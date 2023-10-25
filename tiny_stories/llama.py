# code sources:
# https://github.com/karpathy/llama2.c
# https://github.com/tinygrad/tinygrad/blob/master/examples/llama.py

from dataclasses import dataclass
from typing import Optional, Tuple

import rope

from tinygrad.helpers import CI, getenv
from tinygrad.jit import TinyJit
from tinygrad.nn import Embedding, Linear
from tinygrad.tensor import Tensor

class RMSNorm:
  def __init__(self, dim, eps=1e-6):
    self.eps = eps
    self.weight = Tensor.ones(dim)

  def __call__(self, x: Tensor):
    # TODO: convert to float?
    return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight


@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
  bs, seqlen, n_kv_heads, head_dim = x.shape
  if n_rep == 1:
      return x
  return x[:, :, :, None, :].expand(bs, seqlen, n_kv_heads, n_rep, head_dim).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)


class Attention:
  def __init__(self, dim, n_heads, n_kv_heads):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads

    self.wq = Linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wk = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = Linear(self.n_heads * self.head_dim, dim, bias=False)

  def __call__(self, x: Tensor, freqs_cos: Tensor, freqs_sin: Tensor) -> Tensor:
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)
    xq, xk = rope.apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

    keys, values = xk, xv
    keys, values = repeat_kv(keys, self.n_rep).realize(
    ), repeat_kv(values, self.n_rep).realize()
    attn = Tensor.scaled_dot_product_attention(xq.transpose(1, 2), keys.transpose(
        1, 2), values.transpose(1, 2), is_causal=True).transpose(1, 2).reshape(bsz, seqlen, -1)
    return self.wo(attn)


class FeedForward:
  def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
    # TODO: what is this?
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.w1 = Linear(dim, hidden_dim, bias=False)
    self.w2 = Linear(hidden_dim, dim, bias=False)
    self.w3 = Linear(dim, hidden_dim, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    return self.w2(self.w1(x).silu() * self.w3(x))


class TransformerBlock:
  def __init__(self, dim, multiple_of, n_heads, n_kv_heads, norm_eps, ffn_dim_multiplier=None):
    self.attention = Attention(dim, n_heads, n_kv_heads)
    self.feed_forward = FeedForward(
        dim, 4*dim, multiple_of, ffn_dim_multiplier)
    self.attention_norm = RMSNorm(dim, norm_eps)
    self.ffn_norm = RMSNorm(dim, norm_eps)

  def __call__(self, x: Tensor, freqs_cos, freq_sin) -> Tensor:
    output = self.attention(self.attention_norm(
        x), freqs_cos, freq_sin)
    h = x + output
    return h + self.feed_forward(self.ffn_norm(h))


class Transformer:
  def __init__(self, dim, multiple_of, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=2048, ffn_dim_multiplier=None, n_kv_heads=None, rope_theta=10000, **kwargs):
    self.layers = [TransformerBlock(dim, multiple_of, n_heads, n_kv_heads,
                                    norm_eps, ffn_dim_multiplier) for _ in range(n_layers)]
    self.norm = RMSNorm(dim, norm_eps)
    print('vocab_size', vocab_size, 'dim', dim)
    self.tok_embeddings = Embedding(vocab_size, dim)
    self.output = Linear(dim, vocab_size, bias=False)
    self.freqs_cos, self.freq_sin = rope.precompute_freqs_cis(dim // n_heads, 4096, rope_theta)
    self.norm_output = lambda x: self.output(self.norm(x))

  def __call__(self, tokens: Tensor, start_pos: int = 0, temperature: Optional[float] = None):
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings_jitted(tokens)
    for layer in self.layers:
      h = layer(h, self.freqs_cos[0:seqlen], self.freq_sin[0:seqlen])
    return self.output(self.norm(h))
