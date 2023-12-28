# code sources:
# https://github.com/karpathy/llama2.c
# https://github.com/tinygrad/tinygrad/blob/master/examples/llama.py

from dataclasses import dataclass
from turtle import forward
from typing import Optional, Tuple

import rope

from tinygrad.helpers import CI, getenv
from tinygrad.jit import TinyJit
from tinygrad.nn import Embedding, Linear
from tinygrad.tensor import Tensor
from layer import RMSNorm, FeedForward

JIT = getenv("JIT", 0 if CI else 1)

class Attention:
  def __init__(self, dim, n_heads, n_kv_heads):
    self.n_heads = n_heads  # 8
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads  # 4
    self.head_dim = dim // n_heads  # 64/8 = 8
    self.n_rep = self.n_heads // self.n_kv_heads  # 2

    self.wq = Linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wk = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = Linear(self.n_heads * self.head_dim, dim, bias=False)

  @staticmethod
  def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x.reshape(bs, seqlen, n_kv_heads, 1, head_dim).expand(bs, seqlen, n_kv_heads, n_rep, head_dim).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

  def __call__(self, x: Tensor, freqs_cos: Tensor, freqs_sin: Tensor, freq_cis) -> Tensor:
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)
    # xq, xk = rope.apply_rotary_emb((xq, xk), freqs_cos, freqs_sin)
    xq, xk = rope.apply_rotary_emb(xq, xk, freq_cis)

    keys, values = xk, xv
    keys, values = self.repeat_kv(keys, self.n_rep), self.repeat_kv(values, self.n_rep)
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(keys, values, is_causal=True).transpose(1, 2).reshape(bsz, seqlen, -1)
    return self.wo(attn)

class TransformerBlock:
  def __init__(self, dim, n_heads, n_kv_heads, hidden_dim, norm_eps):
    self.attention = Attention(dim, n_heads, n_kv_heads)
    self.feed_forward = FeedForward(dim, hidden_dim)
    self.attention_norm = RMSNorm(dim, norm_eps)
    self.ffn_norm = RMSNorm(dim, norm_eps)

  def __call__(self, x: Tensor, freqs_cos, freq_sin, freq_cis) -> Tensor:
    output = self.attention(self.attention_norm(x), freqs_cos, freq_sin, freq_cis=freq_cis)
    h = x + output
    return (h + self.feed_forward(self.ffn_norm(h))).realize()


class Transformer:
  def __init__(self, dim: int, hidden_dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len, n_kv_heads=None, dropout=0.0, rope_theta=10000.0):
    self.layers = [TransformerBlock(dim, n_heads, n_kv_heads=n_kv_heads, norm_eps=norm_eps, hidden_dim=hidden_dim) for _ in range(n_layers)]
    self.norm = RMSNorm(dim, norm_eps)
    print('vocab_size', vocab_size, 'dim', dim)
    self.tok_embeddings = Embedding(vocab_size, dim)
    self.output = Linear(dim, vocab_size, bias=False)
    # self.freqs_cos, self.freq_sin = rope.precompute_freqs_cis(dim // n_heads, max_seq_len * 2, rope_theta)
    self.freqs_cis = rope.precompute_freqs_cis(dim // n_heads, max_seq_len, rope_theta)
    self.norm_output = lambda x: self.output(self.norm(x))
    self.selector = Selector()

  def __call__(self, tokens: Tensor, loss=None, debug=None):
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)
    # freqs_cos = self.freqs_cos.shrink((None, (start_pos, start_pos+seqlen),None,None,None))
    freq_cis = self.freqs_cis.shrink((None, (0, seqlen), None, None, None))


    for layer in self.layers:
      # h = layer(h, self.freqs_cos[0:seqlen], self.freq_sin[0:seqlen], freq_cis=freq_cis)
      h = layer(h, None, None, freq_cis=freq_cis)
    h = self.output(self.norm(h))
    if loss:
      l = loss(h[:, :-1, :], tokens[:, 1:])
      return l
    else:
      # print(h.shape, h.numpy())
      return self.selector(h.realize())
    if debug:
      debug(h, tokens, l)


class Selector:
  def __call__(self, x: Tensor) -> Tensor:
    return x.argmax(-1)
