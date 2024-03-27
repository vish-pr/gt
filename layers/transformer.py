# code sources:
# https://github.com/karpathy/llama2.c
# https://github.com/tinygrad/tinygrad/blob/master/examples/llama.py

import math
from typing import Any, Callable, Union

from attr import dataclass
import numpy as np

import layers.rope as rope
from layers.norm import RMSNorm
from tinygrad import TinyJit
from tinygrad.nn import Embedding, Linear
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor


class FeedForward:
  def __init__(self, dim, hidden_dim):
    self.w1 = Linear(dim, hidden_dim, bias=False)
    self.w2 = Linear(hidden_dim, dim, bias=False)
    self.w3 = Linear(dim, hidden_dim, bias=False)
    self.dim = dim
    self.hidden_dim = hidden_dim

  def add_lora(self):
    self.weight_in = Tensor.zeros(4, self.dim, requires_grad=True)
    self.weight_out = Tensor.kaiming_uniform(self.hidden_dim, 4, a=math.sqrt(5), requires_grad=True)

  def __call__(self, x: Tensor) -> Tensor:
    if hasattr(self, "weight_in"):
      y = x.linear(self.weight_in.transpose()).linear(self.weight_out.transpose())
      return self.w2((self.w1(x) + y).silu() * self.w3(x))
    return self.w2(self.w1(x).silu() * self.w3(x))


class Attention:
  # Causal attention
  # suports grouped key, value heads
  # supports cache

  # TODO: flash attention

  def __init__(self, dim, n_heads, n_kv_heads=None):
    self.n_heads = n_heads  # 8
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads  # 4
    self.head_dim = dim // n_heads  # 64/8 = 8
    self.n_rep = self.n_heads // self.n_kv_heads  # 2

    self.wq = Linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wk = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = Linear(self.n_heads * self.head_dim, dim, bias=False)
    self.max_context = 5000

  @staticmethod
  def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x.reshape(bs, seqlen, n_kv_heads, 1, head_dim).expand(bs, seqlen, n_kv_heads, n_rep, head_dim).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, use_cache=True) -> Tensor:
      x = x.half()
      xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
      xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
      xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
      xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)
      xq, xk = rope.apply_rotary_emb(xq, xk, freqs_cis)
      bsz, seqlen, n_heads, head_dim = xq.shape

      # create kv cache
      if use_cache:
        if not hasattr(self, "cache_k"):
          self.cache_k = Tensor.zeros(bsz, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype)
          self.cache_v = Tensor.zeros(bsz, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype)
        keys = self.cache_k.shrink((None, (0, start_pos), None, None)).cat(xk, dim=1).contiguous()
        values = self.cache_v.shrink((None, (0, start_pos), None, None)).cat(xv, dim=1).contiguous()

        # update the cache
        # we can not update with cache = ... As this does not work in jit mode hence need to introduce max_context
        self.cache_k.assign(keys.pad((None, (0, self.max_context - start_pos - seqlen), None, None)).contiguous()).realize()
        self.cache_v.assign(values.pad((None, (0, self.max_context - start_pos - seqlen), None, None)).contiguous()).realize()
      else:
        keys, values = xk, xv

      keys, values = self.repeat_kv(keys, self.n_rep), self.repeat_kv(values, self.n_rep)

      xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
      mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=x.dtype).triu(start_pos + 1).realize() if seqlen > 1 else None
      attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2).reshape(bsz, seqlen, -1)
      return self.wo(attn)


class TransformerBlock:
  def __init__(self, dim, n_heads, n_kv_heads, hidden_dim, norm_eps):
    self.attention = Attention(dim, n_heads, n_kv_heads)
    self.feed_forward = FeedForward(dim, hidden_dim)
    self.attention_norm = RMSNorm(dim, norm_eps)
    self.ffn_norm = RMSNorm(dim, norm_eps)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor):
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis)
    return h + self.feed_forward(self.ffn_norm(h))


selector: Callable[[Tensor], Tensor] = lambda x: x.argmax(-1, keepdim=True)


class MaxLogitsSelector:
  def __call__(self, logits: Tensor) -> Tensor:
    return logits.argmax(-1, keepdim=True)


class Explorer:
  def __call__(self, logits: Tensor) -> Tensor:
    prob = -logits.softmax(axis=-1)
    # shape: bsz, vocab_size
    cpu = prob.numpy()
    # sort rows
    # Sort each row and get the indices
    sorted_indices = np.argsort(cpu, axis=-1)

    # Use the indices to get the sorted matrix
    sorted_matrix = np.take_along_axis(cpu, sorted_indices, axis=-1)
    print(-sorted_matrix[0, :20])

    return logits.argmax(-1, keepdim=True)

class Transformer:

  @dataclass
  class Config:
    dim: int
    hidden_dim: int
    n_heads: int
    n_layers: int
    norm_eps: float
    vocab_size: int
    n_kv_heads: int
    rope_theta: float
    max_seq_len: int

  def __init__(self, config: Config, selector=selector):
    self.layers = [TransformerBlock(config.dim, config.n_heads, n_kv_heads=config.n_kv_heads, norm_eps=config.norm_eps,
                                    hidden_dim=config.hidden_dim) for _ in range(config.n_layers)]
    self.norm = RMSNorm(config.dim, config.norm_eps)
    print('vocab_size', config.vocab_size, 'dim', config.dim)
    self.tok_embeddings = Embedding(config.vocab_size, config.dim)
    self.output = Linear(config.dim, config.vocab_size, bias=False)  # weight of shape: vocab_size, dim
    # self.freqs_cos, self.freq_sin = rope.precompute_freqs_cis(dim // n_heads, max_seq_len * 2, rope_theta)
    self.freqs_cis = rope.precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len, config.rope_theta)
    self.max_context = 5000
    self.selector = Explorer()

  @TinyJit
  def predicting_one_token(self, tokens: Tensor, start_pos: int) -> Tensor:
    return self.predicting_multiple_tokens(tokens, start_pos, 1)

  def predicting_multiple_tokens(self, tokens: Tensor, start_pos: int, seq_len: int) -> Tensor:
    freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos + seq_len), None, None, None))
    h = self.tok_embeddings(tokens).realize()
    for layer in self.layers:
      h = layer(h, start_pos, freqs_cis).realize()
    logits = self.output(self.norm(h))
    return self.selector(logits[:, -1]).realize()

  def __call__(self, tokens: Tensor, start_pos: int) -> Tensor:
    _bsz, seqlen = tokens.shape
    if start_pos > 0 and seqlen == 1:
      # start pos > 0 so cache is created, # TODO move cache creation in init and remove this conidtion
      return self.predicting_one_token(tokens, Variable("start_pos", 1, self.max_context).bind(
          start_pos))
    return self.predicting_multiple_tokens(tokens, start_pos, int(seqlen))
