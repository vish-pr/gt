# code sources:
# https://github.com/karpathy/llama2.c
# https://github.com/tinygrad/tinygrad/blob/master/examples/llama.py

from typing import Any, Callable, Union

import layers.rope as rope
from layers.norm import RMSNorm
from tinygrad.jit import TinyJit
from tinygrad.nn import Embedding, Linear
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor


class FeedForward:
  def __init__(self, dim, hidden_dim):
    self.w1 = Linear(dim, hidden_dim, bias=False)
    self.w2 = Linear(hidden_dim, dim, bias=False)
    self.w3 = Linear(dim, hidden_dim, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
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

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor) -> Tensor:
      x = x.half()
      xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
      xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
      xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
      xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)
      xq, xk = rope.apply_rotary_emb(xq, xk, freqs_cis)
      bsz, seqlen, n_heads, head_dim = xq.shape

      # create kv cache
      if not hasattr(self, "cache_k"):
        self.cache_k = Tensor.zeros(bsz, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype)
        self.cache_v = Tensor.zeros(bsz, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype)
      keys = self.cache_k.shrink((None, (0, start_pos), None, None)).cat(xk, dim=1).contiguous()
      values = self.cache_v.shrink((None, (0, start_pos), None, None)).cat(xv, dim=1).contiguous()

      # update the cache
      # we can not update with cache = ... As this does not work in jit mode hence need to introduce max_context
      self.cache_k.assign(keys.pad((None, (0, self.max_context - start_pos - seqlen), None, None)).contiguous()).realize()
      self.cache_v.assign(values.pad((None, (0, self.max_context - start_pos - seqlen), None, None)).contiguous()).realize()

      keys, values = self.repeat_kv(keys, self.n_rep), self.repeat_kv(values, self.n_rep)

      xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
      mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=x.dtype).triu(start_pos + 1).realize() if seqlen > 1 else None
      attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2).reshape(bsz, seqlen, -1)
      return self.wo(attn)

  # def __call__(self, x: Tensor, freq_cis, mask) -> Tensor:
  #   xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
  #   xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
  #   xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
  #   xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)
  #   # xq, xk = rope.apply_rotary_emb((xq, xk), freqs_cos, freqs_sin)
  #   xq, xk = rope.apply_rotary_emb(xq, xk, freq_cis)

  #   bsz, seqlen, n_heads, head_dim = xq.shape

  #   if not hasattr(self, "cache_k") or self.cache_k.shape[0] != bsz:  # TODO: for a different past have LRU cache
  #     start_pos = 0
  #     self.cache_k = xk
  #     self.cache_v = xv
  #   else:
  #     start_pos = self.cache_k.shape[1]
  #     self.cache_k = self.cache_k.cat(xk, dim=1)
  #     self.cache_v = self.cache_v.cat(xv, dim=1)
  #     xk, xv = self.cache_k, self.cache_v
  #   # print(self.cache_k.shape, start_pos, seqlen)
  #   keys, values = self.repeat_kv(xk, self.n_rep), self.repeat_kv(xv, self.n_rep)
  #   # if cache is used, then xq is of shape (bsz, 1, n_heads, head_dim) and we do not need mask in attention else xq (bsz, seqlen, n_heads, head_dim) and need mask
  #   xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)

  #   attn = xq.scaled_dot_product_attention(keys, values, attn_mask=mask).transpose(1, 2).reshape(bsz, seqlen, -1)
  #   return self.wo(attn)


class TransformerBlock:
  def __init__(self, dim, n_heads, n_kv_heads, hidden_dim, norm_eps):
    self.attention = Attention(dim, n_heads, n_kv_heads)
    self.feed_forward = FeedForward(dim, hidden_dim)
    self.attention_norm = RMSNorm(dim, norm_eps)
    self.ffn_norm = RMSNorm(dim, norm_eps)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor):
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis)
    return h + self.feed_forward(self.ffn_norm(h))


selector: Callable[[Tensor], Tensor] = lambda x: x.argmax(-1)


class Selector:
  def call(self, x: Tensor) -> Tensor:

    return x.argmax(-1)


class Transformer:
  def __init__(self, dim: int, hidden_dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len, n_kv_heads=None, rope_theta=10000.0, selector=selector):
    self.layers = [TransformerBlock(dim, n_heads, n_kv_heads=n_kv_heads, norm_eps=norm_eps, hidden_dim=hidden_dim) for _ in range(n_layers)]
    self.norm = RMSNorm(dim, norm_eps)
    print('vocab_size', vocab_size, 'dim', dim)
    self.tok_embeddings = Embedding(vocab_size, dim)
    self.output = Linear(dim, vocab_size, bias=False)
    # self.freqs_cos, self.freq_sin = rope.precompute_freqs_cis(dim // n_heads, max_seq_len * 2, rope_theta)
    self.freqs_cis = rope.precompute_freqs_cis(dim // n_heads, max_seq_len, rope_theta)
    self.norm_output = lambda x: self.output(self.norm(x))
    self.cache_history = None
    self.max_context = 1000
    self.selector = selector
    self.forward_jit = TinyJit[Tensor](self.forward)

  def forward(self, tokens: Tensor, start_pos: Union[Variable, int]) -> Tensor:
    _bsz, seqlen = tokens.shape
    freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos + seqlen), None, None, None))
    h = self.tok_embeddings(tokens)
    for layer in self.layers:
      h = layer(h, start_pos, freqs_cis)
    logits = self.output(self.norm(h))
    if self.selector:
      return self.selector(logits[:, -1]).realize()
    return logits.realize()
    # return (logits[:, -1, :] / (temperature + 1e-6)).softmax().flatten().realize()

  def __call__(self, tokens: Tensor, start_pos: int) -> Tensor:
    # start_pos = self.fix_cache(tokens) # this is slower than maitaing start_pos outside, by 5ms per token
    # tokens = tokens.shrink((None, (start_pos, tokens.shape[1]))).contiguous().realize()
    # TODO: better way to handle the first call v.s. the rest?
    return self.forward_jit(tokens, Variable("start_pos", 1, self.max_context).bind(
      start_pos)) if tokens.shape[0:2] == (1, 1) and start_pos > 0 else self.forward(tokens, start_pos)
    # op = logits[:, -1, :].argmax(-1).realize()

  # def fix_cache(self, tokens: Tensor):

  #   if not self.cache_history:
  #     self.cache_history = tokens
  #     # return tokens
  #     return 0
  #   length = min(self.cache_history.shape[1], tokens.shape[1] - 1)
  #   while length > 0 and (self.cache_history[:, :length - 1] - tokens[:, :length - 1]).abs().sum().item() != 0:
  #     print((self.cache_history[:, :length - 1] - tokens[:, :length - 1]).abs().sum().item(),
  #           self.cache_history[:, :length - 1].numpy(), tokens[:, :length - 1].numpy())
  #     length -= 1
  #   # for layer in self.layers:
  #   #   layer.attention.cache_k = layer.attention.cache_k[:, :length]
  #   #   layer.attention.cache_v = layer.attention.cache_v[:, :length]
  #   self.cache_history = tokens
  #   # return tokens[:, length:]
  #   return length

  # def fix_cache(self, tokens: Tensor):

  #   if not self.cache_history:
  #     self.cache_history = tokens
  #     return tokens
  #   length = min(self.cache_history.shape[1], tokens.shape[1] - 1)
  #   while length > 0 and (self.cache_history[:, :length - 1] - tokens[:, :length - 1]).abs().sum().item() != 0:
  #     print((self.cache_history[:, :length - 1] - tokens[:, :length - 1]).abs().sum().item(),
  #           self.cache_history[:, :length - 1].numpy(), tokens[:, :length - 1].numpy())
  #     length -= 1
  #   for layer in self.layers:
  #     layer.attention.cache_k = layer.attention.cache_k[:, :length]
  #     layer.attention.cache_v = layer.attention.cache_v[:, :length]
  #   self.cache_history = tokens
  #   return tokens[:, length:]

  # def work(self, tokens, mask):
  #   # print(tokens.shape)
  #   h = self.tok_embeddings(tokens)
  #   freq_cis = self.freqs_cis.shrink((None, (100 - tokens.shape[1], 100), None, None, None))
  #   for layer in self.layers:
  #     # h = layer(h, self.freqs_cos[0:seqlen], self.freq_sin[0:seqlen], freq_cis=freq_cis)
  #     h = layer(h, freq_cis, mask)
  #   h = self.output(self.norm(h))
  #   return self.selector(h.realize())

  # @TinyJit
  # def jit_work(self, tokens, mask):
  #   # seqlen = 10
  #   return self.work(tokens, mask)

  # def __call__(self, tokens: Tensor, loss=None, debug=None):
  #   _bsz, seqlen = tokens.shape
  #   # freqs_cos = self.freqs_cos.shrink((None, (start_pos, start_pos+seqlen),None,None,None))

  #   # print(tokens.shape, end='->')
  #   tokens = self.fix_cache(tokens)
  #   mask = Tensor.full((1, 1, tokens.shape[1], seqlen), float("-inf")).triu(tokens.shape[1] + 1).realize()
  #   if tokens.shape[1] == 1:
  #     return self.jit_work(tokens, mask)
  #   return self.work(tokens, mask)
  #   # if loss:
  #   #   l = loss(h[:, :-1, :], tokens[:, 1:])
  #   #   return l
  #   # else:
  #   #   # print(h.shape, h.numpy())
  #   #   return self.selector(h.realize())
  #   # if debug:
  #   #   debug(h, tokens, l)
