
# https://github.com/karpathy/llama2.c
# https://github.com/tinygrad/tinygrad/blob/master/examples/llama.py

import math
from typing import Callable, Union

from tinygrad.jit import TinyJit
from tinygrad.nn import Embedding, Linear
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor

from layers import rope


class RMSNorm:
  def __init__(self, dim, eps=1e-6):
    self.eps = eps
    self.weight = Tensor.ones(dim)

  def __call__(self, x: Tensor):
    # float because half will become inf
    return ((x * (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight).half()


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


class Transformer:
  def __init__(self, dim: int, hidden_dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len, n_kv_heads=None, rope_theta=10000.0, selector=selector):
    self.layers = [TransformerBlock(dim, n_heads, n_kv_heads=n_kv_heads, norm_eps=norm_eps, hidden_dim=hidden_dim) for _ in range(n_layers)]
    self.norm = RMSNorm(dim, norm_eps)
    print('vocab_size', vocab_size, 'dim', dim)
    self.tok_embeddings = Embedding(vocab_size, dim)
    self.output = Linear(dim, vocab_size, bias=False)  # weight of shape: vocab_size, dim
    # self.freqs_cos, self.freq_sin = rope.precompute_freqs_cis(dim // n_heads, max_seq_len * 2, rope_theta)
    self.freqs_cis = rope.precompute_freqs_cis(dim // n_heads, max_seq_len, rope_theta)
    self.norm_output = lambda x: self.output(self.norm(x))
    self.cache_history = None
    self.max_context = 5000
    self.selector = selector
    self.forward_jit = TinyJit[Tensor](self.forward)

  def forward(self, tokens: Tensor, start_pos: Union[Variable, int]) -> Tensor:
    _bsz, seqlen = tokens.shape
    freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos + seqlen), None, None, None))
    h = self.tok_embeddings(tokens).realize()
    for layer in self.layers:
      h = layer(h, start_pos, freqs_cis).realize()
    logits = self.output(self.norm(h))
    if self.selector:
      return self.selector(logits[:, -1]).realize()
    return logits.softmax(axis=-1).realize()
    # return (logits[:, -1, :] / (temperature + 1e-6)).softmax().flatten().realize()

  def __call__(self, tokens: Tensor, start_pos: int) -> Tensor:
    # start_pos = self.fix_cache(tokens) # this is slower than maitaing start_pos outside, by 5ms per token
    # tokens = tokens.shrink((None, (start_pos, tokens.shape[1]))).contiguous().realize()
    # TODO: better way to handle the first call v.s. the rest?
    return self.forward_jit(tokens, Variable("start_pos", 1, self.max_context).bind(
      start_pos)) if tokens.shape[0:2] == (1, 1) and start_pos > 0 else self.forward(tokens, start_pos)
    # op = logits[:, -1, :].argmax(-1).realize()
