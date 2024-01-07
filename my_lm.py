# code sources:
# https://github.com/karpathy/llama2.c
# https://github.com/tinygrad/tinygrad/blob/master/examples/llama.py

import math
from typing import Optional, Tuple

import layers.rope as rope
from layers.transformer import RMSNorm
from tinygrad.dtype import dtypes
from tinygrad.helpers import all_int
from tinygrad.nn import Embedding, Linear
from tinygrad.tensor import Tensor


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
  bs, seqlen, n_kv_heads, head_dim = x.shape
  if n_rep == 1:
      return x
  return x.reshape(bs, seqlen, n_kv_heads, 1, head_dim).expand(bs, seqlen, n_kv_heads, n_rep, head_dim).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)


class Attention:
  def __init__(self, dim, n_heads, n_kv_heads):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads

    self.wq_forward = Linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wq_backward = Linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wk = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = Linear(self.n_heads * self.head_dim, dim, bias=False)

  @staticmethod
  def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0, is_forward: bool = False) -> Tensor:
    # NOTE: it works if key, value have symbolic shape
    assert all_int(query.shape), f"does not support symbolic shape {query.shape}"
    if is_forward:
      attn_mask = Tensor.ones(query.shape[-2], key.shape[-2], requires_grad=False, device=query.device).tril(0).cast(dtypes.bool)
    else:
      attn_mask = Tensor.ones(query.shape[-2], key.shape[-2], requires_grad=False, device=query.device).triu(0).cast(dtypes.bool)
    attn_mask = (attn_mask == 0).where(-float("inf"), attn_mask)
    return (query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1]) + attn_mask).softmax(-1).dropout(dropout_p) @ value

  def __call__(self, x: Tuple[Tensor, Tensor], freqs_cos: Tensor, freqs_sin: Tensor) -> Tuple[Tensor, Tensor]:
    x_forward, x_backward = x
    bsz, seqlen, _ = x_forward.shape
    xq_forward, xq_backward, xk_forward, xk_backward, xv_forward, xv_backward = self.wq_forward(x_forward), self.wq_backward(
      x_backward), self.wk(x_forward), self.wk(x_backward), self.wv(x_forward), self.wv(x_backward)
    xq_forward = xq_forward.reshape(xq_forward.shape[0], xq_forward.shape[1], self.n_heads, self.head_dim)
    xq_backward = xq_backward.reshape(xq_backward.shape[0], xq_backward.shape[1], self.n_heads, self.head_dim)
    xk_forward = xk_forward.reshape(xk_forward.shape[0], xk_forward.shape[1], self.n_kv_heads, self.head_dim)
    xk_backward = xk_backward.reshape(xk_backward.shape[0], xk_backward.shape[1], self.n_kv_heads, self.head_dim)
    xv_forward = xv_forward.reshape(xv_forward.shape[0], xv_forward.shape[1], self.n_kv_heads, self.head_dim)
    xv_backward = xv_backward.reshape(xv_backward.shape[0], xv_backward.shape[1], self.n_kv_heads, self.head_dim)
    xq_forward, xq_backward, xk_forward, xk_backward = rope.apply_rotary_emb((xq_forward, xq_backward, xk_forward, xk_backward), freqs_cos, freqs_sin)

    keys_forward, keys_backward, values_forward, values_backward = repeat_kv(xk_forward, self.n_rep), repeat_kv(
      xk_backward, self.n_rep), repeat_kv(xv_forward, self.n_rep), repeat_kv(xv_backward, self.n_rep)
    attn_forward = self.scaled_dot_product_attention(xq_forward.transpose(1, 2), keys_forward.transpose(
      1, 2), values_forward.transpose(1, 2), is_forward=True).transpose(1, 2).reshape(bsz, seqlen, -1)
    attn_backward = self.scaled_dot_product_attention(xq_backward.transpose(1, 2), keys_backward.transpose(
      1, 2), values_backward.transpose(1, 2), is_forward=False).transpose(1, 2).reshape(bsz, seqlen, -1)
    return self.wo(attn_forward), self.wo(attn_backward)


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


class Output:
  def __init__(self, dim, vocab_size):
    # TODO can share weights https://arxiv.org/abs/2110.06399
    self.norm = RMSNorm(dim, 1e-05)
    self.real_proj = Linear(dim, vocab_size, bias=False)
    # self.imag_proj = Linear(dim, vocab_size, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    norm = self.norm(x)
    return self.real_proj(norm)  # , self.imag_proj(norm)


class TransformerBlock:
  def __init__(self, dim, multiple_of, n_heads, n_kv_heads, norm_eps, vocab_size, ffn_dim_multiplier=None):
    self.attention = Attention(dim, n_heads, n_kv_heads)
    self.feed_forward = FeedForward(
        dim, 4 * dim, multiple_of, ffn_dim_multiplier)
    self.attention_norm = RMSNorm(dim, norm_eps)
    self.ffn_norm = RMSNorm(dim, norm_eps)
    # self.output = Output(dim, vocab_size)

  def __call__(self, x: Tuple[Tensor, Tensor], freqs_cos, freq_sin) -> Tuple[Tensor, Tensor]:
    output = self.attention((self.attention_norm(
        x[0]), self.attention_norm(x[1])), freqs_cos, freq_sin)
    h = x[0] + output[0], x[1] + output[1]
    h = h[0] + self.feed_forward(self.ffn_norm(h[0])), h[1] + self.feed_forward(self.ffn_norm(h[1]))
    return h  # , self.output(h)



class Transformer:
  def __init__(self, dim, multiple_of, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=2048, ffn_dim_multiplier=None, n_kv_heads=None, pad_id=-1, dropout=0):
    self.layers = [TransformerBlock(dim, multiple_of, n_heads, n_kv_heads,
                                    norm_eps, vocab_size, ffn_dim_multiplier) for _ in range(n_layers)]
    # self.norm = RMSNorm(dim, norm_eps)
    print('vocab_size', vocab_size, 'dim', dim)
    self.vocab_size = vocab_size
    self.ouput_forward = Output(dim, vocab_size)
    self.ouput_backward = Output(dim, vocab_size)
    self.tok_embeddings = Embedding(vocab_size, dim)
    self.tok_embeddings.weight.requires_grad = False
    self.freqs_cos, self.freq_sin = rope.precompute_freqs_cis(dim // n_heads, vocab_size)
    # self.norm_output = lambda x: self.output(self.norm(x))
    assert self.freq_sin.requires_grad == False
    self.pad_id = pad_id

  def __call__(self, tokens: Tensor, debug=None) -> Tensor:
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)
    h = h, h
    # op = None
    for layer in self.layers:
      h = layer(h, self.freqs_cos[0:seqlen], self.freq_sin[0:seqlen])
    h = self.ouput_forward(h[0]), self.ouput_backward(h[1])
    # op = cur_op if op is None else op + cur_op
    # return logits, self.find_loss(logits[:, :-1, :], x[:, 1:])
    loss1 = self.find_loss(h[0][:, :-1, :], tokens[:, 1:])
    loss2 = self.find_loss(h[1][:, 1:, :], tokens[:, :-1])
    if debug:
      debug(tokens[0], h[0][0])
    return loss1 + loss2

  def find_loss(self, logits: Tensor, y: Tensor):
    return logits.sparse_categorical_crossentropy(y, ignore_index=self.pad_id)
