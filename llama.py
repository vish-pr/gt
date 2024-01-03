# code sources:
# https://github.com/karpathy/llama2.c
# https://github.com/tinygrad/tinygrad/blob/master/examples/llama.py

from typing import Union

from layers import RMSNorm, rope, TransformerBlock
from tinygrad.jit import TinyJit
from tinygrad.nn import Embedding, Linear
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor


class Transformer:
  def __init__(self, dim: int, hidden_dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len, n_kv_heads=None, rope_theta=10000.0):
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

  def forward(self, tokens: Tensor, start_pos: Union[Variable, int]):
    _bsz, seqlen = tokens.shape
    freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos + seqlen), None, None, None))

    h = self.tok_embeddings(tokens)
    for layer in self.layers:
      h = layer(h, start_pos, freqs_cis)
    logits = self.output(self.norm(h))
    # print('logits', logits.shape, logits.argmax(-1))
    return logits.argmax(-1).realize()
    # return (logits[:, -1, :] / (temperature + 1e-6)).softmax().flatten().realize()

  @TinyJit
  def jit_forward(self, tokens: Tensor, start_pos: Variable):
    return self.forward(tokens, start_pos)

  def __call__(self, tokens: Tensor, start_pos: int):
    # start_pos = self.fix_cache(tokens) # this is slower than maitaing start_pos outside, by 5ms per token
    # tokens = tokens.shrink((None, (start_pos, tokens.shape[1]))).contiguous().realize()
    # TODO: better way to handle the first call v.s. the rest?
    if tokens.shape[0:2] == (1, 1) and start_pos > 0:
      return self.jit_forward(tokens, Variable("start_pos", 1, self.max_context).bind(start_pos))
    return self.forward(tokens, start_pos)

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
