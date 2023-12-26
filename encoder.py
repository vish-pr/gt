# Encoder
# Algorithm
# given a token map to a random vector
# each vector has associated knowledge size (char, token, word, sentence, paragraph, document)
# Apply reversible layers
# After each layer also train a layer to reverse that layer (may share weights) ? share bias?
# These weights are diagnoal half matrix, so each dimension can influence dimension before it not after. How to check bias?
# so for small knowldge size only need to train a small input vector.
# output vector is used for
# * inpur of LM
# * Nearest neighbour search for output of LM (instead of distribution over vocab)
# * A mapping to all avaialble encoding.
# * Can use public available encoding to train group of characters. and map this output to that encoding.


# Noise
# Adding noise to output of encoding generated from random vector, can make LM more robust, and make nearest neighbour search more robust.

from typing import Any, Generator, List

from tinygrad.tensor import Tensor
from llama import FeedForward

from layer import RMSNorm


class UnicodeEncoder:
  def __init__(self) -> None:
    self.vocab_size = 0
    embed_size = 8
    self.weight = Tensor.glorot_uniform(self.vocab_size, embed_size)
    self.id_to_idx = {}

  def encode_char(self, char: str):
    return ord(char)

  def decode_char(self, char: int):
    return chr(char)

  def _get_or_create_idx(self, id: int) -> int:
    if id not in self.id_to_idx:
      self.id_to_idx[id] = self.vocab_size
      self.vocab_size += 1
      # concat a row in weight matrix
      self.weight = self.weight.cat(Tensor.glorot_uniform(1, self.weight.shape[1]), dim=1)
      return self.vocab_size - 1
    return self.id_to_idx[id]

  def encode(self, text: str) -> Generator[Tensor, None, None]:
    for char in text:
      id = ord(char)
      idx = self._get_or_create_idx(id)
      yield self.weight[idx]


class TransformerBlock:
  # ff
  # left attending to one left, right attending to one on right
  # left try to predit right
  #   if sccess:  w
  # right try to predict left

  def __init__(self, dim, multiple_of, n_heads, n_kv_heads, norm_eps, ffn_dim_multiplier=None):
    self.attention = Attention(dim, n_heads, n_kv_heads)
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
    self.feed_forward = FeedForward(
        dim, 4 * dim, multiple_of, ffn_dim_multiplier)
    self.norm = RMSNorm(dim, norm_eps)

  def __call__(self, tokens: Tensor):
    _bsz, seqlen, dim = tokens.shape
    # increase size of encoding at each layer
    # try to predict next token at a layer
    # try to decode from each layer
    for layer in self.layers:
      h = layer(h)
    return self.output(self.norm(h))
