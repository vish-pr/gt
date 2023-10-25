
from tinygrad.nn import Embedding, Linear
from tinygrad.tensor import Tensor


class Transformer:
  def __init__(self, dim, vocab_size, **kwargs):
    self.tok_embeddings = Embedding(vocab_size, dim)
    self.output = Linear(dim, vocab_size, bias=False)

  def __call__(self, tokens: Tensor):
    h = self.tok_embeddings(tokens)
    h = self.output(h)
    return h
