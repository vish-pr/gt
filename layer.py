from tinygrad.tensor import Tensor
from tinygrad.nn import Linear


class LayerNorm:
  def __init__(self, dim, eps=1e-5):
    self.eps = eps
    self.weight = Tensor.ones(dim)
    self.bias = Tensor.zeros(dim)

  def __call__(self, x: Tensor):
    return (x.layernorm(eps=self.eps)) * self.weight + self.bias


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

  def __call__(self, x: Tensor) -> Tensor:
    return self.w2(self.w1(x).silu() * self.w3(x))
