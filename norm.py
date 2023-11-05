from tinygrad.tensor import Tensor

class LayerNorm:
  def __init__(self, dim, eps=1e-5):
    self.eps = eps
    self.weight = Tensor.ones(dim)
    self.bias = Tensor.zeros(dim)

  def __call__(self, x:Tensor):
    return (x.layernorm(eps=self.eps)) * self.weight + self.bias

class RMSNorm:
  def __init__(self, dim, eps=1e-6):
    self.eps = eps
    self.weight = Tensor.ones(dim)

  def __call__(self, x:Tensor):
    # TODO: convert to float?
    return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight