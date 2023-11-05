from typing import List
from tinygrad.nn.optim import Optimizer
from tinygrad.nn.state import get_state_dict
from tinygrad.tensor import Tensor


class Opt(Optimizer):
  def __init__(self, model, lr=0.01):
    params = self._get_params(model)
    super().__init__(params, lr)

  def _get_params(self, model) -> List[Tensor]:
    params = []
    count = 0
    for k, v in get_state_dict(model).items():
      if v.requires_grad == False:
        continue
      # print(f'Optimizing {k}')
      params.append(v)
      count += v.numel()
    print(f'Optimizing params {len(params)} with {count} floating numbers')
    return params

  def step(self) -> None:
    for i, t in enumerate(self.params):
      assert t.grad is not None
      g = t.grad.realize()
      t.assign(t.detach() - g * self.lr)
    self.realize()
