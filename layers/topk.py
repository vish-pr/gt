import numpy
from tinygrad.helpers import Timing
from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor


@TinyJit
def topk(x: Tensor):
  # x has shape, batch, vocab_size

  mask = Tensor.full_like(x, True, requires_grad=False)
  indexes = Tensor.arange(x.shape[-1] - 1, -1, -1, requires_grad=False)
  topk_indexes = []
  topk_values = []
  for i in range(5):
    max_value = (x * mask).max(-1, keepdim=True)
    topk_values.append(max_value.squeeze(1))
    max_positions = max_value == x
    mask = mask * (max_positions == False)
    topk_indexes.append(x.shape[-1] - (max_positions * indexes).max(-1) - 1)
  return Tensor.stack(topk_indexes, -1).realize(), Tensor.stack(topk_values, -1).realize()


def test_topk_speed():
  Tensor.manual_seed(12342)

  ops = []
  values = []
  for i in range(100):
    with Timing("topk "):
      a = Tensor.rand(10, 50000).softmax(-1).realize()
      encodings = Tensor.rand(50000, 400)
      pos, val = topk(a)
      b = encodings.gather(pos, dim=0)
      print(b.numpy())
      ops.append(pos)
      values.append(val)
  for i in range(len(ops)):
    print(f"test {i}")
    # print(ops[i].numpy(), values[i].numpy())


def test_topk_accuracy():
  a = Tensor([[.2, 3, 8.8, 9, 7.99, 1.89, 8.9], [.2, 3.9, 4, .5, .9, 1.89, .8], [.2, 3.9, 4, .5, .9, 1.89, 89]])
  # create one hot encodings of dim 7
  encodings = Tensor([[1, 0, 0, 0, 0, 0, 0],
                      [0, 2, 0, 0, 0, 0, 0],
                      [0, 0, 3, 0, 0, 0, 0],
                      [0, 0, 0, 4, 0, 0, 0],
                      [0, 0, 0, 0, 5, 0, 0],
                      [0, 0, 0, 0, 0, 6, 0],
                      [0, 0, 0, 0, 0, 0, 7]]).unsqueeze(0).repeat((3, 1, 1))
  pos, val = topk(a)
  print(val.shape, pos.shape)
  pos = pos.unsqueeze(-1).repeat((1, 1, 7))
  print(pos.numpy())
  print(pos.shape, encodings.shape)
  b = encodings.gather(pos, dim=1)
  print(b.numpy())
  cum = b.cumsum(-2)
  print(cum.numpy())
  do = -(cum[:, :-1] * b[:, 1:]).sum(-1)
  print(do.numpy())
  do_prob = val[:, :-1] / val[:, 0].unsqueeze(-1)
  print(do_prob.numpy())
  # assert op to be
#   [3 2 6]
# [6 1 2]
# [2 5 1]
# [4 4 5]
# [1 6 4]
  # assert topk(a).numpy() == numpy.array([[3, 2, 6], [6, 1, 2], [2, 5, 1], [4, 4, 5], [1, 6, 4]]"""  """)


test_topk_accuracy()
# test_topk_speed()
