
#!/usr/bin/env python
from importlib.metadata import requires
import unittest
import numpy as np
from tinygrad import nn
from tinygrad.helpers import dtypes, fetch
from tinygrad.nn.optim import SGD
from tinygrad.tensor import Tensor
from tinygrad import Device


class TestMistral(unittest.TestCase):

  def test_create_bfloat_cpu(self):
    b = Tensor([[1.0, 2.0, 3.0], [1.0, 2.9, 3.0]], dtype=dtypes.bfloat16, device='CPU').realize()
    print(b)

  def test_create_bfloat_gpu(self):
    b = Tensor([[1.0, 2.0, 3.0], [1.0, 2.9, 3.0]], dtype=dtypes.bfloat16, device='GPU').realize()
    print(b)

  def test_bfloat_loading(self):
    return
    b = Tensor([[1.0, 2.0, 3.0], [1.0, 2.9, 3.0]], dtype=dtypes.bfloat16, device='GPU').realize()
    url1 = 'https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00001-of-00002.bin'
    url2 = 'https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00002-of-00002.bin'
    # b = Tensor([[1.0, 2.0, 3.0], [1.0, 2.9, 3.0]], dtype=dtypes.float16).realize()
    # print(b)
    # b = b.bitcast(dtypes.uint16).realize()
    # print(b.numpy())
    # b = b.bitcast(dtypes.float16).realize()
    # print(b.numpy())

    filename1 = "weights/" + "mistral-7b-v01" + '-' + url1.split("/")[-1]
    filename2 = "weights/" + "mistral-7b-v01" + '-' + url2.split("/")[-1]
    a = nn.state.torch_load(str(fetch(url1, filename1)))['model.embed_tokens.weight']
    b = nn.state.torch_load(str(fetch(url2, filename2)))['lm_head.weight']
    a = a.bitcast(dtypes.uint16).cast(dtypes.uint32).realize().to('CPU').realize()
    a = a.mul(1 << 16).realize().contiguous().bitcast(dtypes.float32).cast(dtype=dtypes.float16).realize()
    print(a)
    print(a.numpy()[4095])
    b = b.bitcast(dtypes.uint16).cast(dtypes.uint32).realize().to('CPU').realize()
    b = b.mul(1 << 16).realize().contiguous().bitcast(dtypes.float32).cast(dtype=dtypes.float16).realize()
    print(b)
    print(b.numpy()[4095])

  def test_linear_transfomation(self):
    return
    url1 = 'https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00001-of-00002.bin'
    url2 = 'https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00002-of-00002.bin'
    # b = Tensor([[1.0, 2.0, 3.0], [1.0, 2.9, 3.0]], dtype=dtypes.float16).realize()
    # print(b)
    # b = b.bitcast(dtypes.uint16).realize()
    # print(b.numpy())
    # b = b.bitcast(dtypes.float16).realize()
    # print(b.numpy())

    filename1 = "weights/" + "mistral-7b-v01" + '-' + url1.split("/")[-1]
    filename2 = "weights/" + "mistral-7b-v01" + '-' + url2.split("/")[-1]
    a = nn.state.torch_load(str(fetch(url1, filename1)))['model.embed_tokens.weight']
    b = nn.state.torch_load(str(fetch(url2, filename2)))['lm_head.weight']
    a = a.bitcast(dtypes.uint16).cast(dtypes.uint32).realize().to('CPU').realize()
    a = a.mul(1 << 16).realize().contiguous().bitcast(dtypes.float32).realize()
    b = b.bitcast(dtypes.uint16).cast(dtypes.uint32).realize().to('CPU').realize()
    b = b.mul(1 << 16).realize().contiguous().bitcast(dtypes.float32).realize()
    print(a.shape)

    a_train = a[100:10000:2]
    b_train = b[100:10000:2]
    a_test = a[101:10001:2]
    b_test = b[101:10001:2]

    # Initialize T as a variable with requires_grad=True
    T = Tensor.zeros(a_train.shape[1], a_train.shape[1], requires_grad=True) / a_train.shape[1]

    learning_rate = 0.2
    num_iterations = 1000
    opt = SGD([T], lr=learning_rate)

    for i in range(num_iterations):
      # Forward pass: compute the transformation
      b_pred = a_train @ T
      # print(b_pred.numpy())
      # print(b_pred.numpy()[2])
      # print(b_train.numpy()[2])

      # Compute and print loss
      print((b_pred - b_train).numpy())
      loss = (b_pred - b_train).pow(2).sum()
      opt.zero_grad()
      loss.backward()
      y = a_test @ T
      loss_valid = (y - b_test).pow(2).sum()
      print(i, loss.numpy(), loss_valid.numpy())

      # print(T.grad.numpy())
      opt.step()


if __name__ == '__main__':
  unittest.main()
