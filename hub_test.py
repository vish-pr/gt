
#!/usr/bin/env python
import unittest
from importlib.metadata import requires

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hub import config_mistral as config
from hub import config_instruct
from hub import download_model, download_tokenizer
from llama import Transformer
from tinygrad import Device, nn
from tinygrad.helpers import Timing, dtypes, fetch
from tinygrad.nn.optim import SGD
from tinygrad.tensor import Tensor


class TestMistral(unittest.TestCase):

  def test_create_bfloat_cpu(self):
    b = Tensor([[1.0, 2.0, 3.0], [1.0, 2.9, 3.0]], dtype=dtypes.bfloat16, device='CPU').realize()
    print(b)

  def test_create_bfloat_gpu(self):
    # b = Tensor([[1.0, 2.0, 3.0], [1.0, 2.9, 3.0]], dtype=dtypes.bfloat16, device='CUDA').realize()  # This fails so below is hack
    # print(b)
    b = Tensor([[1.0, 2.0, 3.0], [1.0, 2.9, 3.0]], dtype=dtypes.float16, device='CUDA').realize()
    print(b)

  # def test_bfloat_loading(self):
  #   url1 = 'https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00001-of-00002.bin'
  #   filename1 = "weights/" + "mistral-7b-v01" + '-' + url1.split("/")[-1]
  #   a = nn.state.torch_load(str(fetch(url1, filename1)))['model.embed_tokens.weight']
  #   a = a.bitcast(dtypes.uint16).cast(dtypes.uint32).realize().to('CPU').realize()
  #   a = a.mul(1 << 16).realize().contiguous().bitcast(dtypes.float32).cast(dtype=dtypes.float16).realize()
  #   print(a.numpy())

  def test_mistral_instruct(self):
    tokenizer = download_tokenizer(config_instruct)
    model = download_model(config_instruct)


  def test_mistral_signature(self):
    # device = "cuda"  # the device to load the model onto
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    # pytorch_input: torch.Tensor = tokenizer.encode("", return_tensors='pt')
    # pytorch_input = pytorch_input.to(device)

    # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype='auto')
    # model.to(device)

    # for i in range(5):
    #   op = model(pytorch_input).logits
    #   tok = op[:, -1:, :].argmax(dim=-1)
    #   print('hub warming up', tok)

    # with Timing("hub "):
    #   for i in range(100):
    #     # print(encodeds)
    #     op = model(pytorch_input).logits
    #     tok = op[:, -1:, :].argmax(dim=-1)
    #     # pytorch add element to a 1d tensor
    #     pytorch_input = torch.cat([pytorch_input, tok], dim=-1)
    # print(tokenizer.decode(pytorch_input[0].tolist()))
    # print(pytorch_input[0].tolist())

    encodeds_tiny = Tensor([tokenizer.encode("Lord ram is the king of ayodhya.")])
    # encodeds_tiny = Tensor([tokenizer.encode("Year is 2030 and machines are concious now.")])
    # Transformer(dim=config["hidden_size"], hidden_dim=config["intermediate_size"], n_heads=config["num_attention_heads"], n_kv_heads=config["num_key_value_heads"],
    #                     n_layers=config["num_hidden_layers"], vocab_size=config["vocab_size"], norm_eps=config["rms_norm_eps"], max_seq_len=config["rope_length"])
    model = download_model(model, config)
    # assert model is in cuda
    start_pos = 0
    tok = encodeds_tiny
    for i in range(5):
      nump = encodeds_tiny.numpy()
      print(tokenizer.decode(nump[0]))
      # inp = encodeds_tiny[:, start_pos:].contiguous().realize() # this is very slow, compared to using tok
      # inp = Tensor(nump[:, start_pos:])
      tok = model(encodeds_tiny, start_pos)[:, -1:].realize()
      # tok = model(inp, start_pos)[:, -1:].realize()
      start_pos = encodeds_tiny.shape[1] - 1
      print('tinygrad warming up', tok.numpy())
    with Timing("tinygrad "):
        # print('enco', encodeds_tiny.numpy())
      for i in range(100):  # 100 tokens tinygrad 6765.80 ms, 500 in 35007.69 ms
        # nump = encodeds_tiny.numpy()
        # print(tokenizer.decode(nump[0]))
        # inp = encodeds_tiny[:, start_pos:].contiguous().realize()  # this is very slow, compared to using tok
        # inp = Tensor(nump[:, start_pos:])
        tok = model(encodeds_tiny, start_pos)[:, -1:].realize()
        encodeds_tiny = encodeds_tiny.cat(tok, dim=-1).realize()
        start_pos = encodeds_tiny.shape[1] - 1
    print(encodeds_tiny.numpy())
    print(tokenizer.decode(encodeds_tiny.numpy()[0]))

    # load via hub and tinygrad and check speed and signature
    # encodeds_tiny = encodeds.to(device)
    # encodeds_tiny = Tensor([tokenizer.encode("")])
    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    # encodeds_tiny = Tensor([tokenizer.encode("")])
    # print(encodeds_tiny)
    # print(encodeds_tiny.numpy())
    # with Timing("tinygrad one shot"):
    #   tok = model(encodeds_tiny).realize()
    #   print(tok.numpy())
    # matches = 0
    # for a, b in zip(pytorch_input[0].tolist()[1:], tok[0].numpy().tolist()[:-1]):
    #   if a == b:
    #     matches += 1
    # print(matches, '/', pytorch_input[0].shape[0])

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
