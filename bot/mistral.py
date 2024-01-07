
import os
import sys
import time

sys.path.append(os.getcwd())   # fmt: off

import re
from typing import List, Union

import numpy as np
from language_model import LanguageModel

from hub import config_instruct, download_model, download_tokenizer
from tinygrad.device import Device
from tinygrad.tensor import Tensor


class MistralModels(LanguageModel):
  # all models should have same tokenizer
  # def assert_tokenizers_are_same(self, tokenizes: list):
  def __init__(self):
    # call super constructor with model
    super().__init__(download_model(config_instruct))
    self.tokenizer = download_tokenizer(config_instruct)
    self.pattern = re.compile(r'<0x([0-9A-Fa-f]+)>')

  def token_to_string(self, token: int) -> Union[str, None]:
    if token == config_instruct['eos_token_id']:
      return None
    a = str(self.tokenizer.convert_ids_to_tokens(token))  # normal decode is removing spaces even with params set to False
    a = a.replace('▁', ' ')
    output_string = self.pattern.sub(lambda match: chr(int(match.group(1), 16)), a)
    return output_string

  def input_to_tokens(self, inp: str) -> List[int]:
    inp = '[INST] ' + inp + ' [/INST]'
    return self.tokenizer.encode(inp)

from typing import Generator, List, Union

def mistral_test():
  model = MistralModels()
  st = time.perf_counter_ns()
  op_stream: Generator[str, None, None] = model.process('What is a prime number?')
  output = ''.join(op_stream)
  e1 = time.perf_counter_ns() - st
  print(f'{e1*1e-6:6.2f} ms')
  expected = ' A prime number is a positive integer that is greater than 1 and has no positive divisors other than 1 and itself. In other words, a prime number can only be evenly divided by 1 and itself. For example, the first few prime numbers are 2, 3, 5, 7, 11, and 13. These numbers cannot be divided evenly by any number other than 1 and themselves. Prime numbers are important in number theory and have various applications in mathematics and cryptography.'
  assert output == expected, f'output: {output}, expected: {expected}'



  # speed test
  # TODO: changing batch size is not allowed in cache fix it
  inp = Tensor.randint((1, 100), 0, 100).realize()
  length = 100
  start_pos = 0
  for i in range(3):
    inp = model.model(inp , start_pos).realize()
    start_pos += length
    length = 1
  st = time.perf_counter_ns()
  for i in range(50):
    inp = model.model(inp , start_pos).realize()
    start_pos += length
    length = 1
    Device[Device.DEFAULT].synchronize()
  e = time.perf_counter_ns() - st
  per_token = e*1e-6 / 50
  assert per_token < 210 and per_token > 150, f"per token {per_token:6.2f} ms"
  print(f"per token {per_token:6.2f} ms")

# mistral_test()
