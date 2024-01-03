#!/usr/bin/env python3

from typing import Generator, List
from tinygrad.helpers import Timing
from tinygrad.tensor import Tensor

from hub import download_model, download_tokenizer
from hub import config_instruct


Tensor.no_grad = True


class MistralModels:
  # all models should have same tokenizer

  # def assert_tokenizers_are_same(self, tokenizes: list):

  def __init__(self):
    self.model = download_model(config_instruct)
    self.tokenizer = download_tokenizer(config_instruct)
    self.start_pos = 0

  def process(self, inp) -> Generator[str, None, None]:
    inp = '[INST] ' + inp + ' [/INST]'
    print('encoding', self.tokenizer.encode(inp))
    toks = Tensor([self.tokenizer.encode(inp)])
    self.start_pos = 0
    for i in range(100):
      with Timing("tinygrad "):
        toks = self.model(toks, self.start_pos)
        self.start_pos += toks.shape[1]
        toks = toks[:, -1:].contiguous().realize()
        # encodeds_tiny = encodeds_tiny.cat(tok, dim=-1).realize()
      op = toks.item()
      # normal decode is removing spaces even with params set to False
      yield str(self.tokenizer.convert_ids_to_tokens(op)).replace('▁', ' ')
      if toks.item() == 2:
        break
