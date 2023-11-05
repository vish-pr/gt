#!/usr/bin/env python3

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sentencepiece import SentencePieceProcessor

from data_loader import DataLoader
from llama import Transformer
from opt import Opt
from tinygrad.helpers import GlobalCounters
from tinygrad.nn.state import (get_parameters, get_state_dict, load_state_dict,
                               safe_load, safe_save)
from tinygrad.ops import Device
from tinygrad.tensor import Tensor


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


@dataclass
class Stats:
  train_loss: float = float('inf')
  valid_loss: float = float('inf')
  iter: int = 0
  batch_size: int = 0
  # All below stats are for a single iteration
  train_time: float = 0.0
  data_loading_time: float = 0.0
  forward_pass_time: float = 0.0
  forward_ops: int = 0
  backward_pass_time: float = 0.0
  backward_ops: int = 0
  # TFLOPS: float
  number_of_allocs: int = 0


@dataclass
class FilePaths:
  model_path: Optional[str] = None
  stats_path: Optional[str] = None
  tokenizer_path: str = 'tiny_stories/weights/tok512.model'
  train_path: str = 'tiny_stories/data/TinyStoriesV2-GPT4-train.txt'
  valid_path: str = 'tiny_stories/data/TinyStoriesV2-GPT4-valid.txt'


@dataclass
class ModelMetadata:
  model_args: ModelArgs = ModelArgs()
  stats: Stats = Stats()
  file_paths: FilePaths = FilePaths()


class Model:
  def __init__(self, model_metadta: ModelMetadata):
    self.model_metadata = model_metadta
    self.cpu = Device.canonicalize(None).upper() == 'CPU'
    print('Using CPU' if self.cpu else f'Using backend {Device.canonicalize(None)}')
    # assert model_args is not None or checkpoint_path is not None
    tokenizer_path = model_metadta.file_paths.tokenizer_path
    assert Path(tokenizer_path).is_file()
    self.tokenizer = SentencePieceProcessor()
    self.tokenizer.LoadFromFile(tokenizer_path)
    self.data_loader = DataLoader(model_metadata.file_paths.train_path, model_metadata.file_paths.valid_path, 2 if self.cpu else 12, self.tokenizer)
    self.model = Transformer(**vars(model_metadata.model_args))
    if model_metadata.file_paths.model_path:
      load_state_dict(self.model, safe_load(model_metadata.file_paths.model_path), strict=False)
      print('loaded model', model_metadata.file_paths.model_path)
    self.optim = Opt(self.model)
    self.best_loss = float('inf')
    self.stats_dump = ''

  def save(self, loss: float):
    file_name: str = f'stories_dim{self.model_metadata.model_args.dim}_layer{self.model_metadata.model_args.n_layers}'
    stats_path = 'tiny_stories/weights/' + file_name + '.csv'
    with open(stats_path, 'a') as f:
      f.write(self.stats_dump)
      self.stats_dump = ''
    if loss < self.best_loss:
      self.best_loss = loss
      model_path = 'tiny_stories/weights/' + file_name + '.model'
      self.model_metadata.file_paths.model_path = model_path
      self.model_metadata.file_paths.stats_path = stats_path
      safe_save(get_state_dict(self.model), model_path)
      with open('tiny_stories/' + file_name + '.json', 'w') as f:
        f.write(json.dumps(self.model_metadata, default=lambda o: o.__dict__, indent=2))

  def run_with_loss(self, x: Tensor):
    logits = self.model(x).realize()
    return logits, logits[:, :-1:].sparse_categorical_crossentropy(x[:, 1::], ignore_index=self.tokenizer.eos_id())  # TODO do not ignore first eos_id

  def one_train_pass(self, capture_stats=False):
    Tensor.training = True
    Tensor.no_grad = False
    GlobalCounters.reset()
    # print(f's_ ops: {GlobalCounters.global_ops}, mem: {GlobalCounters.global_mem}, params: {GlobalCounters.kernel_count}, mem_used: {GlobalCounters.mem_used/1e9:.2f} GB, mem_cached: {GlobalCounters.mem_cached/1e9:.2f} GB')
    if capture_stats:
      time_start = mt = time.perf_counter()
    x = self.data_loader.get_batch(train=True)
    if capture_stats:
      self.model_metadata.stats.data_loading_time = time.perf_counter() - time_start
      self.model_metadata.stats.forward_ops = GlobalCounters.global_ops
      # self.stats_dump += f'{iter},{data_loading_time},{forward_ops},'
      time_start = time.perf_counter()

    # self.avg_data_loading_time = (self.avg_data_loading_time * self.iter + loading) / (self.iter + 1)

    logits, loss = self.run_with_loss(x)
    if capture_stats:
      self.model_metadata.stats.forward_pass_time = time.perf_counter() - time_start
      # self.stats_dump += f'{forward_pass_time},'
      # self.forward_pass_time = (self.avg_forward_pass_time * self.iter + forward_pass) / (self.iter + 1)
      time_start = time.perf_counter()
    self.optim.zero_grad()
    loss.backward()
    # print(f'e_ ops: {GlobalCounters.global_ops}, mem: {GlobalCounters.global_mem}, params: {GlobalCounters.kernel_count}, mem_used: {GlobalCounters.mem_used/1e9:.2f} GB, mem_cached: {GlobalCounters.mem_cached/1e9:.2f} GB')
    self.optim.step()
    if capture_stats:
      et = time.perf_counter()
      self.model_metadata.stats.backward_pass_time = et - time_start
      self.model_metadata.stats.backward_ops = GlobalCounters.global_ops - self.model_metadata.stats.forward_ops
      self.model_metadata.stats.train_time = et - mt
      self.model_metadata.stats.number_of_allocs = len([x for x in GlobalCounters.allocs_done if x[0]])
      self.stats_dump = f'{self.model_metadata.stats.iter},{self.model_metadata.stats.data_loading_time},{self.model_metadata.stats.forward_pass_time},{self.model_metadata.stats.backward_pass_time},{self.model_metadata.stats.train_time},{self.model_metadata.stats.forward_ops},{self.model_metadata.stats.backward_ops},{self.model_metadata.stats.number_of_allocs}\n'
      # self.stats_dump += f'{backward_pass_time},{backward_ops},{total_train_time}\n'
    # print(f'loading: {loading:.2f}s, forward: {forward_pass:.2f}s, backward: {backward_pass:.2f}s,{GlobalCounters.global_ops*1e-12/(et-mt):.2f} TFLOPS, loss: {loss.numpy():.2f}')
    # self.evaluate(x[0], logits[0])
    return loss

  def train(self):
    # self.one_train_pass()
    # print(f'size of model and optim is {GlobalCounters.mem_used/1e6:.2f} MB')
    end = self.model_metadata.stats.iter + 1025
    for self.model_metadata.stats.iter in range(self.model_metadata.stats.iter, end):
      loss = self.one_train_pass(capture_stats=self.model_metadata.stats.iter % 5 == 0)
      if self.model_metadata.stats.iter % 25 == 0:
        self.save(loss.numpy().item())

  def evaluate(self, x, y):
    next = y.squeeze(0).argmax(-1).numpy().astype(int).tolist()
    x = x.squeeze(0).numpy().astype(int).tolist()
    for index in range(len(x)):
      print(f'{self.tokenizer.Decode([int(x[index])])} -> {self.tokenizer.Decode([int(next[index])])}', end='|')
      if x[index] == self.tokenizer.eos_id():
        print()
        break

  def validate(self):
    Tensor.training = False
    for param in get_parameters(self.model):
      print(param)
    for i in range(1):
      x = self.data_loader.get_batch(train=False)
      print(x.shape)
      logits, loss = self.run_with_loss(x)
      print((logits[:, :-1:].argmax(-1) == x[:, 1::]).numpy())
      accuracy = (logits[:, :-1:].argmax(-1) == x[:, 1::]).mean()
      print(f'accuracy: {accuracy.numpy()}, loss: {loss.numpy()}')

  def __call__(self, str, debug=False):
    tokens = self.tokenizer.Encode(str)
    logits, loss = self.run_with_loss(Tensor([tokens]))
    next = logits.squeeze(0).argmax(-1).numpy().astype(int).tolist()
    if debug:
      for index, t in enumerate(tokens):
        print(
            f'{self.tokenizer.Decode([int(t)])} -> {self.tokenizer.Decode([int(next[index])])}')
      # print(f'loss: {loss.item()}')
      # print(f'input: {str}')
      # print(f'output: {self.tokenizer.Decode(x)}')
    return self.tokenizer.Decode(next), loss


if __name__ == '__main__':
  model_metadata = ModelMetadata()
  if os.path.isfile("tiny_stories/stories_dim64_layer5.json"):
    # read json file
    with open('tiny_stories/stories_dim64_layer5.json', 'r') as myfile:
      data = myfile.read()
      j = json.loads(data)
      model_metadata = ModelMetadata(ModelArgs(**j['model_args']), Stats(**j['stats']), FilePaths(**j['file_paths']))
  else:
    model_metadata.model_args = ModelArgs(dim=64, n_layers=5, n_heads=8, n_kv_heads=4, vocab_size=512,
                                          hidden_dim=None, multiple_of=4, norm_eps=1e-05, max_seq_len=512, dropout=0.05)
  model = Model(model_metadata)
  model.train()

  # tokenizer_path = 'tiny_stories/weights/tok512.model'
  # checkpoint_path = 'tiny_stories/weights/stories15M.pt'
  # checkpoint_path = 'tiny_stories/weights/stories260K.pt'
  # tokenizer_path = 'tiny_stories/weights/tokenizer.model'
  # model = Model(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path)
  # model_args = ModelArgs(dim=8, hidden_dim=16, n_heads=2, n_layers=4)
  # model = Model(tokenizer_path=tokenizer_path, model_args=model_args)

  # def load_model(checkpoint_path: str) -> Transformer:
  #   assert Path(checkpoint_path).is_file()
  #   check_point = torch_load(checkpoint_path)
  #   state_dict = check_point['model']
  #   unwanted_prefix = '_orig_mod.'
  #   for k, v in list(state_dict.items()):
  #     if k.startswith(unwanted_prefix):
  #       state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
  #   model_args = ModelArgs(**check_point['model_args'])
  #   model = Transformer(**vars(model_args))
  #   load_state_dict(model, state_dict, strict=False)
  #   return model
