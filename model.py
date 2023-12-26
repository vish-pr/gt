#!/usr/bin/env python3

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sentencepiece import SentencePieceProcessor
from tinygrad.nn import Embedding, state

from data_loader import DataLoader
from llama import Transformer
# from my_lm import Transformer
from opt import Opt
from tinygrad.helpers import GlobalCounters
from tinygrad.nn.state import (get_parameters, get_state_dict, load_state_dict,
                               safe_load, safe_save, torch_load)
from tinygrad.ops import Device
from tinygrad.tensor import Tensor


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int]
    vocab_size: int
    hidden_dim: Optional[int]
    multiple_of: int
    norm_eps: float
    max_seq_len: int
    dropout: float


@dataclass
class Stats:
  iter: int = 0
  train_time: float = 0.0
  data_loading_time: float = 0.0
  forward_pass_time: float = 0.0
  backward_pass_time: float = 0.0
  forward_giga_ops: float = 0.0
  backward_giga_ops: float = 0.0
  TFLOPS: float = 0.0
  number_of_allocs: int = 0
  max_mem_used: float = 0.0
  batch_size: int = 0
  train_loss: float = float('inf')
  valid_loss: float = float('inf')


@dataclass
class TensorStats:
  name: str
  iter: int
  mean: float
  std: float
  min: float
  max: float

@dataclass
class FilePaths:
  model_path: Optional[str] = None
  stats_path: Optional[str] = None
  tokenizer_path: str = 'tiny_stories/weights/tok512.model'
  train_path: str = 'tiny_stories/data/TinyStoriesV2-GPT4-train.txt'
  valid_path: str = 'tiny_stories/data/TinyStoriesV2-GPT4-valid.txt'


@dataclass
class ModelMetadata:
  model_args: ModelArgs
  stats: Stats
  file_paths: FilePaths


class Model:
  def __init__(self, model_metadta: ModelMetadata):
    self.model_metadata = model_metadta
    self.cpu = Device.canonicalize(None).upper() == 'CPU'
    print('Using CPU' if self.cpu else f'Using backend {Device.canonicalize(None)}')
    tokenizer_path = model_metadta.file_paths.tokenizer_path
    assert Path(tokenizer_path).is_file()
    self.tokenizer = SentencePieceProcessor()
    self.tokenizer.LoadFromFile(tokenizer_path)
    self.data_loader = DataLoader(self.model_metadata.file_paths.train_path, self.model_metadata.file_paths.valid_path,
                                  2 if self.cpu else 30, self.tokenizer, max_seq_len=self.model_metadata.model_args.max_seq_len, cpu=self.cpu)
    self.model_metadata.model_args.vocab_size = self.tokenizer.vocab_size()
    hidden_dim = self.model_metadata.model_args.dim * 4
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    # if self.model_metadata.model_args.ffn_dim_multiplier is not None:
    #   hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    multiple_of = self.model_metadata.model_args.multiple_of
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.model_metadata.model_args.hidden_dim = hidden_dim
    self.model = Transformer(**vars(self.model_metadata.model_args))
    if self.model_metadata.file_paths.model_path:
      load_state_dict(self.model, safe_load(self.model_metadata.file_paths.model_path), strict=False)
      print('loaded model', self.model_metadata.file_paths.model_path)
    self.optim = Opt(self.model)
    self.best_loss = float('inf')
    self.stats_dump = ''
    assert self.model.tok_embeddings.vocab_size == self.tokenizer.vocab_size()
    self.dist = None
    # self.next_x = None

  def save(self):
    iteration = self.model_metadata.stats.iter
    file_name: str = f'my_stories_exp_1_dim{self.model_metadata.model_args.dim}_layer{self.model_metadata.model_args.n_layers}_ckpt{int(iteration/10000)}'
    print('saving', file_name)
    stats_path = 'tiny_stories/weights/' + file_name + '_training_dump' '.csv'
    with open(stats_path, 'a') as f:
      f.write(self.stats_dump)
      self.stats_dump = ''
    model_stats_dump = ''
    for k, v in get_state_dict(self.model).items():
      if v.grad is None:
          continue
      data = v.numpy()
      # get stats like mean, std dev of each param
      model_stats_dump += f'{k},{iteration},{data.mean():.5f},{data.std():.5f},{data.max():.5f},{data.min():.5f}\n'
      # Maybe for multi dimenstion tensors, we can get stats of stats of each dimension
      grad = v.grad.numpy()
      model_stats_dump += f'{k}_grad,{iteration},{grad.mean():.5f},{grad.std():.5f},{grad.max():.5f},{grad.min():.5f}\n'
    model_stats_path = 'tiny_stories/weights/' + file_name + '_model_stats_dump' + '.csv'
    with open(model_stats_path, 'a') as f:
      f.write(model_stats_dump)
    if self.model_metadata.stats.train_loss < self.best_loss:
      self.best_loss = self.model_metadata.stats.train_loss
      model_path = 'tiny_stories/weights/' + file_name + '.model'
      self.model_metadata.file_paths.model_path = model_path
      self.model_metadata.file_paths.stats_path = stats_path
      safe_save(get_state_dict(self.model), model_path)
      with open('tiny_stories/' + file_name + '.json', 'w') as f:
        f.write(json.dumps(self.model_metadata, default=lambda o: o.__dict__, indent=2))

  # def sparse_categorical_crossentropy_with_sim(self, logits: Tensor, Y: Tensor, ignore_index=-1) -> Tensor:
  #   loss_mask = Y != ignore_index  # b, seq_len, 1
  #   if self.dist is None:
  #     self.dist = Embedding(self.model.vocab_size, self.model.vocab_size)
  #     self.dist.weight.requires_grad = False
  #     emb = self.model.tok_embeddings.weight.detach()
  #     magnitude = emb.pow(2).sum(-1).sqrt()
  #     magnitude_product_matrix = magnitude.unsqueeze(-1) * magnitude.unsqueeze(-2)
  #     self.dist.weight = (emb @ emb.transpose(-1, -2) / magnitude_product_matrix).softmax(-1)

  #   cos = self.dist(Y)
  #   return -logits.log_softmax().mul(cos).sum() / loss_mask.sum()

  def loss(self, logits: Tensor, y: Tensor):
    return logits.sparse_categorical_crossentropy(y, ignore_index=self.tokenizer.pad_id())
  #   # multiply the predicted probability with tokens
  #   # create n*n matrix of cosine similariy of matrix
  #   loss2 = self.sparse_categorical_crossentropy_with_sim(logits, y, ignore_index=self.tokenizer.pad_id())
  #   return loss2
  #   #

    # return logits, self.find_loss(logits[:, :-1, :], x[:, 1:])
  def one_train_pass(self, capture_stats=False):
    Tensor.training = True
    Tensor.no_grad = False
    GlobalCounters.reset()
    if capture_stats:
      mt = time.perf_counter()
    x = self.data_loader.get_batch(train=True).realize()
    if capture_stats:
      time_start = time.perf_counter()
      self.model_metadata.stats.data_loading_time = time_start - mt
    # self.next_x = self.data_loader.get_batch(train=True).realize() # TODO if loading is slow, do this in parallel
    loss = self.model(x, loss=self.loss).realize()
    if capture_stats:
      self.model_metadata.stats.forward_pass_time = time.perf_counter() - time_start
      time_start = time.perf_counter()
      self.model_metadata.stats.forward_giga_ops = GlobalCounters.global_ops / 1e9
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    self.model_metadata.stats.train_loss = loss.numpy().item()  # Synchronizes GPU
    if capture_stats:
      et = time.perf_counter()
      self.model_metadata.stats.backward_pass_time = et - time_start
      self.model_metadata.stats.backward_giga_ops = GlobalCounters.global_ops / 1e9 - self.model_metadata.stats.forward_giga_ops
      self.model_metadata.stats.train_time = et - mt
      self.model_metadata.stats.TFLOPS = GlobalCounters.global_ops / 1e12 / self.model_metadata.stats.train_time
      self.model_metadata.stats.number_of_allocs = len([x for x in GlobalCounters.allocs_done if x[0]])
      self.model_metadata.stats.max_mem_used = GlobalCounters.max_mem_used / 1e9
      self.stats_dump += ','.join(f'{value:.3f}' if isinstance(value, float) else str(value)
                                  for value in vars(self.model_metadata.stats).values()) + '\n'
    # self.evaluate(x[0], logits[0])

  def train(self):
    # self.one_train_pass()
    # print(f'size of model and optim is {GlobalCounters.mem_used/1e6:.2f} MB')
    end = self.model_metadata.stats.iter + 10000
    self.model_metadata.stats.batch_size = self.data_loader.batch_size
    for self.model_metadata.stats.iter in range(self.model_metadata.stats.iter, end):
      self.one_train_pass(capture_stats=self.model_metadata.stats.iter % 5 == 0)
      if self.model_metadata.stats.iter % 100 == 0:
        self.validate()
        self.save()

  def evaluate(self, logits, x, loss):
    accuracy = (logits[:, :-1:].argmax(-1) == x[:, 1::]).mean()
    print(f'validation: accuracy: {accuracy.numpy()}, loss: {loss.numpy()}')
    next = logits[0].squeeze(0).argmax(-1).numpy().astype(int).tolist()
    x = x[0].squeeze(0).numpy().astype(int).tolist()
    for index in range(len(x)):
      print(f'{self.tokenizer.Decode([int(x[index])])} -> {self.tokenizer.Decode([int(next[index])])}', end='|')
      if x[index] == self.tokenizer.eos_id():
        print()
        break

  def validate(self):
    Tensor.training = False
    Tensor.no_grad = True
    x = self.data_loader.get_batch(train=False)
    loss = self.model(x, loss=self.loss, debug=self.evaluate).realize()
    # accuracy = (logits[:, :-1:].argmax(-1) == x[:, 1::]).mean()
    # print(f'validation: accuracy: {accuracy.numpy()}, loss: {loss.numpy()}')
    # self.evaluate(x[0], logits[0])
    self.model_metadata.stats.valid_loss = loss.numpy().item()

  # def __call__(self, str, debug=False):
  #   tokens = self.tokenizer.Encode(str)
  #   loss = self.run_with_loss(Tensor([tokens]))
  #   # next = logits.squeeze(0).argmax(-1).numpy().astype(int).tolist()
  #   # if debug:
  #   #   for index, t in enumerate(tokens):
  #   #     print(
  #   #         f'{self.tokenizer.Decode([int(t)])} -> {self.tokenizer.Decode([int(next[index])])}')
  #     # print(f'loss: {loss.item()}')
  #     # print(f'input: {str}')
  #     # print(f'output: {self.tokenizer.Decode(x)}')
  #   return self.tokenizer.Decode(next), loss


def args() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description='Train a Transformer model on a language modeling task')
  parser.add_argument('--model_path', type=str)
  return parser


def load_model(args) -> Model:
  if args.model_path:
    file_name = args.model_path
    assert Path(file_name).is_file()
    with open(file_name, 'r') as myfile:
      data = myfile.read()
      j = json.loads(data)
      model_metadata = ModelMetadata(ModelArgs(**j['model_args']), Stats(**j['stats']), FilePaths(**j['file_paths']))
  else:
    model_metadata = ModelMetadata(ModelArgs(dim=64, n_layers=5, n_heads=8, n_kv_heads=4, multiple_of=4,
                                   norm_eps=1e-05, max_seq_len=512, vocab_size=512, dropout=0.05), Stats(), FilePaths())
  return Model(model_metadata)


if __name__ == '__main__':
  model = load_model(args().parse_args())

  # tokenizer_path = 'tiny_stories/weights/tok512.model'
  # checkpoint_path = 'tiny_stories/weights/stories15M.pt'
  checkpoint_path = 'tiny_stories/weights/stories260K.pt'
  # tokenizer_path = 'tiny_stories/weights/tokenizer.model'
  # model = Model(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path)
  # model_args = ModelArgs(dim=8, hidden_dim=16, n_heads=2, n_layers=4)
  # model = Model(tokenizer_path=tokenizer_path, model_args=model_args)

  assert Path(checkpoint_path).is_file()
  check_point = torch_load(checkpoint_path)
  state_dict = check_point['model']
  unwanted_prefix = '_orig_mod.'
  for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
      state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
  # model_args = ModelArgs(**check_point['model_args'])
  # model = Transformer(**vars(model_args))
  # load_state_dict(model, state_dict, strict=False)
  model.model.tok_embeddings.weight.assign(state_dict['tok_embeddings.weight'].to(model.model.tok_embeddings.weight.device)).realize()

  model.train()
