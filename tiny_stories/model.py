#!/usr/bin/env python3

from pathlib import Path
from time import time
from typing import List

from arrow import get
from data_loader import DataLoader
from llama import ModelArgs, Transformer
from sentencepiece import SentencePieceProcessor

from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.graph import print_tree
from tinygrad.nn.optim import SGD, Optimizer
from tinygrad.nn.state import (get_parameters, get_state_dict, load_state_dict,
                               torch_load)
from tinygrad.tensor import Tensor


class Opt(Optimizer):
  def __init__(self, model: Transformer, lr=0.001, momentum=0, weight_decay=0.0, nesterov=False):
    # params = self._get_params(model)
    params = get_parameters(model)
    super().__init__(params, lr)
    self.momentum, self.wd, self.nesterov = momentum, weight_decay, nesterov
    self.b = [Tensor.zeros(*t.shape, device=t.device, requires_grad=False) for t in self.params] if self.momentum else []

  def _get_params(self, model: Transformer) -> List[Tensor]:

    params = []
    for k, v in get_state_dict(model).items():
      if k.startswith('layers.0'):
        params.append(v)
    print(len(params))
    return params

  def step(self) -> None:
    for i, t in enumerate(self.params):
      assert t.grad is not None
      g = t.grad.realize() + self.wd * t.detach()
      if self.momentum:
        self.b[i].assign(self.momentum * self.b[i] + g).realize()  # NOTE: self.b[i] is zero on the first run, no if required
        g = (g + self.momentum * self.b[i]) if self.nesterov else self.b[i]
      t.assign(t.detach() - g * self.lr)
    self.realize(self.b)


class Model:
  def __init__(self, tokenizer_path, model_args=None, checkpoint_path=None):
    assert model_args is not None or checkpoint_path is not None
    self.model = self.load_model(checkpoint_path) if checkpoint_path else Transformer(**vars(model_args))
    assert Path(tokenizer_path).is_file()
    self.tokenizer = SentencePieceProcessor()
    self.tokenizer.LoadFromFile(tokenizer_path)
    self.data_loader = DataLoader('tiny_stories/data/TinyStoriesV2-GPT4-train.txt',
                                  'tiny_stories/data/TinyStoriesV2-GPT4-valid.txt', 3, self.tokenizer)
    # self.model = Transformer(64, self.tokenizer.vocab_size())

  def run_with_loss(self, x: Tensor):
    logits = self.model(x)
    return logits, logits[:, :-1:].sparse_categorical_crossentropy(x[:, 1::])

  def train(self):
    optim = SGD([self.model.tok_embeddings.weight, self.model.output.weight], lr=3e-4)
    with Tensor.train():
      for i in range(10):
        x = self.data_loader.get_batch(False)
        # x = x.detach()
        logits, loss = self.run_with_loss(x)
        # sched_items = logits.lazydata.schedule()  # for i, s in enumerate(sched_items):
        # for i, s in enumerate(sched_items):
        #   print(f'********** {i} **********')
        #   print(s.ast.__repr__())
        #   lin = Linearizer(s.ast, LinearizerOptions(device='cpu'))
        #   lin.linearize()
        #   print(lin.display_name)
        #   print(lin.function_name)
        #   for u in lin.uops:
        #     print(u)

        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss.numpy())

  def validate(self):
    Tensor.training = False
    for param in get_parameters(self.model):
      print(param)
    for i in range(10):
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

  def load_model(self, checkpoint_path):
    assert Path(checkpoint_path).is_file()
    check_point = torch_load(checkpoint_path)
    state_dict = check_point['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
      if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model_args = ModelArgs(**check_point['model_args'])
    model = Transformer(**vars(model_args))
    # load_state_dict(model, state_dict, strict=False)
    return model


if __name__ == '__main__':
  # checkpoint_path = 'tiny_stories/weights/stories15M.pt'
  checkpoint_path = 'tiny_stories/weights/stories260K.pt'
  # tokenizer_path = 'tiny_stories/weights/tokenizer.model'
  tokenizer_path = 'tiny_stories/weights/tok512.model'
  assert Path(tokenizer_path).is_file()
  model = Model(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path)
  for i in range(1):
    op, loss = model('Once upon a time, there was a king. He was a big and strong king who ruled over his kingdom. \
    One day, he wanted to take a nice and long bath, so he filled up his big bathtub with warm wa\
    ter. He wanted to feel relaxed and so he soaked in the tub for a really long time.\
    When he had finished soaking and stepped out of the bathtub, the king noticed that the water \
    had spilled out of the tub and all over the floor. He felt guilty that he had made such a mes\
    s, so he quickly grabbed a cloth and began to clean it up.\
    The king got so hot from cleaning up the mess that he decided to take another soak in the bat\
    htub. He put a lot of bubbles in the water to make it nice and bubbly. He relaxed again and f\
    elt all the worries wash away.\
    The king was so happy that he had been able to clean up the mess he had made and enjoy a nice\
    soak. He dried off and wrapped himself up in a big towel. Then, the king went back to ruling\
    his kingdom and enjoying his lovely baths.', False)
    print(op)
  # model_args = ModelArgs(dim=8, hidden_dim=16, n_heads=2, n_layers=4)
  # model = Model(tokenizer_path=tokenizer_path, model_args=model_args)
  model.train()

  # for i in range(100):
  #   next = model(Tensor([tokens]), start_pos=0).realize()
  #   next = next.squeeze(0)
  #   next = next.argmax(1).numpy().astype(int).tolist()
  #   print(next)
  #   assert len(next) == len(tokens)
  #   tokens.append(next[-1])
  # print(enc.decode(next.argmax(1).numpy().astype(int).tolist()))
