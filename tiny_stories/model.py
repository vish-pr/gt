from pathlib import Path
from time import time

from data_loader import DataLoader
from sentencepiece import SentencePieceProcessor

from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters, load_state_dict, torch_load
from tinygrad.tensor import Tensor

from llama import ModelArgs, Transformer


class Model:
  def __init__(self, tokenizer_path, model_args=None, checkpoint_path=None):
    assert model_args is not None or checkpoint_path is not None
    self.model = self.load_model(checkpoint_path) if checkpoint_path else Transformer(**vars(model_args))
    assert Path(tokenizer_path).is_file()
    self.tokenizer = SentencePieceProcessor()
    self.tokenizer.LoadFromFile(tokenizer_path)

  def run_with_loss(self, x: Tensor):
    logits = self.model(x)
    return logits, x[:, 1:].sparse_categorical_crossentropy(logits[:, :-1, :])
    # self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

  def train(self):
    data_loader = DataLoader('tiny_stories/data/TinyStoriesV2-GPT4-train.txt',
                             'tiny_stories/data/TinyStoriesV2-GPT4-valid.txt', 1, self.tokenizer)
    optim = SGD(get_parameters(self.model), lr=0.01)
    for i in range(10):
      x = data_loader.get_batch()
      logits, loss = self.run_with_loss(x)
      optim.zero_grad()
      loss.backward()
      optim.step()
      # print(loss.item())

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
    load_state_dict(model, state_dict, strict=False)
    dummy_x = Tensor.zeros(1, 1)
    start_time = time()
    model(dummy_x).realize()
    print(f'dummy run in {((time() - start_time)*1000):.2f}ms')
    return model


if __name__ == '__main__':
  # checkpoint_path = 'tiny_stories/weights/stories15M.pt'
  checkpoint_path = 'tiny_stories/weights/stories260K.pt'
  # tokenizer_path = 'tiny_stories/weights/tokenizer.model'
  tokenizer_path = 'tiny_stories/weights/tok512.model'
  assert Path(tokenizer_path).is_file()
  model = Model(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path)
  op, loss = model('Once upon a time, in a forest', True)
  print(op)
  model = Model(tokenizer_path=tokenizer_path, model_args=ModelArgs())
  model.train()

  # for i in range(100):
  #   next = model(Tensor([tokens]), start_pos=0).realize()
  #   next = next.squeeze(0)
  #   next = next.argmax(1).numpy().astype(int).tolist()
  #   print(next)
  #   assert len(next) == len(tokens)
  #   tokens.append(next[-1])
  # print(enc.decode(next.argmax(1).numpy().astype(int).tolist()))
