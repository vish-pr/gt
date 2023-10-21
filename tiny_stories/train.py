from data_loader import DataLoader
from model import ModelArgs, Transformer as Model


class Train:
  def _build_model(model_args=ModelArgs):
    model = Model(**vars(model_args))
    return model

  def __init__(self, train_path, valid_path, batch_size: int, tokenizer_path):
    self.data_loader = DataLoader(train_path, valid_path, batch_size, tokenizer_path)
    self.model = self._build_model()


def train():
  # read tiny_stories/data/TinyStoriesV2-GPT4-train.txt
  data_loader.preprocess('tiny_stories/data/TinyStoriesV2-GPT4-valid.txt')
  exit()


tokenizer_path = 'tiny_stories/weights/tokenizer.model'
# tokenizer_path = 'tiny_stories/weights/tok512.model'
self.data_loader = DataLoader('tiny_stories/data/TinyStoriesV2-GPT4-train.txt',
                              'tiny_stories/data/TinyStoriesV2-GPT4-valid.txt', 2, tokenizer_path)
