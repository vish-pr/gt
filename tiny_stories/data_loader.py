import random
import time
from pathlib import Path
from typing import List

import numpy as np
from sentencepiece import SentencePieceProcessor

from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor


class DataLoader:
  BYTES_TO_READ = 1024 * 1024 * 100  # 100 MB

  def __init__(self, train_path: str, valid_path: str, batch_size: int, tokenizer: SentencePieceProcessor):
    self.tokenizer = tokenizer
    self.tokenizer_name = '_' + str(tokenizer.Decode(42))
    print('Tokenizer name', self.tokenizer_name)
    self.preprocess(valid_path)
    # self.preprocess(train_path)
    self.valid_data = self.get_data(valid_path)
    # self.train_data = self.get_data(train_path)
    self.batch_size = batch_size

  def preprocess(self, path, string='<|endoftext|>'):
    if Path(path + self.tokenizer_name).is_file():
      return
    start_time = time.time()
    buffer = ''
    delta_writing = 0
    delta_tokenizing = 0
    num_of_samples = 0
    with open(path, 'r') as f, open(path + self.tokenizer_name, 'w') as f2:
      index = 0
      while (chunk := f.read(self.BYTES_TO_READ)) or buffer:
        buffer += chunk if chunk else string
        time_start = time.time()
        chunk = []
        while (found_index := buffer.find(string, index)) != -1:
          num_of_samples += 1
          text = buffer[index: found_index]
          chunk.append(text.strip())
          index = found_index + len(string)
        chunk = self.tokenizer.Encode(chunk, add_eos=True)
        # samples.extend(self.tokenizer.Encode(samples_str))
        buffer = buffer[index:]
        index = 0
        time_end = time.time()
        delta_tokenizing += time_end - time_start
        time_start = time_end
        f2.write('\n'.join(' '.join(map(str, story)) for story in chunk))
        delta_writing += time.time() - time_start

        print(
            f'Loaded {num_of_samples} : Per sample {delta_tokenizing*1000 / num_of_samples:3f}ms tokenizing and {delta_writing*1000 / num_of_samples:3f}ms writing', end='\r')
    print(f'\nFinished preprocessing {num_of_samples} in seconds {time.time() - start_time}')

  def get_data(self, path) -> List[np.ndarray]:
    data = []
    path = path + self.tokenizer_name
    print(f'Loading {path}')
    with open(path, 'r') as f:
      while line := f.readline():
        data.append(np.fromstring(line, sep=' ', dtype=np.int32))
    return data

  def get_batch(self, train=True) -> Tensor:
    data = self.train_data if train else self.valid_data
    batch = random.choices(data, k=self.batch_size)
    max_length = max(len(row) for row in batch)
    return Tensor([np.pad(row, (0, max_length - len(row)), constant_values=self.tokenizer.eos_id()) for row in batch], dtype=dtypes.int32, requires_grad=False)
