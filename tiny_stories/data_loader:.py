from pathlib import Path

from dataclasses import dataclass
import pickle
import random
import time
from typing import List
import numpy as np
from sentencepiece import SentencePieceProcessor
from tinygrad.tensor import Tensor


@dataclass
class Sample:
    offset_bytes: int
    length_bytes: int


class DataLoader:
  BYTES_TO_READ = 1024 * 1024 * 100  # 100 MB

  def __init__(self, train_path, valid_path, batch_size: int, tokenizer: SentencePieceProcessor):
    self.batch_size: int = batch_size
    self.tokenizer = tokenizer
    tokenizer.GetScore
    self.train_samples = self.preprocess(train_path)
    self.valid_samples = self.preprocess(valid_path)
    self.train_path = train_path
    self.valid_path = valid_path

  def process_sample(self, sample: str):
    # ending is important to guess.
    str = sample.strip() + chr(10)
    return str

  def preprocess(self, path, string='<|endoftext|>') -> List[Sample]:
    # read data in chunks
    buffer = ''
    index = 0
    samples = []
    # if file exists, return samples
    if Path(path + '_processed_offsets').is_file():
      with open(path + '_processed_offsets', 'rb') as f:
        return pickle.load(f)
    delta_writing = 0
    delta_tokenizing = 0
    num_of_samples = 0
    with open(path, 'r') as f, open(path + '_processed', 'w') as f2:
      while (chunk := f.read(self.BYTES_TO_READ)) or buffer:
        buffer += chunk
        if not chunk:
          buffer += string
        inputs = []
        while (found_index := buffer.find(string, index)) != -1:
          num_of_samples += 1
          sample = buffer[index: found_index]
          sample = self.process_sample(sample)
          inputs.append(sample)
          index = found_index + len(string)
        time_start = time.time()
        inputs = self.tokenizer.Encode(inputs)
        # print(sample)
        time_end = time.time()
        delta_tokenizing += time_end - time_start
        time_start = time_end
        for input in inputs:
          offset_before = f2.tell()
          # f2.write(str(input))
          f2.write(' '.join(map(str, input)))
          # f2.write(sample)
          offset_after = f2.tell()
          samples.append(Sample(
            offset_before, offset_after - offset_before))
        time_end = time.time()
        delta_writing += time_end - time_start
        print(
            f'Loaded {num_of_samples}: Per sample {delta_tokenizing*1000 / num_of_samples:3f}ms tokenizing and {delta_writing*1000 / num_of_samples:3f}ms writing', end='\r')
        buffer = buffer[index:]
        index = 0
    with open(path + '_processed_offsets', 'wb') as f:
      pickle.dump(samples, f)
    return samples

  def get_batch(self, train=True) -> Tensor:
    samples = self.train_samples if train else self.valid_samples
    path = self.train_path if train else self.valid_path
    select_sampels = random.sample(samples, self.batch_size)
    batch = []
    max_len = 0
    with open(path + '_processed', 'r') as f:
      for sample in select_sampels:
        f.seek(sample.offset_bytes)
        batch.append(np.fromstring(f.read(sample.length_bytes), sep=' ', dtype=np.int32))
        max_len = max(max_len, len(batch[-1]))
    for i in range(len(batch)):
      batch[i] = np.pad(batch[i], (0, max_len - len(batch[i])),
                        'constant', constant_values=self.tokenizer.eos_id())
    return Tensor(np.array(batch))


tokenizer_path = 'tiny_stories/weights/tokenizer.model'
# tokenizer_path = 'tiny_stories/weights/tok512.model'
assert Path(tokenizer_path).is_file()
sp = SentencePieceProcessor()
sp.LoadFromFile(tokenizer_path)
data_loader = DataLoader('tiny_stories/data/TinyStoriesV2-GPT4-train.txt',
                         'tiny_stories/data/TinyStoriesV2-GPT4-valid.txt', 10, sp)
print(data_loader.get_batch().numpy())
