from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generator, List, Union

from layers import Transformer
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor


# create a enum
class Speaker(Enum):
  USER = 0
  ASSISTANT = 1
  SEARCH = 2


@dataclass
class Node:
  parent: int
  children: List[int]
  data: List[int]
  speaker: Speaker


class ConverstationTree:
  # TODO: may also store model cache as heavy-light decompositions

  def __init__(self):
    self.trees: List[Node] = []
    self.current_node_id: int = -1

  def add_node(self, data: List[int], speaker: Speaker):
    self.trees.append(Node(self.current_node_id, [], data, speaker))
    child_id = len(self.trees) - 1
    self.trees[self.current_node_id].children.append(child_id)
    self.current_node_id = child_id
    return child_id


class LanguageModel:  # this is generic language model, and leaves individual models to be implemented

  def __init__(self, model: Transformer):
    self.model = model
    self.start_pos: int = 0
    self.conversation = ConverstationTree()
    Tensor.no_grad = True

  @abstractmethod
  def input_to_tokens(self, inp: str) -> List[int]:
    pass

  @abstractmethod
  def token_to_string(self, token: int) -> Union[str, None]:
    pass

  def process(self, inp) -> Generator[str, None, None]:
    tokens = self.input_to_tokens(inp)
    self.conversation.add_node(tokens, Speaker.USER)
    x = Tensor([tokens], dtype=dtypes.int32)
    length = len(tokens)
    for _ in range(500):
      x = self.model(x, self.start_pos)
      self.start_pos += length
      length = 1
      op: int = int(x.item())
      self.conversation.add_node([op], Speaker.ASSISTANT)
      str_op = self.token_to_string(op)
      print(str_op)
      if str_op:
        yield str_op
      else:
        break
