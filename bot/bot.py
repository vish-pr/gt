# * Observations passive inputs: all source code, searching web

# process

# output

import os
import sys

from tinygrad.helpers import Timing

sys.path.append(os.getcwd())   # fmt: off

# wait for user input
import asyncio
from dataclasses import dataclass
from typing import List

import websockets

from mistral import MistralModels


class Server:
  def __init__(self):
    self.tree = ConverstationTree()
    start_server = websockets.serve(self.handle_client, "0.0.0.0", 8080)
    asyncio.get_event_loop().run_until_complete(start_server)
    print("Listening")
    asyncio.get_event_loop().run_forever()

  async def handle_client(self, websocket, path):
    # Add the new client to the set
    async for message in websocket:
      await websocket.send(f"<br>Me: {message}")
      print('message', message)
      output_multi_stream = self.tree.process(message)
      await websocket.send(f"<br>Z: ")
      for output in output_multi_stream:
        
        with Timing("outter loop "):
          print(f'output ::{output}::')
          await websocket.send(output)

  async def on_connect(self, websocket, path):
    self.clients = websocket

  async def on_disconnect(self, websocket, path):
    self.client = None


@dataclass
class Node:
  parent: int
  children: List[int]
  text: str
  input: bool  # else output


class ConverstationTree:
  # create a node for a tree struct

  def __init__(self):
    self.trees: List[Node] = []
    self.model = MistralModels()
    self.cur = -1

  def add_node(self, text: str, input: bool):
    # add a node to the tree
    self.trees.append(Node(self.cur, [], text, input))
    return len(self.trees) - 1

  def process(self, inp):
    self.cur = self.add_node(inp, True)
    # outputs: List[str] = self.model.process(inp)
    # branches = [self.add_node(output, False) for output in outputs]
    output_stream = self.model.process(inp)
    for output in output_stream:
      self.cur = self.add_node(output, False)
      yield output

server = Server()