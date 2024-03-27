import os
import sys
import time
sys.path.append(os.getcwd())   # fmt: off
from mistral import MistralModels
import asyncio
import websockets


class Server:
  def __init__(self):
    self.model = MistralModels()
    start_server = websockets.serve(self.handle_client, "0.0.0.0", 8080)
    asyncio.get_event_loop().run_until_complete(start_server)
    print("Listening")
    asyncio.get_event_loop().run_forever()

  async def handle_client(self, websocket, path):
    # Add the new client to the set
    async for message in websocket:
      await websocket.send(f"<br>Me: {message}")
      print('message', message)
      output_multi_stream = self.model.process(message)
      await websocket.send(f"<br>Z: ")
      tokens=0
      st = None
      for output in output_multi_stream:
        await websocket.send(output)
        tokens += 1
        if not st and tokens == 4:
          tokens = 0
          st = time.perf_counter_ns()
      if st:
        en = time.perf_counter_ns() - st
        print(f'{en*1e-6 / tokens:6.2f} ms per token')

  async def on_connect(self, websocket, path):
    self.clients = websocket

  async def on_disconnect(self, websocket, path):
    self.client = None


server = Server()