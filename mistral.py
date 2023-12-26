#!/usr/bin/env python3

import os
import sys
import traceback
from contextlib import redirect_stdout
from io import StringIO
from typing import Dict

from extra.models.llama import Transformer
from sentencepiece import SentencePieceProcessor

from llama import Transformer
from tinygrad import Device, Tensor, dtypes, nn
from tinygrad.helpers import Timing, colored, fetch, getenv
from tinygrad.tensor import Tensor

sys.path.append(os.getcwd())


def create_fixed_tokenizer(output_file):
  print("creating fixed tokenizer")
  import extra.junk.sentencepiece_model_pb2 as spb2
  mp = spb2.ModelProto()
  mp.ParseFromString(fetch("https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/resolve/main/tokenizer.model?download=true").read_bytes())
  mp.pieces.append(spb2.ModelProto.SentencePiece(piece="<|im_end|>", score=0))
  mp.pieces.append(spb2.ModelProto.SentencePiece(piece="<|im_start|>", score=0))
  with open(output_file, "wb") as f:
    f.write(mp.SerializeToString())


config_istruct = {
  "name": "mistral-7b-instruct-v02",
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 32768,
  # "model_type": "mistral",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_theta": 1000000.0,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.36.0",
  "use_cache": True,
  "vocab_size": 32000,
  "urls": ['https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/pytorch_model-00001-of-00003.bin',
           'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/pytorch_model-00002-of-00003.bin',
           'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/pytorch_model-00003-of-00003.bin'],
  "tokenizer_url": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/tokenizer.model",
}

config = {
  "name": "mistral-7b-v01",
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 32768,
  # "model_type": "mistral",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0,
  "sliding_window": 4096,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.36.0",
  "use_cache": True,
  "vocab_size": 32000,
  "urls": ['https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00001-of-00002.bin',
           'https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00002-of-00002.bin'],
  "tokenizer_url": "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.model",
}


def convert_from_huggingface(weights: Dict[str, Tensor], model: Transformer, n_heads: int, n_kv_heads: int):
  def permute(v: Tensor, n_heads: int):
    return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1]).transpose(1, 2).reshape(*v.shape[:2])

  # TODO: make it only regexp based, use no params
  keymap = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    **{f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attention_norm.weight" for l in range(len(model.layers))},
    **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"layers.{l}.attention.w{x}.weight" for x in ["q", "k", "v", "o"] for l in range(len(model.layers))},
    **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight" for l in range(len(model.layers))},
    **{f"model.layers.{l}.mlp.{x}_proj.weight": f"layers.{l}.feed_forward.w{y}.weight" for x, y in {"gate": "1", "down": "2", "up": "3"}.items() for l in range(len(model.layers))},
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
  }
  sd = {}
  for k, v in weights.items():
    if ".rotary_emb." in k:
      continue
    v = v.to(Device.DEFAULT)
    if "model.layers" in k:
      if "q_proj" in k:
        v = permute(v, n_heads)
      elif "k_proj" in k:
        v = permute(v, n_kv_heads)
    sd[keymap[k]] = v
  return sd


a, b = Tensor([]), Tensor([])


def download_model():
  # fix bf16, TODO: check if device supports bf16
  def fix_bf16(weights): return {k: v.to('cpu').cast(dtypes.float16) if v.dtype == dtypes.bfloat16 else v for k, v in weights.items()}

  model = Transformer(dim=config["hidden_size"], hidden_dim=config["intermediate_size"], n_heads=config["num_attention_heads"], n_kv_heads=config["num_key_value_heads"],
                      n_layers=config["num_hidden_layers"], vocab_size=config["vocab_size"], norm_eps=config["rms_norm_eps"], max_seq_len=config["max_position_embeddings"])
  with Timing("weights -> model: "):
    # nn.state.load_state_dict(model, fix_bf16(convert_from_huggingface(part1, model, 32, 8)), strict=False)
    # nn.state.load_state_dict(model, fix_bf16(convert_from_huggingface(part2, model, 32, 8)), strict=False)
    for url in config["urls"]:
      filename = "weights/" + config["name"] + '-' + url.split("/")[-1]
      weights = convert_from_huggingface(nn.state.torch_load(str(fetch(url, filename))), model,
                                         config["num_attention_heads"], config["num_key_value_heads"])
      global a, b
      if weights.get("output.weight"):
        b = weights["output.weight"]
      if weights.get("tok_embeddings.weight"):
        a = weights["tok_embeddings.weight"]
      # for k, v in weights.items():
      #   print(k, v.device, v.dtype)
      # weights = nn.state.load_state_dict(model, weights, strict=False)
  return model


# if not os.path.isfile("/tmp/tokenizer.model"):
#   create_fixed_tokenizer("/tmp/tokenizer.model")
# spp = SentencePieceProcessor(model_file="/tmp/tokenizer.model")


model = download_model()
# a = a.to(Device.DEFAULT).cast(dtypes.float16).realize()
# b = b.to(Device.DEFAULT).cast(dtypes.float16).realize()
# a = a.realize()
# b = b.realize()
# a = a.lazydata.realized.bitcast(dtypes.uint16).realize()
b = b.contiguous().bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1 << 16).contiguous().bitcast(dtypes.float32)
print(b.numpy())
print(a.numpy(), b.numpy())
c = a - b
# c = c.cast(dtypes.float16)
print(c.max().numpy(), c.min().numpy(), c.mean().numpy(), c.std().numpy())

spp = SentencePieceProcessor()
spp.LoadFromFile(str(fetch(config["tokenizer_url"], "weights/" + config["name"] + "-tokenizer.model")))
print(spp.__len__())
print(spp.Decode(3232))
print(spp.Encode(["[INST]"]))
print(spp.Encode([]))

start_pos = 0
toks = []
temperature = 0.7
while 1:
  Tensor.no_grad = True
  tok = model(Tensor([toks[start_pos:]]), start_pos, temperature).multinomial().item()
  start_pos = len(toks)
  toks.append(tok)
  print(spp.Decode(toks))
# if __name__ == "__main__":
#   Tensor.no_grad = True

#   # https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/config.json
#   with Timing("create model: "):
#     model = Transformer(4096, 14336, n_heads=32, n_layers=32, norm_eps=1e-5, vocab_size=32002, n_kv_heads=8, max_context=4096)

#   with Timing("download weights: "):
#     part1 = nn.state.torch_load(
#       fetch("https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/resolve/main/pytorch_model-00001-of-00002.bin?download=true"))
#     part2 = nn.state.torch_load(
#       fetch("https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/resolve/main/pytorch_model-00002-of-00002.bin?download=true"))

#   with Timing("weights -> model: "):
#     nn.state.load_state_dict(model, fix_bf16(convert_from_huggingface(part1, model, 32, 8)), strict=False)
#     nn.state.load_state_dict(model, fix_bf16(convert_from_huggingface(part2, model, 32, 8)), strict=False)

#   if not os.path.isfile("/tmp/tokenizer.model"):
#     create_fixed_tokenizer("/tmp/tokenizer.model")
#   # spp = SentencePieceProcessor(model_file=fetch(config["tokenizer_url"], "/weights/" + config["name"] + "-tokenizer.model"))

#   # https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/tokenizer_config.json
#   #   "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
#   IM_END = 32000
#   IM_START = 32001
# def encode_prompt(k, v): return [IM_START] + spp.encode(f"{k}\n{v}") + [IM_END] + spp.encode("\n")
# def start_prompt(k): return [IM_START] + spp.encode(f"{k}\n")

#   def output(outputted, toks, color):
#     cur = spp.decode(toks)[len(outputted):]
#     sys.stdout.write(colored(cur, color))
#     sys.stdout.flush()
#     outputted += cur
#     return outputted

#   # *** app below this line ***

  toks = [spp.bos_id()] + encode_prompt("system",
                                        "You are Quentin. Quentin is a useful assistant who writes Python code to answer questions. He keeps the code as short as possible and doesn't read from user input")

  PROMPT = getenv("PROMPT", 1)
  temperature = getenv("TEMP", 0.7)

  outputted = output("", toks, "green")
  turn = True
  while 1:
    if PROMPT:
      toks += encode_prompt("user", input("Q: ")) + start_prompt("assistant")
    else:
      toks += start_prompt("user" if turn else "assistant")
      turn = not turn
    old_output_len = len(outputted)
    while 1:
      tok = model(Tensor([toks[start_pos:]]), start_pos, temperature).multinomial().item()
      start_pos = len(toks)
      toks.append(tok)
      outputted = output(outputted, toks, "blue" if not turn else "cyan")
      if tok == IM_END:
        break
      if tok == spp.eos_id():
        break
      new_output = outputted[old_output_len:]

      if new_output.endswith("```") and '```python\n' in new_output:
        python_code = new_output.split('```python\n')[1].split("```")[0]
        # AI safety. Warning to user. Do not press y if the AI is trying to do unsafe things.
        if input(colored(f" <-- PYTHON DETECTED, RUN IT? ", "red")).lower() == 'y':
          my_stdout = StringIO()
          try:
            with redirect_stdout(my_stdout):
              exec(python_code)
            result = my_stdout.getvalue()
          except Exception as e:
            result = ''.join(traceback.format_exception_only(e))
          toks += spp.encode(f"\nOutput:\n```\n{result}```")
          outputted = output(outputted, toks, "yellow")
          old_output_len = len(outputted)
    print("")
