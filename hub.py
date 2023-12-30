import os
from typing import Dict

from transformers import AutoTokenizer

from tinygrad import Device, Tensor, dtypes, nn
from tinygrad.helpers import Timing, fetch

from llama import Transformer


def download_tokenizer(config):
  # TODO write a test to use simple tokenizer like sentencepiece or custom, same as hub tokenizer and then use that
  # verify in test that tokenizer exactly matches hub tokenizer, and has same speed
  tokenizer = AutoTokenizer.from_pretrained(config["name"])
  return tokenizer


def download_model(model, config):
  output_file = "weights/" + config["name"]
  if os.path.isfile(output_file):
    with Timing("loading float16 cache:"):
      state_dict = nn.state.safe_load(output_file)
      sd = {}
      for k, v in state_dict.items():
        if 'freq' in k:
          continue
        sd[k] = v
        # print(k, v.shape)
      nn.state.load_state_dict(model, sd, strict=False)
  else:
    with Timing("creating cache: "):
      for url in config["urls"]:
        filename = "weights/" + config["name"] + '-' + url.split("/")[-1]
        weights = convert_from_huggingface(fix_bf16(nn.state.torch_load(str(fetch(url, filename)))), len(
          model.layers), model.layers[0].attention.n_heads, model.layers[0].attention.n_kv_heads)
        # TODO: Verify bf16 to float16 conversion is not overflows or underflows, by writing test to diff max and min from this and huggingface
        nn.state.load_state_dict(model, weights, strict=False)
      output_file = "weights/" + config["name"]
      nn.state.safe_save(nn.state.get_state_dict(model), output_file)
      for url in config["urls"]:
        filename = "weights/" + config["name"] + '-' + url.split("/")[-1]
        os.remove(filename)
  return model


def fix_bf16(weights):
  def fix(a):
    a = a.bitcast(dtypes.uint16).realize()
    a = a.to('CPU').realize()
    a = a.cast(dtype=dtypes.uint32).realize()
    a = a.mul(1 << 16).realize().contiguous().bitcast(dtypes.float32).cast(dtypes.float16)
    a = a.to(Device.DEFAULT).realize()
    return a
  return {k: fix(v) if v.dtype == dtypes.bfloat16 else v for k, v in weights.items()}


def convert_from_huggingface(weights: Dict[str, Tensor], n_layers: int, n_heads: int, n_kv_heads: int):
  def permute(v: Tensor, n_heads: int):
    return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1]).transpose(1, 2).reshape(*v.shape[:2])

  # TODO: make it only regexp based, use no params
  keymap = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    **{f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attention_norm.weight" for l in range(n_layers)},
    **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"layers.{l}.attention.w{x}.weight" for x in ["q", "k", "v", "o"] for l in range(n_layers)},
    **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight" for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.{x}_proj.weight": f"layers.{l}.feed_forward.w{y}.weight" for x, y in {"gate": "1", "down": "2", "up": "3"}.items() for l in range(n_layers)},
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
  }
  sd = {}
  for k, v in weights.items():
    print(k, v.shape)
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


config_mistral = {
  "name": "mistralai/Mistral-7B-v0.1",
  "model": Transformer(dim=4096,
                       hidden_dim=14336,
                       n_heads=32,
                       n_kv_heads=8,
                       n_layers=32,
                       vocab_size=32_000,
                       norm_eps=1e-05,
                       max_seq_len=128_000),
  # "bos_token_id": 1,
  # "eos_token_id": 2,
  # "hidden_act": "silu",
  # "hidden_size": 4096,
  # # "initializer_range": 0.02,
  # "intermediate_size": 14336,
  # "max_position_embeddings": 32768,
  # # "model_type": "mistral",
  # "num_attention_heads": 32,
  # "num_hidden_layers": 32,
  # "num_key_value_heads": 8,
  # "rms_norm_eps": 1e-05,
  # "rope_theta": 10000.0,
  # # "sliding_window": 4096,
  # "rope_theta": 10000.0,
  # "rope_length": 128_000,
  # "torch_dtype": "bfloat16",
  # "use_cache": True,
  # "vocab_size": 32000,
  "urls": ['https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00001-of-00002.bin',
           'https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model-00002-of-00002.bin'],
  "tokenizer_url": "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.model",
}

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
