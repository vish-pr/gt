#!/usr/bin/env python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# print(tokenizer.encode("[INST]"))
# messages = []

# encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
# print(encodeds)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype='auto')
device = "cuda"  # the device to load the model onto

model.to(device)
b = 29360128
print(model.model.embed_tokens.weight[1213])
print(model.model.embed_tokens.weight.max())
# print(model.model.layers[1].mlp.gate_proj.weight.max())
# print(model.model.layers.1.mlp.gate_pro wj.weight)
# print(model.model.layers.1)
# encodeds = encodeds.to(device)
# for i in range(1000):
#   generated_ids = model(encodeds).logits
#   output = generated_ids[:, -1:].argmax(dim=2)
#   decoded = tokenizer.batch_decode(encodeds)
#   print(decoded)
#   encodeds = torch.cat([encodeds, output], dim=-1)
