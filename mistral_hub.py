#!/usr/bin/env python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
print(tokenizer.encode("[INST]"))
messages = []

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
print(encodeds)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype='auto')
device = "cuda"  # the device to load the model onto

print(model.model.embed_tokens.weight.get_device())
print(model.model.embed_tokens.weight.dtype)
model.to(device)
encodeds = encodeds.to(device)
for i in range(1000):
  generated_ids = model(encodeds).logits
  output = generated_ids[:, -1:].argmax(dim=2)
  decoded = tokenizer.batch_decode(encodeds)
  print(decoded)
  encodeds = torch.cat([encodeds, output], dim=-1)
print(model.model.embed_tokens.weight.get_device())
print(model.model.embed_tokens.weight.dtype)
