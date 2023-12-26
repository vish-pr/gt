#!/usr/bin/env python3

import os
import subprocess

subprocess.run(["pip", "install", "pytest"])
  # in tinygrad GRAPH=1 needs it.
# subprocess.run(["pip", "install", "pydot"])
# subprocess.run(["sudo", "apt", "install", "graphviz"])

# download tiny_stories
if not os.path.isfile("tiny_stories/data/TinyStoriesV2-GPT4-valid.txt"):
  subprocess.run(["wget", "-P", "tiny_stories/data", "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"])
if not os.path.isfile("tiny_stories/weights/stories260K.pt"):
  subprocess.run(["wget", "-P", "tiny_stories/weights", "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.pt"])
if not os.path.isfile("tiny_stories/weights/stories15M.pt"):
  subprocess.run(["wget", "-P", "tiny_stories/weights", "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt"])
if not os.path.isfile("tiny_stories/weights/tok512.model"):
  subprocess.run(["wget", "-P", "tiny_stories/weights", "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.model"])
if not os.path.isfile("tiny_stories/weights/tokenizer.model"):
  subprocess.run(["wget", "-P", "tiny_stories/weights", "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"])
if not os.path.isfile("tiny_stories/data/TinyStoriesV2-GPT4-train.txt"):
  subprocess.run(["wget", "-P", "tiny_stories/data", "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"])
