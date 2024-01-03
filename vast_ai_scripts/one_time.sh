#!/bin/bash

sudo apt-get update
sudo apt-get install -y neovim
git clone https://www.github.com/vish-pr/gt
pip install pycuda sentencepiece transformers openai
cd gt
git clone https://www.github.com/tinygrad/tinygrad
cd tinygrad
python3 -m pip install -e .
cd ..