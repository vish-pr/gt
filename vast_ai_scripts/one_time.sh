#!/bin/bash

sudo apt-get update
sudo apt-get install -y neovim
cd ~
git clone https://www.github.com/vish-pr/gt
pip install sentencepiece transformers openai websockets
cd gt
git pull
git clone https://www.github.com/tinygrad/tinygrad
cd tinygrad
python3 -m pip install -e .
cd ..