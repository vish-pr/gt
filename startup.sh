#!/bin/bash

# if current directory is not gt, clone gt
if [[ $(basename "$PWD") != "gt" ]]; then
  git clone https://github.com/vish-pr/gt
  cd gt
fi
git pull

# install tinygrad
# if file not exist, clone tinygrad
if [ ! -f "tinygrad/setup.py" ]; then
  git clone https://github.com/tinygrad/tinygrad.git
fi
cd tinygrad
git pull
python3 -m pip install -e .
cd ..
pip install pytest
pip install pycuda

# used in tiny_stories tokenizer
pip install sentencepiece
# download tiny_stories
if [ ! -f "tiny_stories/data/TinyStoriesV2-GPT4-valid.txt" ]; then
    wget -P tiny_stories/data https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
fi
if [ ! -f "tiny_stories/weights/stories260K.pt" ]; then
    wget -P tiny_stories/weights https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.pt
fi
if [ ! -f "tiny_stories/weights/stories15M.pt" ]; then
    wget -P tiny_stories/weights https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt
fi
if [ ! -f "tiny_stories/weights/tok512.model" ]; then
    wget -P tiny_stories/weights https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.model
fi
if [ ! -f "tiny_stories/weights/tokenizer.model" ]; then
    wget -P tiny_stories/weights https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model
fi
if [ ! -f "tiny_stories/data/TinyStoriesV2-GPT4-train.txt" ]; then
    wget -P tiny_stories/data https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
fi



# in tinygrad GRAPH=1 needs it.
pip install pydot
sudo apt-get install graphviz
