#!/bin/bash

export github="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"
# if current directory is not gt, clone gt
if [[ $(basename "$PWD") != "gt" && ! -d "gt" ]]; then
  git clone https://github.com/vish-pr/gt
  cd gt
fi
git pull

# install tinygrad
# if file not exist, clone tinygrad
if [ ! -d "tinygrad" ]; then
  git clone https://github.com/tinygrad/tinygrad.git
fi
cd tinygrad
git pull
python3 -m pip install -e .
cd ..


# used in tiny_stories tokenizer
pip install sentencepiece

# download tiny_stories
# if stories15M.pt file not exist, download it
if [ ! -f "tiny_stories/weights/stories15M.pt" ]; then
    wget -P tiny_stories/weights https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt
fi
if [ ! -f "tiny_stories/weights/tokenizer.model" ]; then
    wget -P tiny_stories/weights https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model
fi