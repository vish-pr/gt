# install tinygrad
# if file not exist, clone tinygrad
if [ ! -d "tinygrad" ]; then
    git clone https://github.com/tinygrad/tinygrad.git
fi
cd tinygrad
git pull
python3 -m pip install -e .
cd ..

pip install sentencepiece

# download tiny_stories
# if stories15M.pt file not exist, download it
if [ ! -f "tiny_stories/weights/stories15M.pt" ]; then
    wget -P tiny_stories/weights https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt
fi
if [ ! -f "tiny_stories/weights/tokenizer.model" ]; then
    wget -P tiny_stories/weights https://github.com/karpathy/llama2.c/blob/master/tokenizer.model
fi
