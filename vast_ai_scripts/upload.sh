#!/usr/bin/zsh

# ssh -p $PORT $REMOTE mkdir -p /root/gt/tiny_stories/data/ /root/gt/tiny_stories/weights/ 
scp -P $PORT *py $REMOTE:/root/gt/
# scp -P $PORT tinygrad/tinygrad/runtime/*.py $REMOTE:/root/gt/tinygrad/tinygrad/runtime/
# scp -P $PORT tiny_stories/weights/tok512.model $REMOTE:/root/gt/tiny_stories/weights/
# scp -P $PORT tiny_stories/data/TinyStoriesV2-GPT4-valid.txt_\' $REMOTE:/root/gt/tiny_stories/data/
# scp -P $PORT tiny_stories/data/TinyStoriesV2-GPT4-train.txt_\' $REMOTE:/root/gt/tiny_stories/data/