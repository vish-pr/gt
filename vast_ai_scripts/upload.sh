#!/usr/bin/zsh

upload_first() {
  # scp -P $PORT vast_ai_scripts/one_time.sh $REMOTE:/root/
  ssh -p $PORT $REMOTE "bash -s" < vast_ai_scripts/one_time.sh
}
upall() {
  scp -P $PORT *py $REMOTE:/root/gt/
}
up() {
  scp -P $PORT $1 $REMOTE:/root/gt/
}