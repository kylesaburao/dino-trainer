#!/bin/bash
# Hygiene seed survey, 32,32, sequential: bash tools/run-survey.sh <first seed> <last seed>
# Run from the repo root: nohup bash tools/run-survey.sh 13 30 > runs/exp-surv-batch.log 2>&1 &
source ~/.nvm/nvm.sh
BASE="--target 1000000 --game v2 --pop 128 --episodes 4 --sigma 0.1 --lr 0.03 --rank-ties mid --decay decoupled --weight-decay 0.01 --hidden 32,32 --generations 1500 --time-budget 5 --eval-every 50 --wasm"
for seed in $(seq $1 $2); do
  dir=runs/exp-surv-32x32-s$seed
  if [ -e $dir/checkpoint.json ]; then echo "$(date -Is) skip $dir (exists)"; continue; fi
  mkdir -p $dir
  echo "$(date -Is) start $dir"
  node trainer/cli.js train --seed $seed --dir $dir $BASE >> $dir/train.log 2>&1
  echo "$(date -Is) end $dir exit=$?"
done
echo "$(date -Is) all done"
