#!/bin/bash
# Part 2 replay matrix (HANDOFF 9 fix 1): shapes x seeds x {none, --stall-rule}, sequential.
# Run from the repo root: nohup bash tools/run-part2.sh > runs/part2.log 2>&1 &
source ~/.nvm/nvm.sh
BASE="--target 1000000 --game v2 --pop 128 --episodes 4 --sigma 0.1 --lr 0.03 --rank-ties mid --decay decoupled --weight-decay 0.01 --generations 400 --time-budget 4 --eval-every 50"
for shape in 32,32 32,32,32; do
  tag=${shape//,/x}
  for seed in 1 2 3; do
    for arm in none stall; do
      extra=""; [ $arm = stall ] && extra="--stall-rule"
      dir=runs/exp-rep-$tag-s$seed-$arm
      mkdir -p $dir
      echo "$(date -Is) start $dir"
      node trainer/cli.js train --hidden $shape --seed $seed --dir $dir $BASE $extra >> $dir/train.log 2>&1
      echo "$(date -Is) end $dir exit=$?"
    done
  done
done
echo "$(date -Is) all done"
