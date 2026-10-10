#!/bin/bash
# Run B' vs hygiene batch (research/prereg-es-curriculum.md): resume each frozen start
# checkpoint in both arms, one trainer at a time, stopping each run at the validation event.
# Run from the repo root: nohup bash tools/run-bp.sh > runs/exp-bp-batch.log 2>&1 &
source ~/.nvm/nvm.sh
START=runs/exp-bp-start
(cd $START && sha256sum -c --quiet SHA256SUMS) || { echo "start checkpoint checksum mismatch"; exit 1; }
declare -A G0=([4]=1500 [5]=1435 [7]=1500 [8]=1500 [9]=1017 [12]=1500)
COMMON="--wasm --time-budget 30 --eval-every 10 --val-seeds 12"
CUR="--window 25000 --episodes 8 --start-dist points --start-zero-frac 0.5 --start-zero-cap 50000 --window-adapt --fitness-agg geomean"
for s in 12 4 5 7 8 9; do
  for arm in hyg cur; do
    dir=runs/exp-bp-s$s-$arm
    if [ -e $dir/checkpoint.json ]; then echo "$(date -Is) skip $dir (exists)"; continue; fi
    mkdir -p $dir && cp $START/ckpt-s$s.json $dir/checkpoint.json
    extra=""; [ $arm = cur ] && extra="$CUR"
    echo "$(date -Is) start $dir (from gen ${G0[$s]})"
    node trainer/cli.js train --resume --dir $dir --generations $(( ${G0[$s]} + 2000 )) $COMMON $extra >> $dir/train.log 2>&1 &
    tp=$!
    node tools/event-watch.js $dir --after ${G0[$s]} >> $dir/watch.log 2>&1 &
    wp=$!
    wait $tp; echo "$(date -Is) end $dir exit=$?"
    kill $wp 2>/dev/null; wait $wp 2>/dev/null
  done
done
echo "$(date -Is) all done"
