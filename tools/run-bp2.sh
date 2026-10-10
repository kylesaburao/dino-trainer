#!/bin/bash
# Confirmation round (research/prereg-es-curriculum.md §10): resume each frozen start θ in
# each arm, one trainer at a time; the watcher stops at the confirmed event or at the frame
# budget. Start checkpoints: runs/exp-bp2-start/ckpt-sN.json, else runs/exp-bp-start/.
# Usage (repo root): nohup bash tools/run-bp2.sh "abl cur" "12 4 13 14 ..." > runs/exp-bp2-batch.log 2>&1 &
source ~/.nvm/nvm.sh
ARMS=$1; SEEDS=$2
COMMON="--wasm --time-budget 45 --eval-every 10 --val-seeds 12"
declare -A EXTRA=(
  [abl]="--episodes 8"
  [cur]="--window 25000 --episodes 8 --start-dist points --start-zero-frac 0.5 --start-zero-cap 50000 --window-adapt --fitness-agg geomean"
  [ramp]="--episodes 8 --window 1000000 --start-zero-cap 1000000 --start-zero-frac 0.5 --ramp-frac 0.5 --ramp-min 400 --ramp-max 800"
  [hyg]=""
)
for d in runs/exp-bp2-start runs/exp-bp-start; do
  [ -f $d/SHA256SUMS ] && { (cd $d && sha256sum -c --quiet SHA256SUMS) || { echo "checksum mismatch in $d"; exit 1; }; }
done
for s in $SEEDS; do
  src=runs/exp-bp2-start/ckpt-s$s.json; [ -f $src ] || src=runs/exp-bp-start/ckpt-s$s.json
  g0=$(node -e "console.log(require('./$src').generation)")
  for arm in $ARMS; do
    dir=runs/exp-bp2-s$s-$arm
    if [ -e $dir/checkpoint.json ]; then echo "$(date -Is) skip $dir (exists)"; continue; fi
    mkdir -p $dir && cp $src $dir/checkpoint.json
    echo "$(date -Is) start $dir (from $src, gen $g0)"
    node trainer/cli.js train --resume --dir $dir --generations $(( g0 + 20000 )) $COMMON ${EXTRA[$arm]} >> $dir/train.log 2>&1 &
    tp=$!
    node tools/event-watch.js $dir --after $g0 --max-frames 1e10 >> $dir/watch.log 2>&1 &
    wp=$!
    wait $tp; echo "$(date -Is) end $dir exit=$?"
    kill $wp 2>/dev/null; wait $wp 2>/dev/null
  done
done
echo "$(date -Is) all done"
