#!/bin/sh
# One sweep training job (research/prereg-bc-sweep.md §5): bc-train --backend python with
# --snapshots and --wasm, the shipped hyperparameters (15 epochs, batch 1024, lr 2e-3),
# demos runs/exp-bc-e2e, into runs/exp-bc-sweep/d<D>-w<W>-s<SEED>. Runs in the foreground;
# launch it with nohup. /usr/bin/time -l reports the job's CPU (training + tier-A validation).
# Usage: train-one.sh <depth> <width> <seed> <validation workers>
set -eu
D=$1; W=$2; SEED=$3; WORKERS=$4
ROOT=/Users/kyle/Desktop/dino-game
DIR=$ROOT/runs/exp-bc-sweep/d$D-w$W-s$SEED
HIDDEN=$(node -e "console.log(Array($D).fill($W).join(','))")
mkdir -p "$DIR"
export PYTHON=$ROOT/.venv/bin/python OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
cd "$ROOT"
echo "start $(date -u +%FT%TZ) d=$D w=$W seed=$SEED workers=$WORKERS"
/usr/bin/time -l node "$ROOT/trainer/cli.js" bc-train --backend python --snapshots --wasm \
  --dir "$DIR" --demos "$ROOT/runs/exp-bc-e2e" --hidden "$HIDDEN" \
  --epochs 15 --batch 1024 --lr 2e-3 --seed "$SEED" --workers "$WORKERS" --target 1000000
echo "end $(date -u +%FT%TZ)"
