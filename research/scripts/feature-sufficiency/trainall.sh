#!/bin/zsh
# Train every variant x 2 training seeds on the full solver data, 32x32, 15 epochs.
cd /private/tmp/claude-501/-Users-kyle-Desktop-dino-game/a1a47f55-01e9-48b9-afbe-71ea588921bc/scratchpad/features
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
PY=/Users/kyle/Desktop/dino-game/.venv/bin/python
for v in base rescale tau tau_gap key tau_gap_key all; do
  SEED=1 $PY -W ignore trainB.py $v 32,32 B/m_${v}_s1.json 15 B/d1.bin B/d2.bin B/d3.bin B/d4.bin > B/log_${v}_s1.txt &
  SEED=2 $PY -W ignore trainB.py $v 32,32 B/m_${v}_s2.json 15 B/d1.bin B/d2.bin B/d3.bin B/d4.bin > B/log_${v}_s2.txt &
  wait
done
