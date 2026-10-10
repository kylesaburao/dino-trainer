#!/bin/zsh
# Harder regimes: (h8) 44-8-8-3 on all data; (n50k) 44-32-32-3 on 50k random rows.
cd /private/tmp/claude-501/-Users-kyle-Desktop-dino-game/a1a47f55-01e9-48b9-afbe-71ea588921bc/scratchpad/features
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
PY=/Users/kyle/Desktop/dino-game/.venv/bin/python
D="B/d1.bin B/d2.bin B/d3.bin B/d4.bin"
for v in base rescale tau_gap all; do
  for s in 1 2; do
    SEED=$s $PY -W ignore trainB.py $v 8,8 B/m_h8_${v}_s$s.json 15 ${=D} > B/log_h8_${v}_s$s.txt &
    SEED=$s NROWS=50000 $PY -W ignore trainB.py $v 32,32 B/m_n50k_${v}_s$s.json 40 ${=D} > B/log_n50k_${v}_s$s.txt &
    wait
  done
done
for v in base rescale tau_gap all; do
  for s in 1 2; do
    ROLL_SPAN0=3000 ROLL_SPAN=20000 node cloneB.js roll B/m_h8_${v}_s$s.json 100000 100200 B/rw_h8_${v}_s$s.json > /dev/null &
    ROLL_SPAN0=3000 ROLL_SPAN=20000 node cloneB.js roll B/m_n50k_${v}_s$s.json 100000 100200 B/rw_n50k_${v}_s$s.json > /dev/null &
  done
  wait
done
echo done
