#!/bin/zsh
# Roll every clone: 200 fresh seeds per level (3k points from 0, 20k-point windows from
# 2k/50k/300k/900k), then 25 fresh seeds 0 -> 1,000,000. At most 4 processes.
cd /private/tmp/claude-501/-Users-kyle-Desktop-dino-game/a1a47f55-01e9-48b9-afbe-71ea588921bc/scratchpad/features
one() {
  ROLL_SPAN0=3000 ROLL_SPAN=20000 node cloneB.js roll B/m_$1.json 100000 100200 B/rw_$1.json > /dev/null
  FULL=1 ROLL_LEVELS=0 node cloneB.js roll B/m_$1.json 200000 200025 B/rf_$1.json > /dev/null
}
ms=(base_s1 base_s2 rescale_s1 rescale_s2 tau_s1 tau_s2 tau_gap_s1 tau_gap_s2 key_s1 key_s2 tau_gap_key_s1 tau_gap_key_s2 all_s1 all_s2)
i=0
for m in $ms; do
  one $m &
  i=$((i+1)); if (( i % 4 == 0 )); then wait; fi
done
wait
echo done
