#!/bin/sh
# Tier-B scans (research/prereg-bc-sweep.md §5), one at a time: 1,000 full v2 games to 1M on
# seeds 30,003,000-30,003,999 with research/scripts/prereg-bc-robustness/scan.js --wasm,
# 18 workers. scan.js resumes a partial file, so a finished scan is skipped.
# Usage: scan-queue.sh d1-w8-s1 d2-w8-s1 ...
set -eu
ROOT=/Users/kyle/Desktop/dino-game
OUT=$ROOT/runs/exp-bc-sweep/scans
mkdir -p "$OUT"
cd "$ROOT"
for RUN in "$@"; do
  if [ -f "$OUT/$RUN.jsonl" ] && grep -q '"footer"' "$OUT/$RUN.jsonl"; then echo "skip $RUN"; continue; fi
  echo "scan $RUN start $(date -u +%FT%TZ)"
  node "$ROOT/research/scripts/prereg-bc-robustness/scan.js" --model "$ROOT/runs/exp-bc-sweep/$RUN/best.json" \
    --out "$OUT/$RUN.jsonl" --base 30003000 --n 1000 --target 1000000 --workers 18 --wasm > "$OUT/$RUN.log" 2>&1
  echo "scan $RUN end $(date -u +%FT%TZ): $(tail -1 "$OUT/$RUN.log")"
done
echo "scan queue done $(date -u +%FT%TZ)"
