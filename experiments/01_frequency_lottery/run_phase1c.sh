#!/usr/bin/env bash
# Exp 01, Phase 1C: width sweep. H in {4..512} x 8 seeds, p=47, frac 0.5, init scale 1.0.
# Runs stop 2k steps after grokking; the cap is 50k steps (reaching it = "did not grok").
# Usage: bash experiments/01_frequency_lottery/run_phase1c.sh [parallel_jobs] [init_scale]
set -euo pipefail
cd "$(dirname "$0")/../.."
JOBS=${1:-6}
SCALE=${2:-1.0}
OUT=runs/exp01/phase1c
mkdir -p $OUT

# widest first, so the longest-running jobs don't start last
for H in 512 256 128 64 32 16 8 4; do
  for seed in 0 1 2 3 4 5 6 7; do
    echo "$H $seed"
  done
done | xargs -P "$JOBS" -n 2 sh -c '
  d='"$OUT"'/H$0_is'"$SCALE"'_s$1
  if [ -f $d/history.json ]; then echo "skip $d"; exit 0; fi
  mkdir -p $d
  python train.py --model mlp --p 47 --train-frac 0.5 --d-hidden $0 --init-scale '"$SCALE"' --seed $1 \
    --max-steps 50000 --stop-on-grok --post-grok-steps 2000 --threads 1 --out $d > $d/train.log 2>&1 \
    && echo "finished $d: $(grep done $d/train.log)" || echo "FAILED $d"'
