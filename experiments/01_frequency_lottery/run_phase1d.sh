#!/usr/bin/env bash
# Exp 01, Phase 1D: retrain from the edited inits built by `phase1d_interventions.py make`.
# H=128, p=47, frac 0.5; same --seed as the init, so the data split matches the 1C run. Cap 50k steps.
# Usage: bash experiments/01_frequency_lottery/run_phase1d.sh [parallel_jobs]
set -euo pipefail
cd "$(dirname "$0")/../.."
JOBS=${1:-6}
ROOT=runs/exp01/phase1d

ls $ROOT/inits/*.pt | xargs -P "$JOBS" -n 1 sh -c '
  f=$0; name=$(basename $f .pt); seed=$(echo $name | sed "s/^s\([0-9]*\)_.*/\1/")
  d='"$ROOT"'/$name
  if [ -f $d/history.json ]; then echo "skip $d"; exit 0; fi
  mkdir -p $d
  python train.py --model mlp --p 47 --train-frac 0.5 --d-hidden 128 --seed $seed --init-from $f \
    --max-steps 50000 --stop-on-grok --threads 1 --out $d > $d/train.log 2>&1 \
    && echo "finished $d: $(grep done $d/train.log)" || echo "FAILED $d"'
