#!/usr/bin/env bash
# Exp 01, Phase 1B: does the lottery hold? H=512, p=47, 3 seeds x 3 init scales x 2 data regimes.
# Usage: bash experiments/01_frequency_lottery/run_phase1b.sh <grok_train_frac> [parallel_jobs]
set -euo pipefail
cd "$(dirname "$0")/../.."
FRAC=${1:?grokking train-frac from the pilot}
JOBS=${2:-6}
OUT=runs/exp01/phase1b
# checkpoints: every 10 steps to 500, every 250 to 5k, then every 1k
mkdir -p $OUT

jobs=()
for scale in 0.1 0.3 1.0; do
  for seed in 0 1 2; do
    # grokking split: stop 30k steps after grokking, cap at 50k
    jobs+=("--train-frac $FRAC --stop-on-grok --post-grok-steps 30000 --max-steps 50000 --init-scale $scale --seed $seed --out $OUT/grok_is${scale}_s${seed}")
    # full data (no test set, no grokking): fixed 20k steps
    jobs+=("--train-frac 1.0 --max-steps 20000 --init-scale $scale --seed $seed --out $OUT/full_is${scale}_s${seed}")
  done
done

printf '%s\n' "${jobs[@]}" | xargs -P "$JOBS" -I{} sh -c \
  'a="{}"; d=$(echo "$a" | sed "s/.*--out //"); mkdir -p $d; python train.py --model mlp --p 47 --threads 1 --save-init --ckpt-dense-until 500 --ckpt-dense-every 10 --ckpt-every 250 --ckpt-sparse-after 5000 --ckpt-sparse-every 1000 $a > $d/train.log 2>&1 && echo "finished $d"'
