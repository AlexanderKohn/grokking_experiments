#!/usr/bin/env bash
# Exp 01, Phase 1E: run a job list written by phase1e_jobs.py. Each line is "<out dir> <train.py args>".
# Usage: bash experiments/01_frequency_lottery/run_phase1e.sh <jobs file> [parallel_jobs]
set -euo pipefail
cd "$(dirname "$0")/../.."
JOBS_FILE=${1:?jobs file}
JOBS=${2:-6}

xargs -P "$JOBS" -L 1 sh -c '
  d=$0
  if [ -f $d/history.json ]; then echo "skip $d"; exit 0; fi
  mkdir -p $d
  python train.py --model mlp --p 47 --threads 1 --out $d "$@" > $d/train.log 2>&1 \
    && echo "finished $d: $(grep done $d/train.log)" || echo "FAILED $d"' < "$JOBS_FILE"
echo ALL_DONE
