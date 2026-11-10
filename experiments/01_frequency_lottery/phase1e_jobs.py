"""Exp 01, Phase 1E: build the initial weights and the job lists for E2, E5, E6 and the init-vs-split test.

Each job line is "<out dir> <train.py args>"; run_phase1e.sh runs them. Every run is p = 47, the one-hot MLP and
--threads 1. Runs that already have a history.json are skipped by the runner, and runs shared with earlier phases
(e.g. lr 1e-3, wd 1 at a 1C width) are reused rather than rerun.

  e5     learning rate x width. Arm a keeps wd = 1; arm b holds lr*wd = 1e-3 (the decay per step of the default).
  cf     same initial weights under different train/test splits (and vice versa), H in {128, 256, 512}, plus
         H = 512 at init scales 0.1 and 0.3.
  e2cal  plain gradient descent on all pairs, H = 512, seed 0: which lr fits the data within 20k steps?
  e2     GD on all pairs and on the grokking split (needs --gd-lr), and the Adam warmup ablation.
  e6     rigged-init dose-response at H = 128 (inits are written by phase1e_e6_rigged.py make).

Usage:
  python experiments/01_frequency_lottery/phase1e_jobs.py e5 > runs/exp01/phase1e/jobs_e5.txt
  bash   experiments/01_frequency_lottery/run_phase1e.sh runs/exp01/phase1e/jobs_e5.txt 10
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from grokking.model import make_model  # noqa: E402

P = 47
ROOT = Path("runs/exp01/phase1e")
GROK = "--train-frac 0.5 --max-steps 50000 --stop-on-grok --post-grok-steps 2000"
# checkpoints as in 1B: every 10 steps to 500, every 250 to 5k, then every 1k
CKPT = "--save-init --ckpt-every 250 --ckpt-dense-until 500 --ckpt-dense-every 10 --ckpt-sparse-after 5000 " \
       "--ckpt-sparse-every 1000"
E5_PRODUCTS = (0.128, 0.256, 0.512, 1.024, 2.048)  # lr x H
SEEDS5 = range(5)


def fmt(x):
    return f"{x:.6g}"


def e5():
    jobs = []
    for H in (512, 256, 128):
        for c in E5_PRODUCTS:
            lr = c / H
            for arm, wd in (("a", 1.0), ("b", 1e-3 / lr)):
                if abs(lr - 1e-3) < 1e-12:
                    continue  # lr 1e-3, wd 1 is the 1C run of the same seed
                for s in SEEDS5:
                    jobs.append((f"e5/H{H}_lr{fmt(lr)}_wd{fmt(wd)}_s{s}",
                                 f"--d-hidden {H} --lr {fmt(lr)} --wd {fmt(wd)} --seed {s} {GROK}"))
    # the threshold side: does a higher lr rescue H = 64? And one wider point, H = 1024.
    for lr in (4e-3, 8e-3, 1.6e-2):
        for s in SEEDS5:
            jobs.append((f"e5/H64_lr{fmt(lr)}_wd1_s{s}", f"--d-hidden 64 --lr {fmt(lr)} --wd 1 --seed {s} {GROK}"))
    for lr in (5e-4, 1e-3):
        for s in SEEDS5:
            jobs.append((f"e5/H1024_lr{fmt(lr)}_wd1_s{s}", f"--d-hidden 1024 --lr {fmt(lr)} --wd 1 --seed {s} {GROK}"))
    return jobs


def cf():
    """Initial weights from seed i, data split from seed j. Diagonal runs (i == j) equal ordinary runs."""
    jobs = []
    inits = ROOT / "cf" / "inits"
    inits.mkdir(parents=True, exist_ok=True)
    configs = [(512, 1.0), (512, 0.3), (512, 0.1), (256, 1.0), (128, 1.0)]
    for H, scale in configs:
        for i in range(3):
            f = inits / f"H{H}_is{scale}_s{i}.pt"
            if not f.exists():
                torch.save(make_model("mlp", P, i, d_hidden=H, init_scale=scale).state_dict(), f)
            for j in range(3):
                jobs.append((f"cf/H{H}_is{scale}_init{i}_split{j}",
                             f"--d-hidden {H} --seed {j} --init-from {f} {GROK}"))
    return jobs


def e2cal():
    jobs = []
    for scale in (1.0, 0.1):
        for lr in (0.03, 0.1, 0.3, 1.0):
            jobs.append((f"e2/cal_full_is{scale}_lr{lr}",
                         f"--optim sgd --lr {lr} --wd {fmt(1e-3 / lr)} --warmup 1 --init-scale {scale} --seed 0 "
                         f"--train-frac 1.0 --max-steps 20000"))
    return jobs


def e2(gd_lr):
    jobs = []
    gd = f"--optim sgd --lr {gd_lr} --wd {fmt(1e-3 / gd_lr)} --warmup 1"
    for scale in (1.0, 0.1):
        for s in range(3):
            jobs.append((f"e2/gd_full_is{scale}_s{s}",
                         f"{gd} --init-scale {scale} --seed {s} --train-frac 1.0 --max-steps 20000 {CKPT}"))
            jobs.append((f"e2/gd_grok_is{scale}_s{s}", f"{gd} --init-scale {scale} --seed {s} {GROK} {CKPT}"))
    # Adam warmup ablation; warmup 10 is the 1B run. Warmup 1 means none.
    for w in (1, 100, 1000):
        for s in range(3):
            jobs.append((f"e2/adam_grok_warmup{w}_s{s}", f"--warmup {w} --seed {s} {GROK} {CKPT}"))
            jobs.append((f"e2/adam_full_is0.1_warmup{w}_s{s}",
                         f"--warmup {w} --init-scale 0.1 --seed {s} --train-frac 1.0 --max-steps 20000 {CKPT}"))
    return jobs


def e6():
    jobs = []
    for f in sorted((ROOT / "e6" / "inits").glob("*.pt")):
        seed = int(f.name.split("_")[0][1:])
        jobs.append((f"e6/{f.stem}", f"--d-hidden 128 --seed {seed} --init-from {f} {GROK}"))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["e5", "cf", "e2cal", "e2", "e6"])
    ap.add_argument("--gd-lr", type=float, default=None)
    a = ap.parse_args()
    if a.arm == "e2":
        if a.gd_lr is None:
            ap.error("e2 needs --gd-lr (from e2cal)")
        jobs = e2(a.gd_lr)
    else:
        jobs = globals()[a.arm]()
    for name, args in jobs:
        print(f"{ROOT / name} {args}")


if __name__ == "__main__":
    main()
