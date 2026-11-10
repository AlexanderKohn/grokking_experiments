"""Exp 01, Phase 1E / E2: the lottery under plain gradient descent, and the Adam warmup ablation (H = 512).

Conditions (3 seeds each): plain full-batch GD (lr 1.0, L2 1e-3, so the decay per step matches AdamW's lr*wd) on all
pairs and on the grokking split at init scales 1.0 and 0.1; Adam on the grokking split with warmup 1, 100, 1000 (and
10, the 1B run); Adam on all pairs at init scale 0.1 with warmup 1, 100, 1000 (and 10).

For each: the winner predictors applied at checkpoint t (as in E1) and scored against final k* on the final active
neurons, the lock-in step distribution, the share of active neurons whose k* changes between fitting the training
set and the end, and t_fit / t95.

Usage: python experiments/01_frequency_lottery/phase1e_e2_gd_warmup.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from grokking.fourier import active_neurons, neuron_fourier, split_weights, winner_predictions  # noqa: E402
from phase1b_analysis import analyze_run  # noqa: E402

P = 47
E2 = Path("runs/exp01/phase1e/e2")
B1 = Path("runs/exp01/phase1b")
FIG = Path("experiments/01_frequency_lottery/figures")
STEPS = [0, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000]


def conditions():
    c = defaultdict(list)
    for d in sorted(E2.iterdir()):
        m = re.match(r"(gd_full|gd_grok|adam_grok|adam_full)_(is[\d.]+_)?(warmup\d+_)?s(\d)$", d.name)
        if m and (d / "history.json").exists():
            c[d.name.rsplit("_s", 1)[0]].append(d)
    for s in range(3):  # the 1B runs are the warmup-10 arms
        c["adam_grok_warmup10"].append(B1 / f"grok_is1.0_s{s}")
        c["adam_full_is0.1_warmup10"].append(B1 / f"full_is0.1_s{s}")
        c["adam_full_is1.0_warmup10"].append(B1 / f"full_is1.0_s{s}")
        c["adam_grok_is0.1_warmup10"].append(B1 / f"grok_is0.1_s{s}")
    return c


def predictor_over_time(d):
    Wa, Wb, Wo = split_weights(torch.load(d / "final.pt"), P)
    act = active_neurons(Wa, Wb, Wo)
    kfin = neuron_fourier(Wa, Wb, Wo)["kstar"][act]
    out = {}
    for t in STEPS:
        f = d / f"ckpt_{t}.pt"
        if f.exists():
            pred = winner_predictions(neuron_fourier(*split_weights(torch.load(f), P)))["pred_growth"]
            out[t] = (pred[act] == kfin).numpy()
    return out


def final_k(d):
    w = split_weights(torch.load(d / "final.pt"), P)
    return neuron_fourier(*w)["kstar"], active_neurons(*w)


def same_init_agreement(c):
    """Same initial weights and data, one thing changed (warmup or optimizer): the share of neurons active in both
    runs that end on the same frequency, pooled over the 3 seeds. Run lists in c are in seed order."""
    pairs = {
        "all pairs, init 0.1: warmup 1 vs 1000": ("adam_full_is0.1_warmup1", "adam_full_is0.1_warmup1000"),
        "all pairs, init 0.1: warmup 10 vs 100": ("adam_full_is0.1_warmup10", "adam_full_is0.1_warmup100"),
        "all pairs, init 0.1: Adam vs GD": ("adam_full_is0.1_warmup10", "gd_full_is0.1"),
        "all pairs, init 1.0: Adam vs GD": ("adam_full_is1.0_warmup10", "gd_full_is1.0"),
        "grokking split, init 1.0: warmup 1 vs 1000": ("adam_grok_warmup1", "adam_grok_warmup1000"),
        "grokking split, init 1.0: warmup 10 vs 100": ("adam_grok_warmup10", "adam_grok_warmup100"),
        "grokking split, init 1.0: Adam vs GD": ("adam_grok_warmup10", "gd_grok_is1.0"),
        "grokking split, init 0.1: Adam vs GD": ("adam_grok_is0.1_warmup10", "gd_grok_is0.1"),
    }
    out = {}
    for label, (a, b) in pairs.items():
        n = d = 0
        for da, db in zip(sorted(c[a], key=lambda p: p.name[-1]), sorted(c[b], key=lambda p: p.name[-1])):
            (ka, aa), (kb, ab) = final_k(da), final_k(db)
            m = aa & ab
            n, d = n + int((ka[m] == kb[m]).sum()), d + int(m.sum())
        out[label] = {"agree": n / d, "neurons": d}
        print(f"same init and data, {label}: {n / d:.3f} of {d}")
    return out


def main():
    res = {}
    for name, dirs in sorted(conditions().items()):
        hits = defaultdict(list)
        locks, switch, t_fit, t95, final_acc = [], [], [], [], []
        for d in dirs:
            for t, h in predictor_over_time(d).items():
                hits[t].append(h)
            r = analyze_run(d, P, -1)
            locks.append(np.array(r["lock_step"]))
            switch.append(r["switch_frac"])
            t_fit.append(r["summary"]["train_99_step"])
            t95.append(r["summary"]["test_95_step"])
            final_acc.append((r["summary"]["final_train_acc"], r["summary"]["final_test_acc"]))
        lock = np.concatenate(locks)
        res[name] = {
            "n_runs": len(dirs),
            "pred_growth_at_step": {t: float(np.concatenate(v).mean()) for t, v in sorted(hits.items())},
            "lock_median": float(np.median(lock)),
            "lock_q25_q75": [float(np.percentile(lock, 25)), float(np.percentile(lock, 75))],
            "frac_locked_by": {t: float((lock <= t).mean()) for t in (10, 100, 1000)},
            "switch_frac": switch, "t_fit": t_fit, "t95": t95, "final_train_test_acc": final_acc,
        }
        v = res[name]
        print(f"{name:26s} n={len(dirs)} pred@t " +
              " ".join(f"{t}:{a:.2f}" for t, a in v["pred_growth_at_step"].items()) +
              f" | lock med {v['lock_median']:.0f} | switch {[round(x, 2) if x is not None else None for x in switch]}"
              f" | t_fit {t_fit} t95 {t95}")
    res["same_init_agreement"] = same_init_agreement(conditions())
    (FIG / "phase1e_e2_results.json").write_text(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panels = [("all pairs", [("gd_full_is1.0", "#2a78d6", "-", "GD, init 1.0"),
                             ("gd_full_is0.1", "#2a78d6", "--", "GD, init 0.1"),
                             ("adam_full_is1.0_warmup10", "#eb6834", "-", "Adam, init 1.0"),
                             ("adam_full_is0.1_warmup10", "#eb6834", "--", "Adam, init 0.1"),
                             ("adam_full_is0.1_warmup1000", "#1baf7a", "--", "Adam, init 0.1, warmup 1000")]),
              ("grokking split", [("gd_grok_is1.0", "#2a78d6", "-", "GD, init 1.0"),
                                  ("gd_grok_is0.1", "#2a78d6", "--", "GD, init 0.1"),
                                  ("adam_grok_warmup10", "#eb6834", "-", "Adam, warmup 10"),
                                  ("adam_grok_warmup1", "#8a8984", "-", "Adam, no warmup"),
                                  ("adam_grok_warmup1000", "#1baf7a", "-", "Adam, warmup 1000")])]
    for ax, (title, series) in zip(axes, panels):
        for key, col, ls, lab in series:
            if key not in res:
                continue
            pts = res[key]["pred_growth_at_step"]
            ax.plot(list(pts), list(pts.values()), color=col, ls=ls, marker="o", ms=3.5, lw=1.8, label=lab)
        ax.axhline(1 / 23, color="#999", ls=":", lw=1)
        ax.set(xscale="symlog", title=title, xlabel="step whose weights the prediction is made from", ylim=(0, 1))
        ax.set_xticks([0, 10, 100, 1000], ["0", "10", "100", "1k"])
        ax.minorticks_off()
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5)
    axes[0].set_ylabel("neurons predicted correctly")
    fig.tight_layout()
    fig.savefig(FIG / "phase1e_e2_predictor_over_time.png", dpi=130)


if __name__ == "__main__":
    main()
