"""Exp 01, Phase 1E / E1: was the lottery drawn by Adam's first steps? (no new runs)

Applies the winner predictors to the Phase 1B checkpoints at early steps t, instead of only at theta_0, and scores
each against the final k* over the final active neurons.

Usage: python experiments/01_frequency_lottery/phase1e_e1_predictor_over_time.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from grokking.fourier import active_neurons, neuron_fourier, split_weights, winner_predictions  # noqa: E402

P = 47
K = (P - 1) // 2
STEPS = [0, 10, 20, 30, 50, 100, 200]
RUNS = Path("runs/exp01/phase1b")
OUT = Path("experiments/01_frequency_lottery/figures")
COLORS = {0.1: "#2a78d6", 0.3: "#eb6834", 1.0: "#1baf7a"}


def main():
    acc = defaultdict(lambda: defaultdict(list))  # (regime, scale) -> predictor -> [per-step hit arrays]
    for d in sorted(RUNS.iterdir()):
        m = re.match(r"(grok|full)_is([\d.]+)_s(\d+)$", d.name)
        if not m:
            continue
        regime, scale = m.group(1), float(m.group(2))
        Wa, Wb, Wo = split_weights(torch.load(d / "final.pt"), P)
        act = active_neurons(Wa, Wb, Wo)
        kfin = neuron_fourier(Wa, Wb, Wo)["kstar"][act]
        for name in ("pred_power", "pred_growth"):
            hits = []
            for t in STEPS:
                pred = winner_predictions(neuron_fourier(*split_weights(torch.load(d / f"ckpt_{t}.pt"), P)))[name]
                hits.append((pred[act] == kfin).numpy())
            acc[(regime, scale)][name].append(np.stack(hits))  # (T, n_active)

    res = {"steps": STEPS, "chance": 1 / K, "conditions": {}}
    for (regime, scale), d in sorted(acc.items()):
        res["conditions"][f"{regime}_is{scale}"] = {
            name: [float(x) for x in np.concatenate(v, axis=1).mean(1)] for name, v in d.items()}
    (OUT / "phase1e_e1_results.json").write_text(json.dumps(res, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, regime in zip(axes, ("full", "grok")):
        for scale, col in COLORS.items():
            c = res["conditions"].get(f"{regime}_is{scale}")
            if not c:
                continue
            ax.plot(STEPS, c["pred_growth"], color=col, lw=2, marker="o", ms=5, label=f"magnitude × alignment, init {scale}")
            ax.plot(STEPS, c["pred_power"], color=col, lw=1.5, ls="--", marker="s", ms=4, label=f"power only, init {scale}")
        ax.axhline(1 / K, color="#444", ls=":", lw=1, label="chance 1/K")
        ax.set_xscale("symlog", linthresh=10)
        ax.set(xlabel="step t at which the predictor is applied",
               title=f"{'full data' if regime == 'full' else 'grokking split (50%)'}", ylim=(0, 1))
        ax.set_xticks([0, 10, 30, 100, 200], ["0", "10", "30", "100", "200"])
        ax.minorticks_off()
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("predicts final k* (active neurons)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("E1: predicting each neuron's final frequency from checkpoint t instead of θ₀ (H = 512, 3 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "phase1e_e1_predictor_over_time.png", dpi=130)

    print(f"chance {1 / K:.3f}")
    for c, v in res["conditions"].items():
        print(f"{c:12s} growth " + " ".join(f"{x:.2f}" for x in v["pred_growth"]) +
              " | power " + " ".join(f"{x:.2f}" for x in v["pred_power"]))


if __name__ == "__main__":
    main()
