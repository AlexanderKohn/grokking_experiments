"""Exp 01, Phase 1E: what decides a neuron's frequency on the grokking split, the initial weights or the split?

For each config (H, init scale), runs with initial weights from seed i and the data split from seed j, i, j in 0..2.
A neuron "agrees" between two runs if it is active in both and ends on the same frequency. Chance is the agreement
expected if the two runs drew frequencies independently from their own final distributions. Neuron m has no identity
across different inits, so "same split, different init" is at chance by construction; the split's effect is measured
instead by how well the per-frequency contribution shares of two runs correlate.

Also reported: the init-time score's accuracy on the same runs, the share of neurons that end on the same frequency
under all three splits, and the bound an init-only predictor faces: a predictor that is right with probability q
for every neuron and split agrees with itself, so the same-init agreement a >= q^2 when errors don't coincide,
i.e. q <= sqrt(a) roughly.

Usage: python experiments/01_frequency_lottery/phase1e_cf_init_vs_split.py
"""
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from grokking.fourier import (active_neurons, neuron_contribution, neuron_fourier, split_weights,  # noqa: E402
                              winner_predictions)

P = 47
K = (P - 1) // 2
ROOT = Path("runs/exp01/phase1e/cf")
FIG = Path("experiments/01_frequency_lottery/figures")


def final(d):
    w = split_weights(torch.load(d / "final.pt"), P)
    return neuron_fourier(*w)["kstar"], active_neurons(*w)


def share(d):
    """Share of total neuron contribution on each frequency, (K,)."""
    w = split_weights(torch.load(d / "final.pt"), P)
    q = torch.bincount(neuron_fourier(*w)["kstar"], weights=neuron_contribution(*w).double(), minlength=K + 1)[1:]
    return (q / q.sum()).numpy()


def share_corr(shares, pairs):
    """Mean Pearson correlation between the per-frequency shares of two runs. Neuron identity plays no part, so
    this is the fair way to ask whether the split (or the init) shapes *which frequencies* the network uses."""
    return float(np.mean([np.corrcoef(shares[a], shares[b])[0, 1] for a, b in pairs]))


def agree(a, b):
    (ka, aa), (kb, ab) = a, b
    m = aa & ab
    qa = np.bincount(ka[m].numpy(), minlength=K + 1)[1:] / max(int(m.sum()), 1)
    qb = np.bincount(kb[m].numpy(), minlength=K + 1)[1:] / max(int(m.sum()), 1)
    return int((ka[m] == kb[m]).sum()), int(m.sum()), float(qa @ qb) * int(m.sum())


def pool(runs, pairs):
    n = d = ch = 0
    per = []
    for a, b in pairs:
        x, y, c = agree(runs[a], runs[b])
        n, d, ch = n + x, d + y, ch + c
        per.append(x / y)
    return {"agree": n / d, "chance": ch / d, "pairs": d, "per_pair": [round(v, 3) for v in per]}


def main():
    configs = defaultdict(dict)
    shares = defaultdict(dict)
    hist = {}
    for d in sorted(ROOT.iterdir()):
        m = re.match(r"H(\d+)_is([\d.]+)_init(\d)_split(\d)$", d.name)
        if m and (d / "final.pt").exists():
            key = (int(m.group(1)), float(m.group(2)))
            configs[key][(int(m.group(3)), int(m.group(4)))] = final(d)
            shares[key][(int(m.group(3)), int(m.group(4)))] = share(d)
            hist[(key, int(m.group(3)), int(m.group(4)))] = json.loads((d / "history.json").read_text())["summary"]

    res = {}
    for (H, scale), runs in sorted(configs.items()):
        if len(runs) < 9:
            print(f"H{H} is{scale}: only {len(runs)} runs, skipped")
            continue
        same_init = [((i, j), (i, k)) for i in range(3) for j, k in itertools.combinations(range(3), 2)]
        same_split = [((i, j), (k, j)) for j in range(3) for i, k in itertools.combinations(range(3), 2)]
        neither = [((i, j), (k, l)) for i, k in itertools.combinations(range(3), 2) for j in range(3)
                   for l in range(3) if j != l]
        r = {"same init, different split": pool(runs, same_init),
             "same split, different init": pool(runs, same_split),
             "different init and split": pool(runs, neither)}
        r["per-frequency share correlation"] = {
            "same init, different split": share_corr(shares[(H, scale)], same_init),
            "same split, different init": share_corr(shares[(H, scale)], same_split),
            "different init and split": share_corr(shares[(H, scale)], neither)}
        tot = {"pred_growth": [0, 0], "pred_power": [0, 0]}
        cons = [0, 0, 0]
        for i in range(3):
            sd0 = torch.load(ROOT / "inits" / f"H{H}_is{scale}_s{i}.pt")
            pr = winner_predictions(neuron_fourier(*split_weights(sd0, P)))
            for j in range(3):
                k, a = runs[(i, j)]
                for name in tot:
                    tot[name][0] += int((pr[name][a] == k[a]).sum())
                    tot[name][1] += int(a.sum())
            ks = torch.stack([runs[(i, j)][0] for j in range(3)])
            act = torch.stack([runs[(i, j)][1] for j in range(3)]).all(0)
            same = (ks[0] == ks[1]) & (ks[1] == ks[2]) & act
            cons[0] += int(same.sum())
            cons[1] += int(act.sum())
            cons[2] += int((pr["pred_growth"][same] == ks[0][same]).sum())
        r["init score accuracy"] = {n: v[0] / v[1] for n, v in tot.items()}
        r["same frequency under all 3 splits"] = {"frac": cons[0] / cons[1], "n": cons[1],
                                                 "init score names it": cons[2] / max(cons[0], 1)}
        r["bound sqrt(same-init agreement)"] = float(np.sqrt(r["same init, different split"]["agree"]))
        t95 = [hist[((H, scale), i, j)]["test_95_step"] for i in range(3) for j in range(3)]
        r["t95 (init i, split j)"] = t95
        # does the init or the split explain t95? (variance of log t95 across rows vs columns)
        if all(t95):
            T = np.log10(np.array(t95, dtype=float)).reshape(3, 3)
            g = T.mean()
            r["log10 t95 variance"] = {"init (rows)": float(((T.mean(1) - g) ** 2).mean()),
                                       "split (cols)": float(((T.mean(0) - g) ** 2).mean()),
                                       "total": float(((T - g) ** 2).mean())}
        res[f"H{H}_is{scale}"] = r
        print(f"\nH = {H}, init scale {scale}")
        for k2, v in r.items():
            print(f"  {k2}: {v}")
    (FIG / "phase1e_cf_results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
