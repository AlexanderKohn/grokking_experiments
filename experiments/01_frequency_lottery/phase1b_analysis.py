"""Exp 01, Phase 1B analysis: does the init-time lottery predict each neuron's final frequency?

Reads runs/exp01/phase1b/{grok,full}_is{scale}_s{seed}/ (checkpoints from run_phase1b.sh) and writes
phase1b_results.json plus plots to --out.

Usage: python experiments/01_frequency_lottery/phase1b_analysis.py
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from grokking.fourier import (active_neurons, circular_resultant, key_frequencies,  # noqa: E402
                              neuron_contribution, neuron_fourier, split_weights,
                              winner_predictions)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SCALE_COLOR = {0.1: BLUE, 0.3: ORANGE, 1.0: AQUA}
PREDICTORS = ["pred_power", "pred_growth", "pred_growth_pos"]


def gather_k(t, k):
    """t: (H, K) values, k: (H,) frequencies 1..K -> (H,)"""
    return t.gather(-1, (k - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1)


def eff_num_freqs(kstar, contrib, K):
    """1 / sum_k q_k^2, where q_k is the share of total neuron contribution on frequency k."""
    q = torch.bincount(kstar, weights=contrib.double(), minlength=K + 1)[1:]
    q = q / q.sum()
    return (1 / q.pow(2).sum()).item()


def analyze_run(d, p, sign):
    h = json.loads((d / "history.json").read_text())
    summ, hist = h["summary"], h["history"]
    ck = sorted((int(re.search(r"ckpt_(\d+)", f.name).group(1)), f) for f in d.glob("ckpt_*.pt"))
    files = ck + [(summ["final_step"], d / "final.pt")]
    steps = np.array([s for s, _ in files])

    final_sd = torch.load(d / "final.pt")
    Wa, Wb, Wo = split_weights(final_sd, p)
    act = active_neurons(Wa, Wb, Wo)
    Ffin = neuron_fourier(Wa, Wb, Wo, sign=sign)
    kfin = Ffin["kstar"]
    keys = key_frequencies(kfin, act)

    init = neuron_fourier(*split_weights(torch.load(d / "ckpt_0.pt"), p), sign=sign)
    preds = winner_predictions(init)
    growth = init["alpha_a"] * init["alpha_b"] * init["beta"] * init["cosD"]
    top3 = growth.topk(3, dim=-1).indices + 1

    K = (p - 1) // 2
    K_t, ipr_t, R_minus, R_plus, sinD, neff_t = [], [], [], [], [], []
    for _, f in files:
        W = split_weights(torch.load(f), p)
        F = neuron_fourier(*W, sign=sign)
        k = F["kstar"]
        neff_t.append(eff_num_freqs(k, neuron_contribution(*W), K))
        K_t.append(k)
        ipr_t.append(F["ipr"][act].mean().item())
        pa, pb, ps = (gather_k(F[x], k)[act] for x in ("phi_a", "phi_b", "psi"))
        R_minus.append(circular_resultant(pa + pb - ps).item())
        R_plus.append(circular_resultant(pa + pb + ps).item())
        sinD.append(torch.sin(pa + pb + sign * ps).abs().mean().item())
    K_t = torch.stack(K_t)  # (T, H)

    # lock-in: first checkpoint after which k* never changes again
    differs = (K_t != K_t[-1]).numpy()
    last_diff = np.where(differs.any(0), differs.shape[0] - 1 - np.argmax(differs[::-1], axis=0), -1)
    lock_step = steps[np.minimum(last_diff + 1, len(steps) - 1)]
    lock_step[last_diff < 0] = 0

    def k_at(step):
        i = int(np.searchsorted(steps, step, side="right") - 1)
        return K_t[max(i, 0)]

    t_fit, t_grok = summ["train_99_step"], summ["test_99_step"]
    t_end = t_grok if t_grok is not None else steps[-1]
    switch = None
    if t_fit is not None:
        switch = (k_at(t_fit)[act] != k_at(t_end)[act]).float().mean().item()

    am = act.numpy()
    key_set = torch.tensor(sorted(keys)) if keys else torch.tensor([], dtype=torch.long)
    in_keys = lambda x: torch.isin(x, key_set)  # noqa: E731
    return {
        "summary": summ, "hist": hist, "steps": steps.tolist(),
        "n_active": int(act.sum()), "key_freqs": keys,
        "frac_active_on_keys": in_keys(kfin[act]).float().mean().item() if keys else None,
        "correct": {n: (preds[n][act] == kfin[act]).numpy() for n in PREDICTORS},
        "correct_top3": (top3[act] == kfin[act].unsqueeze(-1)).any(-1).numpy(),
        # key-frequency over-representation among init winners (all neurons)
        "init_pred_on_keys": {n: in_keys(preds[n]).float().mean().item() for n in PREDICTORS} if keys else None,
        "keys_expected_frac": len(keys) / ((p - 1) // 2),
        "lock_step": lock_step[am].tolist(), "t_fit": t_fit, "t_grok": t_grok,
        "switch_frac": switch,
        "ipr_t": ipr_t, "neff_t": neff_t, "neff_final": neff_t[-1], "R_minus_t": R_minus, "R_plus_t": R_plus, "sinD_t": sinD,
        "final_ipr_active": Ffin["ipr"][act].mean().item(),
    }


def boot_ci(x, n=2000, rng=np.random.default_rng(0)):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (math.nan, math.nan)
    b = rng.choice(x, size=(n, len(x))).mean(1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/exp01/phase1b")
    ap.add_argument("--p", type=int, default=47)
    ap.add_argument("--sign", type=int, default=-1)
    ap.add_argument("--out", default="experiments/01_frequency_lottery/figures")
    args = ap.parse_args()
    K = (args.p - 1) // 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for d in sorted(Path(args.runs).iterdir()):
        m = re.match(r"(grok|full)_is([\d.]+)_s(\d+)", d.name)
        if m and (d / "final.pt").exists():
            runs[(m.group(1), float(m.group(2)), int(m.group(3)))] = analyze_run(d, args.p, args.sign)
            print(f"analyzed {d.name}", flush=True)

    groups = defaultdict(list)
    for (regime, scale, seed), r in runs.items():
        groups[(regime, scale)].append(r)

    res = {"K": K, "chance": 1 / K, "sign": args.sign, "conditions": {}, "runs": {}}
    for (regime, scale), rs in sorted(groups.items()):
        c = {"n_runs": len(rs), "n_active": sum(r["n_active"] for r in rs)}
        for n in PREDICTORS:
            x = np.concatenate([r["correct"][n] for r in rs])
            c[n] = {"acc": float(x.mean()), "ci95": boot_ci(x)}
        x = np.concatenate([r["correct_top3"] for r in rs])
        c["growth_top3"] = {"acc": float(x.mean()), "ci95": boot_ci(x), "chance": 3 / K}
        c["switch_frac"] = [r["switch_frac"] for r in rs]
        c["t_fit"] = [r["t_fit"] for r in rs]
        c["t_grok"] = [r["t_grok"] for r in rs]
        lock = np.concatenate([r["lock_step"] for r in rs])
        c["lock_median"] = float(np.median(lock))
        if regime == "grok" and all(r["t_grok"] for r in rs):
            c["frac_locked_before_fit"] = float(np.mean(np.concatenate(
                [np.array(r["lock_step"]) <= r["t_fit"] for r in rs])))
            c["frac_locked_before_grok"] = float(np.mean(np.concatenate(
                [np.array(r["lock_step"]) <= r["t_grok"] for r in rs])))
        c["final_R_minus"] = [r["R_minus_t"][-1] for r in rs]
        c["final_R_plus"] = [r["R_plus_t"][-1] for r in rs]
        c["final_ipr"] = [r["final_ipr_active"] for r in rs]
        c["neff_final"] = [r["neff_final"] for r in rs]
        c["key_freqs"] = [r["key_freqs"] for r in rs]
        c["init_pred_on_keys"] = [r["init_pred_on_keys"] for r in rs]
        c["keys_expected_frac"] = [r["keys_expected_frac"] for r in rs]
        c["frac_active_on_keys"] = [r["frac_active_on_keys"] for r in rs]
        res["conditions"][f"{regime}_is{scale}"] = c
    for (regime, scale, seed), r in runs.items():
        res["runs"][f"{regime}_is{scale}_s{seed}"] = {k: r[k] for k in (
            "n_active", "key_freqs", "t_fit", "t_grok", "switch_frac", "final_ipr_active", "neff_final")}
    (out / "phase1b_results.json").write_text(json.dumps(res, indent=2, default=float))

    # ---------- plots ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def clean(ax):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.25)

    conds = [k for k in res["conditions"]]
    # 1. predictor accuracy
    fig, ax = plt.subplots(figsize=(10, 3.8))
    width = 0.2
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    names = PREDICTORS + ["growth_top3"]
    xs = np.arange(len(conds))
    for i, n in enumerate(names):
        acc = [res["conditions"][c][n]["acc"] for c in conds]
        lo = [a - res["conditions"][c][n]["ci95"][0] for a, c in zip(acc, conds)]
        hi = [res["conditions"][c][n]["ci95"][1] - a for a, c in zip(acc, conds)]
        ax.bar(xs + (i - 1.5) * width, acc, width * 0.9, color=cols[i], label=n, yerr=[lo, hi],
               error_kw={"lw": 1, "ecolor": "#444"})
    ax.axhline(1 / K, color="#444", ls="--", lw=1, label="chance 1/K")
    ax.axhline(3 / K, color="#444", ls=":", lw=1, label="chance 3/K (top-3)")
    ax.set_xticks(xs, [c.replace("_is", "\ninit ") for c in conds])
    ax.set(ylabel="fraction of active neurons", title="Init-time predictors vs final k* (H=512, 3 seeds pooled, 95% bootstrap CI)")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    clean(ax)
    fig.tight_layout()
    fig.savefig(out / "phase1b_predictor_accuracy.png", dpi=130)

    # 2. lock-in ECDF (grok regime)
    fig, ax = plt.subplots(figsize=(7, 4))
    for scale, col in SCALE_COLOR.items():
        rs = groups.get(("grok", scale))
        if not rs:
            continue
        lock = np.sort(np.concatenate([r["lock_step"] for r in rs]))
        ax.step(np.maximum(lock, 1), np.arange(1, len(lock) + 1) / len(lock), where="post", color=col,
                lw=2, label=f"init {scale}")
        tf = np.median([r["t_fit"] for r in rs])
        tg = [r["t_grok"] for r in rs if r["t_grok"]]
        ax.axvline(tf, color=col, ls=":", lw=1)
        if tg:
            ax.axvline(np.median(tg), color=col, ls="--", lw=1)
    ax.set(xscale="log", xlabel="lock-in step (k* never changes afterwards)", ylabel="fraction of active neurons",
           title="Lock-in time, grokking split (dotted = median t_fit, dashed = median t_grok)")
    ax.legend(frameon=False)
    clean(ax)
    fig.tight_layout()
    fig.savefig(out / "phase1b_lockin.png", dpi=130)

    # 3. dynamics: accuracy + IPR + phase concentration, seed 0 of each grok condition
    scales = [s for s in SCALE_COLOR if ("grok", s, 0) in runs]
    fig, axes = plt.subplots(2, len(scales), figsize=(5 * len(scales), 6.5), squeeze=False, sharex="col")
    for j, s in enumerate(scales):
        r = runs[("grok", s, 0)]
        hs = np.maximum(r["hist"]["step"], 1)
        a = axes[0, j]
        a.plot(hs, r["hist"]["train_acc"], color=BLUE, lw=2, label="train acc")
        a.plot(hs, r["hist"]["test_acc"], color=ORANGE, lw=2, label="test acc")
        a.set(title=f"grok split, init scale {s}, seed 0", ylabel="accuracy", xscale="log")
        b = axes[1, j]
        st = np.maximum(r["steps"], 1)
        b.plot(st, r["ipr_t"], color=AQUA, lw=2, marker="o", ms=3, label="mean IPR (active)")
        b.plot(st, r["R_minus_t"], color=YELLOW, lw=2, label="R(φa+φb−ψ) at k*")
        b.plot(st, r["R_plus_t"], color="#999", lw=1.5, ls="--", label="R(φa+φb+ψ) at k*")
        b.plot(st, r["sinD_t"], color="#e87ba4", lw=2, label="mean |sin D|")
        b.plot(st, np.array(r["neff_t"]) / K, color="#444", lw=1.5, label="effective # freqs / K")
        b.set(xlabel="step", ylabel="value", xscale="log", ylim=(0, 1.02))
        for ax in (a, b):
            for t, ls in ((r["t_fit"], ":"), (r["t_grok"], "--")):
                if t:
                    ax.axvline(max(t, 1), color="#444", ls=ls, lw=0.8)
            clean(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "phase1b_dynamics.png", dpi=130)

    # console summary
    print(f"\nchance 1/K = {1 / K:.3f}")
    for c, v in res["conditions"].items():
        print(f"{c:14s} n_act={v['n_active']:4d} | " + " | ".join(
            f"{n.replace('pred_', '')} {v[n]['acc']:.3f}" for n in names) +
            f" | switch {['%.2f' % x if x is not None else '-' for x in v['switch_frac']]}"
            f" | t_fit {v['t_fit']} t_grok {v['t_grok']} | lock med {v['lock_median']:.0f}"
            f" | R- {np.mean(v['final_R_minus']):.2f} R+ {np.mean(v['final_R_plus']):.2f}"
            f" | IPR {np.mean(v['final_ipr']):.2f} | n_eff {['%.1f' % x for x in v['neff_final']]}"
            f" | n_keys {[len(k) for k in v['key_freqs']]}")


if __name__ == "__main__":
    main()
