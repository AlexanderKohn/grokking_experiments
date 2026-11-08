"""Exp 01, Phase 1C analysis: width sweep and the within-width lottery test.

For each run in runs/exp01/phase1c/H{H}_is{scale}_s{seed}/, this records grokking outcome and timing, the
final-network frequency structure, and init-time coverage statistics. The init statistics are recomputed from
make_model(seed, H), which is the same theta_0 the run trained from.

Key test: at fixed H, do seeds with better init coverage or redundancy grok earlier? Reports Spearman rho per
width, with a permutation p-value, plus a width-stratified combined permutation test.

Usage: python experiments/01_frequency_lottery/phase1c_analysis.py [--scale 1.0]
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from grokking.fourier import (active_neurons, key_frequencies, neuron_contribution,  # noqa: E402
                              neuron_fourier, split_weights, winner_predictions)
from grokking.model import make_model  # noqa: E402

RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
ORANGE, BLUE = "#eb6834", "#2a78d6"
INIT_STATS = ["distinct", "min_count", "max_count"]


def init_stats(p, H, seed):
    model = make_model("mlp", p, seed, d_hidden=H)
    preds = winner_predictions(neuron_fourier(*split_weights(model.state_dict(), p)))
    K = (p - 1) // 2
    out = {}
    for name in ("pred_growth", "pred_power"):
        c = np.bincount(preds[name].numpy(), minlength=K + 1)[1:]
        tag = name.replace("pred_", "")
        out[f"{tag}_distinct"] = int((c > 0).sum())
        out[f"{tag}_min_count"] = int(c.min())
        out[f"{tag}_max_count"] = int(c.max())
    return out


def final_stats(d, p):
    K = (p - 1) // 2
    Wa, Wb, Wo = split_weights(torch.load(d / "final.pt"), p)
    F = neuron_fourier(Wa, Wb, Wo)
    act = active_neurons(Wa, Wb, Wo)
    keys = key_frequencies(F["kstar"], act)
    c = neuron_contribution(Wa, Wb, Wo)
    q = torch.bincount(F["kstar"], weights=c.double(), minlength=K + 1)[1:]
    q = q / q.sum()
    return {
        "n_active": int(act.sum()),
        "final_distinct_active": int(F["kstar"][act].unique().numel()),
        "n_key_freqs": len(keys),
        "neurons_per_key_mean": float(np.mean(list(keys.values()))) if keys else 0.0,
        "neurons_per_key_min": int(min(keys.values())) if keys else 0,
        "n_eff_freqs": float(1 / q.pow(2).sum()),
        "final_ipr_active": float(F["ipr"][act].mean()),
    }


def _rank_corr(rx, RY):
    """Pearson correlation of rank vector rx (n,) with each row of RY (m, n) -> (m,)."""
    rx = rx - rx.mean()
    RY = RY - RY.mean(-1, keepdims=True)
    return (RY @ rx) / (np.sqrt((rx**2).sum()) * np.sqrt((RY**2).sum(-1)))


def _perms(ry, n_perm, rng):
    return np.array([rng.permutation(ry) for _ in range(n_perm)])


def spearman_perm(x, y, n_perm=10000, rng=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None
    rng = rng or np.random.default_rng(0)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rho = _rank_corr(rx, ry[None])[0]
    null = _rank_corr(rx, _perms(ry, n_perm, rng))
    return float(rho), float((np.sum(np.abs(null) >= abs(rho) - 1e-12) + 1) / (n_perm + 1))


def stratified_test(groups, n_perm=10000, rng=None):
    """Mean of within-group Spearman rho; null permutes y within each group. groups: list of (x, y)."""
    rng = rng or np.random.default_rng(1)
    usable = [(np.asarray(x, float), np.asarray(y, float)) for x, y in groups
              if np.ptp(x) > 0 and np.ptp(y) > 0]
    if not usable:
        return None, None, 0
    obs, null = 0.0, np.zeros(n_perm)
    for x, y in usable:
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        obs += _rank_corr(rx, ry[None])[0]
        null += _rank_corr(rx, _perms(ry, n_perm, rng))
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)
    return float(obs / len(usable)), float(p), len(usable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/exp01/phase1c")
    ap.add_argument("--p", type=int, default=47)
    ap.add_argument("--scale", default="1.0")
    ap.add_argument("--cap", type=int, default=50000)
    ap.add_argument("--out", default="experiments/01_frequency_lottery/figures")
    args = ap.parse_args()
    p, K = args.p, (args.p - 1) // 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in sorted(Path(args.runs).iterdir()):
        m = re.match(rf"H(\d+)_is{re.escape(args.scale)}_s(\d+)$", d.name)
        if not m or not (d / "history.json").exists():
            continue
        H, seed = int(m.group(1)), int(m.group(2))
        s = json.loads((d / "history.json").read_text())["summary"]
        r = {"H": H, "seed": seed, "grokked": s["test_99_step"] is not None,
             "t_fit": s["train_99_step"], "t_grok": s["test_99_step"], "t_grok_95": s["test_95_step"],
             "grok_ratio": s["grok_ratio"], "log10_grok_ratio": s["log10_grok_ratio"],
             "final_train_acc": s["final_train_acc"], "final_test_acc": s["final_test_acc"]}
        # censored time: runs that never grok count as the cap (ranks them last)
        r["t_grok_cens"] = r["t_grok"] if r["grokked"] else args.cap
        r["grokked_95"] = r["t_grok_95"] is not None
        r["t_grok95_cens"] = r["t_grok_95"] if r["grokked_95"] else args.cap
        r.update(init_stats(p, H, seed))
        r.update(final_stats(d, p))
        rows.append(r)
    rows.sort(key=lambda r: (r["H"], r["seed"]))
    print(f"loaded {len(rows)} runs")

    byH = defaultdict(list)
    for r in rows:
        byH[r["H"]].append(r)
    widths = sorted(byH)

    per_width = {}
    for H in widths:
        rs = byH[H]
        tg = np.array([r["t_grok_cens"] for r in rs])
        g = [r for r in rs if r["grokked"]]
        per_width[H] = {
            "n": len(rs), "p_grok": len(g) / len(rs),
            "p_grok_95": float(np.mean([r["grokked_95"] for r in rs])),
            "t_grok95_cens_q": np.percentile([r["t_grok95_cens"] for r in rs], [25, 50, 75]).tolist(),
            "t_grok_cens_q": np.percentile(tg, [25, 50, 75]).tolist(),
            "t_fit_median": float(np.median([r["t_fit"] if r["t_fit"] is not None else args.cap for r in rs])),
            "fit_frac": float(np.mean([r["t_fit"] is not None for r in rs])),
            "log10_grok_ratio_median": float(np.median([r["log10_grok_ratio"] for r in g])) if g else None,
            "final_test_acc_median": float(np.median([r["final_test_acc"] for r in rs])),
            "n_key_freqs_median": float(np.median([r["n_key_freqs"] for r in g])) if g else None,
            "n_eff_freqs_median": float(np.median([r["n_eff_freqs"] for r in g])) if g else None,
            "final_distinct_median": float(np.median([r["final_distinct_active"] for r in g])) if g else None,
            "init_distinct_median": float(np.median([r["growth_distinct"] for r in rs])),
            "coupon_expected": K * (1 - (1 - 1 / K) ** H),
        }

    # within-width lottery tests. Outcomes: censored t_grok at 99% and 95% (lower is better) and final test
    # accuracy (higher is better). Final-network stats are included as descriptive (not init-time) correlates.
    OUTCOMES = {"t_grok99": "t_grok_cens", "t_grok95": "t_grok95_cens", "final_test_acc": "final_test_acc"}
    PREDS = [f"growth_{s}" for s in INIT_STATS] + [f"power_{s}" for s in INIT_STATS] + \
        ["n_eff_freqs", "final_distinct_active", "neurons_per_key_mean"]
    tests = {}
    for oname, okey in OUTCOMES.items():
        tests[oname] = {}
        for stat in PREDS:
            per = {}
            for H in widths:
                rs = byH[H]
                rho, pv = spearman_perm([r[stat] for r in rs], [r[okey] for r in rs])
                per[H] = {"rho": rho, "p": pv}
            comb_rho, comb_p, n_used = stratified_test(
                [([r[stat] for r in byH[H]], [r[okey] for r in byH[H]]) for H in widths])
            tests[oname][stat] = {"per_width": per, "mean_rho": comb_rho, "stratified_p": comb_p,
                                  "n_widths": n_used}

    res = {"p": p, "K": K, "scale": args.scale, "cap": args.cap, "per_width": per_width, "tests": tests,
           "runs": rows}
    (out / "phase1c_results.json").write_text(json.dumps(res, indent=2, default=float))

    # ---------- plots ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def clean(ax):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.25)
        ax.set_xscale("log")
        ax.set_xticks(widths, [str(h) for h in widths])
        ax.minorticks_off()

    Hs = np.array(widths)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    ax.plot(Hs, [per_width[H]["p_grok"] for H in widths], color=BLUE, lw=2, marker="o", ms=6,
            label="test ≥ 99%")
    ax.plot(Hs, [per_width[H]["p_grok_95"] for H in widths], color=ORANGE, lw=2, marker="s", ms=6, ls="--",
            label="test ≥ 95%")
    ax2 = ax
    ax.set(xlabel="hidden width H", ylabel="P(grok within 50k steps)", ylim=(-0.03, 1.03),
           title="P(grok) vs width (8 seeds)")
    ax.legend(frameon=False, fontsize=8)
    clean(ax)
    ax = axes[1]
    q = np.array([per_width[H]["t_grok95_cens_q"] for H in widths])
    ax.fill_between(Hs, q[:, 0], q[:, 2], color=BLUE, alpha=0.2, lw=0)
    ax.plot(Hs, q[:, 1], color=BLUE, lw=2, marker="o", ms=6, label="median ± IQR")
    for r in rows:
        ax.scatter(r["H"] * np.exp(np.random.default_rng(r["seed"]).uniform(-0.08, 0.08)), r["t_grok95_cens"],
                   s=14, color="#444" if r["grokked_95"] else ORANGE, zorder=3, alpha=0.7)
    ax.axhline(args.cap, color=ORANGE, ls=":", lw=1)
    ax.set(xlabel="hidden width H", ylabel="t_grok, 95% criterion", yscale="log",
           title="t_grok(95%) vs width (orange = never reached 95%, at cap)")
    clean(ax)
    ax = axes[2]
    ax.plot(Hs, [per_width[H]["coupon_expected"] for H in widths], color="#444", ls=":", lw=1.5,
            label="coupon-collector E[coverage]")
    ax.plot(Hs, [per_width[H]["init_distinct_median"] for H in widths], color=BLUE, lw=2, marker="o", ms=5,
            label="init predicted coverage (median)")
    fd = [per_width[H]["final_distinct_median"] for H in widths]
    ax.plot([h for h, f in zip(Hs, fd) if f is not None], [f for f in fd if f is not None], color=ORANGE,
            lw=2, marker="s", ms=5, label="final distinct k* among active (grokked runs)")
    ax.axhline(K, color="#999", lw=0.8)
    ax.set(xlabel="hidden width H", ylabel="frequencies covered", title=f"Coverage (K = {K})")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    clean(ax)
    fig.tight_layout()
    fig.savefig(out / "phase1c_width_sweep.png", dpi=130)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (oname, okey, ylab, ylog) in enumerate([
            ("t_grok95", "t_grok95_cens", "t_grok, 95% criterion (censored at cap)", True),
            ("final_test_acc", "final_test_acc", "final test accuracy", False)]):
        for ax, stat, lab in zip(axes[row], ["growth_distinct", "growth_min_count", "growth_max_count"],
                                 ["init predicted coverage (distinct k)", "init min neurons on any k",
                                  "init max neurons on any k"]):
            for i, H in enumerate(widths):
                rs = byH[H]
                ax.scatter([r[stat] for r in rs], [r[okey] for r in rs], s=22,
                           color=RAMP[i % len(RAMP)], edgecolor="#222", lw=0.4, label=f"H={H}")
            t = tests[oname][stat]
            mr = f"{t['mean_rho']:+.2f}" if t["mean_rho"] is not None else "n/a"
            mp = f"{t['stratified_p']:.2f}" if t["stratified_p"] is not None else "n/a"
            ax.set(xlabel=lab, ylabel=ylab, title=f"within-width mean ρ = {mr}, stratified p = {mp}")
            if ylog:
                ax.set_yscale("log")
            ax.grid(alpha=0.25)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if stat.endswith("max_count"):
                ax.set_xscale("log")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "phase1c_within_width.png", dpi=130)

    # console
    print(f"{'H':>4} {'P(g95)':>6} {'t95 med':>7} {'P(grok)':>7} {'t_grok q25/50/75 (cens)':>26} {'fit':>4} {'log10 ratio':>11} "
          f"{'test acc':>8} {'keys':>5} {'n_eff':>6} {'final dist':>10} {'init dist':>9} {'coupon':>6}")
    for H in widths:
        w = per_width[H]
        fmt = lambda x, f="{:.1f}": "-" if x is None else f.format(x)  # noqa: E731
        print(f"{H:>4} {w['p_grok_95']:>6.2f} {int(w['t_grok95_cens_q'][1]):>7} {w['p_grok']:>7.2f} {'/'.join(str(int(x)) for x in w['t_grok_cens_q']):>26} "
              f"{w['fit_frac']:>4.2f} {fmt(w['log10_grok_ratio_median'], '{:.2f}'):>11} "
              f"{w['final_test_acc_median']:>8.3f} {fmt(w['n_key_freqs_median']):>5} "
              f"{fmt(w['n_eff_freqs_median']):>6} {fmt(w['final_distinct_median']):>10} "
              f"{w['init_distinct_median']:>9.1f} {w['coupon_expected']:>6.1f}")
    print("\nwithin-width Spearman(stat, outcome); per-width rho (p) for widths with variation")
    for oname, ts in tests.items():
        print(f"--- outcome: {oname}")
        for stat, t in ts.items():
            per = " ".join(f"H{H}:{v['rho']:+.2f}({v['p']:.2f})" for H, v in t["per_width"].items()
                           if v["rho"] is not None)
            mr = "n/a" if t["mean_rho"] is None else f"{t['mean_rho']:+.2f}"
            mp = "n/a" if t["stratified_p"] is None else f"{t['stratified_p']:.3f}"
            print(f"  {stat:22s} mean rho {mr} strat p {mp} | {per}")


if __name__ == "__main__":
    main()
