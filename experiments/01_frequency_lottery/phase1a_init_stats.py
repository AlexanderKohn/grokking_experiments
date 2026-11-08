"""Exp 01, Phase 1A: init-only Monte Carlo of predicted frequency winners.

For each width H and seed, builds the exact theta_0 that train.py would use (grokking.model.make_model),
computes each neuron's predicted winning frequency under three predictors, and records per-network
coverage statistics. The predictors are invariant to --init-scale, because all quantities are
scale-normalized.

Outputs (in --out):
  init_stats.csv   one row per (H, seed, predictor): distinct, min_count, max_count, n_none
  counts.npz       raw per-network counts over k (index 0 = no winner)
  summary.json     chi-square uniformity tests, predictor agreement, coupon-collector comparison
  *.png            plots

Usage:
  python experiments/01_frequency_lottery/phase1a_init_stats.py --p 47 --seeds 1000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from grokking.fourier import neuron_fourier, split_weights, winner_predictions  # noqa: E402
from grokking.model import make_model  # noqa: E402

PREDICTORS = ["pred_power", "pred_growth", "pred_growth_pos"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def coupon_expected(K, H):
    return K * (1 - (1 - 1 / K) ** np.asarray(H, dtype=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=47)
    ap.add_argument("--widths", default="4,8,16,32,64,128,256,512,1024")
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--mlp-input", default="onehot", choices=["onehot", "tied"])
    ap.add_argument("--sign", type=int, default=-1, help="D = phi_a + phi_b + sign*psi")
    ap.add_argument("--out", default="runs/exp01/phase1a")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    p, K = args.p, (args.p - 1) // 2
    widths = [int(h) for h in args.widths.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    counts = np.zeros((len(widths), args.seeds, len(PREDICTORS), K + 1), dtype=np.int32)
    agree = np.zeros((len(widths), args.seeds))  # fraction of neurons where power == growth
    t0 = time.time()
    for i, H in enumerate(widths):
        for s in range(args.seeds):
            model = make_model("mlp", p, s, mlp_input=args.mlp_input, d_hidden=H)
            F = neuron_fourier(*split_weights(model.state_dict(), p), sign=args.sign)
            preds = winner_predictions(F)
            for j, name in enumerate(PREDICTORS):
                counts[i, s, j] = np.bincount(preds[name].numpy(), minlength=K + 1)
            agree[i, s] = (preds["pred_power"] == preds["pred_growth"]).float().mean().item()
        print(f"H={H:5d} done ({time.time() - t0:.0f}s)", flush=True)

    # per-network statistics
    rows = []
    for i, H in enumerate(widths):
        for s in range(args.seeds):
            for j, name in enumerate(PREDICTORS):
                c = counts[i, s, j, 1:]
                rows.append((H, s, name, int((c > 0).sum()), int(c.min()), int(c.max()),
                             int(counts[i, s, j, 0])))
    with open(out / "init_stats.csv", "w") as f:
        f.write("H,seed,predictor,distinct,min_count,max_count,n_none\n")
        f.writelines(",".join(map(str, r)) + "\n" for r in rows)
    np.savez_compressed(out / "counts.npz", counts=counts, widths=np.array(widths),
                        predictors=np.array(PREDICTORS), p=p)

    # multinomial reference: H neurons assigned uniformly and independently to K frequencies
    rng = np.random.default_rng(0)
    ref = {}
    for H in widths:
        m = rng.multinomial(H, np.full(K, 1 / K), size=args.seeds)
        ref[H] = {"distinct": (m > 0).sum(1), "min": m.min(1), "max": m.max(1)}

    summary = {"p": p, "K": K, "seeds": args.seeds, "widths": widths, "sign": args.sign,
               "mlp_input": args.mlp_input, "per_width": {}}
    for i, H in enumerate(widths):
        d = {"agree_power_growth": float(agree[i].mean())}
        for j, name in enumerate(PREDICTORS):
            pooled = counts[i, :, j, 1:].sum(0)
            chi2, pval = stats.chisquare(pooled)
            st = counts[i, :, j, 1:]
            d[name] = {
                "chi2": float(chi2), "chi2_p": float(pval),
                "max_rel_dev": float(np.abs(pooled / pooled.mean() - 1).max()),
                "frac_none": float(counts[i, :, j, 0].sum() / (H * args.seeds)),
                "distinct_mean": float((st > 0).sum(1).mean()),
                "min_count_mean": float(st.min(1).mean()),
                "max_count_mean": float(st.max(1).mean()),
            }
        d["coupon_expected_distinct"] = float(coupon_expected(K, H))
        d["multinomial_ref"] = {k: float(v.mean()) for k, v in ref[H].items()}
        summary["per_width"][str(H)] = d
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def q(a):
        return np.percentile(a, [25, 50, 75])

    j = PREDICTORS.index("pred_growth")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    Hs = np.array(widths)
    for ax, key, ylabel in ((axes[0], "distinct", "distinct predicted frequencies"),
                            (axes[1], "min", "min neurons on any frequency"),
                            (axes[2], "max", "max neurons on any frequency")):
        st = counts[:, :, j, 1:]
        obs = {"distinct": (st > 0).sum(-1), "min": st.min(-1), "max": st.max(-1)}[key]
        qs = np.array([q(o) for o in obs])
        rq = np.array([q(ref[H][key]) for H in widths])
        ax.fill_between(Hs, qs[:, 0], qs[:, 2], color=BLUE, alpha=0.2, lw=0)
        ax.plot(Hs, qs[:, 1], color=BLUE, lw=2, marker="o", ms=5, label="init (pred_growth), median ± IQR")
        ax.plot(Hs, rq[:, 1], color=ORANGE, lw=2, ls="--", label="uniform multinomial, median")
        if key == "distinct":
            ax.plot(Hs, coupon_expected(K, Hs), color="#555", lw=1, ls=":", label="coupon-collector E[distinct]")
            ax.axhline(K, color="#999", lw=0.8)
        ax.set(xscale="log", xlabel="hidden width H", ylabel=ylabel)
        ax.set_xticks(Hs, [str(h) for h in Hs])
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Phase 1A: predicted winners at init (p={p}, K={K}, {args.seeds} seeds per width)")
    fig.tight_layout()
    fig.savefig(out / "phase1a_coverage.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    iH = widths.index(512) if 512 in widths else len(widths) - 1
    pooled = counts[iH, :, j, 1:].sum(0)
    ax.bar(np.arange(1, K + 1), pooled / pooled.sum(), color=BLUE, width=0.8)
    ax.axhline(1 / K, color=ORANGE, ls="--", lw=1.5, label="uniform 1/K")
    ax.set(xlabel="frequency k", ylabel="fraction of neurons",
           title=f"pred_growth winners pooled over seeds, H={widths[iH]} "
                 f"(χ² p={summary['per_width'][str(widths[iH])]['pred_growth']['chi2_p']:.2f})")
    ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "phase1a_winner_hist.png", dpi=130)
    print(f"saved to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
