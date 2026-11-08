"""Exp 01, Phase 1D: causal interventions on theta_0 (H = 128, near the grokking threshold).

  make     build the edited initial state_dicts for each seed (runs/exp01/phase1d/inits/)
  analyze  summarize the retrained runs (figures/phase1d_*)

Interventions, applied to Wa, Wb (rows of W1) and W2 (columns); biases are untouched. Every edited row or column is
rescaled to its original L2 norm, and the DC bin is kept throughout, so only the frequency content changes.
  baseline       unedited theta_0 (sanity check: should reproduce the Phase 1C run exactly)
  knockout       remove the seed's top-5 final frequencies (by contribution share in the 1C run)
  knockout_ctrlN remove 5 random frequencies drawn from the rest (control for "removing any 5")
  mono_k{k}      project every neuron onto the single frequency k
  rigged         round-robin assignment k_m = (m mod K) + 1, with that frequency boosted to `boost` x the
                 neuron's largest init magnitude and phase-aligned (psi = phi_a + phi_b, so D = 0)
  rigged_random  same boost and alignment, but k_m drawn i.i.d. uniformly (multinomial coverage)

Usage:
  python experiments/01_frequency_lottery/phase1d_interventions.py make --seeds 0 1 2
  bash   experiments/01_frequency_lottery/run_phase1d.sh
  python experiments/01_frequency_lottery/phase1d_interventions.py analyze
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from grokking.fourier import (active_neurons, neuron_contribution, neuron_fourier,  # noqa: E402
                              project_to_frequencies, split_weights, winner_predictions)
from grokking.model import make_model  # noqa: E402

P, H = 47, 128
K = (P - 1) // 2
ROOT = Path("runs/exp01/phase1d")
MONO_KS = (1, 8, 17)


def contribution_share(sd, p=P):
    Wa, Wb, Wo = split_weights(sd, p)
    F = neuron_fourier(Wa, Wb, Wo)
    c = neuron_contribution(Wa, Wb, Wo)
    q = torch.bincount(F["kstar"], weights=c.double(), minlength=K + 1)[1:]
    return (q / q.sum()).numpy()


def rescale_rows(new, old):
    """Rescale each row of `new` (H, p) to the L2 norm of the matching row of `old`."""
    n_new = new.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return new * old.norm(dim=-1, keepdim=True) / n_new


def assemble(sd, Wa, Wb, Wo):
    out = {k: v.clone() for k, v in sd.items()}
    out["net.0.weight"] = torch.cat([Wa, Wb], dim=1).float()
    out["net.2.weight"] = Wo.T.contiguous().float()
    return out


def project_all(sd, keep):
    Wa, Wb, Wo = split_weights(sd, P)
    new = [rescale_rows(project_to_frequencies(W, keep, keep_dc=True), W) for W in (Wa, Wb, Wo)]
    return assemble(sd, *new)


def rig(sd, assign, boost=2.0):
    """Boost each neuron's assigned frequency and phase-align it. assign: (H,) ints in 1..K."""
    Wa, Wb, Wo = split_weights(sd, P)
    Xa, Xb, Xo = (torch.fft.fft(W.double(), dim=-1) for W in (Wa, Wb, Wo))
    idx = torch.arange(H)
    k = assign.long()
    phi_a, phi_b = Xa[idx, k].angle(), Xb[idx, k].angle()
    psi = phi_a + phi_b                              # D = phi_a + phi_b - psi = 0
    for X, phase in ((Xa, phi_a), (Xb, phi_b), (Xo, psi)):
        mag = X[:, 1:K + 1].abs().max(-1).values * boost
        val = torch.polar(mag, phase)
        X[idx, k] = val
        X[idx, (P - k) % P] = val.conj()
    new = [rescale_rows(torch.fft.ifft(X, dim=-1).real, W) for X, W in ((Xa, Wa), (Xb, Wb), (Xo, Wo))]
    return assemble(sd, *new)


def cmd_make(args):
    out = ROOT / "inits"
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for seed in args.seeds:
        base = make_model("mlp", P, seed, d_hidden=H).state_dict()
        ref = Path(f"runs/exp01/phase1c/H{H}_is1.0_s{seed}/final.pt")
        q = contribution_share(torch.load(ref))
        top5 = sorted((np.argsort(q)[::-1][:5] + 1).tolist())
        rest = [k for k in range(1, K + 1) if k not in top5]
        rng = np.random.default_rng(1000 + seed)
        variants = {"baseline": base,
                    "knockout": project_all(base, [k for k in range(1, K + 1) if k not in top5])}
        ctrl_sets = []
        for c in range(2):
            drop = sorted(rng.choice(rest, size=5, replace=False).tolist())
            ctrl_sets.append(drop)
            variants[f"knockout_ctrl{c}"] = project_all(base, [k for k in range(1, K + 1) if k not in drop])
        for k in MONO_KS:
            variants[f"mono_k{k}"] = project_all(base, [k])
        rr = torch.arange(H) % K + 1
        variants["rigged"] = rig(base, rr, args.boost)
        variants["rigged_random"] = rig(base, torch.from_numpy(rng.integers(1, K + 1, size=H)), args.boost)

        checks = {}
        for name, sd in variants.items():
            torch.save(sd, out / f"s{seed}_{name}.pt")
            F = neuron_fourier(*split_weights(sd, P))
            pred = winner_predictions(F)["pred_growth"]
            counts = np.bincount(pred.numpy(), minlength=K + 1)[1:]
            checks[name] = {"pred_distinct": int((counts > 0).sum()), "pred_max_count": int(counts.max()),
                            "pred_min_count": int(counts.min())}
        checks["rigged"]["pred_matches_assignment"] = float(
            (winner_predictions(neuron_fourier(*split_weights(variants["rigged"], P)))["pred_growth"] == rr)
            .float().mean())
        meta[seed] = {"top5": top5, "top5_share": float(q[np.array(top5) - 1].sum()),
                      "ctrl_sets": ctrl_sets, "init_checks": checks}
        print(f"seed {seed}: top5 {top5} (share {meta[seed]['top5_share']:.2f}), controls {ctrl_sets}")
        for name, c in checks.items():
            print(f"   {name:16s} {c}")
    (ROOT / "meta.json").write_text(json.dumps(meta, indent=2))


def cmd_analyze(args):
    meta = {int(k): v for k, v in json.loads((ROOT / "meta.json").read_text()).items()}
    rows = []
    for d in sorted(ROOT.glob("s*_*")):
        m = re.match(r"s(\d+)_(.+)$", d.name)
        if not m or not (d / "history.json").exists():
            continue
        seed, name = int(m.group(1)), m.group(2)
        h = json.loads((d / "history.json").read_text())
        s = h["summary"]
        sd = torch.load(d / "final.pt")
        q = contribution_share(sd)
        Wa, Wb, Wo = split_weights(sd, P)
        F = neuron_fourier(Wa, Wb, Wo)
        act = active_neurons(Wa, Wb, Wo)
        top5 = meta[seed]["top5"]
        rows.append({
            "seed": seed, "variant": name, "t95": s["test_95_step"], "t99": s["test_99_step"],
            "t_fit": s["train_99_step"], "final_test_acc": s["final_test_acc"],
            "n_eff_freqs": float(1 / (q**2).sum()), "final_distinct_active": int(F["kstar"][act].unique().numel()),
            "share_on_baseline_top5": float(q[np.array(top5) - 1].sum()),
            "share_on_mono_k": {k: float(q[k - 1]) for k in MONO_KS},
            "hist": {k: h["history"][k] for k in ("step", "test_acc", "train_acc")},
        })
    rows.sort(key=lambda r: (r["variant"], r["seed"]))
    byv = defaultdict(list)
    for r in rows:
        byv[r["variant"]].append(r)
    order = ["baseline", "knockout", "knockout_ctrl0", "knockout_ctrl1", "mono_k1", "mono_k8", "mono_k17",
             "rigged", "rigged_random"]
    order = [v for v in order if v in byv]

    # sanity: baseline should reproduce the 1C run exactly
    sanity = {}
    for r in byv.get("baseline", []):
        ref = json.loads(Path(f"runs/exp01/phase1c/H{H}_is1.0_s{r['seed']}/history.json").read_text())["summary"]
        sanity[r["seed"]] = {"t95": r["t95"], "t95_1c": ref["test_95_step"],
                             "final_test_acc": r["final_test_acc"], "final_test_acc_1c": ref["final_test_acc"]}

    out = Path("experiments/01_frequency_lottery/figures")
    summary = {"meta": meta, "sanity_baseline_vs_1c": sanity, "variants": {}}
    print(f"{'variant':16s} {'t95 per seed':>24s} {'final test acc':>22s} {'n_eff':>14s} {'distinct':>10s} "
          f"{'share on base top5':>20s}")
    for v in order:
        rs = byv[v]
        summary["variants"][v] = [{k: r[k] for k in r if k != "hist"} for r in rs]
        print(f"{v:16s} {str([r['t95'] for r in rs]):>24s} "
              f"{str([round(r['final_test_acc'], 3) for r in rs]):>22s} "
              f"{str([round(r['n_eff_freqs'], 1) for r in rs]):>14s} "
              f"{str([r['final_distinct_active'] for r in rs]):>10s} "
              f"{str([round(r['share_on_baseline_top5'], 2) for r in rs]):>20s}")
    print("sanity (baseline vs 1C):", sanity)
    (out / "phase1d_results.json").write_text(json.dumps(summary, indent=2, default=float))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"baseline": "#222222", "knockout": "#eb6834", "knockout_ctrl0": "#f3a27e",
              "knockout_ctrl1": "#f3a27e", "mono_k1": "#2a78d6", "mono_k8": "#6da7ec", "mono_k17": "#184f95",
              "rigged": "#1baf7a", "rigged_random": "#8fd9bd"}
    styles = {"knockout_ctrl0": "--", "knockout_ctrl1": ":", "rigged_random": "--"}
    groups = [("Knockout", ["baseline", "knockout", "knockout_ctrl0", "knockout_ctrl1"]),
              ("Monoculture", ["baseline", "mono_k1", "mono_k8", "mono_k17"]),
              ("Rigged coverage", ["baseline", "rigged", "rigged_random"])]
    seeds = sorted({r["seed"] for r in rows})
    fig, axes = plt.subplots(len(seeds), 3, figsize=(15, 3.4 * len(seeds)), squeeze=False, sharey=True)
    for i, seed in enumerate(seeds):
        for j, (title, vs) in enumerate(groups):
            ax = axes[i, j]
            for v in vs:
                r = next((r for r in byv.get(v, []) if r["seed"] == seed), None)
                if r is None:
                    continue
                ax.plot(np.maximum(r["hist"]["step"], 1), r["hist"]["test_acc"], color=colors[v],
                        ls=styles.get(v, "-"), lw=2 if v != "baseline" else 1.5, label=v)
            ax.axhline(0.95, color="#999", lw=0.8, ls=":")
            ax.set(xscale="log", title=f"{title}, seed {seed}", xlabel="step",
                   ylabel="test accuracy" if j == 0 else None, ylim=(-0.02, 1.02))
            ax.grid(alpha=0.25)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Phase 1D interventions on θ₀ (H={H}, p={P}, frac 0.5); dotted line = 95%")
    fig.tight_layout()
    fig.savefig(out / "phase1d_curves.png", dpi=130)
    print(f"saved {out / 'phase1d_curves.png'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    m.add_argument("--boost", type=float, default=2.0)
    sub.add_parser("analyze")
    args = ap.parse_args()
    torch.set_num_threads(2)
    {"make": cmd_make, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
