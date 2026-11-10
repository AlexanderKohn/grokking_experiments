"""Exp 01, Phase 1E / E6: rigged init, dose-response and decomposition (H = 128, seeds 0-2, the 1C splits).

  make     write the edited initial weights to runs/exp01/phase1e/e6/inits/
  analyze  t95 for every E6 run, and the S0 collapse across 1C, 1D and E6 (figures/phase1e_e6_*)

Variants (each edited row/column is rescaled to its original L2 norm, as in 1D; random assignment k_m ~ U{1..K}):
  dose_b{b}   the assigned frequency set to b x the neuron's largest init magnitude, phase-aligned (D = 0)
  conc_b{b}   the same magnitudes, but the output phase is drawn at random, so D is uniform
  align       no magnitude change: rotate the output phase of each neuron's init-time winner (pred_growth) so D = 0

S0, the initial coherent Fourier signal, uses raw (unnormalized) weights and each neuron's init-time winner:
  S0 = sum_k | sum_{m: k_m = k} alpha_a alpha_b beta e^{iD} |
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
from grokking.fourier import fourier_coeffs, neuron_fourier, split_weights, winner_predictions  # noqa: E402
from grokking.model import make_model  # noqa: E402

P, H = 47, 128
K = (P - 1) // 2
ROOT = Path("runs/exp01/phase1e/e6")
FIG = Path("experiments/01_frequency_lottery/figures")
BOOSTS = (0.25, 0.5, 1.0, 2.0, 4.0)


def rescale_rows(new, old):
    return new * old.norm(dim=-1, keepdim=True) / new.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def assemble(sd, Wa, Wb, Wo):
    out = {k: v.clone() for k, v in sd.items()}
    out["net.0.weight"] = torch.cat([Wa, Wb], dim=1).float()
    out["net.2.weight"] = Wo.T.contiguous().float()
    return out


def edit(sd, assign, boost=None, out_phase="aligned", rng=None):
    """Set frequency assign[m] of neuron m. boost=None keeps the magnitudes; out_phase 'aligned' or 'random'."""
    Wa, Wb, Wo = split_weights(sd, P)
    n = Wa.shape[0]
    Xa, Xb, Xo = (torch.fft.fft(W.double(), dim=-1) for W in (Wa, Wb, Wo))
    idx, k = torch.arange(n), assign.long()
    phi_a, phi_b = Xa[idx, k].angle(), Xb[idx, k].angle()
    if out_phase == "aligned":
        psi = phi_a + phi_b                              # D = phi_a + phi_b - psi = 0
    else:
        psi = torch.from_numpy(rng.uniform(0, 2 * np.pi, n))
    for X, phase in ((Xa, phi_a), (Xb, phi_b), (Xo, psi)):
        mag = X[idx, k].abs() if boost is None else X[:, 1:K + 1].abs().max(-1).values * boost
        val = torch.polar(mag, phase)
        X[idx, k] = val
        X[idx, (P - k) % P] = val.conj()
    new = [rescale_rows(torch.fft.ifft(X, dim=-1).real, W) for X, W in ((Xa, Wa), (Xb, Wb), (Xo, Wo))]
    return assemble(sd, *new)


def s0(sd):
    """Initial coherent Fourier signal (raw weights) and its parts, using each neuron's init-time winner."""
    Wa, Wb, Wo = split_weights(sd, P)
    win = winner_predictions(neuron_fourier(Wa, Wb, Wo))["pred_growth"]
    (aa, pa), (ab, pb), (be, ps) = (fourier_coeffs(W) for W in (Wa, Wb, Wo))
    idx, k = torch.arange(Wa.shape[0]), win - 1
    amp = (aa[idx, k] * ab[idx, k] * be[idx, k]).double()
    z = torch.polar(amp, (pa[idx, k] + pb[idx, k] - ps[idx, k]).double())
    per_k = torch.zeros(K, dtype=torch.complex128).index_add_(0, k, z)
    return {"S0": float(per_k.abs().sum()), "incoherent": float(amp.sum()),
            "mean_cosD": float(torch.cos(z.angle()).mean())}


def cmd_make(args):
    out = ROOT / "inits"
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        base = make_model("mlp", P, seed, d_hidden=H).state_dict()
        rng = np.random.default_rng(2000 + seed)
        assign = torch.from_numpy(rng.integers(1, K + 1, size=H))
        variants = {}
        for b in BOOSTS:
            variants[f"dose_b{b}"] = edit(base, assign, b, "aligned")
            variants[f"conc_b{b}"] = edit(base, assign, b, "random", rng)
        win = winner_predictions(neuron_fourier(*split_weights(base, P)))["pred_growth"]
        variants["align"] = edit(base, win, None, "aligned")
        for name, sd in variants.items():
            torch.save(sd, out / f"s{seed}_{name}.pt")
            print(f"seed {seed} {name:10s} {s0(sd)}")
        print(f"seed {seed} baseline   {s0(base)}")


def t95(d):
    s = json.loads((Path(d) / "history.json").read_text())["summary"]
    return s["test_95_step"], s["final_test_acc"]


def cmd_analyze(args):
    rows = []
    # 1C (H = 128 and every other width that groks), the 1D edits, and E6
    for d in sorted(Path("runs/exp01/phase1c").glob("H*_is1.0_s*")):
        m = re.match(r"H(\d+)_is1.0_s(\d+)$", d.name)
        Hd, seed = int(m.group(1)), int(m.group(2))
        sd = make_model("mlp", P, seed, d_hidden=Hd).state_dict()
        rows.append({"set": "1C width sweep", "H": Hd, "seed": seed, "variant": "baseline", **s0(sd),
                     **dict(zip(("t95", "final_test"), t95(d)))})
    for d in sorted(Path("runs/exp01/phase1d").glob("s*_*")):
        m = re.match(r"s(\d+)_(.+)$", d.name)
        if not m or not (d / "history.json").exists():
            continue
        sd = torch.load(Path("runs/exp01/phase1d/inits") / f"{d.name}.pt")
        rows.append({"set": "1D edits", "H": H, "seed": int(m.group(1)), "variant": m.group(2), **s0(sd),
                     **dict(zip(("t95", "final_test"), t95(d)))})
    for d in sorted(ROOT.glob("s*_*")):
        m = re.match(r"s(\d+)_(.+)$", d.name)
        if not m or not (d / "history.json").exists():
            continue
        sd = torch.load(ROOT / "inits" / f"{d.name}.pt")
        rows.append({"set": "E6", "H": H, "seed": int(m.group(1)), "variant": m.group(2), **s0(sd),
                     **dict(zip(("t95", "final_test"), t95(d)))})

    by = defaultdict(list)
    for r in rows:
        if r["set"] != "1C width sweep":
            by[r["variant"]].append(r)
    base = [r for r in rows if r["set"] == "1C width sweep" and r["H"] == H and r["seed"] in (0, 1, 2)]
    by["baseline (1C)"] = base
    summary = {}
    print(f"{'variant':16s} {'n':>2} {'t95 per seed':>24} {'S0 mean':>9} {'mean cosD':>9}")
    for v in sorted(by, key=lambda v: (v.split("_b")[0], float(v.split("_b")[1]) if "_b" in v else 0)):
        rs = sorted(by[v], key=lambda r: r["seed"])
        summary[v] = {"t95": [r["t95"] for r in rs], "S0": [r["S0"] for r in rs],
                      "mean_cosD": [r["mean_cosD"] for r in rs], "final_test": [r["final_test"] for r in rs]}
        print(f"{v:16s} {len(rs):2d} {str([r['t95'] for r in rs]):>24} {np.mean(summary[v]['S0']):9.4f} "
              f"{np.mean(summary[v]['mean_cosD']):9.3f}")

    # the collapse: log t95 against log S0 over runs that reached 95%
    fit = {}
    for name, sel in (("H=128 edits (1D + E6 + baseline)", lambda r: r["H"] == H),
                      ("1C width sweep", lambda r: r["set"] == "1C width sweep"),
                      ("all", lambda r: True)):
        xs = [(np.log10(r["S0"]), np.log10(r["t95"])) for r in rows if sel(r) and r["t95"]]
        x, y = np.array(xs).T
        b, a = np.polyfit(x, y, 1)
        fit[name] = {"n": len(x), "slope": float(b), "intercept": float(a),
                     "r2": float(np.corrcoef(x, y)[0, 1] ** 2)}
    print(json.dumps(fit, indent=1))
    (FIG / "phase1e_e6_results.json").write_text(json.dumps({"summary": summary, "fit": fit, "rows": rows},
                                                             indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.6))
    groups = [("1C width sweep", None, "#8a8984", "o", "random init, H = 64…512 (1C)"),
              ("1D edits", None, "#eb6834", "s", "1D edits, H = 128"),
              ("E6", "dose", "#2a78d6", "o", "E6 boosted + aligned"),
              ("E6", "conc", "#1baf7a", "^", "E6 boosted, random phase"),
              ("E6", "align", "#1d1d1b", "D", "E6 aligned only")]
    for set_, pre, col, mk, lab in groups:
        pts = [r for r in rows if r["set"] == set_ and (pre is None or r["variant"].startswith(pre)) and r["t95"]]
        if set_ == "1C width sweep":
            pts = [r for r in pts if r["H"] >= 64]
        ax.scatter([r["S0"] for r in pts], [r["t95"] for r in pts], s=26, color=col, marker=mk, label=lab,
                   edgecolor="white", linewidth=0.6, zorder=3)
    ax.set(xscale="log", yscale="log", xlabel="initial coherent Fourier signal S₀ (raw weights)",
           ylabel="steps to 95% test accuracy")
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "phase1e_e6_s0_collapse.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    sub.add_parser("analyze")
    a = ap.parse_args()
    {"make": cmd_make, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
