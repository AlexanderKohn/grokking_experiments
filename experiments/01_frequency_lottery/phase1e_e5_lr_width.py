"""Exp 01, Phase 1E / E5: is the width effect above threshold just a larger effective learning rate?

Reads the E5 runs (runs/exp01/phase1e/e5) and the 1C runs at lr 1e-3, wd 1 (seeds 0-4), and reports steps to 95%
test accuracy (t95) per (H, lr, wd) cell. Two arms: (a) wd = 1, so the decay per step lr*wd moves with lr;
(b) lr*wd = 1e-3 held fixed. A log-linear fit over the runs that reach 95% at H >= 128,
    log t95 = c + a*log(lr) + h*log(H) + d*log(lr*wd),
says whether width acts only through lr*H (h = a) and how much the decay rate matters on its own.

Usage: python experiments/01_frequency_lottery/phase1e_e5_lr_width.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

E5 = Path("runs/exp01/phase1e/e5")
C1 = Path("runs/exp01/phase1c")
FIG = Path("experiments/01_frequency_lottery/figures")
WIDTH_COLOR = {64: "#8a8984", 128: "#2a78d6", 256: "#eb6834", 512: "#1baf7a", 1024: "#1d1d1b"}


def load():
    rows = []
    for d in E5.iterdir():
        m = re.match(r"H(\d+)_lr([\d.e-]+)_wd([\d.e-]+)_s(\d+)$", d.name)
        if m and (d / "history.json").exists():
            s = json.loads((d / "history.json").read_text())["summary"]
            rows.append({"H": int(m.group(1)), "lr": float(m.group(2)), "wd": float(m.group(3)),
                         "seed": int(m.group(4)), "t95": s["test_95_step"], "final_test": s["final_test_acc"],
                         "t_fit": s["train_99_step"], "src": "E5"})
    for d in C1.iterdir():
        m = re.match(r"H(\d+)_is1.0_s(\d+)$", d.name)
        if m and int(m.group(2)) < 5 and int(m.group(1)) >= 64:
            s = json.loads((d / "history.json").read_text())["summary"]
            rows.append({"H": int(m.group(1)), "lr": 1e-3, "wd": 1.0, "seed": int(m.group(2)),
                         "t95": s["test_95_step"], "final_test": s["final_test_acc"], "t_fit": s["train_99_step"],
                         "src": "1C"})
    return rows


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2


def main():
    rows = load()
    cells = defaultdict(list)
    for r in rows:
        cells[(r["H"], r["lr"], round(r["lr"] * r["wd"], 9))].append(r)

    table = []
    print(f"{'H':>5} {'lr':>8} {'wd':>7} {'lr*H':>6} {'lr*wd':>8} {'n95':>5} {'t95 median':>10} {'t95 per seed':>40} "
          f"{'final test':>18}")
    for (H, lr, decay), rs in sorted(cells.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        t = [r["t95"] for r in rs]
        ok = [x for x in t if x is not None]
        med = float(np.median([x if x is not None else np.inf for x in t])) if t else None
        ft = [r["final_test"] for r in rs]
        table.append({"H": H, "lr": lr, "wd": rs[0]["wd"], "lr_H": lr * H, "lr_wd": decay, "n": len(rs),
                      "n95": len(ok), "t95": t, "t95_median": med, "final_test": ft})
        print(f"{H:5d} {lr:8.2e} {rs[0]['wd']:7.3g} {lr * H:6.3f} {decay:8.1e} {len(ok):2d}/{len(rs):<2d} "
              f"{med:10.0f} {str(t):>40} {min(ft):.3f}-{max(ft):.3f}")

    # fit over runs that reach 95%, H >= 128
    fits = {}
    sel = [r for r in rows if r["t95"] and r["H"] >= 128]
    y = np.log10([r["t95"] for r in sel])
    lr, H, dec = (np.log10([f(r) for r in sel]) for f in (lambda r: r["lr"], lambda r: r["H"],
                                                           lambda r: r["lr"] * r["wd"]))
    for name, X, labels in (("lr, H", np.column_stack([lr, H]), ["lr", "H"]),
                            ("lr, H, lr*wd", np.column_stack([lr, H, dec]), ["lr", "H", "lr*wd"]),
                            ("lr*H only", (lr + H)[:, None], ["lr*H"])):
        beta, se, r2 = ols(X, y)
        fits[name] = {"n": len(y), "r2": float(r2), "coef": {l: [float(b), float(s)] for l, b, s in
                                                             zip(labels, beta[1:], se[1:])}}
        print(f"fit [{name}] n={len(y)} R2={r2:.3f} " +
              " ".join(f"{l}: {b:+.2f}±{s:.2f}" for l, b, s in zip(labels, beta[1:], se[1:])))
    # the same fits restricted to each arm
    for arm, f in (("wd = 1", lambda r: r["wd"] == 1.0), ("lr*wd = 1e-3", lambda r: abs(r["lr"] * r["wd"] - 1e-3) < 1e-9)):
        s2 = [r for r in sel if f(r)]
        y2 = np.log10([r["t95"] for r in s2])
        X2 = np.column_stack([np.log10([r["lr"] for r in s2]), np.log10([r["H"] for r in s2])])
        beta, se, r2 = ols(X2, y2)
        fits[f"arm {arm}: lr, H"] = {"n": len(y2), "r2": float(r2),
                                     "coef": {"lr": [float(beta[1]), float(se[1])], "H": [float(beta[2]), float(se[2])]}}
        print(f"fit [arm {arm}: lr, H] n={len(y2)} R2={r2:.3f} lr: {beta[1]:+.2f}±{se[1]:.2f} H: {beta[2]:+.2f}±{se[2]:.2f}")
    (FIG / "phase1e_e5_results.json").write_text(json.dumps({"cells": table, "fits": fits}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    arms = (("wd = 1 (decay per step moves with lr)", lambda r: r["wd"] == 1.0),
            ("lr × wd = 1e-3 (decay per step fixed)", lambda r: abs(r["lr"] * r["wd"] - 1e-3) < 1e-9))
    rng = np.random.default_rng(0)
    for ax, (title, f) in zip(axes, arms):
        for H, col in WIDTH_COLOR.items():
            rs = [r for r in rows if r["H"] == H and f(r)]
            if not rs:
                continue
            x = np.array([r["lr"] * H for r in rs]) * np.exp(rng.normal(0, 0.03, len(rs)))
            y = np.array([r["t95"] if r["t95"] else 60000 for r in rs])
            ax.scatter(x, y, s=22, color=col, label=f"H = {H}", edgecolor="white", linewidth=0.5, zorder=3)
        ax.axhline(60000, color="#bbb", lw=0.8, ls=":")
        ax.text(0.03, 60000, " never reached 95% (50k cap)", fontsize=7, color="#777", va="bottom")
        ax.set(xscale="log", yscale="log", xlabel="lr × H", title=title)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("steps to 95% test accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "phase1e_e5_lr_width.png", dpi=130)


if __name__ == "__main__":
    main()
