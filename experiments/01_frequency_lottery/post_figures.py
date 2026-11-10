"""Figures for the blog post, redrawn from the saved results with plain-language labels.

Reads figures/phase1{b,c,d}_results.json and figures/phase1e_{e1,e2,e5,cf}_results.json plus the run directories
under runs/exp01 (for the lock-in curve and the intervention learning curves) and writes each of the seven figures to --out
twice: NAME.png with the panels side by side, and NAME_narrow.png with them stacked for phone
screens. Nothing is retrained; the numbers are the ones in RESULTS.md.

Usage: python experiments/01_frequency_lottery/post_figures.py --out ../AlexanderKohn.github.io/assets/img/grokking
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, RULE, GRID, REF = "#1f1f1e", "#66655f", "#bdbcb6", "#ebeae6", "#8a8983"
SCALES = ["0.1", "0.3", "1.0"]
SCALE_COLOR = dict(zip(SCALES, [BLUE, ORANGE, AQUA]))
WIDE = 8.8  # inches; shown about 880 CSS px wide, so 9 pt text renders near 12.5 px
PHONE = 4.4  # inches; width of the stacked version
NARROW = False  # set by main() while the stacked versions are drawn

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 9.5, "axes.titlelocation": "left", "axes.titlepad": 9, "axes.titlecolor": INK,
    "axes.labelsize": 9, "axes.labelcolor": MUTED, "axes.edgecolor": RULE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.major.size": 3, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.handlelength": 1.6, "legend.labelcolor": INK,
    "lines.linewidth": 1.8, "lines.solid_capstyle": "round",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 100, "savefig.dpi": 200,
})

pct = PercentFormatter(1.0, decimals=0)


def ksteps(v, _=None):
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


def make_panels(n, height, sharey=False):
    """n panels in a row, or in a column for the phone version. Returns (fig, axes, axes that get a y label)."""
    if NARROW:
        fig, axes = plt.subplots(n, 1, figsize=(PHONE, height * n * 0.92), sharey=sharey)
        return fig, axes, list(axes)
    fig, axes = plt.subplots(1, n, figsize=(WIDE, height), sharey=sharey)
    return fig, axes, [axes[0]]


def save(fig, out, name):
    if NARROW:
        name = name.replace(".png", "_narrow.png")
    fig.savefig(out / name, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", out / name)


def fig_predictor_accuracy(out):
    r = json.loads((HERE / "figures/phase1b_results.json").read_text())
    fig, axes, labelled = make_panels(2, 3.1, sharey=True)
    series = [("pred_power", "largest starting magnitude", BLUE),
              ("pred_growth", "magnitude × phase alignment (the paper's score)", ORANGE)]
    w = 0.3
    for ax, (reg, title) in zip(axes, [("full", "Trained on all pairs"), ("grok", "Trained on half (grokking split)")]):
        for j, (key, label, col) in enumerate(series):
            for i, s in enumerate(SCALES):
                c = r["conditions"][f"{reg}_is{s}"][key]
                x = i + (j - 0.5) * (w + 0.03)
                ax.bar(x, c["acc"], w, color=col, label=label if i == 0 else None)
                ax.plot([x, x], c["ci95"], color=INK, lw=0.9, solid_capstyle="butt")
        ax.axhline(r["chance"], color=REF, lw=1, ls=(0, (4, 3)), label="chance, 1 in 23")
        ax.set_title(title)
        ax.set_xticks(range(3), [f"{s}×" for s in SCALES])
        ax.set_xlabel("initialization scale")
        ax.set_xlim(-0.6, 2.6)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylim(0, 0.6)
    axes[0].yaxis.set_major_formatter(pct)
    for ax in labelled:
        ax.set_ylabel("neurons predicted correctly")
    axes[1].legend(loc="upper right", borderaxespad=0)
    fig.tight_layout(w_pad=2.5, h_pad=1.8)
    save(fig, out, "predictor_accuracy.png")


def fig_lock_in(out, runs):
    from phase1b_analysis import analyze_run
    fig, ax = plt.subplots(figsize=(PHONE if NARROW else 6.6, 3.3))
    t_fit, t_grok = [], []
    for s in SCALES:
        rs = [analyze_run(runs / "phase1b" / f"grok_is{s}_s{seed}", 47, -1) for seed in range(3)]
        lock = np.sort(np.concatenate([r["lock_step"] for r in rs]))
        ax.step(np.maximum(lock, 1), np.arange(1, len(lock) + 1) / len(lock), where="post",
                color=SCALE_COLOR[s], lw=1.6, label=f"init scale {s}×")
        t_fit += [r["t_fit"] for r in rs]
        t_grok += [r["t_grok"] for r in rs]
    for t, label in [(np.median(t_fit), "training set fit"), (np.median(t_grok), "generalizes")]:
        ax.axvline(t, color=REF, lw=1, ls=(0, (4, 3)))
        ax.text(t, 1.045, label, ha="center", va="bottom", color=MUTED, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 3e4)
    ax.set_ylim(0, 1.03)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 1 else ksteps(v)))
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("neurons settled on their final frequency")
    ax.legend(loc="upper left", borderaxespad=0.6)
    fig.tight_layout()
    save(fig, out, "lock_in.png")
    print("  median t_fit", np.median(t_fit), "median t_grok", np.median(t_grok))


def fig_predictor_over_time(out):
    r = json.loads((HERE / "figures/phase1e_e1_results.json").read_text())
    steps = r["steps"]
    fig, axes, labelled = make_panels(2, 3.1, sharey=True)
    for ax, (reg, title) in zip(axes, [("full", "Trained on all pairs"), ("grok", "Trained on half (grokking split)")]):
        for s in SCALES:
            ax.plot(steps, r["conditions"][f"{reg}_is{s}"]["pred_growth"], color=SCALE_COLOR[s],
                    marker="o", ms=4.5, mec="white", mew=1, label=f"init scale {s}×")
        ax.axhline(r["chance"], color=REF, lw=1, ls=(0, (4, 3)))
        ax.set_xscale("symlog", linthresh=10, linscale=0.6)
        ax.set_xticks(steps, [str(s) for s in steps])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(-1.5, 230)
        ax.set_title(title)
        ax.set_xlabel("step whose weights the prediction is made from")
    axes[0].set_ylim(0, 1.02)
    axes[0].yaxis.set_major_formatter(pct)
    for ax in labelled:
        ax.set_ylabel("neurons predicted correctly")
    axes[1].text(225, r["chance"] + 0.02, "chance", ha="right", va="bottom", color=MUTED, fontsize=8.5)
    axes[1].legend(loc="upper left", borderaxespad=0.2)
    fig.tight_layout(w_pad=2.5, h_pad=1.8)
    save(fig, out, "predictor_over_time.png")


def fig_width_sweep(out):
    r = json.loads((HERE / "figures/phase1c_results.json").read_text())
    widths = sorted(int(h) for h in r["per_width"])
    pos = {h: i for i, h in enumerate(widths)}
    rng = np.random.default_rng(0)
    jit = {(run["H"], run["seed"]): rng.uniform(-0.17, 0.17) for run in r["runs"]}
    xs = np.array([pos[run["H"]] + jit[run["H"], run["seed"]] for run in r["runs"]])
    fig, axes, _ = make_panels(3, 3.0)
    dot = dict(s=15, color=BLUE, alpha=0.75, lw=0, zorder=3)

    ax = axes[0]
    ax.scatter(xs, [run["final_test_acc"] for run in r["runs"]], **dot)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("Test accuracy after 50k steps")

    ax = axes[1]
    ok = np.array([run["t_grok_95"] is not None for run in r["runs"]])
    ax.scatter(xs[ok], [run["t_grok_95"] for run in r["runs"] if run["t_grok_95"] is not None], **dot)
    ax.set_yscale("log")
    ax.set_ylim(3000, 50000)
    ax.set_yticks([5000, 10000, 20000, 50000])
    ax.yaxis.set_major_formatter(FuncFormatter(ksteps))
    ax.yaxis.set_minor_formatter(NullFormatter())
    last_fail = max(pos[run["H"]] for run in r["runs"] if run["t_grok_95"] is None)
    ax.axvspan(-0.5, last_fail + 0.5, color="#f3f2ee", lw=0, zorder=0)
    ax.text((last_fail) / 2, 12000, "never reaches 95%", ha="center", va="center", color=MUTED, fontsize=8.5)
    ax.set_title("Steps to 95% test accuracy")

    ax = axes[2]
    K = r["K"]
    ax.axhline(K, color=REF, lw=1, ls=(0, (4, 3)))
    ax.text(-0.4, K + 0.5, f"all {K}", ha="left", va="bottom", color=MUTED, fontsize=8.5)
    ax.plot([pos[h] for h in widths], [r["per_width"][str(h)]["coupon_expected"] for h in widths],
            color=REF, lw=1.4, label="expected from a random draw", zorder=2)
    ax.scatter(xs, [run["growth_distinct"] for run in r["runs"]], label="each seed", **dot)
    ax.set_ylim(0, K + 3.5)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.legend(loc="lower right", borderaxespad=0.5, handletextpad=0.5)
    ax.set_title("Frequencies covered at initialization")

    for ax in axes:
        ax.set_xticks(range(len(widths)), [str(h) for h in widths])
        ax.set_xlim(-0.5, len(widths) - 0.5)
        ax.set_xlabel("hidden width")
        ax.tick_params(axis="x", length=0)
    fig.tight_layout(w_pad=2.0, h_pad=1.8)
    save(fig, out, "width_sweep.png")


def fig_interventions(out, runs):
    panels = [
        ("Frequencies removed", [("knockout", "the network's top 5 removed", BLUE),
                                 ("knockout_ctrl0", "5 other frequencies removed", ORANGE),
                                 ("knockout_ctrl1", None, ORANGE)]),
        ("All neurons on one frequency", [("mono_k1", "one frequency for all neurons", BLUE),
                                           ("mono_k8", None, BLUE), ("mono_k17", None, BLUE)]),
        ("Rigged initialization", [("rigged", "frequencies spread evenly", BLUE),
                                                        ("rigged_random", "frequencies assigned at random", ORANGE)]),
    ]
    fig, axes, labelled = make_panels(3, 3.75, sharey=True)

    def curve(seed, variant):
        h = json.loads((runs / "phase1d" / f"s{seed}_{variant}" / "history.json").read_text())["history"]
        return np.array(h["step"]), np.array(h["test_acc"])

    for ax, (title, variants) in zip(axes, panels):
        for seed in range(3):
            x, y = curve(seed, "baseline")
            ax.plot(x[1:], y[1:], color="#55544f", lw=1.2, label="unedited" if seed == 0 else None, zorder=3)
        for variant, label, col in variants:
            for seed in range(3):
                x, y = curve(seed, variant)
                ax.plot(x[1:], y[1:], color=col, lw=1.2, alpha=0.9, label=label if seed == 0 else None, zorder=4)
        ax.axhline(0.95, color=REF, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(50, 50000)
        ax.set_xticks([100, 1000, 10000], ["100", "1k", "10k"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title)
        ax.set_xlabel("training step (log scale)")
        ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.26), borderaxespad=0, handlelength=1.2)
    axes[0].set_ylim(0, 1.02)
    axes[0].yaxis.set_major_formatter(pct)
    for ax in labelled:
        ax.set_ylabel("test accuracy")
    fig.tight_layout(w_pad=1.6, h_pad=1.8)
    save(fig, out, "interventions.png")


def fig_same_init(out):
    """Same initial weights, one thing changed: how often does a neuron end on the same frequency?"""
    e2 = json.loads((HERE / "figures/phase1e_e2_results.json").read_text())["same_init_agreement"]
    cf = json.loads((HERE / "figures/phase1e_cf_results.json").read_text())["H512_is1.0"]
    groups = [
        ("Trained on all pairs", [
            ("warmup of 1 step vs 1,000 steps", e2["all pairs, init 0.1: warmup 1 vs 1000"]["agree"]),
            ("Adam vs plain gradient descent", e2["all pairs, init 1.0: Adam vs GD"]["agree"])]),
        ("Trained on half (grokking split)", [
            ("warmup of 1 step vs 1,000 steps", e2["grokking split, init 1.0: warmup 1 vs 1000"]["agree"]),
            ("Adam vs plain gradient descent", e2["grokking split, init 1.0: Adam vs GD"]["agree"]),
            ("a different half of the pairs", cf["same init, different split"]["agree"])]),
    ]
    chance = cf["same init, different split"]["chance"]
    fig, ax = plt.subplots(figsize=(PHONE if NARROW else 6.6, 3.4 if NARROW else 2.9))
    y, ticks, labels = 0, [], []
    for gi, (title, rows) in enumerate(groups):
        ax.text(0, y - 0.62, title, ha="left", va="center", color=INK, fontsize=9, fontweight="bold")
        for label, v in rows:
            ax.barh(y, v, 0.62, color=BLUE if gi == 0 else ORANGE)
            ax.text(v + 0.012, y, f"{v:.0%}", va="center", ha="left", color=INK, fontsize=8.5)
            ticks.append(y)
            labels.append(label)
            y += 1
        y += 0.9
    ax.axvline(chance, color=REF, lw=1, ls=(0, (4, 3)))
    ax.text(chance + 0.01, y - 0.95, "chance", color=MUTED, fontsize=8.5, va="center")
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.6, -1.3)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1] if NARROW else np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(pct)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("neurons that end on the same frequency")
    fig.tight_layout()
    save(fig, out, "same_init.png")


def fig_lr_width(out):
    """Steps to 95% test accuracy against learning rate, per width, in both weight-decay arms."""
    cells = json.loads((HERE / "figures/phase1e_e5_results.json").read_text())["cells"]
    colors = {128: BLUE, 256: ORANGE, 512: AQUA}
    arms = [("Weight decay fixed at 1", lambda c: c["wd"] == 1.0),
            ("Decay per step (lr × wd) fixed", lambda c: abs(c["lr_wd"] - 1e-3) < 1e-9)]
    fig, axes, labelled = make_panels(2, 3.1, sharey=True)
    rng = np.random.default_rng(0)
    for ax, (title, keep) in zip(axes, arms):
        for H, col in colors.items():
            cs = sorted((c for c in cells if c["H"] == H and keep(c) and c["n95"] == c["n"]), key=lambda c: c["lr"])
            ax.plot([c["lr"] for c in cs], [c["t95_median"] for c in cs], color=col, lw=1.4, label=f"width {H}",
                    zorder=3)
            for c in cs:
                ax.scatter(c["lr"] * np.exp(rng.normal(0, 0.04, len(c["t95"]))), c["t95"], s=10, color=col,
                           alpha=0.55, lw=0, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([2.5e-4, 1e-3, 4e-3, 1.6e-2], ["0.00025", "0.001", "0.004", "0.016"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title)
        ax.set_xlabel("learning rate")
    axes[0].set_ylim(700, 40000)
    axes[0].set_yticks([1000, 3000, 10000, 30000])
    axes[0].yaxis.set_major_formatter(FuncFormatter(ksteps))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    for ax in labelled:
        ax.set_ylabel("steps to 95% test accuracy")
    axes[1].legend(loc="lower left", borderaxespad=0.3)
    fig.tight_layout(w_pad=2.5, h_pad=1.8)
    save(fig, out, "lr_width.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs/exp01"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out, runs = Path(args.out), Path(args.runs)
    out.mkdir(parents=True, exist_ok=True)
    global NARROW
    for NARROW in (False, True):
        fig_predictor_accuracy(out)
        fig_predictor_over_time(out)
        fig_width_sweep(out)
        fig_interventions(out, runs)
        fig_lock_in(out, runs)
        fig_same_init(out)
        fig_lr_width(out)


if __name__ == "__main__":
    main()
