"""Plot accuracy and loss curves for one or more runs.

Example:
    python plot.py runs/transformer_add_p97_f0.3_wd1.0_s0 -o grok.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--output", default="curves.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(len(args.runs), 2, figsize=(11, 3.4 * len(args.runs)), squeeze=False)
    for row, run in zip(axes, args.runs):
        data = json.loads((Path(run) / "history.json").read_text())
        h, s = data["history"], data["summary"]
        steps = [max(x, 1) for x in h["step"]]
        title = Path(run).name
        row[0].plot(steps, h["train_acc"], label="train", color="#2a6fdb")
        row[0].plot(steps, h["test_acc"], label="test", color="#e8590c")
        row[0].set(xscale="log", ylabel="accuracy", ylim=(-0.02, 1.02), title=f"{title} — accuracy")
        row[1].plot(steps, h["train_loss"], label="train", color="#2a6fdb")
        row[1].plot(steps, h["test_loss"], label="test", color="#e8590c")
        row[1].set(xscale="log", yscale="log", ylabel="loss", title=f"{title} — loss")
        for ax in row:
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False)
            for key, ls in (("train_99_step", ":"), ("test_99_step", "--")):
                if s.get(key):
                    ax.axvline(s[key], color="gray", ls=ls, lw=0.8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
