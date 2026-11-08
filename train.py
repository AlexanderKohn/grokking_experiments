"""Train a small model on modular arithmetic and log train/test curves.

Example:
    python train.py --op add --p 97 --train-frac 0.3 --steps 10000
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from grokking.data import make_dataset
from grokking.model import build_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--op", default="add", choices=["add", "sub", "mul", "div"])
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--model", default="transformer", choices=["transformer", "mlp"])
    ap.add_argument("--mlp-input", default="onehot", choices=["onehot", "scalar", "embed"])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = full batch")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    ap.add_argument("--out", default=None, help="output dir (default runs/<auto-name>)")
    ap.add_argument("--stop-on-grok", action="store_true",
                    help="stop once test acc >= 99%% (saves time on smoke tests)")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits, y).item(), (logits.argmax(-1) == y).float().mean().item()


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    (xtr, ytr), (xte, yte) = make_dataset(args.p, args.op, args.train_frac, args.seed)
    model = build_model(args.model, args.p, mlp_input=args.mlp_input)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))

    name = args.out or f"runs/{args.model}{'-' + args.mlp_input if args.model == 'mlp' else ''}_{args.op}_p{args.p}_f{args.train_frac}_wd{args.wd}_s{args.seed}"
    out = Path(name)
    out.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model} ({n_params:,} params) | {args.op} mod {args.p} | "
          f"train {len(xtr)} / test {len(xte)} | threads {torch.get_num_threads()}")

    hist = {"step": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    t_fit = t_grok = None
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            model.eval()
            trl, tra = evaluate(model, xtr, ytr)
            tel, tea = evaluate(model, xte, yte)
            model.train()
            for k, v in zip(hist, (step, trl, tra, tel, tea)):
                hist[k].append(v)
            if t_fit is None and tra >= 0.99:
                t_fit = step
            if t_grok is None and tea >= 0.99:
                t_grok = step
            if step % (args.eval_every * 20) == 0:
                print(f"step {step:6d} | train loss {trl:.4f} acc {tra:.3f} | "
                      f"test loss {tel:.4f} acc {tea:.3f} | {time.time() - t0:.0f}s", flush=True)
            if args.stop_on_grok and t_grok is not None:
                break
        if step == args.steps:
            break

        if args.batch_size:
            idx = torch.randint(len(xtr), (args.batch_size,))
            xb, yb = xtr[idx], ytr[idx]
        else:
            xb, yb = xtr, ytr
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

    elapsed = time.time() - t0
    summary = {"args": vars(args), "n_params": n_params, "train_99_step": t_fit,
               "test_99_step": t_grok, "final_test_acc": hist["test_acc"][-1],
               "seconds": round(elapsed, 1)}
    (out / "history.json").write_text(json.dumps({"summary": summary, "history": hist}))
    print(f"done in {elapsed:.0f}s | train>=99% at step {t_fit} | test>=99% at step {t_grok}")
    print(f"saved {out / 'history.json'}")


if __name__ == "__main__":
    main()
