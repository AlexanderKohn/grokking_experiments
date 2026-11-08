"""Train a small model on modular arithmetic and log train/test curves.

Example:
    python train.py --op add --p 97 --train-frac 0.3 --steps 10000
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from grokking.data import make_dataset
from grokking.model import make_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--op", default="add", choices=["add", "sub", "mul", "div"])
    ap.add_argument("--train-frac", type=float, default=0.3,
                    help="1.0 = train on all pairs (no test set; test metrics are NaN)")
    ap.add_argument("--model", default="transformer", choices=["transformer", "mlp"])
    ap.add_argument("--mlp-input", default="onehot", choices=["onehot", "tied", "scalar", "embed"])
    ap.add_argument("--d-hidden", type=int, default=512, help="MLP hidden width")
    ap.add_argument("--no-bias", action="store_true", help="MLP without biases")
    ap.add_argument("--init-scale", type=float, default=1.0,
                    help="multiply every parameter by this after the default init")
    ap.add_argument("--steps", "--max-steps", dest="steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = full batch")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="seeds both the init and the data split")
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    ap.add_argument("--out", default=None, help="output dir (default runs/<auto-name>)")
    ap.add_argument("--stop-on-grok", action="store_true",
                    help="stop once test acc >= 99%% (plus --post-grok-steps)")
    ap.add_argument("--post-grok-steps", type=int, default=0,
                    help="with --stop-on-grok, keep training this many steps after grokking")
    # checkpointing / interventions
    ap.add_argument("--save-init", action="store_true", help="save step-0 state_dict to out/init.pt")
    ap.add_argument("--ckpt-every", type=int, default=0, help="save out/ckpt_{step}.pt every N steps")
    ap.add_argument("--ckpt-sparse-after", type=int, default=0,
                    help="after this step, checkpoint every --ckpt-sparse-every steps instead")
    ap.add_argument("--ckpt-sparse-every", type=int, default=1000)
    ap.add_argument("--init-from", default=None, help="load initial weights from this state_dict")
    ap.add_argument("--mask", default=None,
                    help="dict {param_name: bool tensor}; masked weights are held at zero")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, x, y):
    if len(x) == 0:
        return math.nan, math.nan
    logits = model(x)
    return F.cross_entropy(logits, y).item(), (logits.argmax(-1) == y).float().mean().item()


def is_ckpt_step(step, args):
    if not args.ckpt_every:
        return False
    if args.ckpt_sparse_after and step > args.ckpt_sparse_after:
        return step % args.ckpt_sparse_every == 0
    return step % args.ckpt_every == 0


def apply_mask(model, mask_path):
    masks = {k: v.bool() for k, v in torch.load(mask_path).items()}
    params = dict(model.named_parameters())
    unknown = set(masks) - set(params)
    if unknown:
        raise ValueError(f"mask names not in model: {sorted(unknown)}")
    with torch.no_grad():
        for name, m in masks.items():
            params[name].mul_(m)
            params[name].register_hook(lambda g, m=m: g * m)

    def remask():
        with torch.no_grad():
            for name, m in masks.items():
                params[name].mul_(m)

    return remask


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    (xtr, ytr), (xte, yte) = make_dataset(args.p, args.op, args.train_frac, args.seed)
    model = make_model(args.model, args.p, args.seed, mlp_input=args.mlp_input,
                       d_hidden=args.d_hidden, bias=not args.no_bias, init_scale=args.init_scale)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from))
    remask = apply_mask(model, args.mask) if args.mask else None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))

    tag = f"{args.model}-{args.mlp_input}" if args.model == "mlp" else args.model
    name = args.out or f"runs/{tag}_{args.op}_p{args.p}_f{args.train_frac}_wd{args.wd}_s{args.seed}"
    out = Path(name)
    out.mkdir(parents=True, exist_ok=True)
    if args.save_init:
        torch.save(model.state_dict(), out / "init.pt")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model} ({n_params:,} params) | {args.op} mod {args.p} | "
          f"train {len(xtr)} / test {len(xte)} | threads {torch.get_num_threads()}")

    hist = {"step": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    marks = {"train_99_step": None, "test_99_step": None, "train_95_step": None, "test_95_step": None}
    stop_at = None
    t0 = time.time()
    for step in range(args.steps + 1):
        if is_ckpt_step(step, args):
            torch.save(model.state_dict(), out / f"ckpt_{step}.pt")
        if step % args.eval_every == 0:
            model.eval()
            trl, tra = evaluate(model, xtr, ytr)
            tel, tea = evaluate(model, xte, yte)
            model.train()
            for k, v in zip(hist, (step, trl, tra, tel, tea)):
                hist[k].append(v)
            for split, acc in (("train", tra), ("test", tea)):
                for thr in (95, 99):
                    key = f"{split}_{thr}_step"
                    if marks[key] is None and acc >= thr / 100:
                        marks[key] = step
            if step % (args.eval_every * 20) == 0:
                print(f"step {step:6d} | train loss {trl:.4f} acc {tra:.3f} | "
                      f"test loss {tel:.4f} acc {tea:.3f} | {time.time() - t0:.0f}s", flush=True)
            if args.stop_on_grok and stop_at is None and marks["test_99_step"] is not None:
                stop_at = step + args.post_grok_steps
        if step == args.steps or (stop_at is not None and step >= stop_at):
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
        if remask:
            remask()

    torch.save(model.state_dict(), out / "final.pt")
    elapsed = time.time() - t0

    def ratio(g, f):
        return g / max(f, 1) if g is not None and f is not None else None

    gr99, gr95 = ratio(marks["test_99_step"], marks["train_99_step"]), \
        ratio(marks["test_95_step"], marks["train_95_step"])
    summary = {"args": vars(args), "n_params": n_params, **marks,
               "grok_ratio": gr99, "log10_grok_ratio": math.log10(gr99) if gr99 else None,
               "grok_ratio_95": gr95, "log10_grok_ratio_95": math.log10(gr95) if gr95 else None,
               "final_step": step, "final_train_acc": hist["train_acc"][-1],
               "final_test_acc": hist["test_acc"][-1], "seconds": round(elapsed, 1)}
    (out / "history.json").write_text(json.dumps({"summary": summary, "history": hist}))
    print(f"done in {elapsed:.0f}s | train>=99% at step {marks['train_99_step']} | "
          f"test>=99% at step {marks['test_99_step']}")
    print(f"saved {out / 'history.json'}")


if __name__ == "__main__":
    main()
