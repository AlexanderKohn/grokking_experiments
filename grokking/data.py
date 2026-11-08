"""Algorithmic datasets for grokking experiments.

Each example is the token sequence [a, op, b, =] with label c, where c = a <op> b (mod p).
Tokens 0..p-1 are numbers; p is the op token and p+1 is the '=' token.
"""
import torch

OPS = {
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
    "mul": lambda a, b, p: (a * b) % p,
    # division: a / b = a * b^{-1}, only defined for b != 0
    "div": lambda a, b, p: (a * pow(int(b), p - 2, p)) % p,
}


def make_dataset(p: int, op: str, train_frac: float, seed: int = 0):
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; choose from {list(OPS)}")
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    if op == "div":
        keep = b != 0
        a, b = a[keep], b[keep]
    if op == "div":
        c = torch.tensor([OPS[op](int(x), int(y), p) for x, y in zip(a, b)])
    else:
        c = OPS[op](a, b, p)

    x = torch.stack([a, torch.full_like(a, p), b, torch.full_like(a, p + 1)], dim=1)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    n_train = int(train_frac * len(x))
    tr, te = perm[:n_train], perm[n_train:]
    return (x[tr], c[tr]), (x[te], c[te])
