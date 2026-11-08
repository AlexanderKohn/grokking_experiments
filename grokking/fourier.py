"""Per-neuron Fourier analysis of the one-hot (or tied) MLP.

Conventions (see experiments/README.md):
  Wa, Wb : (..., H, p)  input weights from a and b, one row per neuron
  Wout   : (..., H, p)  output weights; row m is neuron m's output vector W2[:, m]
A component alpha*cos(w_k j + phi), with w_k = 2*pi*k/p, has DFT bin X_k = (p*alpha/2) e^{i phi}.
Frequencies are reported as k = 1..K with K = (p - 1) // 2, and k = 0 means "none".
All functions accept leading batch dimensions, e.g. (seeds, H, p).
"""
import math

import torch

TWO_PI = 2 * math.pi


def split_weights(state_dict, p):
    """Return (Wa, Wb, Wout), each (H, p), from an MLP state_dict (onehot or tied)."""
    W1 = state_dict["net.0.weight"].detach().float()
    W2 = state_dict["net.2.weight"].detach().float()
    if W1.shape[-1] == 2 * p:
        Wa, Wb = W1[..., :p], W1[..., p:]
    elif W1.shape[-1] == p:
        Wa = Wb = W1
    else:
        raise ValueError(f"unexpected W1 shape {tuple(W1.shape)} for p={p}")
    return Wa, Wb, W2.transpose(-1, -2)


def fourier_coeffs(W):
    """Magnitude and phase per frequency k = 1..K along the last dim. Returns (alpha, phi), each (..., K)."""
    p = W.shape[-1]
    K = (p - 1) // 2
    X = torch.fft.fft(W.double(), dim=-1)[..., 1:K + 1]
    return (2 * X.abs() / p).float(), X.angle().float()


def _frob(W):
    return W.pow(2).sum(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-30)


def neuron_fourier(Wa, Wb, Wout, sign=-1, tied=None):
    """Per-neuron Fourier statistics.

    sign selects the phase combination D = phi_a + phi_b + sign*psi (-1 by default; see
    experiments/README.md for the empirical sign check).
    Returns a dict of tensors: alpha_a, phi_a, alpha_b, phi_b, beta, psi, P, D, Dt, cosD
    with shape (..., H, K), and kstar (1..K), ipr with shape (..., H).
    """
    if tied is None:
        tied = Wa is Wb
    # Normalize each layer to unit Frobenius norm so that no single layer dominates P.
    n1 = _frob(Wa) if tied else (Wa.pow(2).sum((-1, -2), keepdim=True)
                                  + Wb.pow(2).sum((-1, -2), keepdim=True)).sqrt().clamp_min(1e-30)
    aa, pa = fourier_coeffs(Wa / n1)
    ab, pb = (aa, pa) if tied else fourier_coeffs(Wb / n1)
    be, ps = fourier_coeffs(Wout / _frob(Wout))
    P = aa**2 + ab**2 + be**2
    D = torch.remainder(pa + pb + sign * ps, TWO_PI)
    return {
        "alpha_a": aa, "phi_a": pa, "alpha_b": ab, "phi_b": pb, "beta": be, "psi": ps,
        "P": P, "D": D, "Dt": torch.minimum(D, TWO_PI - D), "cosD": torch.cos(D),
        "kstar": P.argmax(-1) + 1,
        "ipr": P.pow(2).sum(-1) / P.sum(-1).pow(2).clamp_min(1e-30),
    }


def winner_predictions(F):
    """Init-time winner predictors (Exp 01, Phase 1A). Each is (..., H) with values in 1..K (0 = no winner)."""
    growth = F["alpha_a"] * F["alpha_b"] * F["beta"] * F["cosD"]
    g_max, g_arg = growth.max(-1)
    return {
        "pred_power": F["kstar"],
        "pred_growth": g_arg + 1,
        "pred_growth_pos": torch.where(g_max > 0, g_arg + 1, torch.zeros_like(g_arg)),
    }


def neuron_contribution(Wa, Wb, Wout):
    """||W1[m]|| * ||W2[:, m]|| per neuron, shape (..., H)."""
    w1 = Wa.pow(2).sum(-1) if Wa is Wb else Wa.pow(2).sum(-1) + Wb.pow(2).sum(-1)
    return w1.sqrt() * Wout.pow(2).sum(-1).sqrt()


def active_neurons(Wa, Wb, Wout, frac=0.95):
    """Bool mask (H,) for the smallest set of neurons that accounts for `frac` of total contribution."""
    c = neuron_contribution(Wa, Wb, Wout)
    order = c.argsort(descending=True)
    cum = c[order].cumsum(0) / c.sum()
    n = int((cum < frac).sum().item()) + 1
    mask = torch.zeros_like(c, dtype=torch.bool)
    mask[order[:n]] = True
    return mask


def key_frequencies(kstar, active, min_frac=0.02):
    """{k: count} for frequencies that are k* of at least min_frac of the active neurons."""
    ks = kstar[active]
    counts = torch.bincount(ks, minlength=int(ks.max().item()) + 1 if len(ks) else 1)
    thresh = min_frac * len(ks)
    return {int(k): int(c) for k, c in enumerate(counts) if k > 0 and c >= thresh and c > 0}


def circular_resultant(angles, dim=None):
    """Mean resultant length |mean(e^{i theta})|: 1 = fully concentrated, ~0 = uniform."""
    z = torch.polar(torch.ones_like(angles), angles)
    return (z.mean() if dim is None else z.mean(dim)).abs()


def project_to_frequencies(W, S, keep_dc=False, dim=-1):
    """Keep only DFT bins k in S (and their conjugates p-k), plus bin 0 if keep_dc. Returns a real tensor."""
    p = W.shape[dim]
    keep = torch.zeros(p, dtype=torch.bool)
    for k in S:
        keep[k % p] = True
        keep[(-k) % p] = True
    if keep_dc:
        keep[0] = True
    shape = [1] * W.dim()
    shape[dim] = p
    X = torch.fft.fft(W.double(), dim=dim) * keep.view(shape)
    return torch.fft.ifft(X, dim=dim).real.to(W.dtype)


def full_analysis(state_dict, p, sign=-1, active_frac=0.95, key_min_frac=0.02):
    """Convenience wrapper: per-neuron stats plus active set and key frequencies for one network."""
    Wa, Wb, Wout = split_weights(state_dict, p)
    F = neuron_fourier(Wa, Wb, Wout, sign=sign)
    act = active_neurons(Wa, Wb, Wout, frac=active_frac)
    F["active"] = act
    F["key_freqs"] = key_frequencies(F["kstar"], act, min_frac=key_min_frac)
    return F
