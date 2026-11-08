import math

import torch

from grokking.fourier import (active_neurons, circular_resultant, fourier_coeffs, key_frequencies,
                              neuron_fourier, project_to_frequencies, split_weights,
                              winner_predictions)
from grokking.model import make_model

P = 47
K = (P - 1) // 2
J = torch.arange(P, dtype=torch.float64)


def cosine(alpha, k, phi):
    return (alpha * torch.cos(2 * math.pi * k * J / P + phi)).float()


def test_single_component_recovered():
    for alpha, k, phi in [(1.0, 1, 0.0), (0.3, 7, 1.2), (2.5, K, -2.0)]:
        a, ph = fourier_coeffs(cosine(alpha, k, phi))
        assert a.argmax().item() + 1 == k
        assert abs(a[k - 1].item() - alpha) < 1e-5
        assert abs(math.remainder(ph[k - 1].item() - phi, 2 * math.pi)) < 1e-5
        a[k - 1] = 0
        assert a.abs().max().item() < 1e-5


def test_neuron_stats_synthetic():
    ks = [3, 11, 20, 3]
    phis = [(0.4, -0.9, 0.2), (1.0, 2.0, 3.0), (-1.0, 0.5, -0.5), (0.0, 0.0, 0.0)]
    Wa = torch.stack([cosine(1.0, k, pa) for k, (pa, _, _) in zip(ks, phis)])
    Wb = torch.stack([cosine(0.7, k, pb) for k, (_, pb, _) in zip(ks, phis)])
    Wout = torch.stack([cosine(0.5, k, ps) for k, (_, _, ps) in zip(ks, phis)])
    F = neuron_fourier(Wa, Wb, Wout)
    assert F["kstar"].tolist() == ks
    assert torch.allclose(F["ipr"], torch.ones(4), atol=1e-5)
    for m, (k, (pa, pb, ps)) in enumerate(zip(ks, phis)):
        assert abs(math.remainder(F["phi_a"][m, k - 1].item() - pa, 2 * math.pi)) < 1e-4
        assert abs(math.remainder(F["psi"][m, k - 1].item() - ps, 2 * math.pi)) < 1e-4
        expect = abs(math.remainder(pa + pb - ps, 2 * math.pi))
        assert abs(F["Dt"][m, k - 1].item() - expect) < 1e-4
    # magnitudes are reported after per-layer unit-Frobenius normalization; check the ratio of a to b
    m = 0
    assert abs(F["alpha_a"][m, 2].item() / F["alpha_b"][m, 2].item() - 1 / 0.7) < 1e-4


def test_ipr_diffuse_is_small():
    torch.manual_seed(0)
    W = torch.randn(256, P)
    F = neuron_fourier(W, W.clone(), W.clone())
    assert F["ipr"].mean().item() < 0.2  # random weights are spread over many frequencies
    assert F["ipr"].min().item() >= 1 / K - 1e-6


def test_projection():
    W = cosine(1.0, 4, 0.3) + cosine(0.5, 9, 1.0) + 0.2
    Wp = project_to_frequencies(W.unsqueeze(0), [4]).squeeze(0)
    assert torch.allclose(Wp, cosine(1.0, 4, 0.3), atol=1e-5)
    Wp = project_to_frequencies(W.unsqueeze(0), [4, 9], keep_dc=True).squeeze(0)
    assert torch.allclose(Wp, W, atol=1e-5)
    # along dim 0 (e.g. W2 of shape (p, H))
    Wc = torch.stack([W, W], dim=1)
    assert torch.allclose(project_to_frequencies(Wc, [9], dim=0)[:, 1], cosine(0.5, 9, 1.0), atol=1e-5)


def test_active_and_key_freqs():
    Wa = torch.zeros(5, P)
    Wa[:, 0] = torch.tensor([10.0, 5.0, 3.0, 0.01, 0.01])
    Wout = torch.ones(5, P)
    act = active_neurons(Wa, Wa.clone(), Wout, frac=0.95)
    assert act.tolist() == [True, True, True, False, False]
    kstar = torch.tensor([2, 2, 5, 7, 7])
    assert key_frequencies(kstar, act, min_frac=0.4) == {2: 2}


def test_circular_resultant():
    assert abs(circular_resultant(torch.full((10,), 1.3)).item() - 1) < 1e-6
    assert circular_resultant(torch.linspace(0, 2 * math.pi, 1001)[:-1]).item() < 1e-4


def test_real_model_shapes_and_predictors():
    for mode, d_in in (("onehot", 2 * P), ("tied", P)):
        model = make_model("mlp", P, seed=0, mlp_input=mode, d_hidden=32)
        Wa, Wb, Wout = split_weights(model.state_dict(), P)
        assert Wa.shape == (32, P) and Wout.shape == (32, P)
        preds = winner_predictions(neuron_fourier(Wa, Wb, Wout))
        for v in preds.values():
            assert v.shape == (32,) and v.min() >= 0 and v.max() <= K
        assert (preds["pred_growth"] >= 1).all()
