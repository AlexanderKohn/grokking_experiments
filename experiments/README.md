# Experiments

These are instructions for two experiments built on the one-hot MLP baseline from the top-level README.
That baseline is `a + b mod 97`, 30% train, AdamW with lr 1e-3 and wd 1.0, d_hidden 512, and it groks at step 17,200.

| folder | question |
|---|---|
| `01_frequency_lottery/` | Does random initialization decide which Fourier frequency each hidden neuron learns? Does hidden width affect grokking as a coverage ("coupon-collector") problem over frequencies? |
| `02_grokked_tickets/` | Do pruned "grokked tickets" remove the grokking delay in our MLP? Is their advantage explained by which init-time frequencies the mask keeps? |

Each folder's `README.md` is the spec. Put analysis scripts in the folder, put raw runs under `runs/exp01/...` or
`runs/exp02/...` (gitignored), and write findings to a `RESULTS.md` in the folder. Commit the small summary plots.

## Shared infrastructure (build this first; both experiments need it)

Keep every default unchanged, so the top-level README commands still reproduce the baseline exactly.

### `grokking/model.py`
- **`--d-hidden`**: thread `d_hidden` through `build_model`. It is currently hard-coded to 512.
- **`--init-scale c`**: after the default PyTorch init, multiply every parameter by `c`. The default is `1.0`.
- **`--no-bias`** (optional): remove biases from both MLP layers. The analysis in arXiv 2602.16849 uses a bias-free network.
- **`--mlp-input tied`** (optional): `h = onehot(a) + onehot(b)` feeds `Linear(p, H)`. This shares input weights
  between a and b, as in arXiv 2602.16849, where f(x,y) = Σ_m ξ_m σ(⟨h_x + h_y, θ_m⟩). Use it only to replicate
  that paper exactly. Our main runs use the existing untied `onehot` mode.

### `train.py`
- **`--save-init`**: save the state_dict at step 0 to `out/init.pt`.
- **`--ckpt-every N`**: save the state_dict to `out/ckpt_{step}.pt` every N steps. The default is 0 (off).
- **`--init-from path.pt`**: load these weights instead of a fresh init. Seed the data split separately with
  `--seed`, so that "same init, different mask" comparisons use the same split.
- **`--mask path.pt`**: a dict `{param_name: bool tensor}`. Multiply the weights by the mask at load. Zero masked
  gradients with a hook, and re-apply the mask after every `opt.step()`. AdamW's decoupled weight decay keeps zeros
  at zero, but re-masking guards against Adam state leaking through.
- **`--max-steps`**, used with the existing `--stop-on-grok`. Record both `test_99_step` and `test_95_step`, because
  Minegishi et al. use a 95% threshold.
- Add `t_fit_95` and `t_grok_95` to the summary, as well as **`grok_ratio = t_grok / t_fit`**. This is the τ_grok
  of Minegishi et al.; report it log-scaled.

### New module `grokking/fourier.py` (both experiments use it)
For the one-hot MLP: `W1 = net[0].weight` has shape (H, 2p), with `Wa = W1[:, :p]` and `Wb = W1[:, p:]`.
`W2 = net[2].weight` has shape (p, H), and neuron m's output vector is `W2[:, m]`.

With `X = torch.fft.fft(x, dim=-1)` (numpy sign convention), a component `α·cos(ω_k j + φ)` with ω_k = 2πk/p has
`X_k = (pα/2)·e^{iφ}`. For k = 1..K, where K = (p−1)/2:
- **Magnitudes and phases:** `α^a_{m,k} = 2|X^a_{m,k}|/p` and `φ^a_{m,k} = arg X^a_{m,k}`. Compute `(α^b, φ^b)` the
  same way from `Wb`, and `(β, ψ)` from `W2[:, m]`.
- **Power:** `P_{m,k} = (α^a)² + (α^b)² + β²`. Normalize each layer's weights to unit Frobenius norm first, so that
  no single layer dominates.
- **Dominant frequency:** `k*_m = argmax_k P_{m,k}`.
- **Inverse participation ratio** (Gromov 2023; arXiv 2602.16849 with r = 2): `IPR_m = Σ_k P_{m,k}² / (Σ_k P_{m,k})²`.
  It ranges from 1/K for diffuse weights to 1 for a single frequency.
- **Phase misalignment:** `D_{m,k} = (φ^a + φ^b − ψ) mod 2π` and `D̃ = min(D, 2π − D)`. This generalizes the paper's
  `2φ − ψ` to untied inputs, since the logit for c contains cos(ω(a+b−c) + φ^a + φ^b − ψ).
  **Check the sign empirically:** on a grokked network, compute the circular mean resultant length of
  `φ^a + φ^b − ψ` and of `φ^a + φ^b + ψ` at each neuron's k*, and use whichever concentrates. Note which in RESULTS.
- **Active neurons:** a neuron's contribution is `‖W1[m]‖·‖W2[:, m]‖`. Sort neurons by contribution and keep the
  smallest set that accounts for 95% of the total.
- **Key frequencies:** the frequencies that are k* of at least 2% of the active neurons. Report the count per frequency.
- **`project_to_frequencies(W, S)`:** keep only DFT bins k ∈ S (plus their conjugates p−k) and bin 0 if requested,
  then take the real inverse FFT. Both experiments use this for interventions.

Unit test: build synthetic weights `α cos(ω_k j + φ)` and check that magnitude, phase, k* and IPR = 1 are recovered.

### CPU budget notes
The 512-wide one-hot MLP takes about 0.02 s per full-batch step on 3 threads. Small widths are much cheaper.
Run independent jobs in parallel with `--threads` set so the total does not exceed the physical core count
(see the top-level README).
