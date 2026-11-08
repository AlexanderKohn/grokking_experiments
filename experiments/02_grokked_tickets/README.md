# Experiment 02: grokked tickets and what the mask encodes

Build the shared infrastructure in `experiments/README.md` first (`--save-init`, `--ckpt-every`, `--init-from`,
`--mask` and `grokking/fourier.py`). Phase 2F also uses Experiment 01's init-time winner predictors.

## Background

- **Minegishi, Iwasawa & Matsuo, *Bridging Lottery Ticket and Grokking* (TMLR 2025, arXiv 2310.19470):**
  - Magnitude-prune a network after it generalizes, reset the surviving weights to θ₀, and retrain. This
    "grokked ticket" largely removes the grokking delay.
  - Their controls rule out weight norm and sparsity alone as the cause: norm-matched dense networks, random
    pruning, SNIP, GraSP, SynFlow, and tickets taken before generalization.
  - Their MLP setup: p = 67, 40% train, AdamW (lr 1e-3, wd 1.0, β = (0.9, 0.98)), a one-hot → embedding → 48-neuron
    ReLU MLP.
  - Pruning rates tried: 0.1–0.9. Their default is 0.6, τ_grok is minimized near 0.7, and 0.9 fails to generalize.
  - Tickets from checkpoints before generalization give no speedup.
  - They also report that with a ticket, weight decay is no longer needed. Edge-popup can find good masks on frozen θ₀.
  - Their reported structural properties: periodic weight patterns, longer average path length, lower clustering
    coefficient, and fast mask change (Jaccard distance) around the transition.
- **Open question this experiment targets:** *why* does the mask help? Experiment 01's lottery view suggests a
  concrete answer: the mask may be choosing neurons, or within-neuron weights, whose **init-time Fourier
  components** already favor the frequencies that eventually win. If so, an equally good ticket should be buildable
  **from θ₀ alone, without training** (Phase 2F).

## Setup

- **Main runs:** the repo's one-hot MLP baseline: p = 97, frac 0.3, wd 1.0, lr 1e-3, H = 512, which groks at about
  step 17,200. Seeds 0, 1 and 2.
- **Pruning:** layerwise magnitude pruning of `net[0].weight` and `net[2].weight` only. Keep biases dense.
  Also run a global-magnitude variant at the best rate.
- **Retraining:** start from `θ₀ ⊙ m`, with the same data split and optimizer as the dense run, the mask enforced
  every step, and 30k steps max.
- **Metrics:** t_fit and t_grok at both 95% and 99%, plus `grok_ratio = t_grok / t_fit`. Report the ratio
  log-scaled, as Minegishi et al. do.

## Phase 2A: dense reference runs

Train the 3 seeds with `--save-init --ckpt-every 500` for 30k steps.
These checkpoints are the mask sources for everything below.

## Phase 2B: pruning-rate sweep

- **Mask source:** the checkpoint 2,000 steps after the dense run's t_grok.
- **Rates:** {0.3, 0.5, 0.6, 0.7, 0.8, 0.9}.
- **Plot:** grok_ratio and t_grok against rate, with the dense baseline drawn as a horizontal line.
  Pick `k_best` for the later phases.

## Phase 2C: when does a ticket become "grokked"?

- **Rate:** `k_best`.
- **Mask sources:** t ∈ {250, 1k, 2.5k, 5k, 10k, 14k, t_grok − 1k, t_grok, t_grok + 2k, 30k}.
- **Plots:**
  - grok_ratio against the mask's source step, overlaid on the dense run's test-accuracy curve.
  - Jaccard distance between masks 500 steps apart, across training. Minegishi et al. report rapid structural
    change at the transition; check whether it lines up with Experiment 01's frequency lock-in and switching.

## Phase 2D: controls (rate `k_best`, all 3 seeds)

1. **Random mask** with the same per-layer sparsity.
2. **Norm-matched dense:** `θ₀ · r`, with r chosen to match the ticket's L2 norm. Repeat matching L1.
3. **Ticket with wd = 0.** This replicates their weight-decay-free claim.
4. **Ticket with re-randomized init:** the same mask on a fresh θ₀′. This is the standard lottery-ticket-hypothesis
   control, and it separates "the mask alone carries the structure" from "the mask and θ₀ together carry it".
5. **Dense with wd = 0.** This is the reference for control 3.

## Phase 2E: Fourier anatomy of the ticket (the core analysis)

Use the `fourier.py` measures and compare four objects for each seed:
- dense θ₀;
- the ticket at init, `θ₀ ⊙ m`, before any retraining;
- the dense network after grokking;
- the retrained ticket.

Questions:

1. **Neuron selection:** which neurons does the mask keep, measured as the fraction of each row and column kept?
   Are the kept neurons the dense network's final active neurons? Report precision and recall.
2. **Does the mask pre-select frequencies?** Compare the IPR of `θ₀ ⊙ m` with the IPR of θ₀. Also check whether
   each kept neuron's k* at `θ₀ ⊙ m` equals its final k* in the dense run, compared with the same agreement for
   unmasked θ₀ (Experiment 01's lottery accuracy).
3. **Does the mask pre-align phases?** Compare D̃_{m,k*} at `θ₀ ⊙ m` with D̃ at θ₀.
4. **Is the mask itself periodic?** Each row of the W1 mask restricted to the a-block is a binary indicator of a
   subset of Z_p. Take the DFT of the indicator (minus its mean). Does its peak frequency match the neuron's final
   k*? Compare against random masks with the same density. This is the most direct test of whether "good structure"
   in Minegishi et al. means a Fourier filter.
5. **Retrained ticket:** does it use the same key frequencies as the dense run from the same θ₀?
   Does it skip the noisy memorization stage? Track IPR and D over its training.

## Phase 2F: can a ticket be built from θ₀ alone?

Each construction below gets the same sparsity budget as `k_best`. Compare t_grok and grok_ratio against the grokked
ticket and the dense run.

1. **Init-score neuron ticket:** keep the top neurons by Experiment 01's best init predictor score (for example
   `α^a α^b β cos D` at the neuron's predicted winner). Drop whole neurons; do not prune within neurons.
2. **Oracle-frequency neuron ticket** (uses hindsight): keep the neurons whose *init-predicted* winner is among the
   dense run's final key frequencies. This is an upper bound on "knowing which lottery numbers win".
3. **Spectral ticket** (not a binary mask): project each neuron's θ₀ weights onto its init-predicted frequency
   with `project_to_frequencies`, keeping the norm.
4. **Fourier-shaped binary mask:** for each neuron, keep the input weights j for which
   `cos(ω_{k*} j + φ_{m,k*}(0)) > 0`, and do the same for outputs. This tests whether a hand-built periodic mask
   reproduces what 2E.4 finds.
5. **Grokked-ticket neuron set only:** keep the grokked ticket's surviving neurons, but fully dense within them.
   This separates neuron selection from within-neuron sparsity.

Interpretation:
- If 2 or 3 matches the grokked ticket, the ticket's advantage is mostly "pre-selected lottery winners".
- If only the true grokked ticket works, the mask encodes something beyond init-time frequency structure.
  Use 2E to find out what.

## Phase 2G (stretch): strong lottery ticket and width

- **Method:** edge-popup (Ramanujan et al. 2020). Learn per-weight scores on frozen θ₀ and keep the top (1 − k)
  fraction in the forward pass. Use sparsity 0.5.
- **Widths:** H ∈ {128, 256, 512, 1024, 2048}.
- **Question:** what is the smallest width at which a generalizing subnetwork exists without any weight training?
- **Comparison:** relate that minimum width to Experiment 01's coverage statistics, and to the logarithmic
  overparameterization bound of Pensia et al. 2020 (arXiv 2006.07990), which reduces pruning random networks to
  random Subset Sum. A subnetwork that implements the Fourier circuit needs enough random weights per frequency to
  approximate the target cosines, which is a concrete combinatorial target.

## Deliverables (`experiments/02_grokked_tickets/RESULTS.md`)

- a comparison with Minegishi et al., stating whether the speedup and the controls replicate in our setup;
- the rate and checkpoint sweeps;
- the Fourier-anatomy figures, especially the mask DFT against final k*;
- a 2F table: construction vs t_grok vs grok_ratio;
- a short conclusion on what the mask encodes.

## References

- Minegishi, Iwasawa & Matsuo, *Bridging Lottery Ticket and Grokking: Understanding Grokking from Inner Structure of
  Networks*, TMLR 2025 (arXiv 2310.19470). Code: github.com/gouki510/Grokking-Tickets.
- Merrill, Tsilivis & Shukla 2023, *A Tale of Two Circuits: Grokking as Competition of Sparse and Dense Subnetworks*
  (arXiv 2303.11873). They find sparse-subnetwork emergence through selective norm growth in sparse parity.
  As an extra measurement, track per-neuron norm growth of the ticket's neurons in the dense run.
- arXiv 2602.16849: the lottery ticket mechanism for Fourier features in modular addition.
- Frankle & Carbin 2019, *The Lottery Ticket Hypothesis* (arXiv 1803.03635).
- Ramanujan et al. 2020, *What's Hidden in a Randomly Weighted Neural Network?* (arXiv 1911.13299): edge-popup.
- Pensia et al. 2020, *Optimal Lottery Tickets via SubsetSum* (arXiv 2006.07990).
