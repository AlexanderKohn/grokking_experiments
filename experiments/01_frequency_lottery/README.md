# Experiment 01: the frequency lottery and width

Build the shared infrastructure in `experiments/README.md` first. All Fourier quantities (α, φ, β, ψ, P, k*, IPR,
D̃, active neurons, key frequencies) are defined there.

## Background

- Networks that grok modular addition use a Fourier algorithm in which each neuron specializes in one frequency
  (Nanda et al. 2023; Gromov 2023).
- arXiv 2602.16849 (*On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking*)
  analyzes a 2-layer network under small random initialization. In their gradient-flow analysis, frequencies compete
  inside each neuron, and the winner is the one with the **largest initial magnitude and smallest initial phase
  misalignment**. The paper calls this a "lottery ticket mechanism". The network then needs **frequency
  diversification**, meaning the neuron population covers the frequencies it needs, so that noisy single-neuron
  signals cancel by "majority vote".
- They describe grokking (75% train, wd 2.0, p = 23, width 512) as memorization followed by two generalization
  stages, in which weight decay prunes the frequencies that are not features.
- **The paper does not run a width sweep.** It only says the results hold for "sufficiently large" width M.
  That gap is this experiment.

## Hypotheses

- **H1 (lottery):** each active neuron's final k* can be predicted from θ₀ well above chance (1/K).
- **H2 (coverage):** if winners at init are roughly uniform over the K = (p−1)/2 frequencies, the expected number
  covered at width H is `K·(1 − (1 − 1/K)^H)` (coupon collector). If grokking needs a particular set or number of
  frequencies, then P(grok) and t_grok as functions of H should follow statistics computable **from θ₀ alone**.
- **Competing explanations, which the design must be able to tell apart:**
  - **(A) Coverage-limited:** grokking needs the frequencies that win at init to cover enough of the spectrum.
  - **(B) Redundancy-limited:** weight decay leaves only a few key frequencies, so the binding constraint is having
    enough neurons on some small set of frequencies, not coverage of all K. Previous work typically finds about
    3–6 key frequencies after grokking.
  - **(C) Not binding:** neurons switch frequency during training, especially in the memorization stage (the
    paper's "perturbed lottery"), so width matters only through ordinary optimization effects.

## Setup

- **Main sweep:** `--model mlp --mlp-input onehot --op add`. Use **p = 47** (K = 23) so the width sweep is cheap.
  First run a pilot at H = 512 to choose `--train-frac`: pick the smallest value in {0.3, 0.4, 0.5} for which the
  baseline groks in under 20k steps. Freeze it and record it in RESULTS.md.
- **Confirmation:** repeat selected points at p = 97 with frac 0.3, the repo baseline.
- **Optional replication arm:** `--mlp-input tied --no-bias` at p = 23 with frac 0.75 and wd 2.0, to compare
  directly with the paper.
- **Grokking criterion:** test accuracy ≥ 99%. Also log the 95% threshold. Cap runs at 50k steps with
  `--stop-on-grok`, and treat a run as "did not grok" if it reaches the cap.

## Phase 1A: init-only Monte Carlo (no training, minutes of compute)

For H ∈ {4, 8, 16, 32, 64, 128, 256, 512, 1024} and 1,000 seeds each, build θ₀ and compute every neuron's predicted
winner under each predictor:

- **pred_power:** `argmax_k P_{m,k}(0)`
- **pred_growth:** `argmax_k α^a α^b β · cos(D_{m,k}(0))`
- **pred_growth_pos:** the same as pred_growth, but only over k with cos D > 0. If no such k exists, mark the neuron
  "no winner".

Report:
- whether the predicted-winner distribution is uniform over k (χ² test);
- the distribution of the number of distinct predicted frequencies per network, against the coupon-collector curve;
- the distribution of the minimum count of neurons on any one frequency.

Save these as the "init statistics" used in 1C.

*(This is our own operationalization of the winner criterion. If none of these predictors works in 1B, check the
exact form of Corollary 6.1 in the paper.)*

## Phase 1B: does the lottery hold? (H = 512, 3 seeds)

- **Init scales:** `--init-scale` ∈ {0.1, 0.3, 1.0}. The theory assumes small init; 1.0 is PyTorch's default.
- **Data regimes:** full data (`--train-frac 1.0`, no grokking, as in the paper's main setting) and the grokking split.
- **Logging:** use `--ckpt-every 250` up to 5k steps, then every 1,000. At each checkpoint compute k*_m, IPR_m and
  D̃_{m,k*} for every neuron.

Report:
1. Top-1 accuracy of each predictor against final k*, over active neurons only, with chance 1/K and a bootstrap CI.
2. **Lock-in time:** the step after which a neuron's k* never changes. Show its distribution relative to t_fit and
   t_grok.
3. The fraction of active neurons that **switch** frequency between t_fit and t_grok. A high fraction supports (C).
4. Mean IPR and the circular concentration of D over training, overlaid on the accuracy curves. The paper's
   progress measure is |sin D|.
5. Whether final key frequencies were over-represented among predicted winners at init. If they were, weight decay
   is picking survivors from the lottery's results.

## Phase 1C: width sweep (main result)

- **Runs:** H ∈ {4, 8, 16, 32, 64, 128, 256, 512} × 8 seeds × p = 47, with init scale at whatever 1B shows makes
  the lottery most predictive. Also run init scale 1.0 if that differs.
- **Record per run:** grokked (yes/no), t_fit, t_grok, `grok_ratio`, the number of key frequencies, neurons per key
  frequency, and this run's init statistics from 1A (distinct predicted frequencies, minimum count per frequency).
- **Plots:**
  - P(grok) vs H and t_grok vs H, as medians with IQR.
  - The coupon-collector expected coverage on the same x-axis.
  - Scatter of t_grok against init-predicted coverage, and against the init-predicted maximum neurons on any one
    frequency, with the 8 seeds at fixed H as color.
- **The key test:** **at fixed H**, do seeds with better init coverage or redundancy grok earlier?
  Within-width variation controls for capacity, so a correlation there is evidence that the lottery matters, beyond
  the effect of width as capacity. Report Spearman ρ with a permutation p-value.

## Phase 1D: interventions (causal tests; H = 64 or 128, whichever sits near the grokking threshold in 1C)

Use one baseline seed, then edit θ₀ with `project_to_frequencies` and retrain with the same data split:

1. **Knockout:** remove that seed's final key frequencies from every neuron's W1 and W2 at init, then rescale the
   rows to their original norms. Questions: does it still grok, how late, and with which frequencies?
2. **Monoculture:** project every neuron onto a single frequency, rotating through 3 choices of k. With p prime and
   addition, one frequency is enough in principle, so this tests whether diversity matters for *learning dynamics*
   rather than for representability.
3. **Rigged coverage:** keep the width fixed, but assign predicted winners round-robin across all K frequencies.
   Do this by boosting α and β at the assigned k and setting its phase aligned. Compare against a random-assignment
   control with the same boost.

If (A) holds, 1 and 2 should slow or prevent grokking and 3 should speed it up. If (B) holds, 2 may be fine or even
faster. If (C) holds, all three should look like the baseline once training has run long enough.

## Deliverables (`experiments/01_frequency_lottery/RESULTS.md`)

- the train-frac calibration and a sanity check that the baseline groks;
- a table of predictor accuracies (1B);
- the lock-in and switching plots;
- the width-sweep plots and within-width correlations;
- intervention curves;
- one paragraph on which of (A), (B) or (C) the evidence favors, including surprises.

## References

- arXiv 2602.16849: Fourier features, the lottery ticket mechanism, and grokking for modular addition.
- Nanda et al. 2023, *Progress measures for grokking via mechanistic interpretability* (arXiv 2301.05217).
- Gromov 2023, *Grokking modular arithmetic* (arXiv 2301.02679): analytic Fourier solution for the MLP, and IPR.
- Edelman et al. 2023, *Pareto Frontiers in Neural Feature Learning: Data, Compute, Width, and Luck*
  (arXiv 2309.03800): in sparse parity, width acts as parallel search that raises the chance of "lottery ticket"
  neurons. This is the closest precedent for the width-as-luck framing.
- Varma et al. 2023, *Explaining grokking through circuit efficiency* (arXiv 2309.02390): context for why weight
  decay favors the Fourier circuit.
