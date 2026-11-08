# Experiment 01 results: the frequency lottery and width

Status: **in progress.** Shared infrastructure, calibration and Phase 1A are done. Phase 1B is running.
Phases 1C and 1D have not started.

## Reproduce

```bash
python -m pytest -q tests                                              # Fourier unit tests
python experiments/01_frequency_lottery/phase1a_init_stats.py --p 47 --seeds 1000
bash   experiments/01_frequency_lottery/run_phase1b.sh 0.5 6           # 18 runs, about 6 in parallel
python experiments/01_frequency_lottery/phase1b_analysis.py
```

Raw runs go to `runs/exp01/` (gitignored). Summary plots and JSON go to `figures/`.

## Calibration: train-frac at p = 47 (H = 512, init scale 1.0, wd 1.0, lr 1e-3)

Pilot capped at 20k steps, using `--stop-on-grok`:

| train-frac | seed 0 test ≥ 99% | seed 1 test ≥ 99% |
|---|---|---|
| 0.3 | did not grok by 20k | did not grok by 20k |
| 0.4 | 17,900 | did not grok by 20k (97.7% test at 19k) |
| **0.5** | **5,250** | **5,850** |

**Frozen: `--train-frac 0.5`** (1,104 train / 1,105 test pairs). This is the smallest fraction in {0.3, 0.4, 0.5}
at which both pilot seeds grok in under 20k steps. At 0.4 only one seed made it, and narrower networks in 1C would
probably hit the cap. In every case train accuracy reaches 99% by step 150–200.

## Phase 1A: init-only statistics (p = 47, K = 23, 1,000 seeds per width)

The θ₀ for each seed is built with the same `make_model` call that `train.py` uses, so these statistics belong to
the actual networks trained in 1C. All three predictors are scale-invariant: they use Frobenius-normalized layers
and phases. So the statistics hold for every `--init-scale`. Full numbers: `figures/phase1a_summary.json`.
Per-network values: `runs/exp01/phase1a/init_stats.csv`.

![coverage](figures/phase1a_coverage.png)

- **The winners are uniform over k.** A χ² test on winners pooled over seeds finds no departure from uniform at any
  width, for any predictor (all p > 0.1). At H = 512 the largest relative deviation of any frequency from 1/K is
  under 1%. This is expected, because i.i.d. init makes the DFT bins k = 1..K exchangeable.
  ![winners](figures/phase1a_winner_hist.png)
- **Coverage follows the coupon-collector curve exactly.** The mean number of distinct predicted frequencies matches
  `K(1 − (1 − 1/K)^H)` to within 0.1 at every width. Coverage is about 4 at H = 4, 12 at H = 16, 21.7 at H = 64,
  and all 23 frequencies from H = 128 on.
- **The minimum and maximum neurons per frequency match a uniform multinomial.** The minimum is 0 for H ≤ 64,
  about 1.6 at H = 128, 5.3 at 256, 13.6 at 512 and 32 at 1024. The maximum is about 4 at H = 32, 6.4 at 64,
  10.5 at 128 and 32 at 512.
  So the init-time "lottery" is statistically just i.i.d. uniform assignment. Within-width variation in 1C is the
  ordinary multinomial spread.
- **The two predictors mostly disagree.** `pred_power` and `pred_growth` agree on only **23%** of neurons, at every
  width, against 1/K ≈ 4.3% by chance. They are positively related but far from identical, so 1B can tell which
  criterion, if either, matters.
- **`pred_growth_pos` almost always equals `pred_growth`.** It differs only when cos D ≤ 0 at every k, which happened
  for 4 neurons in 128,000 at H = 128 and for none at any other width.

Implication for 1C: under hypothesis (A), coverage is complete at every H ≥ 128, so P(grok) and t_grok should stop
depending on coverage there. Any coverage effect has to show up between H = 16 and H = 64. Under (B), the relevant
init statistic is redundancy, which keeps growing roughly linearly in H.

## Phase 1B: does the lottery hold?

*Running.* 18 runs: H = 512 × init scale {0.1, 0.3, 1.0} × {grokking split at 0.5, full data} × seeds {0, 1, 2}.
Checkpoints every 250 steps to step 5k, then every 1,000. Grokking runs stop 5k steps after test reaches 99% (cap
50k). Full-data runs train for 20k steps.

## Infrastructure notes

- `grokking/model.py`: new `--d-hidden`, `--init-scale`, `--no-bias` and `--mlp-input tied`. `make_model(...)` is
  now the single place a seed's θ₀ is built.
- `train.py`: new `--save-init`, `--ckpt-every` (with `--ckpt-sparse-after/--ckpt-sparse-every` for the 250→1000
  schedule), `--init-from`, `--mask`, `--max-steps` (an alias of `--steps`) and `--post-grok-steps`. Every run also
  saves `final.pt`. The summary now includes `train/test_{95,99}_step`, `grok_ratio`, `log10_grok_ratio` and the
  same ratios at 95%. With `--train-frac 1.0` the test metrics are NaN.
- **Defaults are unchanged.** Rerunning the original one-hot p = 97 baseline for 1,000 steps reproduces the old
  history bit for bit (the maximum difference in losses and accuracies is 0.0).
- `grokking/fourier.py` implements every quantity in the spec. `tests/test_fourier.py` checks recovery of magnitude,
  phase, k* and IPR = 1 on synthetic cosines, plus `project_to_frequencies`, the active set and key frequencies.
