# Experiment 01 follow-ups (Phase 1E)

These come from reviewing `RESULTS.md` and the blog post. Each item says what to run, why, and how to read the
outcome. Unless stated otherwise, the defaults are those of 1C: p = 47, frac 0.5, AdamW with lr 1e-3 and wd 1.0,
init scale 1.0, a 50k-step cap, and the 95% criterion as the primary grokking measure.
Add the results to `RESULTS.md` as "Phase 1E", and update the conclusion where needed.

Priority order: **E1 (free) → E3 → E2 → E5 → E4 → E6**.

**Status (November 9, 2026):** E1, E2, E5 (both lr arms; the μP arm was not needed) and E6 are done, plus an
init-vs-split test that was not in this plan. Results are in `RESULTS.md`, Phase 1E. E3 and E4 are open.

---

## E1. Was the lottery drawn by Adam's first steps? (no new runs)

**Why:** neurons lock in within 10–20 steps, during LR warmup, and `pred_growth` on θ₀ tops out at 44%. Adam's
first updates are roughly lr·sign(g) per coordinate. Those updates can reshape the spectrum in ways the
gradient-flow criterion of arXiv 2602.16849 does not model. That would also explain why a smaller init scale made
θ₀ *less* predictive.

**Run:** apply `pred_power` and `pred_growth` to the 1B checkpoints at t ∈ {0, 10, 20, 30, 50, 100, 200}. The dense
checkpoints already exist every 10 steps up to step 500. Score each against final k* over the active neurons.
Plot accuracy against t, for every init scale and data regime.

**Read:**
- **The lottery is real but optimizer-drawn:** accuracy jumps to ≥ 80% by t = 10–30.
- **Assignment happens gradually through memorization:** accuracy rises slowly instead.

Either way, the post should say which.

## E2. Plain gradient descent and warmup arms

**Why:** the theory assumes gradient flow. Testing the lottery under the optimizer it was derived for separates
"the theory is wrong" from "the theory doesn't transfer to Adam".

**Run (H = 512, seeds 0–2):**
1. **Full-data GD:** full-batch `torch.optim.SGD` with no momentum, `--train-frac 1.0`, and init scale ∈ {0.1, 1.0}.
   Use L2 weight decay through SGD's `weight_decay`, at a value that gives a decay per step comparable to AdamW's
   lr·wd. Calibrate lr in {0.03, 0.1, 0.3, 1.0} on one seed so the run fits the training data within 20k steps.
   This arm only needs predictor accuracy, lock-in and switching, not grokking.
2. **GD on the grokking split:** the same setup at frac 0.5, capped at 50k steps. Report it even if it doesn't grok.
3. **Adam warmup ablation:** `--warmup` ∈ {0, 10, 100, 1000}, all on the grokking split. Does lock-in time track
   the end of warmup? Does predictor accuracy change?

**Read:**
- **GD predictor accuracy ≫ 44% and less switching:** the paper's lottery holds for gradient flow, and Adam blurs it.
- **Similar to Adam:** the paper's criterion is incomplete even in its own regime.

## E3. Does the width threshold scale with K = (p−1)/2? (the most important new test)

**Why:** at p = 47 the threshold sits between H = 64 and 128, which is about 3–6 neurons per frequency. If
generalization needs a fixed number of neurons on each of the K frequencies, the threshold should scale linearly
with K. That would be a clean combinatorial law. If the threshold doesn't move with p, the bottleneck is something
else, such as memorization capacity or optimization.

**Run:**
- **Moduli:** p ∈ {31, 47, 61, 97}, so K ∈ {15, 23, 30, 48}.
- **Widths:** H ∈ {32, 48, 64, 96, 128, 192, 256, 384, 512}.
- **Seeds:** 4. Fixed `--train-frac 0.5`.
- First check that H = 512 groks at every p. If not, record the change.
- Prune the grid adaptively. Once a p has two consecutive widths with 4/4 seeds reaching 95%, skip larger widths
  except 512.

**Report:**
- **H\*(p):** the smallest H at which at least 3 of 4 seeds reach 95% test accuracy within 50k steps.
- A plot of H\* against K, and of H\*/K against p.
- At H\*, each run's final effective number of frequencies and its neurons per frequency.
- **Control for training-set size:** p² grows with p, so also run p = 97 at a train fraction that matches p = 47's
  1,104 training examples (frac ≈ 0.117), at three widths around its H\*. This separates "more frequencies" from
  "more data".

**Read:**
- **A neurons-per-frequency law:** H\* ∝ K, with H\*/K roughly constant, and the threshold insensitive to the
  training-set-size control.
- **Not frequency-limited:** H\* stays flat in p.
- **A data effect:** H\* tracks the number of training examples.

## E4. Embedding bottleneck: why does this MLP use all 23 frequencies?

**Why:** the "no key frequencies" result is probably expected rather than surprising. The one-hot untied MLP has
no bottleneck. As far as we know, the analytic MLP solution (Gromov 2023) spreads neurons over all frequencies, and
arXiv 2602.16849's frequency-diversification condition says the same. The 3–6 key frequencies in the literature
come from transformers, where inputs pass through a shared embedding (d_model = 128) under weight decay.
**Check the Gromov claim against the paper before relying on it.**

**Run:** `--mlp-input embed` at H = 512 with `d_embed` ∈ {4, 8, 16, 32, 128}, 3 seeds each. Expose `--d-embed` if
it isn't already a flag. In addition:
- Run `--mlp-input tied` with `--no-bias` at the same width, as the arXiv 2602.16849 architecture.
- Run the same Fourier analysis on the repo's transformer baseline (p = 97): take the DFT of the token-embedding
  columns for tokens 0..p−1, and of the unembedding.

**Report:** the effective number of frequencies (1/Σq²) against d_embed, and against architecture.

**Read:** each frequency needs about 2 embedding dimensions (cos and sin). If the bottleneck is the cause, the
number of frequencies used should fall to about d_embed/2 or fewer once d_embed/2 < K, with key-frequency sparsity
appearing as in transformers. If it doesn't, the transformer's sparsity comes from somewhere else, such as attention.

## E5. Is the width effect above threshold just effective learning rate?

**Why:** in standard parameterization with Adam at a fixed lr, every parameter moves by about lr per step. So
wider networks change their output faster in function space. "Width adds neurons per frequency" is confounded with
"width raises the effective lr". The 6× speedup from H = 128 to 512 needs this control before it supports the
conclusion.

**Run:**
1. **LR sweep at fixed width:** H = 512, 3 seeds, lr ∈ {1e-3, 5e-4, 2.5e-4, 1.25e-4}, in two arms:
   - (a) wd fixed at 1.0, so decay per step falls with lr;
   - (b) lr·wd held at 1e-3, so wd scales up.

   Can a lower lr at H = 512 reproduce the slower t95 of H = 256 or 128?
2. **μP-style width sweep:** use separate param groups for the 1-hidden-layer MLP. Take base width H₀ = 128.
   - W1 and b1: default init, lr η.
   - W2: init std scaled by H₀/H relative to the default (default std ∝ 1/√H, so the result ∝ 1/H), and
     lr η·(H₀/H).

   Rerun H ∈ {128, 256, 512, 1024} × 4 seeds. Check that H = 128 reproduces standard parameterization exactly.

**Read:**
- **The width effect is mostly parameterization:** under μP, t95 is roughly flat in H above threshold.
- **Supports "more neurons per frequency":** the speedup survives μP and can't be matched by lowering lr.

## E6. Rigged init: dose–response and decomposition

**Why:** the rigged init changes per-neuron concentration (IPR), phase alignment and overall Fourier-signal strength
all at once. The current conclusion ("starting strength of the Fourier signal decides grokking time") needs those
pulled apart. Also, at random init the phases are unaligned, so there is no coherent population signal. The width
claim has to be stated in terms of a measurable quantity.

**Run (H = 128, seeds 0–2, same split, `--init-from`, rows rescaled to their original norm as in 1D):**
1. **Dose–response:** boost ∈ {0, 0.25, 0.5, 1, 2, 4} × the neuron's largest init magnitude, with aligned phase
   (D = 0) and random assignment.
2. **Concentration only:** the same boosts with a **random** phase, so D is uniform.
3. **Alignment only:** no boost. Rotate the phases of each neuron's existing `pred_growth` winner so that D = 0,
   leaving magnitudes unchanged.

**Unifying test:** define the initial coherent Fourier signal of a network as
`S₀ = Σ_k |Σ_{m: k*_m = k} α^a α^b β · e^{iD}|`, using each neuron's init-time winner. Compute S₀ for **every** run
in 1C, 1D and E6, and plot log t95 against log S₀.

**Read:**
- **If the runs collapse onto one curve,** across widths, rigged variants and seeds, then "initial coherent signal
  strength" is the right variable. Fit `t95 ≈ a − b·log S₀`: under exponential growth of the generalizing circuit
  from an initial strength S₀ (a competition picture like Varma et al.'s), time-to-threshold is linear in log S₀.
- **If width runs sit off the curve,** width acts through something else; see E5.
- **If concentration-only or alignment-only alone captures most of the speedup,** that names the mechanism.

---

## Write-up revisions (blog post and RESULTS.md)

1. **Reframe "no key frequencies"** as the expected behavior of an untied one-hot MLP, unlike transformers. Cite
   the frequency-diversification condition in arXiv 2602.16849 and (after checking) Gromov 2023. Update it after E4.
2. **Qualify the width explanation.** "Width adds neurons per frequency" stays a hypothesis until E5 rules out the
   effective-lr confound and E6 tests the S₀ collapse.
3. **Restate the "starting strength" claim precisely.** Random init has no coherent population signal, because
   phases are random. The claim should be stated in terms of S₀ (E6) or of the number of candidate neurons, not as
   an existing signal that width amplifies.
4. **Cite prior work on structured initialization speeding up grokking:** *Let Me Grok for You: Accelerating
   Grokking via Embedding Transfer from a Weaker Model* (GrokTransfer, Xu et al., 2025, arXiv 2504.13292). The new contribution is identifying
   *which* property of the init matters (E6).
5. **Connect the rigged-init result to circuit competition:** Varma et al. 2023 (arXiv 2309.02390), where the
   generalizing circuit is slower to learn but more efficient. A rigged init gives it a head start.
6. **Contrast with sparse parity:** Edelman et al. 2023 (arXiv 2309.03800) find that width acts as parallel search
   for rare "lottery ticket" neurons. The within-width null here suggests modular addition is different: the task
   is dense and full-spectrum, with no rare winning neuron to find. Worth a paragraph.
7. **Link to Experiment 02:** the rigged init is essentially the "spectral ticket" of 2F.3. It predicts that
   grokked tickets will be matched by Fourier-concentrated, phase-aligned initializations, and that grokked-ticket
   masks will have a DFT peak at each neuron's final frequency (2E.4).
