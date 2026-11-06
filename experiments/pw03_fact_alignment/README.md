# pw03_fact_alignment — do active latents map one-to-one to facts? — **DESIGNED — NOT RUN** (2026-10-14)

Proposal item **P3**. Registration only: no analysis or result exists.
Frozen 2026-10-21 together with pw02 and the other analysis, before any
training. Not run: its pilot needs a completed, gated pw02 pilot, and both
pw02 pilots (`pilot_001`, `pilot_002`) ended in `competence_stop`
(2026-10-21; see the pw02 README). Blocked until a pw02 amendment exists.

**Question.** Do the active latents of beta-VAE, beta-TCVAE, a categorical
(Gumbel-softmax) VAE and a top-k SAE align one-to-one with the 30 position
facts / 5 per-block categorical variables of Blocksworld? Scored with
best-latent F1 (coverage, monosemanticity, Hungarian one-to-one), MIG and DCI,
against fact- and block-oracle positive controls, randomly initialized
encoders and a label-permutation null. Metric conventions follow
`docs/METRICS_SPEC.md` (MIG and DCI as in disentanglement_lib).

**No new training:** analyzes the shared sweep trained once by
`pw02_active_units` (checkpoints pinned by hash).

**Hypothesis (required):** H3.2 a k=5 top-k SAE, matching the exact 5-sparse
affine structure of legal states, covers >= 80% of position facts and beats
every Gaussian-latent comparator. Secondary: H3.1 (descriptive: Gaussian
beta-VAE units mostly not single-fact detectors), categorical/TC vs beta-VAE
on block-MIG, activation substrate. Details and forecasts:
`PROTOCOL.md`.

**Files.** `PROTOCOL.md`, `DESIGN.json`, `analyze_alignment.py`, `audit.py`;
shared code `src/pwlib/alignment.py`, `metrics.py`. Compute: see PROTOCOL
section 9.

## Pre-freeze changes (2026-10-14, decided by the project lead; designs not yet frozen)

1. **H3.1 demoted** to a descriptive secondary analysis (half implied by the
   pw02 counting); `h32_topk_sae_covers_position_facts` is the sole required
   criterion.
2. **Seeds** inherited from pw02: 5 per `bw-gauss_vae` cell (the beta-VAE
   comparators of H3.2), 3 otherwise; separation p-values restated.
3. **Metrics aligned with `docs/METRICS_SPEC.md`:** MIG with 20 equal-width
   bins, nats, empirical H (was equal-frequency bins); DCI with
   disentanglement_lib's gradient-boosted classifiers, fixed `random_state=0`,
   balanced accuracy and plain-average completeness added (was L1 logistic
   importances); best-latent F1 adapted for signed latents (both signs, every
   distinct development threshold) with Karvonen's published rule also
   reported for SAEs; raw-state MIG/DCI as exact weighted expectations over the
   866 states. MIG/DCI code cross-checked against disentanglement_lib
   (`tests/crosscheck/`, agreement to 1e-15).
4. **Positive control split** into a fact oracle (Cov30 >= 0.95) and a block
   oracle (block-MIG >= 0.5): the fact oracle cannot reach block-MIG 0.5 by
   construction.
5. **Permutation nulls:** DCI null limited to 10 permutations on controls and
   reference cells (cost); Cov30/MonoFrac/MIG nulls unchanged (100 / 20).
6. **Activation substrate** extended to the width-128 transformer seeds s2/s3
   (pw02 change).

## Pre-freeze changes (2026-10-21)

- The shared sweep (`pw02_active_units/sweep_grid.json`, frozen with this
  design) now carries the pw02 pre-registered A3 contingency: if it fires in
  the pw02 pilot, every pw02 model is trained with a linear KL warm-up (beta
  0 -> target over the first 10% of steps). This analysis is unchanged; the
  amendment is disclosed in its results if it fires.

## Runs

None (blocked by the pw02 pilot competence stop, 2026-10-21).

## Follow-up (2026-10-21)

Superseded for execution by the pilot-triggered amendment `pw03a_fact_alignment_amended` (registered and frozen 2026-10-21, before any confirmation data was opened; see `experiments/pw03a_fact_alignment_amended/`). This record is unchanged.
