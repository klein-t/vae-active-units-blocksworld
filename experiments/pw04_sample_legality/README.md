# pw04_sample_legality — legality of states sampled from the prior — **DESIGNED — NOT RUN** (2026-10-14)

Proposal item **P4**. Registration only: no analysis or result exists.
Frozen 2026-10-21 together with pw02 and the other analysis, before any
training. Not run: its pilot needs a completed, gated pw02 pilot, and both
pw02 pilots (`pilot_001`, `pilot_002`) ended in `competence_stop`
(2026-10-21; see the pw02 README). Blocked until a pw02 amendment exists.

**Question.** What fraction of states decoded from prior samples of the
pw02-trained generative models (beta-VAE grid, beta-TCVAE, categorical VAE,
data-control VAEs) are exactly legal (pw01 checker), compared with exact
factorized baselines (uniform bits, independent bits, per-block product) and
with each model's own aggregate posterior; and does legality fall as the
number of active units rises?

**No new training:** analyzes the shared sweep trained once by
`pw02_active_units` (checkpoints pinned by hash).

**Hypotheses (required):** H4.1 beta=1 VAE prior samples are legal more often
than the per-block product baseline; H4.2 within beta strata, legality
decreases with AU (stratified Spearman, cluster-bootstrap CI below 0).
Secondary: prior-hole gap, categorical vs Gaussian, prod5/code10 controls.
Details and forecasts: `PROTOCOL.md`.

**Files.** `PROTOCOL.md`, `DESIGN.json`, `analyze_legality.py`, `audit.py`;
shared code `src/pwlib/legality.py`. Compute: see PROTOCOL section 9.

## Pre-freeze changes (2026-10-14, decided by the project lead; designs not yet frozen)

1. **Seeds** inherited from pw02: H4.1/H4.2 now rest on 5 seeds per
   `bw-gauss_vae` cell (up to 40 runs per beta stratum); experiment-level
   competence uses a majority of planned seeds (>= 3 of 5).
2. **Taxonomy:** V6 (held block supporting a block) added so that every
   checker rejection has a named violation.
3. **Controls made computable:** `prod5` samples scored by Blocksworld
   legality of their 30 position bits with `clear`/`hand_empty` derived (the
   prod5 data have those columns constant zero); control competence uses the
   in-support rate (valid codeword / one place per block).
4. **Code** in `src/pwlib/legality.py` (vectorized taxonomy, exact baselines,
   Monte Carlo) and thin entry scripts; models loaded from `src/pwlib/models.py`.

## Pre-freeze changes (2026-10-21)

- The shared sweep (`pw02_active_units/sweep_grid.json`, frozen with this
  design) now carries the pw02 pre-registered A3 contingency: if it fires in
  the pw02 pilot, every pw02 model is trained with a linear KL warm-up (beta
  0 -> target over the first 10% of steps). This analysis is unchanged; the
  amendment is disclosed in its results if it fires.

## Runs

None (blocked by the pw02 pilot competence stop, 2026-10-21).

## Follow-up (2026-10-21)

Superseded for execution by the pilot-triggered amendment `pw04a_sample_legality_amended` (registered and frozen 2026-10-21, before any confirmation data was opened; see `experiments/pw04a_sample_legality_amended/`). This record is unchanged.
