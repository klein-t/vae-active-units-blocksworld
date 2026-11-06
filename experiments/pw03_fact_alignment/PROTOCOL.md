# pw03_fact_alignment — protocol (frozen 2026-10-21)

Proposal item **P3**. **No new training.** Analysis of the shared sweep
trained and retained by `pw02_active_units` (`sweep_grid.json`); checkpoints
are loaded by path + SHA-256 from the pw02 run manifests. Random-init null
encoders are created here (initialization only, no training).

## 1. Question

Do the active latents of an unsupervised model align one-to-one with the
ground-truth facts, or with the per-block categorical factors? Compare a
Gaussian beta-VAE, beta-TCVAE, a categorical (Gumbel-softmax) VAE and a
properly trained top-k SAE, with several metrics, since single
disentanglement metrics are unreliable (Locatello et al.; 1911.11791).

Competing expectations: Gaussian VAE latents align with high-variance (PCA-
like) directions rather than facts (Rolinek et al.), and logically coupled
facts entangle (Traeuble et al.); categorical latents disentangle better
(Friede et al.); a top-k SAE matches the data's exact sparse-linear structure
(below); derived facts get absorbed (Chanin et al.).

## 2. Ground truth and identifiability (stated before scoring)

- **Position facts (30):** `on_table(X)` x5, `on(X,Y)` x20, `holding(X)` x5.
  Each block X is in exactly one of 6 places, so these are the one-hot
  indicators of the **5 per-block categorical variables** (L2 targets).
- **Derived facts (6):** `clear(X) = 1 - sum_Y on(Y,X) - holding(X)` and
  `hand_empty = 1 - sum_X holding(X)` (= step parity). They are affine
  functions of position facts, not separate factors: scored and reported,
  **excluded from the required coverage criterion**.
- Consequence for SAEs: every legal state is exactly
  `x = c + sum_{blocks b} d_{pos(b)}`, a nonnegative 5-sparse combination of
  at most 30 atoms. Because the 6 indicators of a block sum to 1, one value per
  block can be absorbed into the bias and represented by absence, so a
  perfect k=5 SAE can legitimately cover between 25 and 30 of the 30 position facts.
- Facts are mutually exclusive within a block and coupled across blocks
  (`on(X,Y)` excludes `on(Z,Y)`), so one-to-one recovery is a hypothesis to
  test, not a guaranteed optimum.

## 3. Models and codes scored

From the pw02 run (gated runs only; pw02 competence gates are reused):

| Family | Cells | Code |
|---|---|---|
| beta-VAE | all 24 `bw-gauss_vae-*` cells (reference A2, lr 3e-4, beta 1 and 4) | posterior mean mu; active set = pw02 AU set |
| beta-TCVAE | `bw-tc_vae-A2-b{2,6}` | mu; active set by the same AU rule |
| categorical VAE | `bw-cat_vae-A2-b1` | per-class posterior probabilities of active variables (6 per variable) |
| top-k SAE | `bw-topk_sae-k{5,10}` | feature activations; active = alive (fires on >= 0.5% of weighted states) |
| plain AE | `bw-plain_ae-A2` | codes (all units) |
| PCA | deterministic | top-25 component scores (PCA with development weights) |
| Substrate (b), secondary | `b_raw_w128` gauss (learned gamma, beta 1), tc_vae, cat_vae, topk_sae k5/k10, plain_ae; `b_lin_*` (all 5 sources) and `b_raw_{w64,w128s2,w128s3,w256}` gauss (learned gamma) | as above, recomputed from the retained checkpoints |

Seeds follow pw02: 5 per `bw-gauss_vae` cell, 3 for every other cell.

## 4. Metrics (frozen)

Implementation `src/pwlib/metrics.py` and `src/pwlib/alignment.py`, following
`docs/METRICS_SPEC.md` (binding); deviations from the reference conventions are
listed at the end of this section.

- **M1, per (code j, fact f) best F1** (Karvonen et al. 2408.00113, adapted):
  classifier `s * code_j > t`; sign s in {+, -} and threshold t chosen to
  maximise weighted F1 on the development rows over **every distinct split of
  the sorted development values** (all-positive included); F1 is then
  evaluated on the confirmation rows. Frequency-weighted over states.
  - `Cov30` = fraction of the 30 position facts with max_j F1 >= 0.9 over active codes.
  - `MonoFrac` = fraction of active codes whose best F1 over all 36 facts is >= 0.9.
  - `O2O` = Hungarian matching of active codes to the 30 position facts on the
    F1 matrix; number of matched pairs with F1 >= 0.9, divided by 30.
  - Also reported: trivial all-positive F1 per fact (prevalence floor) and the
    mean best F1 over position facts.
  - SAE arms additionally report Karvonen coverage as published: phi = I[f >
    t x f_max], f_max over development rows, alive = nonzero firing on
    development rows, t in {0, 0.1, ..., 1.0}; one global t (official code) and
    per-fact t (Eq. 3 as printed); over the 30 position facts and over all 36.
- **M2, MIG** (disentanglement_lib convention, Chen et al. 2018 Eq. 6): codes
  = posterior means (or SAE/AE codes, categorical probabilities), each
  discretised into 20 **equal-width** bins over its own range, plug-in MI in
  nats, H(v) = empirical plug-in entropy; MIG = mean over factors of (top-1 MI
  - top-2 MI) / H(v). Over the 5 block variables (`block-MIG`, used in H3.3)
  and over the 36 facts (`fact-MIG`, facts constant on the evaluation rows
  excluded and listed). Computed on the confirmation distribution, over active
  codes only; with a single active code the runner-up MI is taken as 0, and a
  model with no active code scores 0.
- **M3, DCI** (disentanglement_lib convention, Eastwood & Williams 2018): one
  sklearn `GradientBoostingClassifier` (defaults, `random_state=0`) per block
  variable, block values as classes, importance = `feature_importances_`;
  D = importance-weighted mean of per-code 1 - H_K; C reported both
  dlib-weighted and as the paper's plain average; informativeness = test
  accuracy and balanced accuracy. Raw states: fitted on the 866 states with
  development weights, tested with confirmation weights. Activations: only the
  `b_raw_w128` cells, 10,000 development / 5,000 confirmation positions (seed 0).
  Computed over active codes; with a single active code completeness is 1 by
  definition (entropy base 1 is undefined in the dlib formula).

**Deviations from the reference conventions (recorded before freeze).**
(1) Coverage for signed latents: Karvonen's t x f_max rule assumes nonnegative
features; for every family the primary M1 uses both signs and the
best development threshold over all distinct values (this dominates the
11-value grid, so it is generous to the Gaussian families). (2) MIG and DCI on
raw states use exact weighted expectations over the 866 distinct states
instead of 10,000 / 5,000 i.i.d. draws; with weights `None` the code reduces
exactly to disentanglement_lib. (3) DCI classifier seed fixed
(`random_state=0`; dlib leaves it unset), balanced accuracy and plain-average
completeness added. (4) The earlier draft of this protocol used equal-frequency
MIG bins and L1-logistic DCI importances; both were replaced by the
disentanglement_lib conventions above. Cross-check (2026-10-14, synthetic data,
`tests/crosscheck/`): pwlib MIG equals disentanglement_lib's `_compute_mig`
to 1.1e-16 and pwlib DCI equals `_compute_dci` with the seeded classifier to
3.3e-16 (6.7e-4 against unseeded dlib, tree tie-breaking).

## 5. Hypotheses (frozen) and forecasts

**H3.1 (demoted 2026-10-14 to a descriptive secondary analysis; half
implied by the AU counting of pw02).** For the reference beta-VAE (A2, beta=1,
lr 3e-4), report `MonoFrac` per gated seed and per beta=1 arch x lr cell, and
whether the former rule (MonoFrac < 0.5 in every gated seed of the reference
cell and seed median < 0.5 in >= 7 of 8 beta=1 cells) holds. Not a criterion.
*Forecast:* MonoFrac 0.0-0.3; any monosemantic unit codes `hand_empty`/parity
or a high-prevalence `on_table` fact.

**H3.2 (required; fact-centric, inductive-bias match).** The top-k SAE with
k=5 on raw states reaches `Cov30` >= 0.8 in every gated seed, and its seed
minimum exceeds the seed maximum `Cov30` of each Gaussian-latent cell at A2,
lr 3e-4 (beta-VAE beta {1, 4}, 5 seeds each; beta-TCVAE beta_TC {2, 6}, 3
seeds each). Gated = passing the pw02 competence gate.
*Forecast:* holds (about 70%); k=10 covers fewer facts than k=5 (feature
splitting/absorption with spare capacity).

**H3.3 (secondary; block level).** Block-MIG: categorical VAE > beta-VAE
(beta=1) and beta-TCVAE (beta_TC=6) > beta-VAE (beta=1), each with complete
separation of the seeds (exact one-sided permutation p = 1/56 for 3 vs 5 seeds).
*Forecast:* categorical > Gaussian (55%); TC > VAE (50%; Locatello et al.
found method differences often below seed variance).

**H3.4 (secondary; substrate b).** On `b_raw_w128`, the k=5 SAE's `Cov30` and
the beta-VAE's `MonoFrac` are lower than on raw states, and the SAE still
covers more position facts than the beta-VAE.
*Forecast:* SAE Cov30 on b 0.3-0.7; VAE MonoFrac on b about 0.

**Exploratory (labeled):** alignment vs AU across the 24 cells; absorption
patterns (codes firing for `clear(X)` together with `on(.,X)`); O2O and DCI
for every family.

## 6. Required criteria and outcome

`required_criteria`: `h32_topk_sae_covers_position_facts` (sole required
criterion since H3.1 was demoted). `supported` if it holds, else
`unsupported`. Secondary results do not compensate.

## 7. Gates

- **Task:** pw02 run complete with passing gates and audit; checkpoint hashes
  match the pw02 manifest; labels built from `pwlib.blocksworld`
  (`positions_from_state` round-trips the 866 states); 30/6 fact partition as stated.
- **Numerical:** confirmation F1 of 50 random (code, fact) pairs of the fact
  oracle, at the fitted sign and threshold, recomputed with
  `sklearn.metrics.f1_score` (max abs difference < 1e-9); MIG and DCI checked
  on the oracles and null below; no NaN in any F1 matrix; fail loudly on
  missing runs; pw02 files checked against the pw02 manifest hashes.
- **Competence / metric validity:** only pw02-gated runs are scored. Positive
  controls: the *fact oracle* (the 36 facts + N(0, 0.1^2) noise, padded with
  28 N(0,1) dims) must reach `Cov30` >= 0.95; the *block oracle* (the 5
  block-variable indices + N(0, 0.1^2) noise, padded with 27 N(0,1) dims) must
  reach block-MIG >= 0.5. (The earlier draft required block-MIG >= 0.5 from the
  fact oracle, which cannot reach it: a 6-valued block spread over six
  one-hot dims gives each dim at most H(indicator) of MI, so its MIG gap is
  small by construction.) Null control (randomly initialized A2 encoders,
  seeds 1-3, no training): `Cov30` <= 0.2. If any fails, the metric is invalid
  for this data: `competence_stop`.
- **Label-permutation null:** label rows permuted across evaluation rows
  (weights stay with the codes); 100 permutations on raw states, 20 on (b), for
  Cov30, MonoFrac, block-MIG and fact-MIG of every model; DCI null with 10
  permutations only for the controls and the reference cells (A2 lr 3e-4
  beta-VAE beta {1, 4}, beta-TCVAE, categorical VAE, SAEs, plain AE), for
  cost. Reported as excess over the null 95th percentile.

## 8. Splits, sampling, uncertainty, selection

- Development = val-split state weights (pw02 split) for signs, thresholds and
  DCI fits; confirmation = test-file weights (raw states) and the pw02
  300-sequence confirmation subsample (b); on (b) development = the pw02
  500-sequence val subsample. On raw states the support is the
  same 866 states in both, and each (code, fact) pair fits only a sign and a
  threshold, so overfitting is negligible; the independent unit is the
  training seed (1-3), as in pw02.
- Pilot: analysis of the pw02 pilot run (seed 101); fit on train-split
  weights (on (b): 20,000 train positions, seed 0), score on val weights; the
  test file is not opened.
- Uncertainty: per-seed values; seed min/median/max; family comparisons by
  complete seed separation; permutation null per metric.
- Selection: none. All gated runs of the listed cells are scored; thresholds
  of the criteria fixed above.

## 9. Compute and storage

No training. (b) codes are recomputed from retained pw02 checkpoints and the
pw02 activation cache. CPU, float32 models / float64 metrics, 8 workers.
Measured 2026-10-14 on the scratch smoke (189 models of a full-size sweep,
3 permutations instead of 100/20): 12.8 min wall, 6,100 CPU-s, dominated by
the gradient-boosted DCI fits (about 10 s per raw-state model, 2-4 min per
activation model with up to 256 codes). Scaled to the frozen permutation
counts: **about 4 CPU-h, about 35 min wall** for the full analysis; pilot a
few minutes. Retain: per-model F1/MI/importance matrices, thresholds and
signs, (b) codes, null summaries, results.json. Measured about 270 MB; cap
500 MB. (b) per-example codes are regenerable from pw02 checkpoints.

## 10. Scope

Alignment results are about these architectures, this encoding of the state
(36 facts) and this data distribution. A negative H3.1 does not show that no
VAE can align with facts; it shows that the standard isotropic-prior VAE in
this sweep does not.
