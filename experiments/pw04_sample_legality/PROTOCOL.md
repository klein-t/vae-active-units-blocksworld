# pw04_sample_legality — protocol (frozen 2026-10-21)

Proposal item **P4**. **No new training.** Analysis of the generative models
in the shared sweep trained and retained by `pw02_active_units`
(checkpoints pinned by path + SHA-256 from the pw02 manifests).

## 1. Question

What fraction of states decoded from **prior** samples are legal Blocksworld
states, per model family and beta, how far is that from simple factorized
baselines and from the model's own posterior, and is legality related to the
number of active units? Legality is decided exactly (pw01).

## 2. Legality and violation taxonomy

- A decoded 36-bit vector is legal iff `pwlib.blocksworld.is_legal` (pw01)
  returns true. Independent second check: membership in pw01's
  `legal_states_866.npy` table. The two must agree on every sample (numerical gate).
- Illegal samples are classified by every violated constraint: (V1) a block
  not in exactly one place; (V2) two blocks on the same block; (V3) hand
  inconsistency (more than one held block, or `hand_empty` disagreeing with
  holding); (V4) `clear` inconsistent with the rest; (V5) a cycle; (V6) a held
  block with a block on it (added before freeze: the pw01 checker rejects it
  and V1-V5 do not name it). Reported as any-violated and first-violated
  fractions. Implementation `src/pwlib/legality.py` (unit-tested: the six flags
  are all false exactly on the checker-legal rows).
- `prod5`-trained models emit no `clear`/`hand_empty` (constant-zero columns),
  so their samples are also scored by *position-bit legality*: each block in
  exactly one place in the 30 position bits, and the state with `clear` and
  `hand_empty` derived from them is legal.

## 3. Models and sampling

- Raw-state generative cells from pw02 (pw02-gated runs only): all 24
  `bw-gauss_vae-*` cells (5 seeds each); `bw-tc_vae-A2-b{2,6}`;
  `bw-cat_vae-A2-b1`; data controls `code10-*`, `prod5-*` (4 archs each),
  `indep-*` (3 seeds each).
- N = 20,000 prior samples per model, sampling seed = training seed + 1000.
  Gaussian: z ~ N(0, I) over all latent units. Categorical: each of the 16
  variables uniform over its 6 classes, hard one-hot.
- **Mode decoding (primary):** bit = 1 iff decoder Bernoulli mean > 0.5.
  **Stochastic decoding (secondary):** bits drawn from the Bernoulli means.
- **Aggregate-posterior reference** (per model): x drawn from the evaluation
  state distribution (20,000 draws), z ~ q(z|x), mode decoding.
- Distribution of legal samples (secondary): number of distinct legal states
  hit, and total-variation distance between the legal-sample distribution and
  the evaluation state distribution.
- Substrate (b), exploratory: `b_raw`/`b_lin` Gaussian-decoder cells and the
  `b_raw_w128` TC/categorical cells; decoded activations are read by the pw02
  current-state probe of that source (thresholded) and the read state is
  checked for legality; reference = the same probe on real confirmation activations.

## 4. Exact baselines (model-free)

- B0, uniform random 36 bits: 866 / 2^36 (about 1.3e-8).
- B1, independent bits with the Blocksworld train marginals: exact
  P(legal) = sum over the 866 legal states of the product of per-bit probabilities.
- B2, per-block product: each block's position drawn independently from its
  train marginal over its 6 values, `clear` and `hand_empty` derived; exact
  P(legal) by enumerating the 7,776 combinations (the uniform version is 11.1%,
  pw01). B2 knows the per-block coding but none of the cross-block constraints.
- Each exact baseline is cross-checked by 10^6-sample Monte Carlo (agreement
  within 3 standard errors).

## 5. Hypotheses (frozen) and forecasts

**H4.1 (required).** In every gated beta=1 Gaussian beta-VAE run on raw
states (4 archs x 2 lr x 5 seeds; gated = pw02 competence and the pw04
aggregate-posterior competence below), the mode-decoded prior-sample legality
rate exceeds B2. *Forecast:* legality 0.4-0.85 against B2 about 0.1-0.25.

**H4.2 (required; legality vs AU).** Within each beta stratum
({0.25, 1, 4}, a stratum counts if it has >= 8 gated runs), Spearman rho
between pw02 AU and prior legality across gated runs (8 arch x lr cells x 5
seeds = up to 40 runs per stratum). Pooled statistic = run-count-weighted mean of the stratum rhos; 95%
CI by cluster bootstrap (resample cells within strata, 2,000 replicates).
Supported if the upper CI bound is < 0. Rationale: with more active
dimensions, more prior mass falls outside the region covered by the aggregate
posterior ("prior holes"). *Forecast:* rho about -0.3 to -0.6 (about 60%).
Correlational: architecture and lr move both quantities; the claim is scoped
to association within the sweep.

**H4.3 (secondary; prior holes).** Aggregate-posterior legality minus prior
legality >= 0.05 in every gated beta=1 run. *Forecast:* holds (gap 0.1-0.5).

**H4.4 (secondary; family).** Categorical VAE vs beta-VAE (A2, beta=1)
legality, two-sided, complete seed separation. *No directional forecast:* the
uniform categorical prior puts mass on 6^16 code combinations never seen in training.

**H4.5 (secondary; controls).** (i) Fraction of Blocksworld-legal samples
from `prod5` VAEs is within 0.03 of the uniform per-block rate 0.111 (the
legality of Blocksworld VAEs then reflects learned cross-block constraints,
not the coding). (ii) `code10` VAEs: fraction of valid codewords (all copies of
each bit agree) reported as a constraint-learning reference. *Forecast:* (i)
holds; (ii) > Blocksworld legality.

**Exploratory (labeled):** stochastic-decoding legality; violation taxonomy
by family and beta; coverage/TV of legal samples; substrate (b) probe-read legality.

## 6. Required criteria and outcome

`required_criteria`: `h41_prior_legality_above_product_baseline`,
`h42_legality_decreases_with_au`. `supported` needs both; otherwise `mixed`
or `unsupported`. Secondary results do not compensate.

## 7. Gates

- **Task:** pw02 run complete with passing gates and audit; checkpoint hashes
  match; the legality checker marks all evaluation states legal, reproduces
  the 866-state table, and marks 10^5 uniform random vectors illegal.
- **Numerical:** checker and table agree on every sample; exact baselines
  match Monte Carlo within 3 SE; Wilson 95% intervals computed for every rate;
  fail loudly on a missing model.
- **Competence (per model):** aggregate-posterior legality >= 0.95 (a decoder
  that cannot reproduce legal states from its own posterior has not earned a
  prior-legality rate). For the data controls the same threshold applies to
  in-support rate (`code10`: valid codeword; `prod5`: one place per block and
  derived columns zero); `indep` has no support constraint and is descriptive.
  Experiment-level: every beta=1 cell has a majority of its planned seeds
  passing (>= 3 of 5), else `competence_stop`.

## 8. Splits, sampling, uncertainty, selection

- Evaluation state distribution: confirmation (test-file) weights in the full
  run; val weights in the pilot, which analyzes the pw02 pilot run (seed 101)
  and does not open the test file.
- Uncertainty: Wilson 95% interval per model (N = 20,000 gives about +-0.7
  points); per-seed values; H4.2 cluster bootstrap as above.
- Selection: none; all gated runs of the listed cells; N and thresholds fixed.

## 9. Compute and storage

No training. Measured 2026-10-14 on the scratch smoke (all 228 generative
runs of a full-size sweep, 8 workers): 27 s wall, about 3 CPU-min; the exact
baselines with 10^6-sample Monte Carlo and the bootstrap add under a minute.
Total < 10 CPU-min. Retain: per-model sampled bit vectors (packed, 20,000 x
36 bits), legality flags, violation codes, baseline computations,
results.json. Measured about 30 MB; cap 300 MB.

## 10. Scope

Legality rates are properties of these trained models and of mode decoding
from an isotropic prior; they say nothing about constrained decoders (e.g.
Grammar VAE) or post-hoc prior fitting (two-stage VAE), which are out of scope.
