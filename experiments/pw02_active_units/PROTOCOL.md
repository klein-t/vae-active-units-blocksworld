# pw02_active_units — protocol (frozen 2026-10-21)

Proposal item **P2**. This experiment also **owns the shared training sweep**
(`sweep_grid.json`) whose retained checkpoints and evaluation operands are
reused, without retraining, by `pw03_fact_alignment` (P3) and
`pw04_sample_legality` (P4). Nothing is trained twice.

## 1. Question and competing accounts

Does the number of active latent variables (AU) of an overcomplete VAE track a
known "true number" for Blocksworld, and how much does it move with
architecture, beta and learning rate? On discrete data there is no manifold
dimension, and one continuous latent can in principle index all 866 legal
states, so we do not ask "does AU equal the dimension". We ask **which of
several pre-fixed targets the count behaves like**, and whether it is stable.

- Information account: AU follows how many bits the data carry (log2 866 = 9.76).
- Factor account: AU follows the number of independent generative factors.
- Linear account: AU follows the affine (PCA) dimension of the data (25).
- Architecture/training account (Asperti 1812.07238; Asperti & Trentin
  2002.07514): AU is an architecture- and training-relative quantity that
  moves with depth/width, beta and lr more than with any data target.
- Overshoot (Koehler et al. 2112.06868): AU exceeds the relevant target.

## 2. Targets, fixed before any VAE is trained

Read from the completed pw01 run `analysis_001`:
`experiments/pw01_state_space_ground_truth/results/analysis_001/dimension_candidates.json`
(SHA-256 `07f38203...655b1`, full hash in `DESIGN.json`). The run fails
loudly if a key is missing or the hash differs.

| Symbol | pw01 key | Value |
|---|---|---|
| K_bits | `bits_log2_legal` / `bits_min_binary_code` | 9.76 / 10 |
| K_H | `bits_entropy_empirical` | 9.35 |
| K_blk | `factors_per_block_categorical` | 5 (6-valued, dependent) |
| K_aff | `affine_dim_legal_states` | 25 |
| K_facts | `binary_facts_total` | 36 |
| (K_I3D) | `i3d_hypercube_range` | 11.9-21.3; exploratory, scale-dependent, in no criterion |

Substrate (b) adds two activation-side targets whose **definitions** are frozen
here and whose **values** are computed in step 0 of the run, written to
`targets_b.json`, before any VAE is trained (the script enforces the order):

- K_pca99(w): number of principal components explaining 99% of the variance of
  standardized train activations of source w.
- K_probe(w): smallest q such that 36 logistic probes (sklearn, C=1,
  max_iter=2000) fitted on the top-q PCs reach val bitwise accuracy within
  0.005 of the same probes fitted on all w dimensions; q scanned in increasing
  order over 1..40 and {48, 56, 64, 96, 128, 192, 256} up to w (stop at the
  first q that qualifies). All probes in this experiment (targets, source gate,
  competence probe) are fitted on a fixed subsample of 20,000 train positions
  (seed 0), for cost.

For the data controls (section 4) the targets are known by construction.

## 3. Substrates and splits

**Split (by sequence, fixed seed 0):** the thesis train file (16,004 sequences)
is permuted and cut into train (12,803) and val (3,201). The thesis test file
(4,001 sequences) is the sealed **confirmation** set: not read by the pilot,
not used for any selection. The documented "length > 30 = OOD" rule is not
used (train sequences have 40-60 actions, test 14-60; pw01). A position k of sequence i is
valid iff `X[i,k] != 52` (PAD); its state is `Y[i,k]` (state after k actions).

**(a) Raw states, Bernoulli decoder.** Inputs are the 36-bit states. Because
the encoder is a function of x only, training on positions equals training on
the 866 distinct states weighted by their frequencies: minibatches are drawn
i.i.d. from the train-split empirical state distribution. Evaluation
quantities are exact frequency-weighted expectations over the 866 states
(val weights in the pilot, confirmation weights in the full run). For (a) the
splits share the same 866-state support, so the sealed set protects little;
the real confirmation unit on (a) is the independent training seed. Stated, not hidden.

**(b) Transformer activations, Gaussian decoder.** Sources: five thesis exp46
checkpoints, `d4_w64_s1`, `d4_w128_s1`, `d4_w128_s2`, `d4_w128_s3`,
`d4_w256_s1` (short names `w64`, `w128`, `w128s2`, `w128s3`, `w256`; paths +
SHA-256 in the freeze). Width 128 is replicated over three transformer seeds;
widths 64 and 256 use seed 1 only. Loaded with the verified recipe
(TransformerLens HookedTransformer, pinned `transformer-lens==2.15.4` as in
the thesis, since 4.x removed HookedTransformer; n_heads=4, d_head=W/4, d_mlp=4W,
d_vocab=53, n_ctx=60, input `X[:, :60]`, hook `blocks.{D-1}.hook_resid_post`
= `blocks.3.hook_resid_post`). Recipe re-verified 2026-10-14 on 40 train-file
sequences: all five load with no missing keys, and the exp46 head agrees
better with `Y[k+1]` than with `Y[k]` for every checkpoint (one-step-ahead
convention confirmed). The exp46
head is trained one step ahead; we pair the activation at position k with the
**current** state `Y[k]`. Positions whose next-state target `Y[k+1]` is the
all-zero PAD row (the last valid position of sequences shorter than 60
actions) are excluded, because the head was trained to output zeros there.
Subsamples (fixed generator seeds 0 / 1 / 2 for train / val /
confirmation): 3,000 train sequences, 500 val sequences, 300 confirmation
sequences. Activations are standardized per dimension with train
statistics. Two substrates per source:

- `b_raw`: the standardized activations (carry state, step parity, position, history).
- `b_lin`: the part of `b_raw` linearly explained by the current state:
  least-squares fit of `b_raw` on `[1, Y[k]]` (train positions, min-norm
  `lstsq`), evaluated as `[1, Y[k]] B`. It takes at most 866 distinct values
  lying in an affine space of dimension <= 25: a linear embedding of the raw
  states into activation space.

The (a) vs `b_lin` vs `b_raw` contrast separates decoder likelihood
(Bernoulli vs Gaussian with learnable variance) from non-state content.

## 4. Shared sweep (frozen grid: `sweep_grid.json`)

66 cells, 246 confirmation runs; pilot seed 101 on 23 pilot cells. Seeds per
cell (listed in `sweep_grid.json`): **5 seeds (1-5) for the 24 raw-state
`bw-gauss_vae` cells** (the core of H2.2/H2.3 and the comparators of pw03 H3.2
and pw04 H4.1/H4.2), **3 seeds (1-3) for every other cell** (data controls,
secondary families, activation cells). Each run's RNG seed is derived from
`sha256(cell_id|seed)` (init, minibatch order, reparameterization noise), so
runs with the same seed label are independent across cells. MLP encoders/decoders (inputs are not spatial, so "dense" MLPs are
the only architecture class; no convolutions): A1 = 1x64, A2 = 2x256
(reference), A3 = 4x256, A4 = 2x1024, ReLU. Adam, 30,000 steps, batch 512,
constant lr, no KL warm-up (unless the pre-registered A3 contingency of
section 10 fires), grad-norm clip 10, logvar clamp [-12, 6], log
gamma floor -9. Latent size 32 on (a)-type data, 64 on (b).

- **(a) beta-VAE factorial:** arch {A1..A4} x beta {0.25, 1, 4} x lr {3e-4, 3e-3} = 24 cells.
- **(a) data controls, beta=1, lr 3e-4:** `code10` x 4 archs, `prod5` x 4 archs, `indep` x A2 (9 cells).
  - `code10`: 10 independent fair bits, each copied into 3-4 fixed positions of
    36 (assignment seed 0). 1,024 states; 10 bits; 10 independent factors; affine dim 10.
  - `prod5`: 5 independent uniform 6-valued variables, one-hot (30 bits) + 6
    constant-zero bits. 7,776 states; 12.92 bits; 5 factors; affine dim 25.
    Same per-block coding as Blocksworld (whose legal states are 11.1% of this product).
  - `indep`: 36 independent Bernoulli bits with the Blocksworld train marginals
    (no structure; negative control; count may saturate the 32 latents).
- **(a) other families (consumed by pw03/pw04):** beta-TCVAE beta_TC {2, 6};
  categorical VAE 16x6 Gumbel-softmax; top-k SAE dict 64, k {5, 10}; plain AE.
- **(b):** per source in {w64, w128, w128s2, w128s3, w256}: {`b_raw`, `b_lin`} x gamma
  {learned, fixed=1}, A2, beta=1 (20 cells); `b_raw_w128` beta {0.25, 4}
  with learned gamma; `b_raw_w128` beta-TCVAE (6), categorical VAE, top-k SAE
  (dict 256, k {5, 10}), plain AE.
- **Deterministic baselines (not cells):** PCA counts on every substrate.

Family details are in `sweep_grid.json` (`families`). Each run saves the
final checkpoint (fixed step 30,000; no early stopping) and, on the
evaluation set, the per-example operands: posterior means and log-variances
(Gaussian), class probabilities (categorical), codes (SAE/AE), decoder
outputs, and per-unit KL. On (a) these are 866-row arrays plus weights. On (b)
the decoder outputs are retained as the competence probe's reading of them
(36 bits per position), not as raw activations (disk).

## 5. Metrics

Implementation: `src/pwlib/metrics.py`, following `docs/METRICS_SPEC.md`
(binding). Objectives use the disentanglement_lib convention (reconstruction
NLL summed over inputs, KL summed over latents, unnormalised beta); every run
log records beta and beta_norm = beta x latent / input size.

- **AU (primary):** unit j active iff Var_x(E_q[z_j | x]) > 0.01 (Burda et
  al.), x drawn from the evaluation distribution (exact weighted variance on
  (a); the 300-sequence confirmation positions on (b)).
- Robustness (reported for every run): AU at thresholds {0.001, 0.05, 0.1};
  per-unit mean KL; per-unit A_u values (log-A_u histogram, the bimodality
  that justifies Burda's threshold); Asperti's diagnostic Var(mu_j) +
  E[sigma_j^2] (Eq. 13, about 1 when the aggregate posterior matches the prior
  per unit; a unit is flagged *not converged* if |sum - 1| > 0.1); Asperti's
  reconstruction gain per unit (NLL with mu_j zeroed minus NLL); polarization
  ratio = smallest active Var(mu) / largest inactive Var(mu) (a count is called
  *polarized* if the ratio >= 10).
- Total correlation of the aggregate posterior (all Gaussian runs, 8 batches of
  512 from the evaluation distribution): MWS with Chen's official-code
  constants, as fixed in `docs/METRICS_SPEC.md` section 4, and MSS (Chen
  Appendix C.2). Unit tests on Gaussians show that the Chen-constant MWS value
  equals the true TC plus (D-1) log N when posteriors overlap (the log(NM)
  constant over-subtracts log N per density), so MSS is the value read as nats;
  both are reported, neither enters a criterion.
- Categorical VAE: variable j active iff sum_c Var_x(q_j(c|x)) > 0.01 (secondary; not pooled with Gaussian AU).
- SAE / plain AE: alive codes (fire on >= 0.5% of weighted evaluation
  examples); reported only, not an AU. Plain AE is the AU-metric control: with
  no KL term the covariance criterion should mark nearly all 32 units active.

## 6. Hypotheses (frozen) and forecasts

Each hypothesis names its target. **Forecasts** are the designer's expectations
written before any training, labeled as such; they are not criteria.

**H2.2 (required, central; substrate a; targets K_aff, K_bits vs K_blk vs K_aff).
Merges former H2.1, which it nearly implied.** Two parts, both required.
(i) Every competence-gated beta=1 Bernoulli beta-VAE on Blocksworld (4 archs x
2 lr x 5 seeds) has AU < K_aff = 25 (hence < 36) and is not censored.
(ii) Let A_BW, A_code, A_prod be seed medians of AU at beta=1, lr 3e-4 for one
architecture (gated seeds), and delta = 2 units. The three accounts predict:

| Account | Prediction |
|---|---|
| Information (K_bits: 9.76 / 10 / 12.9) | abs(A_BW - A_code) <= delta and A_prod > A_BW + delta |
| Factor count (K_blk: 5 / 10 / 5) | abs(A_BW - A_prod) <= delta and A_code > A_BW + delta |
| Affine dim (K_aff: 25 / 10 / 25) | abs(A_BW - A_prod) <= delta and A_code < A_BW - delta |

Each architecture gets one verdict (or "none"). **H2.2 is supported if (i)
holds and the information account is the verdict in >= 3 of 4 architectures.**
*Forecast:* (i) holds, median 8-12; information account in 3-4 archs (about
55%); mixed/none (30%); another account (15%). AU(BW) forecast 7-13 at beta=1.

**H2.3 (required, substrate a; architecture/training dependence).** At beta=1,
the range of seed-median AU across the 8 arch x lr cells is >= 3 units AND
larger than the largest within-cell seed range. With 5 seeds per cell the
within-cell range is a stricter yardstick than with 3 (expected range of 5 vs
3 normal draws: 2.33 vs 1.69 sd); the criterion is kept unchanged. *Secondary:*
AU is non-increasing in beta (0.25 -> 1 -> 4) in seed medians in >= 7 of 8 arch
x lr cells (all technically completed runs, competence reported).
*Forecast:* both hold (range 3-6 units; lr effect present, direction not forecast).

**H2.4 (required, substrate b; Gaussian decoder, Dai & Wipf / Zheng et al. contrast).**
With a Gaussian decoder whose variance gamma is learned, gamma can shrink and
the effective reconstruction weight grows (a Gaussian decoder with fixed gamma
is a beta-VAE with beta proportional to gamma); the Bernoulli decoder on (a)
has no such lever. The gamma -> 0 theory is stated for continuous data on a
manifold; `b_lin` is a finite point set in a 25-dim affine space, so the theory
motivates the contrast but gives no numeric prediction here.
(i) On `b_lin`, seed-median AU with learned gamma exceeds fixed gamma by >= 2
units at each of the 5 source checkpoints, with non-overlapping seed ranges.
*Secondary:* (ii) abs(AU(b_lin, learned) - 25) < abs(AU(a, A2, beta=1) - 25);
(iii) AU(b_raw, learned) > AU(b_lin, learned) at each source, and the
direction of change across source widths 64 -> 128 -> 256 (state-driven count
should be flat; geometry-driven count should rise); (iv) b_raw AU compared to
K_pca99(w) and K_probe(w); (v) spread of seed-median AU over the three width-128
transformer seeds (`b_lin` and `b_raw`, learned gamma), descriptive.
*Forecast:* (i) holds; learned gamma hits or approaches the floor on `b_lin`;
AU(b_lin, learned) 12-25; (iii) b_raw > b_lin, rising with width (60%).

## 7. Required criteria and outcome

`required_criteria`: `h22_information_account` (including former H2.1),
`h23_architecture_dependence`, `h24_learned_gamma_more_units`. `supported`
requires all three; otherwise `mixed` or `unsupported` with each hypothesis's
verdict reported separately. Secondary results never compensate for a failed
required criterion. Exploratory analyses (lr direction, TC/categorical AU,
KL-per-unit shapes) are labeled exploratory.

## 8. Gates (prerequisites to efficacy)

- **Task:** data loader reproduces pw01 (866 distinct train states, all legal
  under `pwlib.blocksworld.is_legal`; PAD excluded; position 0 is SOS;
  sequence splits disjoint); control datasets have the stated support sizes
  and entropies (1,024 / 7,776 states; 10 / 12.92 bits); source competence
  for each (b) source: fresh current-state probe val bitwise >= 0.99 and
  exact-state >= 0.80, and the current-state probe beats a next-state probe
  (alignment check).
- **Numerical:** all losses finite; per-unit KL >= -1e-6; PCA count of the
  866 legal states equals 25 (pw01 reproduction); `b_lin` has <= 866 distinct
  rows and affine dim <= 25; AU recomputed from stored operands by a separate
  audit script matches exactly; log gamma at the floor is flagged per run.
- **Competence (per run, on the development evaluation set; a run that fails
  is excluded from criteria and reported):** (a)-type data: frequency-weighted exact-state reconstruction
  from the posterior mean (threshold 0.5) >= 0.95 and bitwise >= 0.99;
  categorical VAE same with argmax codes; SAE/AE same on thresholded
  reconstructions. (b): a current-state probe fitted on original activations
  and applied to reconstructions gives bitwise >= 0.98, and reconstruction
  R^2 >= 0.90. Fixed-gamma (b) cells are a *contrast condition* in H2.4(i):
  their AU is compared regardless of their competence (reported), because the
  hypothesis is about the objective; learned-gamma cells must pass.
- **Experiment-level competence:** every cell entering a required criterion
  has a majority of its planned seeds passing (>= 3 of 5 for the core cells,
  >= 2 of 3 otherwise); else `competence_stop` for the run (outcome untested),
  with the failing cells listed. Medians in the criteria use passing seeds only.
- **Retained for reuse:** per-source current-state probe weights, `b_lin` maps and standardization statistics (used by pw03/pw04).
- **Overcompleteness:** a run with AU equal to its latent size is reported as
  censored (">= latent"); a censored Blocksworld beta=1 run fails H2.2(i) by definition.

## 9. Sampling, uncertainty, selection, stopping

- Unit: independent training seed (init, minibatch order, reparameterization
  noise). Seeds 1-5 (core `bw-gauss_vae` cells) or 1-3 (all other cells)
  confirmation, 101 pilot. Five or three seeds support only coarse claims, so
  comparisons use a fixed tolerance (delta = 2 units) plus non-overlap of seed
  ranges, not p-values; for reference, complete separation has an exact
  one-sided permutation p of 1/252 (5 vs 5), 1/56 (5 vs 3), 1/20 (3 vs 3).
  All per-seed counts are reported.
  Hyperparameter dependence is also summarized by a descriptive two-way
  decomposition of AU variance (arch, lr, residual = seed).
- Selection: none. Every cell's final checkpoint is analyzed; no early
  stopping, no best-seed choice, no threshold tuning (thresholds fixed above).
- Stopping/retry: a run that crashes or produces NaN is re-run once with
  the same seed and config, automatically and logged (CPU training is
  deterministic, so a numerical failure normally repeats); a second failure is a technical stop for
  that run. Any change to the grid, step count or gates after freeze requires
  an amendment directory (e.g. `pw02a_...`).

## 10. Pilot (development data only)

Seed 101, the 23 `pilot_cells` (Blocksworld beta-VAE A2 at both lrs and three betas, A1/A3/A4 at beta=1 lr 3e-4, code10/prod5 at A2 and A3, plus one cell per other family/substrate; A1/A3 added before freeze so that every architecture is exercised, after a 30,000-step synthetic-control check showed A3 collapsing on code10/prod5 at seed 101), train/val splits only (confirmation file not
opened; for (b) the source competence check also uses val). Checks: every gate
is computable; convergence (val objective changes < 2% over the last 20% of
steps) for both lrs and all four architectures in the pilot; AU < latent size
on Blocksworld; wall-clock per run; log gamma behavior. The pilot cannot change
thresholds or hypotheses; if convergence fails, amend (more steps) before the full run.

**Pre-registered A3 contingency (added 2026-10-21, before freeze and before
any real outcome).** Trigger: in `pilot_001`, any A3 (4x256) cell on
raw-state-type data (`bw-gauss_vae-A3-b1-lr0.0003`,
`code10-gauss_vae-A3-b1-lr0.0003`, `prod5-gauss_vae-A3-b1-lr0.0003`) fails
the per-run competence gate (section 8). Effect: a linear KL warm-up is
applied to **all** cells and all architectures, not just A3: the KL-side
penalty is multiplied by min(1, t / 3,000), i.e. beta rises linearly from 0
to its target over the first 10% of the 30,000 steps (for beta-TCVAE the
whole MI + beta TC + DWKL term is scaled; SAE and plain AE have no KL term).
Everything else (grid, steps, thresholds, hypotheses, analyses) is
unchanged. Because the workflow requires the full run to cite a pilot whose
gates passed and whose hyperparameters were exercised, the warm-up is first
run as `pilot_002` (seed 101, the same 23 pilot cells, development data
only); the full run cites `pilot_002`. If `pilot_002` fails any gate, the run
stops and is reported (no further amendment within pw02). Implementation:
`sweep_grid.json` `training.kl_warmup` (default `null` = off) and
`training.kl_warmup_contingency`; `train_sweep.py
--apply-kl-warmup-contingency pilot_001` re-checks the trigger from the pilot's
run files and refuses if it did not fire. If triggered, it is disclosed as a
pilot-triggered amendment in every report of pw02-pw04. If the trigger does
not fire, the full run cites `pilot_001` and trains without warm-up.

## 11. Order of execution across pw02-pw04

1. pw01 completed; its `dimension_candidates.json` hashed. 2. Freeze pw02,
pw03 and pw04 (all three before any training). 3. pw02 pilot (seed 101).
4. pw03 and pw04 pilots on pw02 pilot checkpoints. 5. pw02 full run (seeds
1-3, confirmation opened). 6. pw03 and pw04 full analyses on the pw02 full run.

## 12. Compute and storage

CPU, float32, one thread per worker, 8 parallel workers (machine: 14 cores,
10 performance + 4 efficiency), longest runs scheduled first.
**Measured 2026-10-14** on the scratch smoke (real code, real exp46
checkpoints, synthetic random-walk sequences of the thesis shapes): all 246
runs at 1,000 steps under the 8-worker load took 347 s wall and 2,006
worker-seconds of training. Per-step cost under that load (ms/step): A1 1,
A2 2, A3 4, A4 21, beta-TCVAE 59 (raw) / 125 (activations, 64 latents),
categorical 3-4, SAE 1-2, activation beta-VAE 3. The parallel load slows the
memory-bound A4 and TC runs about 3x and 2x relative to a lone single-thread
run (6.7 and 35 ms/step). Scaled to 30,000 steps:
- **Full sweep: about 17 CPU-h (worker-hours), about 2.5-3 h wall-clock** on 8
  workers. Largest shares: A4 beta-VAEs 6.2 h, beta-TCVAE 6.1 h (9 runs; the
  minibatch-weighted estimator builds a 512 x 512 x D tensor every step),
  activation beta-VAEs 1.7 h, A3 1.3 h. Longest single run about 62 min
  (activation beta-TCVAE).
- Step 0b (activation extraction, probes, targets): about 30 s wall (5
  processes). Analysis and audit: under 1 min.
- **Pilot (23 cells, seed 101): about 1 CPU-h, about 30-35 min wall** (bounded
  by the beta-TCVAE run).
- The original estimate (3.5 CPU-h, 45 min) came from a single-thread
  random-tensor benchmark that omitted the TC estimator and parallel
  contention; the original 174-run grid measures about 13 CPU-h under the same
  conditions, so the seed/transformer amendment adds about 3.5 CPU-h (+26%).
- Budget cap: 30 CPU-h, 5 h wall.
Disk (measured on the smoke run): about 2.6 GB per full run (raw/ 2.1 GB:
activation caches about 1.15 GB plus activation-substrate operands; models/
0.5 GB, A4 checkpoints dominate); pilot about 1.1 GB. Expected 2.8
GB, cap 6 GB. All gitignored under `results/<run>/models` and `results/<run>/raw`.

## 13. Scope and limitations (stated in advance)

- AU is reported as an architecture- and training-relative quantity
  (Asperti), never as "the true dimension" of Blocksworld.
- (b) uses depth-4 transformers: three seeds at width 128, one seed at widths
  64 and 256. The source model is a fixed substrate; claims about (b) are
  scoped to these five checkpoints, and width effects rest on one seed per
  width outside 128.
- The beta=4 cells may fail competence; their AU is descriptive only.
- The log-gamma floor is a fixed choice; if gamma sits at the floor, AU on
  `b_lin` partly reflects the floor (sensitivity not explored).
- `transformer-lens==2.15.4` (with `transformers<5`) is a pinned project
  dependency; `pyproject.toml` and `uv.lock` are frozen with the code.
