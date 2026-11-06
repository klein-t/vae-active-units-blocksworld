# pw05_unit_anatomy — protocol (written 2026-10-27, before any pw05 quantity was computed)

**Post hoc, descriptive.** No training. The inputs are retained artifacts of
pw02a `full_001` (checkpoints, per-state posterior operands, AU sets),
pw03a `full_001` (per-unit x per-fact F1 tables) and the pw04a unregistered
descriptive pass `descriptive/results/pw02a_full_001` (packed mode-decoded
samples). Their confirmation data (thesis test-file weights, seeds 1-5 / 1-3)
were already opened, so no quantity below is a confirmatory test. Six
quantities Q1-Q6 (plus Q7, added before the freeze) are declared here; there are no pass/fail hypotheses. The
required criteria `q1_computed` ... `q7_computed` are true iff the quantity was
computed for every planned run; outcome `supported` would mean only that.
Kind `analysis`, `pilot_required: false` (nothing trained). Before the freeze
the code is smoke-tested only on development artifacts (pw02a/pw03a/pw04a
`pilot_001`, seed 101, val weights) in a scratch directory.

**Conventions.** Evaluation distribution = confirmation weights `conf_weights`
stored by pw02a (Blocksworld: thesis test-file state frequencies over the 866
legal states; code10 / prod5: uniform over 1,024 / 7,776 states). Information
in bits unless stated (nats / ln 2). Units: active iff Burda Var_x(mu_j) > 0.01
under these weights (pw02a `au_per_run.json`). Per cell: median and range
over seeds. Missing file, key, run or mismatched hash = technical stop (fail
loudly). Inputs are checked against the pw02a and pw03a `MANIFEST.json` and
against `input_hashes.json` (pw04a descriptive samples, which have no manifest).

**Planned runs.** R1 = `bw-gauss_vae-{A1,A2,A4}-b1-lr{3e-4,3e-3}` x seeds 1-5
(30); C = `{code10,prod5}-gauss_vae-{A1,A2,A4}-b1-lr3e-4` x seeds 1-3 (18);
R025 = the 30 beta=0.25 Blocksworld runs; R4 = the 30 beta=4 Blocksworld runs
(observed only, they fail pw02a competence); Bl = the 30 `b_lin` runs
(5 sources x {fixed, learned gamma} x 3 seeds).

## Quantities

- **Q1 Rate (R1 + C).** Per-unit KL_j = sum_x w(x) KL(q(z_j|x) || N(0,1)) from
  the stored `conf_kl_per_unit`; total rate R = sum_j KL_j; also the KL in
  active and in inactive units. H(X) = entropy of the weights. Mismatch bound
  (Hoffman & Johnson decomposition, E_x KL(q(z|x)||p) = I(x;z) + KL(q(z)||p),
  I <= H(X)): **bound = R - H(X)** <= KL(q(z)||p).
- **Q2 ELBO by cell (R1 + C).** Distortion D = sum_x w(x) E_{eps}[-log p(x|z)],
  z = mu + exp(logvar/2) eps, eps ~ N(0, I), S = 128 draws per state from
  `torch.Generator().manual_seed(0)` (float32 forward, float64 sums), Bernoulli
  NLL summed over the 36 bits; MC standard error from the per-draw losses.
  Report D, R, -ELBO = D + R and the slack (D + R) - H(X) (nats and bits).
  Highlight A2 lr 3e-3 vs A2 lr 3e-4.
- **Q3 Unit anatomy (R1 + prod5 runs of C).** Codes = stored `conf_mu` on the
  table rows with w > 0. For each active unit and each of the 5 block
  variables (`pwlib.data.block_variables`, 6 values each): weighted plug-in MI
  after discretizing the unit into **20 equal-width bins over its range**
  (`pwlib.metrics.histogram_discretize`, the pw03a / disentanglement_lib
  convention), in bits, and NMI = MI / H(block). Assign the unit to its
  max-MI block (ties: lowest index; count ties) and count units per block.
  F1: Blocksworld runs reuse pw03a `scores_<run>.npz` (`f1_conf`, `sign`);
  prod5 runs recompute them with the pw03a scorer `pwlib.metrics.best_f1`
  (dev codes = `dev_mu`, dev weights; scored on `conf_mu`, conf weights).
  Per active unit: facts with F1 >= 0.9 (count over 36; classes 0, 1, 2, >= 3),
  their signs, and for 2-fact units: same block / different block / involves a
  derived fact (clear, hand_empty; block of a position fact = its subject) and
  opposite vs same sign, and same support (both facts `on(x,y)`, `on(x',y)` with
  the same y; tag added before the freeze after the development smoke test
  showed such pairs); whether the detected position facts' block equals
  the max-MI block. Null: block labels permuted across rows (20 permutations,
  seed 0), 95th percentile of the per-unit max MI.
- **Q4 Linear-VAE prediction on b_lin.** For each source (w64, w128, w128s2,
  w128s3, w256): eigenvalues of the population covariance of `train_lin`
  (pw02a preprocessing: standardized activations projected by the stored
  least-squares map; float64, `numpy.linalg.eigh`). Count lambda > beta*gamma
  with beta = 1 and gamma = 1 and gamma = e^-9 (log-gamma floor); also with each
  learned run's final gamma. Compare with observed AU (pw02a, conf rows) of the
  fixed- and learned-gamma runs. Heuristic: the threshold rule is exact only
  for linear VAEs (Lucas et al. 2019; Wang & Ziyin 2022); here the MLP is A2.
- **Q5 Zero-rate comparison (R1 + C).** D0 = sum over the 36 facts of the
  Bernoulli entropy of their marginals under w (nats; constant columns 0).
  beta* = (D0 - D1) / R1 with D1 = Q2 distortion and R1 = Q1 rate (nats): for
  beta > beta*, the zero-rate solution has lower beta-objective than that beta=1
  solution. Heuristic (ignores re-optimisation at the new beta and the
  train-vs-conf weight difference). Compared with observed AU and rate of R4.
- **Q6 Legality anatomy (R1, R025, prod5 runs of C).** Inputs: pw04a
  `prior_mode_bits` and `aggpost_mode_bits` (20,000 each, sampling seeds
  training seed + 1000 / + 2000, frozen pw04a sampler; unpacked to 36 bits).
  Block x is well formed (WF_x) iff exactly one of its 6 position bits
  (`on_table(x)`, `holding(x)`, `on(x,y)`) is 1. Report WF_x per block, WF_all;
  among WF_all, the fraction whose block positions form a legal state
  (cross-block: `pwlib.legality.legal_from_position_bits`); among those, the
  fraction whose 6 derived bits (clear, hand_empty) also match (= full
  legality, checked equal to the stored `*_legal` flags). Independence
  prediction: p_x(v) = fraction of samples with block x well formed at value v;
  implied WF_all = prod_x WF_x; implied position legality =
  sum over the 866 legal position combinations of prod_x p_x(c_x)
  (`pwlib.legality.baseline_b2` on the unnormalised p); implied cross-block
  rate given WF = their ratio. V1-V6 rates over all samples
  (`pwlib.legality.violations`). Controls: the 866 data states under w
  (positive: all rates 1; implied = B2 with test marginals); prod5 runs
  (trained without cross-block constraints).
- **Q7 Asperti & Trentin count (R1 + C).** Added 2026-10-27 after the
  registration commit and before the freeze, on request;
  an informal audit scratch check of this count (8-11 units at A2 lr 3e-3,
  13-16 elsewhere) had been seen. Criterion of arXiv 2002.07514, footnote 3:
  unit j is inactive iff E_x sigma_j^2(x) = sum_x w(x) exp(conf_logvar_j(x))
  >= 0.8; otherwise active. Report the count next to the Burda count, and the
  size of the symmetric difference between the two active sets.

## Gates

- **task:** every input present and hash-matched; every planned run present;
  legal table equals the brute-force enumeration.
- **numerical:** stored `conf_kl_per_unit` equals recomputation from
  `conf_mu`/`conf_logvar` (rel. 1e-6); D + R >= H(X) - 3 MC SE in nats for every
  run (Gibbs + ELBO bound); `best_f1` reproduces pw03a `f1_conf` on all 30 R1
  runs (max abs diff <= 1e-12; positive control for prod5); own MI reproduces
  pw03a `block_mi` (nats) on R1 (abs <= 1e-9); eigenvalue sum equals covariance
  trace (rel. 1e-8); Q6 funnel product equals the stored legality rate exactly
  and V1 rate = 1 - WF_all.
- **competence:** every R1, C, R025 and Bl run has pw02a status ok and
  competence true (R4 excluded by design: observed only).

Failure of task/numerical = technical stop, competence = competence stop; no
retries under this freeze. Audit: `audit.py` recomputes the headline numbers by
separate code paths (rate from mu/logvar, H(X), D with a different MC seed,
MI by explicit joint tables, F1 counts from the stored matrices, eigenvalues by
SVD, legality funnel by per-sample `is_legal`, independence prediction by Monte
Carlo, one model's prior samples regenerated with the frozen pw04a sampler and
compared byte for byte). Same author and session: not an independent replication.
Compute: CPU, float32 models / float64 sums, 8 workers, < 1 CPU-h.
