# pw05_unit_anatomy — rate, ELBO terms, unit anatomy and legality anatomy of the existing sweep (post hoc, descriptive) — **COMPLETED: DESCRIPTIVE, ALL QUANTITIES COMPUTED** (2026-10-27, run `analysis_001`)

- **Proposal items:** P2 (why 15–16 active units), P3 (what each unit encodes), P4 (why prior samples are illegal).
- **Trigger:** two examiner-style reviews of the report asked why the β-VAE keeps 15–16 active units, what its rate is, and why prior samples are illegal.
- **Nature:** **post hoc and descriptive.** No training. Pure analysis of retained artifacts of pw02a `full_001`, pw03a `full_001` and the pw04a unregistered descriptive pass (`descriptive/results/pw02a_full_001`). The confirmation data were already opened by those runs, so nothing here is a confirmatory test: seven pre-declared quantities (Q1–Q7), no pass/fail hypotheses. **The outcome field `supported` means only that the required completeness criteria `q1_computed` … `q7_computed` hold** (every quantity computed for every planned run); it supports no hypothesis.
- **Frozen method:** `PROTOCOL.md`, `DESIGN.json` (freeze record sha256 `3e1e0c1a…`), `analyze_anatomy.py`, `audit.py`, `input_hashes.json` (sha256 of the 69 pw04a descriptive sample files, which have no manifest). Kind `analysis`, `pilot_required: false`.
- **Pre-freeze changes (disclosed):** the protocol was committed (`01eb241`) before any quantity was computed. Afterwards, before the freeze: (1) **Q7** (Asperti & Trentin count) added on request; an informal audit scratch count of it (8–11 at A2 lr 3e-3, 13–16 elsewhere) had been seen; (2) the Q3 pair tag `same_support` added after the development smoke test. The code was smoke-tested only on development artifacts (pw02a/pw03a/pw04a `pilot_001`, seed 101, val weights) in a scratch directory.
- **Run:** `analysis_001`, status `completed`, gates task / numerical / competence all true; CPU, 8 workers, 1 thread each, float32 models with float64 sums; 18 s wall (analysis), 28 s (audit). One attempt, no retries.

## Gates and completeness criteria

| Item | Value | Pass |
|------|-------|------|
| task: inputs hash-matched (pw02a/pw03a manifests, `input_hashes.json`), all planned runs present (R1 30, code10 9, prod5 9, β=0.25 30, β=4 30, `b_lin` 30), legal table reproduced | all | yes |
| numerical: stored per-unit KL = recomputation (rel ≤ 1e-6); D + R ≥ H(X) − 3 SE every run; `best_f1` reproduces pw03a `f1_conf` (30/30, max diff 0); own MI reproduces pw03a `block_mi` (30/30, max diff 0); eigenvalue sum = trace; Q6 funnel subsets, V1 = not well formed, product = stored legality | all | yes |
| competence: every R1, code10, prod5, β=0.25 and `b_lin` run has pw02a competence (β=4 observed only) | all | yes |
| `q1_computed` … `q7_computed` | 48 / 48 / 39 / 5 sources / 48 + 30 / 69 / 48 runs | yes |

## Results (per cell: median [min, max] over seeds; conf weights = thesis test-file frequencies)

**Q1 rate and Q7 count.** H(X) = 9.34 bits (Blocksworld), 10 (code10), 12.92 (prod5).

| Cell (β=1) | AU Burda | AU Asperti–Trentin (Q7) | rate (bits) | bound R − H(X) (bits) |
|---|---|---|---|---|
| A1 lr 3e-4 | 16 [15, 16] | 15 [15, 16] | 11.74 [11.68, 11.75] | 2.40 [2.35, 2.42] |
| A1 lr 3e-3 | 15 [15, 16] | 15 [15, 16] | 11.72 [11.61, 11.85] | 2.39 [2.28, 2.52] |
| A2 lr 3e-4 | 15 [14, 15] | 15 [14, 15] | 11.31 [11.31, 11.39] | 1.98 [1.97, 2.06] |
| A2 lr 3e-3 | 10 [9, 11] | 9 [8, 11] | 11.42 [11.30, 11.49] | 2.09 [1.97, 2.16] |
| A4 lr 3e-4 | 15 [15, 16] | 14 [13, 14] | 11.20 [11.14, 11.33] | 1.86 [1.81, 1.99] |
| A4 lr 3e-3 | 15 [14, 15] | 13 [13, 14] | 11.40 [11.38, 11.56] | 2.07 [2.05, 2.23] |
| code10 A1 / A2 / A4 | 10 / 10 / 10 | 10 / 10 / 10 | 12.30 / 12.45 / 12.35 | 2.30 / 2.45 / 2.35 |
| prod5 A1 / A2 / A4 | 16 / 17 / 23 | 16 / 17 / 23 | 11.74 / 12.22 / 12.02 | −1.18 / −0.70 / −0.90 (no bound) |

Essentially all rate sits in the active units. Every Blocksworld and code10 run spends about 2–2.5 bits more than the data entropy, so KL(q(z)‖p) ≥ 1.9–2.5 bits; prod5 spends less than its 12.92 bits (lossy reconstruction) and the bound is vacuous. The two counts agree except at A4 (A&T 1–2 lower) and A2 lr 3e-3 (1 lower in the median); they differ by at most 2 units in any run.

**Q2 ELBO terms (nats).** MC SE ≤ 0.015.

| Cell (β=1) | D (distortion) | R (rate) | −ELBO | −ELBO − H(X) (bits) |
|---|---|---|---|---|
| A1 lr 3e-4 | 2.53 [2.45, 2.70] | 8.14 | 10.67 | 6.05 |
| A1 lr 3e-3 | 1.85 [1.78, 1.89] | 8.13 | 9.94 | 5.01 |
| A2 lr 3e-4 (15 units) | 1.57 [1.55, 1.66] | 7.84 [7.84, 7.90] | 9.45 [9.38, 9.50] | 4.30 |
| **A2 lr 3e-3 (10 units)** | **1.18 [1.09, 1.29]** | 7.92 [7.83, 7.96] | **9.12 [8.97, 9.18]** | 3.82 |
| A4 lr 3e-4 | 1.43 [1.41, 1.49] | 7.76 | 9.21 | 3.95 |
| A4 lr 3e-3 | 1.32 [1.25, 1.37] | 7.90 | 9.25 | 4.00 |

The 10-unit A2 lr 3e-3 solution has the **best** ELBO of all cells: same rate as the 15-unit A2 lr 3e-4 solution (7.92 vs 7.84 nats), lower distortion (1.18 vs 1.57), −ELBO lower by 0.33 nats, with seed ranges that do not overlap. Fewer units is not a worse optimum here.

**Q5 zero-rate comparison.** D0 = 15.66 nats (Blocksworld), 24.95 (code10), 13.52 (prod5). β* = 1.6–1.8 for every Blocksworld cell (range 1.59–1.85), 2.7 for code10, 1.25 for prod5. All 30 β=4 runs have 0 active units and rate 0.00 bits, consistent with 4 > β* everywhere (heuristic; ignores re-optimisation at the new β).

**Q3 unit anatomy** (active units, 5 seeds per cell; F1 ≥ 0.9 facts per unit):

| Cell | units with 0 / 1 / 2 facts | 2-fact pairs: same block / cross / derived | units per block (sorted, seed 1) | NMI max / 2nd (median) | null p95 max MI (bits) |
|---|---|---|---|---|---|
| A1 lr 3e-4 | 60 / 13 / 6 | 0 / 6 / 0 | 4,4,3,3,1 | 0.24 / 0.15 | ≤ 0.20 |
| A1 lr 3e-3 | 1 / 26 / 50 | 1 / 49 / 0 | 4,4,4,2,1 | 0.33 / 0.29 | ≤ 0.08 |
| A2 lr 3e-4 | 8 / 11 / 55 | 1 / 54 / 0 | 7,4,2,2,0 | 0.33 / 0.29 | ≤ 0.24 |
| A2 lr 3e-3 | 10 / 25 / 14 | 9 / 4 / 1 | 3,2,2,1,1 | 0.45 / 0.23 | ≤ 0.22 |
| A4 lr 3e-4 | 7 / 14 / 55 | 0 / 55 / 0 | 6,3,2,2,2 | 0.33 / 0.29 | ≤ 0.23 |
| A4 lr 3e-3 | 19 / 16 / 39 | 0 / 39 / 0 | 5,5,3,1,1 | 0.34 / 0.26 | ≤ 0.18 |
| prod5 A1 / A2 / A4 | 33/2/13, 9/12/31, 8/30/30 | 13/0/0, 31/0/0, 27/3/0 | 4,4,3,2,2 / 4,3,3,3,3 / 6,5,4,4,4 | 0.29/0.09, 0.49/0.04, 0.51/0.05 | ≤ 0.01 |

No unit detects 3 or more facts; every 2-fact unit uses opposite signs (antipodal). In the 15-unit Blocksworld cells, a unit's two facts belong to different blocks, and no unit is block-aligned: its MI with the best block is barely above the second best (NMI 0.33 vs 0.29). The max-MI block matches the fact's block mainly for 1-fact units. prod5 units are block-aligned (NMI 0.49–0.51 vs 0.04–0.05, both facts in the same block), so the per-block coding is available to these architectures but not used on Blocksworld. The 10-unit A2 lr 3e-3 cell is the exception (more single-fact and same-block units, NMI 0.45 vs 0.23). `same_support` pairs (`on(x,y)`, `on(x',y)`): 0 everywhere.

**Which two facts (presentation of the stored Q3 fact lists).** This is no new quantity. It was added after the run on request. `presentation/tabulate_pairs.py` → `presentation/q3_pairs.json` gives every 2-fact unit by name and sign, per run. That script re-derived the Blocksworld fact lists from the pw03a `f1_conf` matrices with 0 mismatches. Pair types of all 2-fact units, totals over seeds; every pair has opposite signs:

| Cell | 2-fact units | `on(X,Y)`/`on(Y,X)` | `holding`/`holding` | `on(X,Y)`/`holding` | same subject: `on(X,Y)`/`on(X,Z)`, `on(X,Y)`/`on_table(X)`, `on_table(X)`/`holding(X)` | other |
|---|---|---|---|---|---|---|
| A1 lr 3e-4 | 6 | 6 | 0 | 0 | 0 | 0 |
| A1 lr 3e-3 | 50 | 47 | 2 | 1 (X) | 0 | 0 |
| A2 lr 3e-4 (reference) | 55 | 44 | 6 | 5 (1 X, 2 Y, 2 Z) | 0 | 0 |
| A2 lr 3e-3 (10 units) | 14 | 3 | 1 | 0 | 9 (2, 6, 1) | `clear`/`on` 1 |
| A4 lr 3e-4 | 55 | 48 | 4 | 3 (Z) | 0 | 0 |
| A4 lr 3e-3 | 39 | 35 | 3 | 1 (Z) | 0 | 0 |
| prod5 A1 | 13 | 0 | 0 | 3 (X) | 10 (6, 4, 0) | 0 |
| prod5 A2 | 31 | 0 | 0 | 6 (X) | 25 (16, 6, 3) | 0 |
| prod5 A4 | 30 | 0 | 0 | 10 (7 X, 3 Z) | 20 (13, 7, 0) | 0 |

On Blocksworld, the dominant 2-fact unit is the antipodal detector of one **unordered block pair**: `on(X,Y)` at one sign and `on(Y,X)` at the other. These two facts are mutually exclusive in a legal state. Each run contains at most one such unit per pair, and there are 10 unordered pairs. Units of this kind per run: A1 lr 3e-3 9–10, A2 lr 3e-4 8–9, A4 lr 3e-4 9–10, A4 lr 3e-3 5–10, A1 lr 3e-4 0–3 (most of its units detect no fact at 0.9), and the 10-unit A2 lr 3e-3 cell 0–2. In prod5, `on(X,Y)` and `on(Y,X)` can both hold, and no unit pairs them: every prod5 pair shares the subject block (or involves a third block's holding). So the pair code is specific to the Blocksworld constraints, not to the architecture. A candidate reading of "15" is about 10 pair units plus about 5 others (`holding` and mixed). This is post hoc: a hypothesis for a future registered test, not a finding.

**Q4 linear-VAE prediction on `b_lin`** (rank 25 for every source):

| Source | eigenvalues > 1 (γ=1) | > e^-9 (γ floor) | AU fixed γ | AU learned γ |
|---|---|---|---|---|
| w64 | 18 | 25 | 13, 14, 14 | 14, 13, 14 |
| w128 | 24 | 25 | 14, 14, 16 | 17, 17, 16 |
| w128s2 | 24 | 25 | 16, 17, 17 | 17, 16, 15 |
| w128s3 | 25 | 25 | 16, 13, 18 | 18, 18, 17 |
| w256 | 25 | 25 | 19, 17, 19 | 21, 18, 22 |

The linear threshold rule overpredicts in every source. With γ = 1 it predicts 18–25 units against 13–19 observed. With the learned γ (every run at the floor) it predicts all 25 directions against 13–22 observed. The MLP decoder keeps fewer units than the linear-VAE theory predicts.

**Q6 legality anatomy** (20,000 mode-decoded samples per model; prior, β=1, by cell):

| Cell | WF all | cross-block \| WF | derived bits \| cross | legal | independence-implied cross \| WF |
|---|---|---|---|---|---|
| A1 lr 3e-4 | 0.37 | 0.87 | 0.47 | 0.15 | 0.66 |
| A1 lr 3e-3 | 0.54 | 0.92 | 0.65 | 0.33 | 0.71 |
| A2 lr 3e-4 | 0.64 | 0.93 | 0.74 | 0.43 | 0.64 |
| A2 lr 3e-3 | 0.69 | 0.95 | 0.79 | 0.51 | 0.61 |
| A4 lr 3e-4 | 0.67 | 0.95 | 0.78 | 0.50 | 0.61 |
| A4 lr 3e-3 | 0.72 | 0.96 | 0.81 | 0.56 | 0.62 |

Prior samples are lost mainly at two steps. First, malformed blocks: 28–63% of samples have at least one block in zero or two places. The observed rate of well-formed samples equals the product of the per-block rates within 0.01 in every cell, so blocks fail independently. Second, inconsistent derived bits: clear and hand_empty disagree with the positions in 19–53% of position-legal samples. Cross-block constraints are mostly kept: 0.87–0.96 given well formed, against 0.61–0.71 if the model's blocks were independent. So the model learned the cross-block constraints well beyond its own per-block marginals. β=0.25 prior has the same pattern (WF 0.28–0.68, cross | WF 0.82–0.95, derived 0.35–0.79, legal 0.08–0.51). In the aggregate posterior, WF is 0.77–0.94 at β=1 and 0.93–0.98 at β=0.25, cross | WF is 0.975–0.998, and derived | cross is 0.82–0.99. Controls: the data states give 1 / 1 / 1, implied 0.5405 = B2 with test marginals. prod5 gives cross | WF 0.106–0.111 against an implied 0.108–0.112: matches independence, as it should for a model trained without cross-block constraints.

## Audit

**Pass** (`AUDIT.json`, `audit_recompute.json`). The separate script recomputed the headline numbers by other code paths, with 0 mismatches. It recomputed the rate in float64 and H(X), D with another MC seed and a hand-written Bernoulli NLL (within 4 SE), β*, and the Q7 count. It recomputed MI from explicit joint tables with hand-written binning (≤ 1e-6 bits), units per block, and F1 fact counts from the stored pw03a matrices. It recomputed the `b_lin` eigenvalue counts by SVD. It recomputed the Q6 funnel per sample with the pw01 checker and hand-written cross-block code, and the independence prediction by Monte Carlo (within 4 SE). It regenerated `bw-gauss_vae-A2-b1-lr0.0003_s1` prior and aggregate-posterior samples with the frozen pw04a sampler; they are byte-identical to the stored ones. Negative controls pass: uniform random bits give 0 legal, and permuted block labels give MI below every run's maximum unit MI. **Limits:** same author and session, not an independent replication; same retained operands; per-cell summaries, null percentiles, signs and pair classes were not recomputed.

## Limitations

Post hoc on already-opened confirmation data: descriptive, no inferential claim. Scoped to this sweep, this 36-fact encoding and the test-file weights. β* and the eigenvalue rule are heuristics: β* ignores re-optimisation and the difference between training and evaluation weights, and the eigenvalue rule is exact only for linear VAEs. MI depends on the frozen 20-bin discretisation. The pair-type table is a post-run presentation of stored fact lists (cross-checked against pw03a, not part of the frozen audit); the "10 pairs + 5" reading is a hypothesis. Raw inputs and checkpoints stay local, with no off-machine backup. The record is committed locally only (no remote).

## Evidence

`results/analysis_001/results.json` (per-run and per-unit values: `metrics.per_run`, `legality_per_run`, `q4_b_lin`, `summaries`), `results/analysis_001/audit_recompute.json`, `AUDIT.json`, `MANIFEST.json`, `CLAIM_UPDATES.json`.

## Corrections (2026-10-27)
Found by the numbers gate of the report audit (values re-derived without pwlib; the 4,096-draw negative ELBO per cell is in `negelbo_4096/cell_negelbo_nats.json`). The analysis_001 outputs are unchanged; these notes supersede the wording above where they differ.
- **Q2:** the 128-draw Monte-Carlo distortion is noisy at the second decimal (per-run SE up to 0.015 nats). Two independent 4,096-draw estimates give −ELBO 9.125 (8.984–9.157) for A2 lr 3e-3 against 9.436 (9.406–9.510) for A2 lr 3e-4, and D 1.17 against 1.58. Ordering, disjoint ranges and the 0.3-nat gap stand; the 10-unit cell has the best median, and its range overlaps the A4 lr 3e-4 cell only. β* per run 1.59–1.86.
- **Q1:** prod5 reconstructs exactly in every run; its bound is vacuous because the rate stays below its entropy, not because of lossy reconstruction.
- **Q3:** 53 of 55 reference two-fact units pair mutually exclusive facts (2 pair on(X,Y) with holding(Z), which can co-occur). In prod5, 71 of 74 two-fact units pair values of one block. NMI 0.49–0.51 / 0.04–0.05 for prod5 holds at A2 and A4; Blocksworld 0.33 / 0.29 holds in the three full-pair-code cells.
- **Q6:** "equals the product" → within 0.014 of it. Ranges are cell medians; outward: 0.36–0.72, 0.87–0.97, 0.60–0.72, 0.46–0.81; aggregate posterior well formed 0.76–0.94.
