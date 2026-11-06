# pw01_state_space_ground_truth — exact ground truth for the Blocksworld state space — **COMPLETED: SUPPORTED** (2026-10-14, run `analysis_001`)

- **Proposal item:** P1 (expected factor count fixed before any VAE count).
- **Hypothesis:** 866 legal states (501 arm-empty + 365 holding), equal under
  brute force + checker, STRIPS BFS and the Lah formula; the thesis data contain
  only legal states, cover all 866, and `hand_empty` equals step parity.
- **Frozen method:** `PROTOCOL.md`, `DESIGN.json` (freeze record
  sha256 `45e2b478…`); scripts `run_ground_truth.py`, `audit.py`;
  reusable code `src/pwlib/blocksworld.py` (tests `tests/test_blocksworld.py`,
  `uv run pytest`: 8 passed). Exact model-free computation: no seeds, no pilot
  (Proportionality rule), `--kind analysis`.
- **Run:** `analysis_001`, status `completed`, outcome `supported`, CPU, 1
  process, 2.9 s. One attempt, no retries. Before the freeze the same code was
  smoke-tested on a scratch copy outside `results/` (code check only; the
  numbers were already known from the exploratory script).

## Criteria and gates

| Item | Value | Pass |
|------|-------|------|
| gate task: input hashes, fact order, vocab, PAD=52 suffix, Y=0 at PAD, 1,000,000 transitions match STRIPS, 1,020,005 valid positions | all as specified, 0 transition mismatches | yes |
| gate numerical: 3 counts agree; 31,176 single-bit flips rejected; 200,000 random vectors | 0 flips accepted; 0 random accepted (= 0 in set) | yes |
| count_agreement | brute force = BFS = L(5)+5L(4) = 866; arm-empty 501 = L(5) | yes |
| data_all_legal | 0 illegal of 1,020,005 valid positions (815,457 train + 204,548 test) | yes |
| data_cover_all_legal | 866 / 866 | yes |
| hand_empty_equals_step_parity | 0 violations | yes |

## Results (exact; scoped to this encoding and these data files)

| Quantity | Value |
|----------|-------|
| legal states / arm-empty / holding | 866 / 501 / 365 |
| log2 866 / log2 501 / log2 365 | 9.758 / 8.969 / 8.512 bits |
| H(state), all valid positions | 9.346 bits (train 9.345, test 9.335) |
| H(state) at even k / odd k | 8.473 / 8.216 bits (515,204 / 504,801 positions) |
| H(state \| step parity) | 8.346 bits |
| per-block categorical (5 vars x 6 values) | 12.925 bits; 866/7776 = 11.1% of product legal |
| arm-empty view (5 vars x 5 values) | 11.610 bits; 501/3125 = 16.0% legal |
| total correlation, empirical (all / even k / odd k) | 1.366 / 0.997 / 2.659 bits |
| total correlation, uniform over legal | 2.023 bits |
| mean pairwise MI between block variables (empirical all) | 0.114 bits |
| rank / affine dim, 866 x 36 state matrix | 26 / 25 (same without the hand_empty column) |
| affine dim, arm-empty / holding subsets | 20 / 24 |
| I3D (exploratory, hypercube volumes, r = 3..8, R = 2r) | 11.9 → 21.3 (scale-dependent) |

Positive control for I3D: full {0,1}^8 and {0,1}^10 give exactly 8 and 10 under
the hypercube volume model; the Z^d lattice model gives 2.7–4.5 on them, so its
values on the legal states (2.6–4.6) are not meaningful for binary data. The
I3D numbers are our own re-implementation, not checked against DADApy.

All candidate dimension targets, each with a definition and a
`hand_empty_free` flag (False = hand_empty, fixed by step parity, not counted
as a free factor), are in `results/analysis_001/dimension_candidates.json`.
The dimension candidates are definitions; this experiment makes no claim
about which one a VAE active-unit count should match.

Side finding: the data files are not split at sequence length 30. Train
sequences have 40–60 actions, test 14–60. Here `k <= 30` / `k > 30` are
position ranges (H = 9.051 / 9.646 bits), not the train/test split.

## Audit

`AUDIT.json` status `pass`. `audit.py` (written separately, same author and
session, so not independent replication) re-enumerates the states by
generating tower arrangements without the checker, re-reads the raw arrays
with per-position set lookups, and recomputes entropies, total correlation
and affine dimensions from the retained files. All 10 checks agree. Negative
controls: 0/5000 column-permuted legal states accepted, and TC = 0 for an
independent pair. The I3D estimates were not re-derived.

## Evidence (`results/analysis_001/`)

`results.json`, `dimension_candidates.json`, `state_frequency_table.csv`
(per-state counts: all, train, test, even/odd k, k<=30/k>30),
`legal_states_866.npy` (gitignored, local only, no off-machine backup),
`blocksworld_legality.py` (importable checker copy), `audit_recompute.json`,
`run_stdout.log`, `MANIFEST.json`, `AUDIT.json`, `CLAIM_UPDATES.json`.

## Limitations

Exact for 5 blocks, this 36-fact encoding and the hashed thesis files only.
Entropies describe the thesis data distribution, not a property of the domain.
The per-block variables are dependent (TC > 0), so "5 factors" is not 5
independent factors. I3D is exploratory and strongly scale-dependent.
