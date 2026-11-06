# pw02a_active_units_amended — active units, amended sweep (A3 removed, 60k steps, KL warm-up) — **COMPLETED: MIXED** (2026-10-22, run `full_001`; H2.2 false, H2.3 true, H2.4 false)

Proposal item **P2**. **Pilot-triggered amendment** of `pw02_active_units`
(both pw02 pilots ended in `competence_stop`; see the pw02 README), decided
by the project lead on 2026-10-21 after seeing the pw02 pilot evidence
(development data, seed 101) and **before any confirmation data was
opened**. Frozen 2026-10-21 (FREEZE sha256 `a18b6c05...6d`).

**Changes** (details `PROTOCOL.md`, `DESIGN.json` `amendment_changes`):
A3 (4x256) removed (does not fit raw states under this recipe, even with
warm-up; reported descriptively from the pw02 pilots); linear KL warm-up for
every cell (the pw02 contingency fired); 60,000 steps for every cell; amended
convergence check (absolute tolerance 1e-3 nats/sample for the plain AE and
SAE; learned-gamma cells: reconstruction gate at 80% and 100% of steps and an
identical active-unit set); secondary families unchanged. H2.2 is evaluated on
A1, A2, A4 with the information account required in all three (>= 3,
unchanged count); H2.3 over the 6 beta=1 arch x lr cells. One re-pilot; any
uncovered failure stops; no further amendments.

**Grid:** `sweep_grid.json`, 58 cells, 210 confirmation runs, 20 pilot cells
(seed 101). **Consumed by:** pw03a, pw04a. **Files:** `PROTOCOL.md`,
`DESIGN.json`, `sweep_grid.json`, `extract_activations.py`,
`train_sweep.py`, `analyze_active_units.py`, `audit.py`; shared code
`src/pwlib/` (80% snapshot in `training.py`, `sweep.converged_amended`);
tests `tests/test_amendment_pw02a.py`.

## Runs

**`pilot_001`** (pilot, seed 101, 20 cells, development data only; confirmation file not opened): `completed`, outcome `pilot_only`. Gates task / numerical / competence: true / true / true (all 7 required pilot cells pass: Blocksworld beta=1 A1, A2 both lr, A4; code10 and prod5 A2; `b_lin_w128` learned gamma). Audit pass (0 mismatches). No crash/NaN retries.

- Amended convergence check: 19 of 20 runs converge. The learned-gamma activation cells pass (reconstruction gate at step 48,000 and 60,000, identical active sets: `b_lin_w128` 18 units, `b_raw_w128` 22 units). Plain AE and SAE pass under the absolute tolerance. **Not converged: the categorical VAE** (dev objective -4.0% over the last 20%; it also fails competence, exact 0.28). It is a secondary family that the amendment leaves unchanged (item 5) and enters no required criterion; judged covered, reported descriptively, not a stop.
- `b_lin_w128` learned log gamma reaches the floor (-9.0; flagged; pw02 scope note on the floor applies); `b_raw_w128` -3.35.
- Secondary families still failing competence: beta-TCVAE beta_TC=6 (exact 0.77), categorical VAE (0.28), beta=4 (<= 0.03), `indep` (descriptive).
- Pilot-only AU (development data, seed 101): Blocksworld beta=1 lr 3e-4 A1 15, A2 15, A4 16; A2 lr 3e-3 9; code10 A2 10; prod5 A2 15; `b_lin_w128` learned 18 vs fixed 15; `b_raw_w128` learned 22 vs fixed 14. Source gates and targets_b identical to pw02.
- Compute: CPU, float32, 1 thread x 8 workers; extraction 25 s; sweep 36.5 min wall, 5,169 worker-s (1.44 CPU-h) of training; longest run beta-TCVAE 2,185 s; 1.0 GB.

**`full_001`** (full, cites `pilot_001`; seeds 1-5 for the 18 core `bw-gauss_vae` cells, 1-3 otherwise; 210 runs; **confirmation opened**: thesis test-file state weights on raw states, 300-sequence test subsample on activations): `completed`, outcome **`mixed`**. Gates task / numerical / competence: true / true / true (every required cell has all planned seeds passing). Audit pass (0 mismatches). No crash/NaN retries.

| Required criterion | Verdict | Key numbers (AU on the confirmation distribution, gated seeds) |
|---|---|---|
| `h22_information_account` | **false** | (i) holds: all 30 gated beta=1 runs have AU < 25 (max 16), none censored. (ii) fails: seed medians Blocksworld / code10 / prod5 = A1 16 / 10 / 16 (verdict **affine**), A2 15 / 10 / 17 (**affine**), A4 15 / 10 / 23 (**none**); information account in 0 of 3 architectures (required: 3) |
| `h23_architecture_dependence` | **true** | seed-median AU across the 6 beta=1 arch x lr cells: A1 16 / 15, A2 15 / 10, A4 15 / 15 (lr 3e-4 / 3e-3); between-cell range 6 >= 3 and > max within-cell range 2 |
| `h24_learned_gamma_more_units` | **false** | `b_lin` learned vs fixed gamma (seed values): w64 [14,13,14] vs [13,14,14]; w128 [17,17,16] vs [14,14,16]; w128s2 [17,16,15] vs [16,17,17]; w128s3 [18,18,17] vs [16,13,18]; w256 [21,18,22] vs [19,17,19]: no source has a median gap >= 2 with non-overlapping ranges (w128 median gap 3, but ranges touch at 16) |

Outcome `mixed` (1 of 3 required criteria). Reference targets (pw01): bits 9.76 / 10, per-block factors 5, affine dimension 25, facts 36. AU is an architecture- and training-relative count (Asperti), not the dimension of Blocksworld.

Secondary and descriptive (not criteria):
- Blocksworld beta=1 AU sits at 15-16 for every architecture at lr 3e-4 and for A1/A4 at lr 3e-3; A2 at lr 3e-3 is the outlier (10). The 10-bit code10 control gives exactly 10 units at every architecture and seed (positive control); prod5 (12.9 bits, 5 factors, affine dim 25) gives 16-23, rising with width. So Blocksworld behaves like prod5, not like code10.
- Variance decomposition of beta=1 AU (descriptive, SS): arch 61.8, lr 28.0, interaction 35.3, residual seed 7.2 (total 132.3).
- beta monotone (secondary): holds in 6 of 6 cells; beta=4 collapses to AU 0 everywhere (fails competence, as anticipated), beta=0.25 gives about the same AU as beta=1.
- Thresholds: AU identical at 0.001 / 0.01 / 0.05 / 0.1 for the A2 reference runs; 80% of Gaussian runs are polarized (ratio >= 10).
- H2.4 secondary: `b_raw` learned-gamma AU rises with source width (medians w64 14, w128 20, w256 29; w128 seeds 20 / 22 / 22) and exceeds `b_lin` at 4 of 5 sources (not at w64), so `h24_b_raw_more_than_b_lin` is false as a whole; b_raw AU stays below K_probe at w64-w128 (25-27) and near it at w256 (29 vs 31), far below K_pca99 (39-208). `b_lin` learned AU 13-22 is not closer to 25 than raw-state AU (secondary false).
- All 15 `b_lin` learned-gamma runs end with log gamma at the floor (-9): their AU partly reflects the floor (pw02 scope note).
- Plain AE control: covariance AU 32 / 32 / 32 (all units), as expected without a KL term.
- Secondary families failing competence in every seed (comparisons depending on them are untested downstream): beta-TCVAE beta_TC=6 (exact 0.61-0.62), categorical VAE (0.26-0.35), all beta=4 cells, `indep`; `b_raw_w256` fixed gamma fails its gate (contrast condition, reported). beta-TCVAE beta_TC=2 passes (AU 15-16).
- Convergence (reported in the full run, not a gate there): 22 of 210 runs fail the amended check, including core seeds `bw` A1 beta=1 lr 3e-4 s2/s5 and A2 lr 3e-3 s5 (relative change just over 2%), and 8 of 15 `b_lin` learned-gamma runs whose active-unit set differs between step 48,000 and 60,000 (by about one unit). Stated as a limitation of H2.4 and H2.2.
- **Correction (2026-10-22):** the convergence bullet above miscounts the learned-gamma `b_lin` runs. **6 of 15** (not 8) fail the amended check: `b_lin_w128` s3, `w128s2` s1/s2, `w128s3` s2, `w256` s1/s3. Their development active sets differ between step 48,000 and 60,000 by 1 to 4 units (symmetric difference), not "about one unit". The total of 22 non-converged runs and the per-run flags are correct. Evidence: `results/full_001/results.json` (`metrics.convergence`), `results/full_001/runs/*.json` (`snapshot.active_dev`, `active_dev_final`), independent audit `../audits/independent_audit_001/AUDIT_REPORT.md` (point 2).
- **Literature check (2026-10-27, post hoc; no verdict changes):** Asperti & Trentin (2002.07514, Table 5) find that a higher learning rate leaves more latents active; our one moving cell (A2 lr 3e-3, 10 units) goes the other way, so agreement with [2] is in sensitivity, not direction. The learned-gamma floor on `b_lin` is expected (unbounded Gaussian likelihood on a finite point set; Mattei & Frellsen 2018), so H2.4 compared gamma ~ 1.2e-4 with gamma = 1. A covariance-spectrum fit to 15-16 units holds only under uniform state weights, not under the training frequencies. Details: `../../docs/EXPERIMENT_LOG.md` (pw02a, 2026-10-27).
- Compute: CPU, float32, 1 thread x 8 workers; extraction 26 s; sweep 4.13 h wall, 117,327 worker-s (32.6 CPU-h) of training (estimate 31; cap 45); longest runs activation beta-TCVAE about 2 h; disk 2.6 GB (gitignored raw/ and models/; no off-machine backup).
