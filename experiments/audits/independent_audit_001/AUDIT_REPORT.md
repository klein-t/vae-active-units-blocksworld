# Independent audit 001: pw02a full_001, pw03a full_001, pw04a pilot_001 (+ descriptive pass)

**Date:** 2026-10-22. **Auditor:** a separate session that did not write, run or self-audit these
experiments. **Independence limit:** same machine, same retained artifacts; the recompute code below is
newly written and does not import `pwlib`
(it reuses only the thesis data files, the saved checkpoints/state dicts, the cached activation files,
which are themselves regenerated, and the TransformerLens loader). This is an independent check, not an
external replication. Nothing under `experiments/*/results` or any frozen design was modified.

## Bottom line

**No finding changes a recorded conclusion.** Every headline number was reproduced exactly (AU,
competence, convergence, Cov30, MonoFrac) or within Monte Carlo error (legality), and every
pre-registered verdict re-derives the same from the frozen text: H2.2 false, H2.3 true, H2.4 false,
H3.2 false, pw04a untested.

Points worth knowing (none flips a verdict):

1. **Pilot stop rule was applied by judgment.** In pw02a `pilot_001` the categorical VAE failed the amended
   convergence check (-4.0%). The amendment's stop rule says an *uncovered* convergence failure stops the
   experiment. The analysis ruled it covered by amendment item 5 (secondary families unchanged; they are
   treated as untested if they fail competence, and this model also failed competence, exact 0.28). That
   reading holds up and is disclosed in the README. But the call was made after seeing the
   pilot. Read strictly, pw02a would have stopped at the pilot and all of pw02a/pw03a/pw04a would be untested.
   The project lead should own this call explicitly.
2. **README miscount (pw02a full_001).** It says "8 of 15 `b_lin` learned-gamma runs" fail convergence.
   The correct number is **6 of 15**. Of the 22 non-converged runs, the activation learned-gamma ones are
   6 `b_lin` plus 3 `b_raw` beta=0.25. The total of 22 and the per-run flags in results.json are correct;
   only the sentence is wrong.
3. **H2.3 passes on a single cell.** The 6-unit between-cell range comes entirely from A2 at lr 3e-3
   (median 10). The other five beta=1 cells sit at 15-16. The verdict follows the frozen text, but it should
   be described as one arch x lr interaction, not a broad architecture effect.
4. **The H3.2 verdict is decided by one fact.** k=5 SAE seed minimum 24/30 vs. beta-VAE seed maximum 25/30.
   It stays false under every variant I tried (tie-breaking, active-set weights, dev-selected code).
5. **Convergence sensitivity.** If the 22 non-converged runs are dropped, all three required verdicts stay
   the same. The secondary beta-monotonicity check flips from 6/6 to 5/6, because the A4 lr 3e-4 beta=0.25
   median falls to 13. It is secondary only.
6. pw01 (model-free, before pw02) read the thesis **test** file to count state coverage (1,020,005
   positions = train + val + test). No model selection or criterion used it, so it is harmless. Strictly,
   though, "the test file was never read" holds for pw02/pw02a, not for the project as a whole.

## Checks

| # | Check | Result | Mine vs recorded |
|---|---|---|---|
| 1a | Freeze integrity (pw02a 20 files, pw03a 16, pw04a 16) | **pass** | working copies and `.frozen` snapshots hash-equal to FREEZE.json; all mtimes precede freeze; `git diff 7027d8a..HEAD` touches no frozen code (only READMEs, logs, results, the new descriptive wrapper) |
| 1b | Run/freeze binding | **pass** | RUN.json and results.json `freeze_sha256` = sha256(FREEZE.json) for all 5 runs; every MANIFEST entry (650 + 80 + 170 + 22 + 14 files) matches size and hash; 12 external inputs (thesis data, 5 exp46 checkpoints, pw01 files) match |
| 1c | Confirmation sealed in pilots | **pass** | pw02a pilot has no `w_test`, no `conf_*` arrays in any npz, no `conf` metrics; pw03a/pw04a pilots read only pw02a `pilot_001` (code: test weights only when not pilot) |
| 1d | pw04a descriptive pass used frozen code | **pass** | `analyze_legality.py` byte-identical to snapshot; header `descriptive-unregistered`, no RUN.json |
| 2a | AU, all 207 non-`indep` runs of full_001 + 19 pilot runs (covers every arch/substrate/beta/family) | **pass** | 0 mismatches in AU, robustness counts, active sets, categorical activity, SAE alive, plain-AE covariance AU; my mu vs stored operands max abs diff 4.8e-6 |
| 2b | Competence (dev) for the same runs | **pass** | 0 mismatches; all 17 required cells 5/5 or 3/3 passing; 7 required pilot cells pass |
| 2c | Amended convergence rule | **pass** | 0 mismatches; 22 non-converged runs in full_001 (bw beta=1: A1 lr3e-4 s2/s5 at 2.2%/2.4%, A2 lr3e-3 s5 at 2.7%), 1 in pilot (cat VAE) |
| 2d | H2.2 table (seed medians, gated) | **pass** | A1 16/10/16 (affine), A2 15/10/17 (affine), A4 15/10/23 (none); part (i) 30 runs, max AU 16 < 25 |
| 2e | H2.3 | **pass** | cell medians A1 16/15, A2 15/10, A4 15/15; between 6, max within 2 |
| 2f | H2.4 (learned vs fixed gamma per source) | **pass** | w64 14 vs 14; w128 17 vs 14 (ranges touch at 16); w128s2 16 vs 17; w128s3 18 vs 16 (17 < 18); w256 21 vs 19 (18 < 19); learned log gamma = -9.0 (floor) in all 15, fixed = 0.0 |
| 3 | Verdicts re-derived from the frozen text | **pass** | H2.2 false, H2.3 true, H2.4 false, beta-monotone 6/6 true, experiment competence true; H3.2 false; pw04a competence_stop -> untested |
| 3b | Did the A3-removal rounding make anything easier? | **pass (no)** | H2.2(ii) "3 of 4" -> "3 of 3": passing the new rule implies passing the old one (harder or equal); beta-monotone 7/8 -> 6/6 and H3.1 7/8 -> 6/6 are stricter; pw04a's 8-run stratum minimum is kept with 30 instead of 40 runs (stricter). The only thing that loosened is the cell set itself (A3 is no longer a required cell; for H2.3, fewer cells can shrink the within-cell yardstick). That is the disclosed substance of the amendment, not rounding. Here the margins are wide (H2.2 has 0 information verdicts; H2.3 is 6 vs 2) |
| 4a | pw03a Cov30, own signed best-F1 rule, all gated seeds | **pass** | k=5 SAE 0.800/0.833/0.833; beta-VAE A2 beta=1 0.767/0.833/0.833/0.767/0.833 (max 0.833); beta-TCVAE beta_TC=2 0.667/0.733/0.700; beta=4 and beta_TC=6: 0 gated seeds (my competence recompute) |
| 4b | MonoFrac / n_active | **pass** | k=5 SAE 0.871/0.906/0.909 (31/32/33 alive); beta-VAE beta=1 0.80/0.93/0.93/0.80/1.00 (15/15/14/15/15 active); 0 mismatches against per_model |
| 4c | H3.2 sensitivity | **pass** | false under dev ties broken towards the largest positive set, active set on dev weights, and code chosen by dev F1 instead of max over conf F1 |
| 5 | pw04a legality, own sampler and BFS-866 lookup | **pass** | 4 required pilot cells, aggregate posterior: A1 0.628 (rec 0.630), A2 0.820 (0.822), A2 lr3e-3 0.878 (0.883), A4 0.862 (0.861), all < 0.95, so the stop is real; prior 0.169/0.428/0.499/0.505 (rec 0.168/0.435/0.496/0.502); decoding at z = mu reconstructs exactly (1.000), so the sampler is not the cause; 56 runs total, max abs z of the difference 2.87 over 112 comparisons; B1 2.543e-4 and B2 0.53995 exact match; descriptive "24 of 30 below B2" reproduced |
| 6a | Pairing Y[k] (current state) | **pass** | all 1,000,000 thesis transitions satisfy Y[k] = apply(X[k], Y[k-1]) under my STRIPS; the 5 activation caches regenerate bit-exactly (acts diff 0.0, ycur, (seq, k) and b_lin equal to 1.4e-7); exp46 head agrees more with Y[k+1] (0.92-0.97) than Y[k] (0.89-0.92), and the stored probe reads the current state (0.998-0.9998) rather than the next (0.87) |
| 6b | Splits, weights, training distribution | **pass** | split re-derived (12,803 / 3,201, disjoint, complete); w_train / w_val / w_test recomputed from Y at X != PAD equal stored to 0.0 (652,451 / 163,006 / 204,548 positions; all 866 states present in every split); TableData samples from `w_train` (code) |
| 6c | Seed leakage | **pass** | logged run seeds equal sha256(cell|seed) derivation; all unique; no pilot/full overlap; confirmation subsample drawn from the test file only |
| 6d | Off-by-one / latent sizes | **pass** | latent 32 (raw) / 64 (activations) from the checkpoint shapes; AU strict > 0.01; censored = AU == latent (none) |
| NC | Negative controls of my checkers | **pass** | 0 of 100,000 random vectors and 0 of 31,176 single-bit flips accepted as legal; permuted labels give k=5 SAE Cov30 0.0 (true labels 0.8); every cached row differs from its next-state row, so a Y[k+1] pairing would be detected |

Not re-derived: the `indep` control (eval sample generator-defined; descriptive only), TC diagnostics,
K_pca99/K_probe targets, pw03a MIG/DCI/oracle gates, pw04a violation taxonomy and H4.2 bootstrap
(stopped experiment, descriptive only), A3 (not trained).

## Files

- `audit_freeze.py` -> `freeze_check.json` (check 1)
- `recompute_data.py` -> `data_check.json` (6a transitions, 6b splits/weights, legal set, controls)
- `verify_activations.py` -> `activation_check.json` (6a activation pairing)
- `recompute_pw02a.py` -> `pw02a_recompute.json` (2, 3, 6c, 6d; per-run rows)
- `recompute_pw03a.py` -> `pw03a_recompute.json` (4)
- `recompute_pw04a.py` -> `pw04a_recompute.json` (5)
- `summarize.py` -> `recomputed_values.json` (negative controls + compact headline values)
- `ind_common.py`: own STRIPS/BFS legality, split, weights, forward passes

Run from this directory with the project venv, in that order (about 1 min total on CPU):
`../../../.venv/bin/python <script>.py`.
