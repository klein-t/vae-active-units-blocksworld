# pw06_encoding_and_latent_size — re-encoding the same states, latent size and copied fair bits — **COMPLETED: MIXED** (2026-10-27, run `full_001`; P1 false, P2 true, P3 true)

- **Proposal item:** P2 (what sets the active-unit count). Follow-up to three reviews of report v1.1, registered after pw02a–pw05 were known.
- **Questions** (full text, predictions and rules: [`DESIGN.md`](DESIGN.md)):
  - **Q1 encoding:** with network, β and lr fixed, does the count on the same 866 states change when they are re-encoded (36 facts, 30 position facts, 10-bit random index)? P1: both 10-bit encodings collapse (median ≤ 3, max ≤ 5, at least 10 below E36). S2: E30 ≤ E36.
  - **Q2 latent size:** does the count at the reference cell stop growing with L? P3: medians at L = 24, 32, 64 within 2. S1: at L = 8, 10, 12 the median is at least L − 1.
  - **Q3 copies:** does a fair bit get a unit only when copied? P2: c = 1 median ≤ 1; c = 2, 3, 4 median 10 with every seed in 9–11.
- **Gates:** pipeline (E36-L32 median within 1 of 15, all 5 runs reconstruct; else technical stop); capacity for P1 (E10a plain AE exact ≥ 0.95 in ≥ 2 of 3 seeds).
- **Frozen design:** `DESIGN.md`, SHA-256 `904ea2366846496c08b6c68668861d0f93677f06cf232c546967104349a1b129` (also recorded in each `RUN.json` as `freeze_sha256`; the run and analysis scripts refuse to run if the file changed).
- **Files:** `sweep_grid.json` (cells, seeds, training block), `pw06_encodings.py` (E36, E30, E10a/b, code10 copies, model-free table), `run_pw06.py` (training), `pilot_check.py` (pilot health, added after registration; no counts), `analyze_pw06.py` (registered analysis: every count, check, gate and verdict); tests `tests/test_pw06_encodings.py`, `tests/test_pw06_rules.py`.

## Implementation choices (routine, fixed before any pw06 run)

- **Same computation as pw02a.** Training, evaluation, numerical flags and the convergence rule are pwlib code unchanged (`training.train`/`evaluate`, `sweep.numerical_flags`/`converged_amended`): 2×256 mirrored ReLU MLP, Bernoulli decoder, Adam, batch 512, clip 10, 60,000 steps, linear KL warm-up over 6,000 steps, final checkpoint, CPU float32, one thread per worker. The E36-L32 cell keeps pw02a's cell id `bw-gauss_vae-A2-b1-lr0.0003`, so its run RNG seeds (sha256 of `cell_id|seed`) are pw02a's: it is the same computation as the pw02a reference cell (seeds 1–5; pilot seed 101). All other cell ids are new, so their seeds are independent. pw02a's 80% weight snapshot is dropped (it served only learned-γ cells and computes active sets during training).
- **Data.** Every encoding is a row-wise function of the pw01 table `legal_states_866.npy`, trained and evaluated with the pw02a state weights (train split for training, val = development, thesis test file = test split). code10 copies are uniform over their 1,024 states.
- **E10 index.** Code of pw01 state i = `perm[i]`, `perm = default_rng(seed).permutation(866)`, 10 bits MSB first. This is the reading that reproduces the design's total-correlation row (0.391–0.413 nats for seeds 0–3; the inverse permutation gives 0.401–0.407).
- **code10 copies.** Same 10-bit generator as pw02a (`itertools.product`); columns placed by pw02a's assignment seed 0 over 40 columns: `perm[:10c]` carry bit k mod 10, the other 40 − 10c columns are constant zero.
- **Analysis.** Counts, reconstruction check and descriptive quantities on the test split (as pw02a `full_001`). Rate = stored per-unit KL sum; distortion = 128-draw Monte Carlo (pw05 definition); nats. P1/P2 are untested if a needed cell has no eligible run (relevant only for the convergence-filtered variant). The primary outcome label is `supported` (P1–P3 all hold), `unsupported` (all fail), `untested` (all untested) or `mixed`.

## Model-free table (recomputed with `pw06_encodings.py`, pw01 `count_train` frequencies)

| Quantity | DESIGN.md | Recomputed |
|---|---|---|
| H(X) | 9.345 bits = 6.478 nats | 9.3452 bits = 6.4776 nats |
| E36 zero-rate distortion | 15.662 nats | 15.6623 nats |
| E10 total correlation, permutation seeds 0–3 | 0.39–0.41 nats | 0.4001, 0.3986, 0.3910, 0.4131 |

## Cells (78 runs; run id = `<cell id>_s<seed>`)

| Cell | Cell id | Seeds | Rule |
|---|---|---|---|
| E36-L32 | `bw-gauss_vae-A2-b1-lr0.0003` | 1–5 | pipeline gate, P1 reference, P3, S2 |
| E30-L32 | `e30-gauss_vae-A2-b1-lr0.0003` | 1–5 | S2 |
| E10a-L32 | `e10a-gauss_vae-A2-b1-lr0.0003` | 1–5 | P1 |
| E10b-L32 | `e10b-gauss_vae-A2-b1-lr0.0003` | 1–5 | P1 |
| E10a plain AE | `e10a-plain_ae-A2-lr0.0003` | 1–3 | capacity check |
| E10a β = 0.25 | `e10a-gauss_vae-A2-b0.25-lr0.0003` | 1–5 | descriptive |
| E36-L8, L10, L12 | `bw-gauss_vae-A2-b1-lr0.0003-L{8,10,12}` | 1–5 | S1 |
| E36-L16 | `bw-gauss_vae-A2-b1-lr0.0003-L16` | 1–5 | descriptive |
| E36-L24, L64 | `bw-gauss_vae-A2-b1-lr0.0003-L{24,64}` | 1–5 | P3 |
| code10 c = 1, 2, 3, 4 | `code10c{1,2,3,4}-gauss_vae-A2-b1-lr0.0003` | 1–5 | P2 |

Pilot (`pilot_001`, seed 101, development data, technical only, no counts): E36-L32, E36-L64, E10a plain AE, code10 c = 2.

## Reproduce

```bash
# RUN.json (written once per run: experiment_id, run_id, kind pilot|full, freeze_sha256 = sha256 of DESIGN.md, pilot_run)
PYTHONPATH=src .venv/bin/python -u experiments/pw06_encoding_and_latent_size/run_pw06.py --run-id pilot_001 --workers 4
PYTHONPATH=src .venv/bin/python -u experiments/pw06_encoding_and_latent_size/run_pw06.py --run-id full_001 --workers 8
PYTHONPATH=src .venv/bin/python experiments/pw06_encoding_and_latent_size/analyze_pw06.py --run-id full_001
.venv/bin/python -m pytest -q tests/test_pw06_encodings.py tests/test_pw06_rules.py
```

Weights (`models/`) and raw posterior arrays (`raw/*.npz`: per split `mu`, `logvar`, `weights`, per-unit KL, reconstruction bits) stay local and gitignored.

## Runs

Registered in commit `ebb472e` (design, code, analysis script and log entry) before any pw06 model was trained.

**`pilot_001`** (pilot, seed 101, 4 cells, development data only; technical; no active-unit count computed or looked at): passed, no amendment. All 4 runs completed in one attempt, objective finite, KL non-negative, dev objective converged, dev exact reconstruction 1.0. Runtime 64–111 s per run (plain AE 64 s, E36-L64 111 s); 116 s wall with 4 workers × 1 thread. The E36-L32 seed-101 training curve is bit-identical to pw02a `pilot_001` (max abs difference 0). Evidence: `results/pilot_001/pilot_health.json` (`pilot_check.py`), `runs/`, `MANIFEST.json`.

**`full_001`** (main, cites `pilot_001`; 78 runs; test-split weights): status `completed`, outcome **`mixed`**. 78/78 runs ok in one attempt, none numerically flagged. CPU, float32, 8 workers × 1 thread, torch 2.14.1, Python 3.12.11; 1,497 s wall (25 min), 11,561 worker-s (3.2 CPU-h) of training; 92 MB on disk (gitignored `raw/`, `models/`). The analysis ran once (`analyze_pw06.py`).

| Gate | Result |
|---|---|
| task (data, encodings, model-free table) | pass |
| pipeline: E36-L32 median within 1 of 15, 5/5 reconstruct | pass (median 15, seeds 15, 15, 14, 15, 15; 5/5) |
| capacity for P1: E10a plain AE exact ≥ 0.95 in ≥ 2/3 | pass (exact 1.0 in 3/3) |

The E36-L32 posterior means are bit-identical to pw02a `full_001` for all 5 seeds (same counts 15, 15, 14, 15, 15): the pipeline reproduces the reference cell exactly.

| Rule | All runs (registered) | Without non-converged runs | Key numbers (counts per seed) |
|---|---|---|---|
| **P1** (Q1) | **false** | false | E10a 9, 6, 6, 6, 8 (median 6); E10b 7, 7, 7, 6, 7 (median 7); all three conditions fail for both: median > 3, max > 5, gap to E36 (15) 9 and 8 < 10 |
| **P2** (Q3) | **true** | true | c = 1: 0 in every seed; c = 2, 3, 4: 10 in every seed |
| **P3** (Q2) | **true** | true | medians L = 24 / 32 / 64: 15 / 15 / 15 (spread 0) |
| S1 (Q2) | true | true | L = 8, 10, 12: 8, 10, 12 in every seed (every unit active) |
| S2 (Q1) | true | true | E30 11, 12, 11, 11, 11 (median 11) ≤ E36 15 |

Non-converged runs (pw02a rule): `E36-L12` s3 and `E36-L16` s5. Neither enters P1, P2 or P3, and S1 at L = 12 keeps 12 with the 4 converged runs, so every verdict is the same with and without them.

Per-cell table (test split, nats; median over seeds; count = Burda active units at 0.01):

| Cell | Count median (range) | Recon pass | −ELBO | Rate | Distortion | Rate − H(X) |
|---|---|---|---|---|---|---|
| E36-L32 | 15 (14–15) | 5/5 | 9.45 | 7.84 | 1.57 | 1.37 |
| E30-L32 | 11 (11–12) | 5/5 | 8.44 | 6.49 | 1.97 | 0.02 |
| E10a-L32 | 6 (6–9) | 0/5 | 6.70 | 2.12 | 4.58 | −4.35 |
| E10b-L32 | 7 (6–7) | 0/5 | 6.71 | 2.07 | 4.63 | −4.40 |
| E10a plain AE | 32 (32–32), covariance count | 3/3 | n/a | n/a | 0.00 | n/a |
| E10a β = 0.25 | 10 (10–10) | 5/5 | 8.79 | 8.47 | 0.34 | 2.00 |
| E36-L8 | 8 (8–8) | 5/5 | 9.30 | 7.95 | 1.37 | 1.48 |
| E36-L10 | 10 (10–10) | 5/5 | 9.53 | 7.94 | 1.55 | 1.47 |
| E36-L12 | 12 (12–12) | 5/5 | 9.63 | 7.93 | 1.67 | 1.46 |
| E36-L16 | 14 (13–15) | 5/5 | 9.42 | 7.92 | 1.55 | 1.45 |
| E36-L24 | 15 (15–16) | 5/5 | 9.59 | 7.87 | 1.72 | 1.40 |
| E36-L64 | 15 (15–16) | 5/5 | 9.62 | 7.89 | 1.73 | 1.41 |
| code10 c = 1 | 0 (0–0) | 0/5 | 6.93 | 0.00 | 6.93 | −6.93 |
| code10 c = 2 | 10 (10–10) | 5/5 | 9.11 | 6.89 | 2.23 | −0.04 |
| code10 c = 3 | 10 (10–10) | 5/5 | 9.92 | 8.20 | 1.72 | 1.27 |
| code10 c = 4 | 10 (10–10) | 5/5 | 10.38 | 8.82 | 1.56 | 1.89 |

H(X) on the test split: 6.48 nats for the Blocksworld encodings, 10 ln 2 = 6.93 nats for code10.

**Reading (descriptive, not a criterion).** Re-encoding the same 866 states changes the count at fixed network, β and lr: 15 (E36), 11 (E30), 6–7 (E10). The 10-bit encodings do not collapse as P1 predicted: they keep 6–9 units, reconstruct lossily (exact 0.33–0.67, bitwise 0.89–0.97) and spend a rate of about 2.1 nats, well below H(X). At β = 0.25 E10a uses exactly 10 units and reconstructs perfectly. On code10 the balance account's prediction holds exactly: a single copy of a fair bit gets no unit (rate 0.00 nats, distortion 10 ln 2), two or more copies get one unit each. The count grows with L until about 15 and stays there from L = 24 to 64; below that every unit is used.

**Deviations from DESIGN.md:** none. The routine implementation choices above were fixed before any run. **Audit:** pass (2026-10-27). A separate recount was written from the saved posterior arrays (`audit_recount.py`; it does not import or reuse `analyze_pw06.py`, `run_pw06.py` or pwlib's count code). It verified all 241 `MANIFEST.json` hashes and the DESIGN.md freeze, rebuilt the test weights from pw01 `count_test` and the encodings from DESIGN.md, and for all 78 runs recomputed the counts (0.01, 0.001, 0.05, 0.1), active sets, reconstruction, numerical and convergence checks, rate and H(X) (exact match) and the Monte Carlo distortion and −ELBO (within 6 SE, independent draws); its own forward pass reproduces the stored posterior means and reconstruction bits. Gates and P1–P3, S1, S2, with all runs and converged-only, re-derived identically: 1,847 checks, 0 mismatches (`results/full_001/AUDIT.json`, `audit_recompute.json`).

**Limitations:** one architecture (A2), one lr, β = 1 (plus one β = 0.25 cell); E10 results come from two random index permutations only; counts are architecture- and training-relative (Asperti), not a dimension of the data. Raw arrays and checkpoints stay local, with no off-machine backup; the commit is local (no remote).

Evidence: `results/full_001/results.json` (`gates`, `verdicts.all_runs`, `verdicts.converged_only`, `cells`, `per_run`, `descriptive.reproduction_vs_pw02a`), `results/full_001/SUMMARY.md`, `runs/*.json`, `runs_index.json`, `task_gate.json`, `MANIFEST.json`.
