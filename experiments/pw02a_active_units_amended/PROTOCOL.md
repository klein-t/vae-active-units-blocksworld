# pw02a_active_units_amended — protocol (amendment of pw02, frozen 2026-10-21)

**Pilot-triggered amendment** of `pw02_active_units`, registered 2026-10-21
after both pw02 pilots (`pilot_001`, `pilot_002`) ended in
`competence_stop`, and **before any confirmation data was opened** (the
thesis test file has never been read by pw02 or pw02a). The decisions were
made by the project lead after seeing the pw02 pilot evidence (development
data, seed 101), including the pilot AU values logged in the pw02 README.
That disclosure applies to every result of this amendment.

The pw02 protocol (`experiments/pw02_active_units/PROTOCOL.md`, frozen in
the pw02 freeze record `a1559081...42`) remains binding for everything not listed
below: question, targets, substrates, splits, families, latent sizes, batch,
lr, seeds, metrics, AU threshold, gates and thresholds, hypotheses H2.2-H2.4
(as adapted in section 2), forecasts, selection, uncertainty, scope.

## 1. Changes (all pilot-triggered)

1. **A3 removed.** The 4x256 network does not reconstruct raw-state data
   under this recipe, even with KL warm-up (pw02 `pilot_002`: exact
   reconstruction `bw` 0.87, `prod5` 0.08; gate 0.95). The 8 A3 cells
   (`bw` x 3 beta x 2 lr, `code10`, `prod5`; 36 runs) are not trained. A3's
   pilot failure is reported descriptively in pw02, never as an AU result.
   Grid: 58 cells, 210 confirmation runs (24 -> 18 core `bw-gauss_vae`
   cells at 5 seeds); pilot: 20 cells at seed 101.
2. **KL warm-up kept for every cell**, because the pw02 contingency fired:
   KL-side penalty x min(1, t / 6,000), i.e. beta rises linearly from 0 to
   its target over the first 10% of the steps (same schedule as pw02
   `pilot_002`).
3. **60,000 steps for every cell** (was 30,000). Evaluation every 1,000
   steps unchanged; final checkpoint analyzed; no early stopping.
4. **Convergence check amended** (pilot check, pw02 section 10):
   - Cells with a plain objective: dev objective change over the last 20% of
     steps < 2% relative (unchanged). For the plain AE and the top-k SAE,
     whose objective is about 0 at perfect reconstruction, it is also
     accepted when the absolute change is < 1e-3 nats per sample.
   - Learned-gamma cells (activation substrates, gamma learned): the
     objective is unbounded below as gamma -> 0 (Dai & Wipf 2019), so the
     objective test is not meaningful. Converged means: the per-run
     reconstruction gate (pw02 section 8: probe-on-reconstruction bitwise
     >= 0.98 and R^2 >= 0.90) passes both at 80% of training (step 48,000)
     and at the end, **and** the set of active units on the development
     evaluation set is identical at both checkpoints (Gaussian: Burda AU at
     0.01; categorical: activity > 0.01).
   - Implementation: each run evaluates a copy of its weights at step 48,000
     and stores the development metrics, competence and active set
     (`runs/<run>.json` `snapshot`, `active_dev_final`);
     `pwlib.sweep.converged_amended`.
5. **Secondary families unchanged** (beta-TCVAE, categorical VAE, beta=4).
   If they still fail competence in the full run, comparisons that depend
   on them are reported as untested (pw03a, pw04a).

## 2. Hypothesis adaptations forced by removing A3 (decided here, before the full run)

Rule used: keep each threshold's fraction and round **up**, so no criterion
becomes easier to pass.

- **H2.2 (required)** is evaluated on A1, A2 and A4. (i) every gated beta=1
  Bernoulli beta-VAE run (3 archs x 2 lr x 5 seeds) has AU < 25 and is not
  censored; (ii) the information account is the verdict in **>= 3
  architectures, i.e. all three** (was >= 3 of 4). The code10/prod5
  comparators at A1, A2, A4 are unchanged.
- **H2.3 (required)**: the range of seed-median AU across the **6** beta=1
  arch x lr cells is >= 3 units and larger than the largest within-cell seed
  range (was 8 cells). Secondary beta monotonicity: non-increasing in **6 of
  6** cells (was 7 of 8).
- **H2.4 (required)** unchanged (A2 activation cells).
- Experiment-level competence: every remaining required cell (6 `bw` beta=1
  cells, `code10`/`prod5` at A1/A2/A4, 5 `b_lin` learned-gamma cells) has a
  majority of planned seeds passing; else `competence_stop`.

## 3. Runs, order and stop rule

1. Freeze pw02a, pw03a, pw04a together.
2. `pilot_001` (seed 101, 20 cells, development data only; confirmation file
   not opened). Gates task/numerical/competence as in pw02, plus the amended
   convergence check for every run.
3. **Stop rule:** if any gate or the convergence check fails in a way this
   amendment does not cover, stop and report. **No further amendments.**
4. If the pilot passes: pw03a and pw04a pilots on pw02a `pilot_001`; then
   pw02a `full_001` (seeds 1-5 / 1-3, confirmation opened), then pw03a and
   pw04a `full_001` on pw02a `full_001`.

## 4. Compute and storage

CPU, float32, one thread per worker, 8 workers. Scaled from the pw02
measurements (pw02 `pilot_002`: 0.84 CPU-h for 23 cells at 30k steps; full
sweep estimate 17 CPU-h at 30k, A3 share 1.3 CPU-h): **pilot about 1.5 CPU-h,
about 40 min wall** (bounded by the 60k-step beta-TCVAE run); **full sweep
about 31 CPU-h, about 4-5 h wall** (longest run, activation beta-TCVAE, about
2 h). Cap 45 CPU-h, 8 h wall. Disk about 1 GB (pilot) and about 2.6 GB (full),
cap 6 GB; gitignored `raw/` and `models/`.
