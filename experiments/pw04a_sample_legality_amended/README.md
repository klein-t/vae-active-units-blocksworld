# pw04a_sample_legality_amended — prior-sample legality on the amended sweep — **PILOT COMPETENCE STOP: ENDS UNTESTED** (2026-10-21, run `pilot_001`)

Proposal item **P4**. **Pilot-triggered amendment** of
`pw04_sample_legality` (never run; blocked by the pw02 pilot competence
stop), registered with pw02a on 2026-10-21 before any confirmation data was
opened. Same question, legality definition, baselines, gates and hypotheses
as pw04 (`experiments/pw04_sample_legality/`).

**Changes** (`PROTOCOL.md`): input is the pw02a amended sweep; A3 removed
from the beta=1 cells of H4.1, H4.2 and the competence gate (A1, A2, A4 x 2
lr x 5 seeds); H4.4 untested when a side has no gated seeds.

**Files:** `PROTOCOL.md`, `DESIGN.json`, `analyze_legality.py`, `audit.py`.

## Runs

**`pilot_001`** (pilot on pw02a `pilot_001`; val-weight evaluation distribution; test file not opened): **`competence_stop`**, outcome untested. Gates task / numerical / competence: true / true / **false**. Task: checker marks all evaluation states legal, reproduces the 866 table, rejects 10^5 random vectors. Numerical: exact baselines agree with 10^6-sample Monte Carlo within 3 SE (B0 1.26e-8; B1 2.54e-4; **B2 0.540** with train marginals; B2 uniform 0.111). **Competence failed:** aggregate-posterior legality (z ~ q(z|x), mode decoding) is below 0.95 in every required beta=1 cell: A1 0.630, A2 lr 3e-4 0.822, A2 lr 3e-3 0.883, A4 0.861 (all at their pw02a-gated runs, which reconstruct exactly from the posterior mean; the sampler was checked: z = mu + exp(logvar/2) eps). The pre-registered pw04 stopping rule makes this a terminal `competence_stop` for pw04a: the full run cannot start and no H4.1/H4.2 verdict exists. Pilot-only, descriptive (seed 101): prior mode-decoded legality at beta=1: A1 0.168, A2 0.435 / 0.496, A4 0.502; beta=0.25 A2 cells pass competence (0.954 / 0.970). code10 valid codewords 0.952 (prior), prod5 in-support 0.796 (fails control competence). H4.4 untested (no gated categorical VAE). Audit pass (stopped run; rates, B2 and criteria recomputed, negative controls). Compute 7 s wall.

## Final status and descriptive evidence (2026-10-22)

**pw04 / pw04a end untested.** The pilot competence stop stands; there is no full run and no H4.1/H4.2 verdict. No further amendment (project lead, 2026-10-22). Everything below is **descriptive only: not a hypothesis test** and not evidence for or against H4.1-H4.5.

**(a) Pilot `pilot_001`: development data (val weights), one seed (101).** Aggregate-posterior legality (z ~ q(z|x), mode decoding) at beta=1: A1 0.630, A2 0.822 (lr 3e-4) / 0.883 (lr 3e-3), A4 0.861 (gate 0.95); beta=0.25 A2 0.954 / 0.970. Exact baselines (model-free): B0 1.26e-8, B1 2.54e-4, B2 0.540 with train block marginals (0.111 uniform), all matching 10^6-sample Monte Carlo within 3 SE. code10 valid codewords: 0.952 (prior), in-support 0.986 (aggregate posterior). Prior legality at beta=1: 0.17-0.50.

**(b) Unregistered descriptive pass on pw02a `full_001`** (requested by the project lead): the frozen pw04a analysis code, unchanged (working copy checked byte-identical to the frozen snapshot), applied to the 210-run full sweep with test-file state weights, via `descriptive_full_legality.py`; output `descriptive/results/pw02a_full_001/results.json` (header `freeze_sha256: descriptive-unregistered`; no RUN.json; not a pw04a run). Its own competence gate fails again (0 of 5 seeds pass in every beta=1 cell). 5 seeds per Blocksworld cell, 3 per control:

| Cell (beta=1 unless stated) | Prior legality (mode) per seed | Aggregate-posterior legality |
|---|---|---|
| A1 lr 3e-4 / 3e-3 | 0.11-0.17 / 0.32-0.34 | 0.55-0.64 / 0.73-0.74 |
| A2 lr 3e-4 / 3e-3 | 0.41-0.46 / 0.46-0.55 | 0.81-0.85 / 0.86-0.90 |
| A4 lr 3e-4 / 3e-3 | 0.49-0.54 / 0.54-0.58 | 0.86-0.87 / 0.89-0.90 |
| beta=0.25, A2 / A4 (both lr) | 0.34-0.47 / 0.41-0.52 | 0.945-0.974 (passes 0.95 in 19 of 20 runs) |
| beta-TCVAE beta_TC=2 (A2) | 0.46 | 0.74-0.75 |
| code10 A1 / A2 / A4 | valid codewords 0.99 / 0.95 / 0.98 | in-support 0.996-0.998 / 0.988 / 0.994 |
| prod5 A1 / A2 / A4 | Blocksworld-legal position bits 0.05-0.07 (uniform per-block rate 0.111) | in-support 0.70-0.84 |

Against B2 = 0.540, beta=1 prior legality is below the per-block product baseline in 24 of 30 runs; above it only in 4 of 5 A4 lr 3e-3 seeds (0.559-0.578) and 2 of 5 A2 lr 3e-3 seeds (0.545, 0.550). Stratified Spearman (frozen code, beta=0.25 stratum only, the only stratum with >= 8 gated runs): rho -0.25, bootstrap 95% CI [-0.69, 0.37]. Compute 22 s wall.
