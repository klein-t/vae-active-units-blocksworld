# pw02_active_units — active latent units vs pre-fixed target counts — **PILOT COMPETENCE STOP: full run not started** (2026-10-21, runs `pilot_001`, `pilot_002`)

Proposal item **P2**. Registered 2026-10-14, frozen 2026-10-21 (freeze record
sha256 `a1559081...42`). Two pilots ran; both ended in `competence_stop`
because the A3 (4x256) architecture does not reconstruct raw-state data. The
pre-registered KL warm-up contingency fired and did not rescue A3, so per the
frozen rule pw02 stops before the full run. **No hypothesis has been tested.**

**Question.** In overcomplete VAEs (32 latents on raw states, 64 on
activations), does the number of active units (Burda criterion, threshold
0.01) behave like the information content, the per-block factor count, or the
affine dimension of Blocksworld, and how much does it move with architecture,
beta and learning rate? Substrates: (a) raw 36-bit states with a Bernoulli
decoder; (b) final-layer residual activations of five thesis exp46
transformers (d4; width 128 at seeds 1-3, widths 64 and 256 at seed 1) with a Gaussian decoder, as raw
activations (`b_raw`) and as their state-explained linear part (`b_lin`).

**Shared sweep.** This experiment trains the one retained sweep
(`sweep_grid.json`: 66 cells, 246 runs: 5 seeds for the 24 raw-state
beta-VAE cells, 3 for the rest; compute measured in `PROTOCOL.md` section 12) that pw03 and pw04 analyze without retraining.

**Hypotheses (required):** H2.2 (absorbs former H2.1) every gated beta=1 VAE
on raw states has AU < 25, and the count follows the information account
(matches the 10-bit code10 control, stays below the 12.9-bit prod5 control)
in >= 3/4 architectures; H2.3 AU varies across architecture x lr beyond seed
noise; H2.4 a learned-variance Gaussian decoder switches on >= 2 more units
than a fixed one on `b_lin` at each of the 5 source checkpoints.
Forecasts and decision rules: `PROTOCOL.md`.

**Files.** `PROTOCOL.md`, `DESIGN.json`, `sweep_grid.json` (grid), entry
scripts `extract_activations.py` (step 0b, targets_b.json), `train_sweep.py`
(shared sweep), `analyze_active_units.py`, `audit.py`. Shared code in
`src/pwlib/` (`data`, `models`, `training`, `metrics`, `activations`,
`sweep`, `runio`); tests in `tests/`.

**Depends on:** pw01 completed run (`dimension_candidates.json`, legality
checker `src/pwlib/blocksworld.py`). **Consumed by:** pw03, pw04.

**Open before freeze:** none (decisions recorded below).

## Pre-freeze changes (2026-10-14, decided by the project lead; designs not yet frozen)

1. **Seeds.** 5 seeds (1-5) for the 24 raw-state `bw-gauss_vae` cells
   (network size x beta x lr); 3 seeds for controls, secondary families and
   activation cells. Majority competence rule per cell (>= 3/5, >= 2/3);
   power/tolerance language updated (PROTOCOL sections 6, 9). Run RNG seed
   derived from `sha256(cell_id|seed)`.
2. **Transformer seeds.** Width 128 now uses exp46 `d4_w128_s1/s2/s3`
   (8 new `b_*_w128s2/s3` cells); widths 64/256 stay at s1. H2.4(i) is
   required at each of the 5 source checkpoints; new secondary (v) reports the
   width-128 transformer-seed spread.
3. **H2.1 merged into H2.2** as part (i); required criteria are now
   `h22_information_account`, `h23_architecture_dependence`,
   `h24_learned_gamma_more_units`.
4. **Kept:** empirical state frequencies for training, the 2x1024 network
   (A4), dense MLPs only.
5. **Environment.** `transformer-lens==2.15.4` and `transformers<5` pinned
   (TL 4.x removed HookedTransformer); exp46 recipe re-verified on train-file
   sequences for all five checkpoints; activations paired with the current
   state Y[k].
6. **Metric conventions** per `docs/METRICS_SPEC.md`: Asperti Eq. 13 check
   (tolerance 0.1) and reconstruction gain logged per unit; beta_norm logged;
   TC reported in nats by MWS with Chen's constants (as specified) and by MSS,
   because tests show the Chen-constant MWS carries a (D-1) log N offset.
7. **Implementation details fixed:** all probes fit on 20,000 train positions
   (cost); K_probe scanned in increasing q with early stop; per-split
   subsample generator seeds 0/1/2; competence evaluated on development data;
   (b) reconstructions retained as probe-read bits; automatic single same-seed
   retry on a crash/NaN.
8. **Pilot cells 19 -> 23:** Blocksworld beta-VAE A1 and A3 (beta=1, lr
   3e-4) and code10/prod5 at A3 added so every architecture is exercised
   before the full run. Reason: a 30,000-step check on the synthetic controls
   (seed 101, no Blocksworld data) showed A1, A2, A4 reconstructing code10 and
   prod5 exactly, but A3 (4x256) collapsing (code10 exact 0.60, prod5 exact
   0.00); if this recurs in the pilot, the A3 control cells cannot pass
   competence and H2.2 would end in `competence_stop` unless amended.
9. **Compute measured** (PROTOCOL section 12): about 17 CPU-h and 2.5-3 h wall
   on 8 workers for the full sweep, about 1 CPU-h / 35 min for the pilot; the
   earlier 3.5 CPU-h estimate missed the beta-TCVAE cost and parallel
   contention.
10. **Code.** Stubs replaced by the implementation in `src/pwlib/` and thin
   entry scripts; `vae_models.py` removed (models live in `src/pwlib/models.py`).

## Pre-freeze changes (2026-10-21, decided by the project lead before freeze and before any real outcome)

11. **Pre-registered A3 contingency (KL warm-up).** If the pilot (`pilot_001`)
   shows any A3 (4x256) cell on raw-state-type data
   (`bw`/`code10`/`prod5-gauss_vae-A3-b1-lr0.0003`) failing the per-run
   competence gate, the full run applies a linear KL warm-up to **all**
   architectures and all cells, not just A3: beta rises linearly from 0 to its
   target over the first 10% of steps (3,000 of 30,000). It is first exercised
   in `pilot_002` (seed 101, the same 23 pilot cells, development data only),
   because the full run must cite a pilot whose gates passed with the same
   hyperparameters; a `pilot_002` gate failure stops the run. Decided before
   seeing any real outcome; if it fires it is disclosed as a pilot-triggered
   amendment. Implemented behind `sweep_grid.json` `training.kl_warmup`
   (default `null` = off) and `training.kl_warmup_contingency`;
   `train_sweep.py --apply-kl-warmup-contingency pilot_001` re-checks the
   trigger; unit tests in `tests/test_pipeline_parts.py`. Recorded in
   `PROTOCOL.md` section 10 and `DESIGN.json` (`contingencies`).

## Runs

| Run | Kind | Config | Status | Outcome | Gates (task / numerical / competence) | Audit |
|---|---|---|---|---|---|---|
| `pilot_001` | pilot, seed 101, 23 cells, dev data only | frozen grid, no KL warm-up | `competence_stop` | untested | true / true / **false** | pass |
| `pilot_002` | pilot, seed 101, same 23 cells, dev data only | pre-registered A3 contingency: linear KL warm-up (beta 0 -> target over the first 3,000 of 30,000 steps) on all cells | `competence_stop` | untested | true / true / **false** | pass |

Confirmation data (thesis test file) were never opened. No retries were needed
(no crash or NaN in 46 training runs).

### What failed (required-cell competence, pilot evaluation on val weights)

Per-run gate on raw-state-type data: exact reconstruction >= 0.95 and bitwise >= 0.99.

| Required pilot cell | pilot_001 exact / bitwise | pilot_002 (warm-up) exact / bitwise |
|---|---|---|
| `bw-gauss_vae-A3-b1-lr0.0003` | 0.101 / 0.902 (fail) | 0.872 / 0.994 (fail) |
| `code10-gauss_vae-A3-b1-lr0.0003` | 0.600 / 0.971 (fail) | 1.000 / 1.000 (pass) |
| `prod5-gauss_vae-A3-b1-lr0.0003` | 0.000 / 0.916 (fail) | 0.080 / 0.974 (fail) |
| all other required pilot cells (A1, A2 both lr, A4 on bw; A2 code10/prod5; `b_lin_w128` learned gamma) | pass | pass |

The contingency trigger (any A3 raw-state-type cell failing) fired in
`pilot_001` (all three failed; re-checked by `train_sweep.py
--apply-kl-warmup-contingency pilot_001`, record
`results/pilot_002/kl_warmup_contingency.json`). The warm-up rescued A3 on
`code10` only. Under the frozen rule a `pilot_002` gate failure stops the run:
H2.2 needs A3 cells (part (i) and the per-architecture verdict), so the full
run would end in `competence_stop` for H2.2. This is a competence stop of the
4x256 network, not evidence about any hypothesis.

### Other pilot checks (PROTOCOL section 10)

- **Convergence (dev objective change < 2% over the last 20% of steps): fails**
  in both pilots, which under the frozen protocol requires an amendment (more
  steps) before any full run. Non-converged in `pilot_002`: `b_lin_w128` and
  `b_raw_w128` learned-gamma VAEs (objective still falling as log gamma
  shrinks; `pilot_002` change over the last 20%: b_lin -269 -> -341 nats,
  -27%; b_raw -9.8%), the categorical VAE (-4.1%; Gumbel temperature still
  annealing), A3 on `bw` (-3.5%),
  and the plain AE and k=5 SAE (objective about 0 at perfect reconstruction, so
  the relative criterion is degenerate there). All Blocksworld beta-VAE cells
  at A1, A2 (both lr) and A4 converged in both pilots.
- AU < latent size on Blocksworld: holds (no censored run).
- Log gamma: learned-gamma activation VAEs end at -7.8 (`b_lin`, floor -9) and
  -3.2 (`b_raw`); not at the floor.
- Secondary families failing per-run competence (not required for pw02, but
  consumed by pw03/pw04): beta-TCVAE beta_TC=6 (exact 0.38 / 0.50),
  categorical VAE (0.40 / 0.54), beta=4 beta-VAEs (exact <= 0.03; anticipated
  in section 13), `indep` (no support; descriptive).
- Pilot AU values (development data, seed 101, pilot evidence only, not
  confirmation): `pilot_001` / `pilot_002` Blocksworld beta=1 lr 3e-4: A1 15 / 16,
  A2 10 / 15, A4 14 / 16; A2 lr 3e-3: 4 / 10; `code10` A2 10 / 10;
  `prod5` A2 18 / 15; `b_lin_w128` learned vs fixed gamma 12 vs 12 / 18 vs 14.
  Source gates (all five exp46 checkpoints): current-state probe val bitwise
  0.998-1.000, exact 0.94-0.99, beats the next-state probe; `targets_b`:
  K_pca99 39 / 88 / 91 / 93 / 208 and K_probe 25 / 26 / 27 / 27 / 31 for
  w64 / w128 / w128s2 / w128s3 / w256.

### Compute (CPU, float32, 1 thread per worker, 8 workers, M4 Max 14 cores)

| Run | Extraction wall | Sweep wall | Training worker-seconds | Peak RSS (parent) |
|---|---|---|---|---|
| `pilot_001` | 38 s | 22.4 min | 4,829 (1.34 CPU-h) | 3.1 GB (extraction) |
| `pilot_002` | about 25 s (slowest source 22 s) | 18.5 min | 3,012 (0.84 CPU-h) | 1.2 GB (sweep) |

Longest run: beta-TCVAE (1,339 s / 1,108 s). Disk 1.1 GB per pilot
(gitignored `raw/`, `models/`).

### Audit

Both pilots: `audit.py` pass (AU recount from retained posterior means with
explicit loops, competence recomputed from stored bits, negative controls
constant-mean = 0 units and duplicated unit counted per copy; 0 mismatches).
Same author and session (not independent replication).

### Consequence and open decision (for the project lead)

pw02 cannot proceed to the full run without a new amendment directory
(e.g. `pw02a_...`), which the frozen rules require for any change to steps,
grid or gates and which is not pre-registered. Two issues need a decision:
(1) A3 does not reach competence on raw states even with warm-up; (2) the
convergence check fails for the learned-gamma activation cells (required for
H2.4) and the categorical VAE. pw03 and pw04 pilots were not started: both
require a completed, gated and audited pw02 run, which does not exist.

Evidence: `results/pilot_00{1,2}/` (`results.json`, `au_table.json`,
`task_gate.json`, `source_gate.json`, `targets_b.json`, `runs/*.json`,
`audit_recompute.json`, `AUDIT.json`, `MANIFEST.json`; raw operands and
checkpoints local only, no off-machine backup).

## Follow-up (2026-10-21)

Superseded for execution by the pilot-triggered amendment `pw02a_active_units_amended` (registered and frozen 2026-10-21, before any confirmation data was opened; see `experiments/pw02a_active_units_amended/`). This record is unchanged.
