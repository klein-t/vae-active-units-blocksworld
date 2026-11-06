# pw03a_fact_alignment_amended — fact alignment on the amended sweep — **COMPLETED: UNSUPPORTED** (2026-10-22, run `full_001`)

Proposal item **P3**. **Pilot-triggered amendment** of `pw03_fact_alignment`
(never run; blocked by the pw02 pilot competence stop), registered with
pw02a on 2026-10-21 before any confirmation data was opened. Same question,
metrics, gates and hypotheses as pw03 (`experiments/pw03_fact_alignment/`).

**Changes** (`PROTOCOL.md`): input is the pw02a amended sweep; **H3.2 bug
fix**: a comparator cell with zero gated seeds now scores untested, not
unsupported (three-valued rule; an untested H3.2 gives status
`inconclusive`), unit-tested; H3.3 untested when a side has no gated seeds;
H3.1 (descriptive) over the 6 remaining beta=1 cells.

**Files:** `PROTOCOL.md`, `DESIGN.json`, `analyze_alignment.py`, `audit.py`;
shared `src/pwlib/alignment.py` (`h32_verdict`, `separated_or_untested`).

## Runs

**`pilot_001`** (pilot on pw02a `pilot_001`; fits on train weights, scored on val weights; test file not opened): `completed`, `pilot_only`. Gates task / numerical / competence (metric validity): true / true / true. Fact oracle Cov30 1.00 (gate >= 0.95); block oracle block-MIG 0.926 (gate >= 0.5); random-init null Cov30 0.00 / 0.00 / 0.00 (gate <= 0.2); sklearn F1 check max abs diff 2.2e-16. 17 models scored (gated pw02a runs + controls). H3.3 comparisons came out untested (beta-TCVAE and categorical VAE have no gated seed), as the amendment specifies. Pilot-only values (seed 101, val weights): k=5 SAE Cov30 0.83; beta-VAE A2 beta=1 lr 3e-4 Cov30 0.83, MonoFrac 0.93. No hypothesis tested. Audit pass (sklearn Cov30 recount, brute-force threshold search reproduced, permuted-oracle negative control). Compute: 4.0 min wall (8 workers), 1,332 CPU-s.

**`full_001`** (full on pw02a `full_001`, cites `pilot_001`; signs/thresholds fit on val weights, scored on test-file weights; activations: 500 val / 300 confirmation sequences; confirmation opened): `completed`, outcome **`unsupported`**. Gates task / numerical / competence (metric validity): true / true / true (fact oracle Cov30 1.00; block oracle block-MIG 0.927; random-init null Cov30 0.00 x 3; sklearn F1 check 2.2e-16). 123 models scored. Audit pass (Cov30 recount with sklearn, brute-force threshold search reproduced, permuted-oracle control; H3.2 re-derived separately with the three-valued rule).

| Required criterion | Verdict | Key numbers (Cov30 = fraction of 30 position facts with best-latent F1 >= 0.9, per seed) |
|---|---|---|
| `h32_topk_sae_covers_position_facts` | **false** | k=5 SAE 0.800 / 0.833 / 0.833 (>= 0.8 in every seed: part 1 holds). Comparators (seed max): beta-VAE beta=1 **0.833** (seeds 0.83, 0.77, 0.83, 0.77, 0.83), which reaches the SAE minimum 0.800, so the measured part fails; beta-TCVAE beta_TC=2 0.733 (below); beta-VAE beta=4 and beta-TCVAE beta_TC=6 have **zero gated seeds (untested comparators)**. Under the frozen three-valued rule a failed measured part makes H3.2 false, not untested |

Secondary and descriptive (not criteria):
- 25/30 = 0.833 is the ceiling the protocol anticipated (one position value per block can be represented by absence); the SAE (k=5) and the A2/A4 beta=1 beta-VAEs both sit at it. The k=10 SAE reaches 0.83-0.87.
- H3.1 (descriptive, former hypothesis): the "beta-VAE units are not single-fact detectors" rule does **not** hold: MonoFrac at the reference cell (A2 beta=1 lr 3e-4) is 0.80-1.00; other beta=1 cells 0.6-1.0 except A1 lr 3e-4 (0.13-0.44). Contrary to the designer's forecast (0.0-0.3).
- H3.3 (block-MIG, categorical vs Gaussian, TC vs VAE): **untested** (categorical VAE and beta-TCVAE beta_TC=6 have no gated seed). Block-MIG is low for every trained model (mostly 0.00-0.11; A2 lr 3e-3 up to 0.39), against 0.93 for the block oracle.
- **Correction (2026-10-22):** "A2 lr 3e-3 up to 0.39" in the bullet above is the maximum over both betas; the 0.39 comes from the beta=0.25 cell (seeds 0.10-0.39). The beta=1 cell (A2 lr 3e-3) peaks at **0.31** (seeds 0.16-0.31). Evidence: `results/full_001/results.json` (`metrics.per_model.*.block_mig`); independent audit `../audits/independent_audit_001/AUDIT_REPORT.md`.
- **Correction (2026-10-27):** MonoFrac counts a unit once when its best F1 is >= 0.9, and a signed threshold can isolate up to two mutually exclusive facts (one per sign). At the reference cell the seeds cover 23-25 position facts with at most 15 active units, so at least 8 units per seed (10 in the seeds covering 25) detect two facts. Read MonoFrac 0.80-1.00 as "high-F1 detectors of one or two facts", not "single-fact detectors". Evidence and literature: `../../docs/EXPERIMENT_LOG.md` (pw03a, correction of 2026-10-27).
- H3.4 (activation substrate `b_raw_w128`): holds as stated: SAE k=5 Cov30 0.57 median (raw states 0.83), beta-VAE MonoFrac 0.14 (raw 0.93), SAE > VAE Cov30 (0.57 vs 0.10).
- Plain AE codes: Cov30 0; PCA top-25: 0.03.
- Compute: CPU, float64 metrics, 8 workers; 16.1 min wall, 7,250 CPU-s; peak parent RSS 0.76 GB; 299 MB on disk (raw/ gitignored).
