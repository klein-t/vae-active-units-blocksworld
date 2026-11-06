# pw07_unit_kl_and_matched_legality — KL per latent unit, and sample legality on the baseline's footing (post hoc, descriptive) — **COMPLETED: DESCRIPTIVE** (2026-10-30, run `analysis_001`)

- **Proposal items:** P2 (why about 15 active units), P4 (are decoded samples legal).
- **Added 2026-10-30, after the main results, in response to review.** Two examiner-style reviews of report v1.4/v1.5 raised two objections:
  - **A (examiner, "Classic theory" and weakness 1):** one Gaussian latent can index all 866 states, so the active count is bounded by no data dimension; it is the rate divided by the KL each unit carries (7.84 nats / 15 ≈ 0.5, 7.92 / 10 ≈ 0.8). The KL per unit is never shown.
  - **B (both reviews):** the per-block baseline (0.540) derives clear(X) and hand_empty exactly from its sampled positions and is well formed by construction, while the VAE must generate all 36 bits. "Prior samples fall below the baseline in 24 of 30 runs at β = 1" partly reflects this asymmetry.
- **Nature:** post hoc and descriptive. No training, no forward pass: only retained arrays. No hypothesis, no pass/fail criterion, no threshold chosen after seeing the outcome. The only thresholds are Burda's 0.01 (the project's AU rule), the per-block baseline (fixed in pw04a) and 95% Wilson intervals. **Not registered or frozen** (no DESIGN/FREEZE/RUN.json); the confirmation data were already opened by pw02a–pw06.
- **Inputs (hash-checked):** pw02a `full_001` `raw/<run>.npz`, `au_table.json`, `au_per_run.json`, `raw/data_context.npz` (against pw02a `MANIFEST.json`); pw06 `full_001` `raw/<run>.npz`, `results.json` (against pw06 `MANIFEST.json`); pw04a descriptive samples `descriptive/results/pw02a_full_001/raw/samples_*.npz` (against pw05 `input_hashes.json`; the three β-TCVAE sample files had no recorded hash, so theirs are written to `results.json`).
- **Run:** `analysis_001`, CPU, 8 worker processes, about 27 s wall. One attempt for the numbers; the script was re-run twice only to change the figure layout and to split the closed-form KL check into a relative check (units above 0.01 nats) and an absolute one (below). Numbers did not change between runs.

## Method

**Split.** Everything uses the test split (`conf` weights), as the report's counts and rates (pw02a `analyze_active_units.py` uses `key = 'conf'` for full runs; pw05 and pw06 likewise).

**A. KL per unit.** For every raw-state Gaussian β-VAE run of pw02a at β = 1 (Blocksworld: 3 networks × 2 learning rates × 5 seeds = 30; code10 and prod5: 3 networks × 3 seeds at lr 3·10⁻⁴) and, supplementary, at β = 0.25 (30 runs), and for every Gaussian-latent run of pw06 (re-encodings, code10 with c copies, latent sizes 8–64; 75 runs): stored per-unit KL `conf_kl_per_unit` (nats, frequency-weighted mean of the closed-form Gaussian KL), sorted; total rate R = its sum; active units by Burda (Var_x μ > 0.01, recomputed and required to equal the stored count and set); KL in active vs inactive units; mean, median, min and max KL per active unit; active units above ln 2 (one bit); R − H(X). Per configuration: medians and ranges over seeds.

**B. Legality on equal terms.** For the 20,000 mode-decoded prior samples and 20,000 aggregate-posterior samples of each Blocksworld run with stored samples (Gaussian β-VAE at β = 1 and 0.25, 30 runs each; β-TCVAE at β_TC = 2, 3 runs; no samples exist at β = 4): (i) original legality (pw01 checker, required to equal the 866-table and the stored flags); (ii) **re-derived legality**: clear(X) and hand_empty replaced by their values derived from the sample's 30 position facts, exactly as the baseline derives them, then the checker (required to equal `pwlib.legality.legal_from_position_bits` and pw05's `position_legal`); (iii) the funnel: well formed (every block in exactly one place), the product of the per-block well-formed fractions, constraints between blocks given well formed (= re-derived legal / well formed), the same quantity if the model's blocks were independent with its own per-block marginals, and derived facts consistent given legal positions. Baseline: B2 = 0.53995 (each block drawn independently from its training marginal, clear and hand_empty derived; recomputed exactly and equal to pw04a's stored value; 0.5405 with test marginals). A run is "below" when its rate is below 0.53995; the Wilson 95% interval (n = 20,000, half-width about 0.007) is reported alongside.

## Results

### A. The count is the rate divided by the KL per unit

Per configuration, median [range] over seeds; nats; β = 1 unless stated.

| Configuration | Active units | Rate R | R − H(X) | KL in inactive units | Mean KL per active unit | Median KL per active unit | Smallest–largest active unit | Active units > 1 bit |
|---|---|---|---|---|---|---|---|---|
| 1×64, lr 3e-4 | 16 [15, 16] | 8.14 [8.10, 8.15] | 1.67 | 0.0006 | 0.51 [0.51, 0.54] | 0.53 | 0.30–0.64 | 0 |
| 1×64, lr 3e-3 | 15 [15, 16] | 8.13 [8.05, 8.22] | 1.66 | 0.0042 | 0.54 [0.51, 0.54] | 0.62 | 0.28–0.66 | 0 |
| **2×256, lr 3e-4 (reference)** | 15 [14, 15] | 7.84 [7.84, 7.90] | 1.37 | 0.0012 | **0.52** [0.52, 0.56] | 0.60 | 0.18–0.73 | 1 |
| **2×256, lr 3e-3** | 10 [9, 11] | 7.92 [7.83, 7.96] | 1.45 | 0.0067 | **0.79** [0.72, 0.88] | 0.79 | 0.16–1.34 | 6 |
| 2×1024, lr 3e-4 | 15 [15, 16] | 7.76 [7.72, 7.85] | 1.29 | 0.0017 | 0.52 [0.49, 0.52] | 0.59 | 0.16–1.02 | 1 |
| 2×1024, lr 3e-3 | 15 [14, 15] | 7.90 [7.89, 8.01] | 1.43 | 0.0067 | 0.53 [0.53, 0.56] | 0.59 | 0.12–0.96 | 3 |
| code10 1×64 / 2×256 / 2×1024 | 10 / 10 / 10 | 8.52 / 8.63 / 8.56 | 1.59 / 1.70 / 1.63 | ≤ 0.002 | 0.85 / 0.86 / 0.86 | 0.86–0.89 | 0.79–0.94 | 10 |
| prod5 1×64 | 16 [15, 17] | 8.14 [8.02, 8.20] | −0.82 | 0.0005 | 0.51 [0.47, 0.54] | 0.56 | 0.23–0.67 | 0 |
| prod5 2×256 | 17 [16, 19] | 8.47 [8.40, 8.51] | −0.49 | 0.0009 | 0.50 [0.44, 0.53] | 0.41 | 0.22–0.83 | 5 |
| prod5 2×1024 | 23 [22, 23] | 8.33 [8.32, 8.35] | −0.63 | 0.0010 | 0.36 [0.36, 0.38] | 0.30 | 0.22–0.82 | 2 |
| *β = 0.25, six configurations* | 15–16 (11 at 2×256 lr 3e-3) | 10.0–12.4 | 3.5–5.9 | ≤ 0.013 | 0.68–0.78 (0.91 at 2×256 lr 3e-3) | 0.76–1.02 | | 8–11 |

pw06 follow-up (2×256, lr 3·10⁻⁴, 5 seeds; median [range]):

| Condition | Active units | Rate R | R − H(X) | Mean KL per active unit | Active units > 1 bit |
|---|---|---|---|---|---|
| 36 facts, L = 32 | 15 [14, 15] | 7.84 [7.84, 7.90] | 1.37 | 0.52 [0.52, 0.56] | 1 |
| 30 position facts | 11 [11, 12] | 6.49 [6.46, 6.53] | 0.02 | 0.59 [0.54, 0.59] | 4 |
| 10-bit index, order 1 / 2 | 6 / 7 | 2.12 / 2.07 | −4.35 / −4.40 | 0.32 / 0.30 | 1 / 1 |
| 10-bit index, order 1, β = 0.25 | 10 [10, 10] | 8.47 [8.36, 8.51] | 2.00 | 0.85 [0.84, 0.85] | 9 |
| L = 8 / 10 / 12 | 8 / 10 / 12 (all) | 7.95 / 7.94 / 7.93 | 1.46–1.48 | **0.99 / 0.79 / 0.66** | 8 / 8 / 5 |
| L = 16 / 24 / 64 | 14 / 15 / 15 | 7.92 / 7.87 / 7.89 | 1.40–1.45 | 0.57 / 0.52 / 0.52 | 3 / 1 / 1 |
| code10, c = 1 | 0 | 0.00 | −6.93 | – | – |
| code10, c = 2 / 3 / 4 | 10 / 10 / 10 | 6.89 / 8.20 / 8.82 | −0.04 / 1.27 / 1.89 | **0.69** / 0.82 / 0.88 | 5 / 10 / 10 |

- **The examiner's arithmetic holds.** Reference: median rate 7.841 nats, median count 15, ratio 0.523; median of the per-run mean KL per active unit 0.523 nats, below one bit (ln 2 = 0.693). 2×256 at lr 3·10⁻³: 7.917 / 10 = 0.792; per-run mean 0.791.
- **Across the 30 Blocksworld runs at β = 1 the rate varies little, the count a lot:** rate 7.72–8.22 nats (about ±3%), count 9–16, mean KL per active unit 0.49–0.88 nats. Rank correlation of count with mean KL per active unit −0.91, with rate +0.40. Since count × mean KL per active unit = rate minus the inactive units' KL (< 0.0075 nats; at least 99.9% of the rate sits in the active units), the −0.91 is close to an identity given a near-constant rate; the informative fact is the near-constant rate.
- **Latent size makes it visible.** With 8, 10 or 12 latents every latent is active, every seed reconstructs every test state exactly, and the rate is the same as with 32 (7.93–7.95 nats); each unit then carries more KL, 0.99, 0.79 and 0.66 nats on average. From 24 latents on, the model settles at 15 units of about 0.52 nats. The L = 8 runs have a lower median −ELBO than L = 32 (9.30 vs 9.45, pw06's 128-draw estimator).
- **Units are not bits, and not equal.** Only code10 with two copies sits at one bit per unit (0.689 nats against ln 2 = 0.693; active units span 0.66–0.71 across seeds). In the median curves of the five 15–16-unit Blocksworld configurations, the ten largest units carry about 0.5–0.7 nats each (rank 1 up to 1.0) and the remaining five or six 0.01–0.51, with a step after rank 10 in four of the five (the 1×64 network at lr 3·10⁻⁴ declines gradually; figure, panel a). That step sits where pw05's post hoc "ten pair units plus about five others" reading would put it; whether the top ten are the reciprocal units was not checked here. The 10-unit configuration spreads more (0.16–1.34; six units above a bit).
- **β = 0.25** raises the rate by 2.2–4.3 nats while the count stays at 15–16 (11 at 2×256 lr 3·10⁻³): the extra rate goes into the same units (0.68–0.78 nats each). prod5 keeps its rate below H(X) and its count rises with the network (16 → 23) as the KL per unit falls (0.51 → 0.36).
- **Per-unit KL is not per-unit information.** Each unit's KL is its mutual information with x plus the mismatch between its aggregate posterior and the prior, and units need not be independent. The rate exceeds H(X) by 1.25–1.75 nats in every β = 1 run, which bounds the joint mismatch between aggregate posterior and prior from below; it does not say how that mismatch is shared among the units.

**Figure** `results/analysis_001/fig_unit_kl.png` (and `.pdf`; draft, not in `report/`): sorted per-unit KL (linear scale, nats), thin lines per seed, thick line the per-configuration median, dashed line one bit; legend gives median units and median rate. Panels: (a) the six Blocksworld configurations at β = 1; (b) code10 and prod5 at the three networks; (c) pw06 re-encodings and code10 copies; (d) pw06 latent size. Four panels rather than two because (c) and (d) carry the cleanest evidence: (d) holds the network and data fixed and shows the same rate split over 8, 10, 12 or 15 units. **What a reader should take from it:** on Blocksworld the total rate stays near 7.8–8.1 nats across networks, learning rates and latent sizes, and the count is how many units share it, about 0.5 nats (less than a bit) each when latents are plentiful and more when they are scarce. Only the twice-copied fair bits of code10 come out at one bit per unit, so a count of active units reads as a number of bits only there.

### B. Legality on equal terms

Per configuration, median [range] over seeds; prior samples unless stated; baseline 0.540. "Below" counts runs (of 5, or 3 for the β-TCVAE) whose rate is below 0.53995.

| Configuration | Prior legal (original) | Prior legal (re-derived) | Below baseline, original / re-derived | Well formed | max abs(WF − product) | Between-block constraints given WF | Same, independent blocks | Derived consistent given legal positions | Aggregate posterior, original / re-derived |
|---|---|---|---|---|---|---|---|---|---|
| 1×64, lr 3e-4 | 0.152 [0.110, 0.166] | 0.323 [0.282, 0.347] | 5 / 5 | 0.369 | 0.006 | 0.870 | 0.656 | 0.465 | 0.615 / 0.748 |
| 1×64, lr 3e-3 | 0.326 [0.321, 0.336] | 0.500 [0.492, 0.513] | 5 / 5 | 0.540 | 0.002 | 0.924 | 0.710 | 0.652 | 0.734 / 0.816 |
| 2×256, lr 3e-4 | 0.431 [0.413, 0.464] | 0.593 [0.581, 0.629] | 5 / 0 | 0.635 | 0.007 | 0.933 | 0.635 | 0.737 | 0.828 / 0.885 |
| 2×256, lr 3e-3 | 0.511 [0.463, 0.550] | 0.655 [0.612, 0.685] | 3 / 0 | 0.690 | 0.013 | 0.948 | 0.607 | 0.788 | 0.876 / 0.918 |
| 2×1024, lr 3e-4 | 0.500 [0.494, 0.539] | 0.639 [0.636, 0.681] | 5 / 0 | 0.673 | 0.012 | 0.949 | 0.614 | 0.784 | 0.865 / 0.908 |
| 2×1024, lr 3e-3 | 0.560 [0.539, 0.578] | 0.692 [0.675, 0.715] | 1 / 0 | 0.719 | 0.012 | 0.963 | 0.622 | 0.809 | 0.892 / 0.929 |
| β = 0.25, 1×64, lr 3e-4 | 0.078 | 0.225 | 5 / 5 | 0.278 | 0.009 | 0.815 | 0.656 | 0.345 | 0.883 / 0.931 |
| β = 0.25, 1×64, lr 3e-3 | 0.367 | 0.544 | 5 / 2 | 0.572 | 0.005 | 0.941 | 0.773 | 0.670 | 0.939 / 0.962 |
| β = 0.25, 2×256, lr 3e-4 | 0.368 | 0.522 | 5 / 4 | 0.578 | 0.013 | 0.909 | 0.630 | 0.706 | 0.957 / 0.973 |
| β = 0.25, 2×256, lr 3e-3 | 0.457 | 0.592 | 5 / 0 | 0.633 | 0.018 | 0.931 | 0.624 | 0.774 | 0.970 / 0.981 |
| β = 0.25, 2×1024, lr 3e-4 | 0.430 | 0.581 | 5 / 0 | 0.624 | 0.016 | 0.932 | 0.631 | 0.742 | 0.964 / 0.977 |
| β = 0.25, 2×1024, lr 3e-3 | 0.511 | 0.648 | 5 / 0 | 0.681 | 0.024 | 0.950 | 0.633 | 0.788 | 0.973 / 0.983 |
| β-TCVAE β_TC = 2, 2×256 | 0.459 | 0.626 | 3 / 0 | 0.655 | 0.005 | 0.953 | 0.640 | 0.733 | 0.745 / 0.828 |

Counts below the baseline (prior / aggregate posterior):

| Set | Original | Re-derived (Wilson interval entirely below / entirely above) | Between-block constraints given WF |
|---|---|---|---|
| β = 1, 30 runs (headline) | **24** / 0 | **10** (10 / 20) / 0 | 0 / 0 |
| β = 0.25, 30 runs | 30 / 0 | 11 (11 / 17; 2 overlap 0.540) / 0 | 0 / 0 |
| β-TCVAE β_TC = 2, 3 runs | 3 / 0 | 0 (0 / 3) / 0 | 0 / 0 |

- **On equal terms, prior samples beat the baseline in 20 of the 30 β = 1 runs**: every run of the 2×256 and 2×1024 networks, each with its Wilson interval above 0.540. The 10 runs below are the 10 runs of the 1×64 network. The original "24 of 30 below" falls to 10 once the derived facts are judged as the baseline judges them: re-derivation adds 0.13–0.17 (median per configuration) to prior legality.
- **What still separates the two is well-formedness.** Re-derived legality cannot exceed the well-formed fraction, and the baseline is well formed by construction. In the 1×64 runs only 0.33–0.56 of prior samples are well formed; in eight of the ten runs this alone keeps them below 0.540, and in the other two (lr 3·10⁻³, seeds 3 and 4: 0.548 and 0.555 well formed) the constraints between blocks also fall short. Among well-formed samples, constraints between blocks hold in 0.87–0.96 (configuration medians), above the baseline's 0.540 and above the 0.61–0.71 that independent blocks with the model's own marginals would give, **in every run**.
- **Aggregate posterior:** above the baseline in every run, original or re-derived (re-derived 0.75–0.93 at β = 1).
- **Failure types among well-formed prior samples** (β = 1, medians; may overlap): two blocks on one block 0.016–0.062, a cycle 0.012–0.028, a held block supporting another 0.006–0.033, two blocks held 0.004–0.016.

**Reading.** The reviewers' objection is borne out. Most of the gap between the VAE's prior samples and the per-block baseline came from the derived facts, which the baseline never has to generate. On the baseline's own footing, the two larger networks beat it in every run; the smallest network does not, because 44–67% of its samples misplace a block. The fair summary is that malformed blocks and inconsistent derived facts, not the constraints between blocks, limit legality. **Limits:** re-derivation discards the model's own clear/hand_empty bits, so it repairs the samples; the baseline's well-formedness advantage remains and is not removed here. Projecting a malformed block onto its most probable position would need the decoder's Bernoulli means, which were not stored, so it would require a forward pass, outside this analysis's brief. "Constraints given well formed" conditions on a model-dependent subset. Everything is mode-decoded, at one sample size per run and a few seeds, and descriptive.

## Checks (all recomputed from the raw arrays by `analyze_pw07.py`)

- A: recomputed active sets and counts equal pw02a's stored ones in every run (the script raises otherwise) and pw06's counts in every run; rate equals pw05's `rate_bits` (|Δ| < 1e-9) and pw06's `rate_nats` (|Δ| < 1e-9); pw06 E36-L32 per-unit KL equals the pw02a reference cell's; stored per-unit KL equals the closed form from stored μ, log σ² to 1.8·10⁻⁷ relative (units above 0.01 nats) and 5·10⁻⁹ nats absolute (below).
- B: the checker equals the 866-table on every original and re-derived sample; stored legal flags and pw04a rates reproduced exactly; re-derived legality equals `legal_from_position_bits` and pw05's `position_legal` exactly; no originally legal sample turns illegal; every re-derived legal sample is well formed; baseline equals pw04a's stored B2 to 1e-12.
- **Against report v1.5 draft** (`report/versions/report-v1.5-draft.pdf`, text extracted and matched to `v1.5_src`): 7.84 nats / 15 units and 7.92 / 10 (medians 7.841 and 7.917) match; the rate excess "1.2–1.8 nats per run" is 1.25–1.75; baseline 0.540 matches; "24 of 30" below at β = 1 and "no run beats the baseline" at β = 0.25 match the original legality; prior legality 0.1096–0.578 (printed 0.109–0.578, outward) and aggregate posterior 0.5465–0.8983 (printed 0.546–0.899); well formed 0.369–0.719 (printed 0.36–0.72); max abs(WF − product) 0.0134 (printed "within 0.014"); between-block given WF 0.870–0.963 (printed 0.87–0.97), independent 0.607–0.710 (printed 0.60–0.72), derived 0.465–0.809 (printed 0.46–0.81), aggregate-posterior WF 0.767–0.936 (printed 0.76–0.94) and between-block minimum 0.966 (printed "at least 0.966"). The follow-up table's rates (7.84, 6.49, 2.12, 2.07, 0.00, 6.89, 8.20, 8.82, 7.92, 7.87, 7.89; 7.93–7.95 printed as 7.92–7.96) all match. **No mismatch**; every printed range is the outward-rounded range of these values. Note that "within 0.014" holds at β = 1 only (β = 0.25 reaches 0.024; the report states it for β = 1).

## Limitations

Post hoc on already-opened confirmation data; descriptive, no inferential claim; not frozen. Same author and session wrote and checked the code; the cross-checks are against values produced earlier by other scripts (pw02a, pw04a, pw05, pw06), not an independent replication. Scoped to these MLPs, this encoding (except the pw06 re-encodings), the test-split weights and a few seeds per configuration. The analysis shows what the count is made of (rate and KL per unit); it does not explain why the KL per unit settles near 0.5 nats. Raw inputs stay local; the record is uncommitted (no remote).

## Files

- `analyze_pw07.py` — the analysis (usage in its docstring).
- `results/analysis_001/results.json` — per run and per configuration: `A_unit_kl` (`per_run`, `per_run_pw06`, `per_cell`, `per_cell_pw06`, `examiner_check`, `global_`, `checks`) and `B_matched_legality` (`baseline`, `per_run`, `per_cell`, `counts`, `verify_report`, `checks`, `input_hashes_recorded_here`); `report_numbers_checked` lists the report values compared.
- `results/analysis_001/fig_unit_kl.png`, `.pdf` — draft figure.

**Independent check (2026-10-30).** Every number in results.json was re-derived from the raw arrays by a separately written verifier (without this script or src/pwlib): all match. Six README sentences were corrected after that check (inactive-KL bound, code10 per-unit range, tail range, the 1×64 below-baseline inference, the malformed fraction, the mismatch allocation).
