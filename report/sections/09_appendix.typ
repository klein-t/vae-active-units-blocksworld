
= Appendix A: Planned predictions, changes to the plan and code map

#text(size: 9pt)[*Planned predictions.* @tab-plan lists the main predictions we wrote down before the main sweep or the follow-up, with their outcomes, and the experiment READMEs list the secondary ones. Counts are medians at β = 1 on raw states unless stated. Models that did not reconstruct the data (Section 4.3) entered no prediction, except those for the index, the copies and β, which use every run.

#figure(
  placement: none,
  table(
    columns: (1fr, auto),
    align: (left, left),
    table.header([*Prediction*], [*Outcome*]),
    [Fewer than 25 active units on Blocksworld], [holds (at most 16)],
    [Information hypothesis at every network (Section 3.2)], [fails at all three],
    [Affine hypothesis (decision rule only, no predicted outcome)], [holds at 1×64, barely at 2×256],
    [Factor-count hypothesis (decision rule only, no predicted outcome)], [holds at no network],
    [Spread of configuration medians at least 3 and above the seed spread], [holds, from one configuration],
    [Learned γ adds ≥ 2 units on state-projected activations (@tab-act), ranges separated, at every transformer], [fails at all five],
    [k = 5 SAE has detectors for ≥ 24 of the 30 position facts, more than every Gaussian-latent model], [fails (24 against 25)],
    [Prior samples beat the per-block baseline, legality falls with the count], [not tested (pilot below 95%)],
    [Index keeps median ≤ 3, no run > 5, ≥ 10 fewer than 36 facts (follow-up)], [fails on all three],
    [code10 keeps ≤ 1 unit with one copy, 10 (runs 9–11) with 2–4 (follow-up)], [holds],
    [Medians at 24, 32 and 64 latents within two units (follow-up)], [holds],
  ),
  caption: [Main planned predictions and their outcomes. The Section 3.2 hypotheses were decided per network at the lower learning rate.],
) <tab-plan>

*Changes to the plan.* After the pilots we changed five things before looking at the main sweep. We dropped a 4×256 network, which could not reconstruct the data even with a KL warm-up, and adapted the predictions from four networks to three. We added the KL warm-up to every run, as the plan prescribed in this case, and trained for 60,000 steps instead of 30,000, since the pilot as a whole had not converged, with an extra convergence rule for the models whose objective is near zero and one for a learned γ. We left out comparison models that never reconstruct the data, and we kept the categorical VAE, whose objective still moved by 4.0% in its pilot, a case the plan did not cover, although strictly the plan would have stopped the sweep.

*Other notes.* Leaving out the 22 runs that did not converge changes no primary outcome, and of the secondary ones only the one on β. The follow-up ran as planned. On 30 October 2026 we added the KL per active unit and legality on equal terms from stored outputs only. Separately written re-implementations of the scoring reproduced every count and outcome, but they were written by models of the same family as those that ran the analyses, so they are not external replications.

*Additional figure and tables.* @fig-variance shows the gap between active and inactive latents of Section 5.1, @tab-act the counts on transformer activations of Section 5.4 and @tab-align the alignment of units with facts of Section 5.5.

#figure(
  image("../figures/fig4_variance.pdf", width: 100%),
  placement: none,
  caption: [Variance of the posterior mean (filled) and mean posterior variance (hollow) of each latent, sorted by the former, over the test states (all states, equally weighted, for code10 and prod5). Each panel shows the seed with the median count at 2×256, β = 1 and lr 3·10#super[−4], and (a) is the reference configuration.],
) <fig-variance>

#figure(
  placement: none,
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    stroke: (x, y) => if y == 1 { (bottom: 0.6pt) } else { none } + if x == 3 { (left: 0.4pt) } else { none },
    table.header(
      [], table.cell(colspan: 2)[*Comparison counts*], table.cell(colspan: 3)[*Active units (of 64)*],
      [*Transformer*], [*K#sub[probe]*], [*K#sub[99%]*], [*Projected, γ = 1*], [*Projected, learned γ*], [*Full, learned γ*],
    ),
    [T64], [25], [39], [14 (13–14)], [14 (13–14)], [14 (14–16)],
    [T128a], [26], [88], [14 (14–16)], [17 (16–17)], [20 (20–21)],
    [T128b], [27], [91], [17 (16–17)], [16 (15–17)], [22 (21–22)],
    [T128c], [27], [93], [16 (13–18)], [18 (17–18)], [22 (20–23)],
    [T256], [31], [208], [19 (17–19)], [21 (18–22)], [29 (28–30)],
    table.hline(stroke: 0.4pt),
    [T128a, β = 0.25], [], [], [], [], [62 (61–63)],
    [T128a, β = 4], [], [], [], [], [14 (13–16)],
  ),
  caption: [Active units on transformer activations, median (min–max) over three seeds, for the 2×256 β-VAE at β = 1 and lr 3·10#super[−4]. K#sub[probe] is the fewest principal components with which linear probes come within 0.5 percentage points of their accuracy on all coordinates, and K#sub[99%] the fewest that explain 99% of the variance. Projected stands for the state-projected activations, the least-squares prediction of the activations from the current facts (rank 25). The last two rows vary β on T128a, where no run converges at β = 0.25.],
) <tab-act>

#figure(
  placement: none,
  table(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    table.header([*Model (raw states)*], [*Coverage (of 30)*], [*Detector fraction*], [*DCI*], [*Reciprocal units*]),
    [top-k SAE, k = 5], [24, 25, 25], [0.87–0.91], [0.96–1.00], [n/a],
    [top-k SAE, k = 10], [25, 25, 26], [0.71–0.93], [0.95–1.00], [n/a],
    [β-TCVAE, β#sub[TC] = 2, 2×256], [20–22], [0.81–0.94], [0.58–0.65], [n/a],
    [β-VAE, reference configuration], [23, 25, 25, 23, 25], [0.80–1.00], [0.55–0.65], [8–9],
    [β-VAE, 1×64], [3–10], [0.12–0.44], [0.27–0.38], [0–3],
    [β-VAE, 1×64, lr 3·10#super[−3]], [25], [0.93–1.00], [0.60–0.69], [9–10],
    [β-VAE, 2×256, lr 3·10#super[−3]], [6–12], [0.66–0.91], [0.39–0.54], [0–2],
    [β-VAE, 2×1024], [24–25], [0.86–0.94], [0.56–0.61], [9–10],
    [β-VAE, 2×1024, lr 3·10#super[−3]], [14–24], [0.60–0.94], [0.51–0.63], [5–10],
    [β = 4, β-TCVAE at β#sub[TC] = 6, categorical VAE], table.cell(colspan: 4)[no reconstruction (all β-VAEs at β = 4), not scored],
  ),
  caption: [Fact alignment on raw states at β = 1 and lr 3·10#super[−4] unless stated. Coverage is the number of the 30 position facts with a detector among the active units (for the SAE, the alive units), per seed for the models compared in the text and a range otherwise, and 25 is a natural level since a code can leave one position per block implicit. The detector fraction is the share of active units (for the SAE, of alive units, firing on at least 0.5% of test states) that detect a fact. DCI is the disentanglement score of @eastwood2018dci, which measures how much each unit is used for one block variable only.],
) <tab-align>

*Code map.* Each directory under `experiments/` holds its plan, a `README.md` with its status and its scripts, with JSON results where it ran. The shared code is in `src/pwlib/`, and `pw02`–`pw04` hold the plans replaced after the pilots by `pw02a`–`pw04a`. `report/make_figures.py` draws the figures and writes the numbers behind them to `report/figures/tables.md`, with their sources.

#align(center)[#table(
  columns: (auto, auto, auto, auto),
  align: left,
  inset: (x: 5pt, y: 2.5pt),
  stroke: none,
  table.hline(),
  table.header([*Report*], [*Directory*], [*Report*], [*Directory*]),
  table.hline(stroke: 0.5pt),
  [Section 3.1], [`pw01_state_space_ground_truth`], [5.5, alignment], [`pw03a_fact_alignment_amended`],
  [counts of 5.1, 5.2, 5.4], [`pw02a_active_units_amended`], [5.5, post-hoc analyses], [`pw05_unit_anatomy`],
  [5.3, follow-up], [`pw06_encoding_and_latent_size`], [5.6, legality], [`pw04a_sample_legality_amended`],
  [KL per unit, equal terms], [`pw07_unit_kl_and_matched_legality`], [full chronology], [`docs/EXPERIMENT_LOG.md`],
  table.hline(),
)]]
