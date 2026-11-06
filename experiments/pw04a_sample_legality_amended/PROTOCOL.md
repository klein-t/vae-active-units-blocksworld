# pw04a_sample_legality_amended — protocol (amendment of pw04, frozen 2026-10-21)

**Pilot-triggered amendment** of `pw04_sample_legality`, registered
2026-10-21 together with `pw02a_active_units_amended`, before any
confirmation data was opened. pw04 never ran (blocked by the pw02 pilot
competence stop). The pw04 protocol
(`experiments/pw04_sample_legality/PROTOCOL.md`, frozen in the pw04 freeze record
`68a9e8b5...`) remains binding for everything not listed here: legality
definition, taxonomy, sampling (20,000 prior samples, mode decoding),
baselines B0-B2, gates and thresholds, hypotheses and forecasts.

## Changes

1. **Input:** the amended sweep `pw02a_active_units_amended` (A3 removed,
   60k steps, KL warm-up on all cells). Pilot on pw02a `pilot_001`; full
   run on pw02a `full_001`.
2. **A3 removed from the beta=1 cells.** H4.1 covers every gated beta=1
   Gaussian beta-VAE run at A1, A2, A4 x 2 lr x 5 seeds (up to 30 runs);
   H4.2 strata hold up to 30 runs (a stratum still counts with >= 8 gated
   runs); the experiment-level competence gate covers the 6 remaining beta=1
   cells (majority of planned seeds, >= 3 of 5).
3. **Secondary comparisons with an unscorable side are untested**: H4.4
   (categorical vs Gaussian) reports `separated: null` when either cell has
   no gated seed (was false).

Compute and storage as pw04 (< 10 CPU-min).
