# pw03a_fact_alignment_amended — protocol (amendment of pw03, frozen 2026-10-21)

**Pilot-triggered amendment** of `pw03_fact_alignment`, registered
2026-10-21 together with `pw02a_active_units_amended`, before any
confirmation data was opened. pw03 never ran: it was blocked by the pw02
pilot competence stop. The pw03 protocol
(`experiments/pw03_fact_alignment/PROTOCOL.md`, frozen in the pw03 freeze record
`da7e8521...`) remains binding for everything not listed here: question,
identifiability, models scored, metrics M1-M3, gates and thresholds, nulls,
splits, hypotheses and forecasts.

## Changes

1. **Input:** the amended sweep `pw02a_active_units_amended` (A3 removed,
   60k steps, KL warm-up on all cells). Pilot on pw02a `pilot_001`; full
   run on pw02a `full_001`.
2. **H3.2 bug fix (no change to the hypothesis).** The pw03 code scored
   H3.2 as false (unsupported) when a comparator cell had zero
   pw02-gated seeds. In pw03a the criterion is three-valued (Kleene): it is
   **false** if any measured part fails (an SAE seed below Cov30 0.8, or a
   scored comparator whose seed maximum reaches the SAE seed minimum);
   **untested** if the k=5 SAE or any comparator cell has zero gated seeds
   and nothing measured fails; **true** otherwise. An untested H3.2 gives run
   status `inconclusive`, outcome `untested`. Implementation
   `pwlib.alignment.h32_verdict`, unit-tested in
   `tests/test_amendment_pw02a.py`; the audit re-derives it separately.
3. **Secondary comparisons with an unscorable side are untested**, not
   false: H3.3 complete separation (`pwlib.alignment.separated_or_untested`).
4. **H3.1 (descriptive)** over the 6 remaining beta=1 cells: the former
   "seed median < 0.5 in >= 7 of 8 cells" becomes 6 of 6 (fraction kept,
   rounded up).

Compute and storage as pw03 (about 4 CPU-h, about 35 min wall for the full
analysis; slightly less with 210 instead of 246 pw02 runs).
