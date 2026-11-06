# pw01_state_space_ground_truth — protocol

**Proposal item:** P1 (fix the expected number of factors before any VAE count).
**Kind:** exact, model-free analysis (`--kind analysis`, `pilot_required: false`).

## Question

What exactly is the legal state space of 5-block Blocksworld in the thesis
36-fact encoding, what do the thesis data arrays contain, and which
"true dimension" numbers can later VAE active-unit counts be compared with?
On discrete data the dimension is not unique, so all candidates are frozen
here, as definitions, before any VAE is trained.

## Encoding (verified against `blocksworld_metadata.json`)

36 facts: `on_table(A..E)` [0-4], `on(X,Y)` row-major over X, then Y != X
[5-24], `clear(A..E)` [25-29], `holding(A..E)` [30-34], `hand_empty` [35].
X token 52 = PAD; a position k is valid iff `X[i,k] != 52`; `Y[i,k]` is the
state after k actions (`Y[i,0]` initial). The run fails if PAD is not a suffix,
position 0 is not SOS, or Y is nonzero at any PAD position.

Legality (`src/pwlib/blocksworld.py::is_legal`): every block is in exactly one
place (table, hand, or on exactly one other block); at most one block on any
block; at most one block held and `hand_empty` iff none held; `clear(y)` iff y
is not held and nothing is on y; a held block supports nothing; no cycles.

## Primary criteria (all required; outcome `supported` only if all hold)

1. `count_agreement`: brute force over the 6^5 per-block placements filtered by
   the checker, BFS over the 50 STRIPS actions from the all-on-table state, and
   the formula L(5) + 5 L(4) with L(n) = sum_k C(n-1,k-1) n!/k! (Lah numbers)
   all give 866 identical states, 501 with the hand empty.
2. `data_all_legal`: zero illegal states at valid positions of train+test.
3. `data_cover_all_legal`: all 866 legal states occur in the data.
4. `hand_empty_equals_step_parity`: `Y[i,k,35] == (k even)` at every valid position.

## Gates

- **task:** input SHA-256 equal to `DESIGN.json`; fact order and action
  vocabulary match the metadata; PAD = 52; Y = 0 at PAD; every valid
  transition `Y[i,k-1] -> Y[i,k]` equals the STRIPS successor under action
  `X[i,k]`; 1,020,005 valid positions (expected from the prior exploratory count).
- **numerical:** the three counts agree; every enumerated state passes the
  checker; all 866 x 36 single-bit flips are rejected; on 200,000 random 36-bit
  vectors (seed 0) the checker accepts exactly those in the enumerated set.

## Reported quantities (descriptive, exact unless marked)

- Counts: legal, arm-empty, holding; data positions; coverage.
- Entropy (bits) of the empirical state distribution: all positions; train;
  test; even k; odd k; k <= 30 and k > 30; H(state | step parity).
- Per-block categorical factorization (block in {T, H, on one of 4 others}):
  product space 6^5, legal fraction; arm-empty view 5^5. For the empirical
  distribution (all, even k, odd k) and for the uniform distribution over
  legal states: marginal entropies, pairwise mutual information, and total
  correlation TC = sum_i H(X_i) - H(X_1..X_5) (state <-> 5-tuple is a bijection,
  so H(X_1..X_5) = H(state)).
- Linear structure: rank and affine-hull dimension of the legal-state matrix,
  with and without the `hand_empty` column, and per parity class.
- Dimension candidates JSON: each candidate with a definition and a flag
  `hand_empty_free` (False = hand_empty, which step parity determines, is not
  counted as a free factor).
- **Exploratory:** discrete intrinsic dimension with the binomial estimator of
  Macocco et al. (2207.09688), on the 866 legal states with Hamming distance,
  radii (r, R) = (3,6), (4,8), (5,10), (6,12), (7,14), (8,16) (the minimum
  Hamming distance between legal states is 3), two volume models (Z^d L1
  lattice; Hamming hypercube), positive controls on full {0,1}^8 and {0,1}^10.
  Ball counts include the centre point. Own re-implementation, not checked
  against DADApy; scale-dependent.

## Audit

`audit.py` recomputes the count with a generative tower enumeration (no
checker), legality/coverage/parity/entropy with per-position set lookups on
the raw arrays, conditional entropies, total correlation and affine dimensions
from the retained files; negative controls: column-permuted legal states
under the retained checker copy, and TC = 0 for an independent pair. Same
author and session: not an independent replication.

## Compute and storage

CPU, one process, numpy; under a minute expected. Outputs under 1 MB in
`results/<run_id>/`. `legal_states_866.npy` is gitignored and stays local.
