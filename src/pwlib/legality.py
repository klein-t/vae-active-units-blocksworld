"""Vectorized legality checks, violation taxonomy and exact factorized baselines (pw04).

`pwlib.blocksworld.is_legal` is the reference checker (pw01); `table_member`
is the independent second check (membership in the 866-state table).
"""
import itertools

import numpy as np

from . import blocksworld as bw
from .data import LegalIndex, POSITION_COLUMNS, block_variables

NB = len(bw.BLOCKS)
ON = np.full((NB, NB), -1)
for _i, _x in enumerate(bw.BLOCKS):
    for _j, _y in enumerate(bw.BLOCKS):
        if _x != _y:
            ON[_i, _j] = bw.INDEX[f'on({_x},{_y})']
TABLE = np.array([bw.INDEX[f'on_table({x})'] for x in bw.BLOCKS])
HOLD = np.array([bw.INDEX[f'holding({x})'] for x in bw.BLOCKS])
CLEAR = np.array([bw.INDEX[f'clear({x})'] for x in bw.BLOCKS])
VIOLATIONS = ('V1_place', 'V2_two_on_one', 'V3_hand', 'V4_clear', 'V5_cycle', 'V6_held_supports')


def checker_flags(S):
    """pw01 reference checker on every row."""
    return np.array([bw.is_legal(v) for v in np.asarray(S)], dtype=bool)


def table_member(S, index: LegalIndex):
    return index.lookup(S) >= 0


def on_matrix(S):
    """[n, 5, 5] on(x, y) adjacency (diagonal zero)."""
    S = np.asarray(S)
    A = np.zeros((len(S), NB, NB), dtype=np.int64)
    for i in range(NB):
        for j in range(NB):
            if i != j:
                A[:, i, j] = S[:, ON[i, j]]
    return A


def violations(S):
    """Boolean [n, 6] matrix of violated constraints (V1-V6, see PROTOCOL.md section 2):
    V1 a block not in exactly one place; V2 two blocks on the same block; V3 hand
    inconsistency (> 1 held, or hand_empty disagreeing with holding); V4 clear inconsistent
    with on/holding; V5 a cycle of on() facts; V6 a held block with a block on it."""
    S = np.asarray(S).astype(np.int64)
    A = on_matrix(S)
    held = S[:, HOLD]
    places = S[:, TABLE] + held + A.sum(2)
    v1 = (places != 1).any(1)
    covered = A.sum(1)  # number of blocks on y
    v2 = (covered > 1).any(1)
    n_held = held.sum(1)
    v3 = (n_held > 1) | (S[:, bw.HAND_EMPTY] != (n_held == 0))
    clear_expected = ((held == 0) & (covered == 0)).astype(np.int64)
    v4 = (S[:, CLEAR] != clear_expected).any(1)
    P, cyc = np.eye(NB, dtype=np.int64)[None].repeat(len(S), 0), np.zeros(len(S), dtype=bool)
    for _ in range(NB):
        P = np.minimum(P @ A, 1)
        cyc |= np.trace(P, axis1=1, axis2=2) > 0
    v6 = ((held == 1) & (covered > 0)).any(1)
    return np.stack([v1, v2, v3, v4, cyc, v6], 1)


def first_violation(V):
    """Index of the first violated constraint per row (-1 if none)."""
    return np.where(V.any(1), V.argmax(1), -1)


def legal_from_position_bits(S):
    """Blocksworld legality of the 30 position bits alone, with clear and hand_empty derived
    (used for prod5-trained models whose derived columns are constant zero)."""
    S = np.asarray(S).astype(np.int8)
    ok = np.zeros(len(S), dtype=bool)
    one_place = np.ones(len(S), dtype=bool)
    for x in bw.BLOCKS:
        cols = [bw.INDEX[f'on_table({x})'], bw.INDEX[f'holding({x})']] + \
               [bw.INDEX[f'on({x},{y})'] for y in bw.BLOCKS if y != x]
        one_place &= S[:, cols].sum(1) == 1
    rows = np.flatnonzero(one_place)
    if len(rows):
        codes = block_variables(S[rows])
        for r, c in zip(rows, codes):
            pos = {x: bw.POSITION_VALUES[x][c[b]] for b, x in enumerate(bw.BLOCKS)}
            ok[r] = bw.is_legal(bw.state_from_positions(pos))
    return ok


# ---------------------------------------------------------- exact baselines

def baseline_b0():
    return 866 / 2 ** bw.N_FACTS


def baseline_b1(legal, marginals):
    """Exact P(legal) for independent bits with the given marginals."""
    p = np.asarray(marginals, dtype=np.float64)
    L = np.asarray(legal, dtype=np.float64)
    return float(np.exp((L * np.log(p) + (1 - L) * np.log1p(-p)).sum(1)).sum())


def baseline_b2(block_marginals):
    """Exact P(legal) when each block's position is drawn independently from its marginal over
    the 6 values (clear, hand_empty derived): enumeration of the 7,776 combinations."""
    q = np.asarray(block_marginals, dtype=np.float64)  # [5, 6]
    total = 0.0
    for combo in itertools.product(range(6), repeat=NB):
        pos = {x: bw.POSITION_VALUES[x][c] for x, c in zip(bw.BLOCKS, combo)}
        if bw.is_legal(bw.state_from_positions(pos)):
            total += float(np.prod([q[b, c] for b, c in enumerate(combo)]))
    return total


def block_marginals(legal, weights):
    """[5, 6] marginal distribution of each block's position under state weights."""
    codes = block_variables(legal)
    out = np.zeros((NB, 6))
    for b in range(NB):
        np.add.at(out[b], codes[:, b], weights)
    return out / out.sum(1, keepdims=True)


def mc_b1(marginals, n, rng, index):
    S = (rng.random((n, bw.N_FACTS)) < np.asarray(marginals)).astype(np.int8)
    return int(table_member(S, index).sum())


def mc_b2(block_marg, n, rng, index):
    S = np.zeros((n, bw.N_FACTS), dtype=np.int8)
    for b, x in enumerate(bw.BLOCKS):
        c = rng.choice(6, size=n, p=block_marg[b])
        for v, p in enumerate(bw.POSITION_VALUES[x]):
            col = bw.INDEX[f'on_table({x})'] if p == 'T' else bw.INDEX[f'holding({x})'] if p == 'H' \
                else bw.INDEX[f'on({x},{p})']
            S[c == v, col] = 1
    held = S[:, HOLD].sum(1)
    S[:, bw.HAND_EMPTY] = held == 0
    A = on_matrix(S)
    S[:, CLEAR] = ((S[:, HOLD] == 0) & (A.sum(1) == 0)).astype(np.int8)
    return int(table_member(S, index).sum())


def mc_b0(n, rng, index):
    S = rng.integers(0, 2, size=(n, bw.N_FACTS)).astype(np.int8)
    return int(table_member(S, index).sum())


def codeword_valid(S, bit_of_pos):
    """code10 validity: every copy of each source bit agrees."""
    S = np.asarray(S)
    ok = np.ones(len(S), dtype=bool)
    for b in np.unique(bit_of_pos):
        cols = S[:, bit_of_pos == b]
        ok &= (cols == cols[:, :1]).all(1)
    return ok


assert len(POSITION_COLUMNS) == 30
