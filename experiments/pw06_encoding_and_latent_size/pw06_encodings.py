"""pw06 encodings of the 866 Blocksworld states and the code10 copy controls (DESIGN.md, "Data and cells").

Every encoding is a deterministic function of the state, row-aligned with the pw01 table
`legal_states_866.npy` (row i = pw01 `state_index` i), so the pw02a state weights apply unchanged.
- E36: the 36 facts (identity).
- E30: the 30 position facts (on_table, on, holding) in pwlib fact order; drops clear(X), hand_empty.
- E10a / E10b: 10-bit binary index, most significant bit first. Index of state i =
  perm[i] with perm = numpy.random.default_rng(seed).permutation(866), seed 0 (E10a) / 1 (E10b).
  This reading of "rank under the permutation" is the one that reproduces the design's model-free
  table (total correlation 0.39-0.41 nats for seeds 0-3); the inverse reading gives 0.40-0.41.
- code10 with c copies: the 1,024 states of 10 fair bits (pw02a generator,
  itertools.product((0, 1), repeat=10)); each bit copied c times; 40 inputs, the other 40 - 10c
  inputs constant zero. Column placement mirrors pw02a's code10: permutation of the 40 columns
  with seed 0 (pwlib.data.CODE10_ASSIGN_SEED); columns perm[:10c] carry bit (k mod 10).
"""
import itertools

import numpy as np

from pwlib import data as D

N_STATES = 866
E10_SEEDS = dict(e10a=0, e10b=1)
N_BITS = 10
CODE10_INPUTS = 40
LN2 = float(np.log(2))


def e30(legal):
    return np.asarray(legal)[:, D.POSITION_COLUMNS]


def e10_index(seed):
    """Index (0..865) of each pw01 state under the fixed permutation."""
    return np.random.default_rng(seed).permutation(N_STATES)


def to_bits(idx, n=N_BITS):
    idx = np.asarray(idx, dtype=np.int64)
    return ((idx[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.int8)


def from_bits(B):
    B = np.asarray(B, dtype=np.int64)
    return (B << np.arange(B.shape[1] - 1, -1, -1)).sum(1)


def e10(seed):
    return to_bits(e10_index(seed))


def code10_copies(c):
    """(states [1024, 40] int8, bit_of_col [40] with -1 for constant-zero columns)."""
    if c not in (1, 2, 3, 4):
        raise ValueError(f'c must be in 1..4, got {c}')
    perm = np.random.default_rng(D.CODE10_ASSIGN_SEED).permutation(CODE10_INPUTS)
    bit_of_col = np.full(CODE10_INPUTS, -1, dtype=np.int64)
    bit_of_col[perm[:N_BITS * c]] = np.arange(N_BITS * c) % N_BITS
    codes = np.array(list(itertools.product((0, 1), repeat=N_BITS)), dtype=np.int8)
    S = np.zeros((len(codes), CODE10_INPUTS), dtype=np.int8)
    used = bit_of_col >= 0
    S[:, used] = codes[:, bit_of_col[used]]
    return S, bit_of_col


def encode(name, legal):
    """State table for an encoding name used in sweep_grid.json substrates."""
    if name == 'bw':
        return np.asarray(legal)
    if name == 'e30':
        return e30(legal)
    if name in E10_SEEDS:
        return e10(E10_SEEDS[name])
    if name.startswith('code10c'):
        return code10_copies(int(name[len('code10c'):]))[0]
    raise ValueError(f'Unknown encoding {name}')


# ------------------------------------------------------------- model-free quantities (nats)

def entropy_nats(w):
    w = np.asarray(w, dtype=np.float64)
    w = w[w > 0]
    return float(-(w * np.log(w)).sum())


def bernoulli_entropy_nats(p):
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    out[m] = -(p[m] * np.log(p[m]) + (1 - p[m]) * np.log1p(-p[m]))
    return out


def zero_rate_distortion(S, w):
    """Sum of the marginal Bernoulli entropies of the encoded bits under weights w."""
    w = np.asarray(w, dtype=np.float64)
    return float(bernoulli_entropy_nats((w[:, None] * np.asarray(S, dtype=np.float64)).sum(0)).sum())


def total_correlation(S, w):
    """Sum of marginal bit entropies minus the joint entropy (codes are a bijection of the states)."""
    return zero_rate_distortion(S, w) - entropy_nats(w)


def model_free_table(legal, w):
    """The design's model-free table under state weights w (train frequencies)."""
    H = entropy_nats(w)
    return dict(H_nats=H, H_bits=H / LN2, zero_rate_distortion_e36_nats=zero_rate_distortion(legal, w),
                tc_e10_nats={str(s): total_correlation(e10(s), w) for s in range(4)})
