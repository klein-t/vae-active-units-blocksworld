"""Data for the pw02-pw04 sweep: Blocksworld states from the earlier thesis data, sequence splits, controls.

Conventions (pw02 PROTOCOL.md section 3):
- thesis train file permuted by sequence with seed 0 and cut into train (12,803)
  and val (3,201); the thesis test file is the sealed confirmation set and is
  only read when `include_test=True` (full runs).
- A position k of sequence i is valid iff X[i, k] != PAD (52); its state is
  Y[i, k] (state after k actions). Raw-state training draws i.i.d. from the
  train-split empirical distribution over the 866 legal states.
- Every input file is checked against its expected SHA-256 before use.
"""
import itertools
import os
from pathlib import Path

import numpy as np
import torch

from . import blocksworld as bw
from .runio import sha256

# Blocksworld sequences of the earlier thesis work: blocksworld_{train,test}_{X,Y}.npy, blocksworld_metadata.json
BW_DATA = Path(os.environ.get('PW_BLOCKSWORLD_DATA', Path(__file__).resolve().parents[2] / 'data' / 'blocksworld'))
SPLIT_SEED = 0
N_TRAIN_FILE = 16004
N_TRAIN_SPLIT = 12803
PW01_RUN = Path('experiments/pw01_state_space_ground_truth/results/analysis_001')
INDEP_EVAL_N = 20_000
INDEP_EVAL_SEED = 0
CODE10_ASSIGN_SEED = 0


def check_hash(path, expected):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')
    got = sha256(path)
    if got != expected:
        raise ValueError(f'SHA-256 mismatch for {path}: {got} != {expected}')
    return got


def hash_key(path):
    """Key of `path` in the recorded hashes: files of the data directory are recorded as $PW_BLOCKSWORLD_DATA/<name>."""
    path = Path(path)
    return f'$PW_BLOCKSWORLD_DATA/{path.name}' if path.parent == BW_DATA else str(path)


def load_array(path, hashes):
    """np.load after checking the hash recorded for this path (repo-relative, or $PW_BLOCKSWORLD_DATA/<name>)."""
    key = hash_key(path)
    if key not in hashes:
        raise KeyError(f'No expected hash recorded for {key}')
    check_hash(path, hashes[key])
    return np.load(path)


def state_codes(S):
    """Integer code of each 36-bit row (bit i -> 2^i)."""
    S = np.asarray(S)
    if S.ndim != 2 or S.shape[1] != bw.N_FACTS:
        raise ValueError(f'Expected (n, 36) states, got {S.shape}')
    return (S.astype(np.int64) << np.arange(bw.N_FACTS, dtype=np.int64)).sum(1)


class LegalIndex:
    """Row index of each state in the pw01 866-state table (sorted by bytes)."""

    def __init__(self, legal):
        self.legal = np.asarray(legal, dtype=np.int8)
        codes = state_codes(self.legal)
        self.order = np.argsort(codes)
        self.sorted_codes = codes[self.order]
        if len(np.unique(codes)) != len(codes):
            raise ValueError('Duplicate legal states')

    def lookup(self, S):
        """Index into `legal` for each row, -1 if not a legal state."""
        c = state_codes(S)
        pos = np.searchsorted(self.sorted_codes, c)
        pos = np.clip(pos, 0, len(self.sorted_codes) - 1)
        hit = self.sorted_codes[pos] == c
        return np.where(hit, self.order[pos], -1)

    def index(self, S):
        idx = self.lookup(S)
        if (idx < 0).any():
            raise ValueError(f'{int((idx < 0).sum())} states not in the legal table')
        return idx


def load_legal_table(root, hashes):
    path = Path(root) / PW01_RUN / 'legal_states_866.npy'
    legal = load_array(path, {str(path): hashes[str(PW01_RUN / 'legal_states_866.npy')]})
    if not np.array_equal(legal, bw.enumerate_legal_bruteforce()):
        raise ValueError('pw01 legal table differs from the pwlib enumeration')
    return legal


def split_train_file():
    perm = np.random.default_rng(SPLIT_SEED).permutation(N_TRAIN_FILE)
    return np.sort(perm[:N_TRAIN_SPLIT]), np.sort(perm[N_TRAIN_SPLIT:])


def load_sequences(hashes, include_test):
    """{'train','val'[, 'test']} -> (X, Y) arrays, hash-checked."""
    Xtr = load_array(BW_DATA / 'blocksworld_train_X.npy', hashes)
    Ytr = load_array(BW_DATA / 'blocksworld_train_Y.npy', hashes)
    if Xtr.shape != (N_TRAIN_FILE, 61) or Ytr.shape != (N_TRAIN_FILE, 61, bw.N_FACTS):
        raise ValueError(f'Unexpected train file shapes {Xtr.shape} {Ytr.shape}')
    tr, va = split_train_file()
    out = dict(train=(Xtr[tr], Ytr[tr]), val=(Xtr[va], Ytr[va]))
    if include_test:
        out['test'] = (load_array(BW_DATA / 'blocksworld_test_X.npy', hashes),
                       load_array(BW_DATA / 'blocksworld_test_Y.npy', hashes))
    return out


def valid_states(X, Y):
    """States at all valid positions (X != PAD) and the per-position mask."""
    valid = X != bw.PAD_TOKEN
    if X.shape[1] != Y.shape[1]:
        raise ValueError('X and Y position counts differ')
    if Y[~valid].any():
        raise ValueError('Non-zero state at a PAD position')
    return Y[valid], valid


def state_weights(seqs, index):
    """Per-split frequency weights over the 866 legal states (sum to 1) and position counts."""
    weights, counts = {}, {}
    for name, (X, Y) in seqs.items():
        S, _ = valid_states(X, Y)
        c = np.bincount(index.index(S), minlength=len(index.legal)).astype(np.float64)
        counts[name] = int(c.sum())
        weights[name] = c / c.sum()
    return weights, counts


def code10_states():
    """10 fair bits, each copied into 3-4 of the 36 positions (assignment seed 0); all 1,024 states."""
    perm = np.random.default_rng(CODE10_ASSIGN_SEED).permutation(bw.N_FACTS)
    bit_of_pos = np.empty(bw.N_FACTS, dtype=np.int64)
    bit_of_pos[perm] = np.arange(bw.N_FACTS) % 10
    codes = np.array(list(itertools.product((0, 1), repeat=10)), dtype=np.int8)
    return codes[:, bit_of_pos], bit_of_pos


def prod5_states():
    """5 independent uniform 6-valued blocks one-hot in the Blocksworld position-fact columns
    (on_table, on, holding); clear and hand_empty columns constant zero. 7,776 states."""
    rows = []
    for combo in itertools.product(*[bw.POSITION_VALUES[x] for x in bw.BLOCKS]):
        v = bw.state_from_positions(dict(zip(bw.BLOCKS, combo)))
        v[DERIVED_COLUMNS] = 0
        rows.append(v)
    return np.stack(rows)


POSITION_COLUMNS = np.array([bw.INDEX[f] for f in bw.FACTS
                             if f.startswith(('on_table', 'on(', 'holding'))])
DERIVED_COLUMNS = np.array([bw.INDEX[f] for f in bw.FACTS if f.startswith('clear') or f == 'hand_empty'])
assert len(POSITION_COLUMNS) == 30 and len(DERIVED_COLUMNS) == 6


def block_variables(S):
    """(n, 5) per-block categorical codes (index into POSITION_VALUES[x]) from position bits.
    Fails loudly if a block is not in exactly one place."""
    S = np.asarray(S)
    out = np.empty((len(S), len(bw.BLOCKS)), dtype=np.int64)
    for b, x in enumerate(bw.BLOCKS):
        cols = [bw.INDEX[f'on_table({x})'] if p == 'T' else bw.INDEX[f'holding({x})'] if p == 'H'
                else bw.INDEX[f'on({x},{p})'] for p in bw.POSITION_VALUES[x]]
        block = S[:, cols]
        if not (block.sum(1) == 1).all():
            raise ValueError(f'Block {x} not in exactly one place in some rows')
        out[:, b] = block.argmax(1)
    return out


# --------------------------------------------------------------------- datasets

class TableData:
    """Finite support with sampling probabilities; evaluation is exact over the table rows."""
    kind = 'table'

    def __init__(self, states, train_p, eval_weights, n_train_points, name):
        self.x = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        p = np.asarray(train_p, dtype=np.float64)
        if p.shape != (len(self.x),) or abs(p.sum() - 1) > 1e-9:
            raise ValueError('train_p must be a distribution over the rows')
        self.p = torch.as_tensor(p)
        self.eval_weights = {k: np.asarray(v, dtype=np.float64) for k, v in eval_weights.items()}
        self.n_train_points = int(n_train_points)
        self.name = name
        self.d_in = self.x.shape[1]
        self.train_mean = (self.p[:, None] * self.x.double()).sum(0).float()

    def sample(self, batch, gen):
        return self.x[torch.multinomial(self.p, batch, replacement=True, generator=gen)]

    def eval_sets(self):
        """name -> (rows, weights). Table data share one row set across weightings."""
        return {k: (self.x, w) for k, w in self.eval_weights.items()}


class BernoulliData:
    """Independent bits with given marginals (indep control); evaluation on a fixed sample."""
    kind = 'bernoulli'

    def __init__(self, marginals, name, keys=('dev', 'conf')):
        self.keys = tuple(keys)
        self.marg = torch.as_tensor(np.asarray(marginals), dtype=torch.float32)
        g = torch.Generator().manual_seed(INDEP_EVAL_SEED)
        self.eval_x = torch.bernoulli(self.marg.expand(INDEP_EVAL_N, -1), generator=g)
        self.n_train_points = INDEP_EVAL_N
        self.name = name
        self.d_in = len(self.marg)
        self.train_mean = self.marg.clone()

    def sample(self, batch, gen):
        return torch.bernoulli(self.marg.expand(batch, -1), generator=gen)

    def eval_sets(self):
        w = np.full(INDEP_EVAL_N, 1.0 / INDEP_EVAL_N)
        return {k: (self.eval_x, w) for k in self.keys}


class ArrayData:
    """Activation positions: uniform sampling over train rows; evaluation on held-out rows."""
    kind = 'array'

    def __init__(self, train, evals, name):
        self.train = torch.as_tensor(np.asarray(train), dtype=torch.float32)
        self.evals = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in evals.items()}
        self.n_train_points = len(self.train)
        self.name = name
        self.d_in = self.train.shape[1]
        self.train_mean = self.train.mean(0)

    def sample(self, batch, gen):
        return self.train[torch.randint(len(self.train), (batch,), generator=gen)]

    def eval_sets(self):
        return {k: (v, np.full(len(v), 1.0 / len(v))) for k, v in self.evals.items()}
