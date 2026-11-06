"""Independent audit 001: shared helpers written from the protocols, not from pwlib.

Nothing here imports pwlib. Fact indices come from the thesis metadata file; the legal state set is
re-derived by BFS over my own STRIPS operators; model forward passes are re-implemented from the
checkpoint state dicts (Linear/ReLU stacks), so a bug in pwlib's model or metric code would show up as a
mismatch against the recorded values.
"""
import ast
import hashlib
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('PW_BLOCKSWORLD_DATA', ROOT / 'data' / 'blocksworld'))  # earlier thesis data
PW02 = ROOT / 'experiments/pw02a_active_units_amended/results'
PAD = 52
BLOCKS = 'ABCDE'


def _maybe_literal(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


META = json.loads((DATA / 'blocksworld_metadata.json').read_text())
FIDX = _maybe_literal(META['probe_indices'])
VOCAB = _maybe_literal(META['vocab'])
INV_VOCAB = {v: k for k, v in VOCAB.items()}
POSITION_FACTS = [i for f, i in FIDX.items() if f.startswith(('on_table', 'on(', 'holding'))]
assert len(FIDX) == 36 and len(POSITION_FACTS) == 30


def f(name):
    return FIDX[name]


# --------------------------------------------------------------- STRIPS (own implementation)

def apply_action(s, act):
    """Apply a STRIPS action string to a 36-bit state; returns the new state or None if inapplicable."""
    s = s.copy()
    parts = act.strip('()').split()
    op, args = parts[0], parts[1:]
    if op == 'pick-up':
        x, = args
        if not (s[f(f'clear({x})')] and s[f(f'on_table({x})')] and s[f('hand_empty')]):
            return None
        s[f(f'holding({x})')] = 1
        s[f(f'on_table({x})')] = s[f(f'clear({x})')] = s[f('hand_empty')] = 0
    elif op == 'put-down':
        x, = args
        if not s[f(f'holding({x})')]:
            return None
        s[f(f'on_table({x})')] = s[f(f'clear({x})')] = s[f('hand_empty')] = 1
        s[f(f'holding({x})')] = 0
    elif op == 'stack':
        x, y = args
        if not (s[f(f'holding({x})')] and s[f(f'clear({y})')]):
            return None
        s[f(f'on({x},{y})')] = s[f(f'clear({x})')] = s[f('hand_empty')] = 1
        s[f(f'holding({x})')] = s[f(f'clear({y})')] = 0
    elif op == 'unstack':
        x, y = args
        if not (s[f(f'on({x},{y})')] and s[f(f'clear({x})')] and s[f('hand_empty')]):
            return None
        s[f(f'holding({x})')] = s[f(f'clear({y})')] = 1
        s[f(f'on({x},{y})')] = s[f(f'clear({x})')] = s[f('hand_empty')] = 0
    else:
        raise ValueError(act)
    return s


def bfs_legal():
    s0 = np.zeros(36, dtype=np.int8)
    for x in BLOCKS:
        s0[f(f'on_table({x})')] = s0[f(f'clear({x})')] = 1
    s0[f('hand_empty')] = 1
    seen = {s0.tobytes(): s0}
    q = deque([s0])
    acts = [a for a in VOCAB if a.startswith('(')]
    while q:
        s = q.popleft()
        for a in acts:
            t = apply_action(s, a)
            if t is not None and t.tobytes() not in seen:
                seen[t.tobytes()] = t
                q.append(t)
    return seen


LEGAL_SET = None


def legal_set():
    global LEGAL_SET
    if LEGAL_SET is None:
        LEGAL_SET = set(bfs_legal().keys())
    return LEGAL_SET


def is_legal_rows(B):
    """Exact legality by lookup in my BFS-derived set (rows of 0/1 int8)."""
    L = legal_set()
    B = np.ascontiguousarray(B.astype(np.int8))
    return np.array([r.tobytes() in L for r in B])


# --------------------------------------------------------------- data (own implementation from protocol text)

def load_thesis_data():
    return dict(Xtr=np.load(DATA / 'blocksworld_train_X.npy'), Ytr=np.load(DATA / 'blocksworld_train_Y.npy'),
                Xte=np.load(DATA / 'blocksworld_test_X.npy'), Yte=np.load(DATA / 'blocksworld_test_Y.npy'))


def split_indices(n=16004, n_train=12803):
    """pw02 PROTOCOL section 3: train file permuted by sequence with seed 0 and cut into train / val."""
    perm = np.random.default_rng(0).permutation(n)
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def state_counts(X, Y, row_of):
    valid = X != PAD
    S = Y[valid].astype(np.int8)
    idx = np.array([row_of[r.tobytes()] for r in np.ascontiguousarray(S)])
    return np.bincount(idx, minlength=len(row_of)).astype(np.float64)


# --------------------------------------------------------------- forward passes (own implementation)

def linear_stack(sd, prefix, x):
    """Sequential of Linear (even indices) with ReLU between, as stored in the state dict."""
    idx = sorted({int(k[len(prefix):].split('.')[0]) for k in sd if k.startswith(prefix) and k.endswith('.weight')})
    h = x
    for n, i in enumerate(idx):
        h = h @ sd[f'{prefix}{i}.weight'].T + sd[f'{prefix}{i}.bias']
        if n < len(idx) - 1:
            h = torch.relu(h)
    return h


def load_ckpt(run_dir, run):
    ck = torch.load(Path(run_dir) / 'models' / f'{run}.pt', map_location='cpu', weights_only=False)
    return ck, {k: v.float() for k, v in ck['state_dict'].items()}


@torch.no_grad()
def gauss_encode(sd, x, clamp=(-12.0, 6.0)):
    h = linear_stack(sd, 'enc.', x)
    L = h.shape[1] // 2
    return h[:, :L], h[:, L:].clamp(*clamp)


@torch.no_grad()
def decode(sd, z):
    return linear_stack(sd, 'dec.net.', z)


def wvar(a, w):
    a = np.asarray(a, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64) / np.sum(w)
    m = w @ a
    return w @ (a - m) ** 2


def run_seed(cell_id, seed):
    return int.from_bytes(hashlib.sha256(f'{cell_id}|{seed}'.encode()).digest()[:4], 'big') & 0x7FFFFFFF


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
