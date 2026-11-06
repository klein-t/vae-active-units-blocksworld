"""Exact 5-block Blocksworld state space (36 binary facts), model-free.

Fact order matches `probe_indices` in the thesis data's `blocksworld_metadata.json`:
on_table(A..E) [0-4], on(X,Y) row-major over X then Y != X [5-24],
clear(A..E) [25-29], holding(A..E) [30-34], hand_empty [35].

A state is legal iff it is one of the states reachable by the four STRIPS
operators (pick-up, put-down, stack, unstack) from the all-on-table state;
`is_legal` checks this with explicit constraints on the raw 36-bit vector.
Per-block categorical view: each block sits on the table ('T'), in the hand
('H') or on one of the other four blocks (6 values per block).
"""
import itertools
import math
from collections import deque

import numpy as np

BLOCKS = ('A', 'B', 'C', 'D', 'E')
FACTS = (
    [f'on_table({x})' for x in BLOCKS]
    + [f'on({x},{y})' for x in BLOCKS for y in BLOCKS if x != y]
    + [f'clear({x})' for x in BLOCKS]
    + [f'holding({x})' for x in BLOCKS]
    + ['hand_empty']
)
N_FACTS = len(FACTS)
INDEX = {f: i for i, f in enumerate(FACTS)}
HAND_EMPTY = INDEX['hand_empty']
PAD_TOKEN = 52
# Value order of the per-block categorical variable: table, hand, then the
# other blocks in alphabetical order.
POSITION_VALUES = {x: ('T', 'H') + tuple(y for y in BLOCKS if y != x) for x in BLOCKS}


def check_fact_order(probe_indices):
    """Fail loudly unless the metadata fact order equals FACTS."""
    names = sorted(probe_indices, key=probe_indices.get)
    if names != list(FACTS) or sorted(probe_indices.values()) != list(range(N_FACTS)):
        raise ValueError('Metadata probe order differs from pwlib.blocksworld.FACTS')


def is_legal(v):
    """True iff the 36-vector v is a legal 5-block Blocksworld state."""
    v = np.asarray(v)
    if v.shape != (N_FACTS,) or not np.isin(v, (0, 1)).all():
        return False
    v = v.astype(int)
    on = {(x, y): v[INDEX[f'on({x},{y})']] for x in BLOCKS for y in BLOCKS if x != y}
    held = [x for x in BLOCKS if v[INDEX[f'holding({x})']]]
    # Each block is in exactly one place: table, hand, or on one other block.
    for x in BLOCKS:
        places = v[INDEX[f'on_table({x})']] + v[INDEX[f'holding({x})']] + sum(on[x, y] for y in BLOCKS if y != x)
        if places != 1:
            return False
    # At most one block on top of any block.
    for y in BLOCKS:
        if sum(on[x, y] for x in BLOCKS if x != y) > 1:
            return False
    # One-block hand; hand_empty iff nothing is held.
    if len(held) > 1 or v[HAND_EMPTY] != int(not held):
        return False
    # clear(y) iff y is not held and nothing is on y (a held block supports nothing).
    for y in BLOCKS:
        covered = any(on[x, y] for x in BLOCKS if x != y)
        if y in held and covered:
            return False
        if v[INDEX[f'clear({y})']] != int(y not in held and not covered):
            return False
    # No cycles: following on() from every block ends at the table or hand.
    below = {x: y for (x, y), b in on.items() if b}
    for x in BLOCKS:
        seen, cur = set(), x
        while cur in below:
            if cur in seen:
                return False
            seen.add(cur)
            cur = below[cur]
    return True


def state_from_positions(pos):
    """36-vector from {block: 'T' | 'H' | other block} (no legality check)."""
    v = np.zeros(N_FACTS, dtype=np.int8)
    for x in BLOCKS:
        p = pos[x]
        if p == 'T':
            v[INDEX[f'on_table({x})']] = 1
        elif p == 'H':
            v[INDEX[f'holding({x})']] = 1
        else:
            v[INDEX[f'on({x},{p})']] = 1
    v[HAND_EMPTY] = int(not any(pos[x] == 'H' for x in BLOCKS))
    for y in BLOCKS:
        v[INDEX[f'clear({y})']] = int(pos[y] != 'H' and not any(pos[x] == y for x in BLOCKS))
    return v


def positions_from_state(v):
    """Per-block categorical codes (index into POSITION_VALUES[x]) of a legal state."""
    if not is_legal(v):
        raise ValueError('positions_from_state needs a legal state')
    codes = []
    for x in BLOCKS:
        hits = [k for k, p in enumerate(POSITION_VALUES[x]) if
                v[INDEX[f'on_table({x})' if p == 'T' else f'holding({x})' if p == 'H' else f'on({x},{p})']]]
        codes.append(hits[0])
    return tuple(codes)


def enumerate_legal_bruteforce():
    """All legal states: the 6^5 placements filtered by `is_legal`, sorted by bytes."""
    legal = set()
    for combo in itertools.product(*[POSITION_VALUES[x] for x in BLOCKS]):
        v = state_from_positions(dict(zip(BLOCKS, combo)))
        if is_legal(v):
            legal.add(v.tobytes())
    return np.stack([np.frombuffer(b, dtype=np.int8) for b in sorted(legal)])


def action_names():
    return ([f'(pick-up {x})' for x in BLOCKS] + [f'(put-down {x})' for x in BLOCKS]
            + [f'(stack {x} {y})' for x in BLOCKS for y in BLOCKS if x != y]
            + [f'(unstack {x} {y})' for x in BLOCKS for y in BLOCKS if x != y])


def apply_action(state, action):
    """STRIPS successor of a fact set, or None if preconditions fail."""
    op, *args = action.strip('()').split()
    if op == 'pick-up':
        x, = args
        pre, add = {f'clear({x})', f'on_table({x})', 'hand_empty'}, {f'holding({x})'}
    elif op == 'put-down':
        x, = args
        pre, add = {f'holding({x})'}, {f'clear({x})', f'on_table({x})', 'hand_empty'}
    elif op == 'stack':
        x, y = args
        pre, add = {f'holding({x})', f'clear({y})'}, {f'on({x},{y})', f'clear({x})', 'hand_empty'}
    elif op == 'unstack':
        x, y = args
        pre, add = {f'on({x},{y})', f'clear({x})', 'hand_empty'}, {f'holding({x})', f'clear({y})'}
    else:
        raise ValueError(f'Unknown action: {action}')
    if not pre <= state:
        return None
    return frozenset((state - pre) | add)


def enumerate_legal_bfs():
    """States reachable from all-on-table via STRIPS actions, sorted by bytes."""
    start = frozenset([f'on_table({x})' for x in BLOCKS] + [f'clear({x})' for x in BLOCKS] + ['hand_empty'])
    seen, queue = {start}, deque([start])
    actions = action_names()
    while queue:
        s = queue.popleft()
        for a in actions:
            t = apply_action(s, a)
            if t is not None and t not in seen:
                seen.add(t)
                queue.append(t)
    out = set()
    for s in seen:
        v = np.zeros(N_FACTS, dtype=np.int8)
        for f in s:
            v[INDEX[f]] = 1
        out.add(v.tobytes())
    return np.stack([np.frombuffer(b, dtype=np.int8) for b in sorted(out)])


def lah_total(n):
    """Number of ways to arrange n labelled blocks into unordered towers (sum of Lah numbers)."""
    if n == 0:
        return 1
    return sum(math.comb(n - 1, k - 1) * math.factorial(n) // math.factorial(k) for k in range(1, n + 1))


def legal_count_formula(n=len(BLOCKS)):
    """Arm-empty states L(n) plus n choices of held block times L(n-1)."""
    return lah_total(n) + n * lah_total(n - 1)
