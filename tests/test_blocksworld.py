import numpy as np
import pytest

from pwlib import blocksworld as bw


@pytest.fixture(scope='module')
def legal():
    return bw.enumerate_legal_bruteforce()


def test_fact_order():
    assert bw.N_FACTS == 36
    assert bw.FACTS[:5] == [f'on_table({x})' for x in 'ABCDE']
    assert bw.FACTS[5] == 'on(A,B)' and bw.FACTS[24] == 'on(E,D)'
    assert bw.FACTS[25] == 'clear(A)' and bw.FACTS[30] == 'holding(A)' and bw.FACTS[35] == 'hand_empty'


def test_three_counts_agree(legal):
    assert len(legal) == 866
    assert int(legal[:, bw.HAND_EMPTY].sum()) == 501
    assert bw.lah_total(5) == 501 and bw.lah_total(4) == 73
    assert bw.legal_count_formula() == 866
    assert np.array_equal(bw.enumerate_legal_bfs(), legal)


def test_small_lah_values():
    # 1 block: 1 tower; 2 blocks: 2 single towers or 2 orders of one tower; 3 blocks: 13.
    assert [bw.lah_total(n) for n in range(4)] == [1, 1, 3, 13]


def test_single_bit_flips_rejected(legal):
    for v in legal:
        assert bw.is_legal(v)
        for i in range(bw.N_FACTS):
            w = v.copy()
            w[i] ^= 1
            assert not bw.is_legal(w)


def test_random_vectors_rejected():
    rng = np.random.default_rng(0)
    assert not any(bw.is_legal(r) for r in rng.integers(0, 2, size=(5000, 36)))


def test_handcrafted_illegal_states():
    s = bw.state_from_positions({'A': 'B', 'B': 'A', 'C': 'T', 'D': 'T', 'E': 'T'})  # 2-cycle
    assert not bw.is_legal(s)
    s = bw.state_from_positions({'A': 'H', 'B': 'H', 'C': 'T', 'D': 'T', 'E': 'T'})  # two held
    assert not bw.is_legal(s)
    s = bw.state_from_positions({'A': 'C', 'B': 'C', 'C': 'T', 'D': 'T', 'E': 'T'})  # two on C
    assert not bw.is_legal(s)
    s = bw.state_from_positions({'A': 'B', 'B': 'H', 'C': 'T', 'D': 'T', 'E': 'T'})  # held B supports A
    assert not bw.is_legal(s)
    assert not bw.is_legal(np.full(36, 2))
    assert not bw.is_legal(np.zeros(35))


def test_positions_roundtrip(legal):
    seen = set()
    for v in legal:
        codes = bw.positions_from_state(v)
        pos = {x: bw.POSITION_VALUES[x][c] for x, c in zip(bw.BLOCKS, codes)}
        assert np.array_equal(bw.state_from_positions(pos), v)
        seen.add(codes)
    assert len(seen) == 866


def test_actions_preserve_legality(legal):
    for v in legal[::17]:
        s = frozenset(f for f, b in zip(bw.FACTS, v) if b)
        for a in bw.action_names():
            t = bw.apply_action(s, a)
            if t is not None:
                w = np.array([int(f in t) for f in bw.FACTS])
                assert bw.is_legal(w) and w[bw.HAND_EMPTY] != v[bw.HAND_EMPTY]
