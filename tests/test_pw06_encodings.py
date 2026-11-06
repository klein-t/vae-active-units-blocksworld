"""Unit tests for the pw06 encodings (experiments/pw06_encoding_and_latent_size/pw06_encodings.py)."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np

from pwlib import blocksworld as bw
from pwlib import data as D

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'pw06_encodings', ROOT / 'experiments/pw06_encoding_and_latent_size/pw06_encodings.py')
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)
LEGAL = np.load(ROOT / D.PW01_RUN / 'legal_states_866.npy')


def test_e10_codes_unique_and_invertible():
    for name, seed in E.E10_SEEDS.items():
        B = E.encode(name, LEGAL)
        assert B.shape == (866, 10) and set(np.unique(B)) <= {0, 1}
        assert len(np.unique(B, axis=0)) == 866
        idx = E.from_bits(B)
        assert np.array_equal(idx, np.random.default_rng(seed).permutation(866))
        assert np.array_equal(np.sort(idx), np.arange(866))
        # inverse map: code -> state row
        row_of_code = {int(i): r for r, i in enumerate(idx)}
        assert all(row_of_code[int(i)] == r for r, i in enumerate(idx))


def test_e10_msb_first():
    assert E.to_bits(np.array([1]))[0].tolist() == [0] * 9 + [1]
    assert E.to_bits(np.array([512]))[0].tolist() == [1] + [0] * 9
    assert E.from_bits(E.to_bits(np.arange(1024))).tolist() == list(range(1024))


def test_e10a_e10b_differ():
    assert not np.array_equal(E.encode('e10a', LEGAL), E.encode('e10b', LEGAL))


def test_e30_is_the_30_position_facts():
    X = E.encode('e30', LEGAL)
    names = [bw.FACTS[i] for i in D.POSITION_COLUMNS]
    assert X.shape == (866, 30)
    assert names == [f for f in bw.FACTS if f.startswith(('on_table', 'on(', 'holding'))]
    assert not any(f.startswith('clear') or f == 'hand_empty' for f in names)
    assert np.array_equal(X, LEGAL[:, [bw.INDEX[f] for f in names]])
    # dropped facts are affine functions of the positions: E30 keeps all 866 states distinct
    assert len(np.unique(X, axis=0)) == 866


def test_code10_copies():
    codes = np.array(list(itertools.product((0, 1), repeat=10)), dtype=np.int8)
    for c in (1, 2, 3, 4):
        S, boc = E.code10_copies(c)
        assert S.shape == (1024, 40)
        assert (boc < 0).sum() == 40 - 10 * c and not S[:, boc < 0].any()
        for b in range(10):
            cols = np.flatnonzero(boc == b)
            assert len(cols) == c
            for col in cols:
                assert np.array_equal(S[:, col], codes[:, b])
        assert len(np.unique(S, axis=0)) == 1024


def test_identity_e36():
    assert np.array_equal(E.encode('bw', LEGAL), LEGAL)


def test_model_free_quantities_uniform_sanity():
    u = np.full(866, 1 / 866)
    mf = E.model_free_table(LEGAL, u)
    assert abs(mf['H_nats'] - np.log(866)) < 1e-12
    for v in mf['tc_e10_nats'].values():
        assert v >= -1e-12
