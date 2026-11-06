"""Decision-rule tests for the pw06 registered analysis on synthetic rows (no model, no data)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'analyze_pw06', ROOT / 'experiments/pw06_encoding_and_latent_size/analyze_pw06.py')
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

GATES_OK = {'pipeline': {'pass': True}, 'capacity_p1': {'pass': True}}


def rows_for(label, counts, recon=True, conv=True):
    return [dict(label=label, count=c, numerical_ok=True, recon_ok=recon, converged=conv, status='ok')
            for c in counts]


def base():
    r = rows_for('E36-L32', [15, 15, 14, 16, 15]) + rows_for('E30-L32', [14] * 5)
    r += rows_for('E10a-L32', [0, 1, 1, 2, 0], recon=False) + rows_for('E10b-L32', [1, 1, 0, 3, 5], recon=False)
    r += rows_for('code10-c1', [0, 1, 0, 2, 1], recon=False)
    for c in (2, 3, 4):
        r += rows_for(f'code10-c{c}', [10, 10, 9, 11, 10])
    for L, v in ((8, 7), (10, 9), (12, 11), (16, 14), (24, 15), (64, 17)):
        r += rows_for(f'E36-L{L}', [v] * 5)
    return r


def test_all_hold():
    v = A.verdicts(base(), False, GATES_OK)
    assert [v[k]['verdict'] for k in ('P1', 'P2', 'P3', 'S1', 'S2')] == [True] * 5
    assert A.outcome(v) == 'supported'


def test_p1_fails_on_max_or_gap():
    r = [x for x in base() if x['label'] != 'E10b-L32'] + rows_for('E10b-L32', [1, 1, 0, 3, 6])
    assert A.p1(r, False)['verdict'] is False
    r = [x for x in base() if x['label'] != 'E10a-L32'] + rows_for('E10a-L32', [6, 6, 6, 6, 6])
    assert A.p1(r, False)['per_encoding']['E10a-L32']['conditions']['gap_ge_10'] is False


def test_p2_band_and_c1():
    r = [x for x in base() if x['label'] != 'code10-c3'] + rows_for('code10-c3', [10, 10, 10, 12, 10])
    assert A.p2(r, False)['verdict'] is False
    r = [x for x in base() if x['label'] != 'code10-c1'] + rows_for('code10-c1', [2, 2, 1, 0, 2])
    assert A.p2(r, False)['verdict'] is False


def test_p3_spread_and_untested():
    r = [x for x in base() if x['label'] != 'E36-L64'] + rows_for('E36-L64', [18] * 5)
    assert A.p3(r, False)['verdict'] is False  # 18 - 15 = 3 > 2
    r = [x for x in base() if x['label'] != 'E36-L24'] + rows_for('E36-L24', [15] * 2) + \
        rows_for('E36-L24', [15] * 3, recon=False)
    assert A.p3(r, False)['verdict'] is None  # only 2 reconstruction-passing runs


def test_s1_s2():
    r = [x for x in base() if x['label'] != 'E36-L10'] + rows_for('E36-L10', [8] * 5)
    assert A.s1(r, False)['verdict'] is False
    r = [x for x in base() if x['label'] != 'E30-L32'] + rows_for('E30-L32', [16] * 5)
    assert A.s2(r, False)['verdict'] is False


def test_converged_only_filter():
    r = [x for x in base() if x['label'] != 'E10a-L32']
    r += rows_for('E10a-L32', [0, 0, 0], conv=True) + rows_for('E10a-L32', [9, 9], conv=False)
    assert A.p1(r, False)['verdict'] is False  # max 9 > 5 with all runs
    assert A.p1(r, True)['verdict'] is True


def test_gates_make_untested():
    v = A.verdicts(base(), False, {'pipeline': {'pass': False}, 'capacity_p1': {'pass': True}})
    assert all(v[k]['verdict'] is None for k in v) and A.outcome(v) == 'untested'
    v = A.verdicts(base(), False, {'pipeline': {'pass': True}, 'capacity_p1': {'pass': False}})
    assert v['P1']['verdict'] is None and v['P2']['verdict'] is True
    assert A.outcome(v) == 'mixed'
