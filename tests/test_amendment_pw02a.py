"""Unit tests for the pw02a/pw03a amendment: 80% snapshot, amended convergence rule, three-valued H3.2."""
import json
from pathlib import Path

import numpy as np
import torch

from pwlib import alignment as AL
from pwlib import data as D
from pwlib import sweep as S
from pwlib import training as T

ROOT = Path(__file__).resolve().parents[1]
GRID_A = json.loads((ROOT / 'experiments/pw02a_active_units_amended/sweep_grid.json').read_text())


def test_amended_grid():
    assert GRID_A['training']['steps'] == 60000
    assert GRID_A['training']['kl_warmup'] == dict(schedule='linear', fraction=0.1)
    assert GRID_A['training']['convergence_snapshot_fraction'] == 0.8
    assert 'A3' not in GRID_A['architectures'] and not any(c.get('arch') == 'A3' for c in GRID_A['cells'])
    assert not any('-A3-' in c for c in GRID_A['pilot_cells'])
    assert len(GRID_A['cells']) == 58 and sum(len(c['seeds']) for c in GRID_A['cells']) == 210
    assert len(GRID_A['pilot_cells']) == 20


# ------------------------------------------------------------- H3.2 / separation

def test_h32_comparator_without_gated_seeds_is_untested():
    comp = {'vae_b1': [0.2, 0.3], 'tc_b6': []}
    assert AL.h32_verdict([0.9, 0.95], comp) is None


def test_h32_false_when_measured_part_fails():
    assert AL.h32_verdict([0.7, 0.95], {'vae_b1': [0.2], 'tc_b6': []}) is False  # SAE below 0.8
    assert AL.h32_verdict([0.9, 0.95], {'vae_b1': [0.92], 'tc_b6': []}) is False  # scored comparator reaches SAE min


def test_h32_true_and_missing_sae():
    assert AL.h32_verdict([0.9, 0.95], {'vae_b1': [0.2], 'tc_b6': [0.5]}) is True
    assert AL.h32_verdict([], {'vae_b1': [0.2]}) is None


def test_separated_or_untested():
    assert AL.separated_or_untested([], [0.1]) is None
    assert AL.separated_or_untested([0.5, 0.6], [0.1, 0.4]) is True
    assert AL.separated_or_untested([0.3, 0.6], [0.1, 0.4]) is False


# ------------------------------------------------------------- convergence rule

def _curve(vals):
    return [dict(step=1000 * (i + 1), dev_objective=v) for i, v in enumerate(vals)]


def test_objective_converged_absolute_tolerance_near_zero():
    c = _curve([1.0] * 7 + [2e-4, 1.5e-4, 1e-4])  # 80% point = 2e-4, final 1e-4: -50% relative, 1e-4 absolute
    assert S.objective_converged(c) is False
    assert S.objective_converged(c, abs_tol=1e-3) is True
    plain = dict(substrate='bw', family='plain_ae')
    vae = dict(substrate='bw', family='gauss_vae')
    log = dict(log=dict(curve=c))
    assert S.converged_amended(plain, log) is True and S.converged_amended(vae, log) is False


def test_learned_gamma_convergence_uses_gate_and_active_set():
    cell = dict(substrate='b_lin_w128', family='gauss_vae', gamma='learned')
    falling = dict(log=dict(curve=_curve([10.0 - 30 * i for i in range(10)])), competence=True,
                   snapshot=dict(competence=True, active_dev=[0, 3, 7]), active_dev_final=[0, 3, 7])
    assert S.converged_amended(cell, falling) is True  # objective still falling, but gate + set stable
    assert S.converged_amended(cell, dict(falling, active_dev_final=[0, 3])) is False
    assert S.converged_amended(cell, dict(falling, snapshot=dict(competence=False, active_dev=[0, 3, 7]))) is False
    assert S.converged_amended(cell, dict(falling, competence=False)) is False
    fixed = dict(cell, gamma='fixed')
    assert S.converged_amended(fixed, falling) is False  # fixed gamma: plain objective test


# ------------------------------------------------------------- snapshot plumbing

def test_train_snapshot_at_fraction():
    S10, _ = D.code10_states()
    u = np.full(len(S10), 1 / len(S10))
    data = D.TableData(S10, u, {'dev': u}, len(S10), 'code10')
    cell = dict(cell_id='code10-gauss_vae', substrate='code10', family='gauss_vae', arch='A1', beta=1.0, lr=3e-4,
                latent=32)
    torch.set_num_threads(1)
    model, log, snap = T.train(cell, 1, data, GRID_A, steps=20, log_every=10, snapshot_fraction=0.8)
    assert snap['step'] == 16 and log['kl_warmup'] == GRID_A['training']['kl_warmup']
    final = model.state_dict()
    assert any(not torch.equal(snap['state_dict'][k], final[k]) for k in final)
    _, log2 = T.train(cell, 1, data, GRID_A, steps=20, log_every=10)  # default: no snapshot, same trajectory
    assert [r['train_loss'] for r in log2['curve']] == [r['train_loss'] for r in log['curve']]


def test_execute_run_records_snapshot(tmp_path):
    S10, bit_of_pos = D.code10_states()
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'models').mkdir()
    (tmp_path / 'runs').mkdir()
    np.savez(tmp_path / 'raw' / 'data_context.npz', code10_states=S10, code10_bit_of_pos=bit_of_pos)
    grid = json.loads(json.dumps(GRID_A))
    grid['training']['eval_every_steps'] = 10
    (tmp_path / 'sweep_grid_used.json').write_text(json.dumps(grid))
    cell = dict(cell_id='code10-gauss_vae-A1-b1-lr0.0003', substrate='code10', family='gauss_vae', arch='A1',
                beta=1.0, lr=3e-4, latent=32)
    out = S.execute_run((cell, 1, str(tmp_path), True, 20))
    assert out['status'] == 'ok', out
    rec = json.loads((tmp_path / 'runs' / 'code10-gauss_vae-A1-b1-lr0.0003_s1.json').read_text())
    assert rec['snapshot']['step'] == 16 and 'exact' in rec['snapshot']['metrics_dev']
    assert isinstance(rec['snapshot']['active_dev'], list) and isinstance(rec['active_dev_final'], list)
    assert S.converged_amended(cell, rec) in (True, False)
