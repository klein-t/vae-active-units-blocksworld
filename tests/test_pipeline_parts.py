"""Unit tests for data controls, models, training determinism and the legality tools."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pwlib import blocksworld as bw
from pwlib import data as D
from pwlib import legality as LG
from pwlib import training as T
from pwlib.models import CatVAE, TopKSAE, gumbel_tau

ROOT = Path(__file__).resolve().parents[1]
GRID = json.loads((ROOT / 'experiments/pw02_active_units/sweep_grid.json').read_text())


@pytest.fixture(scope='module')
def legal():
    return bw.enumerate_legal_bruteforce()


# ------------------------------------------------------------------ data

def test_code10_control():
    S, bit_of_pos = D.code10_states()
    assert S.shape == (1024, 36) and len(np.unique(S, axis=0)) == 1024
    counts = np.bincount(bit_of_pos)
    assert sorted(counts) == [3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
    assert np.linalg.matrix_rank(S - S.mean(0)) == 10
    assert LG.codeword_valid(S, bit_of_pos).all()


def test_prod5_control(legal):
    S = D.prod5_states()
    assert S.shape == (7776, 36) and len(np.unique(S, axis=0)) == 7776
    assert not S[:, D.DERIVED_COLUMNS].any()
    assert np.linalg.matrix_rank(S - S.mean(0)) == 25
    assert abs(LG.legal_from_position_bits(S).mean() - 866 / 7776) < 1e-12


def test_legal_index_and_block_variables(legal):
    idx = D.LegalIndex(legal)
    assert np.array_equal(idx.index(legal[::-1]), np.arange(866)[::-1])
    assert idx.lookup(np.zeros((1, 36), dtype=np.int8))[0] == -1
    B = D.block_variables(legal)
    assert np.array_equal(B, np.array([bw.positions_from_state(v) for v in legal]))


def test_split_is_disjoint_and_fixed():
    tr, va = D.split_train_file()
    assert len(tr) == 12803 and len(va) == 3201 and len(np.intersect1d(tr, va)) == 0
    tr2, _ = D.split_train_file()
    assert np.array_equal(tr, tr2)


# ------------------------------------------------------------- legality

def test_violations_on_legal_states_are_empty(legal):
    assert not LG.violations(legal).any()


def test_violation_taxonomy_examples():
    s = bw.state_from_positions({'A': 'B', 'B': 'A', 'C': 'T', 'D': 'T', 'E': 'T'})
    assert LG.violations(s[None])[0, 4]  # V5 cycle
    s = bw.state_from_positions({'A': 'H', 'B': 'H', 'C': 'T', 'D': 'T', 'E': 'T'})
    assert LG.violations(s[None])[0, 2]  # V3 two held
    s = bw.state_from_positions({'A': 'C', 'B': 'C', 'C': 'T', 'D': 'T', 'E': 'T'})
    assert LG.violations(s[None])[0, 1]  # V2 two on one
    v = bw.state_from_positions({x: 'T' for x in 'ABCDE'})
    v[bw.INDEX['clear(A)']] = 0
    assert LG.violations(v[None])[0, 3] and not LG.violations(v[None])[0, :3].any()  # V4 only
    v = bw.state_from_positions({x: 'T' for x in 'ABCDE'})
    v[bw.INDEX['on_table(A)']] = 0
    assert LG.violations(v[None])[0, 0]  # V1


def test_violation_flags_agree_with_checker():
    rng = np.random.default_rng(0)
    legal = bw.enumerate_legal_bruteforce()
    S = legal[rng.integers(0, 866, 3000)].copy()
    flips = rng.integers(0, 36, 3000)
    S[np.arange(3000)[::2], flips[::2]] ^= 1  # corrupt half
    assert np.array_equal(~LG.violations(S).any(1), LG.checker_flags(S))


def test_exact_baselines(legal):
    w = np.full(866, 1 / 866)
    assert LG.baseline_b0() == 866 / 2 ** 36
    assert abs(LG.baseline_b2(np.full((5, 6), 1 / 6)) - 866 / 7776) < 1e-12
    marg = (w[:, None] * legal).sum(0)
    b1 = LG.baseline_b1(legal, marg)
    assert 0 < b1 < 1
    idx = D.LegalIndex(legal)
    rng = np.random.default_rng(1)
    n = 200_000
    bm = LG.block_marginals(legal, w)
    b2 = LG.baseline_b2(bm)
    mc = LG.mc_b2(bm, n, rng, idx) / n
    assert abs(mc - b2) < 4 * np.sqrt(b2 * (1 - b2) / n)
    assert LG.mc_b0(10_000, rng, idx) == 0


# ---------------------------------------------------------------- models

def test_gumbel_schedule():
    assert gumbel_tau(0) == 1.0 and gumbel_tau(999) == 1.0
    assert abs(gumbel_tau(1000) - np.exp(-0.03)) < 1e-12 and gumbel_tau(10 ** 6) == 0.5


def test_topk_sae_sparsity_and_unit_norm():
    m = TopKSAE(36, 64, 5, torch.zeros(36))
    x = torch.rand(32, 36).round()
    out = m(x)
    assert ((out['z'] > 0).sum(1) <= 5).all()
    m.W_dec.data *= 3
    m.renorm()
    assert torch.allclose(m.W_dec.norm(dim=1), torch.ones(64))


def test_cat_vae_kl_zero_at_uniform():
    m = CatVAE(36, [16], 4, 6, 'bernoulli')
    for p in m.enc.parameters():
        p.data.zero_()
    out = m(torch.rand(8, 36).round(), 1.0, torch.Generator().manual_seed(0))
    assert torch.allclose(out['kl_per_var'], torch.zeros_like(out['kl_per_var']), atol=1e-6)


def _code10_data():
    S, _ = D.code10_states()
    u = np.full(len(S), 1 / len(S))
    return D.TableData(S, u, {'dev': u}, len(S), 'code10')


@pytest.mark.parametrize('family,extra', [('gauss_vae', dict(arch='A1', beta=1.0, latent=32)),
                                          ('tc_vae', dict(arch='A1', beta=6.0, latent=32)),
                                          ('cat_vae', dict(arch='A1', beta=1.0, latent='16x6')),
                                          ('topk_sae', dict(k=5, latent=64)),
                                          ('plain_ae', dict(arch='A1', latent=32))])
def test_training_runs_and_is_deterministic(family, extra):
    cell = dict(cell_id=f'code10-{family}', substrate='code10', family=family, lr=3e-4, **extra)
    data = _code10_data()
    torch.set_num_threads(1)
    m1, log1 = T.train(cell, 1, data, GRID, steps=20, log_every=10)
    m2, log2 = T.train(cell, 1, data, GRID, steps=20, log_every=10)
    assert [r['train_loss'] for r in log1['curve']] == [r['train_loss'] for r in log2['curve']]
    arrays, metrics = T.evaluate(m1, cell, data)
    assert 'dev_recon_bits' in arrays and 0 <= metrics['dev']['bitwise'] <= 1
    _, log3 = T.train(cell, 2, data, GRID, steps=20, log_every=10)
    assert log3['run_seed'] != log1['run_seed']


def test_gaussian_decoder_gamma_floor():
    cell = dict(cell_id='b_lin_w64-x', substrate='b_lin_w64', family='gauss_vae', arch='A1', beta=1.0,
                gamma='learned', lr=3e-4, latent=8)
    m = T.build_model(cell, GRID, 16, torch.zeros(16))
    m.dec.log_gamma.data.fill_(-20.0)
    assert float(m.dec.effective_log_gamma().detach()) == GRID['training']['log_gamma_floor']
    fixed = T.build_model(dict(cell, gamma='fixed'), GRID, 16, torch.zeros(16))
    assert not fixed.dec.log_gamma.requires_grad


def test_kl_warmup_schedule_and_default_off():
    assert GRID['training']['kl_warmup'] is None  # default off
    spec = GRID['training']['kl_warmup_contingency']
    w = dict(schedule=spec['schedule'], fraction=spec['fraction'])
    assert T.kl_weight(1, 30000, None) == 1.0 and T.kl_weight(30000, 30000, None) == 1.0
    assert T.kl_weight(1, 30000, w) == 1 / 3000
    assert T.kl_weight(1500, 30000, w) == 0.5
    assert T.kl_weight(3000, 30000, w) == 1.0 and T.kl_weight(20000, 30000, w) == 1.0
    with pytest.raises(ValueError):
        T.kl_weight(1, 10, dict(schedule='cyclic', fraction=0.1))


@pytest.mark.parametrize('family,extra', [('gauss_vae', dict(arch='A1', beta=4.0, latent=32)),
                                          ('tc_vae', dict(arch='A1', beta=6.0, latent=32)),
                                          ('cat_vae', dict(arch='A1', beta=4.0, latent='16x6'))])
def test_kl_warmup_scales_kl_side_and_changes_training(family, extra):
    cell = dict(cell_id=f'code10-{family}', substrate='code10', family=family, lr=3e-4, **extra)
    data = _code10_data()
    torch.set_num_threads(1)
    model = T.build_model(cell, GRID, data.d_in, data.train_mean)
    x = data.sample(64, torch.Generator().manual_seed(0))
    l0, p0 = T.batch_loss(model, cell, x, 1, data.n_train_points, torch.Generator().manual_seed(1), kl_w=0.0)
    l1, p1 = T.batch_loss(model, cell, x, 1, data.n_train_points, torch.Generator().manual_seed(1), kl_w=1.0)
    assert abs(l0.item() - p0['rec']) < 1e-5  # weight 0: reconstruction only
    assert l1.item() > l0.item() and p1['kl_weight'] == 1.0
    warm = json.loads(json.dumps(GRID))
    warm['training']['kl_warmup'] = dict(schedule='linear', fraction=0.5)
    _, log_off = T.train(cell, 1, data, GRID, steps=20, log_every=10)
    _, log_on = T.train(cell, 1, data, warm, steps=20, log_every=10)
    assert log_off['kl_warmup'] is None and log_on['kl_warmup'] == warm['training']['kl_warmup']
    assert [r['kl_weight'] for r in log_on['curve']] == [1.0, 1.0] and log_off['curve'][0]['kl_weight'] == 1.0
    assert log_on['curve'][0]['train_loss'] != log_off['curve'][0]['train_loss']
