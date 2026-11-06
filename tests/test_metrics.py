"""Metric unit tests with analytically known answers (docs/METRICS_SPEC.md)."""
import math

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from pwlib import metrics as M
from pwlib.models import GaussVAE, mss_log_densities, mws_log_densities


def factors(n=4000, K=4, card=5, seed=0):
    return np.random.default_rng(seed).integers(0, card, size=(n, K))


# ------------------------------------------------------------- active units

def test_au_constructed_encoder_known_active_dims():
    """Linear 'encoder' mu = x W with exactly 3 informative units of variance 1 and 5 near-dead ones."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 3))
    mu = np.c_[x, 1e-3 * rng.normal(size=(5000, 5))]
    n, mask, A = M.active_units(mu)
    assert n == 3 and mask[:3].all() and not mask[3:].any()
    assert M.polarization_ratio(A) > 1e4


def test_au_weighted_equals_expanded():
    """Weighted variance over distinct rows equals the variance of the expanded sample."""
    rng = np.random.default_rng(1)
    rows = rng.normal(size=(20, 4))
    counts = rng.integers(1, 9, size=20)
    expanded = np.repeat(rows, counts, axis=0)
    assert np.allclose(M.wvar(rows, counts), expanded.var(0))


def test_au_threshold_is_strict():
    mu = np.r_[np.zeros(32), np.full(32, 0.5)][:, None]  # variance exactly 0.0625 in binary floating point
    assert M.wvar(mu, None)[0] == 0.0625
    assert M.active_units(mu, threshold=0.0625)[0] == 0
    assert M.active_units(mu, threshold=0.0624)[0] == 1


def test_asperti_sum_is_one_for_prior_matched_unit():
    rng = np.random.default_rng(2)
    mu = rng.normal(0, math.sqrt(0.7), size=(20000, 1))
    logvar = np.full_like(mu, math.log(0.3))
    d = M.asperti_diagnostic(mu, logvar)
    assert abs(d['sum'][0] - 1.0) < 0.03 and not d['not_converged'][0]


def test_categorical_activity():
    probs = np.zeros((100, 2, 3))
    probs[:, 0] = [1, 0, 0]  # constant variable: inactive
    probs[:50, 1] = [1, 0, 0]
    probs[50:, 1] = [0, 1, 0]
    act = M.categorical_activity(probs)
    assert abs(act[0]) < 1e-12 and abs(act[1] - 0.5) < 1e-12


# --------------------------------------------------------------------- MIG

def test_mig_perfect_code_is_one():
    v = factors()
    rng = np.random.default_rng(3)
    codes = v + 0.01 * rng.normal(size=v.shape)
    assert M.mig(codes, v)['mig'] > 0.99


def test_mig_duplicated_code_is_zero():
    v = factors()
    codes = np.c_[v[:, 0], v[:, 0], v[:, 1], v[:, 1], v[:, 2], v[:, 2], v[:, 3], v[:, 3]].astype(float)
    assert abs(M.mig(codes, v)['mig']) < 1e-12


def test_mig_rotation_is_low():
    v = factors(card=20)
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    codes = (v - v.mean(0)) / v.std(0) @ Q
    assert M.mig(codes, v)['mig'] < 0.3


def test_mig_weighted_equals_expanded():
    v = factors(n=60, K=2, card=3, seed=5)
    rng = np.random.default_rng(5)
    codes = v + 0.3 * rng.normal(size=v.shape)
    counts = rng.integers(1, 6, size=60)
    a = M.mig(codes, v, w=counts)
    b = M.mig(np.repeat(codes, counts, 0), np.repeat(v, counts, 0))
    assert abs(a['mig'] - b['mig']) < 1e-12


def test_mig_excludes_constant_factor():
    v = factors(K=2)
    v = np.c_[v, np.zeros(len(v), dtype=int)]
    out = M.mig(v.astype(float), v)
    assert list(out['constant_factors']) == [2] and out['mig'] > 0.99


def test_mig_single_code_gap_is_top_mi():
    v = factors(K=2, card=2)
    out = M.mig(v[:, :1].astype(float), v)
    assert abs(out['per_factor'][0] - 1.0) < 1e-12 and abs(out['per_factor'][1]) < 0.01


def test_discrete_mi_is_natural_log():
    a = np.r_[np.zeros(500), np.ones(500)]
    assert abs(M.discrete_mi(a, a) - math.log(2)) < 1e-12


# --------------------------------------------------------------------- DCI

def test_dci_perfect_disentanglement():
    v = factors(n=3000, K=3, card=4, seed=6)
    rng = np.random.default_rng(6)
    codes = v + 0.05 * rng.normal(size=v.shape)
    d = M.dci(codes[:2000], v[:2000], codes[2000:], v[2000:])
    assert d['disentanglement'] > 0.95 and d['completeness'] > 0.95 and d['informativeness_test'] > 0.99
    assert d['balanced_accuracy_test'] > 0.99


def test_dci_rotation_low_disentanglement():
    v = factors(n=3000, K=3, card=4, seed=7)
    Q = np.array([[1, 1, 1], [1, -1, 0], [1, 1, -2]]) / np.sqrt([[3], [2], [6]])  # Helmert: every code mixes
    codes = ((v - v.mean(0)) / v.std(0)) @ Q.T
    d = M.dci(codes[:2000], v[:2000], codes[2000:], v[2000:])
    assert d['disentanglement'] < 0.6


def test_dci_seed_fixed_is_deterministic():
    v = factors(n=1000, K=2, card=3, seed=8)
    codes = v + np.random.default_rng(8).normal(size=v.shape)
    a, b = M.dci(codes, v, codes, v), M.dci(codes, v, codes, v)
    assert np.array_equal(a['importance'], b['importance'])


def test_dci_entropy_bases():
    R = np.eye(3)
    assert abs(M.dci_disentanglement(R) - 1) < 1e-9 and abs(M.dci_completeness(R) - 1) < 1e-9
    U = np.ones((4, 3))
    assert abs(M.dci_disentanglement(U)) < 1e-9 and abs(M.dci_completeness(U)) < 1e-9


def test_dci_single_code_is_finite():
    R = np.array([[0.3, 0.7]])
    assert M.dci_completeness(R) == 1.0 and np.isfinite(M.dci_disentanglement(R))


def test_balanced_accuracy_skips_zero_weight_class():
    y = np.array([0, 0, 1, 2])
    p = np.array([0, 1, 1, 0])
    w = np.array([0.5, 0.5, 0.0, 0.0])
    assert M.balanced_accuracy(y, p, w) == 0.5


def test_balanced_accuracy_rare_class():
    y = np.r_[np.zeros(95), np.ones(5)].astype(int)
    p = np.zeros(100, dtype=int)
    w = np.full(100, 0.01)
    assert M.balanced_accuracy(y, p, w) == 0.5


# ---------------------------------------------------------------- best F1

def test_best_f1_perfect_and_signed():
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, size=(500, 2))
    codes = np.c_[y[:, 0] + 0.1 * rng.normal(size=500), -y[:, 1] + 0.1 * rng.normal(size=500), rng.normal(size=500)]
    bf = M.best_f1(codes, y, codes, y)
    assert bf['f1_conf'][0, 0] == 1.0 and bf['f1_conf'][1, 1] == 1.0
    assert bf['sign'][1, 1] == -1 and bf['f1_conf'][2].max() < 0.9


def test_best_f1_matches_sklearn_on_confirmation():
    rng = np.random.default_rng(10)
    y = (rng.random((300, 3)) < [0.1, 0.4, 0.7]).astype(int)
    codes = y @ rng.normal(size=(3, 4)) + rng.normal(size=(300, 4))
    wd, wc = rng.random(300), rng.random(300)
    bf = M.best_f1(codes, y, codes, y, wd, wc)
    for j in range(4):
        for f in range(3):
            pred = bf['sign'][j, f] * codes[:, j] > bf['threshold'][j, f]
            ref = f1_score(y[:, f], pred, sample_weight=wc, zero_division=0.0)
            assert abs(ref - bf['f1_conf'][j, f]) < 1e-12


def test_best_f1_is_the_dev_optimum():
    """Brute force over all midpoints confirms the selected development F1 is maximal."""
    rng = np.random.default_rng(11)
    y = (rng.random((200, 1)) < 0.3).astype(int)
    c = (y[:, 0] + rng.normal(scale=0.8, size=200))[:, None]
    bf = M.best_f1(c, y, c, y)
    best = 0
    for s in (1, -1):
        for t in np.r_[np.unique(s * c), -np.inf]:
            best = max(best, f1_score(y[:, 0], s * c[:, 0] > t - 1e-12 if np.isfinite(t) else np.ones(200, bool)))
    assert abs(best - bf['f1_dev'][0, 0]) < 1e-12


def test_alignment_scores_counts():
    f1 = np.zeros((3, 36))
    f1[0, 0] = f1[1, 5] = 0.95
    f1[2, 25] = 0.99  # clear(A): derived fact, not in Cov30
    pos = [i for i in range(36) if not (25 <= i <= 29 or i == 35)]
    s = M.alignment_scores(f1, np.ones(3, bool), pos)
    assert s['cov30'] == pytest.approx(2 / 30) and s['mono_frac'] == 1.0 and s['o2o'] == pytest.approx(2 / 30)


def test_karvonen_coverage_perfect_sae_code():
    rng = np.random.default_rng(12)
    y = rng.integers(0, 2, size=(400, 3))
    codes = y * rng.uniform(1, 2, size=(400, 3))
    k = M.karvonen_coverage(codes, codes, y)
    assert k['global_t'] == pytest.approx(1.0) and k['best_t'] == 0.0 and k['per_fact_t'] == pytest.approx(1.0)


def test_wilson_interval():
    lo, hi = M.wilson(50, 100)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 1e-12
    assert M.wilson(0, 20000)[0] == 0.0


# --------------------------------------------------------------------- TC

def _tc_gaussian(rho, n=2000, s=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = torch.tensor([[1.0, rho], [rho, 1.0]], dtype=torch.float64)
    X = torch.randn((n, 2), generator=g, dtype=torch.float64) @ torch.linalg.cholesky(C).T
    logvar = torch.full_like(X, math.log(s * s))
    z = X + s * torch.randn(X.shape, generator=g, dtype=torch.float64)
    Ceff = C + s * s * torch.eye(2, dtype=torch.float64)
    true = 0.5 * (torch.log(torch.diag(Ceff)).sum() - torch.logdet(Ceff)).item()
    return z, X, logvar, true


@pytest.mark.parametrize('rho', [0.0, 0.8])
def test_tc_mss_recovers_gaussian_tc(rho):
    z, mu, lv, true = _tc_gaussian(rho)
    a, b = mss_log_densities(z, mu, lv, len(z))
    assert abs((a - b).mean().item() - true) < 0.06


@pytest.mark.parametrize('rho', [0.0, 0.8])
def test_tc_mws_chen_constant_offset(rho):
    """MWS with Chen's log(NM) constants equals the log(M)-normalised estimate plus (D-1) log N,
    which is close to the true TC: the Chen-constant value carries a (D-1) log N offset."""
    z, mu, lv, true = _tc_gaussian(rho)
    N = M_ = len(z)
    a, b = mws_log_densities(z, mu, lv, N)
    chen = (a - b).mean().item()
    assert abs(chen - math.log(N) - true) < 0.06
    a2 = a + math.log(N * M_) - math.log(M_)
    b2 = b + 2 * (math.log(N * M_) - math.log(M_))
    assert abs((a2 - b2).mean().item() - true) < 0.06


def test_gauss_vae_kl_matches_closed_form():
    m = GaussVAE(4, [8], 3, 'bernoulli')
    x = torch.rand(10, 4).round()
    out = m(x, torch.Generator().manual_seed(0))
    mu, lv = out['mu'], out['logvar']
    q = torch.distributions.Normal(mu, (0.5 * lv).exp())
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
    assert torch.allclose(out['kl_per_unit'], torch.distributions.kl_divergence(q, p), atol=1e-6)
