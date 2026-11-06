"""Metrics for pw02-pw04, following docs/METRICS_SPEC.md (binding).

Weighted variants take per-row weights (rows = distinct states with frequency
weights on raw data); with `w=None` the MIG and DCI functions reduce exactly to
the disentanglement_lib computations (`mig.py::_compute_mig`,
`dci.py::_compute_dci`, commit 86a644d4ed35). Deviations, all recorded in the
protocols: DCI classifier with a fixed `random_state`, balanced accuracy and
plain-average completeness added; coverage adapted to signed latents.
"""
import numpy as np
import scipy.stats
from scipy.optimize import linear_sum_assignment
from sklearn import ensemble
from sklearn.metrics import mutual_info_score

AU_THRESHOLD = 0.01
AU_ROBUSTNESS = (0.001, 0.05, 0.1)
POLARIZED_RATIO = 10.0
ASPERTI_TOL = 0.1
MIG_BINS = 20
F1_THRESHOLD = 0.9
KARVONEN_GRID = tuple(np.round(np.arange(0.0, 1.01, 0.1), 1))
DCI_RANDOM_STATE = 0


def _norm(w, n):
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (n,) or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Invalid weights')
    return w / w.sum()


def wmean(x, w):
    x = np.asarray(x, dtype=np.float64)
    return np.tensordot(_norm(w, len(x)), x, axes=(0, 0))


def wvar(x, w):
    """Population variance over rows with weights (Burda's Cov_x over the evaluation distribution)."""
    x = np.asarray(x, dtype=np.float64)
    w = _norm(w, len(x))
    m = np.tensordot(w, x, axes=(0, 0))
    return np.tensordot(w, (x - m) ** 2, axes=(0, 0))


# ------------------------------------------------------------- active units

def active_units(mu, w=None, threshold=AU_THRESHOLD):
    """Burda et al.: unit u active iff A_u = Var_x(E_q[u|x]) > threshold (strict)."""
    A = wvar(mu, w)
    mask = A > threshold
    return int(mask.sum()), mask, A


def asperti_diagnostic(mu, logvar, w=None, tol=ASPERTI_TOL):
    """Per unit: Var_x(mu), E_x[sigma^2], their sum (Asperti Eq. 13, about 1), not-converged flag."""
    var_mu = wvar(mu, w)
    mean_s2 = wmean(np.exp(np.asarray(logvar, dtype=np.float64)), w)
    s = var_mu + mean_s2
    return dict(var_mu=var_mu, mean_sigma2=mean_s2, sum=s, not_converged=np.abs(s - 1) > tol)


def polarization_ratio(A, threshold=AU_THRESHOLD):
    """Smallest active A_u / largest inactive A_u; None when one side is empty."""
    act, ina = A[A > threshold], A[A <= threshold]
    if len(act) == 0 or len(ina) == 0 or ina.max() <= 0:
        return None
    return float(act.min() / ina.max())


def categorical_activity(probs, w=None):
    """Per categorical variable: sum over classes of Var_x(q(c|x)) (pw02 secondary rule)."""
    n, v, c = probs.shape
    return wvar(probs.reshape(n, v * c), w).reshape(v, c).sum(1)


# --------------------------------------------------------------------- MIG

def histogram_discretize(x, num_bins=MIG_BINS):
    """disentanglement_lib `_histogram_discretize` for one code vector: equal-width bins."""
    return np.digitize(x, np.histogram(x, num_bins)[1][:-1])


def discrete_mi(a, b, w=None):
    """Plug-in mutual information in nats; unweighted path is sklearn.mutual_info_score (as dlib)."""
    if w is None:
        return mutual_info_score(a, b)
    w = _norm(w, len(a))
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(joint, (ia, ib), w)
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())


def _weighted_mi(ia, ib, w):
    ka, kb = ia.max() + 1, ib.max() + 1
    joint = np.bincount(ia * kb + ib, weights=w, minlength=ka * kb).reshape(ka, kb)
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())


def mig(codes, factors, w=None, num_bins=MIG_BINS):
    """MIG (disentanglement_lib convention): codes discretised into equal-width bins over each
    code's range, plug-in MI in nats, H(v_k) = empirical entropy. Factors constant on the
    evaluation rows are excluded and listed (dlib would divide by zero)."""
    codes, factors = np.asarray(codes), np.asarray(factors)
    if w is not None:
        keep = np.asarray(w) > 0
        codes, factors, w = codes[keep], factors[keep], np.asarray(w)[keep]
    J, K = codes.shape[1], factors.shape[1]
    disc = np.stack([histogram_discretize(codes[:, j], num_bins) for j in range(J)], 1)
    if w is None:  # disentanglement_lib path, sklearn plug-in MI
        m = np.array([[discrete_mi(disc[:, j], factors[:, k]) for k in range(K)] for j in range(J)])
        H = np.array([discrete_mi(factors[:, k], factors[:, k]) for k in range(K)])
    else:  # same plug-in estimator with weights, label indices computed once
        w = _norm(w, len(codes))
        inv_c = [np.unique(disc[:, j], return_inverse=True)[1] for j in range(J)]
        inv_f = [np.unique(factors[:, k], return_inverse=True)[1] for k in range(K)]
        m = np.array([[_weighted_mi(a, b, w) for b in inv_f] for a in inv_c])
        H = np.array([_weighted_mi(b, b, w) for b in inv_f])
    const = H <= 1e-12
    sm = np.sort(m, axis=0)[::-1]
    if J == 1:  # a single code: the runner-up MI is taken as 0 (undefined in dlib, which needs >= 2 codes)
        sm = np.vstack([sm, np.zeros((1, K))])
    per = np.full(K, np.nan)
    per[~const] = (sm[0, ~const] - sm[1, ~const]) / H[~const]
    if const.all():
        raise ValueError('All factors constant on the evaluation rows')
    return dict(mig=float(np.mean(per[~const])), per_factor=per, mi=m, entropy=H,
                constant_factors=np.flatnonzero(const))


# --------------------------------------------------------------------- DCI

def dci(x_train, y_train, x_test, y_test, w_train=None, w_test=None, random_state=DCI_RANDOM_STATE):
    """DCI with disentanglement_lib's gradient-boosted classifiers (sklearn defaults, factors as
    classes) plus a fixed random_state; rows are points, columns codes / factors.

    Returns importance matrix R [codes, factors], disentanglement (dlib weighted), completeness
    (dlib weighted and plain average), informativeness as accuracy (train/test) and balanced accuracy."""
    J, K = x_train.shape[1], y_train.shape[1]
    R = np.zeros((J, K))
    acc_tr, acc_te, bacc_te = [], [], []
    wtr = None if w_train is None else _norm(w_train, len(x_train))
    wte = _norm(w_test, len(x_test))
    for k in range(K):
        model = ensemble.GradientBoostingClassifier(random_state=random_state)
        model.fit(x_train, y_train[:, k], sample_weight=wtr)
        R[:, k] = np.abs(model.feature_importances_)
        ptr = model.predict(x_train)
        pte = model.predict(x_test)
        acc_tr.append(float(np.sum(_norm(w_train, len(x_train)) * (ptr == y_train[:, k]))))
        acc_te.append(float(np.sum(wte * (pte == y_test[:, k]))))
        bacc_te.append(balanced_accuracy(y_test[:, k], pte, wte))
    return dict(importance=R, disentanglement=dci_disentanglement(R), completeness=dci_completeness(R),
                completeness_plain=float(np.mean(completeness_per_factor(R))),
                disentanglement_per_code=disentanglement_per_code(R),
                completeness_per_factor=completeness_per_factor(R),
                informativeness_train=float(np.mean(acc_tr)), informativeness_test=float(np.mean(acc_te)),
                balanced_accuracy_test=float(np.mean(bacc_te)), balanced_accuracy_test_per_factor=bacc_te)


def balanced_accuracy(y, p, w):
    """Weighted mean over classes with positive weight in y of per-class recall."""
    rec = []
    for c in np.unique(y):
        m = y == c
        if np.sum(w[m]) <= 0:
            continue
        rec.append(np.sum(w[m] * (p[m] == c)) / np.sum(w[m]))
    return float(np.mean(rec))


def disentanglement_per_code(R):
    if R.shape[1] == 1:  # one factor: entropy base 1 is undefined; every code is trivially disentangled
        return np.ones(R.shape[0])
    return 1.0 - scipy.stats.entropy(R.T + 1e-11, base=R.shape[1])


def dci_disentanglement(R):
    per = disentanglement_per_code(R)
    if R.sum() == 0.0:
        R = np.ones_like(R)
    return float(np.sum(per * R.sum(axis=1) / R.sum()))


def completeness_per_factor(R):
    if R.shape[0] == 1:  # one code: entropy base 1 is undefined; every factor is trivially complete
        return np.ones(R.shape[1])
    return 1.0 - scipy.stats.entropy(R + 1e-11, base=R.shape[0])


def dci_completeness(R):
    per = completeness_per_factor(R)
    if R.sum() == 0.0:
        R = np.ones_like(R)
    return float(np.sum(per * R.sum(axis=0) / R.sum()))


# ------------------------------------------------------- best-latent F1

def _f1(tp, pp, p):
    den = pp + p
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)


def best_f1(codes_dev, y_dev, codes_conf, y_conf, w_dev=None, w_conf=None, signs=(1, -1)):
    """Per (code j, fact f): classifier s * code_j > t, with sign s and threshold t chosen to
    maximise weighted F1 on the development rows over every distinct split of the sorted
    development values (all-positive included); F1 then evaluated on the confirmation rows.

    Returns dict with f1_conf [J, F], f1_dev [J, F], sign [J, F], threshold [J, F] and the
    confirmation positive mass per fact."""
    cd = np.asarray(codes_dev, dtype=np.float64)
    cc = np.asarray(codes_conf, dtype=np.float64)
    yd = np.asarray(y_dev, dtype=np.float64)
    yc = np.asarray(y_conf, dtype=np.float64)
    wd, wc = _norm(w_dev, len(cd)), _norm(w_conf, len(cc))
    J, F = cd.shape[1], yd.shape[1]
    Pd = wd @ yd
    Pc = wc @ yc
    best = np.full((J, F), -1.0)
    sign = np.ones((J, F))
    thr = np.full((J, F), -np.inf)
    for j in range(J):
        for s in signs:
            v = s * cd[:, j]
            order = np.argsort(-v, kind='stable')
            vs = v[order]
            ends = np.flatnonzero(np.r_[vs[1:] != vs[:-1], True])  # last index of each value group
            tp = np.cumsum(wd[order, None] * yd[order], axis=0)[ends]
            pp = np.cumsum(wd[order])[ends]
            f1 = _f1(tp, pp[:, None], Pd[None, :])  # [groups, F]
            g = np.argmax(f1, axis=0)
            val = f1[g, np.arange(F)]
            t = np.where(g + 1 < len(ends), 0.5 * (vs[ends[g]] + vs[np.minimum(ends[g] + 1, len(vs) - 1)]), -np.inf)
            better = val > best[j]
            best[j, better], sign[j, better], thr[j, better] = val[better], s, t[better]
    f1c = np.zeros((J, F))
    for j in range(J):
        pred = (sign[j][None, :] * cc[:, j][:, None]) > thr[j][None, :]  # [n, F]
        tp = (wc[:, None] * pred * yc).sum(0)
        pp = (wc[:, None] * pred).sum(0)
        f1c[j] = _f1(tp, pp, Pc)
    return dict(f1_conf=f1c, f1_dev=best, sign=sign, threshold=thr, positive_mass_conf=Pc)


def alignment_scores(f1, active, position_facts, threshold=F1_THRESHOLD):
    """Cov30, MonoFrac and Hungarian O2O from an F1 matrix [codes, 36 facts]."""
    f1a = f1[np.asarray(active, dtype=bool)]
    pos = np.asarray(position_facts)
    if len(f1a) == 0:
        return dict(cov30=0.0, mono_frac=None, o2o=0.0, n_active=0, best_per_fact=np.zeros(f1.shape[1]))
    best_fact = f1a.max(0)
    cov = float(np.mean(best_fact[pos] >= threshold))
    mono = float(np.mean(f1a.max(1) >= threshold))
    r, c = linear_sum_assignment(f1a[:, pos], maximize=True)
    o2o = float(np.sum(f1a[:, pos][r, c] >= threshold) / len(pos))
    return dict(cov30=cov, mono_frac=mono, o2o=o2o, n_active=int(len(f1a)), best_per_fact=best_fact,
                mean_best_f1_position=float(best_fact[pos].mean()))


def prevalence_floor_f1(y, w=None):
    """F1 of the trivial all-positive classifier per fact: 2p / (1 + p)."""
    p = wmean(y, w)
    return 2 * p / (1 + p)


def karvonen_coverage(codes_dev, codes_conf, y_conf, w_dev=None, w_conf=None, grid=KARVONEN_GRID,
                      facts=None):
    """Karvonen et al. Eq. 2-3 for nonnegative features: phi = I[f > t * f_max], f_max over the
    development rows, alive = nonzero firing on development rows; coverage with one global t
    (official code) and with a per-fact t (Eq. 3 as printed)."""
    cd, cc, yc = np.asarray(codes_dev), np.asarray(codes_conf), np.asarray(y_conf, dtype=np.float64)
    if (cd < 0).any():
        raise ValueError('Karvonen coverage assumes nonnegative features')
    wd, wc = _norm(w_dev, len(cd)), _norm(w_conf, len(cc))
    alive = (wd @ (cd > 0)) > 0
    if not alive.any():
        return dict(global_t=0.0, best_t=None, per_fact_t=0.0, n_alive=0)
    fmax = cd[:, alive].max(0)
    facts = np.arange(yc.shape[1]) if facts is None else np.asarray(facts)
    P = wc @ yc[:, facts]
    per_t = []
    for t in grid:
        pred = (cc[:, alive] > t * fmax).astype(np.float64)  # [n, A]
        tp = pred.T @ (wc[:, None] * yc[:, facts])  # [A, F]
        pp = wc @ pred
        per_t.append(_f1(tp, pp[:, None], P[None, :]).max(0))  # best feature per fact
    per_t = np.array(per_t)  # [T, F]
    means = per_t.mean(1)
    return dict(global_t=float(means.max()), best_t=float(grid[int(np.argmax(means))]),
                per_fact_t=float(per_t.max(0).mean()), n_alive=int(alive.sum()))


# ------------------------------------------------------------ intervals

def wilson(k, n, z=1.959963984540054):
    if n <= 0:
        raise ValueError('n must be positive')
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(centre - half), float(centre + half)
