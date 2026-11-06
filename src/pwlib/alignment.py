"""Latent-to-fact alignment scoring for pw03 (one code matrix at a time).

Metrics (docs/METRICS_SPEC.md sections 5-7; pw03 PROTOCOL.md section 4):
M1 best-latent F1 with sign and threshold fitted on development rows (adapted
coverage rule for signed latents), Cov30 / MonoFrac / Hungarian O2O at F1 >= 0.9,
Karvonen coverage for nonnegative SAE codes; M2 MIG (disentanglement_lib
convention) over the 5 block variables and over the 36 facts; M3 DCI with
gradient-boosted classifiers on the 5 block variables. Label-permutation nulls
permute the label rows across evaluation rows (weights stay with the codes).
"""
import numpy as np

from . import metrics as M
from .data import POSITION_COLUMNS


def m1(codes_dev, codes_conf, Y_dev, Y_conf, w_dev, w_conf, active):
    bf = M.best_f1(codes_dev, Y_dev, codes_conf, Y_conf, w_dev, w_conf)
    sc = M.alignment_scores(bf['f1_conf'], active, POSITION_COLUMNS)
    return bf, sc


def mig_pair(codes_conf, Y_conf, B_conf, w_conf, active):
    a = np.asarray(active, dtype=bool)
    if not a.any():
        return dict(mig=0.0), dict(mig=0.0)
    return M.mig(codes_conf[:, a], B_conf, w_conf), M.mig(codes_conf[:, a], Y_conf, w_conf)


def score(codes_dev, codes_conf, Y_dev, Y_conf, B_dev, B_conf, w_dev, w_conf, active, n_perm=0, seed=0,
          dci=None, karvonen=False, perm_same_rows=True, n_perm_dci=0):
    """All alignment metrics for one model; returns (summary dict, arrays dict).

    dci: None (skip) or dict(x_train, y_train, x_test, y_test, w_train, w_test) built by the caller
    (rows for the classifiers may differ from the F1 rows on activation data)."""
    active = np.asarray(active, dtype=bool)
    bf, sc = m1(codes_dev, codes_conf, Y_dev, Y_conf, w_dev, w_conf, active)
    bmig, fmig = mig_pair(codes_conf, Y_conf, B_conf, w_conf, active)
    out = dict(cov30=sc['cov30'], mono_frac=sc['mono_frac'], o2o=sc['o2o'], n_active=sc['n_active'],
               mean_best_f1_position=sc.get('mean_best_f1_position'), block_mig=bmig['mig'], fact_mig=fmig['mig'],
               fact_mig_constant_facts=fmig.get('constant_factors', np.array([])).tolist(),
               prevalence_floor_f1=M.prevalence_floor_f1(Y_conf, w_conf))
    arrays = dict(f1_conf=bf['f1_conf'], f1_dev=bf['f1_dev'], sign=bf['sign'], threshold=bf['threshold'],
                  active=active)
    if 'mi' in bmig:
        arrays['block_mi'], arrays['fact_mi'] = bmig['mi'], fmig['mi']
    if karvonen:
        a = active
        out['karvonen_position'] = M.karvonen_coverage(codes_dev[:, a], codes_conf[:, a], Y_conf, w_dev, w_conf,
                                                       facts=POSITION_COLUMNS)
        out['karvonen_all36'] = M.karvonen_coverage(codes_dev[:, a], codes_conf[:, a], Y_conf, w_dev, w_conf)
    if dci is not None and active.any():
        d = M.dci(dci['x_train'][:, active], dci['y_train'], dci['x_test'][:, active], dci['y_test'],
                  dci['w_train'], dci['w_test'])
        out['dci'] = {k: v for k, v in d.items() if k in ('disentanglement', 'completeness', 'completeness_plain',
                                                          'informativeness_train', 'informativeness_test',
                                                          'balanced_accuracy_test')}
        arrays['dci_importance'] = d['importance']
    if n_perm:
        rng = np.random.default_rng(seed)
        null = dict(cov30=[], mono_frac=[], block_mig=[], fact_mig=[], dci_disentanglement=[], dci_completeness=[])
        for i in range(n_perm):
            pc = rng.permutation(len(Y_conf))
            pd = pc if perm_same_rows else rng.permutation(len(Y_dev))
            _, s = m1(codes_dev, codes_conf, Y_dev[pd], Y_conf[pc], w_dev, w_conf, active)
            b, f = mig_pair(codes_conf, Y_conf[pc], B_conf[pc], w_conf, active)
            null['cov30'].append(s['cov30'])
            null['mono_frac'].append(s['mono_frac'])
            null['block_mig'].append(b['mig'])
            null['fact_mig'].append(f['mig'])
            if dci is not None and i < n_perm_dci and active.any():
                ptr = rng.permutation(len(dci['y_train']))
                pte = ptr if perm_same_rows else rng.permutation(len(dci['y_test']))
                d = M.dci(dci['x_train'][:, active], dci['y_train'][ptr], dci['x_test'][:, active], dci['y_test'][pte],
                          dci['w_train'], dci['w_test'])
                null['dci_disentanglement'].append(d['disentanglement'])
                null['dci_completeness'].append(d['completeness'])
        out['null_p95'] = {k: (float(np.percentile([x for x in v if x is not None], 95)) if any(x is not None for x in v)
                               else None) for k, v in null.items()}
        out['excess_over_null_p95'] = {
            k: (None if out['null_p95'][k] is None or obs is None else float(obs - out['null_p95'][k]))
            for k, obs in (('cov30', out['cov30']), ('mono_frac', out['mono_frac']), ('block_mig', out['block_mig']),
                           ('fact_mig', out['fact_mig']),
                           ('dci_disentanglement', out.get('dci', {}).get('disentanglement')),
                           ('dci_completeness', out.get('dci', {}).get('completeness')))}
        out['n_perm'] = n_perm
        out['n_perm_dci'] = min(n_perm, n_perm_dci) if dci is not None else 0
    return out, arrays


def h32_verdict(sae_cov30, comparator_cov30, min_cov=0.8):
    """pw03a H3.2 with three-valued (Kleene) logic. sae_cov30: Cov30 of the gated k=5 SAE seeds;
    comparator_cov30: {cell: [Cov30 of its gated seeds]}. False if a measured part fails (SAE seed below
    min_cov, or a scored comparator reaching the SAE minimum); None (untested) if the SAE or any comparator
    has zero gated seeds and nothing measured fails; True otherwise."""
    if sae_cov30 and min(sae_cov30) < min_cov:
        return False
    s_min = min(sae_cov30) if sae_cov30 else None
    if s_min is not None and any(v and max(v) >= s_min for v in comparator_cov30.values()):
        return False
    if s_min is None or any(not v for v in comparator_cov30.values()):
        return None
    return True


def separated_or_untested(a, b):
    """Complete seed separation min(a) > max(b); None (untested) if either side has no gated seeds."""
    if not a or not b:
        return None
    return bool(min(a) > max(b))
