"""pw07 (post hoc, descriptive, added 2026-10-30 after the main results in response to review):
(A) KL per latent unit and (B) legality of decoded samples on the same footing as the per-block baseline.

No training and no model forward passes: only retained arrays of pw02a full_001, pw06 full_001 and the pw04a
descriptive sample pass. Every input is checked against its recorded SHA-256 (pw02a/pw06 MANIFEST.json, pw05
input_hashes.json); files without a recorded hash have theirs written into results.json. No hypothesis test,
no threshold chosen on the outcome (README.md).

Usage:
    uv run python -u experiments/pw07_unit_kl_and_matched_legality/analyze_pw07.py --run-id analysis_001 [--workers 8]
"""
import argparse
import math
import re
import statistics
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pwlib import blocksworld as bw  # noqa: E402
from pwlib import data as D  # noqa: E402
from pwlib import legality as LG  # noqa: E402
from pwlib import metrics as M  # noqa: E402
from pwlib.runio import dump_json, load_json, sha256  # noqa: E402

EXP = Path(__file__).resolve().parent
PW02 = ROOT / 'experiments' / 'pw02a_active_units_amended' / 'results' / 'full_001'
PW04D = ROOT / 'experiments' / 'pw04a_sample_legality_amended' / 'descriptive' / 'results' / 'pw02a_full_001'
PW05 = ROOT / 'experiments' / 'pw05_unit_anatomy'
PW06 = ROOT / 'experiments' / 'pw06_encoding_and_latent_size' / 'results' / 'full_001'
LN2 = math.log(2)
KEY = 'conf'  # test split = confirmation weights, as pw02a/pw05/pw06 and the report's counts and rates
ARCH_LABEL = {'A1': '1x64', 'A2': '2x256', 'A4': '2x1024'}
ARCHS, LRS = ('A1', 'A2', 'A4'), ('0.0003', '0.003')

# Numbers printed in report/versions/report-v1.5-draft.pdf (Typst source v1.5_src) that this analysis re-derives.
REPORT = dict(
    rate_ref_nats=7.84, rate_lr3e3_nats=7.92, rate_excess_range_nats=(1.2, 1.8), baseline=0.540,
    below_baseline_beta1='24 of 30', beta025_beats_baseline='none of 30',
    prior_legal_range_beta1=(0.109, 0.578), agg_legal_range_beta1=(0.546, 0.899),
    wf_range=(0.36, 0.72), wf_product_tol=0.014, cross_given_wf_range=(0.87, 0.97),
    implied_cross_range=(0.60, 0.72), derived_given_cross_range=(0.46, 0.81),
    agg_wf_range=(0.76, 0.94), agg_cross_min=0.966,
    pw06_rates={'E36-L32': 7.84, 'E30-L32': 6.49, 'E10a-L32': 2.12, 'E10b-L32': 2.07, 'code10-c1': 0.00,
                'code10-c2': 6.89, 'code10-c3': 8.20, 'code10-c4': 8.82, 'E36-L16': 7.92, 'E36-L24': 7.87,
                'E36-L64': 7.89, 'L8/10/12 range': (7.92, 7.96)})


def med(v):
    v = [x for x in v if x is not None]
    return float(statistics.median(v)) if v else None


def summ(v):
    return dict(median=med(v), min=float(min(v)), max=float(max(v)), values=[float(x) for x in v])


def manifest(run_dir):
    return {r['path']: r['sha256'] for r in load_json(run_dir / 'MANIFEST.json')['files']}


def checked(path, expected):
    got = sha256(path)
    if got != expected:
        raise ValueError(f'hash mismatch: {path}')
    return path


def parse_cell(cid):
    m = re.fullmatch(r'(\w+)-gauss_vae-(A\d)-b([\d.]+)-lr([\d.]+)', cid)
    if not m:
        raise ValueError(cid)
    return dict(substrate=m[1], arch=m[2], beta=float(m[3]), lr=m[4])


# ------------------------------------------------------------------------ A: KL per unit

def unit_kl_row(z, active_mask_stored):
    """Per-unit KL (nats, conf weights), rate, active units and their split, from one raw npz. The closed-form
    check is relative for units above 0.01 nats and absolute below (float32 cancellation in the stored sums)."""
    mu, lv, w, klu = (z[f'{KEY}_{k}'].astype(np.float64) for k in ('mu', 'logvar', 'weights', 'kl_per_unit'))
    kl_closed = ((0.5 * (mu ** 2 + np.exp(lv) - lv - 1)) * w[:, None]).sum(0)
    rel_all = np.abs(kl_closed - klu) / np.maximum(np.abs(klu), 1e-9)
    rel = float(rel_all[klu > 0.01].max()) if (klu > 0.01).any() else 0.0  # float32 cancellation dominates below
    abs_small = float(np.abs(kl_closed - klu)[klu <= 0.01].max()) if (klu <= 0.01).any() else 0.0
    n_au, mask, A = M.active_units(z[f'{KEY}_mu'], z[f'{KEY}_weights'])
    if active_mask_stored is not None and not np.array_equal(mask, active_mask_stored):
        raise ValueError('recomputed active set differs from the stored one')
    R = float(klu.sum())
    act = klu[mask]
    p = w / w.sum()
    HX = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    order = np.argsort(-klu)
    return dict(
        latent=int(len(klu)), au=int(n_au), rate_nats=R, rate_bits=R / LN2, H_X_nats=HX, rate_minus_H_nats=R - HX,
        kl_sorted_nats=klu[order].tolist(), active_sorted=mask[order].tolist(),
        kl_active_nats=float(act.sum()), kl_inactive_nats=float(klu[~mask].sum()),
        kl_inactive_max_nats=float(klu[~mask].max()) if (~mask).any() else None,
        frac_rate_active=float(act.sum() / R) if R > 0 else None,
        mean_kl_per_active_nats=float(act.mean()) if len(act) else None,
        median_kl_per_active_nats=float(np.median(act)) if len(act) else None,
        min_kl_active_nats=float(act.min()) if len(act) else None,
        max_kl_active_nats=float(act.max()) if len(act) else None,
        cv_kl_active=float(act.std() / act.mean()) if len(act) > 1 else None,
        n_active_above_ln2=int((act > LN2).sum()),
        rate_over_au_nats=R / n_au if n_au else None,
        kl_closed_form_max_rel=rel, kl_closed_form_max_abs_small_units_nats=abs_small)


def part_a():
    man02, man06 = manifest(PW02), manifest(PW06)
    au_table = {r['run']: r for r in load_json(checked(PW02 / 'au_table.json', man02['au_table.json']))}
    au_sets = load_json(checked(PW02 / 'au_per_run.json', man02['au_per_run.json']))
    pw05 = load_json(PW05 / 'results' / 'analysis_001' / 'results.json')
    pw05_q = {r['name']: r for r in pw05['metrics']['per_run']}
    rows, checks = [], dict(au_equal_stored=True, rate_equal_pw05=True, kl_closed_form_max_rel=0.0,
                            kl_closed_form_max_abs_small_units_nats=0.0)
    cells = [f'bw-gauss_vae-{a}-b{b}-lr{lr}' for b in ('1', '0.25') for a in ARCHS for lr in LRS] + \
            [f'{s}-gauss_vae-{a}-b1-lr0.0003' for s in ('code10', 'prod5') for a in ARCHS]
    for cid in cells:
        seeds = sorted(r['seed'] for r in au_table.values() if r['cell_id'] == cid)
        if not seeds:
            raise KeyError(cid)
        for s in seeds:
            name = f'{cid}_s{s}'
            rec = au_table[name]
            if rec['status'] != 'ok':
                raise ValueError(f'{name} status {rec["status"]}')
            stored = np.isin(np.arange(rec['latent']), au_sets[name])
            with np.load(checked(PW02 / 'raw' / f'{name}.npz', man02[f'raw/{name}.npz'])) as z:
                row = unit_kl_row(z, stored)
            if row['au'] != rec['au']:
                raise ValueError(f'{name}: AU {row["au"]} != stored {rec["au"]}')
            if name in pw05_q:
                d = abs(row['rate_bits'] - pw05_q[name]['q1']['rate_bits'])
                row['pw05_rate_abs_diff_bits'] = d
                checks['rate_equal_pw05'] &= d < 1e-9
            checks['kl_closed_form_max_abs_small_units_nats'] = max(
                checks['kl_closed_form_max_abs_small_units_nats'], row['kl_closed_form_max_abs_small_units_nats'])
            checks['kl_closed_form_max_rel'] = max(checks['kl_closed_form_max_rel'], row['kl_closed_form_max_rel'])
            rows.append(dict(name=name, cell_id=cid, seed=s, competence=bool(rec['competence']),
                             converged=bool(rec['converged']), **parse_cell(cid), **row))
    # pw06 follow-up: re-encodings, copies and latent sizes at the reference configuration
    p6 = load_json(checked(PW06 / 'results.json', man06['results.json']))
    rows6 = []
    for r in p6['per_run']:
        if r['family'] != 'gauss_vae':
            continue
        with np.load(checked(PW06 / 'raw' / f'{r["run"]}.npz', man06[f'raw/{r["run"]}.npz'])) as z:
            stored = np.isin(np.arange(r['latent']), r['active'])
            row = unit_kl_row(z, stored)
        if row['au'] != r['count']:
            raise ValueError(f'pw06 {r["run"]}: count {row["au"]} != {r["count"]}')
        d = abs(row['rate_nats'] - r['rate_nats'])
        checks['rate_equal_pw06'] = checks.get('rate_equal_pw06', True) and d < 1e-9
        checks['kl_closed_form_max_abs_small_units_nats'] = max(
            checks['kl_closed_form_max_abs_small_units_nats'], row['kl_closed_form_max_abs_small_units_nats'])
        checks['kl_closed_form_max_rel'] = max(checks['kl_closed_form_max_rel'], row['kl_closed_form_max_rel'])
        rows6.append(dict(name=r['run'], label=r['label'], seed=r['seed'], recon_ok=r['recon_ok'],
                          converged=r['converged'], **row))
    # E36-L32 in pw06 reuses the pw02a reference cell and seeds: per-unit KL must be identical
    ref02 = {r['seed']: r for r in rows if r['cell_id'] == 'bw-gauss_vae-A2-b1-lr0.0003'}
    checks['pw06_E36_L32_equals_pw02a_reference'] = all(
        np.allclose(r['kl_sorted_nats'], ref02[r['seed']]['kl_sorted_nats'], rtol=1e-6, atol=1e-9)
        for r in rows6 if r['label'] == 'E36-L32')
    checks['au_equal_stored'] = True  # every mismatch above raises
    fields = ('au', 'rate_nats', 'rate_minus_H_nats', 'kl_active_nats', 'kl_inactive_nats', 'frac_rate_active',
              'mean_kl_per_active_nats', 'median_kl_per_active_nats', 'min_kl_active_nats', 'max_kl_active_nats',
              'cv_kl_active', 'n_active_above_ln2', 'rate_over_au_nats')

    def cell_summary(group):
        out = {}
        for f in fields:
            vals = [g[f] for g in group if g[f] is not None]
            out[f] = summ(vals) if vals else None
        L = group[0]['latent']
        out['median_sorted_kl_nats'] = np.median(np.array([g['kl_sorted_nats'] for g in group]), 0).tolist() \
            if all(g['latent'] == L for g in group) else None
        out['n_runs'] = len(group)
        return out
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r['cell_id'], []).append(r)
    by_label = {}
    for r in rows6:
        by_label.setdefault(r['label'], []).append(r)
    summaries = {k: cell_summary(v) for k, v in by_cell.items()}
    summaries6 = {k: cell_summary(v) for k, v in by_label.items()}
    # the examiner's arithmetic: median rate / median count at 2x256, both learning rates
    exam = {}
    for cid, (R_rep, n_rep) in {'bw-gauss_vae-A2-b1-lr0.0003': (7.84, 15),
                                'bw-gauss_vae-A2-b1-lr0.003': (7.92, 10)}.items():
        s = summaries[cid]
        exam[cid] = dict(examiner_rate=R_rep, examiner_units=n_rep, examiner_ratio=R_rep / n_rep,
                         median_rate=s['rate_nats']['median'], median_au=s['au']['median'],
                         ratio_of_medians=s['rate_nats']['median'] / s['au']['median'],
                         median_of_per_run_mean_kl_active=s['mean_kl_per_active_nats']['median'],
                         median_of_per_run_median_kl_active=s['median_kl_per_active_nats']['median'])
    beta1 = [r for r in rows if r['beta'] == 1.0 and r['substrate'] == 'bw']
    global_ = dict(
        beta1_rate_minus_H_range_nats=(min(r['rate_minus_H_nats'] for r in beta1),
                                       max(r['rate_minus_H_nats'] for r in beta1)),
        beta1_rate_range_nats=(min(r['rate_nats'] for r in beta1), max(r['rate_nats'] for r in beta1)),
        beta1_au_range=(min(r['au'] for r in beta1), max(r['au'] for r in beta1)),
        beta1_mean_kl_active_range_nats=(min(r['mean_kl_per_active_nats'] for r in beta1),
                                         max(r['mean_kl_per_active_nats'] for r in beta1)),
        beta1_frac_rate_active_min=min(r['frac_rate_active'] for r in beta1),
        beta1_max_inactive_unit_kl_nats=max(r['kl_inactive_max_nats'] for r in beta1),
        beta1_spearman_au_vs_mean_kl_active=float(
            _spearman([r['au'] for r in beta1], [r['mean_kl_per_active_nats'] for r in beta1])),
        beta1_spearman_au_vs_rate=float(_spearman([r['au'] for r in beta1], [r['rate_nats'] for r in beta1])))
    return dict(per_run=rows, per_run_pw06=rows6, per_cell=summaries, per_cell_pw06=summaries6,
                examiner_check=exam, global_=global_, checks=checks)


def _spearman(x, y):
    from scipy.stats import spearmanr
    return spearmanr(x, y)[0]


# ------------------------------------------------------------------------ B: legality on equal terms

def rederive(S):
    """Replace clear(X) and hand_empty by their values derived from the 30 position bits, exactly as the
    per-block baseline derives them (pwlib.legality.mc_b2)."""
    S = S.copy()
    held = S[:, LG.HOLD]
    S[:, bw.HAND_EMPTY] = (held.sum(1) == 0).astype(np.int8)
    S[:, LG.CLEAR] = ((held == 0) & (LG.on_matrix(S).sum(1) == 0)).astype(np.int8)
    return S


def block_place_cols(x):
    return [bw.INDEX[f'on_table({x})'] if v == 'T' else bw.INDEX[f'holding({x})'] if v == 'H'
            else bw.INDEX[f'on({x},{v})'] for v in bw.POSITION_VALUES[x]]


def anatomy(S, legal_table, b2):
    """Original legality, re-derived legality and the funnel for one set of 36-bit samples."""
    index = D.LegalIndex(legal_table)
    n = len(S)
    chk = LG.checker_flags(S)
    tab = LG.table_member(S, index)
    if not np.array_equal(chk, tab):
        raise ValueError('checker and 866-table disagree (original)')
    R = rederive(S)
    chk_r = LG.checker_flags(R)
    if not np.array_equal(chk_r, LG.table_member(R, index)):
        raise ValueError('checker and 866-table disagree (re-derived)')
    if not np.array_equal(chk_r, LG.legal_from_position_bits(S)):
        raise ValueError('re-derived legality differs from pwlib.legality.legal_from_position_bits')
    if (chk & ~chk_r).any():
        raise ValueError('an originally legal sample is illegal after re-derivation')
    wf_b = np.zeros((n, 5), dtype=bool)
    p = np.zeros((5, 6))
    for b, x in enumerate(bw.BLOCKS):
        blk = S[:, block_place_cols(x)]
        wf_b[:, b] = blk.sum(1) == 1
        np.add.at(p[b], blk[wf_b[:, b]].argmax(1), 1.0 / n)
    wf = wf_b.all(1)
    if (chk_r & ~wf).any():
        raise ValueError('re-derived legal sample that is not well formed')
    implied_pos = LG.baseline_b2(p)  # P(position-legal) if blocks were independent with the model's own marginals
    prod_wf = float(np.prod(wf_b.mean(0)))
    V = LG.violations(R)  # constraints between blocks, judged on re-derived states
    n_held = R[:, LG.HOLD].sum(1)
    wfi = wf & ~chk_r
    k, kr = int(chk.sum()), int(chk_r.sum())
    ci = M.wilson(kr, n)
    return dict(
        n=n, legal=k / n, legal_rederived=kr / n, legal_rederived_wilson95=ci,
        gain_from_rederivation=(kr - k) / n,
        wf_all=float(wf.mean()), wf_per_block=wf_b.mean(0).tolist(), wf_product_of_blocks=prod_wf,
        wf_minus_product=float(wf.mean()) - prod_wf,
        cross_given_wf=kr / int(wf.sum()) if wf.any() else None,
        cross_given_wf_wilson95=M.wilson(kr, int(wf.sum())) if wf.any() else None,
        implied_cross_given_wf_independent_blocks=implied_pos / prod_wf if prod_wf else None,
        implied_position_legal_independent_blocks=implied_pos,
        derived_given_cross=k / kr if kr else None,
        wf_failures_by_type=dict(  # fractions of well-formed samples, types may overlap
            two_on_one_block=float((V[:, 1] & wfi).sum() / max(wf.sum(), 1)),
            two_blocks_held=float(((n_held > 1) & wfi).sum() / max(wf.sum(), 1)),
            cycle=float((V[:, 4] & wfi).sum() / max(wf.sum(), 1)),
            held_block_supports=float((V[:, 5] & wfi).sum() / max(wf.sum(), 1))),
        below_baseline=dict(original=bool(k / n < b2), rederived=bool(kr / n < b2),
                            rederived_ci_entirely_below=bool(ci[1] < b2),
                            rederived_ci_entirely_above=bool(ci[0] > b2),
                            cross_given_wf=bool(wf.any() and kr / int(wf.sum()) < b2)))


def legality_task(t):
    out = dict(name=t['name'])
    with np.load(t['path']) as z:
        for tag in ('prior_mode', 'aggpost_mode'):
            S = np.unpackbits(z[f'{tag}_bits'], axis=1)[:, :bw.N_FACTS].astype(np.int8)
            a = anatomy(S, t['legal'], t['b2'])
            if not np.array_equal(LG.checker_flags(S), z[f'{tag}_legal']):
                raise ValueError(f'{t["name"]} {tag}: stored legal flags differ')
            out[tag] = a
    return out


def part_b(workers):
    man02 = manifest(PW02)
    with np.load(checked(PW02 / 'raw' / 'data_context.npz', man02['raw/data_context.npz'])) as z:
        legal, w_train, w_test = z['legal'], z['w_train'], z['w_test']
    if not np.array_equal(bw.enumerate_legal_bruteforce(), legal):
        raise ValueError('legal table not reproduced')
    b2 = LG.baseline_b2(LG.block_marginals(legal, w_train))
    b2_test = LG.baseline_b2(LG.block_marginals(legal, w_test))
    pw04 = load_json(PW04D / 'results.json')
    base_stored = pw04['metrics']['baselines']['exact']['B2']
    if abs(b2 - base_stored) > 1e-12:
        raise ValueError('baseline differs from the stored pw04a value')
    hashes = load_json(PW05 / 'input_hashes.json')['pw04a_descriptive_samples']
    names = [f'bw-gauss_vae-{a}-b{b}-lr{lr}_s{s}' for b in ('1', '0.25') for a in ARCHS for lr in LRS
             for s in range(1, 6)] + [f'bw-tc_vae-A2-b2-lr0.0003_s{s}' for s in (1, 2, 3)]
    tasks, new_hashes = [], {}
    for nm in names:
        rel = f'samples_{nm}.npz'
        path = PW04D / 'raw' / rel
        if rel in hashes:
            checked(path, hashes[rel])
        else:  # beta-TCVAE samples: no prior recorded hash; record it here
            new_hashes[rel] = sha256(path)
        tasks.append(dict(name=nm, path=str(path), legal=legal, b2=b2))
    with get_context('spawn').Pool(workers) as pool:
        out = {o['name']: o for o in pool.imap_unordered(legality_task, tasks)}
    pm = pw04['metrics']['per_model']
    pw05 = load_json(PW05 / 'results' / 'analysis_001' / 'results.json')['metrics']['legality_per_run']
    checks = dict(baseline_equals_pw04a=True, legal_equals_pw04a_rate=True, rederived_equals_pw05_position_legal=True)
    rows = []
    for nm in names:
        o = out[nm]
        for tag in ('prior_mode', 'aggpost_mode'):
            checks['legal_equals_pw04a_rate'] &= abs(o[tag]['legal'] - pm[nm][tag]['rate']) < 1e-12
            if nm in pw05:
                checks['rederived_equals_pw05_position_legal'] &= \
                    abs(o[tag]['legal_rederived'] - pw05[nm][tag]['position_legal']) < 1e-12
        fam = 'tc_vae' if '-tc_vae-' in nm else 'gauss_vae'
        m = re.fullmatch(r'bw-\w+-(A\d)-b([\d.]+)-lr([\d.]+)_s(\d)', nm)
        rows.append(dict(name=nm, cell_id=nm.rsplit('_s', 1)[0], family=fam, arch=m[1], beta=float(m[2]), lr=m[3],
                         seed=int(m[4]), prior=o['prior_mode'], aggpost=o['aggpost_mode']))

    def count(group, tag, key):
        return sum(r[tag]['below_baseline'][key] for r in group)

    sets = dict(beta1=[r for r in rows if r['family'] == 'gauss_vae' and r['beta'] == 1.0],
                beta025=[r for r in rows if r['family'] == 'gauss_vae' and r['beta'] == 0.25],
                tc_vae_b2=[r for r in rows if r['family'] == 'tc_vae'])
    counts = {}
    for sname, g in sets.items():
        counts[sname] = dict(n_runs=len(g), **{
            f'{tag}_below_{key}': count(g, tag, key) for tag in ('prior', 'aggpost')
            for key in ('original', 'rederived', 'rederived_ci_entirely_below', 'cross_given_wf')},
            **{f'{tag}_rederived_ci_entirely_above': sum(r[tag]['below_baseline']['rederived_ci_entirely_above']
                                                          for r in g) for tag in ('prior', 'aggpost')})
    fields = ('legal', 'legal_rederived', 'gain_from_rederivation', 'wf_all', 'wf_product_of_blocks',
              'wf_minus_product', 'cross_given_wf', 'implied_cross_given_wf_independent_blocks', 'derived_given_cross')
    per_cell = {}
    for r in rows:
        per_cell.setdefault(r['cell_id'], []).append(r)
    per_cell = {cid: {tag: {f: summ([r[tag][f] for r in g]) for f in fields} |
                      {'below_baseline_original': sum(r[tag]['below_baseline']['original'] for r in g),
                       'below_baseline_rederived': sum(r[tag]['below_baseline']['rederived'] for r in g),
                       'wf_failures_by_type_median': {k: med([r[tag]['wf_failures_by_type'][k] for r in g])
                                                      for k in g[0][tag]['wf_failures_by_type']}}
                      for tag in ('prior', 'aggpost')} | {'n_runs': len(g)}
                for cid, g in per_cell.items()}

    def cell_med_range(beta, tag, f):
        v = [per_cell[f'bw-gauss_vae-{a}-b{beta}-lr{lr}'][tag][f]['median'] for a in ARCHS for lr in LRS]
        return (min(v), max(v))
    b1 = sets['beta1']
    verify = dict(  # against the report's printed numbers
        prior_legal_run_range_beta1=(min(r['prior']['legal'] for r in b1), max(r['prior']['legal'] for r in b1)),
        agg_legal_run_range_beta1=(min(r['aggpost']['legal'] for r in b1), max(r['aggpost']['legal'] for r in b1)),
        wf_cell_median_range=cell_med_range('1', 'prior', 'wf_all'),
        wf_minus_product_max_abs_run=max(abs(r['prior']['wf_minus_product']) for r in b1),
        cross_given_wf_cell_median_range=cell_med_range('1', 'prior', 'cross_given_wf'),
        implied_cross_cell_median_range=cell_med_range('1', 'prior', 'implied_cross_given_wf_independent_blocks'),
        derived_given_cross_cell_median_range=cell_med_range('1', 'prior', 'derived_given_cross'),
        agg_wf_cell_median_range=cell_med_range('1', 'aggpost', 'wf_all'),
        agg_cross_given_wf_min_run=min(r['aggpost']['cross_given_wf'] for r in b1),
        rederived_prior_run_range_beta1=(min(r['prior']['legal_rederived'] for r in b1),
                                         max(r['prior']['legal_rederived'] for r in b1)),
        rederived_prior_cell_median_range_beta1=cell_med_range('1', 'prior', 'legal_rederived'),
        rederived_prior_cell_median_range_beta025=cell_med_range('0.25', 'prior', 'legal_rederived'),
        rederived_agg_cell_median_range_beta1=cell_med_range('1', 'aggpost', 'legal_rederived'))
    return dict(baseline=dict(B2_train_marginals=b2, B2_test_marginals=b2_test, pw04a_stored=base_stored),
                per_run=rows, per_cell=per_cell, counts=counts, verify_report=verify, checks=checks,
                input_hashes_recorded_here=new_hashes)


# ------------------------------------------------------------------------ figure (draft, not for report/)

def figure(a, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                         'legend.frameon': False, 'savefig.dpi': 300})
    blue, verm, green, orange, grey = '#0072B2', '#D55E00', '#009E73', '#E69F00', '#7f7f7f'
    fig, axes = plt.subplots(2, 2, figsize=(6.45, 5.0), sharey=True)
    axes = axes.ravel()
    XMAX = 25

    def curves(ax, groups, title):
        for label, runs, col, ls in groups:
            for r in runs:
                ax.plot(np.arange(1, r['latent'] + 1), r['kl_sorted_nats'], color=col, lw=0.5, alpha=0.35, ls=ls)
            L = runs[0]['latent']
            medc = np.median(np.array([r['kl_sorted_nats'] for r in runs]), 0)
            nau = med([r['au'] for r in runs])
            R = med([r['rate_nats'] for r in runs])
            ax.plot(np.arange(1, L + 1), medc, color=col, lw=1.4, ls=ls, label=f'{label}: {nau:g} units, {R:.2f} nats')
        ax.axhline(LN2, color=grey, lw=0.7, ls=(0, (4, 3)))
        ax.text(XMAX, LN2 + 0.02, '1 bit', color=grey, fontsize=7, ha='right', va='bottom')
        ax.set_ylim(0, 1.95)
        ax.set_xlim(0.5, XMAX)
        ax.set_title(title, loc='left', fontsize=9)
        ax.legend(fontsize=6.5, loc='upper right', bbox_to_anchor=(1.0, 0.97), handlelength=2.2)
    rows = a['per_run']
    sel = lambda cid: [r for r in rows if r['cell_id'] == cid]  # noqa: E731
    cols = {'A1': green, 'A2': blue, 'A4': verm}
    g1 = [(f'{ARCH_LABEL[ar]}, lr {"3e-4" if lr == "0.0003" else "3e-3"}', sel(f'bw-gauss_vae-{ar}-b1-lr{lr}'),
           cols[ar], '-' if lr == '0.0003' else '--') for ar in ARCHS for lr in LRS]
    curves(axes[0], g1, '(a) Blocksworld, $\\beta$ = 1, 32 latents')
    g2 = [(f'{s} {ARCH_LABEL[ar]}', sel(f'{s}-gauss_vae-{ar}-b1-lr0.0003'), cols[ar], '-' if s == 'prod5' else ':')
          for s in ('code10', 'prod5') for ar in ARCHS]
    curves(axes[1], g2, '(b) controls, $\\beta$ = 1, lr 3e-4')
    r6 = a['per_run_pw06']
    s6 = lambda lab: [r for r in r6 if r['label'] == lab]  # noqa: E731
    g3 = [('36 facts', s6('E36-L32'), blue, '-'), ('30 position facts', s6('E30-L32'), orange, '-'),
          ('10-bit index, order 1', s6('E10a-L32'), verm, '-'), ('10-bit index, order 2', s6('E10b-L32'), verm, '--'),
          ('code10, 2 copies', s6('code10-c2'), green, ':'), ('code10, 4 copies', s6('code10-c4'), green, '-')]
    curves(axes[2], g3, '(c) re-encodings, 2x256, lr 3e-4')
    seq = plt.get_cmap('viridis')(np.linspace(0, 0.85, 6))
    g4 = [(f'L = {L}', s6(lab), c, '-') for (L, lab), c in zip(
        [(8, 'E36-L8'), (10, 'E36-L10'), (12, 'E36-L12'), (16, 'E36-L16'), (32, 'E36-L32'), (64, 'E36-L64')], seq)]
    curves(axes[3], g4, '(d) latent size, 36 facts, 2x256, lr 3e-4')
    for ax in axes[2:]:
        ax.set_xlabel('latent, sorted by KL')
    for ax in axes[::2]:
        ax.set_ylabel('KL per latent (nats)')
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'fig_unit_kl.{ext}', bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    out_dir = EXP / 'results' / args.run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    a = part_a()
    b = part_b(args.workers)
    figure(a, out_dir)
    res = dict(experiment_id='pw07_unit_kl_and_matched_legality', run_id=args.run_id, kind='analysis',
               nature='post hoc, descriptive; added 2026-10-30 after the main results in response to review; '
                      'no hypothesis test, no registration or freeze',
               eval_split=KEY, report_numbers_checked=REPORT,
               inputs=dict(pw02a=str(PW02.relative_to(ROOT)), pw06=str(PW06.relative_to(ROOT)),
                           pw04a_descriptive=str(PW04D.relative_to(ROOT)),
                           pw02a_manifest_sha256=sha256(PW02 / 'MANIFEST.json'),
                           pw06_manifest_sha256=sha256(PW06 / 'MANIFEST.json'),
                           pw05_input_hashes_sha256=sha256(PW05 / 'input_hashes.json'),
                           pw05_results_sha256=sha256(PW05 / 'results' / 'analysis_001' / 'results.json'),
                           pw04a_descriptive_results_sha256=sha256(PW04D / 'results.json')),
               A_unit_kl=a, B_matched_legality=b, wall_seconds=time.time() - t0)
    dump_json(res, out_dir / 'results.json')
    print(dict(checks_A=a['checks'], checks_B=b['checks'], counts=b['counts'], seconds=round(time.time() - t0, 1)))


if __name__ == '__main__':
    main()
