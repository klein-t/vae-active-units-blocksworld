"""Independent audit 001, check 4: pw03a coverage (Cov30) and single-fact fractions (MonoFrac) from
checkpoints, with my own implementation of the signed best-F1 rule (pw03 PROTOCOL section 4 / METRICS_SPEC
section 7 project note):

classifier s * code_j > t; sign s in {+,-} and t chosen to maximise frequency-weighted F1 on the
development weights (pw02 val split) over every distinct split of the sorted values (all-positive
included); F1 then evaluated with the confirmation weights (thesis test file). Cov30 = fraction of the 30
position facts with max_j F1_conf >= 0.9 over active codes; MonoFrac = fraction of active codes whose best
F1_conf over the 36 facts is >= 0.9. Active codes: Burda AU set (conf weights) for Gaussian models, alive
(fires on >= 0.5% of conf-weighted states) for SAEs.

Sensitivity variants reported: dev-tie broken towards the largest vs the smallest positive set; active
set on dev instead of conf weights; code j chosen by dev F1 instead of max over conf F1.
H3.2 re-derived with the frozen three-valued rule (pw03a PROTOCOL change 2). Writes pw03a_recompute.json.
"""
import json
import sys

import numpy as np
import torch

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):  # usage without the data present
    print(__doc__)
    sys.exit(0)

import ind_common as C  # noqa: E402

CELLS = {'sae_k5': 'bw-topk_sae-k5-lr0.0003', 'sae_k10': 'bw-topk_sae-k10-lr0.0003',
         'vae_b1': 'bw-gauss_vae-A2-b1-lr0.0003', 'vae_b4': 'bw-gauss_vae-A2-b4-lr0.0003',
         'tc_b2': 'bw-tc_vae-A2-b2-lr0.0003', 'tc_b6': 'bw-tc_vae-A2-b6-lr0.0003'}
COMPARATORS = ('vae_b1', 'vae_b4', 'tc_b2', 'tc_b6')


def f1_table(v, Y, wd, wc, tie):
    """Best dev split per fact for one code; returns (dev F1, conf F1) arrays over the 36 facts."""
    best_d = np.full(Y.shape[1], -1.0)
    best_c = np.zeros(Y.shape[1])
    Pd, Pc = wd @ Y, wc @ Y
    for s in (1.0, -1.0):
        sv = s * v
        order = np.argsort(-sv, kind='stable')
        svs = sv[order]
        ends = np.r_[np.flatnonzero(svs[:-1] != svs[1:]), len(svs) - 1]  # cut after each distinct-value group
        TPd = np.cumsum(wd[order, None] * Y[order], 0)[ends]
        TPc = np.cumsum(wc[order, None] * Y[order], 0)[ends]
        PPd, PPc = np.cumsum(wd[order])[ends], np.cumsum(wc[order])[ends]
        F1d = 2 * TPd / np.maximum(PPd[:, None] + Pd[None, :], 1e-300)
        F1c = 2 * TPc / np.maximum(PPc[:, None] + Pc[None, :], 1e-300)
        if tie == 'small':  # first maximum = smallest positive set
            idx = F1d.argmax(0)
        else:  # last maximum = largest positive set
            idx = len(ends) - 1 - F1d[::-1].argmax(0)
        cand_d = F1d[idx, np.arange(Y.shape[1])]
        cand_c = F1c[idx, np.arange(Y.shape[1])]
        better = cand_d > best_d + 1e-15
        best_d = np.where(better, cand_d, best_d)
        best_c = np.where(better, cand_c, best_c)
    return best_d, best_c


def codes_for(run_dir, run, fam, k, legal):
    ck, sd = C.load_ckpt(run_dir, run)
    x = torch.as_tensor(legal, dtype=torch.float32)
    with torch.no_grad():
        if fam == 'topk_sae':
            p = (x - sd['b_dec']) @ sd['W_enc'] + sd['b_enc']
            v, i = p.topk(k, dim=1)
            z = torch.zeros_like(p).scatter(1, i, torch.relu(v))
            dec = z @ sd['W_dec'] + sd['b_dec']
            bits = dec.numpy() > 0.5
            return z.numpy().astype(np.float64), bits
        mu, _ = C.gauss_encode(sd, x)
        bits = C.decode(sd, mu).numpy() > 0
        return mu.numpy().astype(np.float64), bits


def score(codes, active, Y, wd, wc, tie='small'):
    D = np.zeros((len(active), Y.shape[1]))
    Cf = np.zeros_like(D)
    for n, j in enumerate(active):
        D[n], Cf[n] = f1_table(codes[:, j], Y, wd, wc, tie)
    pos = C.POSITION_FACTS
    if len(active) == 0:
        return dict(cov30=0.0, mono_frac=0.0, cov30_dev_selected=0.0, n_active=0)
    cov = float((Cf[:, pos].max(0) >= 0.9).mean())
    mono = float((Cf.max(1) >= 0.9).mean())
    jsel = D[:, pos].argmax(0)
    cov_devsel = float((Cf[jsel, pos] >= 0.9).mean())
    return dict(cov30=cov, mono_frac=mono, cov30_dev_selected=cov_devsel, n_active=len(active),
                covered_facts=[int(pos[i]) for i in np.flatnonzero(Cf[:, pos].max(0) >= 0.9)])


def main():
    run_dir = C.PW02 / 'full_001'
    ctx = np.load(run_dir / 'raw/data_context.npz')
    legal = ctx['legal'].astype(np.int8)
    Y = legal.astype(np.float64)
    # own weights (recompute_data.py verified they equal the stored ones; recomputed again here)
    d = C.load_thesis_data()
    tr, va = C.split_indices()
    row_of = {r.tobytes(): i for i, r in enumerate(legal)}
    cd = C.state_counts(d['Xtr'][va], d['Ytr'][va], row_of)
    cc = C.state_counts(d['Xte'], d['Yte'], row_of)
    wd, wc = cd / cd.sum(), cc / cc.sum()
    rec = json.loads((C.ROOT / 'experiments/pw03a_fact_alignment_amended/results/full_001/results.json').read_text())
    per_model = rec['metrics']['per_model']
    out, mism = {}, []
    for tag, cid in CELLS.items():
        fam = 'topk_sae' if 'topk' in cid else 'gauss'
        k = int(cid.split('-k')[1].split('-')[0]) if fam == 'topk_sae' else None
        seeds = [1, 2, 3, 4, 5] if cid.startswith('bw-gauss_vae') else [1, 2, 3]
        rows = []
        for s in seeds:
            run = f'{cid}_s{s}'
            codes, bits = codes_for(run_dir, run, fam, k, legal)
            eq = bits == legal.astype(bool)
            comp = bool(wd @ eq.all(1) >= 0.95 and wd @ eq.mean(1) >= 0.99)
            if fam == 'topk_sae':
                act_c = np.flatnonzero(wc @ (codes > 0) >= 0.005)
                act_d = np.flatnonzero(wd @ (codes > 0) >= 0.005)
            else:
                act_c = np.flatnonzero(C.wvar(codes, wc) > 0.01)
                act_d = np.flatnonzero(C.wvar(codes, wd) > 0.01)
            r = dict(run=run, seed=s, competence=comp)
            if comp:
                r['main'] = score(codes, act_c, Y, wd, wc, 'small')
                r['tie_large'] = score(codes, act_c, Y, wd, wc, 'large')
                r['active_dev'] = score(codes, act_d, Y, wd, wc, 'small')
                pm = per_model.get(run)
                r['recorded'] = None if pm is None else dict(cov30=pm['cov30'], mono_frac=pm['mono_frac'], n_active=pm['n_active'])
                if pm is None:
                    mism.append(dict(run=run, issue='gated run not scored in pw03a'))
                else:
                    for fld in ('cov30', 'mono_frac', 'n_active'):
                        if abs(r['main'][fld] - pm[fld]) > 1e-12:
                            mism.append(dict(run=run, field=fld, mine=r['main'][fld], recorded=pm[fld]))
            elif run in per_model:
                mism.append(dict(run=run, issue='ungated run scored in pw03a'))
            rows.append(r)
        out[tag] = dict(cell_id=cid, rows=rows)

    def h32(variant):
        sae = [r[variant]['cov30'] for r in out['sae_k5']['rows'] if r['competence']]
        comps = {t: [r[variant]['cov30'] for r in out[t]['rows'] if r['competence']] for t in COMPARATORS}
        fail = bool(sae) and (min(sae) < 0.8 or any(v and max(v) >= min(sae) for v in comps.values()))
        untested = (not sae) or any(not v for v in comps.values())
        verdict = False if fail else (None if untested else True)
        return dict(sae_k5_cov30=sae, comparator_cov30=comps,
                    comparator_max={t: (max(v) if v else None) for t, v in comps.items()}, verdict=verdict)

    res = dict(mismatches=mism, cells=out, h32={v: h32(v) for v in ('main', 'tie_large', 'active_dev')},
               h32_dev_selected=None)
    # variant: code chosen by dev F1
    sae = [r['main']['cov30_dev_selected'] for r in out['sae_k5']['rows'] if r['competence']]
    comps = {t: [r['main']['cov30_dev_selected'] for r in out[t]['rows'] if r['competence']] for t in COMPARATORS}
    res['h32_dev_selected'] = dict(sae_k5=sae, comparators=comps)
    res['h32_recorded'] = rec['metrics']['h32'] | dict(criterion=rec['criteria'])
    (C.OUT / 'pw03a_recompute.json').write_text(json.dumps(C.jsonable(res), indent=1))
    print('mismatches', mism)
    for v, h in res['h32'].items():
        print(v, h)
    print('dev-selected', res['h32_dev_selected'])
    for tag in out:
        print(tag, [(r['seed'], r['competence'], r.get('main', {}).get('cov30'), r.get('main', {}).get('mono_frac'),
                     r.get('main', {}).get('n_active')) for r in out[tag]['rows']])


if __name__ == '__main__':
    main()
