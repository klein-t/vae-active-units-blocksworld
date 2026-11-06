"""Independent audit 001, checks 2-3 and 6 for pw02a: AU, competence, convergence and verdicts.

For every run of pw02a full_001 (all 210, which covers and exceeds the requested stratified sample of
>= 40) and pilot_001 (20), from the saved checkpoint with my own forward pass:
- posterior means on the evaluation rows; AU = #units with weighted population Var_x(mu) > 0.01 (Burda,
  strict), on the confirmation distribution in the full run (dev in the pilot); robustness thresholds;
- competence on the development set (raw states: exact >= 0.95 and bitwise >= 0.99 from the decoded
  posterior mean, frequency-weighted; activations: stored current-state probe on the reconstruction,
  bitwise >= 0.98 and pooled R^2 >= 0.90);
- categorical activity, SAE alive counts, plain-AE covariance AU;
- the amended convergence rule from the logged curves / snapshot and my recomputed final active set;
- my mu vs the stored operands, my AU / active set vs au_table.json / au_per_run.json.
Raw-state evaluation weights are my own recomputation (recompute_data.py showed them equal to the stored
ones; here they are recomputed again from the thesis files). Activation rows come from the cached
activation files, which verify_activations.py regenerates independently.
Then H2.2 / H2.3 / H2.4 and the experiment-level competence gate are re-derived from the frozen criteria
text (pw02 PROTOCOL sections 6 and 8 as adapted by pw02a PROTOCOL section 2) with my numbers only.
Writes pw02a_recompute.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):  # usage without the data present
    print(__doc__)
    sys.exit(0)

import ind_common as C  # noqa: E402

torch.set_num_threads(4)
LRS = ('0.0003', '0.003')
ARCHS = ('A1', 'A2', 'A4')
SOURCES = ('w64', 'w128', 'w128s2', 'w128s3', 'w256')


def raw_weights():
    d = C.load_thesis_data()
    tr, va = C.split_indices()
    ctx = np.load(C.PW02 / 'full_001/raw/data_context.npz')
    legal = ctx['legal'].astype(np.int8)
    row_of = {r.tobytes(): i for i, r in enumerate(legal)}
    w = {}
    for k, X, Y in (('train', d['Xtr'][tr], d['Ytr'][tr]), ('dev', d['Xtr'][va], d['Ytr'][va]),
                    ('conf', d['Xte'], d['Yte'])):
        c = C.state_counts(X, Y, row_of)
        w[k] = c / c.sum()
    return legal, w, ctx


def bits_scores(xhat_bits, x, w):
    eq = (xhat_bits == x.astype(bool))
    w = w / w.sum()
    return float(w @ eq.mean(1)), float(w @ eq.all(1))


def eval_run(run_dir, run, cell, data, key_eval):
    ck, sd = C.load_ckpt(run_dir, run)
    assert ck['cell']['cell_id'] == cell['cell_id'] and ck['seed'] == int(run.rsplit('_s', 1)[1])
    fam, sub = cell['family'], cell['substrate']
    out = dict(run=run, cell_id=cell['cell_id'], family=fam, substrate=sub, seed=ck['seed'],
               run_seed_logged=ck['run_seed'], run_seed_mine=C.run_seed(cell['cell_id'], ck['seed']))
    sets = data(sub)
    res = {}
    for key, (x, w, extra) in sets.items():
        xt = torch.as_tensor(x, dtype=torch.float32)
        r = {}
        if fam in ('gauss_vae', 'tc_vae'):
            mu, lv = C.gauss_encode(sd, xt)
            r['latent'] = mu.shape[1]
            dec = C.decode(sd, mu)
            A = C.wvar(mu.numpy(), w)
            r['A'] = A
            r['mu'] = mu.numpy()
        elif fam == 'cat_vae':
            logits = C.linear_stack(sd, 'enc.', xt).view(len(xt), 16, 6)
            q = torch.softmax(logits, -1)
            hard = torch.nn.functional.one_hot(q.argmax(-1), 6).float().flatten(1)
            dec = C.decode(sd, hard)
            act = C.wvar(q.reshape(len(q), -1).numpy(), w).reshape(16, 6).sum(1)
            r['activity'] = act
        elif fam == 'topk_sae':
            p = (xt - sd['b_dec']) @ sd['W_enc'] + sd['b_enc']
            v, i = p.topk(int(cell['k']), dim=1)
            z = torch.zeros_like(p).scatter(1, i, torch.relu(v))
            dec = z @ sd['W_dec'] + sd['b_dec']
            fire = (w / w.sum()) @ (z.numpy() > 0)
            r['fire'] = fire
            r['latent'] = z.shape[1]
        else:  # plain_ae
            z = C.linear_stack(sd, 'enc.', xt)
            dec = C.decode(sd, z)
            r['A'] = C.wvar(z.numpy(), w)
            r['latent'] = z.shape[1]
        dec = dec.numpy().astype(np.float64)
        if sub.startswith('b_'):
            probe, labels = extra
            if fam == 'topk_sae' or fam == 'plain_ae':
                recon = dec
            else:
                recon = dec  # Gaussian decoder mean = network output
            bits = (recon @ probe['W'].T + probe['b']) > 0
            r['probe_bitwise'], r['probe_exact'] = bits_scores(bits, labels, w)
            ww = w / w.sum()
            xm = ww @ x
            r['r2'] = float(1 - (ww @ ((x - recon) ** 2).sum(1)) / (ww @ ((x - xm) ** 2).sum(1)))
        else:
            if fam == 'topk_sae':
                bits = dec > 0.5
            else:  # Bernoulli logits; mean > 0.5 iff logit > 0
                bits = dec > 0
            r['bitwise'], r['exact'] = bits_scores(bits, x, w)
        res[key] = r
    dev = res['dev']
    if sub.startswith('b_'):
        out['competence'] = bool(dev['probe_bitwise'] >= 0.98 and dev['r2'] >= 0.90)
        out['dev_probe_bitwise'], out['dev_r2'] = dev['probe_bitwise'], dev['r2']
    else:
        out['competence'] = bool(dev['exact'] >= 0.95 and dev['bitwise'] >= 0.99)
        out['dev_exact'], out['dev_bitwise'] = dev['exact'], dev['bitwise']
    ev = res[key_eval]
    if fam in ('gauss_vae', 'tc_vae'):
        A = ev['A']
        out.update(au=int((A > 0.01).sum()), latent=int(ev['latent']), active=[int(i) for i in np.flatnonzero(A > 0.01)],
                   au_robust={str(t): int((A > t).sum()) for t in (0.001, 0.05, 0.1)},
                   active_dev=[int(i) for i in np.flatnonzero(dev['A'] > 0.01)])
        stored = np.load(Path(run_dir) / 'raw' / f'{run}.npz')
        out['mu_max_abs_diff_vs_stored'] = float(np.abs(stored[f'{key_eval}_mu'] - ev['mu']).max())
    elif fam == 'cat_vae':
        out['au_categorical'] = int((ev['activity'] > 0.01).sum())
        out['active'] = [int(i) for i in np.flatnonzero(ev['activity'] > 0.01)]
        out['active_dev'] = [int(i) for i in np.flatnonzero(dev['activity'] > 0.01)]
    elif fam == 'topk_sae':
        out['alive'] = int((ev['fire'] >= 0.005).sum())
        out['active'] = [int(i) for i in np.flatnonzero(ev['fire'] >= 0.005)]
    else:
        out['au_covariance'] = int((ev['A'] > 0.01).sum())
        out['active'] = [int(i) for i in np.flatnonzero(ev['A'] > 0.01)]
    return out


def converged(cell, log, mine):
    """pw02a PROTOCOL section 1 item 4, re-implemented."""
    if cell['substrate'].startswith('b_') and cell.get('gamma') == 'learned':
        snap = log['snapshot']
        return bool(snap['competence'] and mine['competence'] and snap['active_dev'] == mine['active_dev'])
    curve = log['log']['curve']
    last = curve[-1]
    ref = min(curve, key=lambda r: abs(r['step'] - 0.8 * last['step']))
    a, b = ref['dev_objective'], last['dev_objective']
    ok = abs(b - a) / abs(a) < 0.02
    if cell['family'] in ('plain_ae', 'topk_sae'):
        ok = ok or abs(b - a) < 1e-3
    return bool(ok), dict(step_ref=ref['step'], obj_ref=a, obj_final=b, rel=abs(b - a) / abs(a))


def median(v):
    return float(np.median(v)) if v else None


def verdicts(rows):
    by = {}
    for r in rows:
        by.setdefault(r['cell_id'], []).append(r)
    g = lambda cid: [r['au'] for r in by[cid] if r['competence']]  # noqa: E731
    out = {}
    beta1 = [f'bw-gauss_vae-{a}-b1-lr{lr}' for a in ARCHS for lr in LRS]
    part_i_runs = [r for c in beta1 for r in by[c] if r['competence']]
    part_i = bool(part_i_runs) and all(r['au'] < 25 and r['au'] < r['latent'] for r in part_i_runs)
    per_arch = {}
    for a in ARCHS:
        A_bw, A_code, A_prod = (median(g(f'{d}-gauss_vae-{a}-b1-lr0.0003')) for d in ('bw', 'code10', 'prod5'))
        acc = []
        if abs(A_bw - A_code) <= 2 and A_prod > A_bw + 2:
            acc.append('information')
        if abs(A_bw - A_prod) <= 2 and A_code > A_bw + 2:
            acc.append('factor')
        if abs(A_bw - A_prod) <= 2 and A_code < A_bw - 2:
            acc.append('affine')
        per_arch[a] = dict(A_bw=A_bw, A_code10=A_code, A_prod5=A_prod, accounts=acc,
                           seeds_bw=g(f'bw-gauss_vae-{a}-b1-lr0.0003'), seeds_code10=g(f'code10-gauss_vae-{a}-b1-lr0.0003'),
                           seeds_prod5=g(f'prod5-gauss_vae-{a}-b1-lr0.0003'))
    n_info = sum('information' in v['accounts'] for v in per_arch.values())
    out['H2.2'] = dict(part_i=part_i, part_i_n=len(part_i_runs), part_i_max=max(r['au'] for r in part_i_runs),
                       per_arch=per_arch, n_information=n_info, verdict=bool(part_i and n_info >= 3))
    cells = {c: dict(values=g(c), median=median(g(c)), range=max(g(c)) - min(g(c))) for c in beta1}
    between = max(v['median'] for v in cells.values()) - min(v['median'] for v in cells.values())
    within = max(v['range'] for v in cells.values())
    out['H2.3'] = dict(cells=cells, between=between, max_within=within, verdict=bool(between >= 3 and between > within))
    mono = {}
    for a in ARCHS:
        for lr in LRS:
            m = [median([r['au'] for r in by[f'bw-gauss_vae-{a}-b{b}-lr{lr}']]) for b in ('0.25', '1', '4')]
            mono[f'{a}-lr{lr}'] = dict(medians=m, nonincreasing=bool(m[0] >= m[1] >= m[2]))
    out['H2.3_secondary_beta_monotone'] = dict(cells=mono, n=sum(v['nonincreasing'] for v in mono.values()),
                                               verdict=sum(v['nonincreasing'] for v in mono.values()) >= 6)
    per_src = {}
    for s in SOURCES:
        L = g(f'b_lin_{s}-gauss_vae-A2-b1-glearned-lr0.0003')
        F = [r['au'] for r in by[f'b_lin_{s}-gauss_vae-A2-b1-gfixed-lr0.0003']]  # contrast: regardless of competence
        per_src[s] = dict(learned=L, fixed=F, med_learned=median(L), med_fixed=median(F),
                          ok=bool(median(L) - median(F) >= 2 and min(L) > max(F)))
    out['H2.4'] = dict(per_source=per_src, verdict=all(v['ok'] for v in per_src.values()))
    req = beta1 + [f'{d}-gauss_vae-{a}-b1-lr0.0003' for d in ('code10', 'prod5') for a in ARCHS] + \
        [f'b_lin_{s}-gauss_vae-A2-b1-glearned-lr0.0003' for s in SOURCES]
    comp = {c: dict(planned=len(by[c]), passing=sum(r['competence'] for r in by[c])) for c in req}
    out['experiment_competence'] = dict(cells=comp, verdict=all(v['passing'] >= v['planned'] // 2 + 1 for v in comp.values()))
    return out


def main():
    legal, w, ctx_full = raw_weights()
    out = {}
    for run_id, key_eval in (('pilot_001', 'dev'), ('full_001', 'conf')):
        run_dir = C.PW02 / run_id
        grid = json.loads((run_dir / 'sweep_grid_used.json').read_text())
        cells = {c['cell_id']: c for c in grid['cells']}
        ctx = np.load(run_dir / 'raw/data_context.npz')
        cache = {}

        def data(sub):
            if sub in cache:
                return cache[sub]
            if sub == 'bw':
                d = {'dev': (legal, w['dev'], None)}
                if key_eval == 'conf':
                    d['conf'] = (legal, w['conf'], None)
            elif sub in ('code10', 'prod5'):
                S = ctx[f'{sub}_states']
                u = np.full(len(S), 1 / len(S))
                d = {'dev': (S, u, None), 'conf': (S, u, None)}
            elif sub == 'indep':
                d = None
            else:
                _, kind, src = sub.split('_', 2)
                a = np.load(run_dir / f'raw/activations_{src}.npz')
                p = np.load(run_dir / f'raw/probes_{src}.npz')
                probe = dict(W=p['cur_W'], b=p['cur_b'])
                fld = 'acts' if kind == 'raw' else 'lin'
                d = {'dev': (a[f'val_{fld}'].astype(np.float64), np.full(len(a[f'val_{fld}']), 1.0), (probe, a['val_ycur']))}
                if key_eval == 'conf':
                    d['conf'] = (a[f'conf_{fld}'].astype(np.float64), np.full(len(a[f'conf_{fld}']), 1.0),
                                 (probe, a['conf_ycur']))
            cache[sub] = d
            return d

        index = json.loads((run_dir / 'runs_index.json').read_text())
        table = {r['run']: r for r in json.loads((run_dir / 'au_table.json').read_text())}
        per_run = json.loads((run_dir / 'au_per_run.json').read_text())
        rows, mism = [], []
        for rec in index['runs']:
            cell = cells[rec['cell_id']]
            if cell['substrate'] == 'indep':
                continue  # descriptive negative control; eval sample is generator-defined, not re-derived
            log = json.loads((run_dir / 'runs' / f'{rec["run"]}.json').read_text())
            m = eval_run(run_dir, rec['run'], cell, data, key_eval)
            conv = converged(cell, log, m)
            m['converged'], m['convergence_detail'] = (conv, None) if isinstance(conv, bool) else conv
            t = table[rec['run']]
            for fld in ('au', 'competence', 'alive', 'au_categorical', 'au_covariance', 'converged'):
                if fld in m and fld in t and m[fld] != t[fld]:
                    mism.append(dict(run=rec['run'], field=fld, mine=m[fld], recorded=t[fld]))
            if 'au_robust' in m and m['au_robust'] != t.get('au_robust'):
                mism.append(dict(run=rec['run'], field='au_robust', mine=m['au_robust'], recorded=t.get('au_robust')))
            if 'active' in m and rec['run'] in per_run and m['active'] != per_run[rec['run']]:
                mism.append(dict(run=rec['run'], field='active_set', mine=m['active'], recorded=per_run[rec['run']]))
            if 'active_dev' in m and 'active_dev_final' in log and m['active_dev'] != log['active_dev_final']:
                mism.append(dict(run=rec['run'], field='active_dev_final', mine=m['active_dev'],
                                 recorded=log['active_dev_final']))
            rows.append(m)
        seeds = [r['run_seed_mine'] for r in rows]
        res = dict(n_runs_recomputed=len(rows), mismatches=mism,
                   run_seed_logged_equals_derivation=all(r['run_seed_logged'] == r['run_seed_mine'] for r in rows),
                   run_seeds_unique=len(set(seeds)) == len(seeds),
                   max_mu_abs_diff_vs_stored=max(r.get('mu_max_abs_diff_vs_stored', 0.0) for r in rows),
                   latent_sizes=sorted({(r['substrate'][:2] == 'b_', r.get('latent')) for r in rows if r.get('latent')}),
                   not_converged=[r['run'] for r in rows if r['converged'] is False],
                   rows=rows)
        if run_id == 'full_001':
            res['verdicts'] = verdicts(rows)
            keep = [r for r in rows if r['converged'] is not False]
            res['verdicts_excluding_not_converged'] = verdicts(keep)
        out[run_id] = res
    out['seed_overlap_pilot_full'] = sorted(set(r['run_seed_mine'] for r in out['pilot_001']['rows'])
                                            & set(r['run_seed_mine'] for r in out['full_001']['rows']))
    (C.OUT / 'pw02a_recompute.json').write_text(json.dumps(C.jsonable(out), indent=1))
    for k in ('pilot_001', 'full_001'):
        o = out[k]
        print(k, dict(n=o['n_runs_recomputed'], mismatches=len(o['mismatches']), seeds_ok=o['run_seed_logged_equals_derivation'],
                      unique=o['run_seeds_unique'], mu_diff=o['max_mu_abs_diff_vs_stored'], latents=o['latent_sizes'],
                      not_conv=len(o['not_converged'])))
    print(json.dumps(C.jsonable(out['full_001']['mismatches'][:20]), indent=0))
    v = out['full_001']['verdicts']
    print({k: v[k]['verdict'] for k in v})
    print(json.dumps(C.jsonable(v), indent=0)[:4000])
    print('excl not converged', {k: out['full_001']['verdicts_excluding_not_converged'][k]['verdict']
                                 for k in out['full_001']['verdicts_excluding_not_converged']})


if __name__ == '__main__':
    main()
