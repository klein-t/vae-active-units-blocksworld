"""pw02 audit: recompute the headline numbers from retained operands with separately written code.

Usage: uv run python experiments/pw02_active_units/audit.py --run-id ID
Recomputes, without pwlib.metrics: every Gaussian run's AU (E[mu^2] - E[mu]^2 > 0.01 with explicit
loops over units), every run's competence from stored reconstruction bits / probe bits, and the
required-criterion booleans from the recomputed counts. Negative controls: a constant posterior
mean must give AU = 0; a duplicated-unit code must count each copy. Writes audit_recompute.json,
MANIFEST.json and AUDIT.json. Same author and session as the analysis: not independent replication.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pwlib.runio import dump_json, write_audit, write_manifest  # noqa: E402

EXP = Path(__file__).resolve().parent


def au_count(mu, w):
    w = w / w.sum()
    n = 0
    for j in range(mu.shape[1]):
        x = mu[:, j].astype(np.float64)
        m1 = float(np.dot(w, x))
        m2 = float(np.dot(w, x * x))
        n += (m2 - m1 * m1) > 0.01
    return int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    run_dir = EXP / 'results' / args.run_id
    res = json.loads((run_dir / 'results.json').read_text())
    table = {r['run']: r for r in json.loads((run_dir / 'au_table.json').read_text())}
    key = res['metrics']['eval_set']
    mismatches, n_checked = [], 0
    recomputed = {}
    for run, row in table.items():
        if row['status'] != 'ok':
            continue
        z = np.load(run_dir / 'raw' / f'{run}.npz')
        log = json.loads((run_dir / 'runs' / f'{run}.json').read_text())
        if f'{key}_mu' in z.files:
            au = au_count(z[f'{key}_mu'], z[f'{key}_weights'])
            recomputed[run] = au
            n_checked += 1
            if au != row['au']:
                mismatches.append(dict(run=run, analysis=row['au'], audit=au))
        wd = z['dev_weights'] / z['dev_weights'].sum()
        if 'dev_recon_bits' in z.files:
            data = np.load(run_dir / 'raw' / 'data_context.npz')
            sub = row['substrate']
            if sub == 'indep':
                comp = None  # evaluation sample is regenerated deterministically; checked via the run log only
            else:
                X = data['legal'] if sub == 'bw' else data[f'{sub}_states']
                eq = z['dev_recon_bits'] == X
                comp = bool(np.dot(wd, eq.all(1)) >= 0.95 and np.dot(wd, eq.mean(1)) >= 0.99)
        else:
            eq = z['dev_recon_probe_bits'] == z['dev_labels']
            comp = bool(np.dot(wd, eq.mean(1)) >= 0.98) if log['metrics']['dev']['r2'] >= 0.90 else False
        if comp is not None and comp != row['competence']:
            mismatches.append(dict(run=run, field='competence', analysis=row['competence'], audit=comp))
    # negative controls
    rng = np.random.default_rng(0)
    w = rng.random(866)
    neg_constant = au_count(np.ones((866, 32)) * 3.0, w) == 0
    x = rng.normal(size=(866, 1))
    neg_duplicate = au_count(np.repeat(x, 4, 1), w) == 4
    # criteria recomputed from audit counts (full runs only)
    crit = {}
    if res['status'] == 'completed' and key == 'conf':
        def med(cid, gated=True):
            v = [recomputed[r] for r, row in table.items() if row['cell_id'] == cid and r in recomputed
                 and (row['competence'] or not gated)]
            return (float(np.median(v)), min(v), max(v)) if v else (None, None, None)
        beta1 = [f'bw-gauss_vae-{a}-b1-lr{lr}' for a in ('A1', 'A2', 'A3', 'A4') for lr in ('0.0003', '0.003')]
        part_i = all(recomputed[r] < 25 for r, row in table.items() if row['cell_id'] in beta1 and row['competence'])
        info = 0
        for a in ('A1', 'A2', 'A3', 'A4'):
            b, c, p = (med(f'{d}-gauss_vae-{a}-b1-lr0.0003')[0] for d in ('bw', 'code10', 'prod5'))
            if None not in (b, c, p) and abs(b - c) <= 2 and p > b + 2:
                info += 1
        crit['h22_information_account'] = bool(part_i and info >= 3)
        meds = [med(c)[0] for c in beta1]
        rngs = [med(c)[2] - med(c)[1] for c in beta1 if med(c)[0] is not None]
        crit['h23_architecture_dependence'] = bool(max(meds) - min(meds) >= 3 and max(meds) - min(meds) > max(rngs))
        ok = True
        for s in ('w64', 'w128', 'w128s2', 'w128s3', 'w256'):
            L = med(f'b_lin_{s}-gauss_vae-A2-b1-glearned-lr0.0003')
            F = med(f'b_lin_{s}-gauss_vae-A2-b1-gfixed-lr0.0003', gated=False)
            ok &= L[0] is not None and F[0] is not None and L[0] - F[0] >= 2 and L[1] > F[2]
        crit['h24_learned_gamma_more_units'] = bool(ok)
        for k, v in crit.items():
            if res['criteria'].get(k) != v:
                mismatches.append(dict(criterion=k, analysis=res['criteria'].get(k), audit=v))
    checks = dict(au_counts_match=not any('analysis' in m and 'field' not in m and 'criterion' not in m for m in mismatches),
                  competence_match=not any(m.get('field') == 'competence' for m in mismatches),
                  criteria_match=not any('criterion' in m for m in mismatches),
                  negative_constant_mu=neg_constant, negative_duplicate_units=neg_duplicate)
    dump_json(dict(n_au_checked=n_checked, mismatches=mismatches, criteria_recomputed=crit, checks=checks),
              run_dir / 'audit_recompute.json')
    write_manifest(run_dir)
    ok = all(checks.values())
    stopped = res['status'] != 'completed'
    write_audit(run_dir, status='pass' if ok else 'fail',
                method='Separate script: explicit-loop weighted AU recount from retained posterior means, competence '
                       'recomputed from stored reconstruction/probe bits, required criteria re-derived from the '
                       'recounted AU; negative controls (constant mean -> 0 units; duplicated unit counted per copy).',
                limitations='Same author and session wrote the run, analysis and audit (not independent replication); '
                            'indep-control competence and TC diagnostics not recomputed.',
                evidence=['audit_recompute.json', 'results.json', 'au_table.json'], checks=checks,
                reason='Stopped run: audit limited to recomputation of available operands.' if stopped else None)
    print(dict(audit='pass' if ok else 'fail', checks=checks, mismatches=len(mismatches)))


if __name__ == '__main__':
    main()
