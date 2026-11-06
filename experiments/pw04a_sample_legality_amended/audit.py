"""pw04a audit (A3 removed): recompute legality rates and the required criteria from retained sample arrays.

Usage: uv run python experiments/pw04a_sample_legality_amended/audit.py --run-id ID --pw02-run PW02_RUN_ID
Legality recounted by Python-set membership of the unpacked bit rows in the 866 legal states
(no LegalIndex, no checker); B2 recomputed independently as the sum over the 866 legal states of
the product of block marginals; H4.1 and the H4.2 point estimate re-derived. Negative controls:
uniform random bits and column-permuted legal states. Same author and session: not independent
replication; the H4.2 bootstrap interval is not re-derived.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pwlib.runio import dump_json, write_audit, write_manifest  # noqa: E402

EXP = Path(__file__).resolve().parent
PW02 = ROOT / 'experiments' / 'pw02a_active_units_amended'
BLOCKS = 'ABCDE'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--pw02-run', required=True)
    args = ap.parse_args()
    run_dir = EXP / 'results' / args.run_id
    res = json.loads((run_dir / 'results.json').read_text())
    pw02 = PW02 / 'results' / args.pw02_run
    ctx = np.load(pw02 / 'raw' / 'data_context.npz')
    legal = ctx['legal'].astype(np.int8)
    table = {r.tobytes() for r in legal}
    per = res['metrics']['per_model']
    mism, rates = [], {}
    for name, m in per.items():
        if 'prior_mode' not in m:
            continue
        z = np.load(run_dir / 'raw' / f'samples_{name}.npz')
        bits = np.unpackbits(z['prior_mode_bits'], axis=1)[:, :36].astype(np.int8)
        rate = float(np.mean([r.tobytes() in table for r in bits]))
        rates[name] = rate
        if abs(rate - m['prior_mode']['rate']) > 1e-12:
            mism.append(dict(model=name, analysis=m['prior_mode']['rate'], audit=rate))
    # B2 independently: each legal state is one block-position combination
    names = [f'on_table({x})' for x in BLOCKS] + [f'on({x},{y})' for x in BLOCKS for y in BLOCKS if x != y] + \
        [f'clear({x})' for x in BLOCKS] + [f'holding({x})' for x in BLOCKS] + ['hand_empty']
    col = {n: i for i, n in enumerate(names)}
    w = ctx['w_train'] / ctx['w_train'].sum()
    pos_cols = {x: [col[f'on_table({x})'], col[f'holding({x})']] + [col[f'on({x},{y})'] for y in BLOCKS if y != x]
                for x in BLOCKS}
    q = {x: (w[:, None] * legal[:, pos_cols[x]]).sum(0) for x in BLOCKS}
    b2 = float(sum(np.prod([q[x][int(np.argmax(s[pos_cols[x]]))] for x in BLOCKS]) for s in legal))
    b2_ok = abs(b2 - res['metrics']['baselines']['exact']['B2']) < 1e-12
    # criteria
    crit = {}
    if res['status'] == 'completed' and json.loads((run_dir / 'RUN.json').read_text())['kind'] != 'pilot':
        beta1 = {f'bw-gauss_vae-{a}-b1-lr{lr}' for a in ('A1', 'A2', 'A4') for lr in ('0.0003', '0.003')}
        h41 = [rates[n] for n, m in per.items() if m.get('cell_id') in beta1 and m.get('competence')]
        crit['h41'] = bool(h41) and min(h41) > b2
        rows = res['metrics']['h42_rows']
        by = {}
        for r in rows:
            by.setdefault(r['beta'], []).append((r['au'], r['legal']))
        num = den = 0
        for pts in by.values():
            if len(pts) >= 8:
                rho = spearmanr(*zip(*pts))[0]
                if np.isfinite(rho):
                    num += rho * len(pts)
                    den += len(pts)
        rho = num / den if den else None
        crit['h42_point'] = rho
        if crit['h41'] != res['criteria']['h41_prior_legality_above_product_baseline']:
            mism.append(dict(criterion='h41'))
        obs = res['metrics']['h42']['pooled_rho']
        if (rho is None) != (obs is None) or (rho is not None and abs(rho - obs) > 1e-9):
            mism.append(dict(criterion='h42_point', analysis=obs, audit=rho))
    rng = np.random.default_rng(9)
    neg_random = sum(r.tobytes() in table for r in rng.integers(0, 2, (100_000, 36)).astype(np.int8)) == 0
    shuffled = legal[:, rng.permutation(36)]
    neg_perm = float(np.mean([r.tobytes() in table for r in shuffled])) < 0.05
    checks = dict(rates_match=not any('model' in m for m in mism), b2_match=bool(b2_ok),
                  criteria_match=not any('criterion' in m for m in mism), negative_random=bool(neg_random),
                  negative_column_permuted=bool(neg_perm))
    dump_json(dict(rates=rates, b2=b2, criteria=crit, mismatches=mism, checks=checks), run_dir / 'audit_recompute.json')
    write_manifest(run_dir)
    ok = all(checks.values())
    write_audit(run_dir, status='pass' if ok else 'fail',
                method='Separate script: set-membership recount of mode-decoded prior legality per model, B2 by summation '
                       'over legal states, H4.1 and the H4.2 point estimate re-derived; negative controls (uniform random '
                       'bits; column-permuted legal states).',
                limitations='Same author and session (not independent replication); bootstrap interval, violation '
                            'taxonomy and activation-substrate rates not recomputed.',
                evidence=['audit_recompute.json', 'results.json'], checks=checks,
                reason=None if res['status'] == 'completed' else 'Stopped run: audit of available arrays only.')
    print(dict(audit='pass' if ok else 'fail', checks=checks))


if __name__ == '__main__':
    main()
