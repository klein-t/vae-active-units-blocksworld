"""pw03a audit (H3.2 three-valued, separately re-derived): recompute the H3.2 headline from retained arrays with separately written code.

Usage: uv run python experiments/pw03a_fact_alignment_amended/audit.py --run-id ID --pw02-run PW02_RUN_ID
(1) For the k=5 SAE and the four Gaussian comparator cells: confirmation F1 of every (active code,
position fact) pair recomputed with sklearn.metrics.f1_score from the stored sign/threshold, Cov30
recounted and H3.2 re-derived. (2) For one SAE run, the whole best-threshold search is redone by
brute force (prediction matrices over every distinct development value) and must reproduce the
stored development F1. Negative control: the fact oracle with label rows permuted must give Cov30 <= 0.2.
Same author and session as the analysis: not independent replication.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pwlib.runio import dump_json, write_audit, write_manifest  # noqa: E402

EXP = Path(__file__).resolve().parent
PW02 = ROOT / 'experiments' / 'pw02a_active_units_amended'
CELLS = ('bw-topk_sae-k5-lr0.0003', 'bw-gauss_vae-A2-b1-lr0.0003', 'bw-gauss_vae-A2-b4-lr0.0003',
         'bw-tc_vae-A2-b2-lr0.0003', 'bw-tc_vae-A2-b6-lr0.0003')
POS = [i for i in range(36) if not (25 <= i <= 29 or i == 35)]  # on_table, on, holding columns


def brute_best_dev_f1(c, y, w):
    """Best weighted F1 over thresholds at every distinct value, both signs (prediction-matrix form)."""
    best = 0.0
    P = np.dot(w, y)
    for s in (1, -1):
        v = s * c
        u = np.unique(v)
        cuts = np.r_[(u[1:] + u[:-1]) / 2, -np.inf]
        pred = v[:, None] > cuts[None, :]
        tp = (w * y) @ pred
        pp = w @ pred
        f1 = np.where(pp + P > 0, 2 * tp / np.maximum(pp + P, 1e-300), 0)
        best = max(best, float(f1.max()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--pw02-run', required=True)
    args = ap.parse_args()
    run_dir = EXP / 'results' / args.run_id
    res = json.loads((run_dir / 'results.json').read_text())
    pw02 = PW02 / 'results' / args.pw02_run
    run_kind = json.loads((run_dir / 'RUN.json').read_text())['kind']
    ctx = np.load(pw02 / 'raw' / 'data_context.npz')
    legal = ctx['legal'].astype(int)
    w_dev, w_conf = (ctx['w_train'], ctx['w_val']) if run_kind == 'pilot' else (ctx['w_val'], ctx['w_test'])
    w_dev, w_conf = w_dev / w_dev.sum(), w_conf / w_conf.sum()
    active_sets = json.loads((pw02 / 'au_per_run.json').read_text())
    per = res['metrics']['per_model']
    mism, cov = [], {}
    for name, m in per.items():
        if m.get('cell_id') not in CELLS:
            continue
        z = np.load(pw02 / 'raw' / f'{name}.npz')
        codes = z['dev_mu'] if 'dev_mu' in z.files else z['dev_codes']
        act = np.array(active_sets[name], dtype=int)
        sc = np.load(run_dir / 'raw' / f'scores_{name}.npz')
        hits = 0
        for f in POS:
            best = 0.0
            for j in act:
                pred = (sc['sign'][j, f] * codes[:, j]) > sc['threshold'][j, f]
                best = max(best, f1_score(legal[:, f], pred, sample_weight=w_conf, zero_division=0.0))
            hits += best >= 0.9
        cov[name] = hits / len(POS)
        if abs(cov[name] - m['cov30']) > 1e-12:
            mism.append(dict(model=name, analysis=m['cov30'], audit=cov[name]))
    # full search reproduced for one SAE run
    search_ok = None
    sae = sorted(n for n in per if per[n].get('cell_id') == 'bw-topk_sae-k5-lr0.0003')
    if sae:
        z = np.load(pw02 / 'raw' / f'{sae[0]}.npz')
        sc = np.load(run_dir / 'raw' / f'scores_{sae[0]}.npz')
        worst = 0.0
        for j in range(z['dev_codes'].shape[1]):
            for f in range(36):
                worst = max(worst, abs(brute_best_dev_f1(z['dev_codes'][:, j].astype(float), legal[:, f], w_dev)
                                       - sc['f1_dev'][j, f]))
        search_ok = worst < 1e-9
    # negative control: permuted labels on the fact oracle
    rng = np.random.default_rng(5)
    oracle = np.c_[legal + 0.1 * rng.normal(size=legal.shape), rng.normal(size=(len(legal), 28))]
    perm = legal[rng.permutation(len(legal))]
    neg_hits = sum(max(brute_best_dev_f1(oracle[:, j], perm[:, f], w_dev) for j in range(oracle.shape[1])) >= 0.9
                   for f in POS)
    neg_ok = neg_hits / len(POS) <= 0.2
    crit = None
    if res['status'] in ('completed', 'inconclusive') and run_kind != 'pilot':
        s = [cov[n] for n in cov if per[n]['cell_id'] == CELLS[0]]
        comp = {c: [cov[n] for n in cov if per[n]['cell_id'] == c] for c in CELLS[1:]}
        if s and (min(s) < 0.8 or any(v and max(v) >= min(s) for v in comp.values())):
            crit = False
        elif not s or any(not v for v in comp.values()):
            crit = None
        else:
            crit = True
        if crit != res['criteria']['h32_topk_sae_covers_position_facts']:
            mism.append(dict(criterion='h32', analysis=res['criteria']['h32_topk_sae_covers_position_facts'], audit=crit))
    checks = dict(cov30_match=not any('model' in m for m in mism), criterion_match=not any('criterion' in m for m in mism),
                  full_search_reproduced=search_ok is not False, negative_permuted_oracle=bool(neg_ok))
    dump_json(dict(cov30_recomputed=cov, mismatches=mism, h32_recomputed=crit, negative_permuted_oracle_cov30=neg_hits / len(POS),
                   checks=checks), run_dir / 'audit_recompute.json')
    write_manifest(run_dir)
    ok = all(checks.values())
    write_audit(run_dir, status='pass' if ok else 'fail',
                method='Separate script: sklearn f1_score recount of Cov30 for the H3.2 cells from stored signs/thresholds, '
                       'brute-force threshold search reproducing stored development F1 for one SAE run, H3.2 re-derived; '
                       'negative control: label-permuted fact oracle.',
                limitations='Same author and session (not independent replication); MIG, DCI and the activation substrate '
                            'are not recomputed.',
                evidence=['audit_recompute.json', 'results.json'], checks=checks,
                reason=None if res['status'] == 'completed' else 'Stopped or inconclusive run: audit of available arrays only.')
    print(dict(audit='pass' if ok else 'fail', checks=checks))


if __name__ == '__main__':
    main()
