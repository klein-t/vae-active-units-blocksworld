"""pw01 audit: recompute the headline numbers with separately written code.

Uses a generative enumeration (towers as ordered lists, no constraint checker), the thesis arrays and the
retained run files. Writes audit_recompute.json, then MANIFEST.json and AUDIT.json for the run.
Usage: uv run python experiments/pw01_state_space_ground_truth/audit.py --run-id ID
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from itertools import permutations
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True  # importing the retained checker copy must not add files to the run dir

EXP = Path(__file__).resolve().parent
DATA = Path(os.environ.get('PW_BLOCKSWORLD_DATA', Path(__file__).resolve().parents[2] / 'data' / 'blocksworld'))
CONTROL = {'RUN.json', 'MANIFEST.json', 'AUDIT.json', 'CLAIM_UPDATES.json'}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def towers_partitions(items):
    """All ways to arrange `items` into a set of towers (each tower a bottom-to-top tuple)."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    # The tower containing `first`: choose its other members and their order.
    for mask in range(1 << len(rest)):
        members = [first] + [rest[i] for i in range(len(rest)) if mask >> i & 1]
        others = [rest[i] for i in range(len(rest)) if not mask >> i & 1]
        for tower in permutations(members):
            for tail in towers_partitions(others):
                yield [tower] + tail


def encode(towers, held, names):
    fact = {n: 0 for n in names}
    for t in towers:
        fact[f'on_table({t[0]})'] = 1
        for lower, upper in zip(t, t[1:]):
            fact[f'on({upper},{lower})'] = 1
        fact[f'clear({t[-1]})'] = 1
    if held is None:
        fact['hand_empty'] = 1
    else:
        fact[f'holding({held})'] = 1
    return tuple(fact[n] for n in names)


def H(c):
    p = np.asarray(c, float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    run_id = ap.parse_args().run_id
    out = EXP / 'results' / run_id
    meta = json.loads((DATA / 'blocksworld_metadata.json').read_text())
    names = sorted(meta['probe_indices'], key=meta['probe_indices'].get)
    blocks = meta['blocks']

    gen = set()
    for t in towers_partitions(blocks):
        gen.add(encode(t, None, names))
    n_empty = len(gen)
    for b in blocks:
        for t in towers_partitions([x for x in blocks if x != b]):
            gen.add(encode(t, b, names))
    L = np.load(out / 'legal_states_866.npy')
    run_set = {tuple(int(x) for x in v) for v in L}
    a = dict(generative_count=len(gen), generative_arm_empty=n_empty, generative_equals_run_set=gen == run_set)

    # Data recheck with set lookups (not the run's packed-code search).
    illegal = parity_bad = positions = 0
    counts = {}
    for split in ('train', 'test'):
        X = np.load(DATA / f'blocksworld_{split}_X.npy')
        Y = np.load(DATA / f'blocksworld_{split}_Y.npy')
        for i in range(len(X)):
            n = int((X[i] != meta['vocab']['[PAD]']).sum())
            for k in range(n):
                s = tuple(int(x) for x in Y[i, k])
                positions += 1
                illegal += s not in gen
                parity_bad += s[names.index('hand_empty')] != int(k % 2 == 0)
                counts[s] = counts.get(s, 0) + 1
    a.update(positions=positions, illegal=illegal, parity_violations=parity_bad, distinct=len(counts),
             entropy_all=H(list(counts.values())))

    # Recompute from the retained frequency table.
    rows = list(csv.DictReader(open(out / 'state_frequency_table.csv')))
    tab = {tuple(int(c) for c in r['bits']): int(r['count_all']) for r in rows}
    a['table_matches_recount'] = tab == counts
    ev = np.array([int(r['count_even_k']) for r in rows])
    od = np.array([int(r['count_odd_k']) for r in rows])
    pe = ev.sum() / (ev.sum() + od.sum())
    a.update(entropy_even=H(ev), entropy_odd=H(od), entropy_given_parity=pe * H(ev) + (1 - pe) * H(od))
    # Total correlation from the 'positions' strings.
    pos = [dict(p.split(':') for p in r['positions'].split()) for r in rows]
    w = np.array([int(r['count_all']) for r in rows], float)
    marg = []
    for b in blocks:
        m = {}
        for pr, wi in zip(pos, w):
            m[pr[b]] = m.get(pr[b], 0) + wi
        marg.append(H(list(m.values())))
    a.update(sum_marginal_entropies=sum(marg), total_correlation=sum(marg) - H(w))
    Lf = L.astype(float)
    a.update(affine_dim=int(np.linalg.matrix_rank(Lf - Lf.mean(0))), rank=int(np.linalg.matrix_rank(Lf)),
             affine_dim_arm_empty=int(np.linalg.matrix_rank(Lf[L[:, 35] == 1] - Lf[L[:, 35] == 1].mean(0))))

    # Negative controls: column-permuted legal states should almost never be legal under the
    # retained checker copy; a product-of-marginals distribution has zero total correlation.
    spec = importlib.util.spec_from_file_location('bwcopy', out / 'blocksworld_legality.py')
    bwc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bwc)
    rng = np.random.default_rng(1)
    perm_legal = sum(bwc.is_legal(L[i][rng.permutation(36)]) for i in rng.integers(0, len(L), 5000))
    a['negative_control_column_permuted_accepted_of_5000'] = int(perm_legal)
    p = np.ones((6,) * 2) / 36
    a['negative_control_tc_independent_pair'] = float(2 * math.log2(6) - H(p.ravel()))

    res = json.loads((out / 'results.json').read_text())
    m = res['metrics']
    close = lambda x, y: abs(x - y) < 1e-9  # noqa: E731
    checks = dict(
        count=a['generative_count'] == m['n_legal'] == 866 and a['generative_arm_empty'] == m['n_arm_empty'],
        set=a['generative_equals_run_set'],
        data=a['illegal'] == m['n_illegal_positions_in_data'] and a['positions'] == m['task']['valid_positions'],
        coverage=a['distinct'] == m['legal_states_covered'],
        parity=a['parity_violations'] == m['hand_empty_parity_violations'],
        entropy=close(a['entropy_all'], m['entropy_bits']['all']) and close(a['entropy_even'], m['entropy_bits']['even_k'])
        and close(a['entropy_odd'], m['entropy_bits']['odd_k']) and close(a['entropy_given_parity'], m['entropy_bits_given_parity']),
        total_correlation=close(a['total_correlation'], m['factorisation']['empirical_all']['total_correlation_bits']),
        table=a['table_matches_recount'],
        linear=a['affine_dim'] == m['linear']['affine_dim_all'] and a['rank'] == m['linear']['rank_all']
        and a['affine_dim_arm_empty'] == m['linear']['affine_dim_arm_empty'],
        negative_controls=a['negative_control_column_permuted_accepted_of_5000'] < 50
        and close(a['negative_control_tc_independent_pair'], 0.0),
    )
    checks = {k: bool(v) for k, v in checks.items()}
    a['checks'] = checks
    (out / 'audit_recompute.json').write_text(json.dumps(a, indent=2) + '\n')

    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name not in CONTROL)
    rec = [dict(path=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=digest(p)) for p in files]
    (out / 'MANIFEST.json').write_text(json.dumps(dict(files=rec), indent=2) + '\n')
    ev_names = {'results.json', 'audit_recompute.json', 'legal_states_866.npy', 'state_frequency_table.csv'}
    audit = dict(
        run_id=run_id, results_sha256=digest(out / 'results.json'),
        status='pass' if all(checks.values()) else 'fail',
        method=('Separate script: generative enumeration of tower arrangements (no constraint checker), '
                'per-position set lookups over the raw thesis arrays for legality/coverage/parity/entropy, '
                'recomputation of conditional entropies, total correlation and affine dimensions from the retained '
                'frequency table and legal-state array; negative controls: column-permuted legal states under the '
                'retained checker copy, and zero total correlation for an independent pair.'),
        limitations=('Same author and session wrote run and audit (not independent replication). The I3D '
                     'dimension estimates are exploratory and not re-derived here.'),
        checks=checks, evidence=[r for r in rec if r['path'] in ev_names])
    (out / 'AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(status=audit['status'], checks=checks), indent=2))


if __name__ == '__main__':
    main()
