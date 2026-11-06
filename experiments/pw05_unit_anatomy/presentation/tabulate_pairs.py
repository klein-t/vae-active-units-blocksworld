"""pw05 presentation only (no new quantity; not frozen; written after analysis_001 on request):
tabulate WHICH two facts each 2-fact unit detects (names, signs) and the pair types per cell, from the per-unit fact
lists stored in results/analysis_001/results.json. Cross-checks the Blocksworld fact lists against the pw03a f1_conf
matrices directly. Usage: uv run python experiments/pw05_unit_anatomy/presentation/tabulate_pairs.py OUT.json
"""
import json, collections, sys
import numpy as np
from pathlib import Path
if {"-h", "--help"} & set(sys.argv[1:]):
    sys.exit(print(__doc__))
ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, ROOT + '/src')
from pwlib import blocksworld as bw
r = json.load(open(f'{ROOT}/experiments/pw05_unit_anatomy/results/analysis_001/results.json'))['metrics']
au = json.load(open(f'{ROOT}/experiments/pw02a_active_units_amended/results/full_001/au_per_run.json'))
def args(f): return f[f.index('(')+1:-1].split(',')
def ptype(a, b):
    pa, pb = a.split('(')[0], b.split('(')[0]
    if pa == pb == 'on':
        (x1, y1), (x2, y2) = args(a), args(b)
        if (x1, y1) == (y2, x2): return 'on(X,Y)|on(Y,X)'
        if x1 == x2: return 'on(X,Y)|on(X,Z)'
        if y1 == y2: return 'on(X,Y)|on(Z,Y)'
        return 'on|on other'
    key = tuple(sorted([pa, pb]))
    if key == ('holding', 'on'):
        h = args(a if pa == 'holding' else b)[0]; xy = args(b if pa == 'holding' else a)
        return 'on(X,Y)|holding(X)' if h == xy[0] else 'on(X,Y)|holding(Y)' if h == xy[1] else 'on(X,Y)|holding(Z)'
    if key == ('on', 'on_table'):
        t = args(a if pa == 'on_table' else b)[0]; xy = args(b if pa == 'on_table' else a)
        return 'on(X,Y)|on_table(X)' if t == xy[0] else 'on(X,Y)|on_table(Y)' if t == xy[1] else 'on(X,Y)|on_table(Z)'
    return f'{key[0]}|{key[1]}'
cells = collections.defaultdict(lambda: collections.Counter()); signs = collections.defaultdict(collections.Counter)
names = collections.defaultdict(list); mism = 0
for o in r['per_run']:
    if 'q3' not in o: continue
    if o['substrate'] == 'bw':  # cross-check fact lists against pw03a stored F1 directly
        f1 = np.load(f'{ROOT}/experiments/pw03a_fact_alignment_amended/results/full_001/raw/scores_{o["name"]}.npz')['f1_conf']
        for u in o['q3']['units']:
            mism += [bw.FACTS[i] for i in np.flatnonzero(f1[u['unit']] >= 0.9)] != u['facts']
    for u in o['q3']['units']:
        if u['n_facts'] == 2:
            t = ptype(*u['facts']); cells[o['cell_id']][t] += 1
            signs[o['cell_id']][t] += u['signs'][0] != u['signs'][1]
            names[o['name']].append('/'.join(f'{f}{"+" if s > 0 else "-"}' for f, s in zip(u['facts'], u['signs'])))
print('fact-list mismatches vs pw03a:', mism)
for c, cnt in cells.items():
    print(c, sum(cnt.values()), dict(cnt.most_common()), 'opposite:', dict(signs[c]))
pairs_per_run = {}
for o in r['per_run']:
    if 'q3' in o:
        ps = {frozenset(args(u['facts'][0])) for u in o['q3']['units'] if u['n_facts'] == 2 and ptype(*u['facts']) == 'on(X,Y)|on(Y,X)'}
        pairs_per_run[o['name']] = (o['n_active'], sum(ptype(*u['facts']) == 'on(X,Y)|on(Y,X)' for u in o['q3']['units'] if u['n_facts'] == 2), len(ps))
for k, v in pairs_per_run.items(): print(k, v)
json.dump(dict(pair_types_per_cell={c: dict(v) for c, v in cells.items()}, named_pairs_per_run=names,
               reversed_units_per_run=pairs_per_run), open(sys.argv[1], 'w'), indent=1)
