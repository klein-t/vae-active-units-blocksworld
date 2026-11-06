"""Independent audit 001: data conventions (state pairing, splits, weights, legal set, controls).

- Legal set re-derived by BFS over my STRIPS operators; compared with pw01's table and pw02a's data_context.
- Every transition of every thesis sequence (train + test file) checked: Y[k] == apply(X[k], Y[k-1]),
  i.e. Y[k] is the state AFTER the action token at position k (current state for a causal model at k).
- Train/val split re-derived from the protocol text; frequency weights per split recomputed from Y at
  X != PAD and compared with the stored w_train / w_val / w_test.
- code10 / prod5 control tables checked structurally (support size, block structure).
Writes data_check.json.
"""
import json
import sys

import numpy as np

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):  # usage without the data present
    print(__doc__)
    sys.exit(0)

import ind_common as C  # noqa: E402


def main():
    out = {}
    L = C.bfs_legal()
    out['bfs_legal_count'] = len(L)
    pw01 = np.load(C.ROOT / 'experiments/pw01_state_space_ground_truth/results/analysis_001/legal_states_866.npy')
    out['pw01_table_equals_bfs_set'] = {r.tobytes() for r in pw01.astype(np.int8)} == set(L)
    res = {}
    for run in ('pilot_001', 'full_001'):
        ctx = np.load(C.PW02 / run / 'raw/data_context.npz')
        legal = ctx['legal'].astype(np.int8)
        row_of = {r.tobytes(): i for i, r in enumerate(legal)}
        res[run] = dict(ctx=ctx, legal=legal, row_of=row_of,
                        legal_equals_bfs={r.tobytes() for r in legal} == set(L))
    d = C.load_thesis_data()
    # transitions
    trans = {}
    for name, X, Y in (('train_file', d['Xtr'], d['Ytr']), ('test_file', d['Xte'], d['Yte'])):
        n_ok = n_bad = 0
        sos_ok = bool((X[:, 0] == C.VOCAB['[SOS]']).all())
        pad_rows_zero = bool((Y[X == C.PAD] == 0).all())
        y0_legal = bool(C.is_legal_rows(Y[:, 0]).all())
        for i in range(len(X)):
            for k in range(1, X.shape[1]):
                if X[i, k] == C.PAD:
                    break
                t = C.apply_action(Y[i, k - 1].astype(np.int8), C.INV_VOCAB[int(X[i, k])])
                if t is not None and np.array_equal(t, Y[i, k]):
                    n_ok += 1
                else:
                    n_bad += 1
        trans[name] = dict(transitions_checked=n_ok + n_bad, mismatches=n_bad, sos_token_constant=sos_ok,
                           y_zero_at_pad=pad_rows_zero, y0_legal=y0_legal,
                           valid_positions=int((X != C.PAD).sum()))
    out['transitions_Yk_is_state_after_token_k'] = trans
    # split + weights
    tr, va = C.split_indices()
    out['split'] = dict(n_train=len(tr), n_val=len(va), disjoint=len(np.intersect1d(tr, va)) == 0,
                        union_is_all=len(np.union1d(tr, va)) == 16004)
    for run, r in res.items():
        ctx, row_of = r['ctx'], r['row_of']
        cmp = {}
        for key, X, Y in (('w_train', d['Xtr'][tr], d['Ytr'][tr]), ('w_val', d['Xtr'][va], d['Ytr'][va]),
                          ('w_test', d['Xte'], d['Yte'])):
            if key not in ctx.files:
                cmp[key] = 'absent (sealed)'
                continue
            c = C.state_counts(X, Y, row_of)
            cmp[key] = dict(max_abs_diff=float(np.abs(c / c.sum() - ctx[key]).max()), positions=int(c.sum()),
                            zero_weight_states=int((c == 0).sum()))
        c = C.state_counts(d['Xtr'][tr], d['Ytr'][tr], row_of)
        cmp['n_train_positions_matches'] = int(c.sum()) == int(ctx['n_train_positions'])
        cmp['legal_equals_bfs'] = r['legal_equals_bfs']
        # controls
        c10 = ctx['code10_states'].astype(np.int8)
        groups = {}
        for j in range(36):
            groups.setdefault(c10[:, j].tobytes(), []).append(j)
        p5 = ctx['prod5_states'].astype(np.int8)
        derived = [i for fct, i in C.FIDX.items() if fct.startswith('clear') or fct == 'hand_empty']
        blocks_ok = True
        for x in C.BLOCKS:
            cols = [C.f(f'on_table({x})'), C.f(f'holding({x})')] + [C.f(f'on({x},{y})') for y in C.BLOCKS if y != x]
            blocks_ok &= bool((p5[:, cols].sum(1) == 1).all())
        cmp['code10'] = dict(distinct=len({r.tobytes() for r in c10}), n_column_groups=len(groups),
                             group_sizes=sorted(len(v) for v in groups.values()))
        cmp['prod5'] = dict(distinct=len({r.tobytes() for r in p5}), one_place_per_block=blocks_ok,
                            derived_cols_zero=bool((p5[:, derived] == 0).all()))
        out[f'weights_{run}'] = cmp
    (C.OUT / 'data_check.json').write_text(json.dumps(C.jsonable(out), indent=1))
    print(json.dumps(C.jsonable(out), indent=1))


if __name__ == '__main__':
    main()
