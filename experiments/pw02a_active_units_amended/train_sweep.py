"""pw02a entry point: the amended shared training sweep for pw02a-pw04a (A3 removed, 60k steps, KL warm-up).

Usage (after `uv run python src/pwlib/runio.py start ... --run-id ID`, and after extract_activations.py):
  uv run python -u experiments/pw02a_active_units_amended/train_sweep.py --run-id ID --workers 8

Pilot vs full is read from RUN.json (pilot: pilot_cells x seed 101, development data only; the
thesis test file is not opened). Steps: (0) task gate and data context; (1) every (cell, seed) on
CPU, one thread per worker, longest first; (2) runs_index.json. AU analysis is a separate step
(analyze_active_units.py). Smoke flags exist only for code checks on a scratch copy and are
recorded in every output.
"""
import argparse
import fnmatch
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pwlib import blocksworld as bw  # noqa: E402
from pwlib import data as D  # noqa: E402
from pwlib import sweep as S  # noqa: E402
from pwlib.runio import dump_json, load_json, read_run, sha256  # noqa: E402

EXP = Path(__file__).resolve().parent
PW01_KEYS = ('bits_log2_legal', 'bits_min_binary_code', 'bits_entropy_empirical', 'factors_per_block_categorical',
             'affine_dim_legal_states', 'binary_facts_total', 'i3d_hypercube_range')
N_CELLS = 58


def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def task_gate(run_dir, pilot, hashes):
    """Data loading, splits, pw01 reproduction and control definitions; writes raw/data_context.npz."""
    gate = {}
    seqs = D.load_sequences(hashes, include_test=not pilot)
    legal = D.load_legal_table(ROOT, hashes)
    index = D.LegalIndex(legal)
    weights, counts = D.state_weights(seqs, index)
    tr, va = D.split_train_file()
    gate['splits_disjoint'] = bool(len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == D.N_TRAIN_FILE)
    gate['sos_at_position_0'] = bool(all((X[:, 0] == 50).all() for X, _ in seqs.values()))
    gate['train_distinct_states'] = int((weights['train'] > 0).sum())
    gate['all_train_states_legal'] = bool(all(bw.is_legal(v) for v in legal[weights['train'] > 0]))
    gate['n_train_866'] = gate['train_distinct_states'] == 866
    code10, bit_of_pos = D.code10_states()
    prod5 = D.prod5_states()
    gate['code10'] = dict(states=int(len(np.unique(code10, axis=0))), bits=float(np.log2(len(np.unique(code10, axis=0)))))
    gate['prod5'] = dict(states=int(len(np.unique(prod5, axis=0))), bits=float(np.log2(len(np.unique(prod5, axis=0)))))
    gate['controls_ok'] = bool(gate['code10']['states'] == 1024 and gate['prod5']['states'] == 7776)
    cand_path = ROOT / D.PW01_RUN / 'dimension_candidates.json'
    D.check_hash(cand_path, hashes[str(D.PW01_RUN / 'dimension_candidates.json')])
    cand = load_json(cand_path)['candidates']
    targets = {k: cand[k]['value'] for k in PW01_KEYS}  # KeyError = fail loudly
    gate['pw01_targets'] = targets
    centered = legal - legal.mean(0)
    gate['pca_count_866'] = int(np.linalg.matrix_rank(centered.astype(np.float64)))
    gate['numerical_pca_25'] = gate['pca_count_866'] == targets['affine_dim_legal_states'] == 25
    gate['position_counts'] = counts
    gate['entropy_bits'] = {k: entropy_bits(w) for k, w in weights.items()}
    gate['pass'] = bool(gate['splits_disjoint'] and gate['sos_at_position_0'] and gate['n_train_866']
                        and gate['all_train_states_legal'] and gate['controls_ok'] and gate['numerical_pca_25'])
    marg = (weights['train'][:, None] * legal).sum(0)
    ctx = dict(legal=legal, w_train=weights['train'], w_val=weights['val'], marginals_train=marg,
               n_train_positions=np.array(counts['train']), code10_states=code10, code10_bit_of_pos=bit_of_pos,
               prod5_states=prod5)
    if not pilot:
        ctx['w_test'] = weights['test']
    np.savez(run_dir / 'raw' / 'data_context.npz', **ctx)
    return gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--smoke-steps', type=int, default=None, help='SMOKE ONLY: override training steps')
    ap.add_argument('--smoke-cells', default=None, help='SMOKE ONLY: fnmatch pattern on cell ids')
    args = ap.parse_args()
    run_dir = EXP / 'results' / args.run_id
    run = read_run(run_dir)
    pilot = run['kind'] == 'pilot'
    design = load_json(EXP / 'DESIGN.json')
    grid_path = EXP / 'sweep_grid.json'
    grid = S.load_grid(grid_path)
    if len(grid['cells']) != N_CELLS:
        raise ValueError('grid changed: amend, do not edit')
    for d in ('models', 'raw', 'runs'):
        (run_dir / d).mkdir(exist_ok=True)
    dump_json(grid, run_dir / 'sweep_grid_used.json')
    t0 = time.time()
    gate = task_gate(run_dir, pilot, design['external_inputs'])
    dump_json(gate, run_dir / 'task_gate.json')
    if not gate['pass']:
        raise SystemExit(f'Task gate failed: see {run_dir / "task_gate.json"}')
    runs = S.planned_runs(grid, pilot)
    if args.smoke_cells:
        runs = [(c, s) for c, s in runs if fnmatch.fnmatch(c['cell_id'], args.smoke_cells)]
    src_gate = load_json(run_dir / 'source_gate.json') if (run_dir / 'source_gate.json').exists() else None
    skipped, tasks = [], []
    for cell, seed in runs:
        kind, src = S.source_of(cell['substrate'])
        if src is not None and (src_gate is None or not src_gate['sources'][src]['gate']['pass']):
            skipped.append(dict(run=S.run_name(cell['cell_id'], seed), reason=f'source {src} gate missing or failed'))
            continue
        if (run_dir / 'runs' / f'{S.run_name(cell["cell_id"], seed)}.json').exists():
            raise FileExistsError(f'Run already recorded: {cell["cell_id"]} s{seed} (no silent re-runs)')
        tasks.append((cell, seed, str(run_dir), pilot, args.smoke_steps))
    tasks.sort(key=lambda t: -S.cost_rank(t[0]))
    print(f'{len(tasks)} runs, {len(skipped)} skipped, {args.workers} workers', flush=True)
    with get_context('spawn').Pool(args.workers) as pool:
        results = []
        for r in pool.imap_unordered(S.execute_run, tasks):
            results.append(r)
            print(f'[{len(results)}/{len(tasks)}] {r["run"]} {r["status"]} competence={r["competence"]} '
                  f'{r.get("wall_seconds", 0):.0f}s', flush=True)
    results.sort(key=lambda r: r['run'])
    dump_json(dict(pilot=pilot, smoke=dict(steps=args.smoke_steps, cells=args.smoke_cells),
                   workers=args.workers, wall_seconds=time.time() - t0,
                   cpu_seconds_training=sum(r.get('wall_seconds', 0) for r in results),
                   grid_sha256=sha256(grid_path), runs=results, skipped=skipped), run_dir / 'runs_index.json')


if __name__ == '__main__':
    main()
