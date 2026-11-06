"""pw06 training entry point (DESIGN.md): 78 runs (full) or 4 pilot runs (seed 101, development data only).

Usage:
  PYTHONPATH=src .venv/bin/python -u experiments/pw06_encoding_and_latent_size/run_pw06.py --run-id ID --workers 8

RUN.json must exist in results/ID (kind 'pilot' or 'full', freeze_sha256 = SHA-256 of DESIGN.md).
Reuses the pw02a code paths unchanged: pwlib.training.train / evaluate (Bernoulli beta-VAE, plain AE,
linear KL warm-up, Adam, clip 10, 60k steps), pwlib.sweep competence / numerical_flags, CPU, one thread
per worker, spawn pool, longest first. Only the dataset builder is new: each encoding is a row-wise
function of the pw01 866-state table with the pw02a state weights (code10 copies: uniform over 1,024).
The worker mirrors pwlib.sweep.execute_run without the 80% snapshot, so no active-unit set is
computed during training (the pilot must not see counts). Smoke flags are for code checks on a
scratch copy only (--results-root) and are recorded in every output.
"""
import argparse
import fnmatch
import os
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(EXP))
from pwlib import blocksworld as bw  # noqa: E402
from pwlib import data as D  # noqa: E402
from pwlib import sweep as S  # noqa: E402
from pwlib import training as T  # noqa: E402
from pwlib.runio import dump_json, load_json, read_run, sha256  # noqa: E402
import pw06_encodings as E  # noqa: E402

N_RUNS_FULL = 78
# Read-only inputs (same files and hashes as pw02a DESIGN.json external_inputs).
INPUT_HASHES = {
    '$PW_BLOCKSWORLD_DATA/blocksworld_train_X.npy': '6a8d71fd22fde8a25e146619f0f5a09f45dc55cea4483377719cf5dd53a05716',
    '$PW_BLOCKSWORLD_DATA/blocksworld_train_Y.npy': '34421f110850147fd91f12b4795388b235125563d25cbf0a467c756594591093',
    '$PW_BLOCKSWORLD_DATA/blocksworld_test_X.npy': '21a223dcf19ba8add03c7e18ab738541dbf3d4a901e1154d8d040e899cf93845',
    '$PW_BLOCKSWORLD_DATA/blocksworld_test_Y.npy': '54aac4bc4800b08f6acf0bdf7f8d3dcb5e05ba25967105857811c44808401512',
    str(D.PW01_RUN / 'legal_states_866.npy'): 'e1173a49582f5ebccefbfa65d997ee2fa6933fa8d08f01a3e1b216cb9a8a88ca',
}
PW01_FREQ = D.PW01_RUN / 'state_frequency_table.csv'
# DESIGN.md model-free table (train frequencies), checked at the stated rounding.
DESIGN_MODEL_FREE = dict(H_bits=9.345, H_nats=6.478, zero_rate_distortion_e36_nats=15.662, tc_e10_range=(0.39, 0.41))
SUBSTRATES = ('bw', 'e30', 'e10a', 'e10b', 'code10c1', 'code10c2', 'code10c3', 'code10c4')


def pw01_train_weights(legal):
    """State weights from the pw01 frequency table (count_train), row-aligned with the legal table."""
    import csv
    rows = list(csv.DictReader(open(ROOT / PW01_FREQ)))
    if [int(r['state_index']) for r in rows] != list(range(E.N_STATES)):
        raise ValueError('pw01 frequency table rows are not state_index 0..865')
    bits = np.array([[int(ch) for ch in r['bits']] for r in rows], dtype=np.int8)
    if not np.array_equal(bits, legal):
        raise ValueError('pw01 frequency table bits differ from legal_states_866.npy')
    c = np.array([float(r['count_train']) for r in rows])
    return c / c.sum()


def model_free_check(legal):
    mf = E.model_free_table(legal, pw01_train_weights(legal))
    tc = list(mf['tc_e10_nats'].values())
    lo, hi = DESIGN_MODEL_FREE['tc_e10_range']
    mf['matches_design'] = dict(
        H_bits=round(mf['H_bits'], 3) == DESIGN_MODEL_FREE['H_bits'],
        H_nats=round(mf['H_nats'], 3) == DESIGN_MODEL_FREE['H_nats'],
        zero_rate=round(mf['zero_rate_distortion_e36_nats'], 3) == DESIGN_MODEL_FREE['zero_rate_distortion_e36_nats'],
        tc_range=(round(min(tc), 2), round(max(tc), 2)) == (lo, hi))
    mf['pass'] = bool(all(mf['matches_design'].values()))
    return mf


def encoding_checks(legal):
    out = {}
    for name in ('e10a', 'e10b'):
        B = E.encode(name, legal)
        idx = E.from_bits(B)
        out[name] = dict(unique=int(len(np.unique(B, axis=0))) == E.N_STATES,
                         invertible=bool(np.array_equal(idx, E.e10_index(E.E10_SEEDS[name]))),
                         in_range=bool(idx.max() < 1024 and idx.min() >= 0))
    X30 = E.encode('e30', legal)
    out['e30'] = dict(shape_ok=X30.shape == (E.N_STATES, 30), unique=int(len(np.unique(X30, axis=0))) == E.N_STATES,
                      position_facts=[bw.FACTS[i] for i in D.POSITION_COLUMNS] == [
                          f for f in bw.FACTS if f.startswith(('on_table', 'on(', 'holding'))])
    for c in (1, 2, 3, 4):
        Sc, boc = E.code10_copies(c)
        ok = all(np.array_equal(Sc[:, boc == b], np.repeat(Sc[:, [np.flatnonzero(boc == b)[0]]], c, 1))
                 for b in range(10)) and all((boc == b).sum() == c for b in range(10))
        out[f'code10c{c}'] = dict(states=int(len(np.unique(Sc, axis=0))), copies_ok=bool(ok),
                                  zero_cols=int((boc < 0).sum()), zero_cols_constant=bool(not Sc[:, boc < 0].any()))
    out['pass'] = bool(all(v['unique'] and v['invertible'] and v['in_range'] for k, v in out.items() if k.startswith('e10'))
                       and all(out['e30'].values())
                       and all(out[f'code10c{c}']['states'] == 1024 and out[f'code10c{c}']['copies_ok']
                               and out[f'code10c{c}']['zero_cols'] == 40 - 10 * c and out[f'code10c{c}']['zero_cols_constant']
                               for c in (1, 2, 3, 4)))
    return out


def task_gate(run_dir, pilot):
    """Data loading, splits and encodings; writes raw/data_context.npz (state tables per substrate)."""
    gate = {}
    seqs = D.load_sequences(INPUT_HASHES, include_test=not pilot)
    legal = D.load_legal_table(ROOT, INPUT_HASHES)
    weights, counts = D.state_weights(seqs, D.LegalIndex(legal))
    tr, va = D.split_train_file()
    gate['splits_disjoint'] = bool(len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == D.N_TRAIN_FILE)
    gate['train_distinct_states'] = int((weights['train'] > 0).sum())
    gate['n_train_866'] = gate['train_distinct_states'] == 866
    gate['position_counts'] = counts
    gate['encodings'] = encoding_checks(legal)
    gate['model_free'] = model_free_check(legal)
    gate['pass'] = bool(gate['splits_disjoint'] and gate['n_train_866'] and gate['encodings']['pass']
                        and gate['model_free']['pass'])
    ctx = dict(w_train=weights['train'], w_val=weights['val'], n_train_positions=np.array(counts['train']))
    for sub in SUBSTRATES:
        ctx[f'states_{sub}'] = E.encode(sub, legal)
    if not pilot:
        ctx['w_test'] = weights['test']
    np.savez(run_dir / 'raw' / 'data_context.npz', **ctx)
    return gate


def make_dataset(cell, run_dir, pilot):
    """Same TableData construction as pwlib.sweep.make_dataset for 'bw' and 'code10'."""
    with np.load(Path(run_dir) / 'raw' / 'data_context.npz') as f:
        ctx = {k: f[k] for k in f.files}
    sub = cell['substrate']
    states = ctx[f'states_{sub}']  # KeyError = fail loudly
    if sub.startswith('code10c'):
        u = np.full(len(states), 1.0 / len(states))
        keys = ('dev',) if pilot else ('dev', 'conf')
        return D.TableData(states, u, {k: u for k in keys}, len(states), sub)
    w = dict(train=ctx['w_train'], dev=ctx['w_val'])
    if not pilot:
        w['conf'] = ctx['w_test']
    return D.TableData(states, ctx['w_train'], w, int(ctx['n_train_positions']), sub)


def execute_run(task):
    """Worker: train, evaluate and save one (cell, seed); one same-seed retry on failure (as pw02a)."""
    cell, seed, run_dir, pilot, steps, grid = task
    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    name = S.run_name(cell['cell_id'], seed)
    attempts = []
    for attempt in (1, 2):
        t0 = time.time()
        try:
            data = make_dataset(cell, run_dir, pilot)
            model, log = T.train(cell, seed, data, grid, steps=steps)
            arrays, metrics = T.evaluate(model, cell, data)
            flags = S.numerical_flags(cell, log, metrics)
            if not (flags['finite'] and flags['kl_nonnegative']):
                raise FloatingPointError(f'numerical flags failed: {flags}')
            torch.save(dict(state_dict=model.state_dict(), cell=cell, seed=seed, run_seed=log['run_seed']),
                       run_dir / 'models' / f'{name}.pt')
            np.savez_compressed(run_dir / 'raw' / f'{name}.npz', **arrays)
            summary = dict(run=name, cell_id=cell['cell_id'], label=cell['label'], seed=seed, status='ok',
                           attempts=attempts + [dict(attempt=attempt, status='ok', seconds=time.time() - t0)],
                           competence=S.competence(cell, metrics), metrics=metrics, numerical=flags, log=log,
                           d_in=int(data.d_in), pid=os.getpid())
            dump_json(summary, run_dir / 'runs' / f'{name}.json')
            return {k: summary[k] for k in ('run', 'cell_id', 'seed', 'status', 'competence', 'attempts')} | dict(
                wall_seconds=log['wall_seconds'])
        except Exception as exc:
            attempts.append(dict(attempt=attempt, status='failed', seconds=time.time() - t0,
                                 error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-2000:]))
    summary = dict(run=name, cell_id=cell['cell_id'], label=cell['label'], seed=seed, status='technical_failure',
                   attempts=attempts, competence=False)
    dump_json(summary, run_dir / 'runs' / f'{name}.json')
    return {k: summary[k] for k in ('run', 'cell_id', 'seed', 'status', 'competence', 'attempts')}


def planned_runs(grid, pilot):
    cells = {c['cell_id']: c for c in grid['cells']}
    if pilot:
        return [(cells[c], s) for c in grid['pilot_cells'] for s in grid['seeds']['pilot']]
    return [(c, s) for c in grid['cells'] for s in c['seeds']]


def cost(cell):
    return cell['latent'] + (0 if cell['family'] == 'plain_ae' else 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--results-root', default=None, help='SMOKE ONLY: scratch results directory')
    ap.add_argument('--smoke-steps', type=int, default=None, help='SMOKE ONLY: override training steps')
    ap.add_argument('--smoke-cells', default=None, help='SMOKE ONLY: fnmatch pattern on cell ids')
    args = ap.parse_args()
    smoke = args.smoke_steps is not None or args.smoke_cells is not None
    if smoke and args.results_root is None:
        raise SystemExit('Smoke flags need --results-root (scratch copy only)')
    run_dir = Path(args.results_root or EXP / 'results') / args.run_id
    run = read_run(run_dir)
    if run['experiment_id'] != 'pw06_encoding_and_latent_size':
        raise ValueError('RUN.json belongs to another experiment')
    if run['freeze_sha256'] != sha256(EXP / 'DESIGN.md'):
        raise ValueError('DESIGN.md changed since RUN.json was written')
    pilot = run['kind'] == 'pilot'
    grid_path = EXP / 'sweep_grid.json'
    grid = S.load_grid(grid_path)
    planned = planned_runs(grid, pilot)
    if not pilot and len(planned) != N_RUNS_FULL:
        raise ValueError('grid changed: amend, do not edit')
    for d in ('models', 'raw', 'runs'):
        (run_dir / d).mkdir(exist_ok=True)
    dump_json(grid, run_dir / 'sweep_grid_used.json')
    t0 = time.time()
    gate = task_gate(run_dir, pilot)
    dump_json(gate, run_dir / 'task_gate.json')
    if not gate['pass']:
        raise SystemExit(f'Task gate failed: see {run_dir / "task_gate.json"}')
    runs = planned
    if args.smoke_cells:
        runs = [(c, s) for c, s in runs if fnmatch.fnmatch(c['cell_id'], args.smoke_cells)]
    tasks = []
    for cell, seed in runs:
        if (run_dir / 'runs' / f'{S.run_name(cell["cell_id"], seed)}.json').exists():
            raise FileExistsError(f'Run already recorded: {cell["cell_id"]} s{seed} (no silent re-runs)')
        tasks.append((cell, seed, str(run_dir), pilot, args.smoke_steps, grid))
    tasks.sort(key=lambda t: -cost(t[0]))
    print(f'{len(tasks)} runs, {args.workers} workers', flush=True)
    with get_context('spawn').Pool(args.workers) as pool:
        results = []
        for r in pool.imap_unordered(execute_run, tasks):
            results.append(r)
            print(f'[{len(results)}/{len(tasks)}] {r["run"]} {r["status"]} {r.get("wall_seconds", 0):.0f}s', flush=True)
    results.sort(key=lambda r: r['run'])
    dump_json(dict(pilot=pilot, smoke=dict(steps=args.smoke_steps, cells=args.smoke_cells, root=args.results_root),
                   workers=args.workers, threads_per_worker=1, device='cpu', precision='float32',
                   wall_seconds=time.time() - t0, cpu_seconds_training=sum(r.get('wall_seconds', 0) for r in results),
                   grid_sha256=sha256(grid_path), design_sha256=sha256(EXP / 'DESIGN.md'),
                   torch=torch.__version__, python=sys.version.split()[0], runs=results),
              run_dir / 'runs_index.json')


if __name__ == '__main__':
    main()
