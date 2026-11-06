"""Shared-sweep plumbing for pw02 (training) and pw03/pw04 (reuse without retraining).

The pw02 run directory holds everything downstream analyses need:
  raw/data_context.npz          legal table, state weights per split, control datasets
  raw/activations_<src>.npz     standardized exp46 activations, b_lin, labels (substrate b)
  raw/probes_<src>.npz          current/next-state probes, standardization, b_lin map
  models/<cell>_s<seed>.pt      final checkpoints
  raw/<cell>_s<seed>.npz        evaluation operands
  runs/<cell>_s<seed>.json      per-run log (timing, curve, competence, numerical flags)
"""
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import training as T
from .runio import dump_json

AU_GATES = dict(raw_state_recon_exact_min=0.95, raw_state_recon_bitwise_min=0.99,
                activation_probe_on_recon_bitwise_min=0.98, activation_recon_r2_min=0.90)
CONV_REL = 0.02            # relative change of the dev objective over the last 20% of steps
CONV_ABS = 1e-3            # pw02a: absolute tolerance (nats per sample) for near-zero objectives
CONV_ABS_FAMILIES = ('plain_ae', 'topk_sae')


def load_grid(path):
    g = json.loads(Path(path).read_text())
    for c in g['cells']:
        for k in ('cell_id', 'substrate', 'family', 'lr', 'latent', 'seeds', 'uses'):
            if k not in c:
                raise KeyError(f'grid cell missing {k}: {c}')
    if len({c['cell_id'] for c in g['cells']}) != len(g['cells']):
        raise ValueError('Duplicate cell ids')
    return g


def run_name(cell_id, seed):
    return f'{cell_id}_s{seed}'


def planned_runs(grid, pilot):
    cells = {c['cell_id']: c for c in grid['cells']}
    if pilot:
        missing = [c for c in grid['pilot_cells'] if c not in cells]
        if missing:
            raise KeyError(f'pilot cells not in grid: {missing}')
        return [(cells[c], s) for c in grid['pilot_cells'] for s in grid['seeds']['pilot']]
    return [(c, s) for c in grid['cells'] for s in c['seeds']]


def source_of(substrate):
    """'b_raw_w128s2' -> ('raw', 'w128s2'); raw-state substrates -> (None, None)."""
    if not substrate.startswith('b_'):
        return None, None
    _, kind, src = substrate.split('_', 2)
    return kind, src


def eval_keys(pilot):
    return ('dev',) if pilot else ('dev', 'conf')


def cost_rank(cell):
    """Rough relative cost for longest-first scheduling (measured ms/step ordering)."""
    fam = cell['family']
    if fam == 'tc_vae':
        return 100 if cell['substrate'].startswith('b_') else 50
    return {'A1': 1, 'A2': 1.3, 'A3': 2.4, 'A4': 6.7}.get(cell.get('arch'), 1.5) + (1 if cell['substrate'].startswith('b_') else 0)


# ---------------------------------------------------------------- datasets

_CACHE = {}


def _npz(path):
    if path not in _CACHE:
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing sweep input: {path}')
        with np.load(path) as f:
            _CACHE[path] = {k: f[k] for k in f.files}
    return _CACHE[path]


def make_dataset(cell, run_dir, pilot):
    """Dataset object for a cell plus (probe, labels) for activation substrates."""
    run_dir = Path(run_dir)
    sub = cell['substrate']
    keys = eval_keys(pilot)
    ctx = _npz(str(run_dir / 'raw' / 'data_context.npz'))
    if sub == 'bw':
        w = dict(train=ctx['w_train'], dev=ctx['w_val'])
        if not pilot:
            w['conf'] = ctx['w_test']
        return D.TableData(ctx['legal'], ctx['w_train'], w, int(ctx['n_train_positions']), sub), None, None
    if sub in ('code10', 'prod5'):
        S = ctx[f'{sub}_states']
        u = np.full(len(S), 1.0 / len(S))
        return D.TableData(S, u, {k: u for k in keys}, len(S), sub), None, None
    if sub == 'indep':
        return D.BernoulliData(ctx['marginals_train'], sub, keys), None, None
    kind, src = source_of(sub)
    acts = _npz(str(run_dir / 'raw' / f'activations_{src}.npz'))
    probes = _npz(str(run_dir / 'raw' / f'probes_{src}.npz'))
    field = {'raw': 'acts', 'lin': 'lin'}[kind]
    split_of = dict(dev='val', conf='conf')
    evals = {k: acts[f'{split_of[k]}_{field}'] for k in keys}
    labels = {k: acts[f'{split_of[k]}_ycur'] for k in keys}
    probe = dict(W=probes['cur_W'], b=probes['cur_b'])
    return D.ArrayData(acts[f'train_{field}'], evals, sub), probe, labels


def competence(cell, metrics):
    """Per-run competence on the development evaluation set (pw02 PROTOCOL.md section 8)."""
    m = metrics['dev']
    if cell['substrate'].startswith('b_'):
        return bool(m['probe_bitwise'] >= AU_GATES['activation_probe_on_recon_bitwise_min']
                    and m['r2'] >= AU_GATES['activation_recon_r2_min'])
    return bool(m['exact'] >= AU_GATES['raw_state_recon_exact_min']
                and m['bitwise'] >= AU_GATES['raw_state_recon_bitwise_min'])


def is_learned_gamma(cell):
    return cell['substrate'].startswith('b_') and cell.get('gamma') == 'learned'


def active_set(cell, arrays, key='dev'):
    """Sorted indices of active latent units (Gaussian: Burda AU; categorical: activity > 0.01); None otherwise."""
    from . import metrics as M
    if cell['family'] in ('gauss_vae', 'tc_vae'):
        return [int(i) for i in np.flatnonzero(M.active_units(arrays[f'{key}_mu'], arrays[f'{key}_weights'])[1])]
    if cell['family'] == 'cat_vae':
        act = M.categorical_activity(arrays[f'{key}_probs'], arrays[f'{key}_weights'])
        return [int(i) for i in np.flatnonzero(act > M.AU_THRESHOLD)]
    return None


def objective_converged(curve, abs_tol=None):
    """|obj(final) - obj(80%)| / |obj(80%)| < 2% on the dev objective; with abs_tol, also accepted when
    |obj(final) - obj(80%)| < abs_tol (nats per sample). None if the curve is too short."""
    if len(curve) < 2:
        return None
    steps = [r['step'] for r in curve]
    ref = min(range(len(steps)), key=lambda i: abs(steps[i] - 0.8 * steps[-1]))
    a, b = curve[ref]['dev_objective'], curve[-1]['dev_objective']
    rel_ok = abs(b - a) / max(abs(a), 1e-12) < CONV_REL
    return bool(rel_ok or (abs_tol is not None and abs(b - a) < abs_tol))


def converged_amended(cell, run_log):
    """pw02a convergence rule. Learned-gamma cells (objective unbounded below as gamma -> 0): the
    reconstruction gate passes at the 80% snapshot and at the end, and the active-unit set is identical
    at both. Other cells: objective test, with the absolute tolerance for plain AE / SAE."""
    if is_learned_gamma(cell):
        snap = run_log['snapshot']  # KeyError = fail loudly
        return bool(snap['competence'] and run_log['competence'] and snap['active_dev'] == run_log['active_dev_final'])
    return objective_converged(run_log['log']['curve'], CONV_ABS if cell['family'] in CONV_ABS_FAMILIES else None)


def numerical_flags(cell, log, metrics):
    finite = all(np.isfinite(r['train_loss']) and np.isfinite(r['dev_objective']) for r in log['curve'])
    kl_ok = all(m.get('min_unit_kl', 0.0) >= -1e-6 for m in metrics.values())
    return dict(finite=bool(finite), kl_nonnegative=bool(kl_ok),
                log_gamma_at_floor=bool(log.get('log_gamma_at_floor', False)))


def execute_run(task):
    """Worker entry: train, evaluate and save one (cell, seed); one same-seed retry on failure."""
    cell, seed, run_dir, pilot, steps = task
    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    name = run_name(cell['cell_id'], seed)
    attempts = []
    for attempt in (1, 2):
        t0 = time.time()
        try:
            data, probe, labels = make_dataset(cell, run_dir, pilot)
            grid = _grid(run_dir)
            snap_frac = grid['training'].get('convergence_snapshot_fraction')  # absent in pre-pw02a grids
            snapshot = None
            if snap_frac is None:
                model, log = T.train(cell, seed, data, grid, steps=steps)
            else:
                model, log, snap = T.train(cell, seed, data, grid, steps=steps, snapshot_fraction=snap_frac)
                m80 = T.build_model(cell, grid, data.d_in, data.train_mean)
                m80.load_state_dict(snap['state_dict'])
                arr80, met80 = T.evaluate(m80, cell, data, probe, labels)
                snapshot = dict(step=snap['step'], metrics_dev=met80['dev'], competence=competence(cell, met80),
                                active_dev=active_set(cell, arr80))
            arrays, metrics = T.evaluate(model, cell, data, probe, labels)
            flags = numerical_flags(cell, log, metrics)
            if not (flags['finite'] and flags['kl_nonnegative']):
                raise FloatingPointError(f'numerical flags failed: {flags}')
            torch.save(dict(state_dict=model.state_dict(), cell=cell, seed=seed, run_seed=log['run_seed']),
                       run_dir / 'models' / f'{name}.pt')
            np.savez_compressed(run_dir / 'raw' / f'{name}.npz', **arrays)
            summary = dict(run=name, cell_id=cell['cell_id'], seed=seed, status='ok', attempts=attempts + [
                dict(attempt=attempt, status='ok', seconds=time.time() - t0)], competence=competence(cell, metrics),
                metrics=metrics, numerical=flags, log=log, pid=os.getpid())
            if snapshot is not None:
                summary['snapshot'] = snapshot
                summary['active_dev_final'] = active_set(cell, arrays)
            dump_json(summary, run_dir / 'runs' / f'{name}.json')
            return {k: summary[k] for k in ('run', 'cell_id', 'seed', 'status', 'competence', 'attempts')} | dict(
                wall_seconds=log['wall_seconds'])
        except Exception as exc:  # retried once, then recorded as a technical failure
            attempts.append(dict(attempt=attempt, status='failed', seconds=time.time() - t0,
                                 error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-2000:]))
    summary = dict(run=name, cell_id=cell['cell_id'], seed=seed, status='technical_failure', attempts=attempts,
                   competence=False)
    dump_json(summary, run_dir / 'runs' / f'{name}.json')
    return {k: summary[k] for k in ('run', 'cell_id', 'seed', 'status', 'competence', 'attempts')}


def _grid(run_dir):
    return _npz_json(str(Path(run_dir) / 'sweep_grid_used.json'))


_JSON = {}


def _npz_json(path):
    if path not in _JSON:
        _JSON[path] = json.loads(Path(path).read_text())
    return _JSON[path]


def load_model(run_dir, cell, seed, grid, d_in, train_mean):
    """Rebuild a retained checkpoint (pw03/pw04 reuse)."""
    ck = torch.load(Path(run_dir) / 'models' / f'{run_name(cell["cell_id"], seed)}.pt', map_location='cpu',
                    weights_only=False)
    if ck['cell']['cell_id'] != cell['cell_id'] or ck['seed'] != seed:
        raise ValueError('Checkpoint identity mismatch')
    model = T.build_model(cell, grid, d_in, train_mean)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model
