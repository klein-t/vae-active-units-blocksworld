"""pw06 pilot health (DESIGN.md "Pilot"): runs complete, objective finite, runtime. No active-unit counts.

Usage: PYTHONPATH=src .venv/bin/python experiments/pw06_encoding_and_latent_size/pilot_check.py --run-id pilot_001
Writes results/ID/pilot_health.json. Technical extra: E36-L32 seed 101 shares pw02a's cell id and run seed,
so its training curve is compared with pw02a pilot_001's (same computation check; no counts involved).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from pwlib import sweep as S  # noqa: E402
from pwlib.runio import dump_json, load_json, read_run  # noqa: E402

PW02A_PILOT = ROOT / 'experiments/pw02a_active_units_amended/results/pilot_001/runs'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    run_dir = EXP / 'results' / args.run_id
    if read_run(run_dir)['kind'] != 'pilot':
        raise SystemExit('pilot runs only')
    grid = S.load_grid(run_dir / 'sweep_grid_used.json')
    cells = {c['cell_id']: c for c in grid['cells']}
    index = load_json(run_dir / 'runs_index.json')
    out = {}
    for cid in grid['pilot_cells']:
        for seed in grid['seeds']['pilot']:
            name = S.run_name(cid, seed)
            log = load_json(run_dir / 'runs' / f'{name}.json')
            rec = dict(status=log['status'], attempts=len(log['attempts']))
            if log['status'] == 'ok':
                curve = log['log']['curve']
                rec.update(steps=log['log']['steps'], finite=log['numerical']['finite'],
                           kl_nonnegative=log['numerical']['kl_nonnegative'],
                           wall_seconds=log['log']['wall_seconds'], final_dev_objective=curve[-1]['dev_objective'],
                           converged=S.converged_amended(cells[cid], log),
                           dev_exact=log['metrics']['dev']['exact'], dev_bitwise=log['metrics']['dev']['bitwise'],
                           threads=log['log']['threads'], torch=log['log']['torch'])
                ref = PW02A_PILOT / f'{name}.json'
                if ref.is_file():
                    rc = load_json(ref)['log']['curve']
                    a = np.array([r['dev_objective'] for r in curve])
                    b = np.array([r['dev_objective'] for r in rc])
                    rec['same_as_pw02a_pilot'] = dict(curve_identical=bool(len(a) == len(b) and np.array_equal(a, b)),
                                                      max_abs_diff=float(np.max(np.abs(a - b))) if len(a) == len(b) else None)
            out[name] = rec
    ok = all(r['status'] == 'ok' and r['finite'] and r['kl_nonnegative'] for r in out.values())
    dump_json(dict(pass_complete_and_finite=ok, wall_seconds_total=index['wall_seconds'], workers=index['workers'],
                   threads_per_worker=index['threads_per_worker'], device=index['device'], runs=out),
              run_dir / 'pilot_health.json')
    print('pilot complete and finite:', ok)


if __name__ == '__main__':
    main()
