"""Run-directory helpers shared by the pwNN entry scripts.

JSON-safe dumping (numpy scalars/arrays converted, NaN/inf rejected), sha256,
RUN.json writing and reading, and MANIFEST.json / AUDIT.json writing.

Register a run before its entry script (writes experiments/<pwNN_name>/results/<ID>/RUN.json):
    uv run python src/pwlib/runio.py start experiments/<pwNN_name> --run-id ID --kind {pilot,full,analysis}
"""
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CONTROL_FILES = {'RUN.json', 'MANIFEST.json', 'AUDIT.json', 'CLAIM_UPDATES.json'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def jsonable(obj, path='$'):
    """Recursively convert numpy types; raise on non-finite floats (fail loudly, naming the key path)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v, f'{path}.{k}') for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, f'{path}[{i}]') for i, v in enumerate(obj)]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist(), path)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not math.isfinite(v):
            raise ValueError(f'Non-finite value in JSON output at {path}: {v}')
        return v
    if obj is None or isinstance(obj, (str, int)):
        return obj
    raise TypeError(f'Not JSON-serializable: {type(obj)}')


def dump_json(obj, path, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(obj), indent=1, allow_nan=False) + '\n'
    with open(path, 'x' if exclusive else 'w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing required file: {path}')
    return json.loads(path.read_text())


def require_keys(d, keys, where):
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f'{where}: missing keys {missing}')
    return d


def read_run(run_dir):
    """RUN.json of a registered run (created by `start_run`); fail if absent."""
    run = load_json(Path(run_dir) / 'RUN.json')
    require_keys(run, ('experiment_id', 'run_id', 'kind', 'freeze_sha256'), 'RUN.json')
    return run


def file_records(base, paths):
    base = Path(base)
    return [dict(path=str(p), bytes=(base / p).stat().st_size, sha256=sha256(base / p)) for p in paths]


def write_manifest(run_dir):
    """MANIFEST.json over every run file except the four control records."""
    run_dir = Path(run_dir)
    paths = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob('*')
                   if p.is_file() and not (p.parent == run_dir and p.name in CONTROL_FILES))
    records = file_records(run_dir, paths)
    dump_json(dict(files=records), run_dir / 'MANIFEST.json')
    return records


def write_audit(run_dir, status, method, limitations, evidence, reason=None, checks=None):
    run_dir = Path(run_dir)
    run = read_run(run_dir)
    audit = dict(run_id=run['run_id'], results_sha256=sha256(run_dir / 'results.json'), status=status,
                 method=method, limitations=limitations, checks=checks or {},
                 evidence=file_records(run_dir, evidence))
    if reason:
        audit['reason'] = reason
    dump_json(audit, run_dir / 'AUDIT.json')
    return audit


def results_header(run_dir):
    run = read_run(run_dir)
    return dict(experiment_id=run['experiment_id'], run_id=run['run_id'], freeze_sha256=run['freeze_sha256'])


def start_run(exp_dir, run_id, kind, pilot_run=None):
    """Create results/<run_id>/RUN.json in an experiment directory; fail if the run directory exists.

    freeze_sha256 is the SHA-256 of the experiment's plan (DESIGN.md if present, else DESIGN.json).
    """
    exp_dir = Path(exp_dir)
    if kind not in ('pilot', 'full', 'analysis'):
        raise ValueError(f'Invalid run kind: {kind}')
    plan = exp_dir / 'DESIGN.md' if (exp_dir / 'DESIGN.md').is_file() else exp_dir / 'DESIGN.json'
    run_dir = exp_dir / 'results' / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    record = dict(schema_version=1, experiment_id=exp_dir.resolve().name, run_id=run_id, kind=kind,
                  freeze_sha256=sha256(plan), pilot_run=pilot_run, started_at=datetime.now(timezone.utc).isoformat())
    dump_json(record, run_dir / 'RUN.json', exclusive=True)
    return record


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Register a run: write experiments/<pwNN_name>/results/<ID>/RUN.json.')
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('start', help='create the run directory and its RUN.json')
    s.add_argument('experiment', type=Path, help='experiment directory, e.g. experiments/pw01_state_space_ground_truth')
    s.add_argument('--run-id', required=True)
    s.add_argument('--kind', choices=['pilot', 'full', 'analysis'], required=True)
    s.add_argument('--pilot-run', help='pilot run ID cited by a full run')
    args = ap.parse_args()
    print(json.dumps(start_run(args.experiment, args.run_id, args.kind, args.pilot_run), indent=2))


if __name__ == '__main__':
    main()
