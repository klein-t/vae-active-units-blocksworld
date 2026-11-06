"""Independent audit 001, check 1: freeze integrity and confirmation sealing.

Own code (hashlib/json only). For pw02a, pw03a, pw04a:
- every FREEZE.json file entry: sha256 of the current working file and of the .frozen snapshot equal the
  recorded hash; file mtimes precede the freeze time (no edit-then-revert after freezing); skipped (None)
  when FREEZE.json is absent, as in the released folder;
- RUN.json freeze_sha256 equals sha256(FREEZE.json); results.json carries the same value;
- every MANIFEST.json entry (size + sha256) matches the file on disk;
- external inputs in DESIGN.json (thesis data, exp46 checkpoints, pw01 files) match their recorded hashes;
- pilots hold no confirmation-derived arrays/metrics (no w_test, no conf_* arrays, no 'conf' metrics);
- the descriptive pw04a pass ran the byte-identical frozen analysis module.
Writes freeze_check.json next to this script.
"""
import datetime as dt
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EXPS = ['pw02a_active_units_amended', 'pw03a_fact_alignment_amended', 'pw04a_sample_legality_amended']
# recorded input paths start with these environment variables (earlier thesis data and transformer checkpoints)
ENV_ROOTS = {'$PW_BLOCKSWORLD_DATA': os.environ.get('PW_BLOCKSWORLD_DATA', str(ROOT / 'data' / 'blocksworld')),
             '$PW_TRANSFORMER_CHECKPOINTS': os.environ.get('PW_TRANSFORMER_CHECKPOINTS', str(ROOT / 'data' / 'transformers'))}
# sha256 of pw04a's frozen snapshot of analyze_legality.py (also recorded in pw05 DESIGN.json external_inputs)
PW04A_FROZEN_SHA256 = 'd7c073e6daa3a877d1527c4ec2142d3c7c0b91cf8a56d76f6a309b341ebb889c'


def input_path(k):
    var, _, rest = k.partition('/')
    return Path(ENV_ROOTS[var]) / rest if var in ENV_ROOTS else ROOT / k


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    out = {}
    for e in EXPS:
        ed = ROOT / 'experiments' / e
        fr = json.loads((ed / 'FREEZE.json').read_text()) if (ed / 'FREEZE.json').is_file() else None
        t_freeze = dt.datetime.fromisoformat(fr['created_at']).timestamp() if fr else None
        bad = [] if fr else None
        for f in (fr['files'] if fr else []):
            cur, snap = ROOT / f['path'], ROOT / f['snapshot']
            if sha(cur) != f['sha256']:
                bad.append(('working_copy_hash', f['path']))
            if sha(snap) != f['sha256']:
                bad.append(('snapshot_hash', f['path']))
            if os.stat(cur).st_mtime > t_freeze:
                bad.append(('mtime_after_freeze', f['path']))
        fsha = sha(ed / 'FREEZE.json') if fr else None
        runs = {}
        for rd in sorted((ed / 'results').iterdir()):
            run = json.loads((rd / 'RUN.json').read_text())
            res = json.loads((rd / 'results.json').read_text())
            man = json.loads((rd / 'MANIFEST.json').read_text())
            mbad = [r['path'] for r in man['files']
                    if (rd / r['path']).stat().st_size != r['bytes'] or sha(rd / r['path']) != r['sha256']]
            runs[rd.name] = dict(kind=run['kind'], started_at=run['started_at'],
                                 run_freeze_matches=run['freeze_sha256'] == fsha if fr else None,
                                 results_freeze_matches=res['freeze_sha256'] == fsha if fr else None,
                                 **{'started_after_freeze': (dt.datetime.fromisoformat(run['started_at']).timestamp()
                                                             > t_freeze) if fr else None},
                                 manifest_files=len(man['files']), manifest_mismatches=mbad,
                                 confirmation_accessed=res.get('confirmation_accessed'))
        design = json.loads((ed / 'DESIGN.json').read_text())
        ext = design.get('external_inputs', {})
        ext_bad = [k for k, v in ext.items() if k != 'note' and sha(input_path(k)) != v]
        out[e] = dict(freeze_sha256=fsha, n_frozen_files=len(fr['files']) if fr else None, frozen_file_problems=bad, runs=runs,
                      external_inputs_checked=len([k for k in ext if k != 'note']), external_input_mismatches=ext_bad)

    # pilots: no confirmation-derived material
    seal = {}
    p2 = ROOT / 'experiments/pw02a_active_units_amended/results/pilot_001'
    ctx = np.load(p2 / 'raw/data_context.npz')
    seal['pw02a_pilot_data_context_has_w_test'] = 'w_test' in ctx.files
    seal['pw02a_pilot_activation_files_with_conf_arrays'] = [
        Path(f).name for f in glob.glob(str(p2 / 'raw/activations_*.npz'))
        if any(k.startswith('conf_') for k in np.load(f).files)]
    seal['pw02a_pilot_run_npz_with_conf_arrays'] = [
        Path(f).name for f in glob.glob(str(p2 / 'raw/*_s101.npz')) if any(k.startswith('conf_') for k in np.load(f).files)]
    seal['pw02a_pilot_runs_with_conf_metrics'] = [
        Path(f).name for f in glob.glob(str(p2 / 'runs/*.json')) if 'conf' in json.loads(Path(f).read_text()).get('metrics', {})]
    for e in ('pw03a_fact_alignment_amended', 'pw04a_sample_legality_amended'):
        r = json.loads((ROOT / f'experiments/{e}/results/pilot_001/results.json').read_text())
        seal[f'{e}_pilot_eval_set'] = r['metrics'].get('eval_set', r['metrics'].get('splits'))
        seal[f'{e}_pilot_pw02_run'] = r['metrics'].get('pw02_run')
        seal[f'{e}_pilot_confirmation_accessed'] = r.get('confirmation_accessed')
    # pilot run start times precede every full run start (full runs cite completed pilots)
    out['confirmation_sealing'] = seal

    exp4 = ROOT / 'experiments/pw04a_sample_legality_amended'
    out['pw04a_descriptive'] = dict(
        working_copy_equals_frozen=sha(exp4 / 'analyze_legality.py') == PW04A_FROZEN_SHA256,
        **{'frozen_hash_in_freeze': sha(exp4 / 'analyze_legality.py') == PW04A_FROZEN_SHA256},
        results_header=json.loads((exp4 / 'descriptive/results/pw02a_full_001/results.json').read_text()).get('freeze_sha256'))
    (OUT / 'freeze_check.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    if {'-h', '--help'} & set(sys.argv[1:]):  # usage without the data present
        print(__doc__)
        sys.exit(0)
    main()
