"""Descriptive, unregistered analysis (2026-10-22, requested by the project lead after the pw04a pilot
competence stop): apply the FROZEN pw04a analysis code unchanged to the pw02a full_001 checkpoints.

This is NOT a pw04a run: pw04a ended in a pilot competence_stop, so no full run can be registered and no
hypothesis verdict exists. The frozen module (working copy checked against the sha256 of pw04a's frozen snapshot) is loaded and run with its
output redirected to descriptive/results/pw02a_<pw02_run>/ (no RUN.json is created or faked; the header fields say
'descriptive'). Its own gate and criterion fields are retained only as descriptive numbers.

Usage: uv run python experiments/pw04a_sample_legality_amended/descriptive_full_legality.py --pw02-run full_001
"""
import argparse
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
FROZEN = EXP / 'analyze_legality.py'
# sha256 of pw04a's frozen snapshot of analyze_legality.py (also recorded in pw05 DESIGN.json external_inputs)
FROZEN_SHA256 = 'd7c073e6daa3a877d1527c4ec2142d3c7c0b91cf8a56d76f6a309b341ebb889c'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pw02-run', required=True)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / 'src'))
    # The working copy must be byte-identical to the frozen snapshot (module paths resolve from its location).
    from pwlib.runio import sha256
    if sha256(FROZEN) != FROZEN_SHA256:
        raise RuntimeError('analyze_legality.py differs from its frozen snapshot')
    sys.path.insert(0, str(EXP))  # spawn workers inherit sys.path and import the same module
    import analyze_legality as mod
    out_root = EXP / 'descriptive'
    label = f'pw02a_{args.pw02_run}'
    (out_root / 'results' / label).mkdir(parents=True, exist_ok=False)
    mod.EXP = out_root  # the frozen code writes to EXP/results/<label> = descriptive/results/<label>
    out_dir, real_read_run = (out_root / 'results' / label).resolve(), mod.read_run
    fake = dict(kind='full', run_id=label, experiment_id='pw04a_sample_legality_amended',
                freeze_sha256='descriptive-unregistered')
    mod.read_run = lambda d: fake if Path(d).resolve() == out_dir else real_read_run(d)  # pw02a RUN.json read for real
    mod.results_header = lambda d: dict(experiment_id='pw04a_sample_legality_amended', run_id=label,
                                        freeze_sha256='descriptive-unregistered',
                                        note='DESCRIPTIVE ONLY: not a registered run; pw04a competence gate failed '
                                             'in its pilot; no hypothesis verdict')
    sys.argv = [str(FROZEN), '--run-id', label, '--pw02-run', args.pw02_run, '--workers', str(args.workers)]
    mod.main()


if __name__ == '__main__':
    main()
