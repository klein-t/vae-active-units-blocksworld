"""pw02a step 0b (unchanged from pw02): exp46 final-layer activations for substrate (b), source gates and targets_b.json.

Usage (after freeze + start): uv run python -u experiments/pw02a_active_units_amended/extract_activations.py --run-id ID
Pilot runs (RUN.json kind 'pilot') use development splits only; the thesis test file is not opened.
Must run before train_sweep.py (which refuses b cells without these files); targets_b.json is
written here, before any VAE is trained. Conventions: pwlib/activations.py and PROTOCOL.md section 3.
"""
import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pwlib import activations as A  # noqa: E402
from pwlib import data as D  # noqa: E402
from pwlib.runio import dump_json, load_json, read_run  # noqa: E402

EXP = Path(__file__).resolve().parent
GATE = dict(source_probe_val_bitwise_min=0.99, source_probe_val_exact_min=0.80)


def extract_source(args):
    src, run_dir, pilot, hashes, smoke_seqs = args
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    name = A.SOURCES[src]
    D.check_hash(A.checkpoint_path(name), hashes[A.checkpoint_key(name)])
    model, depth, width, missing = A.load_exp46(name)
    seqs = D.load_sequences(hashes, include_test=not pilot)
    splits = {'train': seqs['train'], 'val': seqs['val']}
    if not pilot:
        splits['conf'] = seqs['test']
    tabs, head = {}, {}
    for split, (X, Y) in splits.items():
        idx = A.subsample_sequences(len(X), split)
        if smoke_seqs:
            idx = idx[:smoke_seqs]
        H, L, hook = A.residuals(model, depth, X[idx])
        tabs[split] = A.position_table(X[idx], Y[idx], H, L)
        pred = tabs[split]['head_logits'] > 0
        head[split] = dict(vs_next_bitwise=float((pred == tabs[split]['ynext'].astype(bool)).mean()),
                           vs_current_bitwise=float((pred == tabs[split]['ycur'].astype(bool)).mean()))
    mean, std = A.standardize_stats(tabs['train']['acts'])
    for t in tabs.values():
        t['acts'] = ((t['acts'] - mean) / std).astype(np.float32)
    tr, va = tabs['train'], tabs['val']
    cur = A.fit_probes(tr['acts'], tr['ycur'])
    nxt = A.fit_probes(tr['acts'], tr['ynext'])
    cur_val = A.probe_accuracy(cur, va['acts'], va['ycur'])
    nxt_val = A.probe_accuracy(nxt, va['acts'], va['ynext'])
    B = A.b_lin_map(tr['acts'], tr['ycur'])
    for t in tabs.values():
        t['lin'] = A.apply_b_lin(B, t['ycur'])
    Vt, ratios = A.pca_basis(tr['acts'])
    kp, scan = A.k_probe(tr['acts'], tr['ycur'], va['acts'], va['ycur'], cur_val['bitwise'], Vt, tr['acts'].mean(0))
    U = np.unique(tr['lin'].round(5), axis=0).astype(np.float64)
    lin_distinct = int(len(U))
    sv = np.linalg.svd(U - U.mean(0), compute_uv=False)
    lin_affine = int((sv > sv.max() * 1e-5).sum())  # relative tolerance: float32 storage of an exact affine map
    arrays = {f'{s}_{k}': v for s, t in tabs.items() for k, v in t.items() if k != 'head_logits'}
    np.savez(Path(run_dir) / 'raw' / f'activations_{src}.npz', **arrays)
    np.savez(Path(run_dir) / 'raw' / f'probes_{src}.npz', cur_W=cur['W'], cur_b=cur['b'], next_W=nxt['W'],
             next_b=nxt['b'], mean=mean, std=std, b_lin=B, pca_Vt=Vt, pca_ratios=ratios)
    gate = dict(current_probe_val=cur_val, next_probe_val=nxt_val,
                current_beats_next=bool(cur_val['bitwise'] > nxt_val['bitwise']),
                pass_=bool(cur_val['bitwise'] >= GATE['source_probe_val_bitwise_min']
                           and cur_val['exact'] >= GATE['source_probe_val_exact_min']
                           and cur_val['bitwise'] > nxt_val['bitwise']))
    return src, dict(checkpoint=name, depth=depth, width=width, hook=hook, ignored_missing_keys=missing,
                     positions={s: int(len(t['acts'])) for s, t in tabs.items()},
                     sequences={s: int(len(np.unique(t['seq']))) for s, t in tabs.items()},
                     exp46_head=head, gate=gate,
                     numerical=dict(b_lin_distinct_rows_train=lin_distinct, b_lin_affine_dim_train=lin_affine,
                                    b_lin_ok=bool(lin_distinct <= 866 and lin_affine <= 25)),
                     targets=dict(K_pca99=A.k_pca99(ratios), K_probe=kp, K_probe_scan=scan,
                                  probe_fit_n=A.PROBE_FIT_N), seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--smoke-seqs', type=int, default=0, help='SMOKE ONLY: cap sequences per split')
    args = ap.parse_args()
    run_dir = EXP / 'results' / args.run_id
    run = read_run(run_dir)
    pilot = run['kind'] == 'pilot'
    design = load_json(EXP / 'DESIGN.json')
    hashes = design['external_inputs']
    (run_dir / 'raw').mkdir(exist_ok=True)
    if (run_dir / 'targets_b.json').exists():
        raise FileExistsError('targets_b.json already written for this run')
    if any((run_dir / 'models').glob('*.pt')):
        raise RuntimeError('Models exist already: targets must be fixed before any VAE is trained')
    tasks = [(s, str(run_dir), pilot, hashes, args.smoke_seqs) for s in A.SOURCES]
    with get_context('spawn').Pool(min(args.workers, len(tasks))) as pool:
        out = dict(pool.map(extract_source, tasks))
    for v in out.values():
        v['gate']['pass'] = v['gate'].pop('pass_')
    dump_json(dict(pilot=pilot, smoke_seqs=args.smoke_seqs, sources=out), run_dir / 'source_gate.json')
    dump_json(dict(definitions=dict(K_pca99='PCs explaining 99% of the variance of standardized train activations',
                                    K_probe='smallest q whose 36 top-q-PC probes reach val bitwise accuracy within '
                                            '0.005 of all-dims probes (probes fit on 20,000 train positions)'),
                   values={s: {k: v['targets'][k] for k in ('K_pca99', 'K_probe')} for s, v in out.items()},
                   smoke_seqs=args.smoke_seqs), run_dir / 'targets_b.json', exclusive=True)
    print({s: dict(gate=v['gate']['pass'], seconds=round(v['seconds'], 1)) for s, v in out.items()})


if __name__ == '__main__':
    main()
