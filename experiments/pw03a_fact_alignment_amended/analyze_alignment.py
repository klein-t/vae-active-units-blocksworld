"""pw03a entry point: latent-to-fact alignment on the amended pw02a shared sweep (no training).

pw03a changes (PROTOCOL.md): pw02a input; H3.2 three-valued (a comparator cell with zero gated seeds
scores untested, not unsupported); H3.3 untested when a side has no gated seeds; H3.1 over the 6
remaining beta=1 cells (A3 removed).

Usage (after freeze + start): uv run python -u experiments/pw03a_fact_alignment_amended/analyze_alignment.py \
    --run-id ID --pw02-run PW02_RUN_ID [--workers 8] [--recompute-codes-only]
Pilot vs full from RUN.json; the pw02 run must be of the same kind, completed, gated and audited.
pw02 files read here are checked against the pw02 MANIFEST.json hashes. Definitions: PROTOCOL.md.
"""
import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pwlib import alignment as AL  # noqa: E402
from pwlib import blocksworld as bw  # noqa: E402
from pwlib import data as D  # noqa: E402
from pwlib import sweep as S  # noqa: E402
from pwlib.models import GaussVAE  # noqa: E402
from pwlib.runio import dump_json, load_json, read_run, results_header, sha256  # noqa: E402

EXP = Path(__file__).resolve().parent
PW02 = ROOT / 'experiments' / 'pw02a_active_units_amended'
N_PERM = {'raw': 100, 'b': 20}
N_PERM_DCI_RAW = 10
DCI_REF_CELLS = ('bw-gauss_vae-A2-b1-lr0.0003', 'bw-gauss_vae-A2-b4-lr0.0003', 'bw-tc_vae-A2-b2-lr0.0003',
                 'bw-tc_vae-A2-b6-lr0.0003', 'bw-cat_vae-A2-b1-lr0.0003', 'bw-topk_sae-k5-lr0.0003',
                 'bw-topk_sae-k10-lr0.0003', 'bw-plain_ae-A2-lr0.0003')
B_CELLS = ['b_raw_w128-gauss_vae-A2-b1-glearned-lr0.0003', 'b_raw_w128-tc_vae-A2-b6-glearned-lr0.0003',
           'b_raw_w128-cat_vae-A2-b1-glearned-lr0.0003', 'b_raw_w128-topk_sae-k5-lr0.0003',
           'b_raw_w128-topk_sae-k10-lr0.0003', 'b_raw_w128-plain_ae-A2-lr0.0003'] + \
          [f'b_lin_{s}-gauss_vae-A2-b1-glearned-lr0.0003' for s in ('w64', 'w128', 'w128s2', 'w128s3', 'w256')] + \
          [f'b_raw_{s}-gauss_vae-A2-b1-glearned-lr0.0003' for s in ('w64', 'w128s2', 'w128s3', 'w256')]
B_DCI_PREFIX = 'b_raw_w128-'
B_DEV_PILOT_N = 20_000
DCI_B_TRAIN, DCI_B_TEST = 10_000, 5_000
ORACLE_NOISE = 0.1
F1_THRESHOLD = 0.9


def verify_pw02(pw02_dir, pilot):
    run = read_run(pw02_dir)
    if (run['kind'] == 'pilot') != pilot:
        raise ValueError('pw02 run kind does not match this run kind')
    res = load_json(pw02_dir / 'results.json')
    audit = load_json(pw02_dir / 'AUDIT.json')
    ok = res['status'] == 'completed' and all(v is True for v in res['gates'].values()) and audit['status'] == 'pass'
    manifest = {r['path']: r for r in load_json(pw02_dir / 'MANIFEST.json')['files']}
    return ok, manifest


def checked(pw02_dir, manifest, rel):
    if rel not in manifest:
        raise KeyError(f'pw02 manifest lacks {rel}')
    p = pw02_dir / rel
    if sha256(p) != manifest[rel]['sha256']:
        raise ValueError(f'pw02 file changed since its manifest: {rel}')
    return p


def raw_context(pw02_dir, manifest, pilot):
    with np.load(checked(pw02_dir, manifest, 'raw/data_context.npz')) as z:
        ctx = {k: z[k] for k in z.files}
    legal = ctx['legal']
    w_dev, w_conf = (ctx['w_train'], ctx['w_val']) if pilot else (ctx['w_val'], ctx['w_test'])
    return dict(legal=legal, Y=legal.astype(np.int64), B=D.block_variables(legal), w_dev=w_dev, w_conf=w_conf)


def raw_codes(pw02_dir, manifest, cell, run, active_ids):
    """Codes on the 866 legal states and the active mask, from retained pw02 operands."""
    with np.load(checked(pw02_dir, manifest, f'raw/{run}.npz')) as z:
        fam = cell['family']
        if fam in ('gauss_vae', 'tc_vae'):
            codes = z['dev_mu']
            active = np.isin(np.arange(codes.shape[1]), active_ids)
        elif fam == 'cat_vae':
            probs = z['dev_probs']
            codes = probs[:, np.asarray(active_ids, dtype=int), :].reshape(len(probs), -1)
            active = np.ones(codes.shape[1], dtype=bool)
        elif fam == 'topk_sae':
            codes = z['dev_codes']
            active = np.isin(np.arange(codes.shape[1]), active_ids)
        else:
            codes = z['dev_codes']
            active = np.ones(codes.shape[1], dtype=bool)
    return codes.astype(np.float64), active


def b_codes(pw02_dir, manifest, grid, cell, seed, pilot):
    """Recompute codes on activation substrate (b) from the retained checkpoint and activation cache."""
    kind, src = S.source_of(cell['substrate'])
    field = {'raw': 'acts', 'lin': 'lin'}[kind]
    with np.load(checked(pw02_dir, manifest, f'raw/activations_{src}.npz')) as z:
        tr = z[f'train_{field}']
        if pilot:
            idx = np.sort(np.random.default_rng(0).choice(len(tr), min(B_DEV_PILOT_N, len(tr)), replace=False))
            dev, ydev = tr[idx], z['train_ycur'][idx]
            conf, yconf = z[f'val_{field}'], z['val_ycur']
        else:
            dev, ydev = z[f'val_{field}'], z['val_ycur']
            conf, yconf = z[f'conf_{field}'], z['conf_ycur']
        train_mean = torch.as_tensor(tr.mean(0))
    checked(pw02_dir, manifest, f'models/{S.run_name(cell["cell_id"], seed)}.pt')
    model = S.load_model(pw02_dir, cell, seed, grid, dev.shape[1], train_mean)
    out = []
    with torch.no_grad():
        for X in (dev, conf):
            x = torch.as_tensor(X, dtype=torch.float32)
            fam = cell['family']
            if fam in ('gauss_vae', 'tc_vae'):
                c = model.encode(x)[0]
            elif fam == 'cat_vae':
                c = model.logits(x).softmax(-1)
            elif fam == 'topk_sae':
                c = model.encode(x)
            else:
                c = model.enc(x)
            out.append(c.numpy().astype(np.float64))
    return out[0], out[1], ydev.astype(np.int64), yconf.astype(np.int64)


def score_task(task):
    torch.set_num_threads(1)
    t0 = time.time()
    kind = task['kind']
    if kind == 'raw':
        c = task['ctx']
        codes, active = task['codes'], task['active']
        dci = dict(x_train=codes, y_train=c['B'], x_test=codes, y_test=c['B'], w_train=c['w_dev'], w_test=c['w_conf'])
        summary, arrays = AL.score(codes, codes, c['Y'], c['Y'], c['B'], c['B'], c['w_dev'], c['w_conf'], active,
                                   n_perm=task['n_perm'], seed=task['perm_seed'], dci=dci,
                                   karvonen=task['karvonen'], perm_same_rows=True, n_perm_dci=task['n_perm_dci'])
    else:
        pw02_dir, manifest = Path(task['pw02_dir']), task['manifest']
        cdev, cconf, ydev, yconf = b_codes(pw02_dir, manifest, task['grid'], task['cell'], task['seed'], task['pilot'])
        if task['family'] == 'cat_vae':
            ids = np.asarray(task['active_ids'], dtype=int)
            cdev = cdev[:, ids, :].reshape(len(cdev), -1)
            cconf = cconf[:, ids, :].reshape(len(cconf), -1)
            active = np.ones(cdev.shape[1], dtype=bool)
        elif task['family'] in ('plain_ae',):
            active = np.ones(cdev.shape[1], dtype=bool)
        else:
            active = np.isin(np.arange(cdev.shape[1]), task['active_ids'])
        np.savez_compressed(Path(task['out_dir']) / f'codes_{task["name"]}.npz', dev=cdev.astype(np.float32),
                            conf=cconf.astype(np.float32))
        if task['codes_only']:
            return dict(name=task['name'], codes_only=True)
        bdev, bconf = D.block_variables(ydev), D.block_variables(yconf)
        wd, wc = np.full(len(cdev), 1 / len(cdev)), np.full(len(cconf), 1 / len(cconf))
        dci = None
        if task['cell']['cell_id'].startswith(B_DCI_PREFIX):
            r = np.random.default_rng(0)
            itr = r.choice(len(cdev), min(DCI_B_TRAIN, len(cdev)), replace=False)
            ite = r.choice(len(cconf), min(DCI_B_TEST, len(cconf)), replace=False)
            dci = dict(x_train=cdev[itr], y_train=bdev[itr], x_test=cconf[ite], y_test=bconf[ite], w_train=None,
                       w_test=None)
        summary, arrays = AL.score(cdev, cconf, ydev, yconf, bdev, bconf, wd, wc, active, n_perm=N_PERM['b'],
                                   seed=task['perm_seed'], dci=dci, karvonen=task['karvonen'], perm_same_rows=False)
    np.savez_compressed(Path(task['out_dir']) / f'scores_{task["name"]}.npz', **arrays)
    return dict(name=task['name'], cell_id=task.get('cell_id'), seed=task.get('seed'), family=task['family'],
                substrate=task['substrate'], seconds=time.time() - t0, **summary)


def controls(ctx):
    """Fact oracle, block oracle, random-init A2 encoders (seeds 1-3), PCA top-25 scores."""
    rng = np.random.default_rng(0)
    n = len(ctx['Y'])
    fact_oracle = np.c_[ctx['Y'] + ORACLE_NOISE * rng.normal(size=(n, 36)), rng.normal(size=(n, 28))]
    block_oracle = np.c_[ctx['B'] + ORACLE_NOISE * rng.normal(size=(n, 5)), rng.normal(size=(n, 27))]
    out = dict(fact_oracle=fact_oracle, block_oracle=block_oracle)
    grid = load_json(PW02 / 'sweep_grid.json')
    for s in (1, 2, 3):
        torch.manual_seed(s)
        m = GaussVAE(36, grid['architectures']['A2']['hidden'], 32, 'bernoulli')
        with torch.no_grad():
            out[f'random_init_s{s}'] = m.encode(torch.as_tensor(ctx['legal'], dtype=torch.float32))[0].numpy().astype(np.float64)
    w = ctx['w_dev'] / ctx['w_dev'].sum()
    Xc = ctx['legal'] - w @ ctx['legal']
    _, _, Vt = np.linalg.svd(np.sqrt(w)[:, None] * Xc, full_matrices=False)
    out['pca25'] = Xc @ Vt[:25].T
    return out


def f1_numerical_check(codes, ctx, arrays_f1, sign, thr):
    """Recompute 50 random (code, fact) confirmation F1 values with sklearn.metrics.f1_score."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(50):
        j, f = rng.integers(codes.shape[1]), rng.integers(36)
        pred = (sign[j, f] * codes[:, j]) > thr[j, f]
        ref = f1_score(ctx['Y'][:, f], pred, sample_weight=ctx['w_conf'], zero_division=0.0)
        worst = max(worst, abs(ref - arrays_f1[j, f]))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--pw02-run', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--recompute-codes-only', action='store_true')
    ap.add_argument('--smoke-perm', type=int, default=None, help='SMOKE ONLY: override permutation counts')
    args = ap.parse_args()
    run_dir = EXP / 'results' / args.run_id
    run = read_run(run_dir)
    pilot = run['kind'] == 'pilot'
    pw02_dir = PW02 / 'results' / args.pw02_run
    pw02_ok, manifest = verify_pw02(pw02_dir, pilot)
    out_dir = run_dir / 'raw'
    out_dir.mkdir(exist_ok=True)
    grid = S.load_grid(checked(pw02_dir, manifest, 'sweep_grid_used.json'))
    cells = {c['cell_id']: c for c in grid['cells']}
    index = load_json(checked(pw02_dir, manifest, 'runs_index.json'))
    active_sets = load_json(checked(pw02_dir, manifest, 'au_per_run.json'))
    gated = {r['run']: r for r in index['runs'] if r['status'] == 'ok' and r['competence']}
    ctx = raw_context(pw02_dir, manifest, pilot)
    n_perm_raw = N_PERM['raw'] if args.smoke_perm is None else args.smoke_perm
    if args.smoke_perm is not None:
        N_PERM['b'] = args.smoke_perm
    # task gate: labels from pwlib round-trip and the 30/6 partition
    roundtrip = all(np.array_equal(bw.state_from_positions(
        {x: bw.POSITION_VALUES[x][c] for x, c in zip(bw.BLOCKS, bw.positions_from_state(v))}), v) for v in ctx['legal'])
    blocks_match = np.array_equal(ctx['B'], np.array([bw.positions_from_state(v) for v in ctx['legal']]))
    partition = len(D.POSITION_COLUMNS) == 30 and len(D.DERIVED_COLUMNS) == 6 and \
        len(np.intersect1d(D.POSITION_COLUMNS, D.DERIVED_COLUMNS)) == 0
    task_gate = bool(pw02_ok and roundtrip and blocks_match and partition)

    tasks = []
    base = dict(out_dir=str(out_dir), codes_only=args.recompute_codes_only)
    if not args.recompute_codes_only:
        for name, codes in controls(ctx).items():
            tasks.append(base | dict(kind='raw', name=f'control_{name}', family='control', substrate='bw', ctx=ctx,
                                     codes=codes, active=np.ones(codes.shape[1], dtype=bool), n_perm=n_perm_raw,
                                     perm_seed=0, karvonen=False, n_perm_dci=min(N_PERM_DCI_RAW, n_perm_raw)))
    for run_name, rec in sorted(gated.items()):
        cell = cells[rec['cell_id']]
        if cell['substrate'] == 'bw' and cell['family'] in ('gauss_vae', 'tc_vae', 'cat_vae', 'topk_sae', 'plain_ae'):
            if args.recompute_codes_only:
                continue
            codes, active = raw_codes(pw02_dir, manifest, cell, run_name, active_sets[run_name])
            tasks.append(base | dict(kind='raw', name=run_name, cell_id=cell['cell_id'], seed=rec['seed'],
                                     family=cell['family'], substrate='bw', ctx=ctx, codes=codes, active=active,
                                     n_perm=n_perm_raw, perm_seed=rec['seed'], karvonen=cell['family'] == 'topk_sae',
                                     n_perm_dci=min(N_PERM_DCI_RAW, n_perm_raw) if cell['cell_id'] in DCI_REF_CELLS else 0))
        elif cell['cell_id'] in B_CELLS:
            tasks.append(base | dict(kind='b', name=run_name, cell_id=cell['cell_id'], seed=rec['seed'], cell=cell,
                                     family=cell['family'], substrate=cell['substrate'], grid=grid, pilot=pilot,
                                     pw02_dir=str(pw02_dir), manifest=manifest, active_ids=active_sets[run_name],
                                     perm_seed=rec['seed'], karvonen=cell['family'] == 'topk_sae'))
    t0 = time.time()
    with get_context('spawn').Pool(args.workers) as pool:
        scored = []
        for r in pool.imap_unordered(score_task, tasks):
            scored.append(r)
            print(f'[{len(scored)}/{len(tasks)}] {r["name"]} {r.get("seconds", 0):.1f}s', flush=True)
    if args.recompute_codes_only:
        return
    scored = {r['name']: r for r in scored}
    # numerical gate
    fo = np.load(out_dir / 'scores_control_fact_oracle.npz')
    f1_err = f1_numerical_check(controls(ctx)['fact_oracle'], ctx, fo['f1_conf'], fo['sign'], fo['threshold'])
    no_nan = all(np.isfinite(np.load(out_dir / f'scores_{n}.npz')['f1_conf']).all() for n in scored)
    numerical = bool(f1_err < 1e-9 and no_nan)
    # metric-validity (competence) gate
    validity = dict(fact_oracle_cov30=scored['control_fact_oracle']['cov30'],
                    block_oracle_block_mig=scored['control_block_oracle']['block_mig'],
                    random_init_cov30=[scored[f'control_random_init_s{s}']['cov30'] for s in (1, 2, 3)])
    competence = bool(validity['fact_oracle_cov30'] >= 0.95 and validity['block_oracle_block_mig'] >= 0.5
                      and max(validity['random_init_cov30']) <= 0.2)
    gates = dict(task=task_gate, numerical=numerical, competence=competence)

    def vals(cid, field):
        return [r[field] for r in scored.values() if r.get('cell_id') == cid and r[field] is not None]

    criteria, secondary = {}, {}
    sae = vals('bw-topk_sae-k5-lr0.0003', 'cov30')
    comparators = ['bw-gauss_vae-A2-b1-lr0.0003', 'bw-gauss_vae-A2-b4-lr0.0003', 'bw-tc_vae-A2-b2-lr0.0003',
                   'bw-tc_vae-A2-b6-lr0.0003']
    comp_max = {c: max(vals(c, 'cov30'), default=None) for c in comparators}
    h32 = AL.h32_verdict(sae, {c: vals(c, 'cov30') for c in comparators})
    beta1 = [f'bw-gauss_vae-{a}-b1-lr{lr}' for a in ('A1', 'A2', 'A4') for lr in ('0.0003', '0.003')]
    mono = {c: vals(c, 'mono_frac') for c in beta1}
    ref = mono['bw-gauss_vae-A2-b1-lr0.0003']
    h31 = bool(ref) and max(ref) < 0.5 and sum(bool(v) and np.median(v) < 0.5 for v in mono.values()) >= 6

    def separated(a, b):
        return AL.separated_or_untested(vals(a, 'block_mig'), vals(b, 'block_mig'))
    secondary['h31_vae_latents_not_monosemantic_descriptive'] = dict(rule_holds=h31, mono_frac=mono)
    secondary['h33_categorical_vs_gaussian_block_mig'] = separated('bw-cat_vae-A2-b1-lr0.0003', 'bw-gauss_vae-A2-b1-lr0.0003')
    secondary['h33_tcvae_vs_vae_block_mig'] = separated('bw-tc_vae-A2-b6-lr0.0003', 'bw-gauss_vae-A2-b1-lr0.0003')
    b_sae = vals('b_raw_w128-topk_sae-k5-lr0.0003', 'cov30')
    b_vae_mono = vals('b_raw_w128-gauss_vae-A2-b1-glearned-lr0.0003', 'mono_frac')
    b_vae_cov = vals('b_raw_w128-gauss_vae-A2-b1-glearned-lr0.0003', 'cov30')
    med = lambda v: float(np.median(v)) if v else None  # noqa: E731
    secondary['h34_activation_substrate_alignment'] = dict(
        sae_cov30_b=med(b_sae), sae_cov30_raw=med(sae), vae_mono_b=med(b_vae_mono), vae_mono_raw=med(ref),
        vae_cov30_b=med(b_vae_cov),
        holds=None if None in (med(b_sae), med(sae), med(b_vae_mono), med(ref), med(b_vae_cov)) else bool(
            med(b_sae) < med(sae) and med(b_vae_mono) < med(ref) and med(b_sae) > med(b_vae_cov)))
    if pilot:
        status = 'completed' if all(gates.values()) else ('technical_stop' if not gates['task'] or not gates['numerical']
                                                          else 'competence_stop')
        outcome = 'pilot_only' if status == 'completed' else 'untested'
        interp = 'Pilot on the pw02 pilot run (fit on train weights, scored on val weights); no hypothesis tested.'
        criteria = {}
    elif not all(gates.values()):
        status = 'technical_stop' if not (gates['task'] and gates['numerical']) else 'competence_stop'
        outcome, interp = 'untested', 'Gate failure; alignment metrics not interpreted.'
        criteria = {'h32_topk_sae_covers_position_facts': None}
    elif h32 is None:
        status, outcome = 'inconclusive', 'untested'
        interp = ('H3.2 untested: the k=5 SAE or a Gaussian-latent comparator cell has no pw02a-gated seed, and no '
                  'measured part of the criterion fails (three-valued rule, PROTOCOL.md).')
        criteria = {'h32_topk_sae_covers_position_facts': None}
    else:
        status = 'completed'
        criteria['h32_topk_sae_covers_position_facts'] = bool(h32)
        outcome = 'supported' if h32 else 'unsupported'
        interp = ('Alignment of codes to the 30 position facts on the confirmation distribution, scoped to these '
                  'architectures, this 36-fact encoding and data distribution.')
    metrics = dict(validity=validity, f1_sklearn_max_abs_diff=f1_err, n_models=len(scored),
                   h32=dict(sae_k5_cov30=sae, comparator_max_cov30=comp_max), wall_seconds=time.time() - t0,
                   per_model={k: {f: v[f] for f in v if f not in ('name',)} for k, v in scored.items()},
                   smoke_perm=args.smoke_perm, pw02_run=args.pw02_run,
                   pw02_results_sha256=sha256(pw02_dir / 'results.json'))
    res = results_header(run_dir) | dict(status=status, outcome=outcome, interpretation=interp,
                                         confirmation_accessed=not pilot, claims_changed=False, gates=gates,
                                         criteria=criteria, secondary=secondary, metrics=metrics)
    dump_json(res, run_dir / 'results.json')
    print(dict(status=status, outcome=outcome, gates=gates))


if __name__ == '__main__':
    main()
