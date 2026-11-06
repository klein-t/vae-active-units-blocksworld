"""pw04a entry point: legality of decoded prior samples from the amended pw02a sweep (no training).

pw04a changes (PROTOCOL.md): pw02a input; beta=1 cells A1/A2/A4 (A3 removed); H4.4 untested when a side
has no gated seeds.

Usage (after freeze + start): uv run python -u experiments/pw04a_sample_legality_amended/analyze_legality.py \
    --run-id ID --pw02-run PW02_RUN_ID [--workers 8]
Pilot vs full from RUN.json; the pw02 run must be of the same kind, completed, gated and audited.
Legality: pwlib.blocksworld.is_legal plus membership in the pw01 866-state table (must agree on
every sample). Definitions: PROTOCOL.md sections 2-8.
"""
import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pwlib import blocksworld as bw  # noqa: E402
from pwlib import data as D  # noqa: E402
from pwlib import legality as LG  # noqa: E402
from pwlib import metrics as M  # noqa: E402
from pwlib import sweep as S  # noqa: E402
from pwlib.runio import dump_json, load_json, read_run, results_header, sha256  # noqa: E402

EXP = Path(__file__).resolve().parent
PW02 = ROOT / 'experiments' / 'pw02a_active_units_amended'
N_SAMPLES = 20_000
SAMPLE_SEED_OFFSET = 1000
AGG_SEED_OFFSET = 2000
BOOTSTRAP_REPS = 2000
N_MC = 1_000_000
N_RANDOM_GATE = 100_000
AGG_LEGAL_MIN = 0.95
STRATUM_MIN_RUNS = 8
B2_UNIFORM = 866 / 7776
RAW_FAMILIES = ('gauss_vae', 'tc_vae', 'cat_vae')
B_FAMILIES = ('gauss_vae', 'tc_vae', 'cat_vae')


def verify_pw02(pw02_dir, pilot):
    run = read_run(pw02_dir)
    if (run['kind'] == 'pilot') != pilot:
        raise ValueError('pw02 run kind does not match this run kind')
    res = load_json(pw02_dir / 'results.json')
    audit = load_json(pw02_dir / 'AUDIT.json')
    ok = res['status'] == 'completed' and all(v is True for v in res['gates'].values()) and audit['status'] == 'pass'
    return ok, {r['path']: r for r in load_json(pw02_dir / 'MANIFEST.json')['files']}


def checked(pw02_dir, manifest, rel):
    if rel not in manifest:
        raise KeyError(f'pw02 manifest lacks {rel}')
    if sha256(pw02_dir / rel) != manifest[rel]['sha256']:
        raise ValueError(f'pw02 file changed since its manifest: {rel}')
    return pw02_dir / rel


@torch.no_grad()
def decode_samples(model, cell, n, gen, x_eval=None, w_eval=None):
    """Bernoulli means (or Gaussian means) for prior samples, or for aggregate-posterior samples
    when x_eval/w_eval are given (x ~ evaluation distribution, z ~ q(z|x))."""
    fam = cell['family']
    if x_eval is not None:
        idx = torch.multinomial(torch.as_tensor(w_eval), n, replacement=True, generator=gen)
        x = x_eval[idx]
    if fam in ('gauss_vae', 'tc_vae'):
        if x_eval is None:
            z = torch.randn((n, model.latent), generator=gen)
        else:
            mu, lv = model.encode(x)
            z = mu + (0.5 * lv).exp() * torch.randn(mu.shape, generator=gen)
        return model.dec.mean(model.dec(z))
    if fam == 'cat_vae':
        if x_eval is None:
            cls = torch.randint(model.n_classes, (n, model.n_vars), generator=gen)
        else:
            q = model.logits(x).softmax(-1)
            cls = torch.multinomial(q.reshape(-1, model.n_classes), 1, generator=gen).view(n, model.n_vars)
        y = torch.nn.functional.one_hot(cls, model.n_classes).float()
        return model.dec.mean(model.dec(y.flatten(1)))
    raise ValueError(fam)


def legality_block(S_bits, index):
    chk = LG.checker_flags(S_bits)
    tab = LG.table_member(S_bits, index)
    if not np.array_equal(chk, tab):
        raise ValueError('Checker and 866-table disagree')
    V = LG.violations(S_bits)
    ill = ~chk
    return chk, V, dict(rate=float(chk.mean()), wilson95=M.wilson(int(chk.sum()), len(chk)),
                        any_violated={v: float(V[ill, i].mean()) if ill.any() else 0.0
                                      for i, v in enumerate(LG.VIOLATIONS)},
                        first_violated={v: float((LG.first_violation(V[ill]) == i).mean()) if ill.any() else 0.0
                                        for i, v in enumerate(LG.VIOLATIONS)})


def legal_distribution(S_bits, legal_flags, index, w_eval):
    idx = index.lookup(S_bits[legal_flags])
    if len(idx) == 0:
        return dict(distinct_legal=0, tv_to_eval=None)
    p = np.bincount(idx, minlength=866) / len(idx)
    return dict(distinct_legal=int((p > 0).sum()), tv_to_eval=float(0.5 * np.abs(p - w_eval / w_eval.sum()).sum()))


def sample_task(task):
    torch.set_num_threads(1)
    t0 = time.time()
    cell, seed = task['cell'], task['seed']
    pw02_dir, manifest = Path(task['pw02_dir']), task['manifest']
    ctx = task['ctx']
    index = D.LegalIndex(ctx['legal'])
    sub = cell['substrate']
    name = S.run_name(cell['cell_id'], seed)
    checked(pw02_dir, manifest, f'models/{name}.pt')
    out, arrays = dict(name=name, cell_id=cell['cell_id'], seed=seed, family=cell['family'], substrate=sub), {}
    if sub.startswith('b_'):
        kind, src = S.source_of(sub)
        with np.load(checked(pw02_dir, manifest, f'raw/probes_{src}.npz')) as p:
            W, b = p['cur_W'], p['cur_b']
        with np.load(checked(pw02_dir, manifest, f'raw/activations_{src}.npz')) as a:
            tr = a[f'train_{"acts" if kind == "raw" else "lin"}']
        model = S.load_model(pw02_dir, cell, seed, task['grid'], tr.shape[1], torch.as_tensor(tr.mean(0)))
        g = torch.Generator().manual_seed(seed + SAMPLE_SEED_OFFSET)
        acts = decode_samples(model, cell, N_SAMPLES, g).numpy()
        bits = ((acts @ W.T + b) > 0).astype(np.int8)
        legal, V, lb = legality_block(bits, index)
        out['probe_read_prior'] = lb
        arrays['prior_probe_bits'] = np.packbits(bits, axis=1)
    else:
        X = torch.as_tensor(ctx['eval_states'][sub], dtype=torch.float32)
        w_eval = ctx['eval_weights'][sub]
        model = S.load_model(pw02_dir, cell, seed, task['grid'], 36, X.mean(0))
        g = torch.Generator().manual_seed(seed + SAMPLE_SEED_OFFSET)
        mean = decode_samples(model, cell, N_SAMPLES, g).numpy()
        mode = (mean > 0.5).astype(np.int8)
        stoch = (np.random.default_rng(seed + SAMPLE_SEED_OFFSET).random(mean.shape) < mean).astype(np.int8)
        ga = torch.Generator().manual_seed(seed + AGG_SEED_OFFSET)
        agg = (decode_samples(model, cell, N_SAMPLES, ga, X, w_eval).numpy() > 0.5).astype(np.int8)
        for tag, Sb in (('prior_mode', mode), ('prior_stochastic', stoch), ('aggpost_mode', agg)):
            legal, V, lb = legality_block(Sb, index)
            if sub == 'bw':
                lb.update(legal_distribution(Sb, legal, index, ctx['eval_weights']['bw']))
            if sub == 'prod5':
                lb['bw_legal_position_bits'] = float(LG.legal_from_position_bits(Sb).mean())
                lb['in_support'] = float(((Sb[:, D.DERIVED_COLUMNS] == 0).all(1) & ~LG.violations(Sb)[:, 0]).mean())
            if sub == 'code10':
                lb['codeword_valid'] = float(LG.codeword_valid(Sb, ctx['code10_bit_of_pos']).mean())
                lb['in_support'] = lb['codeword_valid']
            out[tag] = lb
            arrays[f'{tag}_bits'] = np.packbits(Sb, axis=1)
            arrays[f'{tag}_legal'] = legal
            arrays[f'{tag}_violations'] = np.packbits(V, axis=1)
        agg_rate = out['aggpost_mode']['in_support'] if sub in ('code10', 'prod5') else out['aggpost_mode']['rate']
        out['competence'] = None if sub == 'indep' else bool(agg_rate >= AGG_LEGAL_MIN)
    np.savez_compressed(Path(task['out_dir']) / f'samples_{name}.npz', **arrays)
    out['seconds'] = time.time() - t0
    return out


def baselines(ctx, index):
    rng = np.random.default_rng(0)
    marg, bm = ctx['marginals_train'], LG.block_marginals(ctx['legal'], ctx['w_train'])
    exact = dict(B0=LG.baseline_b0(), B1=LG.baseline_b1(ctx['legal'], marg), B2=LG.baseline_b2(bm),
                 B2_uniform=LG.baseline_b2(np.full((5, 6), 1 / 6)))
    mc = dict(B0=LG.mc_b0(N_MC, rng, index) / N_MC, B1=LG.mc_b1(marg, N_MC, rng, index) / N_MC,
              B2=LG.mc_b2(bm, N_MC, rng, index) / N_MC)
    agree = {k: bool(abs(mc[k] - exact[k]) <= 3 * max(np.sqrt(exact[k] * (1 - exact[k]) / N_MC), 1e-12) + 1e-12)
             for k in mc}
    return dict(exact=exact, monte_carlo=mc, agree_3se=agree, block_marginals=bm)


def stratified_spearman(rows, reps, seed=0):
    """Pooled run-count-weighted Spearman(AU, prior legality) within beta strata; cluster bootstrap over cells."""
    strata = {}
    for r in rows:
        strata.setdefault(r['beta'], {}).setdefault(r['cell_id'], []).append((r['au'], r['legal']))
    strata = {b: c for b, c in strata.items() if sum(len(v) for v in c.values()) >= STRATUM_MIN_RUNS}

    def pooled(st):
        num, den = 0.0, 0
        for cells in st.values():
            pts = [p for v in cells.values() for p in v]
            a, l_ = zip(*pts)
            rho = spearmanr(a, l_)[0]
            if np.isfinite(rho):
                num += rho * len(pts)
                den += len(pts)
        return num / den if den else np.nan
    obs = pooled(strata)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(reps):
        st = {}
        for b, cells in strata.items():
            keys = list(cells)
            pick = rng.choice(len(keys), len(keys), replace=True)
            st[b] = {f'{keys[i]}#{j}': cells[keys[i]] for j, i in enumerate(pick)}
        boots.append(pooled(st))
    boots = np.array(boots)
    finite = boots[np.isfinite(boots)]
    ci = (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))) if len(finite) else (None, None)
    return dict(pooled_rho=None if not np.isfinite(obs) else float(obs), ci95=ci, strata=sorted(strata),
                nonfinite_reps=int((~np.isfinite(boots)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--pw02-run', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--smoke-bootstrap', type=int, default=None, help='SMOKE ONLY: bootstrap replicates')
    args = ap.parse_args()
    run_dir = EXP / 'results' / args.run_id
    run = read_run(run_dir)
    pilot = run['kind'] == 'pilot'
    pw02_dir = PW02 / 'results' / args.pw02_run
    pw02_ok, manifest = verify_pw02(pw02_dir, pilot)
    (run_dir / 'raw').mkdir(exist_ok=True)
    grid = S.load_grid(checked(pw02_dir, manifest, 'sweep_grid_used.json'))
    cells = {c['cell_id']: c for c in grid['cells']}
    index_runs = load_json(checked(pw02_dir, manifest, 'runs_index.json'))['runs']
    au = {r['run']: r for r in load_json(checked(pw02_dir, manifest, 'au_table.json'))}
    with np.load(checked(pw02_dir, manifest, 'raw/data_context.npz')) as z:
        dc = {k: z[k] for k in z.files}
    w_eval = dc['w_val'] if pilot else dc['w_test']
    ctx = dict(legal=dc['legal'], w_train=dc['w_train'], marginals_train=dc['marginals_train'],
               code10_bit_of_pos=dc['code10_bit_of_pos'],
               eval_states=dict(bw=dc['legal'], code10=dc['code10_states'], prod5=dc['prod5_states'],
                                indep=None),
               eval_weights=dict(bw=w_eval, code10=np.full(1024, 1 / 1024), prod5=np.full(7776, 1 / 7776)))
    rng = np.random.default_rng(1)
    indep_eval = (rng.random((N_SAMPLES, 36)) < dc['marginals_train']).astype(np.int8)
    ctx['eval_states']['indep'] = indep_eval
    ctx['eval_weights']['indep'] = np.full(N_SAMPLES, 1 / N_SAMPLES)
    index = D.LegalIndex(ctx['legal'])
    # task gate
    rand = np.random.default_rng(2).integers(0, 2, size=(N_RANDOM_GATE, 36)).astype(np.int8)
    task_checks = dict(pw02_ok=pw02_ok,
                       eval_states_legal=bool(LG.checker_flags(ctx['legal'][w_eval > 0]).all()),
                       table_reproduced=bool(np.array_equal(bw.enumerate_legal_bruteforce(), ctx['legal'])),
                       random_rejected=bool(not LG.checker_flags(rand).any() and not LG.table_member(rand, index).any()))
    tasks = []
    for r in index_runs:
        cell = cells[r['cell_id']]
        if r['status'] != 'ok' or not r['competence']:
            continue
        raw_ok = cell['substrate'] in ('bw', 'code10', 'prod5', 'indep') and cell['family'] in RAW_FAMILIES
        b_ok = cell['substrate'].startswith('b_') and cell['family'] in B_FAMILIES
        if raw_ok or b_ok:
            tasks.append(dict(cell=cell, seed=r['seed'], pw02_dir=str(pw02_dir), manifest=manifest, ctx=ctx,
                              grid=grid, out_dir=str(run_dir / 'raw')))
    t0 = time.time()
    with get_context('spawn').Pool(args.workers) as pool:
        out = []
        for o in pool.imap_unordered(sample_task, tasks):
            out.append(o)
            print(f'[{len(out)}/{len(tasks)}] {o["name"]} {o["seconds"]:.1f}s', flush=True)
    out = {o['name']: o for o in out}
    base = baselines(ctx, index)
    gates = dict(task=bool(all(task_checks.values())), numerical=bool(all(base['agree_3se'].values())))
    # competence: every beta=1 bw cell has a majority of seeds passing aggregate-posterior legality
    beta1 = [f'bw-gauss_vae-{a}-b1-lr{lr}' for a in ('A1', 'A2', 'A4') for lr in ('0.0003', '0.003')]
    if pilot:
        beta1 = [c for c in beta1 if c in grid['pilot_cells']]
    comp = {}
    for cid in beta1:
        n_planned = 1 if pilot else len(cells[cid]['seeds'])
        passing = sum(bool(o['competence']) for o in out.values() if o['cell_id'] == cid)
        comp[cid] = dict(planned=n_planned, passing=passing)
    gates['competence'] = bool(all(v['passing'] >= v['planned'] // 2 + 1 for v in comp.values()))
    gated = [o for o in out.values() if o.get('competence')]
    b2 = base['exact']['B2']
    criteria, secondary = {}, {}
    h41_runs = [o for o in gated if o['cell_id'] in beta1]
    h41 = bool(h41_runs) and all(o['prior_mode']['rate'] > b2 for o in h41_runs)
    rows = [dict(cell_id=o['cell_id'], beta=cells[o['cell_id']]['beta'], au=au[o['name']]['au'],
                 legal=o['prior_mode']['rate']) for o in gated
            if o['substrate'] == 'bw' and o['family'] == 'gauss_vae']
    reps = BOOTSTRAP_REPS if args.smoke_bootstrap is None else args.smoke_bootstrap
    h42 = stratified_spearman(rows, reps) if rows else dict(pooled_rho=None, ci95=(None, None), strata=[])
    secondary['h43_prior_hole_gap'] = bool(h41_runs) and all(
        o['aggpost_mode']['rate'] - o['prior_mode']['rate'] >= 0.05 for o in h41_runs)
    cat = [o['prior_mode']['rate'] for o in gated if o['cell_id'] == 'bw-cat_vae-A2-b1-lr0.0003']
    gau = [o['prior_mode']['rate'] for o in gated if o['cell_id'] == 'bw-gauss_vae-A2-b1-lr0.0003']
    secondary['h44_categorical_vs_gaussian_legality'] = dict(
        cat=cat, gauss=gau, separated=None if not (cat and gau) else bool(min(cat) > max(gau) or max(cat) < min(gau)))
    prod = [o['prior_mode']['bw_legal_position_bits'] for o in gated if o['substrate'] == 'prod5']
    code = [o['prior_mode']['codeword_valid'] for o in gated if o['substrate'] == 'code10']
    secondary['h45_prod5_and_code10_controls'] = dict(
        prod5_bw_legal=prod, prod5_within_0p03_of_uniform=bool(prod) and all(abs(p - B2_UNIFORM) <= 0.03 for p in prod),
        code10_codeword_valid=code)
    if pilot:
        status = 'completed' if all(gates.values()) else ('technical_stop' if not (gates['task'] and gates['numerical'])
                                                          else 'competence_stop')
        outcome = 'pilot_only' if status == 'completed' else 'untested'
        interp = 'Pilot on the pw02 pilot run (val-weight evaluation distribution); no hypothesis tested.'
    elif not all(gates.values()):
        status = 'technical_stop' if not (gates['task'] and gates['numerical']) else 'competence_stop'
        outcome, interp = 'untested', 'Gate failure; legality rates not interpreted.'
        criteria = {'h41_prior_legality_above_product_baseline': None, 'h42_legality_decreases_with_au': None}
    else:
        status = 'completed'
        criteria['h41_prior_legality_above_product_baseline'] = h41
        criteria['h42_legality_decreases_with_au'] = bool(h42['ci95'][1] is not None and h42['ci95'][1] < 0)
        n = sum(criteria.values())
        outcome = 'supported' if n == 2 else 'unsupported' if n == 0 else 'mixed'
        interp = ('Mode-decoded prior-sample legality of these trained models against exact factorized baselines; '
                  'H4.2 is an association within the sweep, not a causal effect of AU.')
    metrics = dict(baselines={k: v for k, v in base.items() if k != 'block_marginals'},
                   block_marginals_train=base['block_marginals'], task_checks=task_checks,
                   competence_cells=comp, h41=dict(n_runs=len(h41_runs), min_rate=min(
                       (o['prior_mode']['rate'] for o in h41_runs), default=None), B2=b2),
                   h42=h42, h42_rows=rows, per_model=out, wall_seconds=time.time() - t0,
                   eval_set='val' if pilot else 'test', pw02_run=args.pw02_run,
                   pw02_results_sha256=sha256(pw02_dir / 'results.json'), smoke_bootstrap=args.smoke_bootstrap)
    res = results_header(run_dir) | dict(status=status, outcome=outcome, interpretation=interp,
                                         confirmation_accessed=not pilot, claims_changed=False, gates=gates,
                                         criteria=criteria, secondary=secondary, metrics=metrics)
    dump_json(res, run_dir / 'results.json')
    print(dict(status=status, outcome=outcome, gates=gates))


if __name__ == '__main__':
    main()
