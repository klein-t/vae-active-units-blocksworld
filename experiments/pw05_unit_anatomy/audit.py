"""pw05 audit: recompute the headline numbers of a pw05 run by separate code paths (PROTOCOL.md, Gates).

Rate in float64 from mu/logvar; H(X) from data_context; distortion with a different MC seed and a hand-written
Bernoulli NLL; Q7 from logvar; MI from explicit joint tables with hand-written binning; F1 fact counts from the
stored pw03a matrices; eigenvalues by SVD; legality funnel per sample with the pw01 checker and hand-written
well-formedness / cross-block code; independence prediction by Monte Carlo; one model's prior and aggregate-
posterior samples regenerated with the frozen pw04a sampler and compared byte for byte. Negative controls:
uniform random bit vectors through the funnel; permuted block labels for MI.

Usage: uv run python experiments/pw05_unit_anatomy/audit.py --run-id analysis_001 [--smoke-dir DIR]
"""
import argparse
import importlib.util
import math
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pwlib import blocksworld as bw  # noqa: E402
from pwlib import data as D  # noqa: E402
from pwlib import sweep as S  # noqa: E402
from pwlib.runio import dump_json, load_json, write_audit, write_manifest  # noqa: E402

EXP = Path(__file__).resolve().parent
# working copy, byte-identical to pw04a's frozen snapshot (sha256 in DESIGN.json external_inputs)
PW04_FROZEN = ROOT / 'experiments/pw04a_sample_legality_amended/analyze_legality.py'
LN2 = math.log(2)
REGEN_MODEL = 'bw-gauss_vae-A2-b1-lr0.0003'
SUCC = None  # filled per block below


def place_cols(x):
    return [bw.INDEX[f'on_table({x})'], bw.INDEX[f'holding({x})']] + \
        [bw.INDEX[f'on({x},{y})'] for y in bw.BLOCKS if y != x]


PLACES = {x: ['T', 'H'] + [y for y in bw.BLOCKS if y != x] for x in bw.BLOCKS}


def positions_legal(pos):
    """Cross-block legality of a dict block -> place, written independently of pwlib."""
    held = [x for x, p in pos.items() if p == 'H']
    if len(held) > 1:
        return False
    supports = [p for p in pos.values() if p not in ('T', 'H')]
    if len(supports) != len(set(supports)) or any(p in held for p in supports):
        return False
    for x in pos:  # no cycles
        cur, seen = x, set()
        while pos[cur] not in ('T', 'H'):
            if cur in seen:
                return False
            seen.add(cur)
            cur = pos[cur]
    return True


def audit_funnel(Sb):
    n = len(Sb)
    wf = np.stack([Sb[:, place_cols(x)].sum(1) == 1 for x in bw.BLOCKS], 1)
    cross = np.zeros(n, dtype=bool)
    legal = np.array([bw.is_legal(v) for v in Sb])
    counts = np.zeros((5, 6))
    for i in np.flatnonzero(wf.all(1)):
        pos = {x: PLACES[x][int(np.argmax(Sb[i, place_cols(x)]))] for x in bw.BLOCKS}
        cross[i] = positions_legal(pos)
    for b, x in enumerate(bw.BLOCKS):
        rows = wf[:, b]
        np.add.at(counts[b], Sb[rows][:, place_cols(x)].argmax(1), 1)
    p = counts / n
    return dict(wf_all=float(wf.all(1).mean()), position_legal=float(cross.mean()), legal=float(legal.mean()),
                implied_wf_all=float(np.prod(wf.mean(0))), p=p)


def mc_implied(p, n=200_000, seed=7):
    """Monte Carlo of independent per-block draws (value or 'malformed'); returns rate of legal positions."""
    rng = np.random.default_rng(seed)
    draws = []
    for b in range(5):
        q = np.r_[p[b], max(0.0, 1 - p[b].sum())]
        draws.append(rng.choice(7, size=n, p=q / q.sum()))
    draws = np.stack(draws, 1)
    ok = 0
    for row in draws[(draws < 6).all(1)]:
        ok += positions_legal({x: PLACES[x][row[b]] for b, x in enumerate(bw.BLOCKS)})
    rate = ok / n
    return rate, math.sqrt(max(rate * (1 - rate), 1e-12) / n)


def funnel_task(args):
    name, tag, path = args
    with np.load(path) as z:
        Sb = np.unpackbits(z[f'{tag}_bits'], axis=1)[:, :36].astype(np.int8)
    f = audit_funnel(Sb)
    f['mc_implied'] = mc_implied(f.pop('p'))
    return name, tag, f


def bin20(x):
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros(len(x), dtype=int)
    return np.clip(np.floor((x - lo) / (hi - lo) * 20).astype(int), 0, 19)


def mi_bits(a, b, w):
    J = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(J, (a, b), w)
    pa, pb = J.sum(1), J.sum(0)
    nz = J > 0
    return float((J[nz] * np.log2(J[nz] / np.outer(pa, pb)[nz])).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id')
    ap.add_argument('--smoke-dir', type=Path)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    run_dir = args.smoke_dir if args.smoke_dir else EXP / 'results' / args.run_id
    res = load_json(run_dir / 'results.json')
    met = res['metrics']
    smoke = met['smoke']
    key = met['eval_set']
    pw02 = ROOT / 'experiments/pw02a_active_units_amended/results' / ('pilot_001' if smoke else 'full_001')
    pw03 = ROOT / 'experiments/pw03a_fact_alignment_amended/results' / ('pilot_001' if smoke else 'full_001')
    pw04raw = (ROOT / 'experiments/pw04a_sample_legality_amended/results/pilot_001/raw' if smoke else
               ROOT / 'experiments/pw04a_sample_legality_amended/descriptive/results/pw02a_full_001/raw')
    with np.load(pw02 / 'raw/data_context.npz') as z:
        dc = {k: z[k] for k in z.files}
    w_eval = dc['w_val'] if smoke else dc['w_test']
    states = dict(bw=dc['legal'], code10=dc['code10_states'], prod5=dc['prod5_states'])
    grid = S.load_grid(pw02 / 'sweep_grid_used.json')
    cells = {c['cell_id']: c for c in grid['cells']}
    au_sets = load_json(pw02 / 'au_per_run.json')
    mism, rec = [], {}
    torch.set_num_threads(4)
    for o in met['per_run']:
        name, sub = o['name'], o['substrate']
        with np.load(pw02 / f'raw/{name}.npz') as z:
            mu, lv, w = z[f'{key}_mu'].astype(np.float64), z[f'{key}_logvar'].astype(np.float64), z[f'{key}_weights']
        X = states[sub]
        r = {}
        r['rate_bits'] = float((0.5 * (mu ** 2 + np.exp(lv) - lv - 1) * w[:, None]).sum() / LN2)
        r['H_X_bits'] = float(-(w[w > 0] * np.log(w[w > 0])).sum() / LN2)
        if sub == 'bw' and not np.allclose(w, w_eval):
            mism.append(dict(run=name, check='bw weights differ from data_context eval weights'))
        r['q7_at'] = int(((np.exp(lv) * w[:, None]).sum(0) < 0.8).sum())
        # distortion, different MC seed, hand-written Bernoulli NLL
        model = S.load_model(pw02, cells[o['cell_id']], o['seed'], grid, 36, torch.as_tensor(X, dtype=torch.float32).mean(0))
        g = torch.Generator().manual_seed(12345)
        xt = torch.as_tensor(X, dtype=torch.float64)
        mu_t, sd_t = torch.as_tensor(mu, dtype=torch.float32), torch.as_tensor(np.exp(0.5 * lv), dtype=torch.float32)
        per = []
        with torch.no_grad():
            for _ in range(128):
                logits = model.dec(mu_t + sd_t * torch.randn(mu_t.shape, generator=g)).double()
                nll = -(xt * torch.nn.functional.logsigmoid(logits) + (1 - xt) * torch.nn.functional.logsigmoid(-logits)).sum(1)
                per.append(float((nll.numpy() * w).sum()))
        r['D_nats'], r['D_se'] = float(np.mean(per)), float(np.std(per, ddof=1) / np.sqrt(128))
        D0 = 0.0
        for pf in (w[:, None] * X).sum(0):
            if 0 < pf < 1:
                D0 -= pf * math.log(pf) + (1 - pf) * math.log(1 - pf)
        r['beta_star'] = (D0 - o['q2']['D_nats']) / (r['rate_bits'] * LN2)
        checks = [('rate_bits', o['q1']['rate_bits'], r['rate_bits'], 1e-5 * abs(r['rate_bits'])),
                  ('H_X_bits', o['q1']['H_X_bits'], r['H_X_bits'], 1e-9),
                  ('q7', o['q7']['au_asperti_trentin'], r['q7_at'], 0),
                  ('beta_star', o['q5']['beta_star'], r['beta_star'], 1e-4 * abs(r['beta_star'])),
                  ('D_nats', o['q2']['D_nats'], r['D_nats'], 4 * math.hypot(r['D_se'], o['q2']['D_se_nats']))]
        for k, a, b, tol in checks:
            if abs(a - b) > tol:
                mism.append(dict(run=name, check=k, analysis=a, audit=b))
        if 'q3' in o:
            act = np.asarray(au_sets[name], dtype=int)
            B = D.block_variables(X)
            keep = w > 0
            mi = np.array([[mi_bits(bin20(mu[keep, j]), B[keep, k], w[keep] / w[keep].sum()) for k in range(5)]
                           for j in act])
            mi_an = np.array([u['mi_bits'] for u in o['q3']['units']])
            r['mi_max_abs_diff_bits'] = float(np.max(np.abs(mi - mi_an)))
            if r['mi_max_abs_diff_bits'] > 1e-6:
                mism.append(dict(run=name, check='mi_bits', diff=r['mi_max_abs_diff_bits']))
            per_block = np.bincount(mi.argmax(1), minlength=5).tolist()
            if per_block != o['q3']['units_per_block']:
                mism.append(dict(run=name, check='units_per_block', analysis=o['q3']['units_per_block'], audit=per_block))
            if sub == 'bw':
                with np.load(pw03 / f'raw/scores_{name}.npz') as s:
                    f1 = s['f1_conf']
                nf = [int((f1[j] >= 0.9).sum()) for j in act]
                cnt = {'0': nf.count(0), '1': nf.count(1), '2': nf.count(2), '3+': sum(n >= 3 for n in nf)}
                if cnt != o['q3']['n_units_by_facts']:
                    mism.append(dict(run=name, check='n_units_by_facts', analysis=o['q3']['n_units_by_facts'], audit=cnt))
            # negative control: permuted block labels
            perm = np.random.default_rng(3).permutation(int(keep.sum()))
            r['mi_perm_max_bits'] = float(max(mi_bits(bin20(mu[keep, j]), B[keep][perm, k], w[keep] / w[keep].sum())
                                              for j in act for k in range(5)))
        rec[name] = r
    # Q4 eigenvalues by SVD
    q4 = {}
    for src, row in met['q4_b_lin'].items():
        with np.load(pw02 / f'raw/activations_{src}.npz') as z:
            A = z['train_lin'].astype(np.float64)
        s = np.linalg.svd(A - A.mean(0), compute_uv=False)
        lam = s ** 2 / len(A)
        q4[src] = dict(n1=int((lam > 1).sum()), nfloor=int((lam > math.exp(-9)).sum()))
        if q4[src]['n1'] != row['n_above_gamma1'] or q4[src]['nfloor'] != row['n_above_gamma_floor']:
            mism.append(dict(source=src, check='eigen_counts', analysis=row, audit=q4[src]))
    # Q6 funnel per sample
    tasks = [(n, t, pw04raw / f'samples_{n}.npz') for n in met['legality_per_run'] for t in ('prior_mode', 'aggpost_mode')]
    q6 = {}
    with get_context('spawn').Pool(args.workers) as pool:
        for n, t, f in pool.imap_unordered(funnel_task, tasks):
            q6[f'{n}|{t}'] = f
            an = met['legality_per_run'][n][t]
            for k in ('wf_all', 'position_legal', 'legal'):
                if abs(an[k] - f[k]) > 1e-12:
                    mism.append(dict(run=n, tag=t, check=k, analysis=an[k], audit=f[k]))
            if abs(an['implied_wf_all'] - f['implied_wf_all']) > 1e-9:
                mism.append(dict(run=n, tag=t, check='implied_wf_all'))
            rate, se = f['mc_implied']
            if abs(an['implied_position_legal'] - rate) > 4 * se:
                mism.append(dict(run=n, tag=t, check='implied_position_legal_mc', analysis=an['implied_position_legal'],
                                 audit=rate, se=se))
    # regenerate one model's samples with the frozen pw04a sampler
    spec = importlib.util.spec_from_file_location('pw04a_frozen', PW04_FROZEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seed = 101 if smoke else 1
    cell = cells[REGEN_MODEL]
    name = S.run_name(REGEN_MODEL, seed)
    X = torch.as_tensor(dc['legal'], dtype=torch.float32)
    model = S.load_model(pw02, cell, seed, grid, 36, X.mean(0))
    torch.set_num_threads(1)
    prior = (mod.decode_samples(model, cell, mod.N_SAMPLES, torch.Generator().manual_seed(seed + mod.SAMPLE_SEED_OFFSET))
             .numpy() > 0.5).astype(np.int8)
    agg = (mod.decode_samples(model, cell, mod.N_SAMPLES, torch.Generator().manual_seed(seed + mod.AGG_SEED_OFFSET), X,
                              w_eval).numpy() > 0.5).astype(np.int8)
    with np.load(pw04raw / f'samples_{name}.npz') as z:
        regen = dict(prior_identical=bool(np.array_equal(np.packbits(prior, axis=1), z['prior_mode_bits'])),
                     aggpost_identical=bool(np.array_equal(np.packbits(agg, axis=1), z['aggpost_mode_bits'])))
    # negative control: uniform random bits through the funnel
    rnd = audit_funnel(np.random.default_rng(11).integers(0, 2, (20_000, 36)).astype(np.int8))
    rnd.pop('p')
    checks = dict(no_mismatches=not mism, regenerated_samples_identical=all(regen.values()),
                  negative_random_bits_illegal=rnd['legal'] == 0.0,
                  negative_permuted_mi_below_unit_mi=all(
                      r['mi_perm_max_bits'] < max(max(u['mi_bits']) for u in o['q3']['units'])
                      for o in met['per_run'] if 'q3' in o for r in [rec[o['name']]]))
    dump_json(dict(per_run=rec, q4=q4, q6=q6, regenerated=dict(model=name, **regen), negative_random=rnd,
                   mismatches=mism, checks=checks), run_dir / 'audit_recompute.json')
    ok = all(checks.values())
    if smoke:
        print(dict(audit='pass' if ok else 'fail', checks=checks, mismatches=mism[:10]))
        return
    write_manifest(run_dir)
    write_audit(run_dir, status='pass' if ok else 'fail',
                method='Separate script recomputing headline numbers by different code paths: rate in float64, H(X), '
                       'distortion with another MC seed and hand-written Bernoulli NLL (4 SE), beta*, Q7 count, MI by '
                       'explicit joint tables with hand-written 20-bin discretization, units per block, F1 fact counts '
                       'from stored pw03a matrices, b_lin eigenvalue counts by SVD, Q6 funnel per sample with the pw01 '
                       'checker and hand-written cross-block code, independence prediction by Monte Carlo (4 SE), one '
                       "model's samples regenerated with the frozen pw04a sampler (byte comparison). Negative controls: "
                       'uniform random bits through the funnel; permuted block labels for MI.',
                limitations='Same author and session (not independent replication); same retained operands as the '
                            'analysis; per-cell summaries, null percentiles, signs and pair classes not recomputed.',
                evidence=['audit_recompute.json', 'results.json'], checks=checks)
    print(dict(audit='pass' if ok else 'fail', checks=checks, mismatches=mism[:10]))


if __name__ == '__main__':
    main()
