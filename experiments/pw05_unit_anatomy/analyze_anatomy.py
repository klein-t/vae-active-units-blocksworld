"""pw05 entry point: rate, ELBO terms, unit anatomy and legality anatomy of the existing sweep (no training).

Post hoc and descriptive (PROTOCOL.md): Q1-Q6 on retained artifacts of pw02a full_001, pw03a full_001 and the
pw04a descriptive pass. Every definition is in PROTOCOL.md; this script implements it and fails loudly on any
missing file, key, run or hash mismatch.

Usage (after freeze + start --kind analysis):
    uv run python -u experiments/pw05_unit_anatomy/analyze_anatomy.py --run-id analysis_001 [--workers 8]
Smoke test on development artifacts only (pw02a/pw03a/pw04a pilot_001, seed 101, val weights), any directory:
    uv run python -u experiments/pw05_unit_anatomy/analyze_anatomy.py --smoke-out /path/to/scratch
"""
import argparse
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pwlib import blocksworld as bw  # noqa: E402
from pwlib import data as D  # noqa: E402
from pwlib import legality as LG  # noqa: E402
from pwlib import metrics as M  # noqa: E402
from pwlib import sweep as S  # noqa: E402
from pwlib.runio import dump_json, load_json, read_run, results_header, sha256  # noqa: E402

EXP = Path(__file__).resolve().parent
PW02 = ROOT / 'experiments' / 'pw02a_active_units_amended' / 'results'
PW03 = ROOT / 'experiments' / 'pw03a_fact_alignment_amended' / 'results'
PW04 = ROOT / 'experiments' / 'pw04a_sample_legality_amended'
LN2 = math.log(2)
MC_DRAWS = 128
MC_SEED = 0
F1_MIN = 0.9
N_NULL_PERM = 20
NULL_SEED = 0
LOG_GAMMA_FLOOR = -9.0
AT_INACTIVE_MIN = 0.8  # Asperti & Trentin 2002.07514 fn. 3: inactive iff mean posterior variance >= 0.8
ARCHS, LRS = ('A1', 'A2', 'A4'), ('0.0003', '0.003')
SOURCES = ('w64', 'w128', 'w128s2', 'w128s3', 'w256')
BLOCK_OF_FACT = {bw.INDEX[f]: f[f.index('(') + 1] for f in bw.FACTS
                 if f.startswith(('on_table', 'on(', 'holding'))}  # position fact -> subject block


def bits(x):
    return float(x) / LN2


def entropy_bits(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def bernoulli_entropy_nats(p):
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    out[m] = -(p[m] * np.log(p[m]) + (1 - p[m]) * np.log1p(-p[m]))
    return out


def summary(v):
    v = [x for x in v]
    return dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)), n=len(v))


def checked(path, expected):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')
    got = sha256(path)
    if got != expected:
        raise ValueError(f'Hash mismatch {path}: {got} != {expected}')
    return path


def manifest_of(run_dir):
    return {r['path']: r['sha256'] for r in load_json(Path(run_dir) / 'MANIFEST.json')['files']}


def in_manifest(run_dir, manifest, rel):
    if rel not in manifest:
        raise KeyError(f'{run_dir} manifest lacks {rel}')
    return checked(Path(run_dir) / rel, manifest[rel])


# ------------------------------------------------------------------ configuration

def config(smoke):
    if smoke:
        pw02, pw03, pw04 = PW02 / 'pilot_001', PW03 / 'pilot_001', PW04 / 'results' / 'pilot_001'
        return dict(smoke=True, pw02=pw02, pw03=pw03, pw04_raw=pw04 / 'raw', pw04_hashes={
            k: v for k, v in manifest_of(pw04).items() if k.startswith('raw/')}, key='dev', fit_key='dev',
            f1_weights=('w_train', 'w_val'))
    hashes = load_json(EXP / 'input_hashes.json')
    return dict(smoke=False, pw02=PW02 / 'full_001', pw03=PW03 / 'full_001',
                pw04_raw=PW04 / 'descriptive' / 'results' / 'pw02a_full_001' / 'raw',
                pw04_hashes=hashes['pw04a_descriptive_samples'], key='conf', fit_key='dev',
                f1_weights=('w_val', 'w_test'))


def planned(grid, cfg):
    """{set name: [(cell, seed)]} for R1, C (code10, prod5), R025, R4, Bl; smoke keeps pilot cells only."""
    cells = {c['cell_id']: c for c in grid['cells']}

    def runs(ids):
        out = []
        for cid in ids:
            if cid not in cells:
                raise KeyError(f'cell {cid} not in grid')
            if cfg['smoke']:
                if cid in grid['pilot_cells']:
                    out += [(cells[cid], s) for s in grid['seeds']['pilot']]
            else:
                out += [(cells[cid], s) for s in cells[cid]['seeds']]
        return out
    bwc = lambda b: [f'bw-gauss_vae-{a}-b{b}-lr{lr}' for a in ARCHS for lr in LRS]  # noqa: E731
    return dict(R1=runs(bwc(1)), code10=runs([f'code10-gauss_vae-{a}-b1-lr0.0003' for a in ARCHS]),
                prod5=runs([f'prod5-gauss_vae-{a}-b1-lr0.0003' for a in ARCHS]), R025=runs(bwc(0.25)),
                R4=runs(bwc(4)),
                Bl=runs([f'b_lin_{s}-gauss_vae-A2-b1-g{g}-lr0.0003' for s in SOURCES for g in ('fixed', 'learned')]))


# ------------------------------------------------------------------ per-run work

def mc_distortion(model, mu, logvar, X, w, draws=MC_DRAWS, seed=MC_SEED):
    """Q2: D = sum_x w(x) E_eps[-log p(x | mu + exp(logvar/2) eps)]; returns (D, MC SE) in nats."""
    g = torch.Generator().manual_seed(seed)
    mu_t, sd_t = torch.as_tensor(mu), torch.as_tensor(np.exp(0.5 * logvar.astype(np.float64)).astype(np.float32))
    x = torch.as_tensor(X, dtype=torch.float32)
    per = []
    with torch.no_grad():
        for _ in range(draws):
            z = mu_t + sd_t * torch.randn(mu_t.shape, generator=g)
            nll = model.dec.nll(x, model.dec(z)).double().numpy()
            per.append(float((nll * w).sum()))
    per = np.array(per)
    return float(per.mean()), float(per.std(ddof=1) / np.sqrt(draws))


def unit_block_mi(codes, B, w):
    """Q3: weighted plug-in MI (nats) [units, 5] after 20 equal-width bins; H(block) (nats)."""
    keep = w > 0
    codes, B, w = codes[keep], B[keep], w[keep] / w[keep].sum()
    inv_b = [np.unique(B[:, k], return_inverse=True)[1] for k in range(B.shape[1])]
    mi = np.zeros((codes.shape[1], B.shape[1]))
    for j in range(codes.shape[1]):
        a = np.unique(M.histogram_discretize(codes[:, j]), return_inverse=True)[1]
        mi[j] = [M._weighted_mi(a, b, w) for b in inv_b]
    H = np.array([M._weighted_mi(b, b, w) for b in inv_b])
    return mi, H, (codes, inv_b, w)


def null_max_mi(codes, B, w, active, n_perm=N_NULL_PERM, seed=NULL_SEED):
    """Q3 null: block labels permuted across rows; 95th percentile of per-unit max MI (bits)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        mi, _, _ = unit_block_mi(codes[:, active], B[rng.permutation(len(B))], w)
        vals += list(mi.max(1) / LN2)
    return float(np.percentile(vals, 95)) if vals else None


def fact_anatomy(f1, sign, active, mi_block):
    """Q3: per active unit, facts at F1 >= 0.9, signs, pair class, agreement with the max-MI block."""
    units = []
    for r, j in enumerate(np.flatnonzero(active)):
        facts = [int(f) for f in np.flatnonzero(f1[j] >= F1_MIN)]
        blocks = [BLOCK_OF_FACT.get(f) for f in facts]
        u = dict(unit=int(j), n_facts=len(facts), facts=[bw.FACTS[f] for f in facts],
                 signs=[int(sign[j, f]) for f in facts], max_mi_block=bw.BLOCKS[int(mi_block[r])])
        pos_blocks = {b for b in blocks if b is not None}
        u['fact_block_matches_mi_block'] = None if not pos_blocks else bool(pos_blocks == {u['max_mi_block']})
        if len(facts) == 2:
            u['pair'] = 'derived' if None in blocks else ('same_block' if blocks[0] == blocks[1] else 'cross_block')
            u['pair_opposite_signs'] = bool(u['signs'][0] != u['signs'][1])
            sup = [f[f.index(',') + 1] if f.startswith('on(') else None for f in u['facts']]
            u['pair_same_support'] = bool(sup[0] is not None and sup[0] == sup[1])  # on(x,y), on(x',y)
        units.append(u)
    return units


def model_task(task):
    """Q1, Q2 (+ Q3 for R1/prod5) for one beta=1 raw-state run."""
    torch.set_num_threads(1)
    t0 = time.time()
    cfg, cell, seed, ctx = task['cfg'], task['cell'], task['seed'], task['ctx']
    sub, name, key = cell['substrate'], S.run_name(cell['cell_id'], task['seed']), cfg['key']
    pw02, man02 = cfg['pw02'], task['man02']
    with np.load(in_manifest(pw02, man02, f'raw/{name}.npz')) as z:
        need = [f'{key}_mu', f'{key}_logvar', f'{key}_weights', f'{key}_kl_per_unit',
                f'{cfg["fit_key"]}_mu', f'{cfg["fit_key"]}_weights']
        missing = [k for k in need if k not in z.files]
        if missing:
            raise KeyError(f'{name}: missing keys {missing}')
        mu, lv, w, klu = (z[f'{key}_{k}'] for k in ('mu', 'logvar', 'weights', 'kl_per_unit'))
        mu_fit, w_fit = z[f'{cfg["fit_key"]}_mu'], z[f'{cfg["fit_key"]}_weights']
    X = ctx['states'][sub]
    active = np.isin(np.arange(mu.shape[1]), task['active_ids'])
    # numerical: stored per-unit KL against recomputation from mu / logvar (same float32 expression as pw02a)
    mt, lt = torch.as_tensor(mu), torch.as_tensor(lv)
    kl_re = ((0.5 * (mt ** 2 + lt.exp() - lt - 1)).double().numpy() * w[:, None]).sum(0)
    l64 = lv.astype(np.float64)
    kl_rel = float(np.max(np.abs(kl_re - klu) / np.maximum(np.abs(klu), 1e-12)))
    R = float(klu.sum())
    HX = entropy_bits(w)
    # Q7: Asperti & Trentin criterion, inactive iff E_x sigma^2 >= 0.8
    mean_s2 = (np.exp(l64) * w[:, None]).sum(0)
    at_active = mean_s2 < AT_INACTIVE_MIN
    out = dict(name=name, cell_id=cell['cell_id'], seed=seed, substrate=sub, arch=cell['arch'], lr=cell['lr'],
               n_active=int(active.sum()),
               q1=dict(kl_per_unit_bits=[bits(v) for v in klu], rate_bits=bits(R),
                       rate_active_bits=bits(klu[active].sum()), rate_inactive_bits=bits(klu[~active].sum()),
                       H_X_bits=HX, mismatch_bound_bits=bits(R) - HX, kl_recompute_max_rel=kl_rel),
               q7=dict(au_asperti_trentin=int(at_active.sum()), au_burda=int(active.sum()),
                       sym_diff=int((at_active ^ active).sum()), mean_sigma2=mean_s2.tolist()))
    # Q2: distortion by Monte Carlo through the retained decoder
    model = S.load_model(pw02, cell, seed, task['grid'], 36, torch.as_tensor(X, dtype=torch.float32).mean(0))
    in_manifest(pw02, man02, f'models/{name}.pt')
    Dn, se = mc_distortion(model, mu, lv, X, w)
    out['q2'] = dict(D_nats=Dn, D_se_nats=se, R_nats=R, neg_elbo_nats=Dn + R, D_bits=bits(Dn), R_bits=bits(R),
                     neg_elbo_bits=bits(Dn + R), slack_nats=Dn + R - HX * LN2, slack_bits=bits(Dn + R) - HX,
                     gibbs_bound_ok=bool(Dn + R >= HX * LN2 - 3 * se))
    # Q5 operand: zero-rate distortion
    D0 = float(bernoulli_entropy_nats((w[:, None] * X).sum(0)).sum())
    out['q5'] = dict(D0_nats=D0, beta_star=(D0 - Dn) / R)
    # Q3 for Blocksworld and prod5
    if sub in ('bw', 'prod5'):
        B = D.block_variables(X)
        mi, H, _ = unit_block_mi(mu[:, active].astype(np.float64), B, w)
        assign = mi.argmax(1)
        ties = int(sum((np.isclose(row, row.max(), rtol=0, atol=1e-12)).sum() > 1 for row in mi))
        if sub == 'bw':
            with np.load(in_manifest(cfg['pw03'], task['man03'], f'raw/scores_{name}.npz')) as s:
                for k in ('f1_conf', 'sign', 'active', 'block_mi'):
                    if k not in s.files:
                        raise KeyError(f'pw03a scores_{name}: missing {k}')
                f1, sgn, act03, bmi03 = s['f1_conf'], s['sign'], s['active'], s['block_mi']
            wd, wc = (ctx['weights'][k] for k in cfg['f1_weights'])
            re = M.best_f1(mu_fit.astype(np.float64), X, mu_fit.astype(np.float64), X, wd, wc)
            num = dict(f1_reproduced_max_abs=float(np.max(np.abs(re['f1_conf'] - f1))),
                       pw03_active_equal=bool(np.array_equal(act03, active)),
                       mi_vs_pw03_max_abs=float(np.max(np.abs(mi - bmi03))) if bmi03.shape == mi.shape else None)
        else:
            bf = M.best_f1(mu_fit.astype(np.float64), X, mu.astype(np.float64), X, w_fit, w)
            f1, sgn = bf['f1_conf'], bf['sign']
            num = {}
        units = fact_anatomy(f1, sgn, active, assign)
        per_block = np.bincount(assign, minlength=5)
        for u, row, h in zip(units, mi, [H] * len(units)):
            u['mi_bits'] = [bits(v) for v in row]
            u['nmi'] = [float(v / hh) for v, hh in zip(row, h)]
        n_facts = [u['n_facts'] for u in units]
        pairs = [u for u in units if u['n_facts'] == 2]
        out['q3'] = dict(units=units, units_per_block=per_block.tolist(),
                         units_per_block_sorted=sorted(per_block.tolist(), reverse=True), ties=ties,
                         H_block_bits=[bits(h) for h in H],
                         n_units_by_facts={'0': n_facts.count(0), '1': n_facts.count(1), '2': n_facts.count(2),
                                           '3+': sum(n >= 3 for n in n_facts)},
                         pairs=dict(n=len(pairs), same_block=sum(u['pair'] == 'same_block' for u in pairs),
                                    cross_block=sum(u['pair'] == 'cross_block' for u in pairs),
                                    derived=sum(u['pair'] == 'derived' for u in pairs),
                                    opposite_signs=sum(u['pair_opposite_signs'] for u in pairs),
                                    same_support=sum(u['pair_same_support'] for u in pairs)),
                         fact_block_matches_mi_block=dict(
                             yes=sum(u['fact_block_matches_mi_block'] is True for u in units),
                             no=sum(u['fact_block_matches_mi_block'] is False for u in units),
                             no_position_fact=sum(u['fact_block_matches_mi_block'] is None for u in units)),
                         null_p95_max_mi_bits=null_max_mi(mu.astype(np.float64), B, w, active),
                         numerical=num)
    out['seconds'] = time.time() - t0
    return out


def unpack(packed):
    return np.unpackbits(packed, axis=1)[:, :bw.N_FACTS].astype(np.int8)


def funnel(Sb, legal_flags, index, w=None):
    """Q6 legality anatomy of 36-bit rows (optional row weights for the data control)."""
    w = np.full(len(Sb), 1.0 / len(Sb)) if w is None else w / w.sum()
    wf = np.zeros((len(Sb), 5), dtype=bool)
    p = np.zeros((5, 6))
    for b, x in enumerate(bw.BLOCKS):
        cols = [bw.INDEX[f'on_table({x})'] if v == 'T' else bw.INDEX[f'holding({x})'] if v == 'H'
                else bw.INDEX[f'on({x},{v})'] for v in bw.POSITION_VALUES[x]]
        blk = Sb[:, cols]
        wf[:, b] = blk.sum(1) == 1
        np.add.at(p[b], blk[wf[:, b]].argmax(1), w[wf[:, b]])
    wf_all = wf.all(1)
    cross = LG.legal_from_position_bits(Sb)
    member = index.lookup(Sb) >= 0
    V = LG.violations(Sb)
    num = dict(member_equals_stored=bool(np.array_equal(member, legal_flags)),
               legal_subset_cross=bool(not (member & ~cross).any()), cross_subset_wf=bool(not (cross & ~wf_all).any()),
               v1_equals_not_wf=bool(np.array_equal(V[:, 0], ~wf_all)))
    r = lambda m: float((w * m).sum())  # noqa: E731
    WF, CR, LE = r(wf_all), r(cross), r(member)
    implied_pos = LG.baseline_b2(p)
    implied_wf = float(np.prod(wf.T @ w))
    num['funnel_product_equals_rate'] = bool(abs((WF * (CR / WF if WF else 0) * (LE / CR if CR else 0)) - LE) < 1e-12)
    return dict(wf_per_block={x: r(wf[:, b]) for b, x in enumerate(bw.BLOCKS)}, wf_all=WF,
                cross_given_wf=CR / WF if WF else None, derived_given_cross=LE / CR if CR else None,
                position_legal=CR, legal=LE, implied_wf_all=implied_wf, implied_position_legal=implied_pos,
                implied_cross_given_wf=implied_pos / implied_wf if implied_wf else None,
                observed_over_implied_position_legal=CR / implied_pos if implied_pos else None,
                violations={v: r(V[:, i]) for i, v in enumerate(LG.VIOLATIONS)}, numerical=num)


def legality_task(task):
    cfg, name = task['cfg'], task['name']
    index = D.LegalIndex(task['legal'])
    rel = f'samples_{name}.npz'
    if rel not in cfg['pw04_hashes'] and f'raw/{rel}' not in cfg['pw04_hashes']:
        raise KeyError(f'No recorded hash for pw04a {rel}')
    h = cfg['pw04_hashes'].get(rel, cfg['pw04_hashes'].get(f'raw/{rel}'))
    out = dict(name=name)
    with np.load(checked(cfg['pw04_raw'] / rel, h)) as z:
        for tag in ('prior_mode', 'aggpost_mode'):
            for k in (f'{tag}_bits', f'{tag}_legal'):
                if k not in z.files:
                    raise KeyError(f'{rel}: missing {k}')
            out[tag] = funnel(unpack(z[f'{tag}_bits']), z[f'{tag}_legal'], index)
    return out


# ------------------------------------------------------------------ Q4

def q4(cfg, man02, au):
    rows = {}
    for src in SOURCES:
        fixed = [r for r in au.values() if r['cell_id'] == f'b_lin_{src}-gauss_vae-A2-b1-gfixed-lr0.0003']
        learned = [r for r in au.values() if r['cell_id'] == f'b_lin_{src}-gauss_vae-A2-b1-glearned-lr0.0003']
        if not fixed and not learned:
            continue
        with np.load(in_manifest(cfg['pw02'], man02, f'raw/activations_{src}.npz')) as z:
            if 'train_lin' not in z.files:
                raise KeyError(f'activations_{src}: missing train_lin')
            A = z['train_lin'].astype(np.float64)
        Ac = A - A.mean(0)
        C = Ac.T @ Ac / len(A)
        lam = np.sort(np.linalg.eigh(C)[0])[::-1]
        lg = [max(r['log_gamma'], LOG_GAMMA_FLOOR) for r in learned]
        rows[src] = dict(eigenvalues_top30=lam[:30].tolist(), trace=float(np.trace(C)),
                         eig_sum_rel_err=float(abs(lam.sum() - np.trace(C)) / np.trace(C)),
                         numerical_rank=int((lam > lam[0] * 1e-10).sum()),
                         n_above_gamma1=int((lam > 1.0).sum()), n_above_gamma_floor=int((lam > math.exp(LOG_GAMMA_FLOOR)).sum()),
                         n_above_learned_gamma=[int((lam > math.exp(g)).sum()) for g in lg],
                         learned_log_gamma_effective=lg,
                         au_fixed=[r['au'] for r in sorted(fixed, key=lambda r: r['seed'])],
                         au_learned=[r['au'] for r in sorted(learned, key=lambda r: r['seed'])])
    return rows


# ------------------------------------------------------------------ main

def per_cell(rows, getter):
    cells = {}
    for o in rows:
        cells.setdefault(o['cell_id'], []).append(getter(o))
    return {c: summary(v) for c, v in sorted(cells.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id')
    ap.add_argument('--smoke-out', type=Path, help='SMOKE ONLY: development artifacts (pilot_001), scratch output')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    if (args.run_id is None) == (args.smoke_out is None):
        raise SystemExit('Give exactly one of --run-id or --smoke-out')
    cfg = config(args.smoke_out is not None)
    if cfg['smoke']:
        out_dir = args.smoke_out
        out_dir.mkdir(parents=True, exist_ok=False)
        header = dict(experiment_id='pw05_unit_anatomy', run_id='SMOKE', freeze_sha256='smoke-unfrozen')
    else:
        out_dir = EXP / 'results' / args.run_id
        if read_run(out_dir)['kind'] != 'analysis':
            raise ValueError('pw05 runs are analysis runs')
        header = results_header(out_dir)
    t0 = time.time()
    pw02 = cfg['pw02']
    man02, man03 = manifest_of(pw02), manifest_of(cfg['pw03'])
    for d in (pw02, cfg['pw03']):
        res, aud = load_json(d / 'results.json'), load_json(d / 'AUDIT.json')
        if res['status'] != 'completed' or aud['status'] != 'pass':
            raise ValueError(f'{d} is not a completed, audited run')
    grid = S.load_grid(in_manifest(pw02, man02, 'sweep_grid_used.json'))
    au = {r['run']: r for r in load_json(in_manifest(pw02, man02, 'au_table.json'))}
    active_sets = load_json(in_manifest(pw02, man02, 'au_per_run.json'))
    if active_sets.get('_eval_set') != cfg['key']:
        raise ValueError(f'pw02a AU sets were computed on {active_sets.get("_eval_set")}, expected {cfg["key"]}')
    with np.load(in_manifest(pw02, man02, 'raw/data_context.npz')) as z:
        dc = {k: z[k] for k in z.files}
    legal = dc['legal']
    plan = planned(grid, cfg)
    w_eval = dc['w_val'] if cfg['smoke'] else dc['w_test']
    ctx = dict(states=dict(bw=legal, code10=dc['code10_states'], prod5=dc['prod5_states']),
               weights={k: dc[k] for k in ('w_train', 'w_val') + (() if cfg['smoke'] else ('w_test',))})
    # task gate
    task = dict(legal_table_reproduced=bool(np.array_equal(bw.enumerate_legal_bruteforce(), legal)),
                pw03_scores_listed=all(f'raw/scores_{S.run_name(c["cell_id"], s)}.npz' in man03 for c, s in plan['R1']),
                runs_present=all(S.run_name(c['cell_id'], s) in au for v in plan.values() for c, s in v),
                planned_counts={k: len(v) for k, v in plan.items()})
    if not task['runs_present']:
        raise KeyError('Planned runs missing from the pw02a AU table')
    comp_sets = ('R1', 'code10', 'prod5', 'R025', 'Bl')
    comp = {k: all(au[S.run_name(c['cell_id'], s)]['status'] == 'ok' and au[S.run_name(c['cell_id'], s)]['competence']
                   for c, s in plan[k]) for k in comp_sets}
    # per-run work
    mtasks = [dict(cfg=cfg, cell=c, seed=s, ctx=ctx, grid=grid, man02=man02, man03=man03,
                   active_ids=active_sets[S.run_name(c['cell_id'], s)])
              for k in ('R1', 'code10', 'prod5') for c, s in plan[k]]
    ltasks = [dict(cfg=cfg, name=S.run_name(c['cell_id'], s), legal=legal)
              for k in ('R1', 'R025', 'prod5') for c, s in plan[k]]
    with get_context('spawn').Pool(args.workers) as pool:
        mres = []
        for o in pool.imap_unordered(model_task, mtasks):
            mres.append(o)
            print(f'[model {len(mres)}/{len(mtasks)}] {o["name"]} {o["seconds"]:.1f}s', flush=True)
        lres = {o['name']: o for o in pool.imap_unordered(legality_task, ltasks)}
    mres.sort(key=lambda o: (o['substrate'], o['cell_id'], o['seed']))
    index = D.LegalIndex(legal)
    data_control = funnel(legal, np.ones(len(legal), dtype=bool), index, w_eval)
    data_control['B2_eval_marginals'] = LG.baseline_b2(LG.block_marginals(legal, w_eval))
    q4rows = q4(cfg, man02, au)
    # beta=4 observed (Q5 comparison): AU and rate from stored operands
    r4 = []
    for c, s in plan['R4']:
        name = S.run_name(c['cell_id'], s)
        with np.load(in_manifest(pw02, man02, f'raw/{name}.npz')) as z:
            r4.append(dict(name=name, cell_id=c['cell_id'], au=au[name]['au'],
                           rate_bits=bits(z[f'{cfg["key"]}_kl_per_unit'].sum())))
    # gates
    num = dict(
        kl_recompute=all(o['q1']['kl_recompute_max_rel'] <= 1e-6 for o in mres),
        gibbs_elbo_bound=all(o['q2']['gibbs_bound_ok'] for o in mres),
        f1_reproduces_pw03=all(o['q3']['numerical']['f1_reproduced_max_abs'] <= 1e-12 for o in mres
                               if o['substrate'] == 'bw'),
        pw03_active_equal=all(o['q3']['numerical']['pw03_active_equal'] for o in mres if o['substrate'] == 'bw'),
        mi_reproduces_pw03=all(o['q3']['numerical']['mi_vs_pw03_max_abs'] is not None and
                               o['q3']['numerical']['mi_vs_pw03_max_abs'] <= 1e-9 for o in mres if o['substrate'] == 'bw'),
        eig_sum_trace=all(r['eig_sum_rel_err'] <= 1e-8 for r in q4rows.values()),
        legality_funnel=all(all(f[t]['numerical'].values()) for f in lres.values() for t in ('prior_mode', 'aggpost_mode'))
        and all(data_control['numerical'].values()))
    gates = dict(task=bool(task['legal_table_reproduced'] and task['pw03_scores_listed'] and task['runs_present']),
                 numerical=bool(all(num.values())), competence=bool(all(comp.values())))
    # criteria: completeness of each quantity over its planned runs
    names = {k: {S.run_name(c['cell_id'], s) for c, s in v} for k, v in plan.items()}
    got = {o['name'] for o in mres}
    q3_names = {o['name'] for o in mres if 'q3' in o}
    criteria = dict(
        q1_computed=bool(names['R1'] | names['code10'] | names['prod5'] <= got),
        q2_computed=bool(all('q2' in o for o in mres) and names['R1'] | names['code10'] | names['prod5'] <= got),
        q3_computed=bool(names['R1'] | names['prod5'] <= q3_names),
        q4_computed=bool(set(q4rows) == ({'w128'} if cfg['smoke'] else set(SOURCES))),
        q5_computed=bool(all('q5' in o for o in mres) and len(r4) == len(plan['R4'])),
        q6_computed=bool(names['R1'] | names['R025'] | names['prod5'] <= set(lres)),
        q7_computed=bool(all('q7' in o for o in mres) and names['R1'] | names['code10'] | names['prod5'] <= got))
    # per-cell summaries
    cell_rows = [o for o in mres]
    summaries = dict(
        q1_rate_bits=per_cell(cell_rows, lambda o: o['q1']['rate_bits']),
        q1_mismatch_bound_bits=per_cell(cell_rows, lambda o: o['q1']['mismatch_bound_bits']),
        q1_rate_active_bits=per_cell(cell_rows, lambda o: o['q1']['rate_active_bits']),
        q1_H_X_bits=per_cell(cell_rows, lambda o: o['q1']['H_X_bits']),
        n_active=per_cell(cell_rows, lambda o: o['n_active']),
        q7_au_asperti_trentin=per_cell(cell_rows, lambda o: o['q7']['au_asperti_trentin']),
        q7_sym_diff=per_cell(cell_rows, lambda o: o['q7']['sym_diff']),
        q2_D_nats=per_cell(cell_rows, lambda o: o['q2']['D_nats']),
        q2_R_nats=per_cell(cell_rows, lambda o: o['q2']['R_nats']),
        q2_neg_elbo_nats=per_cell(cell_rows, lambda o: o['q2']['neg_elbo_nats']),
        q2_slack_bits=per_cell(cell_rows, lambda o: o['q2']['slack_bits']),
        q5_beta_star=per_cell(cell_rows, lambda o: o['q5']['beta_star']),
        q5_D0_nats=per_cell(cell_rows, lambda o: o['q5']['D0_nats']),
        q5_beta4_au=per_cell(r4, lambda o: o['au']), q5_beta4_rate_bits=per_cell(r4, lambda o: o['rate_bits']))
    q3_rows = [o for o in mres if 'q3' in o]
    summaries['q3'] = {}
    for cid in sorted({o['cell_id'] for o in q3_rows}):
        rs = [o['q3'] for o in q3_rows if o['cell_id'] == cid]
        summaries['q3'][cid] = dict(
            units_per_block_sorted_per_seed=[r['units_per_block_sorted'] for r in rs],
            n_units_by_facts_total={k: sum(r['n_units_by_facts'][k] for r in rs) for k in ('0', '1', '2', '3+')},
            pairs_total={k: sum(r['pairs'][k] for r in rs) for k in ('n', 'same_block', 'cross_block', 'derived',
                                                                     'opposite_signs', 'same_support')},
            fact_block_matches_mi_block_total={k: sum(r['fact_block_matches_mi_block'][k] for r in rs)
                                               for k in ('yes', 'no', 'no_position_fact')},
            max_nmi_median=float(np.median([max(u['nmi']) for r in rs for u in r['units']])),
            second_nmi_median=float(np.median([sorted(u['nmi'])[-2] for r in rs for u in r['units']])),
            null_p95_max_mi_bits_max=float(max(r['null_p95_max_mi_bits'] for r in rs)),
            ties_total=sum(r['ties'] for r in rs))
    cell_of = {S.run_name(c['cell_id'], s): c['cell_id'] for k in ('R1', 'R025', 'prod5') for c, s in plan[k]}
    summaries['q6'] = {}
    for cid in sorted(set(cell_of.values())):
        fs = [f for n, f in lres.items() if cell_of[n] == cid]
        summaries['q6'][cid] = {t: {k: summary([f[t][k] for f in fs]) for k in (
            'wf_all', 'cross_given_wf', 'derived_given_cross', 'legal', 'implied_wf_all', 'implied_position_legal',
            'implied_cross_given_wf', 'position_legal', 'observed_over_implied_position_legal')}
            for t in ('prior_mode', 'aggpost_mode')}
    metrics = dict(per_run=mres, legality_per_run=lres, legality_data_control=data_control, q4_b_lin=q4rows,
                   beta4_observed=r4, summaries=summaries, task_checks=task, numerical_checks=num,
                   competence_sets=comp, wall_seconds=time.time() - t0, eval_set=cfg['key'],
                   mc=dict(draws=MC_DRAWS, seed=MC_SEED), smoke=cfg['smoke'])
    if not (gates['task'] and gates['numerical']):
        status, outcome = 'technical_stop', 'untested'
    elif not gates['competence']:
        status, outcome = 'competence_stop', 'untested'
    else:
        status = 'completed'
        outcome = 'supported' if all(criteria.values()) else 'inconclusive'
    interp = ('Post hoc, descriptive quantities Q1-Q6 on artifacts whose confirmation data were already opened; '
              'no hypothesis tested. Outcome refers only to the completeness criteria (every pre-declared quantity '
              'computed for every planned run). Scoped to this sweep, encoding and test-file weights.')
    res = header | dict(status=status, outcome=outcome, interpretation=interp, confirmation_accessed=True,
                        claims_changed=False, gates=gates, criteria=criteria, metrics=metrics)
    dump_json(res, out_dir / 'results.json')
    print(dict(status=status, outcome=outcome, gates=gates, criteria=criteria,
               seconds=round(time.time() - t0, 1)))


if __name__ == '__main__':
    main()
