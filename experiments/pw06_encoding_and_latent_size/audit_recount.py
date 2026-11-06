"""pw06 independent recount (DESIGN.md "Analysis and checks": a separately written recount from the saved
posterior arrays).

Written from DESIGN.md, README.md and the array/JSON formats only. It does not import or reuse
analyze_pw06.py, run_pw06.py or pwlib's active-unit / convergence code.

Steps
1. Hash every file listed in results/full_001/MANIFEST.json and the frozen DESIGN.md; refuse on mismatch.
2. Test-split weights from pw01's state_frequency_table.csv (count_test), matched to the encoded states by
   their 36-bit string; compared with the stored conf_weights and data_context w_test. code10: uniform 1/1024.
3. Encodings re-derived from the definitions: E30 = E36 minus 6 columns that are functions of the others;
   E10a/E10b = 10-bit MSB-first of perm[i], perm = default_rng(seed).permutation(866); code10 c copies of
   each of 10 fair bits, other columns constant zero, all 1,024 bit patterns.
4. Per run (test split = 'conf'):
   - Burda count: weighted Var_x(mu_j) > 0.01 (also 0.001, 0.05, 0.1); explicit two-pass weighted variance;
     plain AE: the same on the deterministic codes (covariance count).
   - Own forward pass from the checkpoint: encoder mean vs stored mu; decoder(mu) logits > 0 vs stored
     reconstruction bits.
   - Exact-state and bitwise reconstruction, frequency weighted, from the reconstruction bits vs the states.
   - Numerical check: run status ok, finite stored objective, finite mu, per-unit KL >= 0.
   - Rate: analytic weighted KL from mu/logvar, compared with the stored per-unit KL sum.
   - Distortion: own 128-draw Monte Carlo of -log p(x | mu + sigma eps) (different RNG; compared within a
     tolerance, it is a Monte Carlo quantity); -ELBO = D + R; H(X) and R - H(X) on the test weights.
   - Convergence: from the logged curve, |obj(final) - obj(step nearest 80%)| / |obj(80%)| < 0.02
     (plain AE also accepted below 1e-3 nats absolute, pw02a rule).
5. Gates and verdicts P1, P2, P3, S1, S2 from DESIGN.md's decision rules, with all runs and without
   non-converged runs. Compare with results.json; write audit_recompute.json and AUDIT.json.

Usage: .venv/bin/python experiments/pw06_encoding_and_latent_size/audit_recount.py
"""
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
RUN = EXP / 'results' / 'full_001'
PW01_TABLE = ROOT / 'experiments/pw01_state_space_ground_truth/results/analysis_001/state_frequency_table.csv'
FREEZE = '904ea2366846496c08b6c68668861d0f93677f06cf232c546967104349a1b129'
THRESHOLDS = (0.001, 0.01, 0.05, 0.1)
torch.set_num_threads(4)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 1. hashes
def check_hashes():
    man = json.loads((RUN / 'MANIFEST.json').read_text())['files']
    bad = [f['path'] for f in man if sha256(RUN / f['path']) != f['sha256'] or (RUN / f['path']).stat().st_size != f['bytes']]
    design_ok = sha256(EXP / 'DESIGN.md') == FREEZE
    run_json = json.loads((RUN / 'RUN.json').read_text())
    return dict(n_files=len(man), mismatched=bad, design_sha_ok=design_ok,
                run_json_freeze_ok=run_json['freeze_sha256'] == FREEZE), {f['path'] for f in man}


# ---------------------------------------------------------------- cell map (from DESIGN.md "Data and cells")
def cell_of(run):
    cid, seed = run.rsplit('_s', 1)
    seed = int(seed)
    if cid == 'bw-gauss_vae-A2-b1-lr0.0003':
        return 'E36-L32', seed
    if cid.startswith('bw-gauss_vae-A2-b1-lr0.0003-L'):
        return 'E36-L' + cid.rsplit('-L', 1)[1], seed
    table = {'e30-gauss_vae-A2-b1-lr0.0003': 'E30-L32', 'e10a-gauss_vae-A2-b1-lr0.0003': 'E10a-L32',
             'e10b-gauss_vae-A2-b1-lr0.0003': 'E10b-L32', 'e10a-plain_ae-A2-lr0.0003': 'E10a-AE',
             'e10a-gauss_vae-A2-b0.25-lr0.0003': 'E10a-b0.25'}
    if cid in table:
        return table[cid], seed
    if cid.startswith('code10c'):
        return f'code10-c{cid[7]}', seed
    raise KeyError(cid)


def expected_runs():
    cells = ['E36-L32', 'E30-L32', 'E10a-L32', 'E10b-L32', 'E10a-b0.25'] + [f'E36-L{L}' for L in (8, 10, 12, 16, 24, 64)] \
        + [f'code10-c{c}' for c in (1, 2, 3, 4)]
    return {(c, s) for c in cells for s in range(1, 6)} | {('E10a-AE', s) for s in (1, 2, 3)}


# ---------------------------------------------------------------- 2-3. data
def bits10(v):
    return np.array([(v >> (9 - k)) & 1 for k in range(10)], dtype=np.int8)


def data_checks(ctx):
    out = {}
    rows = list(csv.DictReader(open(PW01_TABLE)))
    by_bits = {r['bits']: r for r in rows}
    bw = ctx['states_bw']
    keys = [''.join(map(str, r.tolist())) for r in bw]
    assert len(set(keys)) == 866 and all(k in by_bits for k in keys), 'states do not match pw01 table'
    ct = np.array([int(by_bits[k]['count_test']) for k in keys], dtype=np.float64)
    w_test = ct / ct.sum()
    out['pw01_state_index_order'] = bool(all(int(by_bits[k]['state_index']) == i for i, k in enumerate(keys)))
    out['w_test_vs_context_maxabs'] = float(np.abs(w_test - ctx['w_test']).max())
    # E30: 30 of the 36 columns; the 6 dropped ones are deterministic functions of the 30.
    e30 = ctx['states_e30']
    col_idx = []
    for j in range(30):
        m = [i for i in range(36) if np.array_equal(bw[:, i], e30[:, j]) and i not in col_idx]
        col_idx.append(m[0] if m else -1)
    dropped = [i for i in range(36) if i not in col_idx]
    key30 = [e30[i].tobytes() for i in range(866)]
    groups = {}
    for i, k in enumerate(key30):
        groups.setdefault(k, []).append(i)
    func = all(len({tuple(bw[i, dropped]) for i in g}) == 1 for g in groups.values())
    out['e30'] = dict(columns_found=bool(-1 not in col_idx and len(set(col_idx)) == 30), dropped_cols=dropped,
                      dropped_are_functions=bool(func), rows_distinct=len(groups) == 866)
    for name, seed in (('e10a', 0), ('e10b', 1)):
        perm = np.random.default_rng(seed).permutation(866)
        mine = np.stack([bits10(int(perm[i])) for i in range(866)])
        out[name] = bool(np.array_equal(mine, ctx[f'states_{name}']))
    for c in (1, 2, 3, 4):
        S = ctx[f'states_code10c{c}']
        zero = [j for j in range(40) if not S[:, j].any()]
        live = [j for j in range(40) if S[:, j].any()]
        uniq = {S[:, j].tobytes() for j in live}
        copies = sorted(sum(S[:, j].tobytes() == u for j in live) for u in uniq)
        allpat = len({r.tobytes() for r in S[:, live]}) == 1024
        out[f'code10c{c}'] = bool(len(zero) == 40 - 10 * c and len(uniq) == 10 and copies == [c] * 10 and allpat
                                  and all(abs(S[:, j].mean() - 0.5) < 1e-12 for j in live))
    return out, w_test


# ---------------------------------------------------------------- 4. per run
def wvar_loop(a, w):
    """Explicit two-pass weighted variance per column."""
    w = np.asarray(w, np.float64)
    w = w / w.sum()
    a = np.asarray(a, np.float64)
    out = np.zeros(a.shape[1])
    for j in range(a.shape[1]):
        m = float(np.sum(w * a[:, j]))
        out[j] = float(np.sum(w * (a[:, j] - m) ** 2))
    return out


def mlp(sd, prefix, x):
    idx = sorted({int(k[len(prefix):].split('.')[0]) for k in sd if k.startswith(prefix) and k.endswith('.weight')})
    h = x
    for n, i in enumerate(idx):
        h = h @ sd[f'{prefix}{i}.weight'].T + sd[f'{prefix}{i}.bias']
        if n < len(idx) - 1:
            h = torch.relu(h)
    return h


def converged(curve, plain):
    last = curve[-1]
    ref = min(curve, key=lambda r: abs(r['step'] - 0.8 * last['step']))
    a, b = ref['dev_objective'], last['dev_objective']
    ok = abs(b - a) / abs(a) < 0.02
    if plain:
        ok = ok or abs(b - a) < 1e-3
    return bool(ok), float(abs(b - a) / abs(a))


@torch.no_grad()
def per_run(run, states, w_ref, gen):
    cell, seed = cell_of(run)
    rj = json.loads((RUN / 'runs' / f'{run}.json').read_text())
    z = np.load(RUN / 'raw' / f'{run}.npz')
    ck = torch.load(RUN / 'models' / f'{run}.pt', map_location='cpu', weights_only=False)
    sd = {k: v.double() for k, v in ck['state_dict'].items()}
    plain = cell == 'E10a-AE'
    w = z['conf_weights'].astype(np.float64)
    r = dict(run=run, cell=cell, seed=seed, family='plain_ae' if plain else 'gauss_vae',
             weights_maxabs_vs_pw01=float(np.abs(w - w_ref).max()))
    x = states.astype(np.float64)
    xt = torch.as_tensor(x)
    h = mlp(sd, 'enc.', xt)
    if plain:
        mu = z['conf_codes'].astype(np.float64)
        mine_mu = h.numpy()
        lv = None
    else:
        mu = z['conf_mu'].astype(np.float64)
        L = h.shape[1] // 2
        mine_mu = h[:, :L].numpy()
        lv = z['conf_logvar'].astype(np.float64)
        r['logvar_maxabs_vs_forward'] = float(np.abs(h[:, L:].clamp(-12, 6).numpy() - lv).max())
    r['latent'] = int(mu.shape[1])
    r['mu_maxabs_vs_forward'] = float(np.abs(mine_mu - mu).max())
    A = wvar_loop(mu, w)
    r['A'] = [float(v) for v in A]
    r['count_by_threshold'] = {str(t): int(sum(1 for v in A if v > t)) for t in THRESHOLDS}
    r['count'] = r['count_by_threshold']['0.01']
    r['active'] = [j for j, v in enumerate(A) if v > 0.01]
    logits = mlp(sd, 'dec.net.', torch.as_tensor(mu)).numpy()
    bits_fwd = (logits > 0).astype(np.int8)
    rb = z['conf_recon_bits']
    r['recon_bits_match_forward_frac'] = float((bits_fwd == rb).mean())
    eq = rb == states
    wn = w / w.sum()
    r['recon_exact'] = float(np.sum(wn * eq.all(1)))
    r['recon_bitwise'] = float(np.sum(wn * eq.mean(1)))
    r['recon_ok'] = bool(r['recon_exact'] >= 0.95 and r['recon_bitwise'] >= 0.99)
    curve = rj['log']['curve']
    r['converged'], r['conv_rel_change'] = converged(curve, plain)
    finite_obj = all(math.isfinite(c['dev_objective']) and math.isfinite(c['train_loss']) for c in curve)
    if plain:
        kl_ok = True
    else:
        kl_ok = bool(all((z[k] >= 0).all() for k in z.files if k.endswith('_kl_per_unit')))
    r['numerical_ok'] = bool(rj['status'] == 'ok' and finite_obj and np.isfinite(mu).all() and kl_ok)
    pos = w > 0
    r['H_X_nats'] = float(-np.sum(wn[pos] * np.log(wn[pos])))
    if not plain:
        kl_unit = 0.5 * (mu ** 2 + np.exp(lv) - 1.0 - lv)
        rate = float(np.sum(wn @ kl_unit))
        r['rate_nats'] = rate
        r['rate_vs_stored_kl_sum'] = float(abs(rate - z['conf_kl_per_unit'].sum()))
        sig = np.exp(0.5 * lv)
        nll = np.zeros(len(x))
        for _ in range(128):
            eps = torch.randn(mu.shape, generator=gen, dtype=torch.float64).numpy()
            lg = mlp(sd, 'dec.net.', torch.as_tensor(mu + sig * eps)).numpy()
            nll += (np.logaddexp(0, lg) - x * lg).sum(1)
        nll /= 128
        r['distortion_nats'] = float(np.sum(wn * nll))
        r['neg_elbo_nats'] = r['distortion_nats'] + rate
        r['rate_minus_H_nats'] = rate - r['H_X_nats']
    else:
        lg = logits
        r['distortion_nats'] = float(np.sum(wn * (np.logaddexp(0, lg) - x * lg).sum(1)))
    return r


# ---------------------------------------------------------------- 5. gates and verdicts
def med(v):
    return float(np.median(v)) if v else None


def verdicts(rows, conv_only):
    use = [r for r in rows if r['converged']] if conv_only else rows
    by = {}
    for r in sorted(use, key=lambda r: r['seed']):
        by.setdefault(r['cell'], []).append(r)
    num = lambda c: [r['count'] for r in by.get(c, []) if r['numerical_ok']]  # noqa: E731
    rec = lambda c: [r['count'] for r in by.get(c, []) if r['recon_ok']]  # noqa: E731
    out = {}
    ref = num('E36-L32')
    p1 = {}
    for e in ('E10a-L32', 'E10b-L32'):
        v = num(e)
        if not v or not ref:
            p1[e] = dict(verdict=None, values=v)
            continue
        cond = dict(median_le_3=med(v) <= 3, max_le_5=max(v) <= 5, gap_ge_10=med(ref) - med(v) >= 10)
        p1[e] = dict(values=v, median=med(v), ref_values=ref, ref_median=med(ref), conditions=cond, verdict=all(cond.values()))
    vs = [p['verdict'] for p in p1.values()]
    out['P1'] = dict(verdict=None if None in vs else all(vs), per_encoding=p1)
    p2 = {}
    for c in (1, 2, 3, 4):
        v = num(f'code10-c{c}')
        if not v:
            p2[str(c)] = dict(verdict=None, values=v)
        elif c == 1:
            p2[str(c)] = dict(values=v, median=med(v), verdict=med(v) <= 1)
        else:
            p2[str(c)] = dict(values=v, median=med(v), verdict=med(v) == 10 and all(9 <= x <= 11 for x in v))
    vs = [p['verdict'] for p in p2.values()]
    out['P2'] = dict(verdict=None if None in vs else all(vs), per_c=p2)
    p3 = {L: rec(f'E36-L{L}') for L in (24, 32, 64)}
    if any(len(v) < 3 for v in p3.values()):
        out['P3'] = dict(verdict=None, per_L=p3)
    else:
        m = [med(v) for v in p3.values()]
        out['P3'] = dict(verdict=max(m) - min(m) <= 2, spread=max(m) - min(m),
                         per_L={str(L): dict(values=v, median=med(v)) for L, v in p3.items()})
    s1 = {}
    for L in (8, 10, 12):
        v = rec(f'E36-L{L}')
        s1[str(L)] = dict(values=v, median=med(v), verdict=None if len(v) < 3 else med(v) >= L - 1)
    vs = [p['verdict'] for p in s1.values()]
    out['S1'] = dict(verdict=None if None in vs else all(vs), per_L=s1)
    e30, e36 = rec('E30-L32'), rec('E36-L32')
    out['S2'] = dict(verdict=None if len(e30) < 3 or len(e36) < 3 else med(e30) <= med(e36),
                     e30=dict(values=e30, median=med(e30)), e36=dict(values=e36, median=med(e36)))
    return out


def gates(rows):
    ref = [r for r in rows if r['cell'] == 'E36-L32']
    ae = [r for r in rows if r['cell'] == 'E10a-AE']
    m = med([r['count'] for r in ref])
    pipe = dict(n=len(ref), median=m, recon_pass=sum(r['recon_ok'] for r in ref),
                values=[r['count'] for r in sorted(ref, key=lambda r: r['seed'])])
    pipe['pass'] = bool(len(ref) == 5 and abs(m - 15) <= 1 and pipe['recon_pass'] == 5)
    cap = dict(exact=[r['recon_exact'] for r in sorted(ae, key=lambda r: r['seed'])])
    cap['pass'] = bool(sum(e >= 0.95 for e in cap['exact']) >= 2 and len(ae) == 3)
    return dict(pipeline=pipe, capacity_p1=cap)


# ---------------------------------------------------------------- comparison
class Cmp:
    def __init__(self):
        self.n = 0
        self.mis = []

    def eq(self, what, mine, theirs, tol=None):
        self.n += 1
        if tol is None:
            ok = mine == theirs
        else:
            ok = theirs is not None and mine is not None and abs(mine - theirs) <= tol
        if not ok:
            self.mis.append(dict(check=what, mine=mine, results_json=theirs))


def apply_gate(v, g, conv_only):
    """DESIGN.md: every verdict untested if the pipeline gate fails; P1 also if the capacity check fails."""
    if not g['pipeline']['pass']:
        for k in v:
            v[k]['verdict'] = None
    elif not g['capacity_p1']['pass']:
        v['P1']['verdict'] = None
    return v


def main():
    hashes, manifest_paths = check_hashes()
    assert not hashes['mismatched'] and hashes['design_sha_ok'] and hashes['run_json_freeze_ok'], hashes
    ctx = np.load(RUN / 'raw' / 'data_context.npz')
    data, w_test = data_checks(ctx)
    res = json.loads((RUN / 'results.json').read_text())
    runs = sorted(p.stem for p in (RUN / 'runs').glob('*.json'))
    for r in runs:
        for sub in (f'runs/{r}.json', f'raw/{r}.npz', f'models/{r}.pt'):
            assert sub in manifest_paths, f'{sub} not hashed in MANIFEST'
    have = {cell_of(r) for r in runs}
    assert have == expected_runs(), (have ^ expected_runs())
    gen = torch.Generator().manual_seed(20261027)
    rows = []
    for run in runs:
        cell, _ = cell_of(run)
        if cell.startswith('code10'):
            st = ctx[f'states_code10c{cell[-1]}']
            wref = np.full(1024, 1 / 1024)
        else:
            enc = {'E30-L32': 'e30', 'E10a-L32': 'e10a', 'E10b-L32': 'e10b', 'E10a-AE': 'e10a', 'E10a-b0.25': 'e10a'}.get(cell, 'bw')
            st = ctx[f'states_{enc}']
            wref = w_test
        rows.append(per_run(run, st, wref, gen))
        print(run, rows[-1]['count'], rows[-1]['recon_ok'], rows[-1]['converged'], flush=True)
    g = gates(rows)
    V = {k: apply_gate(verdicts(rows, k == 'converged_only'), g, k == 'converged_only') for k in ('all_runs', 'converged_only')}
    labels = {k: v['verdict'] for k, v in V['all_runs'].items() if k.startswith('P')}
    outcome = ('supported' if all(v is True for v in labels.values()) else 'unsupported' if all(v is False for v in labels.values())
               else 'untested' if all(v is None for v in labels.values()) else 'mixed')

    # ---- compare
    C = Cmp()
    theirs = {r['run']: r for r in res['per_run']}
    C.eq('run set', sorted(theirs), runs)
    for r in rows:
        t = theirs[r['run']]
        p = r['run']
        C.eq(f'{p} label', r['cell'], t['label'])
        C.eq(f'{p} numerical_ok', r['numerical_ok'], t['numerical_ok'])
        C.eq(f'{p} recon_exact', r['recon_exact'], t['recon_exact'], 1e-12)
        C.eq(f'{p} recon_bitwise', r['recon_bitwise'], t['recon_bitwise'], 1e-12)
        C.eq(f'{p} recon_ok', r['recon_ok'], t['recon_ok'])
        C.eq(f'{p} converged', r['converged'], t['converged'])
        C.eq(f'{p} H_X', r['H_X_nats'], t['H_X_nats'], 1e-9)
        if r['family'] == 'plain_ae':
            C.eq(f'{p} count_covariance', r['count'], t['count_covariance'])
            C.eq(f'{p} distortion', r['distortion_nats'], t['distortion_nats'], 1e-5)
            continue
        C.eq(f'{p} count', r['count'], t['count'])
        C.eq(f'{p} active set', r['active'], t['active'])
        for th in ('0.001', '0.05', '0.1'):
            C.eq(f'{p} count@{th}', r['count_by_threshold'][th], t['count_robust'][th])
        C.eq(f'{p} A_u vector', max(abs(a - b) for a, b in zip(r['A'], res['A_u'][p])), 0.0, 1e-9)
        C.eq(f'{p} rate', r['rate_nats'], t['rate_nats'], 1e-6)
        C.eq(f'{p} rate_minus_H', r['rate_minus_H_nats'], t['rate_minus_H_nats'], 1e-6)
        tol_d = max(6 * t['distortion_se_nats'], 1e-4)  # independent Monte Carlo draws
        C.eq(f'{p} distortion (MC, tol 6 SE)', r['distortion_nats'], t['distortion_nats'], tol_d)
        C.eq(f'{p} neg_elbo (MC, tol 6 SE)', r['neg_elbo_nats'], t['neg_elbo_nats'], tol_d)
    # internal consistency (stored arrays vs my forward pass / pw01 weights)
    for r in rows:
        p = r['run']
        C.eq(f'{p} weights vs pw01', r['weights_maxabs_vs_pw01'], 0.0, 1e-12)
        C.eq(f'{p} mu vs own forward pass', r['mu_maxabs_vs_forward'], 0.0, 1e-4)
        C.eq(f'{p} recon bits vs own decode', r['recon_bits_match_forward_frac'], 1.0, 0.0)
        if 'rate_vs_stored_kl_sum' in r:
            C.eq(f'{p} analytic rate vs stored KL', r['rate_vs_stored_kl_sum'], 0.0, 1e-6)
            C.eq(f'{p} logvar vs own forward pass', r['logvar_maxabs_vs_forward'], 0.0, 1e-4)
    for k, v in data.items():
        if isinstance(v, bool):
            C.eq(f'data {k}', v, True)
    C.eq('data e30', data['e30']['columns_found'] and data['e30']['dropped_are_functions'] and data['e30']['rows_distinct'], True)
    C.eq('data w_test vs context', data['w_test_vs_context_maxabs'], 0.0, 1e-12)
    # cells
    for cell, c in res['cells'].items():
        mine = [r['count'] for r in sorted(rows, key=lambda r: r['seed']) if r['cell'] == cell]
        mine_c = [r['count'] for r in sorted(rows, key=lambda r: r['seed']) if r['cell'] == cell and r['converged']]
        C.eq(f'cell {cell} values', mine, c['count']['values'])
        C.eq(f'cell {cell} median', med(mine), c['count']['median'])
        C.eq(f'cell {cell} values converged', mine_c, c['count_converged_only']['values'])
        C.eq(f'cell {cell} recon_pass', sum(r['recon_ok'] for r in rows if r['cell'] == cell), c['recon_pass'])
        C.eq(f'cell {cell} converged', sum(r['converged'] for r in rows if r['cell'] == cell), c['converged'])
    # gates
    C.eq('gate pipeline pass', g['pipeline']['pass'], res['gates']['pipeline']['pass'])
    C.eq('gate pipeline median', g['pipeline']['median'], res['gates']['pipeline']['median'])
    C.eq('gate pipeline recon_pass', g['pipeline']['recon_pass'], res['gates']['pipeline']['recon_pass'])
    C.eq('gate capacity pass', g['capacity_p1']['pass'], res['gates']['capacity_p1']['pass'])
    for a, b in zip(g['capacity_p1']['exact'], res['gates']['capacity_p1']['exact']):
        C.eq('gate capacity exact', a, b, 1e-12)
    C.eq('non-converged runs', sorted(r['run'] for r in rows if not r['converged']), sorted(res['descriptive']['non_converged_runs']))
    # verdicts
    for variant in ('all_runs', 'converged_only'):
        mv, tv = V[variant], res['verdicts'][variant]
        for rule in ('P1', 'P2', 'P3', 'S1', 'S2'):
            C.eq(f'{variant} {rule} verdict', mv[rule]['verdict'], tv[rule]['verdict'])
        for e, d in mv['P1']['per_encoding'].items():
            C.eq(f'{variant} P1 {e} values', d['values'], tv['P1']['per_encoding'][e]['values'])
            C.eq(f'{variant} P1 {e} conditions', d['conditions'], tv['P1']['per_encoding'][e]['conditions'])
            C.eq(f'{variant} P1 {e} verdict', d['verdict'], tv['P1']['per_encoding'][e]['verdict'])
        for c, d in mv['P2']['per_c'].items():
            C.eq(f'{variant} P2 c={c} values', d['values'], tv['P2']['per_c'][c]['values'])
            C.eq(f'{variant} P2 c={c} verdict', d['verdict'], tv['P2']['per_c'][c]['verdict'])
        C.eq(f'{variant} P3 spread', mv['P3']['spread'], tv['P3']['spread'])
        for L, d in mv['P3']['per_L'].items():
            C.eq(f'{variant} P3 L={L} values', d['values'], tv['P3']['per_L'][L]['values'])
        for L, d in mv['S1']['per_L'].items():
            C.eq(f'{variant} S1 L={L} values', d['values'], tv['S1']['per_L'][L]['values'])
            C.eq(f'{variant} S1 L={L} verdict', d['verdict'], tv['S1']['per_L'][L]['verdict'])
        C.eq(f'{variant} S2 e30', mv['S2']['e30']['values'], tv['S2']['e30']['values'])
        C.eq(f'{variant} S2 e36', mv['S2']['e36']['values'], tv['S2']['e36']['values'])
    C.eq('outcome', outcome, res['outcome'])

    rec_path = RUN / 'audit_recompute.json'
    rec_path.write_text(json.dumps(dict(hashes=hashes, data=data, gates=g, verdicts=V, outcome=outcome,
                                        per_run=[{k: v for k, v in r.items()} for r in rows]), indent=1))
    status = 'pass' if not C.mis and not hashes['mismatched'] else 'fail'
    audit = dict(
        experiment_id='pw06_encoding_and_latent_size', run_id='full_001',
        auditor='separately written recount, independent of run_pw06.py / analyze_pw06.py',
        date=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='experiments/pw06_encoding_and_latent_size/audit_recount.py',
        results_sha256=sha256(RUN / 'results.json'),
        status=status,
        method=('Separately written recount (no import or reading of analyze_pw06.py, run_pw06.py or pwlib active-unit/'
                'convergence code). All MANIFEST hashes and the DESIGN.md freeze verified first. Test-split weights '
                'rebuilt from pw01 count_test (code10 uniform); encodings E30/E10a/E10b/code10 re-derived from DESIGN.md. '
                'Per run: explicit-loop weighted variance of the stored posterior mean (Burda, >0.01; also 0.001/0.05/0.1; '
                'plain AE covariance count); own encoder forward pass vs stored mu/logvar; own decoder(mu) vs stored '
                'reconstruction bits; exact/bitwise reconstruction; numerical check; convergence from the logged curve; '
                'analytic rate; own 128-draw Monte Carlo distortion (independent RNG, compared within 6 SE). Gates and '
                'P1-P3, S1, S2 re-derived from DESIGN.md decision rules, all runs and converged-only.'),
        limitations=('Distortion and -ELBO are Monte Carlo and agree only within tolerance (independent draws); same '
                     'machine and library stack as the run; no independent retraining.'),
        n_runs=len(rows), n_checks=C.n, n_mismatches=len(C.mis), mismatches=C.mis,
        checks=dict(manifest_hashes=not hashes['mismatched'], **{'design_freeze': hashes['design_sha_ok']},
                    counts_match=not any('count' in m['check'] or 'active' in m['check'] or 'A_u' in m['check'] for m in C.mis),
                    reconstruction_match=not any('recon' in m['check'] for m in C.mis),
                    convergence_match=not any('converged' in m['check'] for m in C.mis),
                    gates_match=not any(m['check'].startswith('gate') for m in C.mis),
                    verdicts_match=not any(m['check'].startswith(('all_runs', 'converged_only', 'outcome')) for m in C.mis)),
        recount=dict(gates={k: v['pass'] for k, v in g.items()},
                     verdicts_all_runs={k: v['verdict'] for k, v in V['all_runs'].items()},
                     verdicts_converged_only={k: v['verdict'] for k, v in V['converged_only'].items()}, outcome=outcome),
        evidence=[dict(path=p, bytes=(RUN / p).stat().st_size, sha256=sha256(RUN / p))
                  for p in ('audit_recompute.json', 'results.json', 'MANIFEST.json')],
    )
    (RUN / 'AUDIT.json').write_text(json.dumps(audit, indent=1))
    print(json.dumps(dict(status=status, n_checks=C.n, mismatches=C.mis[:20], recount=audit['recount']), indent=1))


if __name__ == '__main__':
    if {'-h', '--help'} & set(sys.argv[1:]):  # usage without the data present
        print(__doc__)
        sys.exit(0)
    main()
