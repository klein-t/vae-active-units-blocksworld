"""pw06 registered analysis (DESIGN.md "Predictions and decision rules", "Gates", "Descriptive").

Usage (once, on the main run):
  PYTHONPATH=src .venv/bin/python experiments/pw06_encoding_and_latent_size/analyze_pw06.py --run-id full_001

Reads results/ID/{RUN.json, sweep_grid_used.json, runs_index.json, task_gate.json, runs/*.json, raw/*.npz,
models/*.pt}; writes results.json and SUMMARY.md. Refuses pilot runs (the pilot never sees counts).

Definitions (all on the test split = confirmation weights 'conf', as in pw02a full_001):
- count: Burda active units, state-frequency-weighted variance of the posterior mean > 0.01
  (pwlib.metrics.active_units on the stored conf_mu / conf_weights, exactly as pw02a).
- numerical check: run status ok, finite objective and non-negative per-unit KL (pwlib.sweep.numerical_flags).
- reconstruction check: from the posterior mean on the test split, exact-state >= 0.95 and bitwise >= 0.99.
- convergence check: pw02a rule (pwlib.sweep.converged_amended: dev objective change over the last 20% of
  steps < 2% relative; plain AE also accepted below 1e-3 nats absolute).
- rate R = sum of the stored per-unit KL (conf weights); distortion D = Monte Carlo
  E_eps[-log p(x | mu + sigma eps)] with 128 draws, seed 0 (pw05 definition); -ELBO = D + R; nats.
Rule inputs: P1 and P2 use every run passing the numerical check; P3, S1, S2 use runs passing the
reconstruction check, and a cell with fewer than 3 such runs is untested. Each verdict is computed with all
runs ('all', the registered verdict) and without runs failing the convergence check ('converged_only').
P1/P2 are untested when a cell they need has no eligible run. Every verdict is untested if the pipeline gate
fails; P1 is also untested if the E10a plain-AE capacity check fails.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from pwlib import metrics as M  # noqa: E402
from pwlib import sweep as S  # noqa: E402
from pwlib import training as T  # noqa: E402
from pwlib.runio import dump_json, load_json, read_run, results_header, sha256  # noqa: E402

RECON_EXACT_MIN, RECON_BITWISE_MIN = 0.95, 0.99
MC_DRAWS, MC_SEED = 128, 0
REF_MEDIAN, REF_TOL = 15, 1
MIN_RUNS = 3
PW02A_FULL = ROOT / 'experiments/pw02a_active_units_amended/results/full_001'
LN2 = math.log(2)


def entropy_nats(w):
    w = np.asarray(w, dtype=np.float64)
    w = w[w > 0]
    return float(-(w * np.log(w)).sum())


def mc_distortion(model, mu, logvar, X, w, draws=MC_DRAWS, seed=MC_SEED):
    """D = sum_x w(x) E_eps[-log p(x | mu + exp(logvar/2) eps)]; returns (D, MC SE) in nats (pw05)."""
    g = torch.Generator().manual_seed(seed)
    mu_t, sd_t = torch.as_tensor(mu), torch.as_tensor(np.exp(0.5 * logvar.astype(np.float64)).astype(np.float32))
    x = torch.as_tensor(X, dtype=torch.float32)
    per = []
    with torch.no_grad():
        for _ in range(draws):
            z = mu_t + sd_t * torch.randn(mu_t.shape, generator=g)
            per.append(float((model.dec.nll(x, model.dec(z)).double().numpy() * w).sum()))
    per = np.array(per)
    return float(per.mean()), float(per.std(ddof=1) / np.sqrt(draws))


def load_model(run_dir, cell, seed, grid, d_in):
    ck = torch.load(Path(run_dir) / 'models' / f'{S.run_name(cell["cell_id"], seed)}.pt', map_location='cpu',
                    weights_only=False)
    if ck['cell']['cell_id'] != cell['cell_id'] or ck['seed'] != seed:
        raise ValueError('Checkpoint identity mismatch')
    model = T.build_model(cell, grid, d_in, torch.zeros(d_in))
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model


def run_row(run_dir, cell, seed, grid, states):
    name = S.run_name(cell['cell_id'], seed)
    log = load_json(run_dir / 'runs' / f'{name}.json')
    row = dict(run=name, cell_id=cell['cell_id'], label=cell['label'], seed=seed, family=cell['family'],
               latent=cell['latent'], status=log['status'])
    if log['status'] != 'ok':
        row.update(numerical_ok=False, recon_ok=False, converged=None)
        return row
    with np.load(run_dir / 'raw' / f'{name}.npz') as z:
        arr = {k: z[k] for k in z.files}
    w = arr['conf_weights']
    X = states[cell['substrate']]
    if len(w) != len(X):
        raise ValueError(f'{name}: weights and states differ in length')
    mc = log['metrics']['conf']
    row.update(numerical_ok=bool(log['numerical']['finite'] and log['numerical']['kl_nonnegative']),
               recon_exact=mc['exact'], recon_bitwise=mc['bitwise'],
               recon_ok=bool(mc['exact'] >= RECON_EXACT_MIN and mc['bitwise'] >= RECON_BITWISE_MIN),
               converged=S.converged_amended(cell, log), dev_exact=log['metrics']['dev']['exact'],
               wall_seconds=log['log']['wall_seconds'], H_X_nats=entropy_nats(w))
    model = load_model(run_dir, cell, seed, grid, X.shape[1])
    if cell['family'] == 'gauss_vae':
        mu, lv = arr['conf_mu'], arr['conf_logvar']
        n, mask, A = M.active_units(mu, w)
        R = float(arr['conf_kl_per_unit'].sum())
        D_, se = mc_distortion(model, mu, lv, X, w)
        row.update(count=n, active=np.flatnonzero(mask).tolist(), A_u=A.tolist(),
                   count_robust={str(t): M.active_units(mu, w, t)[0] for t in M.AU_ROBUSTNESS},
                   rate_nats=R, distortion_nats=D_, distortion_se_nats=se, neg_elbo_nats=D_ + R,
                   rate_minus_H_nats=R - row['H_X_nats'])
    else:  # plain AE: covariance count of the codes (descriptive), deterministic reconstruction NLL
        codes = arr['conf_codes']
        n, mask, A = M.active_units(codes, w)
        with torch.no_grad():
            x = torch.as_tensor(X, dtype=torch.float32)
            D_ = float((model.dec.nll(x, model.dec(model.enc(x))).double().numpy() * w).sum())
        row.update(count_covariance=n, A_u=A.tolist(), distortion_nats=D_)
    return row


def eligible(rows, label, need, converged_only):
    out = [r for r in rows if r['label'] == label and r[need]]
    if converged_only:
        out = [r for r in out if r['converged']]
    return out


def stats(vals):
    if not vals:
        return dict(n=0, values=[], median=None, min=None, max=None)
    return dict(n=len(vals), values=vals, median=float(np.median(vals)), min=min(vals), max=max(vals))


def counts(rows, label, need, conv):
    return [r['count'] for r in eligible(rows, label, need, conv)]


def p1(rows, conv):
    ref = counts(rows, 'E36-L32', 'numerical_ok', conv)
    per = {}
    for lab in ('E10a-L32', 'E10b-L32'):
        v = counts(rows, lab, 'numerical_ok', conv)
        if not v or not ref:
            per[lab] = dict(verdict=None, reason='no eligible run', values=v, ref_values=ref)
            continue
        c = dict(median_le_3=bool(np.median(v) <= 3), max_le_5=bool(max(v) <= 5),
                 gap_ge_10=bool(np.median(ref) - np.median(v) >= 10))
        per[lab] = dict(verdict=bool(all(c.values())), conditions=c, values=v, median=float(np.median(v)),
                        ref_values=ref, ref_median=float(np.median(ref)))
    vs = [p['verdict'] for p in per.values()]
    return dict(verdict=None if None in vs else bool(all(vs)), per_encoding=per)


def p2(rows, conv):
    per = {}
    for c in (1, 2, 3, 4):
        v = counts(rows, f'code10-c{c}', 'numerical_ok', conv)
        if not v:
            per[c] = dict(verdict=None, reason='no eligible run', values=v)
            continue
        ok = bool(np.median(v) <= 1) if c == 1 else bool(np.median(v) == 10 and all(9 <= x <= 11 for x in v))
        per[c] = dict(verdict=ok, values=v, median=float(np.median(v)))
    vs = [p['verdict'] for p in per.values()]
    return dict(verdict=None if None in vs else bool(all(vs)), per_c={str(k): v for k, v in per.items()})


def tested_median(rows, label, conv):
    v = counts(rows, label, 'recon_ok', conv)
    return v, (float(np.median(v)) if len(v) >= MIN_RUNS else None)


def p3(rows, conv):
    per = {L: tested_median(rows, f'E36-L{L}', conv) for L in (24, 32, 64)}
    meds = [m for _, m in per.values()]
    det = {str(L): dict(values=v, median=m) for L, (v, m) in per.items()}
    if None in meds:
        return dict(verdict=None, reason=f'fewer than {MIN_RUNS} reconstruction-passing runs', per_L=det)
    return dict(verdict=bool(max(meds) - min(meds) <= 2), spread=max(meds) - min(meds), per_L=det)


def s1(rows, conv):
    per = {}
    for L in (8, 10, 12):
        v, m = tested_median(rows, f'E36-L{L}', conv)
        per[str(L)] = dict(values=v, median=m, verdict=None if m is None else bool(m >= L - 1))
    vs = [p['verdict'] for p in per.values()]
    verdict = False if False in vs else (None if None in vs else True)
    return dict(verdict=verdict, per_L=per)


def s2(rows, conv):
    v30, m30 = tested_median(rows, 'E30-L32', conv)
    v36, m36 = tested_median(rows, 'E36-L32', conv)
    if m30 is None or m36 is None:
        return dict(verdict=None, reason=f'fewer than {MIN_RUNS} reconstruction-passing runs', e30=v30, e36=v36)
    return dict(verdict=bool(m30 <= m36), e30=dict(values=v30, median=m30), e36=dict(values=v36, median=m36))


def verdicts(rows, conv, gates):
    v = dict(P1=p1(rows, conv), P2=p2(rows, conv), P3=p3(rows, conv), S1=s1(rows, conv), S2=s2(rows, conv))
    if not gates['pipeline']['pass']:
        for k in v:
            v[k] = dict(verdict=None, reason='pipeline gate failed (technical stop)', computed=v[k])
    elif not gates['capacity_p1']['pass']:
        v['P1'] = dict(verdict=None, reason='E10a plain-AE capacity check failed', computed=v['P1'])
    return v


def cell_table(rows, labels):
    table = {}
    for lab in labels:
        rs = [r for r in rows if r['label'] == lab]
        ok = [r for r in rs if r['status'] == 'ok']
        key = 'count' if ok and 'count' in ok[0] else 'count_covariance'
        med = lambda k: float(np.median([r[k] for r in ok])) if ok and all(k in r for r in ok) else None  # noqa: E731
        table[lab] = dict(
            runs=len(rs), ok=len(ok), recon_pass=sum(r['recon_ok'] for r in ok),
            converged=sum(bool(r['converged']) for r in ok), numerical_pass=sum(r['numerical_ok'] for r in ok),
            count_field=key, count=stats([r[key] for r in ok]),
            count_converged_only=stats([r[key] for r in ok if r['converged']]),
            neg_elbo_nats=med('neg_elbo_nats'), rate_nats=med('rate_nats'), distortion_nats=med('distortion_nats'),
            rate_minus_H_nats=med('rate_minus_H_nats'), H_X_nats=med('H_X_nats'),
            recon_exact_median=med('recon_exact'))
    return table


def outcome(v):
    """From the primary verdicts P1-P3: all untested / all hold / all fail; anything else is mixed."""
    prim = [v[k]['verdict'] for k in ('P1', 'P2', 'P3')]
    if all(p is None for p in prim):
        return 'untested'
    if all(p is True for p in prim):
        return 'supported'
    if all(p is False for p in prim):
        return 'unsupported'
    return 'mixed'


def reproduction_vs_pw02a(run_dir, rows):
    """Descriptive: pw06 E36-L32 uses pw02a's cell id and seeds; compare conf_mu and counts per seed."""
    out = {}
    for r in rows:
        if r['label'] != 'E36-L32' or r['status'] != 'ok':
            continue
        ref = PW02A_FULL / 'raw' / f'{r["run"]}.npz'
        if not ref.is_file():
            out[r['run']] = dict(available=False)
            continue
        with np.load(ref) as a, np.load(run_dir / 'raw' / f'{r["run"]}.npz') as b:
            same = bool(np.array_equal(a['conf_mu'], b['conf_mu']))
            n_ref = M.active_units(a['conf_mu'], a['conf_weights'])[0]
            diff = float(np.max(np.abs(a['conf_mu'] - b['conf_mu'])))
        out[r['run']] = dict(available=True, conf_mu_identical=same, max_abs_diff=diff, count_pw02a=n_ref,
                             count_pw06=r['count'])
    return out


def fmt(s):
    if s['n'] == 0:
        return 'n/a'
    return f"{s['median']:g} ({s['min']}–{s['max']})"


def summary_md(header, gates, v_all, v_conv, table, desc, out):
    def vs(x):
        return {True: 'holds', False: 'fails', None: 'untested'}[x['verdict']]
    lines = [f"# pw06 {header['run_id']} summary", '',
             f"Outcome (registered, all runs): **{out}**. DESIGN.md sha256 `{header['freeze_sha256']}`.", '',
             '| Gate | Pass | Detail |', '|---|---|---|',
             f"| task | {gates['task']['pass']} | data, encodings, model-free table |",
             f"| pipeline (E36-L32 median within 1 of 15, 5/5 reconstruct) | {gates['pipeline']['pass']} | "
             f"median {gates['pipeline']['median']}, recon {gates['pipeline']['recon_pass']}/{gates['pipeline']['n']} |",
             f"| capacity for P1 (E10a AE exact >= 0.95 in >= 2/3) | {gates['capacity_p1']['pass']} | "
             f"exact {gates['capacity_p1']['exact']} |", '',
             '| Rule | All runs (registered) | Without non-converged |', '|---|---|---|']
    for k in ('P1', 'P2', 'P3', 'S1', 'S2'):
        lines.append(f'| {k} | {vs(v_all[k])} | {vs(v_conv[k])} |')
    lines += ['', '| Cell | Count median (range), all ok runs | Recon pass | Converged | −ELBO | Rate | Distortion | Rate − H(X) |',
              '|---|---|---|---|---|---|---|---|']
    for lab, t in table.items():
        f2 = lambda x: 'n/a' if x is None else f'{x:.2f}'  # noqa: E731
        lines.append(f"| {lab} | {fmt(t['count'])}{' (cov.)' if t['count_field'] != 'count' else ''} | "
                     f"{t['recon_pass']}/{t['runs']} | {t['converged']}/{t['runs']} | {f2(t['neg_elbo_nats'])} | "
                     f"{f2(t['rate_nats'])} | {f2(t['distortion_nats'])} | {f2(t['rate_minus_H_nats'])} |")
    lines += ['', 'Nats; test-split (confirmation) state weights; counts = Burda active units at 0.01 '
              '(plain AE: covariance count, descriptive).', '',
              f"Median −ELBO per latent size (E36): {desc['neg_elbo_by_L']}.",
              f"E10a β = 0.25 count: {fmt(desc['e10a_beta025_count'])}. L = 16 count: {fmt(desc['l16_count'])}.", '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--results-root', default=None, help='SMOKE ONLY: scratch results directory')
    args = ap.parse_args()
    run_dir = Path(args.results_root or EXP / 'results') / args.run_id
    run = read_run(run_dir)
    if run['kind'] != 'full':
        raise SystemExit('The registered analysis runs on the main run only (the pilot never sees counts).')
    if run['freeze_sha256'] != sha256(EXP / 'DESIGN.md'):
        raise ValueError('DESIGN.md changed since RUN.json was written')
    torch.set_num_threads(1)
    grid = S.load_grid(run_dir / 'sweep_grid_used.json')
    task = load_json(run_dir / 'task_gate.json')
    index = load_json(run_dir / 'runs_index.json')
    with np.load(run_dir / 'raw' / 'data_context.npz') as f:
        states = {k[len('states_'):]: f[k] for k in f.files if k.startswith('states_')}
    planned = [(c, s) for c in grid['cells'] for s in c['seeds']]
    recorded = {r['run'] for r in index['runs']}
    missing = sorted(S.run_name(c['cell_id'], s) for c, s in planned if S.run_name(c['cell_id'], s) not in recorded)
    rows = [run_row(run_dir, c, s, grid, states) for c, s in planned
            if S.run_name(c['cell_id'], s) in recorded]
    labels = [c['label'] for c in grid['cells']]

    ref = [r for r in rows if r['label'] == 'E36-L32']
    ref_ok = [r for r in ref if r['numerical_ok']]
    ref_med = float(np.median([r['count'] for r in ref_ok])) if ref_ok else None
    task_ok = bool(task['pass'] and not missing)
    pipe = {'n': len(ref), 'median': ref_med, 'recon_pass': sum(r['recon_ok'] for r in ref),
            'pass': bool(task_ok and len(ref) == 5 and len(ref_ok) == 5 and abs(ref_med - REF_MEDIAN) <= REF_TOL
                         and all(r['recon_ok'] for r in ref))}
    ae = [r for r in rows if r['label'] == 'E10a-AE' and r['status'] == 'ok']
    cap = {'exact': [r['recon_exact'] for r in ae], 'n': len(ae),
           'pass': bool(sum(r['recon_exact'] >= RECON_EXACT_MIN for r in ae) >= 2)}
    gates = dict(task={'pass': task_ok}, pipeline=pipe, capacity_p1=cap,
                 numerical=dict(technical_failures=[r['run'] for r in rows if r['status'] != 'ok'],
                                numerical_fail=[r['run'] for r in rows if not r['numerical_ok']]))

    v_all = verdicts(rows, False, gates)
    v_conv = verdicts(rows, True, gates)
    table = cell_table(rows, labels)
    desc = dict(
        neg_elbo_by_L={str(L): table[f'E36-L{L}']['neg_elbo_nats'] for L in (8, 10, 12, 16, 24, 32, 64)},
        rate_minus_H_by_cell={lab: table[lab]['rate_minus_H_nats'] for lab in labels},
        e10a_beta025_count=table['E10a-b0.25']['count'], l16_count=table['E36-L16']['count'],
        non_converged_runs=[r['run'] for r in rows if r['status'] == 'ok' and not r['converged']],
        reproduction_vs_pw02a=reproduction_vs_pw02a(run_dir, rows))
    out = outcome(v_all) if gates['pipeline']['pass'] else 'untested'
    status = 'completed' if gates['pipeline']['pass'] else 'technical_stop'
    header = results_header(run_dir)
    per_run = [{k: v for k, v in r.items() if k != 'A_u'} for r in rows]
    dump_json(header | dict(status=status, outcome=out, eval_set='conf', design_sha256=sha256(EXP / 'DESIGN.md'),
                            gates=gates, verdicts=dict(all_runs=v_all, converged_only=v_conv),
                            cells=table, descriptive=desc, model_free=task['model_free'], missing_runs=missing,
                            per_run=per_run, A_u={r['run']: r.get('A_u') for r in rows},
                            definitions=__doc__), run_dir / 'results.json')
    (run_dir / 'SUMMARY.md').write_text(summary_md(header, gates, v_all, v_conv, table, desc, out))
    print(f'status {status}, outcome {out}; written {run_dir / "results.json"}')


if __name__ == '__main__':
    main()
