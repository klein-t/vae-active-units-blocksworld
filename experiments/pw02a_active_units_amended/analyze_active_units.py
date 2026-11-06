"""pw02a analysis: active units, robustness, hypotheses H2.2-H2.4 (pw02 PROTOCOL.md sections 5-9 as amended
by pw02a PROTOCOL.md: A3 removed, amended convergence rule).

Usage: uv run python experiments/pw02a_active_units_amended/analyze_active_units.py --run-id ID
Reads only retained operands (results/ID/raw, runs/) and the pw01 / targets files; writes
au_table.json, au_per_run.json (consumed by pw03/pw04) and results.json. The evaluation set is the
development distribution in a pilot and the confirmation distribution in a full run.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pwlib import metrics as M  # noqa: E402
from pwlib import sweep as S  # noqa: E402
from pwlib.runio import dump_json, load_json, read_run, results_header  # noqa: E402

EXP = Path(__file__).resolve().parent
DELTA = 2
K_AFF = 25
ALIVE_MIN = 0.005
SOURCES = ('w64', 'w128', 'w128s2', 'w128s3', 'w256')
ARCHS = ('A1', 'A2', 'A4')  # pw02a: A3 removed
BETA_MONOTONE_MIN = 6  # pw02a: >= 7 of 8 cells becomes 6 of 6 (fraction kept, rounded up)
N_INFORMATION_MIN = 3  # unchanged count: >= 3 information verdicts, now all 3 architectures
LRS = ('0.0003', '0.003')


def majority(n):
    return n // 2 + 1


def run_row(run_dir, rec, cell, key):
    log = load_json(run_dir / 'runs' / f'{rec["run"]}.json')
    row = dict(run=rec['run'], cell_id=cell['cell_id'], seed=rec['seed'], family=cell['family'],
               substrate=cell['substrate'], status=log['status'], competence=bool(log['competence']))
    if log['status'] != 'ok':
        return row, None
    z = np.load(run_dir / 'raw' / f'{rec["run"]}.npz')
    w = z[f'{key}_weights']
    row.update(converged=S.converged_amended(cell, log), snapshot=log.get('snapshot'),
               wall_seconds=log['log']['wall_seconds'],
               log_gamma=log['log'].get('log_gamma'), log_gamma_at_floor=log['numerical']['log_gamma_at_floor'],
               competence_metrics=log['metrics']['dev'])
    active = None
    if cell['family'] in ('gauss_vae', 'tc_vae'):
        mu, lv = z[f'{key}_mu'], z[f'{key}_logvar']
        n, mask, A = M.active_units(mu, w)
        asp = M.asperti_diagnostic(mu, lv, w)
        row.update(au=n, latent=int(mu.shape[1]), censored=bool(n == mu.shape[1]),
                   au_robust={str(t): M.active_units(mu, w, t)[0] for t in M.AU_ROBUSTNESS},
                   polarization=M.polarization_ratio(A), A_u=A,
                   asperti_sum=asp['sum'], asperti_not_converged=int(asp['not_converged'].sum()),
                   kl_per_unit=z[f'{key}_kl_per_unit'], recon_gain=z[f'{key}_recon_gain'],
                   tc=log['metrics'][key].get('tc'))
        active = mask
    elif cell['family'] == 'cat_vae':
        act = M.categorical_activity(z[f'{key}_probs'], w)
        active = act > M.AU_THRESHOLD
        row.update(au_categorical=int(active.sum()), n_vars=int(len(act)), activity=act)
    elif cell['family'] == 'topk_sae':
        fire = M.wmean(z[f'{key}_codes'] > 0, w)
        active = fire >= ALIVE_MIN
        row.update(alive=int(active.sum()), latent=int(len(fire)))
    else:
        n, active, A = M.active_units(z[f'{key}_codes'], w)
        row.update(au_covariance=n, latent=int(len(A)), A_u=A)
    return row, active


def cell_stats(rows, cid, field='au', gated=True):
    rs = [r for r in rows if r['cell_id'] == cid and r['status'] == 'ok' and (r['competence'] or not gated)]
    vals = [r[field] for r in rs]
    if not vals:
        return dict(values=[], median=None, min=None, max=None, range=None, n=0)
    return dict(values=vals, median=float(np.median(vals)), min=min(vals), max=max(vals),
                range=max(vals) - min(vals), n=len(vals))


def account_verdict(bw_, code, prod):
    if abs(bw_ - code) <= DELTA and prod > bw_ + DELTA:
        return 'information'
    if abs(bw_ - prod) <= DELTA and code > bw_ + DELTA:
        return 'factor'
    if abs(bw_ - prod) <= DELTA and code < bw_ - DELTA:
        return 'affine'
    return 'none'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    run_dir = EXP / 'results' / args.run_id
    run = read_run(run_dir)
    pilot = run['kind'] == 'pilot'
    key = 'dev' if pilot else 'conf'
    grid = S.load_grid(run_dir / 'sweep_grid_used.json')
    cells = {c['cell_id']: c for c in grid['cells']}
    index = load_json(run_dir / 'runs_index.json')
    task = load_json(run_dir / 'task_gate.json')
    src_gate = load_json(run_dir / 'source_gate.json')
    targets_b = load_json(run_dir / 'targets_b.json')
    rows, active_sets = [], {}
    for rec in index['runs']:
        row, active = run_row(run_dir, rec, cells[rec['cell_id']], key)
        rows.append(row)
        if active is not None:
            active_sets[rec['run']] = np.flatnonzero(active)
    planned = {S.run_name(c['cell_id'], s) for c, s in S.planned_runs(grid, pilot)}
    missing = sorted(planned - {r['run'] for r in rows})

    # ---- gates
    sources_ok = all(src_gate['sources'][s]['gate']['pass'] for s in SOURCES)
    lin_ok = all(src_gate['sources'][s]['numerical']['b_lin_ok'] for s in SOURCES)
    tech = [r['run'] for r in rows if r['status'] != 'ok']
    gates = dict(task=bool(task['pass'] and sources_ok and not missing),
                 numerical=bool(task['numerical_pca_25'] and lin_ok and not tech))
    beta1 = [f'bw-gauss_vae-{a}-b1-lr{lr}' for a in ARCHS for lr in LRS]
    ctrl = [f'{d}-gauss_vae-{a}-b1-lr0.0003' for d in ('code10', 'prod5') for a in ARCHS]
    lin_learned = [f'b_lin_{s}-gauss_vae-A2-b1-glearned-lr0.0003' for s in SOURCES]
    required_cells = beta1 + ctrl + lin_learned
    if pilot:
        required_cells = [c for c in required_cells if c in grid['pilot_cells']]
    comp_table = {}
    for cid in required_cells:
        rs = [r for r in rows if r['cell_id'] == cid]
        comp_table[cid] = dict(n=len(rs), passing=sum(r['competence'] for r in rs))
    gates['competence'] = bool(all(v['n'] > 0 and v['passing'] >= majority(v['n']) for v in comp_table.values()))

    metrics = dict(eval_set=key, n_runs=len(rows), technical_failures=tech, missing_runs=missing,
                   competence_required_cells=comp_table,
                   competence_pass_rate=float(np.mean([r['competence'] for r in rows])) if rows else None,
                   targets=dict(pw01=task['pw01_targets'], b=targets_b['values']))
    au_rows = [dict({k: v for k, v in r.items() if k not in ('A_u', 'asperti_sum', 'kl_per_unit', 'recon_gain',
                                                             'activity')}) for r in rows]
    gauss = [r for r in rows if r.get('au') is not None]
    metrics['polarized_fraction'] = float(np.mean([r['polarization'] is not None and r['polarization'] >= M.POLARIZED_RATIO
                                                   for r in gauss])) if gauss else None
    metrics['censored_runs'] = [r['run'] for r in gauss if r['censored']]
    metrics['asperti_not_converged_units'] = {r['run']: r['asperti_not_converged'] for r in gauss}
    metrics['log_gamma_at_floor'] = [r['run'] for r in rows if r.get('log_gamma_at_floor')]
    metrics['convergence'] = {r['run']: r.get('converged') for r in rows}
    metrics['wall_seconds_per_run'] = {r['run']: r.get('wall_seconds') for r in rows}

    criteria, secondary = {}, {}
    if pilot:
        metrics['pilot_checks'] = dict(
            au_below_latent_bw=all(not r['censored'] for r in gauss if r['substrate'] == 'bw'),
            converged_all=all(v for v in metrics['convergence'].values() if v is not None),
            log_gamma={r['run']: r['log_gamma'] for r in rows if r.get('log_gamma') is not None})
        status = 'completed' if all(gates.values()) else ('technical_stop' if not gates['task'] or not gates['numerical']
                                                          else 'competence_stop')
        outcome = 'pilot_only' if status == 'completed' else 'untested'
        interp = ('Pilot (seed 101, development data only): gates computable; convergence, AU < latent size, '
                  'wall-clock and log-gamma behaviour recorded in metrics.pilot_checks. No hypothesis is tested.')
    else:
        stats = lambda cid, **kw: cell_stats(rows, cid, **kw)  # noqa: E731
        # H2.2 (merged with former H2.1)
        part_i = [r for r in gauss if r['cell_id'] in beta1 and r['competence']]
        h21 = bool(part_i) and all(r['au'] < K_AFF and not r['censored'] for r in part_i)
        verdicts = {}
        for a in ARCHS:
            b_, c_, p_ = (stats(f'{d}-gauss_vae-{a}-b1-lr0.0003') for d in ('bw', 'code10', 'prod5'))
            verdicts[a] = dict(bw=b_['median'], code10=c_['median'], prod5=p_['median'],
                               verdict=account_verdict(b_['median'], c_['median'], p_['median'])
                               if None not in (b_['median'], c_['median'], p_['median']) else 'none')
        n_info = sum(v['verdict'] == 'information' for v in verdicts.values())
        criteria['h22_information_account'] = bool(h21 and n_info >= N_INFORMATION_MIN)
        metrics['h22'] = dict(part_i_all_beta1_runs_below_affine=h21, part_i_runs=len(part_i),
                              part_i_max_au=max((r['au'] for r in part_i), default=None),
                              per_architecture=verdicts, n_information=n_info)
        # H2.3
        cs = {cid: stats(cid) for cid in beta1}
        meds = [v['median'] for v in cs.values() if v['median'] is not None]
        between = max(meds) - min(meds) if meds else None
        within = max((v['range'] for v in cs.values() if v['range'] is not None), default=None)
        criteria['h23_architecture_dependence'] = bool(between is not None and within is not None and between >= 3
                                                       and between > within)
        mono = {}
        for a in ARCHS:
            for lr in LRS:
                m = [stats(f'bw-gauss_vae-{a}-b{b}-lr{lr}', gated=False)['median'] for b in ('0.25', '1', '4')]
                mono[f'{a}-lr{lr}'] = dict(medians=m, nonincreasing=None if None in m else bool(m[0] >= m[1] >= m[2]))
        secondary['h23_beta_monotone'] = bool(sum(bool(v['nonincreasing']) for v in mono.values()) >= BETA_MONOTONE_MIN)
        metrics['h23'] = dict(cells=cs, between_cell_range=between, max_within_cell_range=within,
                              beta_monotone=mono, variance_decomposition=variance_decomposition(gauss, beta1))
        # H2.4
        per_src = {}
        for s in SOURCES:
            L = stats(f'b_lin_{s}-gauss_vae-A2-b1-glearned-lr0.0003')
            Fx = stats(f'b_lin_{s}-gauss_vae-A2-b1-gfixed-lr0.0003', gated=False)
            R = stats(f'b_raw_{s}-gauss_vae-A2-b1-glearned-lr0.0003')
            ok = (L['median'] is not None and Fx['median'] is not None and L['median'] - Fx['median'] >= DELTA
                  and L['min'] > Fx['max'])
            per_src[s] = dict(lin_learned=L, lin_fixed=Fx, raw_learned=R, learned_exceeds_fixed=bool(ok),
                              raw_more_than_lin=None if None in (R['median'], L['median']) else bool(R['median'] > L['median']),
                              K_pca99=targets_b['values'][s]['K_pca99'], K_probe=targets_b['values'][s]['K_probe'])
        criteria['h24_learned_gamma_more_units'] = bool(all(v['learned_exceeds_fixed'] for v in per_src.values()))
        ref = stats('bw-gauss_vae-A2-b1-lr0.0003')['median']
        secondary['h24_b_lin_closer_to_affine_than_raw_states'] = bool(ref is not None and all(
            v['lin_learned']['median'] is not None and abs(v['lin_learned']['median'] - K_AFF) < abs(ref - K_AFF)
            for v in per_src.values()))
        secondary['h24_b_raw_more_than_b_lin'] = bool(all(v['raw_more_than_lin'] for v in per_src.values()))
        widths = [per_src[s]['raw_learned']['median'] for s in ('w64', 'w128', 'w256')]
        secondary['h24_width_trend'] = dict(raw_learned_medians_w64_w128_w256=widths)
        secondary['h24_au_vs_pca99_and_probe_dim'] = {s: dict(raw=v['raw_learned']['median'], K_pca99=v['K_pca99'],
                                                              K_probe=v['K_probe']) for s, v in per_src.items()}
        secondary['h24_transformer_seed_spread_w128'] = {
            f: [per_src[s][f]['median'] for s in ('w128', 'w128s2', 'w128s3')] for f in ('lin_learned', 'raw_learned')}
        secondary['threshold_robustness_and_polarization'] = dict(polarized_fraction=metrics['polarized_fraction'])
        ae = [r for r in rows if r['cell_id'] == 'bw-plain_ae-A2-lr0.0003' and r['status'] == 'ok']
        secondary['plain_ae_au_metric_control'] = dict(au_covariance=[r['au_covariance'] for r in ae])
        metrics['h24'] = per_src
        if not all(gates.values()):
            status = 'technical_stop' if not (gates['task'] and gates['numerical']) else 'competence_stop'
            outcome, interp = 'untested', 'Gate failure: hypotheses not evaluated (see gates and metrics).'
        else:
            status = 'completed'
            n_pass = sum(criteria.values())
            outcome = 'supported' if n_pass == len(criteria) else 'unsupported' if n_pass == 0 else 'mixed'
            interp = ('Active-unit counts on the confirmation distribution; AU is architecture- and training-relative '
                      '(Asperti), not the dimension of Blocksworld. Per-hypothesis verdicts in criteria/metrics.')
        if status != 'completed':
            criteria = {k: None for k in criteria}
    res = results_header(run_dir) | dict(status=status, outcome=outcome, interpretation=interp,
                                         confirmation_accessed=not pilot, claims_changed=False, gates=gates,
                                         criteria=criteria, secondary=secondary, metrics=metrics)
    dump_json(au_rows, run_dir / 'au_table.json')
    dump_json({k: v.tolist() for k, v in active_sets.items()} | dict(_eval_set=key), run_dir / 'au_per_run.json')
    dump_json(res, run_dir / 'results.json')
    print(dict(status=status, outcome=outcome, gates=gates))


def variance_decomposition(gauss, beta1):
    """Descriptive two-way decomposition of AU at beta=1 (arch, lr; residual = seed)."""
    rs = [r for r in gauss if r['cell_id'] in beta1 and r['competence']]
    if len(rs) < 3:
        return None
    y = np.array([r['au'] for r in rs], dtype=float)
    arch = np.array([r['cell_id'].split('-')[2] for r in rs])
    lr = np.array([r['cell_id'].split('-lr')[1] for r in rs])
    total = float(((y - y.mean()) ** 2).sum())
    ss = {}
    for name, f in (('arch', arch), ('lr', lr)):
        ss[name] = float(sum((f == v).sum() * (y[f == v].mean() - y.mean()) ** 2 for v in np.unique(f)))
    cellm = {c: y[[r['cell_id'] == c for r in rs]].mean() for c in set(r['cell_id'] for r in rs)}
    ss['residual_seed'] = float(sum((r['au'] - cellm[r['cell_id']]) ** 2 for r in rs))
    ss['interaction'] = total - ss['arch'] - ss['lr'] - ss['residual_seed']
    return dict(total=total, **ss)


if __name__ == '__main__':
    main()
