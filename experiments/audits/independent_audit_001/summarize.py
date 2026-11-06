"""Independent audit 001: negative controls for my own checkers and the compact recomputed_values.json.

Negative controls: (1) my legality lookup accepts 0 of 100,000 uniform random 36-bit vectors and 0 of the
866 x 36 single-bit flips of legal states; (2) my best-F1 scorer gives Cov30 near 0 for the k=5 SAE codes
when the state rows of the labels are permuted (seed 0); (3) the pairing check would detect a Y[k+1]
pairing (every cached row differs from its next-state row; activation_check.json).
Run after the other scripts.
"""
import json
import sys

import numpy as np

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):  # usage without the data present
    print(__doc__)
    sys.exit(0)

import ind_common as C  # noqa: E402
import recompute_pw03a as P3  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    neg = {}
    neg['random_vectors_accepted'] = int(C.is_legal_rows(rng.integers(0, 2, (100_000, 36), dtype=np.int8)).sum())
    legal = np.load(C.PW02 / 'full_001/raw/data_context.npz')['legal'].astype(np.int8)
    flips = np.repeat(legal, 36, axis=0)
    flips[np.arange(len(flips)), np.tile(np.arange(36), len(legal))] ^= 1
    neg['single_bit_flips_accepted'] = int(C.is_legal_rows(flips).sum())
    # permuted-label control on the k=5 SAE, seed 1
    d = C.load_thesis_data()
    tr, va = C.split_indices()
    row_of = {r.tobytes(): i for i, r in enumerate(legal)}
    cd, cc = C.state_counts(d['Xtr'][va], d['Ytr'][va], row_of), C.state_counts(d['Xte'], d['Yte'], row_of)
    wd, wc = cd / cd.sum(), cc / cc.sum()
    codes, _ = P3.codes_for(C.PW02 / 'full_001', 'bw-topk_sae-k5-lr0.0003_s1', 'topk_sae', 5, legal)
    act = np.flatnonzero(wc @ (codes > 0) >= 0.005)
    Yp = legal.astype(np.float64)[rng.permutation(len(legal))]
    neg['sae_k5_s1_cov30_permuted_labels'] = P3.score(codes, act, Yp, wd, wc)['cov30']
    neg['sae_k5_s1_cov30_true_labels'] = P3.score(codes, act, legal.astype(np.float64), wd, wc)['cov30']
    act_chk = json.loads((C.OUT / 'activation_check.json').read_text())
    neg['cached_rows_differing_from_next_state'] = min(v[s]['ycur_is_not_ynext_rows'] for v in act_chk.values()
                                                       for s in ('train', 'val', 'conf'))

    fz = json.loads((C.OUT / 'freeze_check.json').read_text())
    dc = json.loads((C.OUT / 'data_check.json').read_text())
    p2 = json.loads((C.OUT / 'pw02a_recompute.json').read_text())
    p3 = json.loads((C.OUT / 'pw03a_recompute.json').read_text())
    p4 = json.loads((C.OUT / 'pw04a_recompute.json').read_text())
    f2 = p2['full_001']
    v = f2['verdicts']
    out = dict(
        **{'freeze': dict({e: dict(frozen_file_problems=fz[e]['frozen_file_problems'],
                             runs={r: {k: x[k] for k in ('run_freeze_matches', 'results_freeze_matches',
                                                         'manifest_mismatches')} for r, x in fz[e]['runs'].items()})
                     for e in list(fz)[:3]}, confirmation_sealing=fz['confirmation_sealing'],
                    pw04a_descriptive=fz['pw04a_descriptive'])},
        data=dict(bfs_legal=dc['bfs_legal_count'], transitions=dc['transitions_Yk_is_state_after_token_k'],
                  weights_full=dc['weights_full_001']),
        activations={s: dict(conf_acts_max_abs_diff=x['conf']['acts_max_abs_diff'], ycur_equal=x['conf']['ycur_equal'],
                             head_vs_current=x['conf']['head_vs_current_bitwise'], head_vs_next=x['conf']['head_vs_next_bitwise'],
                             b_lin_max_abs_diff=x['conf']['b_lin_max_abs_diff']) for s, x in
                     json.loads((C.OUT / 'activation_check.json').read_text()).items()},
        pw02a=dict(runs_recomputed_full=f2['n_runs_recomputed'], runs_recomputed_pilot=p2['pilot_001']['n_runs_recomputed'],
                   mismatches_full=f2['mismatches'], mismatches_pilot=p2['pilot_001']['mismatches'],
                   max_mu_abs_diff_vs_stored=f2['max_mu_abs_diff_vs_stored'],
                   run_seeds_unique=f2['run_seeds_unique'], seed_overlap_pilot_full=p2['seed_overlap_pilot_full'],
                   not_converged=f2['not_converged'],
                   H22={a: {k: x[k] for k in ('A_bw', 'A_code10', 'A_prod5', 'accounts')} for a, x in v['H2.2']['per_arch'].items()}
                   | dict(part_i=v['H2.2']['part_i'], part_i_max=v['H2.2']['part_i_max'], verdict=v['H2.2']['verdict']),
                   H23=dict(medians={c: x['median'] for c, x in v['H2.3']['cells'].items()}, between=v['H2.3']['between'],
                            max_within=v['H2.3']['max_within'], verdict=v['H2.3']['verdict']),
                   H23_beta_monotone=dict(n=v['H2.3_secondary_beta_monotone']['n'], verdict=v['H2.3_secondary_beta_monotone']['verdict']),
                   H24={s: {k: x[k] for k in ('learned', 'fixed', 'med_learned', 'med_fixed', 'ok')} for s, x in
                        v['H2.4']['per_source'].items()} | dict(verdict=v['H2.4']['verdict']),
                   experiment_competence=v['experiment_competence']['verdict'],
                   verdicts_excluding_not_converged={k: x['verdict'] for k, x in f2['verdicts_excluding_not_converged'].items()}),
        pw03a=dict(mismatches=p3['mismatches'], h32=p3['h32'], h32_dev_selected=p3['h32_dev_selected'],
                   mono_frac={t: [r['main']['mono_frac'] for r in x['rows'] if r['competence']] for t, x in p3['cells'].items()}),
        pw04a=dict(baselines={k: x for k, x in p4['baselines'].items() if k != 'block_marginals'},
                   recorded_baselines=p4['pilot_001']['recorded_baselines'],
                   pilot={r: {k: x[k] for k in ('prior', 'recorded_prior', 'aggpost', 'recorded_aggpost', 'posterior_mean_exact_recon')}
                          for r, x in p4['pilot_001']['rows'].items()},
                   pilot_required_cells_passing=p4['pilot_required_cells_passing'],
                   full_beta1=p4['full_beta1_summary']),
        negative_controls=neg)
    (C.OUT / 'recomputed_values.json').write_text(json.dumps(C.jsonable(out), indent=1))
    print(json.dumps(C.jsonable(neg), indent=1))


if __name__ == '__main__':
    main()
