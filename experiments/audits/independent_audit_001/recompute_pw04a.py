"""Independent audit 001, check 5: pw04a legality with my own sampler and an independent legality check
(lookup in my BFS-derived 866-state set; pwlib's checker is not used).

Per model (N = 20,000 each, numpy RNG unrelated to the repo's seeds):
- prior: z ~ N(0, I) over all latent units, mode decoding (Bernoulli mean > 0.5, i.e. logit > 0);
- aggregate posterior (the pw04 competence quantity): x ~ evaluation state distribution (val weights in
  the pilot, test-file weights for the descriptive full pass), z = mu + exp(logvar / 2) * eps, mode decoding;
- posterior-mean decoding (z = mu) as a reference that must match the pw02a reconstruction competence.
Models: the four required beta=1 pilot cells (seed 101; the registered stop) and the two beta=0.25
pilot cells; the 30 beta=1 and 20 beta=0.25 runs of pw02a full_001 (descriptive pass).
Exact baselines B1 and B2 recomputed from my own train-split state frequencies (B2 by enumerating the
7,776 per-block placements and testing legality by set lookup). Writes pw04a_recompute.json.
"""
import json
import sys

import numpy as np
import torch

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):  # usage without the data present
    print(__doc__)
    sys.exit(0)

import ind_common as C  # noqa: E402

N = 20_000


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def baselines(legal, w_train):
    p_bit = w_train @ legal.astype(np.float64)
    L = legal.astype(np.float64)
    B1 = float(np.prod(np.where(L == 1, p_bit, 1 - p_bit), axis=1).sum())
    # per-block marginals over 6 places: table, hand, on each other block
    places = {}
    for x in C.BLOCKS:
        cols = [C.f(f'on_table({x})'), C.f(f'holding({x})')] + [C.f(f'on({x},{y})') for y in C.BLOCKS if y != x]
        places[x] = (cols, w_train @ legal[:, cols].astype(np.float64))
    import itertools
    LS = C.legal_set()
    B2 = B2u = 0.0
    for combo in itertools.product(range(6), repeat=5):
        s = np.zeros(36, dtype=np.int8)
        p = 1.0
        for x, c in zip(C.BLOCKS, combo):
            cols, marg = places[x]
            s[cols[c]] = 1
            p *= marg[c]
        held = [x for x in C.BLOCKS if s[C.f(f'holding({x})')]]
        for y in C.BLOCKS:
            covered = any(s[C.f(f'on({x},{y})')] for x in C.BLOCKS if x != y)
            s[C.f(f'clear({y})')] = int(not covered and y not in held)
        s[C.f('hand_empty')] = int(not held)
        if s.tobytes() in LS:
            B2 += p
            B2u += 1 / 7776
    return dict(B1=B1, B2=B2, B2_uniform=B2u, block_marginals={x: v[1].tolist() for x, v in places.items()})


def legality(run_dir, run, w_eval, legal, rng):
    _, sd = C.load_ckpt(run_dir, run)
    with torch.no_grad():
        x = torch.as_tensor(legal, dtype=torch.float32)
        mu, lv = C.gauss_encode(sd, x)
        L = mu.shape[1]
        zp = torch.as_tensor(rng.standard_normal((N, L)), dtype=torch.float32)
        prior_bits = (C.decode(sd, zp) > 0).numpy()
        idx = rng.choice(len(legal), size=N, p=w_eval / w_eval.sum())
        eps = torch.as_tensor(rng.standard_normal((N, L)), dtype=torch.float32)
        za = mu[idx] + torch.exp(0.5 * lv[idx]) * eps
        agg_bits = (C.decode(sd, za) > 0).numpy()
        mean_bits = (C.decode(sd, mu) > 0).numpy()
    kp, ka = int(C.is_legal_rows(prior_bits).sum()), int(C.is_legal_rows(agg_bits).sum())
    exact_recon = float(w_eval @ (mean_bits == legal.astype(bool)).all(1) / w_eval.sum())
    return dict(prior=kp / N, prior_wilson=wilson(kp, N), aggpost=ka / N, aggpost_wilson=wilson(ka, N),
                posterior_mean_exact_recon=exact_recon)


def main():
    rng = np.random.default_rng(20261022)
    d = C.load_thesis_data()
    tr, va = C.split_indices()
    out = {}
    for run_id in ('pilot_001', 'full_001'):
        run_dir = C.PW02 / run_id
        legal = np.load(run_dir / 'raw/data_context.npz')['legal'].astype(np.int8)
        row_of = {r.tobytes(): i for i, r in enumerate(legal)}
        w_train = C.state_counts(d['Xtr'][tr], d['Ytr'][tr], row_of)
        w_train /= w_train.sum()
        if run_id == 'pilot_001':
            w_eval = C.state_counts(d['Xtr'][va], d['Ytr'][va], row_of)
            recf = C.ROOT / 'experiments/pw04a_sample_legality_amended/results/pilot_001/results.json'
            runs = [f'bw-gauss_vae-{c}_s101' for c in ('A1-b1-lr0.0003', 'A2-b1-lr0.0003', 'A2-b1-lr0.003',
                                                        'A4-b1-lr0.0003', 'A2-b0.25-lr0.0003', 'A2-b0.25-lr0.003')]
            out['baselines'] = baselines(legal, w_train)
        else:
            w_eval = C.state_counts(d['Xte'], d['Yte'], row_of)
            recf = C.ROOT / 'experiments/pw04a_sample_legality_amended/descriptive/results/pw02a_full_001/results.json'
            runs = [f'bw-gauss_vae-{a}-b{b}-lr{lr}_s{s}' for b in ('1', '0.25') for a in ('A1', 'A2', 'A4')
                    for lr in ('0.0003', '0.003') for s in range(1, 6)]
            runs = [r for r in runs if (run_dir / 'models' / f'{r}.pt').exists()]
        rec = json.loads(recf.read_text())['metrics']
        rows = {}
        for r in runs:
            m = legality(run_dir, r, w_eval, legal, rng)
            pm = rec['per_model'][r]
            m['recorded_prior'] = pm['prior_mode']['rate']
            m['recorded_aggpost'] = pm['aggpost_mode']['rate']
            se = lambda p: np.sqrt(p * (1 - p) / N)  # noqa: E731
            m['prior_diff_z'] = float((m['prior'] - m['recorded_prior']) / np.hypot(se(m['prior']), se(m['recorded_prior'])))
            m['aggpost_diff_z'] = float((m['aggpost'] - m['recorded_aggpost']) /
                                        np.hypot(se(m['aggpost']), se(m['recorded_aggpost'])))
            m['competence_aggpost_ge_095'] = m['aggpost'] >= 0.95
            rows[r] = m
        out[run_id] = dict(rows=rows, recorded_baselines=rec['baselines']['exact'])
        print(run_id)
        for r, m in rows.items():
            print(f"  {r:40s} prior {m['prior']:.4f} (rec {m['recorded_prior']:.4f}, z {m['prior_diff_z']:+.2f})  "
                  f"agg {m['aggpost']:.4f} (rec {m['recorded_aggpost']:.4f}, z {m['aggpost_diff_z']:+.2f})  "
                  f"mean-recon {m['posterior_mean_exact_recon']:.4f}")
    b1 = [m for r, m in out['full_001']['rows'].items() if '-b1-' in r]
    out['full_beta1_summary'] = dict(n=len(b1), below_B2=sum(m['prior'] < out['baselines']['B2'] for m in b1),
                                     max_abs_z=max(max(abs(m['prior_diff_z']), abs(m['aggpost_diff_z']))
                                                   for m in out['full_001']['rows'].values()),
                                     any_aggpost_pass=[r for r, m in out['full_001']['rows'].items()
                                                       if '-b1-' in r and m['competence_aggpost_ge_095']])
    out['pilot_required_cells_passing'] = {r: m['competence_aggpost_ge_095'] for r, m in out['pilot_001']['rows'].items()
                                           if '-b1-' in r}
    print({k: v for k, v in out['baselines'].items() if k != 'block_marginals'}, out['pilot_001']['recorded_baselines'])
    print(out['full_beta1_summary'], out['pilot_required_cells_passing'])
    (C.OUT / 'pw04a_recompute.json').write_text(json.dumps(C.jsonable(out), indent=1))


if __name__ == '__main__':
    main()
