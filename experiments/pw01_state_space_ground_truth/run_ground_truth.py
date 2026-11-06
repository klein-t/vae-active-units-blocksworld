"""pw01: exact, model-free ground truth for the 5-block Blocksworld state space.

Usage: uv run python experiments/pw01_state_space_ground_truth/run_ground_truth.py --run-id ID
Writes results/<ID>/ (RUN.json must already exist: `uv run python src/pwlib/runio.py start ...`).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import binom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pwlib import blocksworld as bw  # noqa: E402

EXP = Path(__file__).resolve().parent
DATA = Path(os.environ.get('PW_BLOCKSWORLD_DATA', ROOT / 'data' / 'blocksworld'))  # earlier thesis data
SPLITS = ('train', 'test')
N_RANDOM = 200_000
RANDOM_SEED = 0
I3D_RADII = [(3, 6), (4, 8), (5, 10), (6, 12), (7, 14), (8, 16)]
I3D_CONTROL_DIMS = (8, 10)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def entropy(counts):
    p = np.asarray(counts, dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def pack(states):
    """Unique int64 code per 36-bit vector (bit j = fact j)."""
    return (states.astype(np.int64) << np.arange(bw.N_FACTS, dtype=np.int64)).sum(-1)


def affine_dim(m):
    m = np.asarray(m, dtype=float)
    return int(np.linalg.matrix_rank(m - m[0]))


def i3d(dist, r, R, model, d_grid):
    """Binomial-likelihood ID for discrete metrics (Macocco et al. 2023): solve V(r,d)/V(R,d) = sum k / sum n.

    k_i, n_i: points within distance r and R of point i, counting i itself (as the ball volume counts its
    centre, so a full hypercube recovers d exactly under the hypercube model). Volume models:
    'lattice' = L1 ball in Z^d, V = sum_k 2^k C(d,k) C(r,k); 'hypercube' = Hamming ball in {0,1}^d,
    V = sum_{k<=r} C(d,k). Exploratory re-implementation; returns the grid d closest to the ratio.
    """
    k = int((dist <= r).sum())
    n = int((dist <= R).sum())
    p_hat = k / n

    def vol(rad, d):
        ks = np.arange(rad + 1)[:, None]
        if model == 'lattice':
            return (2.0 ** ks * binom(d[None, :], ks) * binom(rad, ks)).sum(0)
        return binom(d[None, :], ks).sum(0)

    ratio = vol(r, d_grid) / vol(R, d_grid)
    j = int(np.argmin(np.abs(ratio - p_hat)))
    return dict(r=r, R=R, k_total=k, n_total=n, p_hat=p_hat, d_hat=float(d_grid[j]),
                ratio_at_d_hat=float(ratio[j]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    t0 = time.time()
    out = EXP / 'results' / args.run_id
    run = json.loads((out / 'RUN.json').read_text())
    design = json.loads((EXP / 'DESIGN.json').read_text())
    assert run['run_id'] == args.run_id

    # ---------------- task gate: inputs, fact order, vocabulary, padding ----------------
    hashes = {}
    for name, expected in design['inputs'].items():
        got = sha256(DATA / name)
        if got != expected:
            raise ValueError(f'Input hash mismatch: {name}')
        hashes[name] = got
    meta = json.loads((DATA / 'blocksworld_metadata.json').read_text())
    bw.check_fact_order(meta['probe_indices'])
    vocab = meta['vocab']
    if set(k for k in vocab if k.startswith('(')) != set(bw.action_names()):
        raise ValueError('Action vocabulary differs from pwlib.blocksworld.action_names()')
    if vocab['[PAD]'] != bw.PAD_TOKEN:
        raise ValueError('PAD token is not 52')
    task = {}

    # ---------------- numerical gate: three counts and checker tests ----------------
    L = bw.enumerate_legal_bruteforce()
    L_bfs = bw.enumerate_legal_bfs()
    n_legal = len(L)
    n_empty = int(L[:, bw.HAND_EMPTY].sum())
    formula = bw.legal_count_formula()
    counts_agree = bool(n_legal == len(L_bfs) == formula == 866 and np.array_equal(L, L_bfs)
                        and n_empty == bw.lah_total(5) == 501)
    legal_set = {v.tobytes() for v in L}
    flip_tests, flip_accepted, flip_mismatch = 0, 0, 0
    for v in L:
        for i in range(bw.N_FACTS):
            w = v.copy()
            w[i] ^= 1
            flip_tests += 1
            ok = bw.is_legal(w)
            flip_accepted += int(ok)
            flip_mismatch += int(ok != (w.tobytes() in legal_set))
    rng = np.random.default_rng(RANDOM_SEED)
    rand = rng.integers(0, 2, size=(N_RANDOM, bw.N_FACTS)).astype(np.int8)
    rand_accepted = int(sum(bw.is_legal(r) for r in rand))
    rand_in_set = int(np.isin(pack(rand), pack(L)).sum())
    all_legal_pass = all(bw.is_legal(v) for v in L)
    numerical = dict(counts_agree=counts_agree, all_enumerated_pass_checker=all_legal_pass,
                     single_flip_tests=flip_tests, single_flip_accepted=flip_accepted,
                     single_flip_checker_vs_set_mismatch=flip_mismatch,
                     random_vectors=N_RANDOM, random_accepted_by_checker=rand_accepted,
                     random_in_legal_set=rand_in_set)
    numerical_gate = bool(counts_agree and all_legal_pass and flip_accepted == 0 and flip_mismatch == 0
                          and rand_accepted == rand_in_set)

    # ---------------- data: legality, coverage, parity, transitions ----------------
    codes_L = pack(L)
    order = np.argsort(codes_L)
    sorted_codes = codes_L[order]
    succ = np.full((n_legal, len(vocab)), -1, dtype=np.int64)  # successor state index per action token
    facts_of = [frozenset(f for f, b in zip(bw.FACTS, v) if b) for v in L]
    for s_idx, s in enumerate(facts_of):
        for a, tok in vocab.items():
            if a.startswith('('):
                t = bw.apply_action(s, a)
                if t is not None:
                    w = np.array([int(f in t) for f in bw.FACTS], dtype=np.int64)
                    succ[s_idx, tok] = order[np.searchsorted(sorted_codes, pack(w[None])[0])]

    per = {}
    count_cols = ['all', 'train', 'test', 'even_k', 'odd_k', 'k_le_30', 'k_gt_30']
    counts = {c: np.zeros(n_legal, dtype=np.int64) for c in count_cols}
    illegal_total, pad_nonzero, parity_violations, transition_failures, n_trans = 0, 0, 0, 0, 0
    seq_lengths = {}
    for split in SPLITS:
        X = np.load(DATA / f'blocksworld_{split}_X.npy')
        Y = np.load(DATA / f'blocksworld_{split}_Y.npy')
        if X.shape[1] != 61 or Y.shape[1:] != (61, 36):
            raise ValueError(f'Unexpected shapes in {split}')
        valid = X != bw.PAD_TOKEN
        if not np.array_equal(np.cumsum(~valid, axis=1) > 0, ~valid):
            raise ValueError(f'PAD is not a suffix in {split}')
        if not (X[:, 0] == vocab['[SOS]']).all():
            raise ValueError(f'Position 0 is not SOS in {split}')
        pad_nonzero += int(Y[~valid].sum())
        n_actions = valid.sum(1) - 1
        seq_lengths[split] = dict(min_actions=int(n_actions.min()), max_actions=int(n_actions.max()),
                                  n_sequences=int(len(X)))
        codes = pack(Y)
        pos = np.searchsorted(sorted_codes, codes)
        pos = np.clip(pos, 0, n_legal - 1)
        is_known = sorted_codes[pos] == codes
        illegal_total += int((valid & ~is_known).sum())
        idx = np.where(valid & is_known, order[pos], -1)
        kgrid = np.broadcast_to(np.arange(61), X.shape)
        parity_violations += int((valid & (Y[..., bw.HAND_EMPTY] != (kgrid % 2 == 0))).sum())
        # transitions: action token X[:,k] applied to state at k-1 must give state at k
        prev, nxt, act = idx[:, :-1], idx[:, 1:], X[:, 1:]
        tv = valid[:, 1:]
        pred = succ[np.where(prev >= 0, prev, 0), act.clip(0, len(vocab) - 1)]
        transition_failures += int((tv & ((prev < 0) | (pred != nxt))).sum())
        n_trans += int(tv.sum())
        sel = {
            'all': valid, split: valid, 'even_k': valid & (kgrid % 2 == 0), 'odd_k': valid & (kgrid % 2 == 1),
            'k_le_30': valid & (kgrid <= 30), 'k_gt_30': valid & (kgrid > 30),
        }
        for c, mask in sel.items():
            counts[c] += np.bincount(idx[mask & (idx >= 0)], minlength=n_legal)
        per[split] = int(valid.sum())

    n_positions = int(counts['all'].sum())
    task.update(input_sha256=hashes, fact_order_matches_metadata=True, pad_token=bw.PAD_TOKEN,
                y_sum_at_pad_positions=pad_nonzero, valid_positions=n_positions, valid_positions_by_split=per,
                sequence_lengths=seq_lengths, transitions_checked=n_trans,
                transitions_not_matching_strips=transition_failures)
    task_gate = bool(pad_nonzero == 0 and transition_failures == 0 and n_positions == 1_020_005)

    data_all_legal = illegal_total == 0
    coverage = int((counts['all'] > 0).sum())
    hand_empty_parity = parity_violations == 0

    # ---------------- entropies and conditional quantities ----------------
    H = {c: entropy(counts[c]) for c in count_cols}
    n_even, n_odd = int(counts['even_k'].sum()), int(counts['odd_k'].sum())
    p_even = n_even / n_positions
    H_cond_parity = p_even * H['even_k'] + (1 - p_even) * H['odd_k']
    he = L[:, bw.HAND_EMPTY].astype(bool)
    support = dict(even_k=int((counts['even_k'] > 0).sum()), odd_k=int((counts['odd_k'] > 0).sum()),
                   even_k_all_arm_empty=bool((counts['even_k'][~he] == 0).all()),
                   odd_k_all_holding=bool((counts['odd_k'][he] == 0).all()))

    # ---------------- per-block categorical factorisation ----------------
    C = np.array([bw.positions_from_state(v) for v in L])  # (866, 5), values 0..5

    def factor_stats(w):
        w = np.asarray(w, dtype=float)
        p = w / w.sum()
        marg = [np.bincount(C[:, b], weights=p, minlength=6) for b in range(5)]
        Hm = [entropy(m) for m in marg]
        Hj = entropy(w)
        mi = np.zeros((5, 5))
        for a in range(5):
            for b in range(5):
                if a != b:
                    joint = np.zeros((6, 6))
                    np.add.at(joint, (C[:, a], C[:, b]), p)
                    mi[a, b] = Hm[a] + Hm[b] - entropy(joint.ravel())
        return dict(marginal_entropies_bits=dict(zip(bw.BLOCKS, Hm)), sum_marginal_entropies_bits=float(sum(Hm)),
                    joint_entropy_bits=Hj, total_correlation_bits=float(sum(Hm) - Hj),
                    pairwise_mutual_information_bits={f'{bw.BLOCKS[a]},{bw.BLOCKS[b]}': float(mi[a, b])
                                                      for a in range(5) for b in range(a + 1, 5)},
                    mean_pairwise_mi_bits=float(mi[np.triu_indices(5, 1)].mean()))

    factorisation = dict(
        definition='Block x in {T (table), H (hand), on(x, y) for the 4 other y}: 5 variables x 6 values',
        n_vars=5, cardinality=6, product_space=6 ** 5, product_bits=5 * math.log2(6),
        legal_fraction_of_product=n_legal / 6 ** 5,
        arm_empty_view=dict(definition='Given hand empty, each block has 5 values (T or on one of 4 blocks)',
                            product_space=5 ** 5, product_bits=5 * math.log2(5),
                            legal_fraction_of_product=n_empty / 5 ** 5),
        empirical_all=factor_stats(counts['all']),
        empirical_even_k=factor_stats(counts['even_k']),
        empirical_odd_k=factor_stats(counts['odd_k']),
        uniform_over_legal=factor_stats(np.ones(n_legal)),
    )

    # ---------------- linear structure ----------------
    Lf = L.astype(float)
    no_he = np.delete(Lf, bw.HAND_EMPTY, axis=1)
    linear = dict(
        rank_all=int(np.linalg.matrix_rank(Lf)), affine_dim_all=affine_dim(Lf),
        rank_all_without_hand_empty_column=int(np.linalg.matrix_rank(no_he)),
        affine_dim_all_without_hand_empty_column=affine_dim(no_he),
        affine_dim_arm_empty=affine_dim(Lf[he]), affine_dim_holding=affine_dim(Lf[~he]),
        rank_arm_empty=int(np.linalg.matrix_rank(Lf[he])), rank_holding=int(np.linalg.matrix_rank(Lf[~he])),
        nonconstant_facts_all=int((Lf.min(0) != Lf.max(0)).sum()),
        nonconstant_facts_arm_empty=int((Lf[he].min(0) != Lf[he].max(0)).sum()),
        nonconstant_facts_holding=int((Lf[~he].min(0) != Lf[~he].max(0)).sum()),
    )

    # ---------------- discrete intrinsic dimension (exploratory) ----------------
    dist = (L[:, None, :] != L[None, :, :]).sum(-1)
    d_grid = np.arange(1.0, 60.0001, 0.01)
    i3d_res = {m: [i3d(dist, r, R, m, d_grid) for r, R in I3D_RADII] for m in ('lattice', 'hypercube')}
    controls = {}
    for d in I3D_CONTROL_DIMS:
        cube = ((np.arange(2 ** d)[:, None] >> np.arange(d)) & 1)
        cd = (cube[:, None, :] != cube[None, :, :]).sum(-1)
        controls[f'full_hypercube_d{d}'] = {m: [i3d(cd, r, R, m, d_grid) for r, R in ((1, 2), (2, 4))]
                                            for m in ('lattice', 'hypercube')}
    hamming_spectrum = dict(zip(*[a.tolist() for a in np.unique(dist[np.triu_indices(n_legal, 1)], return_counts=True)]))

    # ---------------- dimension candidates ----------------
    i3d_h = [x['d_hat'] for x in i3d_res['hypercube']]
    candidates = {
        'bits_log2_legal': dict(value=math.log2(n_legal), hand_empty_free=True,
                                definition='log2 of the number of legal states (866): bits to index a uniform legal state'),
        'bits_min_binary_code': dict(value=math.ceil(math.log2(n_legal)), hand_empty_free=True,
                                     definition='ceil(log2 866): fewest binary latents that can index every legal state'),
        'bits_entropy_empirical': dict(value=H['all'], hand_empty_free=True,
                                       definition='Shannon entropy of the state distribution over all valid positions of thesis train+test'),
        'bits_log2_arm_empty': dict(value=math.log2(n_empty), hand_empty_free=False,
                                    definition='log2 501: bits to index a legal state given hand empty (even step index)'),
        'bits_log2_holding': dict(value=math.log2(n_legal - n_empty), hand_empty_free=False,
                                  definition='log2 365: bits to index a legal state given a held block (odd step index)'),
        'bits_min_binary_code_given_parity': dict(value=max(math.ceil(math.log2(n_empty)), math.ceil(math.log2(n_legal - n_empty))),
                                                  hand_empty_free=False,
                                                  definition='max over parity classes of ceil(log2 #states in class) = 9'),
        'bits_entropy_given_parity': dict(value=H_cond_parity, hand_empty_free=False,
                                          definition='H(state | step parity) on the data = p_even H_even + p_odd H_odd'),
        'bits_entropy_even_k': dict(value=H['even_k'], hand_empty_free=False, definition='H(state) at even step index k (arm empty)'),
        'bits_entropy_odd_k': dict(value=H['odd_k'], hand_empty_free=False, definition='H(state) at odd step index k (holding)'),
        'factors_per_block_categorical': dict(value=5, hand_empty_free=False,
                                              definition='Number of per-block 6-valued position variables; hand_empty is a function of them (not a free factor). Variables are dependent (total correlation > 0).'),
        'factor_bits_per_block_categorical': dict(value=5 * math.log2(6), hand_empty_free=False,
                                                  definition='5 log2 6: capacity of the per-block product space (7776 cells, 11.1% legal)'),
        'factor_bits_per_block_categorical_arm_empty': dict(value=5 * math.log2(5), hand_empty_free=False,
                                                            definition='5 log2 5: per-block product space given hand empty (3125 cells, 16.0% legal)'),
        'binary_facts_total': dict(value=bw.N_FACTS, hand_empty_free=True, definition='Number of binary facts in the encoding'),
        'binary_facts_excluding_hand_empty': dict(value=bw.N_FACTS - 1, hand_empty_free=False,
                                                  definition='Facts other than hand_empty'),
        'rank_state_matrix': dict(value=linear['rank_all'], hand_empty_free=True, definition='Rank of the 866 x 36 legal-state matrix'),
        'affine_dim_legal_states': dict(value=linear['affine_dim_all'], hand_empty_free=True,
                                        definition='Dimension of the affine hull of the 866 legal states in R^36 (linear-PCA count of nonzero-variance directions)'),
        'affine_dim_without_hand_empty_column': dict(value=linear['affine_dim_all_without_hand_empty_column'], hand_empty_free=False,
                                                     definition='Affine hull dimension after dropping the hand_empty column (it is affinely 1 - sum holding)'),
        'affine_dim_arm_empty': dict(value=linear['affine_dim_arm_empty'], hand_empty_free=False,
                                     definition='Affine hull dimension of the 501 arm-empty states (even steps)'),
        'affine_dim_holding': dict(value=linear['affine_dim_holding'], hand_empty_free=False,
                                   definition='Affine hull dimension of the 365 holding states (odd steps)'),
        'i3d_hypercube_range': dict(value=[min(i3d_h), max(i3d_h)], hand_empty_free=True,
                                    definition='Exploratory: range over radii (r, 2r), r=3..8, of the binomial discrete-ID estimate with Hamming-hypercube ball volumes on the 866 legal states (uniform). Scale-dependent; see i3d section.'),
    }

    # ---------------- outputs ----------------
    np.save(out / 'legal_states_866.npy', L)
    shutil.copyfile(ROOT / 'src/pwlib/blocksworld.py', out / 'blocksworld_legality.py')
    shape_counts = Counter()
    with open(out / 'state_frequency_table.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['state_index', 'bits', 'positions', 'hand_empty', 'n_towers'] + [f'count_{c}' for c in count_cols])
        for i, v in enumerate(L):
            pos = ' '.join(f'{x}:{bw.POSITION_VALUES[x][c]}' for x, c in zip(bw.BLOCKS, C[i]))
            towers = int(v[:5].sum())
            shape_counts[(int(v[bw.HAND_EMPTY]), towers)] += 1
            wr.writerow([i, ''.join(map(str, v)), pos, int(v[bw.HAND_EMPTY]), towers] + [int(counts[c][i]) for c in count_cols])
    cnt = np.sort(counts['all'])[::-1]
    freq_summary = dict(max=int(cnt[0]), min=int(cnt[-1]), median=float(np.median(cnt)),
                        top10_share=float(cnt[:10].sum() / cnt.sum()),
                        legal_states_by_hand_empty_and_n_towers={f'{k[0]},{k[1]}': v for k, v in sorted(shape_counts.items())})
    dims = dict(experiment_id=design['experiment_id'], run_id=args.run_id,
                note='Candidate "true dimension" targets frozen before any VAE count (P1). hand_empty_free=False marks candidates that do not count hand_empty (fixed by step parity) as a free factor.',
                candidates=candidates, linear=linear,
                i3d=dict(radii=I3D_RADII, legal_states=i3d_res, positive_controls=controls,
                         hamming_distance_spectrum_pairs=hamming_spectrum,
                         caveat='Own re-implementation of the Macocco et al. (2207.09688) binomial estimator, not checked against DADApy; full hypercubes are the positive control for the hypercube volume model.'))
    (out / 'dimension_candidates.json').write_text(json.dumps(dims, indent=2) + '\n')

    criteria = dict(count_agreement=counts_agree, data_all_legal=bool(data_all_legal),
                    data_cover_all_legal=coverage == n_legal, hand_empty_equals_step_parity=bool(hand_empty_parity))
    gates = dict(task=task_gate, numerical=numerical_gate)
    ok = all(gates.values())
    supported = ok and all(criteria.values())
    metrics = dict(
        n_legal=n_legal, n_arm_empty=n_empty, n_holding=n_legal - n_empty, lah5=bw.lah_total(5), lah4=bw.lah_total(4),
        n_illegal_positions_in_data=illegal_total, legal_states_covered=coverage,
        hand_empty_parity_violations=parity_violations, positions_even_k=n_even, positions_odd_k=n_odd,
        entropy_bits=H, entropy_bits_given_parity=H_cond_parity, log2_legal=math.log2(n_legal),
        log2_arm_empty=math.log2(n_empty), log2_holding=math.log2(n_legal - n_empty), parity_support=support,
        factorisation=factorisation, linear=linear, frequency=freq_summary, task=task, numerical=numerical,
        runtime_seconds=None,
        compute=dict(device='cpu', precision='int8/float64 (numpy)', threads=1, python=platform.python_version(),
                     numpy=np.__version__),
    )
    metrics['runtime_seconds'] = round(time.time() - t0, 1)
    results = dict(
        experiment_id=design['experiment_id'], run_id=args.run_id, freeze_sha256=run['freeze_sha256'],
        status='completed' if ok else 'technical_stop',
        outcome=('supported' if supported else 'unsupported') if ok else 'untested',
        interpretation=(
            f'Exact, model-free and scoped to 5-block Blocksworld with this 36-fact encoding and the thesis data files: '
            f'{n_legal} legal states ({n_empty} arm-empty + {n_legal - n_empty} holding) by brute force, BFS and the Lah formula; '
            f'the data contain {illegal_total} illegal states and cover {coverage}/{n_legal} legal states over {n_positions} valid positions; '
            f'hand_empty equals step parity at every valid position ({parity_violations} violations). '
            f'Empirical H = {H["all"]:.3f} bits vs log2 {n_legal} = {math.log2(n_legal):.3f}; H(state | parity) = {H_cond_parity:.3f}. '
            f'Per-block categorical variables are dependent: total correlation {factorisation["empirical_all"]["total_correlation_bits"]:.3f} bits. '
            'Dimension candidates are definitions, not a claim about what a VAE should find.'),
        confirmation_accessed=True, claims_changed=False, gates=gates, criteria=criteria, metrics=metrics)
    (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(dict(gates=gates, criteria=criteria, outcome=results['outcome'],
                          runtime_seconds=metrics['runtime_seconds']), indent=2))


if __name__ == '__main__':
    main()
