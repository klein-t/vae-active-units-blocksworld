"""Independent audit 001, check 6 (activation pairing): regenerate the substrate-(b) caches.

For each of the five exp46 sources: own subsampling from the protocol (seeds 0/1/2 for 3,000 train /
500 val / 300 confirmation sequences of the pw02 splits), own TransformerLens forward via run_with_cache
(pwlib uses run_with_hooks), own position table (valid X[k] != PAD and next row Y[k+1] non-PAD), current
state = Y[k]. Compared with the cached pw02a full_001 arrays: activations (after my own train
standardization), ycur labels, (seq, k) positions, b_lin (own lstsq). Also: exp46 head accuracy against
Y[k] vs Y[k+1] (one-step-ahead convention) and the stored current-state probe against current vs next
state on val. Writes activation_check.json.
"""
import json
import os
import re

import sys

import numpy as np
import torch

if __name__ == '__main__' and {'-h', '--help'} & set(sys.argv[1:]):  # usage without the data present
    print(__doc__)
    sys.exit(0)

import ind_common as C  # noqa: E402

EXP46 = os.environ.get('PW_TRANSFORMER_CHECKPOINTS', str(C.ROOT / 'data' / 'transformers'))  # earlier thesis work
SOURCES = {'w64': 'd4_w64_s1', 'w128': 'd4_w128_s1', 'w128s2': 'd4_w128_s2', 'w128s3': 'd4_w128_s3',
           'w256': 'd4_w256_s1'}


def load_model(name):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    depth, width = map(int, re.match(r'd(\d+)_w(\d+)_s\d+$', name).groups())
    cfg = HookedTransformerConfig(d_model=width, n_layers=depth, n_heads=4, d_head=width // 4, d_mlp=4 * width,
                                  d_vocab=53, n_ctx=60, act_fn='gelu', normalization_type='LN', device='cpu')
    m = HookedTransformer(cfg)
    sd = torch.load(f'{EXP46}/{name}.pt', map_location='cpu', weights_only=True)
    tsd = {k[len('transformer.'):]: v for k, v in sd.items() if k.startswith('transformer.')}
    missing, unexpected = m.load_state_dict(tsd, strict=False)
    assert not unexpected and all(k.endswith(('IGNORE', 'mask')) for k in missing), (missing, unexpected)
    m.eval()
    return m, depth, sd['probe_head.weight'], sd['probe_head.bias']


@torch.no_grad()
def table(model, depth, W, b, X, Y):
    hook = f'blocks.{depth - 1}.hook_resid_post'
    H = []
    for i in range(0, len(X), 200):
        _, cache = model.run_with_cache(torch.as_tensor(X[i:i + 200, :60], dtype=torch.long), names_filter=hook)
        H.append(cache[hook])
    H = torch.cat(H).numpy()
    seqs, ks = [], []
    for i in range(len(X)):
        for k in range(60):
            if X[i, k] != C.PAD and Y[i, k + 1].any():
                seqs.append(i)
                ks.append(k)
    seqs, ks = np.array(seqs), np.array(ks)
    acts = H[seqs, ks]
    head = (acts @ W.numpy().T + b.numpy()) > 0
    return dict(acts=acts, ycur=Y[seqs, ks].astype(np.int8), ynext=Y[seqs, ks + 1].astype(np.int8), seq=seqs, k=ks,
                head=head)


def main():
    d = C.load_thesis_data()
    tr, va = C.split_indices()
    splits = dict(train=(d['Xtr'][tr], d['Ytr'][tr], 0, 3000), val=(d['Xtr'][va], d['Ytr'][va], 1, 500),
                  conf=(d['Xte'], d['Yte'], 2, 300))
    run_dir = C.PW02 / 'full_001'
    out = {}
    for src, name in SOURCES.items():
        model, depth, W, b = load_model(name)
        cached = np.load(run_dir / f'raw/activations_{src}.npz')
        probes = np.load(run_dir / f'raw/probes_{src}.npz')
        tabs = {}
        for s, (X, Y, seed, n) in splits.items():
            idx = np.sort(np.random.default_rng(seed).choice(len(X), n, replace=False))
            tabs[s] = table(model, depth, W, b, X[idx], Y[idx])
        mean, std = tabs['train']['acts'].mean(0), tabs['train']['acts'].std(0)
        r = {}
        for s, t in tabs.items():
            a = (t['acts'] - mean) / std
            r[s] = dict(rows=len(a), rows_cached=len(cached[f'{s}_acts']),
                        acts_max_abs_diff=float(np.abs(a - cached[f'{s}_acts']).max()) if len(a) == len(cached[f'{s}_acts']) else None,
                        ycur_equal=bool(np.array_equal(t['ycur'], cached[f'{s}_ycur'])),
                        ycur_is_not_ynext_rows=float((t['ycur'] != t['ynext']).any(1).mean()),
                        positions_equal=bool(np.array_equal(t['seq'], cached[f'{s}_seq']) and np.array_equal(t['k'], cached[f'{s}_k'])),
                        head_vs_current_bitwise=float((t['head'] == t['ycur'].astype(bool)).mean()),
                        head_vs_next_bitwise=float((t['head'] == t['ynext'].astype(bool)).mean()))
            t['std'] = a
        D = np.c_[np.ones(len(tabs['train']['ycur'])), tabs['train']['ycur']].astype(np.float64)
        B, *_ = np.linalg.lstsq(D, tabs['train']['std'].astype(np.float64), rcond=None)
        for s, t in tabs.items():
            lin = np.c_[np.ones(len(t['ycur'])), t['ycur']] @ B
            r[s]['b_lin_max_abs_diff'] = float(np.abs(lin - cached[f'{s}_lin']).max())
        vA, vy, vn = cached['val_acts'], cached['val_ycur'].astype(bool), cached['val_ynext'].astype(bool)
        pred = (vA @ probes['cur_W'].T + probes['cur_b']) > 0
        r['stored_current_probe_val'] = dict(vs_current_bitwise=float((pred == vy).mean()),
                                             vs_current_exact=float((pred == vy).all(1).mean()),
                                             vs_next_bitwise=float((pred == vn).mean()))
        out[src] = r
        print(src, json.dumps(C.jsonable(r)))
    (C.OUT / 'activation_check.json').write_text(json.dumps(C.jsonable(out), indent=1))


if __name__ == '__main__':
    main()
