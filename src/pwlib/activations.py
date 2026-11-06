"""Activations of the exp46 transformers (earlier thesis work) for pw02 substrate (b).

Verified loading recipe (pw02 PROTOCOL.md section 3): TransformerLens
HookedTransformer, n_heads=4, d_head=W/4, d_mlp=4W, d_vocab=53, n_ctx=60,
gelu, LN; state-dict keys `transformer.*` and `probe_head.*`. Input X[:, :60];
hook `blocks.{depth-1}.hook_resid_post` (final layer, never hard-coded).
The exp46 head is trained one step ahead (target Y[k+1]); activations at
position k are paired with the CURRENT state Y[k]. Positions whose
next-state target Y[k+1] is the all-zero PAD row are excluded.
"""
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from . import blocksworld as bw

# exp46 transformer checkpoints d4_w*_s*.pt of the earlier thesis work
EXP46_MODELS = Path(os.environ.get('PW_TRANSFORMER_CHECKPOINTS',
                                   Path(__file__).resolve().parents[2] / 'data' / 'transformers'))
# Short source name -> exp46 checkpoint name.
SOURCES = {'w64': 'd4_w64_s1', 'w128': 'd4_w128_s1', 'w128s2': 'd4_w128_s2', 'w128s3': 'd4_w128_s3',
           'w256': 'd4_w256_s1'}
SUBSAMPLE = {'train': (0, 3000), 'val': (1, 500), 'conf': (2, 300)}  # split -> (seed, n sequences)
PROBE_FIT_N = 20_000
PROBE_FIT_SEED = 0
PROBE_C = 1.0
PROBE_MAX_ITER = 2000
PCA_VAR = 0.99
K_PROBE_TOL = 0.005
EXTRACT_BATCH = 250


def checkpoint_path(name):
    return EXP46_MODELS / f'{name}.pt'


def checkpoint_key(name):
    """Key of a checkpoint in the recorded hashes (DESIGN.json external_inputs)."""
    return f'$PW_TRANSFORMER_CHECKPOINTS/{name}.pt'


def load_exp46(name):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    depth, width = map(int, re.match(r'd(\d+)_w(\d+)_s\d+$', name).groups())
    cfg = HookedTransformerConfig(d_model=width, n_layers=depth, n_heads=4, d_head=width // 4, d_mlp=4 * width,
                                  d_vocab=53, n_ctx=60, act_fn='gelu', normalization_type='LN', device='cpu')

    class ProbePredictionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = HookedTransformer(cfg)
            self.probe_head = nn.Linear(width, bw.N_FACTS)

    m = ProbePredictionModel()
    sd = torch.load(checkpoint_path(name), map_location='cpu', weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.endswith(('IGNORE', 'mask'))]
    if bad or unexpected:
        raise ValueError(f'{name}: state-dict mismatch, missing={bad}, unexpected={list(unexpected)}')
    m.eval()
    return m, depth, width, list(missing)


@torch.no_grad()
def residuals(model, depth, X):
    """Final-layer residual stream [N, 60, W] and exp46 head logits [N, 60, 36] for X[:, :60]."""
    hook = f'blocks.{depth - 1}.hook_resid_post'
    x = torch.as_tensor(X[:, :60], dtype=torch.long)
    H, L = [], []
    for i in range(0, len(x), EXTRACT_BATCH):
        cap = {}

        def grab(t, hook):
            cap['h'] = t.detach().clone()
            return t
        model.transformer.run_with_hooks(x[i:i + EXTRACT_BATCH], fwd_hooks=[(hook, grab)], return_type=None)
        H.append(cap['h'])
        L.append(model.probe_head(cap['h']))
    return torch.cat(H).numpy(), torch.cat(L).numpy(), hook


def position_table(X, Y, H, logits):
    """Kept positions: valid (X != PAD) and next-state row Y[k+1] not the all-zero PAD row."""
    valid = X[:, :60] != bw.PAD_TOKEN
    keep = valid & Y[:, 1:61].any(-1)
    if not np.array_equal(keep, valid & (X[:, 1:61] != bw.PAD_TOKEN)):
        raise ValueError('PAD conventions disagree between X and Y')
    seq, k = np.nonzero(keep)
    return dict(acts=H[keep].astype(np.float32), ycur=Y[:, :60][keep].astype(np.int8),
                ynext=Y[:, 1:61][keep].astype(np.int8), head_logits=logits[keep].astype(np.float32),
                seq=seq.astype(np.int32), k=k.astype(np.int32))


def subsample_sequences(n, split):
    seed, size = SUBSAMPLE[split]
    if size > n:
        raise ValueError(f'Cannot draw {size} sequences from {n}')
    return np.sort(np.random.default_rng(seed).choice(n, size, replace=False))


def fit_probes(A, T, seed=PROBE_FIT_SEED, n_fit=PROBE_FIT_N):
    """36 logistic probes (sklearn, C=1, max_iter=2000) on a fixed subsample of rows."""
    idx = np.random.default_rng(seed).choice(len(A), min(n_fit, len(A)), replace=False)
    W, b = np.zeros((T.shape[1], A.shape[1])), np.zeros(T.shape[1])
    for j in range(T.shape[1]):
        y = T[idx, j]
        if y.min() == y.max():
            raise ValueError(f'Probe target {bw.FACTS[j]} constant in the fit subsample')
        clf = LogisticRegression(C=PROBE_C, max_iter=PROBE_MAX_ITER).fit(A[idx], y)
        W[j], b[j] = clf.coef_[0], clf.intercept_[0]
    return dict(W=W, b=b)


def probe_accuracy(probe, A, T):
    pred = (A @ probe['W'].T + probe['b']) > 0
    eq = pred == T.astype(bool)
    return dict(bitwise=float(eq.mean()), exact=float(eq.all(1).mean()))


def standardize_stats(A):
    mean, std = A.mean(0), A.std(0)
    if (std <= 1e-8).any():
        raise ValueError('Constant activation dimension; cannot standardize')
    return mean, std


def b_lin_map(A_train_std, Y_train):
    """Least-squares map B with A ~ [1, Y] B (min-norm lstsq)."""
    D = np.c_[np.ones(len(Y_train)), Y_train.astype(np.float64)]
    B, *_ = np.linalg.lstsq(D, A_train_std.astype(np.float64), rcond=None)
    return B


def apply_b_lin(B, Y):
    return (np.c_[np.ones(len(Y)), Y.astype(np.float64)] @ B).astype(np.float32)


def pca_basis(A):
    """Principal axes of standardized train activations (rows of Vt) and explained variance ratios."""
    Ac = A - A.mean(0)
    _, s, Vt = np.linalg.svd(Ac.astype(np.float64), full_matrices=False)
    var = s ** 2
    return Vt, var / var.sum()


def k_pca99(ratios):
    return int(np.searchsorted(np.cumsum(ratios), PCA_VAR) + 1)


def probe_q_grid(width):
    return [q for q in list(range(1, 41)) + [48, 56, 64, 96, 128, 192, 256] if q <= width]


def k_probe(A_train, T_train, A_val, T_val, acc_all, Vt, mean):
    """Smallest q whose top-q-PC probes reach val bitwise accuracy within 0.005 of all-dims probes."""
    scan = []
    for q in probe_q_grid(A_train.shape[1]):
        P = Vt[:q].T
        pr = fit_probes((A_train - mean) @ P, T_train)
        acc = probe_accuracy(pr, (A_val - mean) @ P, T_val)['bitwise']
        scan.append(dict(q=q, val_bitwise=acc))
        if acc >= acc_all - K_PROBE_TOL:
            return q, scan
    return A_train.shape[1], scan
