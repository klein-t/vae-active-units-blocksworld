"""Cross-check pwlib.metrics MIG/DCI against disentanglement_lib (commit 86a644d4ed35) on synthetic data.

Not part of pytest (needs a throwaway env). Setup used on 2026-10-14:
  git clone https://github.com/google-research/disentanglement_lib && git checkout 86a644d4ed35
  uv venv env --python 3.12 && uv pip install --python env/bin/python numpy scipy scikit-learn==1.9.1 six absl-py
  gin stub dir: gin/__init__.py (REQUIRED sentinel, identity `configurable`) and empty gin/tf.py
  (avoids TensorFlow; the metric functions themselves are dlib's unmodified code).
Usage: env/bin/python -I dlib_crosscheck.py <dlib_repo> <gin_stub_dir> <pw_src> <out_json>
mig.gin settings are applied by hand (20-bin histogram discretizer). DCI is compared twice: with
dlib's GradientBoostingClassifier given random_state=0 (the project deviation; must match exactly)
and with dlib unseeded (differences only from tree tie-breaking). Result: dlib_crosscheck_result.json.
"""
import functools, json, sys
if {"-h", "--help"} & set(sys.argv[1:]):
    sys.exit(print(__doc__))
repo, stubs, pwsrc, out = sys.argv[1:5]
sys.path[:0] = [stubs, repo, pwsrc]
import numpy as np
from disentanglement_lib.evaluation.metrics import mig, dci, utils
from pwlib import metrics as M

# mig.gin: discretizer_fn = @histogram_discretizer, num_bins = 20
mig.utils.make_discretizer = lambda t: utils._histogram_discretize(t, 20)

def synthetic(kind, seed, n=10000, K=5, J=10):
    rng = np.random.default_rng(seed)
    if kind == 'binary_facts':      # Blocksworld-like: rare binary factors
        y = (rng.random((n, K)) < rng.uniform(0.03, 0.5, K)).astype(int)
    else:
        y = rng.integers(0, 6, size=(n, K))
    W = rng.normal(size=(K, J)); W[:, :K] += 3 * np.eye(K) * (kind != 'rotated')
    codes = y @ W + rng.normal(scale=0.5, size=(n, J))
    if kind == 'rotated':
        Q, _ = np.linalg.qr(rng.normal(size=(J, J))); codes = codes @ Q
    return codes, y

res = {}
for kind in ('aligned', 'rotated', 'binary_facts'):
    for seed in (0, 1, 2):
        codes, y = synthetic(kind, seed)
        d_mig = mig._compute_mig(codes.T, y.T)['discrete_mig']
        p_mig = M.mig(codes, y)['mig']
        tr, te = slice(0, 7000), slice(7000, None)
        # dlib with the project deviation applied (fixed random_state): must match exactly
        orig = dci.ensemble.GradientBoostingClassifier
        dci.ensemble.GradientBoostingClassifier = functools.partial(orig, random_state=0)
        d_dci = dci._compute_dci(codes[tr].T, y[tr].T, codes[te].T, y[te].T)
        dci.ensemble.GradientBoostingClassifier = orig
        np.random.seed(seed)
        d_dci_unseeded = dci._compute_dci(codes[tr].T, y[tr].T, codes[te].T, y[te].T)
        p = M.dci(codes[tr], y[tr], codes[te], y[te])
        res[f'{kind}_s{seed}'] = dict(
            mig_dlib=d_mig, mig_pw=p_mig, mig_abs_diff=abs(d_mig - p_mig),
            dci_dlib_seeded={k: float(v) for k, v in d_dci.items()},
            dci_pw={k: p[k] for k in ('disentanglement', 'completeness', 'informativeness_train', 'informativeness_test')},
            dci_max_abs_diff_seeded=max(abs(float(d_dci[k]) - p[k]) for k in d_dci),
            dci_max_abs_diff_vs_unseeded_dlib=max(abs(float(d_dci_unseeded[k]) - p[k]) for k in d_dci_unseeded))
summary = dict(max_mig_abs_diff=max(v['mig_abs_diff'] for v in res.values()),
               max_dci_abs_diff_seeded=max(v['dci_max_abs_diff_seeded'] for v in res.values()),
               max_dci_abs_diff_vs_unseeded=max(v['dci_max_abs_diff_vs_unseeded_dlib'] for v in res.values()))
json.dump(dict(summary=summary, cases=res), open(out, 'w'), indent=1)
print(json.dumps(summary, indent=1))
for k, v in res.items():
    print(k, 'MIG', round(v['mig_dlib'], 6), round(v['mig_pw'], 6), 'D', round(v['dci_dlib_seeded']['disentanglement'], 6), round(v['dci_pw']['disentanglement'], 6))
