"""Training and end-of-run evaluation for one (cell, seed) of the pw02 shared sweep.

Every run: CPU, float32, one thread, Adam, constant lr, fixed number of steps
(no early stopping), grad-norm clip, final checkpoint retained. The run's RNG
seed is derived from (cell_id, seed) so runs are independent across cells;
init, minibatch order and reparameterisation noise all come from it.
"""
import copy
import hashlib
import math
import platform
import time

import numpy as np
import torch

from . import metrics as M
from .models import CatVAE, GaussVAE, PlainAE, TopKSAE, gumbel_tau, mss_log_densities, mws_log_densities, \
    tc_vae_terms

BATCH_TC_DIAG = 512
N_TC_DIAG_BATCHES = 8


def run_seed(cell_id, seed):
    """Deterministic 31-bit RNG seed for one run."""
    return int.from_bytes(hashlib.sha256(f'{cell_id}|{seed}'.encode()).digest()[:4], 'big') & 0x7FFFFFFF


def likelihood_of(substrate):
    return 'gaussian' if substrate.startswith('b_') else 'bernoulli'


def latent_shape(cell):
    lat = cell['latent']
    if isinstance(lat, str):
        nv, nc = (int(v) for v in lat.split('x'))
        return nv, nc
    return int(lat)


def build_model(cell, grid, d_in, train_mean):
    fam, tr = cell['family'], grid['training']
    lik = likelihood_of(cell['substrate'])
    if fam in ('gauss_vae', 'tc_vae'):
        return GaussVAE(d_in, grid['architectures'][cell['arch']]['hidden'], latent_shape(cell), lik,
                        cell.get('gamma', 'fixed') if lik == 'gaussian' else 'fixed',
                        tr['logvar_clamp'], tr['log_gamma_floor'])
    if fam == 'cat_vae':
        nv, nc = latent_shape(cell)
        return CatVAE(d_in, grid['architectures'][cell['arch']]['hidden'], nv, nc, lik,
                      cell.get('gamma', 'fixed') if lik == 'gaussian' else 'fixed', tr['log_gamma_floor'])
    if fam == 'topk_sae':
        return TopKSAE(d_in, latent_shape(cell), int(cell['k']), train_mean)
    if fam == 'plain_ae':
        return PlainAE(d_in, grid['architectures'][cell['arch']]['hidden'], latent_shape(cell), lik)
    raise ValueError(f'Unknown family {fam}')


def kl_weight(step, steps, warmup):
    """Multiplier on the KL-side penalty. warmup=None (default): 1. warmup={'schedule': 'linear',
    'fraction': f}: min(1, step / (f * steps)), i.e. beta rises linearly from 0 to its target over
    the first f of the steps (pre-registered pw02 A3 contingency, PROTOCOL.md section 10)."""
    if warmup is None:
        return 1.0
    if warmup['schedule'] != 'linear' or not 0 < warmup['fraction'] <= 1:
        raise ValueError(f'Unsupported KL warm-up: {warmup}')
    return min(1.0, step / (warmup['fraction'] * steps))


def batch_loss(model, cell, x, step, n_train_points, gen, kl_w=1.0):
    """Mean training objective over the batch and its logged components. kl_w scales the whole
    KL-side penalty (beta-VAE: beta*KL; beta-TCVAE: MI + beta*TC + DWKL; categorical: beta*KL)."""
    fam = cell['family']
    if fam == 'gauss_vae':
        out = model(x, gen)
        kl = out['kl_per_unit'].sum(1)
        loss = (out['rec_nll'] + kl_w * cell['beta'] * kl).mean()
        return loss, dict(rec=out['rec_nll'].mean().item(), kl=kl.mean().item(), kl_weight=kl_w)
    if fam == 'tc_vae':
        out = model(x, gen)
        t = tc_vae_terms(out, n_train_points)
        loss = (out['rec_nll'] + kl_w * (t['mi'] + cell['beta'] * t['tc'] + t['dwkl'])).mean()
        return loss, dict(rec=out['rec_nll'].mean().item(), mi=t['mi'].mean().item(),
                          tc_mws_chen=t['tc'].mean().item(), dwkl=t['dwkl'].mean().item(), kl_weight=kl_w)
    if fam == 'cat_vae':
        out = model(x, gumbel_tau(step), gen)
        kl = out['kl_per_var'].sum(1)
        loss = (out['rec_nll'] + kl_w * cell['beta'] * kl).mean()
        return loss, dict(rec=out['rec_nll'].mean().item(), kl=kl.mean().item(), tau=gumbel_tau(step),
                          kl_weight=kl_w)
    if fam == 'topk_sae':
        out = model(x)
        loss = (out['rec_nll'] + model.aux_coef * out['aux']).mean()
        return loss, dict(rec=out['rec_nll'].mean().item(), aux=out['aux'].mean().item(),
                          dead=int((model.steps_since_fired >= model.dead_steps).sum()))
    if fam == 'plain_ae':
        out = model(x)
        loss = out['rec_nll'].mean()
        return loss, dict(rec=loss.item())
    raise ValueError(fam)


@torch.no_grad()
def eval_objective(model, cell, data, key, step):
    """Deterministic-noise objective on one evaluation set (weighted mean), for training curves."""
    x, w = data.eval_sets()[key]
    g = torch.Generator().manual_seed(0)
    model.eval()
    fam = cell['family']
    if fam in ('gauss_vae', 'tc_vae'):
        out = model(x, g)
        per = out['rec_nll'] + cell['beta'] * out['kl_per_unit'].sum(1)
    elif fam == 'cat_vae':
        out = model(x, gumbel_tau(step), g)
        per = out['rec_nll'] + cell['beta'] * out['kl_per_var'].sum(1)
    else:
        per = model(x)['rec_nll']
    model.train()
    return float((per.double().numpy() * w).sum())


def train(cell, seed, data, grid, steps=None, log_every=None, snapshot_fraction=None):
    """Train one run; returns (model, log), or (model, log, snapshot) when snapshot_fraction is given,
    where snapshot = dict(step, state_dict) is a copy of the weights at round(snapshot_fraction * steps)
    (pw02a convergence check). Raises FloatingPointError on a non-finite loss."""
    tr = grid['training']
    steps = int(steps if steps is not None else tr['steps'])
    log_every = int(log_every if log_every is not None else tr['eval_every_steps'])
    rs = run_seed(cell['cell_id'], seed)
    torch.manual_seed(rs)
    gen = torch.Generator().manual_seed(rs)
    model = build_model(cell, grid, data.d_in, data.train_mean)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cell['lr'])
    curve, t0 = [], time.time()
    dev_key = 'dev'
    snap_step = None if snapshot_fraction is None else int(round(snapshot_fraction * steps))
    snapshot = None
    for step in range(1, steps + 1):
        x = data.sample(tr['batch_size'], gen)
        loss, parts = batch_loss(model, cell, x, step, data.n_train_points, gen,
                                 kl_weight(step, steps, tr['kl_warmup']))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, tr['grad_clip_norm'])
        opt.step()
        if isinstance(model, TopKSAE):
            model.renorm()
        if step == snap_step:
            snapshot = dict(step=step, state_dict=copy.deepcopy(model.state_dict()))
        if step % log_every == 0 or step == steps:
            curve.append(dict(step=step, train_loss=loss.item(), dev_objective=eval_objective(model, cell, data, dev_key, step),
                              **parts))
    beta = cell.get('beta')
    lat = latent_shape(cell)
    beta_norm = beta * lat / data.d_in if beta is not None and isinstance(lat, int) else None
    log = dict(run_seed=rs, steps=steps, batch_size=tr['batch_size'], wall_seconds=time.time() - t0,
               kl_warmup=tr['kl_warmup'],
               beta=beta, beta_norm=beta_norm,
               device='cpu', precision='float32', threads=torch.get_num_threads(), torch=torch.__version__,
               python=platform.python_version(), n_train_points=data.n_train_points, curve=curve)
    if hasattr(model, 'dec') and model.dec.likelihood == 'gaussian':
        lg = float(model.dec.log_gamma.detach())
        log['log_gamma'] = lg
        log['log_gamma_at_floor'] = bool(lg <= model.dec.log_gamma_floor)
    if snapshot_fraction is not None:
        return model, log, snapshot
    return model, log


# ----------------------------------------------------------------- evaluation

def bits_metrics(xhat_bits, x, w):
    eq = (xhat_bits == (x > 0.5)).astype(np.float64)
    return dict(bitwise=float((eq.mean(1) * w).sum()), exact=float((eq.min(1) * w).sum()))


def probe_metrics(recon, labels, probe, w):
    """Current-state probe (fitted on original standardized activations) applied to reconstructions."""
    bits = (recon @ probe['W'].T + probe['b'] > 0)
    eq = (bits == labels.astype(bool)).astype(np.float64)
    return bits.astype(np.int8), dict(probe_bitwise=float((eq.mean(1) * w).sum()),
                                      probe_exact=float((eq.min(1) * w).sum()))


def r2(x, xhat, w):
    mean = (x * w[:, None]).sum(0)
    sse = (((x - xhat) ** 2).sum(1) * w).sum()
    sst = (((x - mean) ** 2).sum(1) * w).sum()
    return float(1 - sse / sst)


@torch.no_grad()
def evaluate(model, cell, data, probe=None, labels=None):
    """End-of-run operands and competence metrics on every evaluation set.

    Returns (arrays, metrics): arrays keyed '<set>_<name>' (numpy), metrics keyed by set."""
    model.eval()
    fam, arrays, out_metrics = cell['family'], {}, {}
    binary = data.kind in ('table', 'bernoulli')
    for key, (x, w) in data.eval_sets().items():
        xn = x.numpy()
        arrays[f'{key}_weights'] = w
        m = {}
        if fam in ('gauss_vae', 'tc_vae'):
            mu, logvar = model.encode(x)
            mean_out = model.dec(mu)
            recon = model.dec.mean(mean_out).numpy()
            arrays[f'{key}_mu'] = mu.numpy()
            arrays[f'{key}_logvar'] = logvar.numpy()
            kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1)
            arrays[f'{key}_kl_per_unit'] = (kl.double().numpy() * w[:, None]).sum(0)
            base = model.dec.nll(x, mean_out).double().numpy()
            gain = []
            for j in range(mu.shape[1]):
                mz = mu.clone()
                mz[:, j] = 0
                gain.append(float(((model.dec.nll(x, model.dec(mz)).double().numpy() - base) * w).sum()))
            arrays[f'{key}_recon_gain'] = np.array(gain)
            m['min_unit_kl'] = float(arrays[f'{key}_kl_per_unit'].min())
            m['tc'] = tc_diagnostics(mu, logvar, w, data.n_train_points)
        elif fam == 'cat_vae':
            probs = model.logits(x).softmax(-1)
            recon = model.dec.mean(model.decode_hard(probs)).numpy()
            arrays[f'{key}_probs'] = probs.numpy()
            q = probs.double()
            kl = (q * (torch.log(q + 1e-20) - math.log(1.0 / model.n_classes))).sum(-1).numpy()
            arrays[f'{key}_kl_per_var'] = (kl * w[:, None]).sum(0)
            m['min_unit_kl'] = float(arrays[f'{key}_kl_per_var'].min())
        elif fam == 'topk_sae':
            z = model.encode(x)
            recon = model.decode(z).numpy()
            arrays[f'{key}_codes'] = z.numpy()
        else:
            z = model.enc(x)
            out = model.dec(z)
            recon = (torch.sigmoid(out) if model.dec.likelihood == 'bernoulli' else out).numpy()
            arrays[f'{key}_codes'] = z.numpy()
        if binary:
            bits = recon > 0.5
            arrays[f'{key}_recon_bits'] = bits.astype(np.int8)
            m.update(bits_metrics(bits, xn, w))
        else:
            pbits, pm = probe_metrics(recon, labels[key], probe, w)
            arrays[f'{key}_recon_probe_bits'] = pbits
            arrays[f'{key}_labels'] = labels[key].astype(np.int8)
            m.update(pm)
            m['r2'] = r2(xn.astype(np.float64), recon.astype(np.float64), w)
        out_metrics[key] = m
    model.train()
    return arrays, out_metrics


def tc_diagnostics(mu, logvar, w, n_train_points):
    """Total correlation of the aggregate posterior in nats: MWS with Chen's constants and MSS,
    on batches of 512 drawn from the evaluation distribution (fixed seed)."""
    g = torch.Generator().manual_seed(0)
    p = torch.as_tensor(w)
    res = dict(tc_mws_chen=[], tc_mss=[])
    for _ in range(N_TC_DIAG_BATCHES):
        idx = torch.multinomial(p, BATCH_TC_DIAG, replacement=True, generator=g)
        m, lv = mu[idx].double(), logvar[idx].double()
        z = m + (0.5 * lv).exp() * torch.randn(m.shape, generator=g, dtype=torch.float64)
        a, b = mws_log_densities(z, m, lv, n_train_points)
        res['tc_mws_chen'].append(float((a - b).mean()))
        a, b = mss_log_densities(z, m, lv, n_train_points)
        res['tc_mss'].append(float((a - b).mean()))
    return {k: float(np.mean(v)) for k, v in res.items()} | dict(batch=BATCH_TC_DIAG, batches=N_TC_DIAG_BATCHES,
                                                                n_train_points=n_train_points)


def au_summary(arrays, key, thresholds=(0.01,) + M.AU_ROBUSTNESS):
    """AU counts at the frozen thresholds from stored Gaussian operands (used in run logs)."""
    mu, w = arrays[f'{key}_mu'], arrays[f'{key}_weights']
    return {str(t): M.active_units(mu, w, t)[0] for t in thresholds}
