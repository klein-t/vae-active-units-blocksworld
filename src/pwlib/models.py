"""Model families of the pw02 shared sweep (sweep_grid.json `families`).

Conventions (docs/METRICS_SPEC.md sections 3, 4, 8):
- KL summed over latents, reconstruction NLL summed over input dimensions,
  both averaged over the batch (disentanglement_lib convention, unnormalised beta).
- Bernoulli decoder for 36-bit data, Gaussian decoder with a scalar variance
  gamma for activations (learned log-parameter, init 0, floor; or fixed 1).
- beta-TCVAE: Chen et al. Eq. 4 with alpha = gamma = 1, minibatch-weighted
  sampling (MWS) with the official-code constants.
- Categorical VAE: Jang et al. recipe (relaxed Gumbel-softmax samples,
  analytic KL to the uniform categorical, tau = max(0.5, exp(-r t)) updated
  every 1,000 steps).
- Top-k SAE: Gao et al. 2024 recipe (pre-bias = train mean, unit-norm decoder
  rows renormalised each step, encoder init = decoder^T, AuxK on dead latents).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)


def mlp(d_in, hidden, d_out):
    """ReLU MLP with the given hidden widths (architectures A1-A4); dense layers only."""
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class Decoder(nn.Module):
    """Bernoulli (logits) or Gaussian (mean, scalar variance gamma) likelihood head."""

    def __init__(self, latent, hidden, d_out, likelihood, gamma='learned', log_gamma_floor=-9.0):
        super().__init__()
        if likelihood not in ('bernoulli', 'gaussian'):
            raise ValueError(likelihood)
        if gamma not in ('learned', 'fixed'):
            raise ValueError(gamma)
        self.net = mlp(latent, list(reversed(hidden)), d_out)
        self.likelihood = likelihood
        self.gamma_mode = gamma
        self.log_gamma_floor = log_gamma_floor
        self.log_gamma = nn.Parameter(torch.zeros(()), requires_grad=(gamma == 'learned'))

    def effective_log_gamma(self):
        return self.log_gamma.clamp(min=self.log_gamma_floor)

    def nll(self, x, out):
        """Per-example negative log-likelihood, summed over input dimensions."""
        if self.likelihood == 'bernoulli':
            return F.binary_cross_entropy_with_logits(out, x, reduction='none').sum(1)
        lg = self.effective_log_gamma()
        return 0.5 * (((x - out) ** 2).sum(1) / lg.exp() + x.shape[1] * (LOG2PI + lg))

    def mean(self, out):
        return torch.sigmoid(out) if self.likelihood == 'bernoulli' else out

    def forward(self, z):
        return self.net(z)


class GaussVAE(nn.Module):
    """beta-VAE (and beta-TCVAE body): N(0, I) prior, diagonal Gaussian posterior."""

    def __init__(self, d_in, hidden, latent, likelihood, gamma='learned', logvar_clamp=(-12.0, 6.0),
                 log_gamma_floor=-9.0):
        super().__init__()
        self.enc = mlp(d_in, hidden, 2 * latent)
        self.dec = Decoder(latent, hidden, d_in, likelihood, gamma, log_gamma_floor)
        self.latent = latent
        self.logvar_clamp = tuple(logvar_clamp)

    def encode(self, x):
        h = self.enc(x)
        mu, logvar = h[:, :self.latent], h[:, self.latent:]
        return mu, logvar.clamp(*self.logvar_clamp)

    def forward(self, x, gen=None):
        mu, logvar = self.encode(x)
        eps = torch.randn(mu.shape, generator=gen)
        z = mu + (0.5 * logvar).exp() * eps
        out = self.dec(z)
        kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1)
        return dict(mu=mu, logvar=logvar, z=z, out=out, rec_nll=self.dec.nll(x, out), kl_per_unit=kl)


def log_normal(z, mu, logvar):
    return -0.5 * (LOG2PI + logvar + (z - mu) ** 2 / logvar.exp())


def mws_log_densities(z, mu, logvar, dataset_size):
    """Minibatch-weighted sampling (Chen et al. Eq. S4, official-code constants).

    Returns per-sample log q(z) and log prod_d q(z_d), each with log(M N) subtracted
    once per density (D times in the product term)."""
    M = z.shape[0]
    lq = log_normal(z[:, None, :], mu[None, :, :], logvar[None, :, :])  # [M, M, D]
    c = math.log(M * dataset_size)
    log_qz = torch.logsumexp(lq.sum(2), dim=1) - c
    log_qz_prod = (torch.logsumexp(lq, dim=1) - c).sum(1)
    return log_qz, log_qz_prod


def mss_log_densities(z, mu, logvar, dataset_size):
    """Minibatch-stratified sampling (Chen et al. Appendix C.2, official `_log_importance_weight_matrix`)."""
    B = z.shape[0]
    N, M = dataset_size, B - 1
    if M < 1:
        raise ValueError('MSS needs a batch of at least 2')
    strat = (N - M) / (N * M)
    W = torch.full((B, B), 1.0 / M, dtype=z.dtype)
    W.view(-1)[::M + 1] = 1.0 / N
    W.view(-1)[1::M + 1] = strat
    W[M - 1, 0] = strat
    logW = W.log()
    lq = log_normal(z[:, None, :], mu[None, :, :], logvar[None, :, :])
    log_qz = torch.logsumexp(logW + lq.sum(2), dim=1)
    log_qz_prod = torch.logsumexp(logW[:, :, None] + lq, dim=1).sum(1)
    return log_qz, log_qz_prod


def tc_vae_terms(out, dataset_size):
    """Per-example index-code MI, total correlation and dimension-wise KL (MWS, Chen constants)."""
    z, mu, logvar = out['z'], out['mu'], out['logvar']
    log_qz_x = log_normal(z, mu, logvar).sum(1)
    log_pz = log_normal(z, torch.zeros_like(z), torch.zeros_like(z)).sum(1)
    log_qz, log_qz_prod = mws_log_densities(z, mu, logvar, dataset_size)
    return dict(mi=log_qz_x - log_qz, tc=log_qz - log_qz_prod, dwkl=log_qz_prod - log_pz)


class CatVAE(nn.Module):
    """n_vars categorical latents x n_classes, Gumbel-softmax relaxation, uniform prior."""

    def __init__(self, d_in, hidden, n_vars, n_classes, likelihood, gamma='learned', log_gamma_floor=-9.0):
        super().__init__()
        self.enc = mlp(d_in, hidden, n_vars * n_classes)
        self.dec = Decoder(n_vars * n_classes, hidden, d_in, likelihood, gamma, log_gamma_floor)
        self.n_vars, self.n_classes = n_vars, n_classes

    def logits(self, x):
        return self.enc(x).view(-1, self.n_vars, self.n_classes)

    def forward(self, x, tau, gen=None):
        logits = self.logits(x)
        u = torch.rand(logits.shape, generator=gen).clamp(1e-20, 1.0)
        g = -torch.log(-torch.log(u) + 1e-20)
        y = F.softmax((logits + g) / tau, dim=-1)
        out = self.dec(y.flatten(1))
        q = F.softmax(logits, dim=-1)
        kl = (q * (torch.log(q + 1e-20) - math.log(1.0 / self.n_classes))).sum(-1)  # [B, n_vars]
        return dict(probs=q, out=out, rec_nll=self.dec.nll(x, out), kl_per_var=kl)

    def decode_hard(self, probs):
        """Decoder output for the argmax one-hot code."""
        hard = F.one_hot(probs.argmax(-1), self.n_classes).float()
        return self.dec(hard.flatten(1))


def gumbel_tau(step, r=3e-5, every=1000, tau0=1.0, tau_min=0.5):
    """Jang et al. schedule: tau = max(tau_min, tau0 exp(-r t)), t updated every `every` steps."""
    t = (step // every) * every
    return max(tau_min, tau0 * math.exp(-r * t))


class TopKSAE(nn.Module):
    """Top-k sparse autoencoder (Gao et al. 2024) with AuxK on dead latents; MSE loss."""

    def __init__(self, d_in, n_latents, k, train_mean, dead_steps=1000, aux_coef=1.0 / 32):
        super().__init__()
        self.k, self.n_latents = k, n_latents
        self.k_aux = n_latents // 2
        self.dead_steps, self.aux_coef = dead_steps, aux_coef
        self.b_dec = nn.Parameter(torch.as_tensor(train_mean, dtype=torch.float32).clone())
        W = torch.randn(n_latents, d_in)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W)
        self.W_enc = nn.Parameter(W.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.register_buffer('steps_since_fired', torch.zeros(n_latents, dtype=torch.long))

    def pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x):
        p = self.pre(x)
        v, i = p.topk(self.k, dim=1)
        return torch.zeros_like(p).scatter(1, i, F.relu(v))

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        p = self.pre(x)
        v, i = p.topk(self.k, dim=1)
        z = torch.zeros_like(p).scatter(1, i, F.relu(v))
        xh = self.decode(z)
        mse = ((x - xh) ** 2).sum(1)
        out = dict(z=z, out=xh, rec_nll=mse, aux=torch.zeros_like(mse))
        if self.training:
            fired = (z > 0).any(0)
            self.steps_since_fired += 1
            self.steps_since_fired[fired] = 0
            dead = self.steps_since_fired >= self.dead_steps
            if dead.any():
                k_aux = min(self.k_aux, int(dead.sum()))
                pd = p.masked_fill(~dead[None, :], float('-inf'))
                va, ia = pd.topk(k_aux, dim=1)
                za = torch.zeros_like(p).scatter(1, ia, F.relu(va))
                e = (x - xh).detach()
                out['aux'] = ((e - za @ self.W_dec) ** 2).sum(1)
        return out

    @torch.no_grad()
    def renorm(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-12)


class PlainAE(nn.Module):
    """Deterministic autoencoder, same MLP and latent size as the gauss_vae reference; no KL."""

    def __init__(self, d_in, hidden, latent, likelihood):
        super().__init__()
        self.enc = mlp(d_in, hidden, latent)
        self.dec = Decoder(latent, hidden, d_in, likelihood, gamma='fixed')

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        if self.dec.likelihood == 'bernoulli':
            rec = F.binary_cross_entropy_with_logits(out, x, reduction='none').sum(1)
        else:
            rec = ((x - out) ** 2).sum(1)
        return dict(z=z, out=out, rec_nll=rec)
