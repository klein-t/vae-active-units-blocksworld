# Metrics specification

**Written:** 2026-10-14, from the full texts and reference code listed below
(read in full for the relevant sections; quotes are verbatim from the PDF text
extraction, so some spacing and symbols are flattened). Page numbers are PDF
page numbers of the arXiv/OpenReview version named in each section.

Lines marked **Project note** are our own implementation decisions or
consequences for Blocksworld; they are not claims made by the sources.

Reference code read (commit = repository HEAD on 2026-10-14):

| Repo | Commit | Files read |
|------|--------|------------|
| google-research/disentanglement_lib | `86a644d4ed35` | `evaluation/metrics/{mig,dci,utils}.py`, `methods/unsupervised/vae.py`, `postprocessing/methods.py`, `config/unsupervised_study_v1/metric_configs/{mig,dci}.gin`, `postprocess_configs/mean.gin` |
| rtqichen/beta-tcvae (Chen et al. official) | `1a3577dbb146` | `vae_quant.py`, `disentanglement_metrics.py`, `metric_helpers/mi_metric.py` |
| adamkarvonen/SAE_BoardGameEval (Karvonen et al. official) | `7187d4a19ffa` | `circuits/{f1_analysis,analysis,eval_sae_as_classifier,pipeline_config,utils}.py` |
| ericjang/gumbel-softmax (Jang et al. official) | `4deb8bd47884` | `Categorical VAE.ipynb` |

scikit-learn behaviour quoted below was checked against the installed
docstrings of scikit-learn 1.9.1.

---

## 1. Active units (Burda, Grosse & Salakhutdinov, arXiv 1509.00519, v4, ICLR 2016)

**Definition** (Sec. 5.2, p. 7):

> "we measured activity of a latent dimension u using the statistic
> A_u = Cov_x( E_{u~q(u|x)}[u] ). We defined the dimension u to be active if
> A_u > 10^-2."

- A_u is the variance, over data points x, of the posterior mean of unit u.
  For a Gaussian encoder E_{q(u|x)}[u] = mu_u(x), so
  `A_u = Var_x(mu_u(x))`. Threshold: strictly greater than 0.01.
- Justification given (p. 7-8): "The distribution of A_u for a trained model
  consisted of two widely separated modes" (Appendix C, p. 12, shows a
  histogram of log A_u), and "we evaluated all models with the inactive
  dimensions removed. In all cases, this changed the test log-likelihood by
  less than 0.06 nats."
- The paper does not state which split (train or test) A_u is computed on.
- Burda et al. attribute inactivity to the objective, not only optimisation
  (p. 8): "inactivation of the latent dimensions is driven by the objective
  functions rather than by optimization issues. On the other hand,
  optimization also appears to play a role".

**Reference implementation:** none in disentanglement_lib. Implement directly:
`A = mu.var(axis=0)` (population variance over the evaluation set),
`active = A > 0.01`.

**Project note:** the threshold is in prior units (prior N(0, I)); it is not
scale-invariant (see the Asperti scaling remark below). Also report the
histogram of log A_u, since the threshold's justification is bimodality; if
the distribution is not bimodal, the count is threshold-dependent and must be
reported as such. Compute on the held-out set of distinct legal states, and
separately on the transformer-activation set.

## 2. Asperti's diagnostic (arXiv 1812.07238, "Sparsity in Variational Autoencoders")

**Statistics** (Sec. IV.A, p. 4, Table I): per latent variable, "computed
over the full dataset: the mean of its variance (that we expect to be around 1,
since it should be normally distributed), and the mean of the computed
variance sigma^2_theta(X) (that we expect to be a small value, close to 0)."
Table I columns are "variance" (variance of the latent's mean over data) and
"mean(sigma^2_theta(X))".

**Inactive signature** (p. 4): "their variance is very low (in practice, they
always have value 0), while the variance sigma^2_theta(X) computed by the
network is around 1 for each X." Example rows (Table I, p. 4): inactive
`8.85e-05 | 0.980`; active `0.974 | 0.0087`.

**Stationary condition** (Sec. VI, p. 7, Eqs. 12-13): averaging the per-datum
KL over the data gives a minimum when

> sigma_hat^2(X) + mu_hat^2(X) = 1  (12), "or simply" sigma_hat^2(X) + sigma^2 = 1  (13)

where sigma_hat^2(X) is the mean posterior variance and sigma^2 the variance of
the latent's mean over the data ("assuming the mean of the latent variable to
be 0, mu_hat^2(X) is just the (global) variance sigma^2", p. 7). Empirical
check (p. 7): "the average sum of sigma_hat^2(X) and sigma^2 ... is 0.997 with a
variance of 0.00048", and "condition 13 holds both for active and inactive
variables", e.g. a variable with global variance 0.797 and mean local variance
0.238.

- The condition needs the network to be "sufficiently trained" (p. 7).
- Derivation caveat stated in the paper (p. 7): it replaces the geometric mean
  of the variances by the arithmetic mean ("If we replace the geometric mean
  with an arithmetic mean, we get an expression essentially equivalent").
- Asperti gives **no numeric threshold**; active vs inactive is read off the two
  columns.
- Scaling caveat (Sec. VI, p. 6): "trying to compute relevant statistics for the
  posterior distribution Q(z|X) of latent variables without some kind of
  regularization constraint does not make much sense", because mean and
  standard deviation can be rescaled by a constant and undone in the next layer.
- Second diagnostic, "reconstruction gain" (p. 7): "the difference between the
  reconstruction error when the latent variable is zeroed out with respect to
  the case when it is normally taken into account". Generation check (p. 4,
  Fig. 6): digits generated with the inactive variables zeroed out "are
  indistinguishable".
- Sparsity is linked to the KL term specifically (p. 7): "If we replace the
  Kullback-Leibler component with some other regularization, for instance just
  considering a quadratic penalty on latent variables, the sparsity phenomenon
  disappears."

**Project note:** log, per latent, `var_mu = mu.var(0)`, `mean_s2 = exp(logvar).mean(0)`,
their sum (Eq. 13 check, expected about 1), and the reconstruction gain. Use
Burda's A_u > 0.01 as the counting rule and Asperti's pair as the consistency
check; flag any latent where the sum departs from 1 by more than a frozen
tolerance (to be fixed in the pw design, not here) as "not converged".

## 3. beta-VAE objective (Higgins et al., ICLR 2017, OpenReview Sy2fzU9gl)

**Objective** (Sec. 2, p. 5, Eq. 4):

> L(theta, phi; x, z, beta) = E_{q_phi(z|x)}[log p_theta(x|z)] - beta * D_KL(q_phi(z|x) || p(z))

with p(z) = N(0, I) (p. 5). beta = 1 is the standard VAE; the paper argues
"it is important to set beta > 1" for disentangling (p. 5). It is the
Lagrangian of a KL constraint D_KL < epsilon (Eqs. 2-3, p. 5).

**Normalised beta** (p. 9 and Appendix A.6, Eq. 10, p. 15): beta_norm =
beta * M / N, with M the latent size and N the input size; "it makes sense to
normalise beta by latent z size m and input x size n in order to compare its
different values across different latent layer sizes and different datasets".
Eq. 10 writes the objective with the reconstruction averaged over pixels
(E_n) and the KL averaged over latents (E_m).

**Reference implementation** (disentanglement_lib `methods/unsupervised/vae.py`):
`BetaVAE.regularizer` returns `beta * kl_loss`; `compute_gaussian_kl` is
`mean_batch( 0.5 * sum_latents( mu^2 + exp(logvar) - logvar - 1 ) )`, i.e. the
KL is **summed** over latents and averaged over the batch; the reconstruction
loss is the per-sample loss averaged over the batch (Bernoulli loss via
`sigmoid_cross_entropy_with_logits`, summed over pixels).

**Project note:** use summed-over-inputs Bernoulli log-likelihood and
summed-over-latents KL (the disentanglement_lib convention, unnormalised beta).
Always record both beta and beta_norm = beta * K / 36 in run configs, because
Higgins' published beta values are normalised and a K sweep changes beta_norm
at fixed beta.

## 4. beta-TCVAE and total-correlation estimators (Chen, Li, Grosse & Duvenaud, arXiv 1802.04942)

**Decomposition** (Sec. 3, p. 3, Eq. 2), with n a uniform index over the N
training points, q(z|n) = q(z|x_n), q(z) = sum_n q(z|n) p(n):

> E_{p(n)}[KL(q(z|n) || p(z))] = KL(q(z,n) || q(z)p(n))  [i: index-code MI]
> + KL(q(z) || prod_j q(z_j))  [ii: total correlation]
> + sum_j KL(q(z_j) || p(z_j))  [iii: dimension-wise KL]

assuming a factorised p(z). Proof: Appendix C, p. 19.

**Objective** (Sec. 3.1.1, p. 4, Eq. 4):

> L_beta-TC = E_{q(z|n)p(n)}[log p(n|z)] - alpha I_q(z;n) - beta KL(q(z)||prod_j q(z_j)) - gamma sum_j KL(q(z_j)||p(z_j))

"Our proposed beta-TCVAE uses alpha = gamma = 1 and only modifies the
hyperparameter beta." With alpha = gamma = 1 this equals ELBO - (beta - 1) TC
(disentanglement_lib docstring: "If alpha = gamma = 1, Eq. 4 can be written as
ELBO + (1 - beta) * TC").

**Minibatch-weighted sampling, MWS** (Sec. 3.1, p. 4, Eq. 3; Appendix C.1,
p. 19, Eq. S4). With a minibatch {n_1..n_M} drawn i.i.d. from p(n) and
z(n_i) ~ q(z|n_i):

> E_{q(z)}[log q(z)] ≈ (1/M) sum_{i=1}^M [ log sum_{j=1}^M q(z(n_i)|n_j) - log(NM) ]   (S4)

N = dataset size, M = batch size. "This minibatch estimator is biased, since
its expectation is a lower bound" (p. 4). Footnote 2 (p. 4): "The same argument
holds for the term prod_j q(z_j) and a similar estimator can be constructed."

In the official code (`vae_quant.py`, `elbo`, MWS branch), per sample i:
- `log q(z_i) = logsumexp_j( sum_d log q(z_id | n_j) ) - log(M*N)`
- `log prod_d q(z_id) = sum_d [ logsumexp_j( log q(z_id | n_j) ) - log(M*N) ]`

so log(MN) is subtracted **once per marginal** (D times) in the product term
and once in the joint term. The TC estimate `log q(z) - log prod_d q(z_d)`
therefore carries the constant +(D-1) log(MN).

disentanglement_lib (`vae.py`, `total_correlation`) implements the same
estimator but **drops all constants**: "We ignore the constants as they do not
matter for the minimization. The constant should be equal to
(num_latents - 1) * log(batch_size * dataset_size)". Its returned "TC" is
therefore offset from the Chen estimate by -(D-1) log(MN) and is not a TC value
in nats; gradients are identical.

**Minibatch-stratified sampling, MSS** (Appendix C.2, p. 19-20). Sample a
batch of size M+1 containing n*, with B_hat = {n_1..n_M} the others (drawn
without replacement):

> f(z, n*, B_hat) = (1/N) q(z|n*) + (1/M) sum_{m=1}^{M-1} q(z|n_m) + ((N-M)/(NM)) q(z|n_M)   (S6)

"which is unbiased (ie. q(z) = E[f(z, n*, B_hat)]), and exact if M = N", and

> E_{q(z,n)}[log q(z)] ≈ (1/(M+1)) sum_{i=1}^{M+1} log f(z_i, n_i, B_{M+1} \ {n_i})   (S7)

"this estimator is a lower bound on E_q(z)[log q(z)] due to Jensen's
inequality" (S8, p. 20); "the bias goes to zero if M increases". The same
weighting is used for each marginal q(z_j). Official code
(`_log_importance_weight_matrix`): with batch size B, it sets M = B - 1,
weight 1/M everywhere, 1/N on the diagonal, (N-M)/(NM) on the entry after the
diagonal (and at [M-1, 0]), then `log q(z) = logsumexp(log W + log q(z_i|n_j))`.
Reported (C.2.2, p. 20): "using MSS results in performance similar to MWS."

**Official-code objective switches** (`vae_quant.py`): with `tcvae=True,
include_mutinfo=True, lamb=0` the objective is
`logpx - (logqz_condx - logqz) - beta*(logqz - logqz_prodmarginals) - (logqz_prodmarginals - logpz)`,
i.e. Eq. 4 with alpha = gamma = 1. Optional annealing (`anneal_kl`): beta
ramps linearly 0 -> beta over warm-up only when `--beta-anneal` is set.

**Project note:** report TC in nats with the Chen constants (MWS, eq. S4) so the
value is interpretable; train with either (the constant has no gradient).
N = number of training points as counted by the data loader (if we train on
distinct legal states, N = number of distinct states; if on sequence-sampled
states with duplicates, N = number of samples; record which).

## 5. Mutual Information Gap, MIG (Chen et al. 2018, Sec. 4.1; Appendix B)

**Definition** (p. 5, Eq. 6):

> MIG = (1/K) sum_{k=1}^K (1/H(v_k)) ( I_n(z_{j(k)}; v_k) - max_{j != j(k)} I_n(z_j; v_k) ),
> j(k) = argmax_j I_n(z_j; v_k), K = number of known factors.

"for discrete v_k, 0 <= I(z_j; v_k) <= H(v_k), where H(v_k) =
E_p(v_k)[-log p(v_k)] is the entropy of v_k. As such, we use the normalized
mutual information I(z_j;v_k)/H(v_k)" (p. 5). "MIG is bounded by 0 and 1."
"We perform an entire pass through the dataset to estimate MIG."

**Chen's estimator (continuous latents, full posterior)** (Eq. 5, p. 5; Eq. S1,
Appendix B.1, p. 18):

> I_n(z_j; v_k) = E_{q(z_j,v_k)}[ log sum_{n in X_{v_k}} q(z_j|n) p(n|v_k) ] + H(z_j)

It uses samples from q(z_j|n), not the posterior mean, and assumes "The
generating process p(n|v_k) can be sampled from and is known" and
"p(v_k) and p(n|v_k) are quantized (ie. the empirical distributions)" (p. 18).
"we perform stratified sampling over p(v_k), and use 10,000 samples from
q(n, z_k) for each value of v_k" (p. 18). Official code
(`disentanglement_metrics.py`): I = H(z_j) - H(z_j|v_k), with each entropy
estimated by `estimate_entropies(..., n_samples=10000)` as
`-log q(z) = log N - logsumexp_n log q(z|x_n)`; MIs are sorted and
`clamp(min=0)`; normalisation uses `H(v_k) = log(#values)` (uniform factors:
`factor_entropies = [6, 40, 32, 32]` for dSprites, which **omits the shape
factor**; the paper's dSprites list on p. 6 likewise names only "scale (6),
rotation (40), posX (32), posY (32)").

**disentanglement_lib MIG (the version most papers report)**
(`evaluation/metrics/mig.py::_compute_mig`, `utils.py`):
1. Representation = encoder **mean** (`postprocessing/methods.py::mean_representation`;
   `postprocess_configs/mean.gin`). A `sampled_representation` also exists.
2. `num_train = 10000` points (`mig.gin`).
3. Each latent discretised by `_histogram_discretize`:
   `np.digitize(x, np.histogram(x, num_bins)[1][:-1])`, i.e. equal-width bins
   over that latent's own min-max range. **Default `num_bins = 20`**,
   `discretizer_fn = @histogram_discretizer` (`mig.gin`) — verified.
4. `m[i, j] = sklearn.metrics.mutual_info_score(ys[j], discretized_mus[i])`
   (plug-in discrete MI, **natural log**; checked: MI of a balanced bit with
   itself = 0.6931 = ln 2).
5. `H(v_k) = mutual_info_score(ys[k], ys[k])` = **empirical** plug-in entropy
   in nats (not log of the number of values).
6. `discrete_mig = mean_k( (sorted_m[0,k] - sorted_m[1,k]) / H(v_k) )`.
   No clamping, no epsilon: a factor with H(v_k) = 0 (constant in the sample)
   gives a division by zero.

**Binary / categorical factors.** Both definitions require a discrete v_k; the
binary case is the standard one (H(v_k) <= ln 2 nats). For discrete latents
(e.g. a categorical VAE), the dlib histogram step on values {0, 1} still yields
two occupied bins, so the plug-in MI is exact up to sampling.

**Pitfalls flagged by the sources.**
- Chen Table 1 (p. 5) calls MIG "unbiased"; the caption defines this as "is
  unbiased for all hyperparameter settings" (no classifier hyperparameters),
  not as a statistically unbiased estimator. The entropy/log-density estimators
  themselves are Monte Carlo estimates (Appendix B.1; and Appendix C.2.2, p. 20:
  "neither of them is unbiased for estimating log q(z) due to Jensen's
  inequality", said of the q(z) estimators).
- The two implementations differ (full-posterior samples with known p(n|v_k)
  vs posterior mean plus 20-bin histogram); their numbers are not comparable.

**Project note (deliberate choices to freeze):** use the disentanglement_lib
version (mean representation, 20 equal-width bins, natural log, empirical
H(v_k)) as the headline, because it is the convention in the literature we
compare to. Blocksworld facts are rare (e.g. a given `on(X,Y)` is false in
most states), so H(v_k) is small and the ratio is noisy: report per-fact MIG
with the fact's base rate, exclude facts constant in the evaluation set (log
them), and bootstrap over states. Blocksworld facts are logically coupled
(`clear(X)` is determined by the `on(*,X)` facts), so a latent can carry high MI
with several facts legitimately; per-fact MIG must be read with that in mind.

## 6. DCI (Eastwood & Williams, ICLR 2018, OpenReview By-7dz-AZ)

Setup (Sec. 2, p. 2): train K regressors f_j(c): R^D -> R predicting factor
z_j from code c, giving a D x K matrix R of relative importances, "R_ij denotes
the relative importance of c_i in predicting z_j".

**Disentanglement** (p. 2):

> D_i = 1 - H_K(P_i.),  H_K(P_i.) = - sum_{k=0}^{K-1} P_ik log_K P_ik,  P_ij = R_ij / sum_{k=0}^{K-1} R_ik

Overall: "relative code variable importance rho_i = sum_j R_ij / sum_ij R_ij is
used to construct a weighted average sum_i rho_i D_i" (p. 2).

**Completeness** (p. 2-3):

> C_j = 1 - H_D(P~_.j),  H_D(P~_.j) = - sum_{d=0}^{D-1} P~_dj log_D P~_dj

(P~ normalises each column of R.) Overall completeness in the paper is the
**plain average** over factors: Table 3(b) (p. 11) is headed "Avg." while
disentanglement is "W. Avg." ("'W. Avg.' abbreviates weighted average");
e.g. VAE completeness (0.54+0.37+0.75+0.73+0.73)/5 = 0.62 matches Table 1(a).

**Log bases:** base K (number of factors) for D_i, base D (number of code
variables) for C_j, so both lie in [0, 1].

**Informativeness** (p. 3): "the prediction error E(z_j, z_hat_j) (averaged over
the dataset), where E is an appropriate error function"; the paper reports
test-set NRMSE (Table 1 caption, p. 6; Appendix B.1, p. 10: targets are
normalised to unit standard deviation, so RMSE is relative to the constant
regressor). Lower is better.

**Regressors in the paper** (Sec. 4.3, p. 5): lasso with inputs and targets
standardised, R_ij = |W_ij|, alpha fitted on a validation set; and random
forests with n = 10 trees, depth tuned on validation, importance = "the number
of cases split on c_i over the total number of splits". For InfoGAN the paper
states "we simply take the mean returned by Q(x) as the code" (Appendix A,
p. 10); it does not state the VAE code choice in the text we read.

**disentanglement_lib DCI** (`evaluation/metrics/dci.py`, `dci.gin`):
- Regressor: **gradient boosted trees**, `sklearn.ensemble.GradientBoostingClassifier()`
  with sklearn defaults (`n_estimators=100, learning_rate=0.1, max_depth=3`,
  no `random_state` set), one per factor, factors treated as **classes**
  (not regression). Not lasso, not random forest.
- R[:, j] = `abs(model.feature_importances_)` (impurity-based; sklearn: "The
  values of this array sum to 1, unless all trees are single node trees").
- `num_train = 10000`, `num_test = 5000` (`dci.gin`); representation = mean.
- `disentanglement_per_code = 1 - scipy.stats.entropy(R.T + 1e-11, base=K)`;
  weighted by `R.sum(1) / R.sum()`.
- `completeness_per_factor = 1 - scipy.stats.entropy(R + 1e-11, base=D)`;
  **weighted** by `R.sum(0) / R.sum()` (the paper uses a plain average; with
  sklearn's normalised importances each column sums to 1, so the weights are
  uniform and the two agree unless a factor's trees never split).
- "informativeness_train/test" = mean **classification accuracy**
  (`np.mean(model.predict(x) == y)`), higher is better, despite the variable
  name `train_loss`. This is not the paper's NRMSE.

**Pitfalls flagged by the source.**
- Regressor dependence (p. 3): informativeness "is dependent on the capacity of
  f, with linear regressors only capable of extracting information about z in
  c that is explicitly represented"; Table 1 (p. 6) shows different D/C/I for
  lasso vs random forest on the same codes.
- Rotation (p. 3): if factors are identifiable only up to rotation, "the
  informativeness metric remains valid but the disentanglement and
  completeness metrics do not".

**Project note:** Blocksworld facts are binary, so the dlib classifier version
is the natural one; freeze `GradientBoostingClassifier()` defaults plus an
explicit `random_state`, report accuracy and also balanced accuracy (rare
facts make plain accuracy trivially high; our addition), and report the
plain-average completeness alongside dlib's weighted one.

## 7. Coverage / best-latent F1 (Karvonen et al., arXiv 2408.00113)

**Thresholded classifier** (Sec. 3.2, p. 4, Eq. 2):

> phi_{f_i,t}(x) = I[ f_i(x) > t * f_i^max ]

"where f_i^max is (an empirical estimate of) max_{x~D} f_i(x), the maximum
value that f_i takes over the dataset D of activations".

**Coverage** (p. 4, Eq. 3):

> Cov({f_i}, G) := (1/|G|) sum_{g in G} max_t max_{f_i} F1(phi_{f_i,t}; g)

"we take, for each g in G, the F1-score of the feature that best serves as a
classifier for g, and then take the mean of these maximal F1-scores." G is the
set of board state properties (BSPs), e.g. 8 x 8 x 12 one-hot square/piece
facts for chess (p. 4). Threshold grid: "we sweep over t in {0, 0.1, ..., 0.9}
and take the best coverage score; typically this best t is in {0, 0.1, 0.2}"
(p. 4).

**Discrepancy between Eq. 3 and the official code.** Eq. 3 puts max_t inside
the sum (a threshold per BSP), but the text ("take the best coverage score")
and the code choose **one global t**: `f1_analysis.py::best_f1_average` takes
the max over features per BSP for each t, averages over BSPs, and
`update_dataframe_with_coverage_results` takes `argmax` over t. Code threshold
grid: `pipeline_config.py`: `f1_analysis_thresholds = torch.arange(0.0, 1.1, 0.1)`
(0.0 to 1.0, 11 values, versus the paper's 0 to 0.9). F1 per feature
(`analysis.py::get_F1_per_feature`) uses epsilon 1e-8 in precision, recall
and F1. Only "alive" features (nonzero firing rate on a sample;
`utils.py::get_firing_features`) are evaluated, and f_max is estimated on that
sample.

**Board reconstruction** (p. 4-5, Eqs. 4-5): a feature is assigned to every BSP
for which phi is "a high precision (of at least 0.95) classifier" on a 1000-game
training set; a BSP is predicted on if any assigned feature fires; F1 over the
board on a separate 1000-game test set. "different ways of recovering boards
from feature activations may lead to qualitatively different results" (p. 4).

**Related variant (SynthSAEBench, arXiv 2602.14687, Sec. 4.3, p. 5):** each
latent is matched to the ground-truth feature with the highest decoder cosine,
"A latent is considered to fire when f_j > 0", and precision/recall/F1 are
averaged over latents (not over ground-truth features).

**Project note:** Karvonen's threshold is a fraction of the feature's maximum
and assumes nonnegative (ReLU) features. Gaussian VAE latents are signed with
arbitrary sign, so for VAE latents we must deviate: threshold mu_i at
quantiles or at t x max in **both** sign directions (feature and its negation),
and say so. For the SAE arm, use Eq. 2 as written with the code's global-t
convention, and report both the global-t and per-fact-t variants. G = the 36
Blocksworld facts (positive class = fact true).

## 8. Categorical latents: Gumbel-softmax / Concrete VAE

**Sampling** (Jang, Gu & Poole, arXiv 1611.01144, Sec. 2, p. 2, Eqs. 1-2):
Gumbel-max `z = one_hot(argmax_i [g_i + log pi_i])`, relaxed to

> y_i = exp((log(pi_i) + g_i)/tau) / sum_{j=1}^k exp((log(pi_j) + g_j)/tau)

with g_i i.i.d. Gumbel(0,1). Straight-through variant (Sec. 2.2, p. 3):
"we discretize y using arg max but use our continuous approximation in the
backward pass".

**KL term as used by Jang et al.** (Sec. 4.2, p. 6): "We use a learned
categorical prior rather than a Gumbel-Softmax prior in the training objective.
Thus, the minimization objective during training is no longer a variational
bound if the samples are not discrete." The official notebook instead uses a
**uniform** categorical prior and the analytic categorical KL per latent:
`KL = sum_{N,K} q_y * (log q_y - log(1/K))` (q_y = softmax(logits),
`log q_y = log(q_y + 1e-20)`), with a Bernoulli decoder on binary MNIST.

**Temperature annealing** (Jang Sec. 4.2, p. 7): "tau = max(0.5, exp(-rt)) of
the global training step t, where tau is updated every N steps.
N in {500, 1000} and r in {1e-5, 1e-4}"; p. 8 (semi-supervised): r = 3e-5
updated every 2000 steps. "In practice, we start at a high temperature and
anneal to a small but non-zero temperature" (p. 3). Notebook: `tau0 = 1.0`,
`ANNEAL_RATE = 0.00003`, `MIN_TEMP = 0.5`, update every 1000 iterations,
`tau = max(tau0 * exp(-ANNEAL_RATE * i), MIN_TEMP)`.

**Concrete relaxation of the KL** (Maddison, Mnih & Teh, arXiv 1611.00712,
Sec. 3.3, p. 6, Eqs. 12-13; Appendix C.2, p. 15, Eqs. 20-23). Three options:
(20) Concrete densities for both posterior (temperature lambda_1) and prior
(temperature lambda_2); (21) relaxed log-mass; (22) the analytic discrete KL
`sum_i Q(d(i)|x) log(P(d(i))/Q(d(i)|x))`. "Although (22) or (21) is tempting,
we emphasize that these are NOT necessarily lower bounds on log p(x) in the
relaxed model. (20) is the only objective guaranteed to be a lower bound"; with
(21)/(22) "one should early stop otherwise it will overfit to the spurious 'KL'
component of the loss" (p. 15). Compute in log space via ExpConcrete
(Appendix C.3, p. 16). At test time "the discrete lowerbound" is evaluated
(p. 6).

**Temperatures** (Maddison Appendix C.4, p. 17; D, p. 18): guideline
lambda <= 1/(n-1) is "conservative"; "lambda = 2/3 was the best single
performing temperature across the n in {2, 4, 8} cases"; distinct prior and
posterior temperatures "improve results quite dramatically"; best binary
setting "(prior lambda_2 = 1/2, posterior lambda_1 = 2/3)"; "No temperature
annealing was used."

**Project note:** for the discrete-latent arm, the routine choice is Jang's
recipe (analytic categorical KL against a uniform prior, tau annealed
max(0.5, exp(-r t))), reporting the discrete bound at evaluation, with
Maddison's warning recorded as a known limitation (option 22 is not a bound
during training). Active-unit counting for categorical latents cannot use
Burda's variance of a scalar mean; use per-latent KL to the prior or the
variance of the class-probability vector over data, frozen in the design.

## 9. Summary of defaults to match or deviate from

| Metric | Reference default | Our plan |
|--------|-------------------|----------|
| Active units | Var_x(mu) > 0.01 (Burda) | match; add log-A_u histogram and Asperti Eq. 13 check |
| beta-VAE | KL summed over latents, recon summed over inputs (dlib) | match; log beta and beta_norm |
| TC (MWS) | Chen: constant (D-1)log(NM) kept; dlib: dropped | report in nats with Chen constant |
| MIG | dlib: mean rep, 10k points, 20 equal-width bins, nats, empirical H(v) | match; drop constant facts, report base rates |
| DCI | dlib: GBT classifier defaults, 10k/5k, accuracy, weighted completeness | match + random_state, balanced accuracy, plain-average C |
| Coverage | t x f_max, global t over {0..1.0} step 0.1, alive features only | SAE: match; VAE: deviate (signed latents) |
| Gumbel-softmax | uniform prior, analytic KL, tau = max(0.5, exp(-3e-5 t)) every 1000 steps | match; evaluate discrete bound |
