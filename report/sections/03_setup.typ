
= 3 Data and possible dimensions (Q1)

== 3.1 Blocksworld states

We describe the Blocksworld states @slaney2001blocks of five blocks A–E with 36 binary facts, namely on\_table(X), on(X, Y), holding(X), clear(X) and hand\_empty. Each block is on the table, held, or on one of the other four, so the 30 _position facts_ on\_table, on and holding are a one-hot code of five six-valued _block variables_. The _derived facts_ clear and hand\_empty follow from the positions. In the state of #ref(<fig-examples>)a, for example, A is on B, B is on C, C and D are on the table and E is held, so A and D are clear and hand\_empty is false. There are 866 legal states, a number we checked in three different ways. We can also encode each state by its _binary index_, namely its number in some fixed order written in 10 bits.

Our states come from action sequences that we had generated earlier, split into training, development and test sequences, and we draw training states with their frequencies in the training split. Each split contains all the legal states, so our test scores measure how well a model fits states it has already seen.

Before training we computed several possible dimensions of these states (@tab-targets), none of which is the unique true one. The block variables are not independent, since only 866 of their 7,776 joint values are legal. The _affine dimension_, the dimension of the smallest flat subspace containing all legal states, is 25, as the six position facts of a block always sum to one and the derived facts add nothing.

#show table: set par(justify: false)
#figure(
  table(
    columns: (1fr, auto, auto, auto),
    align: (left, right, right, right),
    table.header([*Possible dimension*], [*Blocksworld*], [*code10*], [*prod5*]),
    [log#sub[2] of the number of states $n$], [9.758 bits], [10 bits], [12.925 bits],
    [bits in a fixed-length state index, $ceil(log_2 n)$], [10], [10], [13],
    [entropy of the state distribution], [9.346 bits], [10 bits], [12.925 bits],
    [block or independent variables], [5, dependent], [10, independent], [5, independent],
    [affine dimension], [25], [10], [25],
    [binary inputs], [36], [36], [36 (6 constant)],
  ),
  caption: [Possible dimensions of Blocksworld and of the two controls of Section 3.2, computed before training. The Blocksworld entropy uses the state frequencies over all sequences, and under the test frequencies it is 6.47 nats. The controls are uniform.],
) <tab-targets>

#figure(
  image("../figures/fig6_examples.pdf", width: 100%),
  caption: [(a) A legal state and its seven true facts. (b) Three prior samples of the reference configuration (Section 4.1, seed 1), decoded at 0.5, each the first we found that breaks only the constraint named above it. Facts in conflict and missing facts are in colour.],
) <fig-examples>

== 3.2 Controls with known structure

To tell these dimensions apart we built two 36-bit controls with known structure.


- _code10_ has ten independent fair bits, each copied into three or four positions. Its entropy, its number of independent variables and its affine dimension are all ten.
- _prod5_ is Blocksworld without the constraints between blocks, namely five independent six-valued variables in one-hot code, with six constant bits in place of the derived facts. It has the same block variables and affine dimension as Blocksworld, but a higher entropy.


Before training we wrote down three hypotheses, each checked per network on the median counts at β = 1 and the lower learning rate. If the count follows the entropy of the states (_information hypothesis_), Blocksworld should stay within two units of code10, and prod5 more than two above Blocksworld. If it follows the affine dimension (_affine hypothesis_), Blocksworld should stay within two units of prod5 and more than two above code10, without having to reach 25. If it follows the number of block or independent variables (_factor-count hypothesis_), Blocksworld should stay within two units of prod5 and more than two below code10. prod5 differs from Blocksworld in three ways at once, so comparing the two cannot isolate one of them. We also train on 36 independent bits, each present once, where a code with zero rate is optimal.

= 4 Models and procedure

== 4.1 Models and training

Our main model is the β-VAE @higgins2017beta with diagonal Gaussian posteriors, whose encoder and decoder are mirrored ReLU networks. We name them by the depth × width of their hidden layers, _1×64_, _2×256_ and _2×1024_, and on the states we use 32 latents and a Bernoulli decoder. Our _reference configuration_ is the 2×256 network on the 36 facts at β = 1 and learning rate (lr) 3·10#super[−4].

In Section 5.4 we train on the last-layer activations of five small transformers that we had trained earlier on Blocksworld sequences, named by their width T64, T128a–c (three seeds) and T256. There we use 64 latents and a Gaussian decoder whose variance γ is fixed at 1 or learned @dai2019diagnosing.

We compare the β-VAE with four other models.


- A plain autoencoder, with no KL term, checks the active-unit metric.
- Since each legal state is a constant vector plus one vector per block position, we train a top-k sparse autoencoder (SAE) @makhzani2014ksparse with 64 units and k ∈ {5, 10}.
- The β-TCVAE @chen2018tcvae penalises the dependence between latents, with weight β#sub[TC] ∈ {2, 6}.
- The categorical VAE has 16 six-valued latents, like the block variables, and is trained on Gumbel-softmax samples @jang2017gumbel.


All models use Adam @kingma2015adam for 60,000 steps with batch 512, lr 3·10#super[−4] or 3·10#super[−3], and β ∈ {0.25, 1, 4} where it applies. Each Blocksworld β-VAE setting has five seeds and the others three, giving 210 runs that we call the _main sweep_. Every run warms up the KL term linearly over the first 10% of training @bowman2016generating. In a single-seed pilot without this warm-up the Blocksworld β-VAEs at β = 1 kept fewer units, among our three networks clearly so only at 2×256 (10 instead of 15 at the lower learning rate), where code10 and prod5 did not. The warm-up is therefore itself a training choice that can move the count, as Sønderby et al. also found @sonderby2016ladder.

== 4.2 Metrics

Besides active units we measure which facts the units detect and whether samples are legal.


- *Fact detectors.* For each unit and fact we fit a signed threshold on the posterior mean on development data and score it by F1 on the test split, adapting @karvonen2024board. A unit with F1 ≥ 0.9 on some fact is a _detector_, and it can detect at most two facts that exclude each other, one at each end of its range.
- *Legality.* A decoded state is _well formed_ when every block has exactly one place, and legal when, in addition, the constraints between blocks hold and clear and hand\_empty agree with the positions.


We decode 20,000 samples from the prior, and as many from the aggregate posterior of the test states, by thresholding the outputs at 0.5. As a baseline we draw each block independently from its training distribution over positions, which gives 54% legal states. Since this baseline derives clear and hand\_empty correctly, we score the VAE samples both _as decoded_ and _on equal terms_, with these facts recomputed from the decoded positions.

== 4.3 Protocol

Before the main sweep we wrote down what we expected, and after a pilot we changed a few things, listed in Appendix A. We only compare models that reconstruct the data, at least 95% of the states exactly (with a few exceptions in Appendix A). We keep the 22 runs that did not converge, whose objective still moved by more than 2% in the last fifth of training. With so few seeds we compare their ranges instead of using statistical tests, and the analyses we chose after the main results, such as the rate, are descriptive.
