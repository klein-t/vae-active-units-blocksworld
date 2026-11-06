#show table: set par(justify: false)

= 5 Results

Unless stated otherwise, results are on the 36 facts at β = 1, and counts are medians over seeds on the test split.

== 5.1 Number of active units (Q2)

Like earlier VAEs with more latents than they need @burda2016iwae @asperti2018sparsity, our β-VAEs leave many latents inactive (@fig-au). Five of the six Blocksworld settings keep 15 or 16 of their 32 latents active, which matches none of the dimensions of @tab-targets, and their seeds differ by at most two units. The sixth, the 2×256 network at the higher learning rate, keeps 10, which happens to be the number of bits of the binary index, but we did not test whether its units form such an index. The gap between active and inactive latents is clear in almost every run (@fig-variance in Appendix A), and the plain autoencoder keeps all 32 latents active, as it should.

#figure(
  image("../figures/fig1_active_units.pdf", width: 100%),
  caption: [Active units (of 32) on raw states at β = 1, per seed and network. Blocksworld has five seeds per learning rate, code10 and prod5 three seeds at lr 3·10#super[−4]. Black bars are medians, and crosses mark runs that did not converge.],
) <fig-au>

code10 keeps 10 units in every run, as all three hypotheses expect, so on its own it cannot tell them apart. At the lower learning rate Blocksworld keeps five or six units more than code10, so the information hypothesis fails, and no Blocksworld run at this learning rate keeps fewer than code10, so the factor-count hypothesis fails too. The affine hypothesis holds at the two smaller networks, only just at 2×256, and only as an ordering, since no Blocksworld run comes near 25 units. prod5 grows with the network while Blocksworld does not, so this might just reflect capacity.

prod5, however, has constant bits where Blocksworld has derived facts. Once we drop these facts (Section 5.3), Blocksworld keeps 11 units at 2×256 and the lower learning rate, close to code10 (10) and far from prod5 (17). This is the pattern of the information hypothesis, although we saw it only after the main results and at one network. As expected, the 36 independent bits keep no active unit.

We also looked at the rate. It splits into the information the code carries about the state and the KL divergence between the aggregate posterior and the prior @hoffman2016surgery, which we call the _prior mismatch_. Since the information cannot exceed the entropy of the states, the rate above the entropy is a lower bound on this mismatch, and on Blocksworld it is at least 1.2 nats in every run. The rate stays near 8 nats while the count goes from 9 to 16 (@tab-kl), so the count is basically how many units share the same total KL, about half a nat per unit on average in the 15-unit settings. This does not explain why it settles there.

#figure(
  table(
    columns: (auto,) * 10,
    align: (left,) + (center,) * 9,
    stroke: (x, y) => if y == 1 { (bottom: 0.6pt) } else { none } + if x in (1, 4, 7) { (left: 0.4pt) } else { none },
    table.header(
      [], table.cell(colspan: 3)[*1×64*], table.cell(colspan: 3)[*2×256*], table.cell(colspan: 3)[*2×1024*],
      [], [*Active*], [*Rate*], [*KL/unit*], [*Active*], [*Rate*], [*KL/unit*], [*Active*], [*Rate*], [*KL/unit*],
    ),
    [Blocksworld, lr 3·10#super[−4]], [16 (15–16)], [8.14], [0.51], [15 (14–15)], [7.84], [0.52], [15 (15–16)], [7.76], [0.52],
    [Blocksworld, lr 3·10#super[−3]], [15 (15–16)], [8.13], [0.54], [10 (9–11)], [7.92], [0.79], [15 (14–15)], [7.90], [0.53],
    [code10], [10], [8.52], [0.85], [10], [8.63], [0.86], [10], [8.56], [0.86],
    [prod5], [16 (15–17)], [8.14], [0.51], [17 (16–19)], [8.47], [0.50], [23 (22–23)], [8.33], [0.36],
  ),
  caption: [Active units (median and range over seeds, one value when all seeds agree), median rate and the median over seeds of the within-run mean KL per active unit, in nats, on raw states at β = 1 with seeds as in @fig-au. The entropy of the states is 6.47 nats for Blocksworld, 6.93 for code10 and 8.96 for prod5.],
) <tab-kl>

== 5.2 Learning rate and β (Q2)

As we expected, the count varies more between settings than between seeds, but only because of the 2×256 network at the higher learning rate. Asperti and Trentin also saw the count move with the learning rate @asperti2020balancing[p. 199446], in a CIFAR-10 comparison with no reported seed spread, a different criterion and learning rates below ours. There a higher learning rate left more units active, unlike in our 2×256 network, so only the sensitivity compares. This 10-unit network even fits the data better than the reference one, with a lower negative ELBO, so for this network the 15-unit codes are not the best ones.

Lowering β to 0.25 changes the medians by at most one unit, although Asperti notes that a smaller KL weight reduces the inactive variables @asperti2019collapse[p. 6]. At β = 4 every Blocksworld run collapses, with no active unit at all.

== 5.3 Encoding and number of latents (Q2)

To see whether the count follows the encoding and the number of latents, we planned a follow-up after the main results, again writing down our expectations first. It keeps the reference configuration with five seeds and changes one of the following.


- The encoding, with only the 30 position facts or with the binary index in two random orders.
- The number of latents, from 8 to 64.
- The data, with code10 whose bits are copied one to four times.


Dropping the derived facts, which add no information, lowers the median from 15 to 11 (#ref(<fig-encoding>)a). The binary index keeps 6 and 7 units in its two orders, more than the 3 we had expected, but these models reconstruct only a third to two thirds of the states exactly, so they do not show how many units the full information needs.

#figure(
  image("../figures/fig3_encoding.pdf", width: 100%),
  caption: [Active units (of 32) per seed in the follow-up, 2×256 at β = 1. (a) Four encodings of the 866 states, with the range of the share of test states reconstructed exactly. (b) code10 with each bit copied c times. Bars are medians, and hollow markers are runs that do not reconstruct the data by the rule of Section 4.3.],
) <fig-encoding>

Copies decide whether code10 uses its latents at all (#ref(<fig-encoding>)b). With each bit present once every seed keeps no active unit, like a decoder that ignores the code, and with two to four copies every seed keeps 10, as we had predicted.

From 24 latents on the median stays at 15 (#ref(<tab-follow>)b), as we expected and as Burda et al. found when adding dimensions @burda2016iwae[p. 8]. With 8, 10 or 12 latents every latent is active and every state is still reconstructed, so 8 latents are enough. The best median objective is at 8 latents, but 10 and 12 do worse than 32 and the seeds at 8 and 32 overlap. Together with Section 5.2, the objective in this network does not favour 15 units over 8 to 10, and training simply settles on the plateau.

#figure(
  table(
    columns: (auto,) + (1fr,) * 10,
    align: (left,) + (center,) * 10,
    stroke: (x, y) => if y == 1 { (bottom: 0.6pt) } else { none } + if x == 4 { (left: 0.4pt) } else { none },
    table.header(
      [], table.cell(colspan: 3)[*(a) Encoding, L = 32*], table.cell(colspan: 7)[*(b) Latents L, 36 facts*],
      [], [*30 facts*], [*index 1*], [*index 2*], [*8*], [*10*], [*12*], [*16*], [*24*], [*32*], [*64*],
    ),
    [Active units], [11], [6], [7], [8], [10], [12], [14], [15], [15], [15],
    [range], [11–12], [6–9], [6–7], [8], [10], [12], [13–15], [15–16], [14–15], [15–16],
    [Rate], [6.49], [2.12], [2.07], [7.95], [7.94], [7.93], [7.92], [7.87], [7.84], [7.89],
    [Negative ELBO], [8.44], [6.70], [6.71], [9.30], [9.53], [9.63], [9.42], [9.59], [9.45], [9.62],
  ),
  caption: [Follow-up, five seeds per condition. Active units as median and range, then median rate and median negative ELBO, in nats. Column 32 of (b) is the reference configuration.],
) <tab-follow>

== 5.4 Transformer activations (Q2)

On the transformer activations with learned γ the count grows with width, from 14 at T64 to 29 at T256 (@tab-act in Appendix A), but we have one transformer at each extreme, and width changes the size of the input too. Part of the growth might come from the objective, which sums the reconstruction over more coordinates against the same KL, and this is why Asperti and Trentin scale the KL by four when the pixels grow four times @asperti2020balancing[p. 199445].

== 5.5 What the units encode (Q3)

The active units do not match the facts one to one. We expected few units to detect a fact, but in the reference configuration most of them do, and most have one fact at each end of their range. The first row of @fig-anatomy is typical, as that unit is low when A is on B and high when B is on A. In 44 of the 74 active units over five seeds the two facts are on(X, Y) and on(Y, X), and we call these units _reciprocal units_.

#figure(
  image("../figures/fig2_unit_anatomy.pdf", width: 100%),
  caption: [Facts detected (F1 ≥ 0.9) by the 15 active units (rows) of the reference configuration, seed 1, the seed with the median number of two-fact units, and the last three detect none. Columns are the 30 position facts, six per block X, with t for on\_table(X), Y for on(X, Y) and h for holding(X). In the row labels X↔Y is a reciprocal unit, XY is on(X, Y), hX is holding(X) and a slash joins the two facts of any other unit.],
) <fig-anatomy>

The detector does not force this pairing, since many other pairs of facts also exclude each other, for example two positions of the same block, which only one unit uses. We did not check whether reciprocal pairs just follow from how often the facts occur together, or whether these units help set the count. The 1×64 network at the lower learning rate keeps a similar count with at most three reciprocal units per run, so similar counts can come with very different codes (@tab-align in Appendix A). The top-k SAE with k = 5, which we chose to match the one vector per block, separates the blocks more cleanly than the β-VAE, but it uses about twice as many units. Contrary to what we expected, it does not detect more facts than the best β-VAE.

== 5.6 Legality of samples (Q4)

In the main sweep the β-VAEs at β = 0.25 and 1 reconstruct every state from their posterior means, but their samples are often illegal (@fig-legality). Since at β = 1 our pilot models already missed the legality level we had planned to test, here we only describe what we see. As decoded, the prior samples of most runs are less often legal than the per-block baseline. On equal terms, at β = 1, the two larger networks beat it in every run, and the smallest one never does.

Samples from the aggregate posterior are better, but at β = 1 none reaches 95% legal states as decoded, and at β = 0.25 the two larger networks do in almost every run. Most illegal samples have a block that is not in exactly one place or a wrong derived fact, as in #ref(<fig-examples>)b, but among well-formed samples the constraints between blocks mostly hold (@tab-fail).

#figure(
  image("../figures/fig5_legality.pdf", width: 100%),
  placement: top,
  caption: [Legality per run of samples thresholded at 0.5, for prior samples as decoded and on equal terms and for aggregate-posterior samples as decoded. (a) β-VAEs on raw states at β = 1 (five seeds) and the β-TCVAE at β#sub[TC] = 2 (three seeds). (b) β-VAEs at β = 0.25. The lower lr is 3·10#super[−4] and the higher 3·10#super[−3], and a cross above a run marks that it did not converge.],
) <fig-legality>

#figure(
  placement: top,
  table(
    columns: (auto, auto, auto),
    column-gutter: 1.2em,
    align: (left, center, center),
    table.header([*Fraction of samples*], [*Prior*], [*Aggregate posterior*]),
    [well formed], [0.36–0.72], [0.76–0.94],
    [constraints between blocks hold (independent blocks)], [0.87–0.97 (0.60–0.72)], [≥ 0.96 (0.53–0.56)],
    [clear and hand\_empty consistent, given legal positions], [0.46–0.81], [0.82–0.97],
  ),
  caption: [Where samples fail, for the β-VAEs on raw states at β = 1. Each cell is the range of the six configuration medians, and ≥ marks the smallest value over runs. Row 2 is computed among well-formed samples, and in parentheses for blocks drawn independently from the model's marginals.],
) <tab-fail>

The gap between prior and aggregate-posterior samples fits the prior mismatch of Section 5.1, but we did not measure how much illegality it causes. At β = 1 part of the failure of the aggregate-posterior samples might come from the noise of the posterior itself, since the posterior means of the same states decode correctly.
