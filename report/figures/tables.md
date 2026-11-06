## Table: AU at beta=1, median (min-max) over seeds

| Cell (beta = 1) | A1 | A2 | A4 |
|---|---|---|---|
| Blocksworld, lr 3e-4 | 16 (15-16) | 15 (14-15) | 15 (15-16) |
| Blocksworld, lr 3e-3 | 15 (15-16) | 10 (9-11) | 15 (14-15) |
| code10 control | 10 (10-10) | 10 (10-10) | 10 (10-10) |
| prod5 control | 16 (15-17) | 17 (16-19) | 23 (22-23) |

## Table: AU over beta (median (min-max); * = fails competence)

| Arch | lr | beta 0.25 | beta 1 | beta 4 |
|---|---|---|---|---|
| A1 | 0.0003 | 16 (16-16) | 16 (15-16) | 0 (0-0) * |
| A1 | 0.003 | 15 (15-15) | 15 (15-16) | 0 (0-0) * |
| A2 | 0.0003 | 15 (15-16) | 15 (14-15) | 0 (0-0) * |
| A2 | 0.003 | 11 (9-13) | 10 (9-11) | 0 (0-0) * |
| A4 | 0.0003 | 15 (13-15) | 15 (15-16) | 0 (0-0) * |
| A4 | 0.003 | 15 (14-16) | 15 (14-15) | 0 (0-0) * |

## Table: activation substrates, AU per seed

| Source | b_lin learned | b_lin fixed | b_raw learned | K_probe | K_pca99 |
|---|---|---|---|---|---|
| w64 | [14, 13, 14] | [13, 14, 14] | [16, 14, 14] | 25 | 39 |
| w128 | [17, 17, 16] | [14, 14, 16] | [20, 21, 20] | 26 | 88 |
| w128s2 | [17, 16, 15] | [16, 17, 17] | [22, 21, 22] | 27 | 91 |
| w128s3 | [18, 18, 17] | [16, 13, 18] | [22, 23, 20] | 27 | 93 |
| w256 | [21, 18, 22] | [19, 17, 19] | [28, 30, 29] | 31 | 208 |

## Table: fact alignment on raw states

| Model (raw states) | Cov30 per seed | MonoFrac per seed | DCI disentanglement per seed |
|---|---|---|---|
| top-k SAE, k = 5 | 0.80, 0.83, 0.83 | 0.87, 0.91, 0.91 | 0.99, 0.96, 1.00 |
| top-k SAE, k = 10 | 0.83, 0.83, 0.87 | 0.87, 0.72, 0.93 | 0.96, 0.97, 0.99 |
| beta-VAE A2, beta 1, lr 3e-4 (H3.2 comparator) | 0.77, 0.83, 0.83, 0.77, 0.83 | 0.80, 0.93, 0.93, 0.80, 1.00 | 0.55, 0.60, 0.56, 0.57, 0.64 |
| beta-TCVAE A2, beta_TC 2 (H3.2 comparator) | 0.67, 0.73, 0.70 | 0.81, 0.87, 0.93 | 0.59, 0.59, 0.65 |
| beta-VAE A1, beta 1, lr 3e-4 | 0.10, 0.33, 0.17, 0.13, 0.10 | 0.13, 0.44, 0.25, 0.25, 0.12 | 0.28, 0.37, 0.34, 0.37, 0.34 |
| beta-VAE A1, beta 1, lr 3e-3 | 0.83, 0.83, 0.83, 0.83, 0.83 | 1.00, 0.94, 1.00, 1.00, 1.00 | 0.60, 0.65, 0.68, 0.62, 0.67 |
| beta-VAE A2, beta 1, lr 3e-3 | 0.20, 0.40, 0.20, 0.37, 0.40 | 0.67, 0.91, 0.78, 0.90, 0.70 | 0.45, 0.54, 0.39, 0.44, 0.52 |
| beta-VAE A4, beta 1, lr 3e-4 | 0.83, 0.83, 0.80, 0.83, 0.83 | 0.93, 0.93, 0.87, 0.93, 0.88 | 0.60, 0.58, 0.61, 0.57, 0.60 |
| beta-VAE A4, beta 1, lr 3e-3 | 0.80, 0.47, 0.50, 0.70, 0.63 | 0.93, 0.60, 0.60, 0.86, 0.73 | 0.63, 0.56, 0.52, 0.57, 0.52 |
| plain autoencoder | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.37, 0.35, 0.30 |
| control_fact_oracle | 1.00 | 0.56 | 1.00 |
| control_pca25 | 0.03 | 0.08 | 0.35 |

## Table: descriptive legality (unregistered pass)

| Cell | prior legality (mode) min-max | aggregate-posterior legality min-max | runs passing 0.95 |
|---|---|---|---|
| bw-gauss_vae-A1-b0.25-lr0.0003 | 0.067-0.084 | 0.878-0.889 | 0/5 |
| bw-gauss_vae-A1-b0.25-lr0.003 | 0.342-0.384 | 0.935-0.943 | 0/5 |
| bw-gauss_vae-A1-b1-lr0.0003 | 0.110-0.166 | 0.546-0.635 | 0/5 |
| bw-gauss_vae-A1-b1-lr0.003 | 0.321-0.336 | 0.726-0.737 | 0/5 |
| bw-gauss_vae-A2-b0.25-lr0.0003 | 0.343-0.396 | 0.945-0.967 | 4/5 |
| bw-gauss_vae-A2-b0.25-lr0.003 | 0.424-0.471 | 0.970-0.972 | 5/5 |
| bw-gauss_vae-A2-b1-lr0.0003 | 0.413-0.464 | 0.811-0.845 | 0/5 |
| bw-gauss_vae-A2-b1-lr0.003 | 0.463-0.550 | 0.864-0.898 | 0/5 |
| bw-gauss_vae-A4-b0.25-lr0.0003 | 0.412-0.451 | 0.962-0.969 | 5/5 |
| bw-gauss_vae-A4-b0.25-lr0.003 | 0.438-0.521 | 0.969-0.974 | 5/5 |
| bw-gauss_vae-A4-b1-lr0.0003 | 0.494-0.539 | 0.858-0.872 | 0/5 |
| bw-gauss_vae-A4-b1-lr0.003 | 0.539-0.578 | 0.888-0.895 | 0/5 |
| bw-tc_vae-A2-b2-lr0.0003 | 0.455-0.463 | 0.739-0.754 | 0/3 |

B2 = 0.5400; beta=1 runs with prior legality above B2: 6

## Table: KL per active unit, beta = 1 (pw07 A, post hoc)

| Cell (beta = 1) | runs | active units, median (min-max) | rate (nats), median (min-max) | mean KL per active unit (nats), median (min-max) |
|---|---|---|---|---|
| Blocksworld, lr 3e-4, A1 | 5 | 16 (15-16) | 8.1368 (8.0978-8.1455) | 0.5087 (0.5061-0.5424) |
| Blocksworld, lr 3e-3, A1 | 5 | 15 (15-16) | 8.1267 (8.0495-8.2163) | 0.5364 (0.5077-0.5434) |
| Blocksworld, lr 3e-4, A2 | 5 | 15 (14-15) | 7.8408 (7.8367-7.8955) | 0.5226 (0.5224-0.5600) |
| Blocksworld, lr 3e-3, A2 | 5 | 10 (9-11) | 7.9173 (7.8338-7.9650) | 0.7911 (0.7223-0.8843) |
| Blocksworld, lr 3e-4, A4 | 5 | 15 (15-16) | 7.7629 (7.7232-7.8502) | 0.5162 (0.4905-0.5198) |
| Blocksworld, lr 3e-3, A4 | 5 | 15 (14-15) | 7.9042 (7.8886-8.0138) | 0.5281 (0.5254-0.5641) |
| code10 control, A1 | 3 | 10 (10-10) | 8.5242 (8.4295-8.6584) | 0.8523 (0.8429-0.8658) |
| code10 control, A2 | 3 | 10 (10-10) | 8.6304 (8.6128-8.7437) | 0.8628 (0.8611-0.8742) |
| code10 control, A4 | 3 | 10 (10-10) | 8.5634 (8.5605-8.6994) | 0.8561 (0.8558-0.8698) |
| prod5 control, A1 | 3 | 16 (15-17) | 8.1401 (8.0174-8.1973) | 0.5123 (0.4716-0.5426) |
| prod5 control, A2 | 3 | 17 (16-19) | 8.4710 (8.3952-8.5129) | 0.4982 (0.4418-0.5320) |
| prod5 control, A4 | 3 | 23 (22-23) | 8.3336 (8.3155-8.3469) | 0.3629 (0.3623-0.3779) |

Over the 30 Blocksworld runs at beta = 1: active units 9-16; rate 7.7232-8.2163 nats (outward 7.72–8.22, 7.7–8.3); mean KL per active unit 0.4905-0.8843 (outward 0.49–0.89); rate minus H(X) 1.2524-1.7455 (outward 1.2–1.8); smallest share of the rate in the active units 0.99906; largest KL of an inactive unit 0.00072 nats.

## Table: pw06 follow-up, KL per active unit (Table 2; pw07 A, post hoc)

| pw06 condition | active units, median (min-max) | rate (nats), median | mean KL per active unit (nats), median (min-max) | negative ELBO (nats), median | H(X) (nats) |
|---|---|---|---|---|---|
| E36-L32 | 15 (14-15) | 7.8408 | 0.5226 (0.5224-0.5600) | 9.4533 | 6.4708 |
| E30-L32 | 11 (11-12) | 6.4886 | 0.5871 (0.5413-0.5933) | 8.4428 | 6.4708 |
| E10a-L32 | 6 (6-9) | 2.1206 | 0.3200 (0.2458-0.3530) | 6.7024 | 6.4708 |
| E10b-L32 | 7 (6-7) | 2.0719 | 0.2951 (0.2825-0.3506) | 6.7068 | 6.4708 |
| code10-c1 | 0 (0-0) | 0.0002 | n/a (no active unit) | 6.9316 | 6.9315 |
| code10-c2 | 10 (10-10) | 6.8910 | 0.6890 (0.6795-0.6964) | 9.1144 | 6.9315 |
| code10-c3 | 10 (10-10) | 8.2008 | 0.8199 (0.8078-0.8352) | 9.9222 | 6.9315 |
| code10-c4 | 10 (10-10) | 8.8243 | 0.8823 (0.8751-0.8994) | 10.3812 | 6.9315 |
| E36-L8 | 8 (8-8) | 7.9519 | 0.9940 (0.9849-1.0047) | 9.3020 | 6.4708 |
| E36-L10 | 10 (10-10) | 7.9380 | 0.7938 (0.7880-0.8063) | 9.5259 | 6.4708 |
| E36-L12 | 12 (12-12) | 7.9292 | 0.6608 (0.6558-0.6716) | 9.6282 | 6.4708 |
| E36-L16 | 14 (13-15) | 7.9217 | 0.5658 (0.5282-0.6015) | 9.4238 | 6.4708 |
| E36-L24 | 15 (15-16) | 7.8716 | 0.5203 (0.5026-0.5324) | 9.5938 | 6.4708 |
| E36-L64 | 15 (15-16) | 7.8853 | 0.5248 (0.4955-0.5302) | 9.6185 | 6.4708 |
| E10a-b0.25 | 10 (10-10) | 8.4684 | 0.8466 (0.8361-0.8506) | 8.7943 | 6.4708 |

## Table: legality on equal terms (pw07 B, post hoc; clear and hand_empty recomputed from the 30 position facts)

| Runs | n | prior below baseline, as decoded | prior below, on equal terms | prior on equal terms, Wilson 95% interval entirely above | aggregate posterior below (as decoded / equal terms) |
|---|---|---|---|---|---|
| Gaussian-latent beta-VAEs, beta = 1 | 30 | 24 | 10 | 20 | 0 / 0 |
| Gaussian-latent beta-VAEs, beta = 0.25 | 30 | 30 | 11 | 17 | 0 / 0 |
| beta-TCVAE, beta_TC = 2, A2 | 3 | 3 | 0 | 3 | 0 / 0 |

| Cell | prior as decoded min-max | prior on equal terms median (min-max) | prior well formed min-max | aggregate posterior as decoded min-max | aggregate posterior on equal terms median (min-max) | prior runs below baseline (decoded / equal terms) |
|---|---|---|---|---|---|---|
| bw-gauss_vae-A1-b0.25-lr0.0003 | 0.0674-0.0842 | 0.2251 (0.2139-0.2406) | 0.2635-0.2949 | 0.8782-0.8893 | 0.9310 (0.9235-0.9335) | 5 / 5 |
| bw-gauss_vae-A1-b0.25-lr0.003 | 0.3415-0.3841 | 0.5442 (0.5171-0.5580) | 0.5496-0.5910 | 0.9355-0.9427 | 0.9619 (0.9585-0.9641) | 5 / 2 |
| bw-gauss_vae-A1-b1-lr0.0003 | 0.1096-0.1663 | 0.3230 (0.2817-0.3474) | 0.3338-0.3925 | 0.5465-0.6353 | 0.7484 (0.6947-0.7642) | 5 / 5 |
| bw-gauss_vae-A1-b1-lr0.003 | 0.3205-0.3361 | 0.4995 (0.4919-0.5127) | 0.5334-0.5552 | 0.7257-0.7372 | 0.8156 (0.8116-0.8202) | 5 / 5 |
| bw-gauss_vae-A2-b0.25-lr0.0003 | 0.3433-0.3965 | 0.5217 (0.4954-0.5439) | 0.5517-0.5959 | 0.9449-0.9669 | 0.9728 (0.9636-0.9790) | 5 / 4 |
| bw-gauss_vae-A2-b0.25-lr0.003 | 0.4238-0.4711 | 0.5924 (0.5476-0.5998) | 0.5995-0.6431 | 0.9699-0.9719 | 0.9810 (0.9781-0.9819) | 5 / 0 |
| bw-gauss_vae-A2-b1-lr0.0003 | 0.4135-0.4637 | 0.5928 (0.5806-0.6292) | 0.6231-0.6681 | 0.8114-0.8447 | 0.8846 (0.8732-0.8951) | 5 / 0 |
| bw-gauss_vae-A2-b1-lr0.003 | 0.4626-0.5500 | 0.6545 (0.6121-0.6849) | 0.6508-0.7176 | 0.8639-0.8983 | 0.9181 (0.9079-0.9347) | 3 / 0 |
| bw-gauss_vae-A4-b0.25-lr0.0003 | 0.4124-0.4511 | 0.5813 (0.5636-0.6007) | 0.6050-0.6344 | 0.9621-0.9685 | 0.9768 (0.9747-0.9777) | 5 / 0 |
| bw-gauss_vae-A4-b0.25-lr0.003 | 0.4377-0.5212 | 0.6483 (0.5865-0.6583) | 0.6249-0.6894 | 0.9690-0.9742 | 0.9826 (0.9795-0.9847) | 5 / 0 |
| bw-gauss_vae-A4-b1-lr0.0003 | 0.4940-0.5395 | 0.6387 (0.6357-0.6810) | 0.6695-0.7097 | 0.8582-0.8720 | 0.9078 (0.9048-0.9149) | 5 / 0 |
| bw-gauss_vae-A4-b1-lr0.003 | 0.5394-0.5780 | 0.6922 (0.6755-0.7145) | 0.7049-0.7419 | 0.8880-0.8950 | 0.9285 (0.9244-0.9315) | 1 / 0 |
| bw-tc_vae-A2-b2-lr0.0003 | 0.4547-0.4632 | 0.6263 (0.6214-0.6283) | 0.6550-0.6592 | 0.7388-0.7538 | 0.8276 (0.8256-0.8346) | 3 / 0 |

Baseline B2 = 0.53995 (pw07 = pw04a stored). Beta = 1, prior on equal terms per run: all 0.2817-0.7145 (outward 0.28–0.72); 1x64 0.28–0.52 (0.2817-0.5127); 2x256 and 2x1024 0.58–0.72 (0.5806-0.7145). Aggregate posterior on equal terms per run: 0.6947-0.9347 (outward 0.69–0.94).
1x64 at beta = 1, prior well formed per run: 0.3338-0.5552 (outward 0.33–0.56). beta-TCVAE prior on equal terms: 0.6214-0.6283 (outward 0.62–0.63).
Beta = 1, ranges of the six configuration medians (prior): between-block constraints among well-formed samples 0.87–0.97; the same for independent blocks with the model's marginals 0.60–0.72; derived facts consistent given legal positions 0.46–0.81; well formed 0.36–0.72. Aggregate posterior: well formed 0.76–0.94; between-block constraints among well-formed samples at least 0.9661 in every run. Largest |well formed - product of per-block fractions| over the beta = 1 runs 0.0134; over all 63 runs 0.0238.
Aggregate posterior, ranges of the six configuration medians at beta = 1 (Table 6): between-block constraints among well-formed samples 0.9752-0.9928; independent blocks with the model's marginals 0.5395-0.5586 (outward 0.53–0.56); clear and hand_empty consistent given legal positions 0.8212-0.9606 (outward 0.82–0.97).

## Supporting numbers that left the PDF in report v1.7 (values printed in v1.6, read from JSON and asserted)

### Data (v1.6 Sections 1, 3.1 and 4.2)

- Sequences: training file 16004 (split by sequence into 12,803 training and 3,201 development sequences; the split sizes are quoted from `docs/EXPERIMENT_LOG.md`, pw02 entry, not recomputed here), test file 4001 (pw01 `metrics.task.sequence_lengths`).
- Legal states with the hand empty / holding a block: 501 / 365 of 866 (pw01 `metrics.n_arm_empty`, `metrics.n_holding`).
- Rank of the 866 x 36 legal-state matrix: 26 (affine dimension 25 plus the offset; pw01 `metrics.linear.rank_all`).
- H(X) under the test-split frequencies: 6.4708 nats = 9.3354 bits (pw06 `cells.*.H_X_nats`).

### Section 5.1

- Rate per run at beta = 1 (pw07 `A_unit_kl.per_run.rate_nats`, outward): code10 8.42–8.75 (8.4295-8.7437), prod5 8.01–8.52 (8.0174-8.5129); Blocksworld in the KL table above. prod5 entropy 8.9588 nats = ln 7776 (pw05 `summaries.q1_H_X_bits` x ln 2).
- Median counts at beta = 1 under the inactivity threshold of Asperti and Trentin (2020) (post hoc; a unit is inactive when its mean posterior variance is at least 0.8; pw05 `summaries.q7_au_asperti_trentin`), network order, lower lr first: 15, 15, 15, 9, 14, 13; with Burda's threshold (`summaries.n_active`): 16, 15, 15, 10, 15, 15. Controls identical under both. Largest shift of a Blocksworld median or range end: 2 units.

### Section 5.2

- 2x256, higher vs lower lr: median distortion 1.1783 vs 1.5692 nats (pw05 `summaries.q2_D_nats`, 128 draws); negative ELBO median (seed range) 9.125 (8.984-9.157) vs 9.436 (9.406-9.510), outward 8.9–9.2 vs 9.4–9.6 (4,096-draw audit re-derivation, `experiments/pw05_unit_anatomy/negelbo_4096/cell_negelbo_nats.json`; the 128-draw pw05 ranges round outward to 9.3-9.6 for the lower lr). Rates 7.92 vs 7.84: KL table above.
- Zero rate: factorised-decoder distortion on Blocksworld 15.6612 nats (pw05 `summaries.q5_D0_nats`); largest rate of a beta = 4 run 4.61e-05 nats, below 1e-4 (pw05 `summaries.q5_beta4_rate_bits` x ln 2).
- Critical-beta heuristic (post hoc; distortion saved by the code against a decoder that ignores it, divided by the rate; re-optimisation at the new beta ignored; pw05 `summaries.q5_beta_star`, per run): Blocksworld 1.5916-1.8541 (outward 1.59–1.86), code10 2.6846-2.7503 (cell medians 2.71-2.73, about 2.7), prod5 1.2471-1.2927 (outward 1.2–1.3).

### Section 5.3

- Index codes (E10a, E10b): rate per run 1.9258-2.2774 nats (outward 1.9–2.3; pw07 `A_unit_kl.per_run_pw06.rate_nats`); zero-rate value (sum of the per-bit Bernoulli entropies = pw06 `model_free.H_nats` + `tc_e10_nats`) E10a 6.8777, E10b 6.8762 nats. Median negative ELBO and H(X): follow-up table above.

### Section 5.4

- T128a full activations, learned gamma: 62 (61-63) active units at beta = 0.25 (no run converges), 14 (13-16) at beta = 4 (pw02a `au_table`).
- T128a full activations: k = 5 SAE coverage 0.5667 (17/30), learned-gamma beta-VAE coverage 0.1000 (3/30) and detector fraction 0.1429 (pw03a `secondary.h34_activation_substrate_alignment`). With learned gamma, the beta-VAE at beta = 4, the beta-TCVAE at beta_TC = 6 and the categorical VAE pass the reconstruction criterion on T128a in all three seeds (the categorical VAE also converges); on full T256 activations the fixed-gamma beta-VAE fails it in all three seeds (pw02a `au_table` `competence`, `converged`).
- State-projected activations, learned gamma: units breaking the variance law per run 2-31 (pw02a `au_table` `asperti_not_converged`, 15 runs).
- Linear-VAE prediction (post hoc; covariance eigenvalues above the collapse threshold beta*gamma; pw05 `q4_b_lin`): 18-25 units at gamma = 1 against 13-19 observed, and the full rank of 25 at the floor against 13-22 observed.

### Section 5.5

- Positive control (the 36 facts plus Gaussian noise of sd 0.1, plus 28 pure-noise coordinates; 64 units, 36 carrying one fact; pw03a `analyze_alignment.py` `controls()`): coverage 1.00 (30/30), detector fraction 0.5625, DCI 1.0000 (pw03a `metrics.per_model.control_fact_oracle`).
- Raw states at beta = 0.25, runs reaching coverage 25/30: bw-gauss_vae-A1-b0.25-lr0.003 seed 4; bw-gauss_vae-A2-b0.25-lr0.0003 seed 1 (pw03a `metrics.per_model.*.cov30`).
- Unit-block normalised mutual information at 2x256 (post hoc; median over units, best and second-best block; pw05 `summaries.q3.*.max_nmi_median`, `second_nmi_median`): prod5 0.4909, 0.0428; Blocksworld 0.3284, 0.2877.
- Figure 5 panel (b) of v1.6 (prod5 seed 1 unit anatomy) is no longer drawn; its seed choice and two-fact totals stay asserted in make_figures.py.

### Section 5.6

- beta-TCVAE (beta_TC = 2, 3 seeds): prior legality as decoded 0.454–0.464, on equal terms 0.62–0.63, aggregate posterior as decoded 0.738–0.754: below the baseline as decoded, above it on equal terms, below the legality criterion in all three seeds (pw04a descriptive pass; pw07 B).

### Moved from the prose in the v1.7 final edit

- Reference configuration, two-fact units over five seeds by pair type (pw05 presentation `q3_pairs.json` `pair_types_per_cell`): on(X,Y)|on(Y,X) 44; two holding facts 6; an on fact with a holding fact 5 (on(X,Y)|holding(X) 1, on(X,Y)|holding(Y) 2, on(X,Y)|holding(Z) 2); total 55. The 2 units pairing on(X,Y) with a third block's holding are the two whose facts can co-occur; all other pairs are mutually exclusive.
- prod5, two-fact units over the three networks: 74; the 3 that do not pair positions of one block are at 2x1024 and pair on(X,Y) with a third block's holding (same file).
- beta = 0.25, prior samples on equal terms below the per-block baseline: 11 of 30 runs (pw07 `B_matched_legality.counts.beta025.prior_below_rederived`).

### Quoted from prose records (not recomputed by this script)

- `experiments/pw05_unit_anatomy/README.md`, Corrections (2026-10-27), Q2: "Two independent 4,096-draw estimates give −ELBO 9.125 (8.984–9.157) for A2 lr 3e-3 against 9.436 (9.406–9.510) for A2 lr 3e-4, and D 1.17 against 1.58."
- `docs/EXPERIMENT_LOG.md`, pw02 entry: "Split by sequence (train 12,803 / val 3,201 from the thesis train file; thesis test file sealed)."
- The status of each planned analysis is in Appendix A of the report (planned predictions and their outcomes) and in each `experiments/pwNN_*/README.md`.
