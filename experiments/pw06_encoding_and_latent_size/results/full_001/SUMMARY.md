# pw06 full_001 summary

Outcome (registered, all runs): **mixed**. DESIGN.md sha256 `904ea2366846496c08b6c68668861d0f93677f06cf232c546967104349a1b129`.

| Gate | Pass | Detail |
|---|---|---|
| task | True | data, encodings, model-free table |
| pipeline (E36-L32 median within 1 of 15, 5/5 reconstruct) | True | median 15.0, recon 5/5 |
| capacity for P1 (E10a AE exact >= 0.95 in >= 2/3) | True | exact [0.9999999999999999, 0.9999999999999999, 0.9999999999999999] |

| Rule | All runs (registered) | Without non-converged |
|---|---|---|
| P1 | fails | fails |
| P2 | holds | holds |
| P3 | holds | holds |
| S1 | holds | holds |
| S2 | holds | holds |

| Cell | Count median (range), all ok runs | Recon pass | Converged | −ELBO | Rate | Distortion | Rate − H(X) |
|---|---|---|---|---|---|---|---|
| E36-L32 | 15 (14–15) | 5/5 | 5/5 | 9.45 | 7.84 | 1.57 | 1.37 |
| E30-L32 | 11 (11–12) | 5/5 | 5/5 | 8.44 | 6.49 | 1.97 | 0.02 |
| E10a-L32 | 6 (6–9) | 0/5 | 5/5 | 6.70 | 2.12 | 4.58 | -4.35 |
| E10b-L32 | 7 (6–7) | 0/5 | 5/5 | 6.71 | 2.07 | 4.63 | -4.40 |
| E10a-AE | 32 (32–32) (cov.) | 3/3 | 3/3 | n/a | n/a | 0.00 | n/a |
| E10a-b0.25 | 10 (10–10) | 5/5 | 5/5 | 8.79 | 8.47 | 0.34 | 2.00 |
| E36-L8 | 8 (8–8) | 5/5 | 5/5 | 9.30 | 7.95 | 1.37 | 1.48 |
| E36-L10 | 10 (10–10) | 5/5 | 5/5 | 9.53 | 7.94 | 1.55 | 1.47 |
| E36-L12 | 12 (12–12) | 5/5 | 4/5 | 9.63 | 7.93 | 1.67 | 1.46 |
| E36-L16 | 14 (13–15) | 5/5 | 4/5 | 9.42 | 7.92 | 1.55 | 1.45 |
| E36-L24 | 15 (15–16) | 5/5 | 5/5 | 9.59 | 7.87 | 1.72 | 1.40 |
| E36-L64 | 15 (15–16) | 5/5 | 5/5 | 9.62 | 7.89 | 1.73 | 1.41 |
| code10-c1 | 0 (0–0) | 0/5 | 5/5 | 6.93 | 0.00 | 6.93 | -6.93 |
| code10-c2 | 10 (10–10) | 5/5 | 5/5 | 9.11 | 6.89 | 2.23 | -0.04 |
| code10-c3 | 10 (10–10) | 5/5 | 5/5 | 9.92 | 8.20 | 1.72 | 1.27 |
| code10-c4 | 10 (10–10) | 5/5 | 5/5 | 10.38 | 8.82 | 1.56 | 1.89 |

Nats; test-split (confirmation) state weights; counts = Burda active units at 0.01 (plain AE: covariance count, descriptive).

Median −ELBO per latent size (E36): {'8': 9.30204272122743, '10': 9.525905205807021, '12': 9.62820383840844, '16': 9.423803120546479, '24': 9.593773565554482, '32': 9.453322932277272, '64': 9.618507472331219}.
E10a β = 0.25 count: 10 (10–10). L = 16 count: 14 (13–15).
