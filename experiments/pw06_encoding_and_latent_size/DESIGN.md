# pw06_encoding_and_latent_size: design (registered before any pw06 run)

Status: designed, not run. Written on 2026-10-27, after the pw02a–pw05 results were known, as a
follow-up to three reviews of report v1.1. All three asked for the direct test of the report's
closing interpretation (re-encode the same states) and for more than one latent size. One review
also gave a reconstruction/KL balance argument for code10. The predictions below are fixed before
any pw06 model is trained. The design text is frozen by its SHA-256, recorded in the README and in
RUN.json.

## Questions
- **Q1 (encoding).** With the network, β and lr fixed, does the active-unit count on the same 866
  Blocksworld states change when the states are re-encoded?
- **Q2 (latent size).** At the reference cell, does the count stop growing with the latent size, as
  the architecture-relative "intrinsic internal dimension" of Asperti (2018) suggests?
- **Q3 (copies).** Does a fair bit receive a unit only when it is copied, as the balance between
  reconstruction and KL at β = 1 implies?

## Fixed setup
Identical to the pw02a reference cell unless stated otherwise:
- Bernoulli β-VAE with Gaussian latents and a 2×256 mirrored ReLU MLP; β = 1; lr 3·10⁻⁴.
- Adam, batch 512, gradient clipping at norm 10, 60,000 steps.
- Linear KL warm-up over the first 10% of training; analytic KL; final checkpoint.
- The pw01/pw02a train, development and test splits and state frequencies.
- Active unit: the state-frequency-weighted variance over the test split of the posterior mean
  exceeds 0.01, computed exactly as in pw02a.
- Reconstruction check, as in pw02a: from the posterior mean on the test split, exact-state
  reconstruction at least 0.95 and bitwise accuracy at least 0.99.
- Convergence check, as in pw02a: the development objective changes by less than 2% over the last 20%
  of training.
- 5 seeds per cell unless stated otherwise.

## Data and cells
**Encodings (Q1).** Every encoding is a deterministic function of the state.
- **E36:** the 36 facts. This is the baseline, re-run inside pw06.
- **E30:** the 30 position facts only. It drops clear(X) and hand_empty, which are affine functions of
  the positions.
- **E10a and E10b:** a 10-bit binary index of the state. The index is the rank of the state under a
  fixed random permutation of the 866 pw01 states (`numpy.random.default_rng(seed).permutation(866)`,
  seed 0 for E10a and seed 1 for E10b), written most significant bit first.

Cells for Q1:
- E36-L32, E30-L32, E10a-L32 and E10b-L32, with 5 seeds each;
- E10a with a plain autoencoder of 32 latents, with 3 seeds (capacity control);
- E10a at β = 0.25, with 5 seeds (descriptive).

**Latent sizes (Q2).** E36 at L ∈ {8, 10, 12, 16, 24, 64}, with 5 seeds each. L = 32 is the E36-L32 cell.

**Copies (Q3).** code10, with each of the 10 fair bits copied c ∈ {1, 2, 3, 4} times and padded with
constant-zero inputs to 40 inputs. It uses the same bit generator as pw02a's code10, with L = 32 and
5 seeds per c.

Total: 78 runs.

## Model-free quantities (computed at design time, train-split frequencies)
| Quantity | Value |
|---|---|
| H(X) | 9.345 bits = 6.478 nats |
| Zero-rate distortion of E36 (sum of the marginal Bernoulli entropies) | 15.662 nats |
| Total correlation of the 10 index bits (sum of marginal entropies − H), permutation seeds 0–3 | 0.39–0.41 nats |

For code10 with c copies, the zero-rate solution costs c·ln 2 nats per bit. Any code costs at least
ln 2 nats of rate per transmitted bit, because the rate is at least the mutual information, and it
also pays the mismatch between the aggregate posterior and the prior.

## Accounts and what they predict
- **Information.** The count follows the information in the states. That information is identical in
  E36, E30 and E10, so the counts should be similar: E10 within 2 of E36, or near 10.
- **Reconstruction/KL balance at β = 1.** A unit is used only when the distortion it removes exceeds
  the rate it costs.
  - For E10, no code can save more than the total correlation, about 0.4 nats per state. That is less
    than the aggregate-posterior mismatch our models pay on E36, at least 1.2 nats per state (pw05).
    E10 should therefore collapse to almost no units.
  - An uncopied fair bit (c = 1) saves at most ln 2 nats and costs at least ln 2, so it gets no unit.
  - With c ≥ 2 copies, a unit pays for itself.
- **Architecture-relative intrinsic dimension (Asperti 2018).** Above some latent size, the count stays
  about constant.

## Predictions and decision rules
**Primary.**
- **P1 (Q1).** For both E10a and E10b, three conditions must hold:
  - the median count is at most 3;
  - the maximum count is at most 5;
  - median(E36-L32) − median(E10) is at least 10.

  P1 holds if all three hold for both encodings, and fails otherwise. It is untested if the pipeline
  gate fails, or if the E10a plain autoencoder fails the capacity check.
- **P2 (Q3).** Two conditions must hold:
  - at c = 1 the median count is at most 1;
  - at each c ∈ {2, 3, 4} the median count is 10 and every seed lies in 9–11.

  P2 holds if both hold.
- **P3 (Q2).** The medians at L ∈ {24, 32, 64} lie within 2 units of each other: the largest minus the
  smallest of the three medians is at most 2.

**Secondary.** A secondary result never compensates for a failed primary one.
- **S1 (Q2).** At each L ∈ {8, 10, 12}, the median count is at least L − 1.
- **S2 (Q1).** median(E30-L32) ≤ median(E36-L32).

**Descriptive (no prediction).**
- Median −ELBO, rate and distortion per cell.
- Rate minus H(X) per encoding.
- The count of E10a at β = 0.25.
- The count at L = 16.

**Which runs enter which rule.**
- P1 and P2 concern whether units are used at all, and their predicted outcome implies a failed
  reconstruction. They therefore use every run that passes the numerical checks (finite objective,
  non-negative KL). Reconstruction is reported but does not gate them.
- P3, S1 and S2 use only runs that pass the reconstruction check. A size or encoding with fewer than 3
  passing runs is untested.
- Every verdict is computed with all runs and again without the runs that fail the convergence check.
  Both are reported, and the verdict with all runs is the registered one.

## Gates
- **Pipeline gate.** The E36-L32 median must lie within 1 of 15, the pw02a reference-cell median, and
  all 5 runs must pass the reconstruction check. If not, the run is a technical stop and nothing is
  tested.
- **Capacity check for P1.** The E10a plain autoencoder must reach exact-state reconstruction of at
  least 0.95 in at least 2 of its 3 seeds.

## Pilot
The pilot is technical only and runs on development data, with one seed each of four cells:
E36-L32, E36-L64, the E10a plain autoencoder and code10 c = 2.

It checks three things:
- the runs complete;
- the objective is finite;
- the runtime.

It does not compute or inspect active-unit counts. Any change forced by the pilot is recorded as a
dated amendment before the main run.

## Analysis and checks
- A registered analysis script, committed before the main run, computes every count, check and verdict
  once on the main run. It writes `results.json`.
- A separately written recount from the saved posterior arrays re-derives every
  count and verdict.
- Results are logged in the README and in `docs/EXPERIMENT_LOG.md`, failures and stops included.
