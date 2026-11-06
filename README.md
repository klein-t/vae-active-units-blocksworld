# Do active units reflect a known state space? β-VAEs on Blocksworld

Project Work in Deep Learning (3 CFU), University of Bologna, prof. Andrea Asperti. Student Klein Tahiraj.

The report is `report/report.pdf`.

## Question

Blocksworld with five blocks has exactly 866 legal states, so we know its state space before training anything.
We train overcomplete β-VAEs (32 latents) on these states and ask how many latent units stay active, and whether
that number matches a property of the domain, like the number of facts or the number of bits needed to index the
states. We also look at what each active unit encodes and whether samples from the prior are legal states.

## Layout

```
src/pwlib/      shared code (domain, data, models, training, metrics, alignment, legality, run records)
experiments/    one folder per experiment, pw01 to pw07, plus a re-implementation of the scoring
docs/           experiment log, proposal, metric definitions
tests/          unit tests, plus a one-off cross-check against disentanglement_lib
report/         report source (Typst), figures, make_figures.py and the compiled PDF
```

Each folder under `experiments/` has its plan (`PROTOCOL.md` with `DESIGN.json`, or `DESIGN.md`), a `README.md`
with its status and results, its scripts, and the aggregate results in `results/<run_id>/`.

| Folder | What it is | Report |
|---|---|---|
| `pw01_state_space_ground_truth` | exact count of the 866 legal states, data checks, candidate target counts | Section 3.1 |
| `pw02_active_units` | first sweep plan, stopped at the pilot (the 4×256 network did not reconstruct) | replaced by pw02a |
| `pw02a_active_units_amended` | the main sweep and the active-unit counts | counts of 5.1, 5.2, 5.4 |
| `pw03_fact_alignment` | first alignment plan, never ran | replaced by pw03a |
| `pw03a_fact_alignment_amended` | which facts each unit detects (coverage, detector fraction) | 5.5, alignment |
| `pw04_sample_legality` | first legality plan, never ran | replaced by pw04a |
| `pw04a_sample_legality_amended` | legality of prior and posterior samples | 5.6, legality |
| `pw05_unit_anatomy` | post hoc analyses of the sweep (rate, ELBO terms, unit anatomy) | 5.5, post hoc analyses |
| `pw06_encoding_and_latent_size` | follow-up with other encodings, latent sizes and copied bits | 5.3, follow-up |
| `pw07_unit_kl_and_matched_legality` | KL per active unit, legality on equal terms | KL per unit, equal terms |
| `audits/independent_audit_001` | a separate re-implementation of the scoring for pw02a, pw03a and pw04a | Appendix A |

`docs/EXPERIMENT_LOG.md` has the full chronology, including the changes made after the pilots.

## Setup

Python 3.12 (3.11 or later should work) and `uv`.

```
uv sync
```

## Data

The Blocksworld sequences and the five small transformers come from my earlier thesis work and are not included
here (about 47 MB of data, plus the checkpoints). Two environment variables point to them.

- `PW_BLOCKSWORLD_DATA` is the folder with `blocksworld_{train,test}_{X,Y}.npy` and `blocksworld_metadata.json`.
  Default `data/blocksworld`.
- `PW_TRANSFORMER_CHECKPOINTS` is the folder with the transformer checkpoints `d4_w64_s1.pt`, `d4_w128_s1.pt`,
  `d4_w128_s2.pt`, `d4_w128_s3.pt` and `d4_w256_s1.pt`. Default `data/transformers`.

Every input file is checked against the SHA-256 recorded in the plans and scripts, so a different copy of the
data fails loudly. pw01, the pw02a sweep and the pw06 training read the Blocksworld data. Only the activation step of pw02a
needs the transformers. All other steps read the outputs of earlier runs. The figures, the report and the tests
need no external data.

Every entry point answers `--help` without the data.

## How to run

Run everything from the repository root. A registered run needs a `RUN.json` first, which this writes.

```
uv run python src/pwlib/runio.py start experiments/<pwNN_name> --run-id <run_id> --kind <pilot|full|analysis>
```

The results we report are already in `results/pilot_001`, `results/full_001` or `results/analysis_001` of each
folder. To redo a run, move its folder away first. `start` refuses an existing run, and later steps read
`pilot_001`, `full_001` and `analysis_001` by name. Model weights and most raw arrays are not shipped, so the later
steps only work after the earlier ones have been run again.

**pw01, ground truth (Section 3.1).**

```
uv run python src/pwlib/runio.py start experiments/pw01_state_space_ground_truth --run-id analysis_001 --kind analysis
uv run python experiments/pw01_state_space_ground_truth/run_ground_truth.py --run-id analysis_001
uv run python experiments/pw01_state_space_ground_truth/audit.py --run-id analysis_001
```

**pw02a, main sweep (Sections 5.1, 5.2 and 5.4).** First the pilot, then the full run that cites it.

```
uv run python src/pwlib/runio.py start experiments/pw02a_active_units_amended --run-id pilot_001 --kind pilot
uv run python -u experiments/pw02a_active_units_amended/extract_activations.py --run-id pilot_001
uv run python -u experiments/pw02a_active_units_amended/train_sweep.py --run-id pilot_001 --workers 8
uv run python experiments/pw02a_active_units_amended/analyze_active_units.py --run-id pilot_001
uv run python experiments/pw02a_active_units_amended/audit.py --run-id pilot_001

uv run python src/pwlib/runio.py start experiments/pw02a_active_units_amended --run-id full_001 --kind full --pilot-run pilot_001
uv run python -u experiments/pw02a_active_units_amended/extract_activations.py --run-id full_001
uv run python -u experiments/pw02a_active_units_amended/train_sweep.py --run-id full_001 --workers 8
uv run python experiments/pw02a_active_units_amended/analyze_active_units.py --run-id full_001
uv run python experiments/pw02a_active_units_amended/audit.py --run-id full_001
```

**pw03a, fact alignment (Section 5.5).** The pilot runs the same way on `pilot_001` of both experiments.

```
uv run python src/pwlib/runio.py start experiments/pw03a_fact_alignment_amended --run-id full_001 --kind full --pilot-run pilot_001
uv run python -u experiments/pw03a_fact_alignment_amended/analyze_alignment.py --run-id full_001 --pw02-run full_001 --workers 8
uv run python experiments/pw03a_fact_alignment_amended/audit.py --run-id full_001 --pw02-run full_001
```

**pw04a, sample legality (Section 5.6).** The registered pw04a stopped at its pilot, so the report uses a
descriptive pass of the same code on the full sweep. It writes to `descriptive/results/pw02a_full_001`.

```
uv run python src/pwlib/runio.py start experiments/pw04a_sample_legality_amended --run-id pilot_001 --kind pilot
uv run python -u experiments/pw04a_sample_legality_amended/analyze_legality.py --run-id pilot_001 --pw02-run pilot_001
uv run python experiments/pw04a_sample_legality_amended/audit.py --run-id pilot_001 --pw02-run pilot_001
uv run python experiments/pw04a_sample_legality_amended/descriptive_full_legality.py --pw02-run full_001
```

**pw05, post hoc analyses (Section 5.5).** The last line rebuilds the table of fact pairs that the figures use.

```
uv run python src/pwlib/runio.py start experiments/pw05_unit_anatomy --run-id analysis_001 --kind analysis
uv run python -u experiments/pw05_unit_anatomy/analyze_anatomy.py --run-id analysis_001 --workers 8
uv run python experiments/pw05_unit_anatomy/audit.py --run-id analysis_001
uv run python experiments/pw05_unit_anatomy/presentation/tabulate_pairs.py experiments/pw05_unit_anatomy/presentation/q3_pairs.json
```

**pw06, follow-up (Section 5.3).**

```
uv run python src/pwlib/runio.py start experiments/pw06_encoding_and_latent_size --run-id pilot_001 --kind pilot
uv run python -u experiments/pw06_encoding_and_latent_size/run_pw06.py --run-id pilot_001 --workers 8
uv run python experiments/pw06_encoding_and_latent_size/pilot_check.py --run-id pilot_001
uv run python src/pwlib/runio.py start experiments/pw06_encoding_and_latent_size --run-id full_001 --kind full --pilot-run pilot_001
uv run python -u experiments/pw06_encoding_and_latent_size/run_pw06.py --run-id full_001 --workers 8
uv run python experiments/pw06_encoding_and_latent_size/analyze_pw06.py --run-id full_001
uv run python experiments/pw06_encoding_and_latent_size/audit_recount.py
```

**pw07, KL per unit and legality on equal terms (Sections 5.1 and 5.6).** Not registered, so no `RUN.json`.

```
uv run python -u experiments/pw07_unit_kl_and_matched_legality/analyze_pw07.py --run-id analysis_001 --workers 8
```

**Figures and report.**

```
uv run python report/make_figures.py
typst compile report/report.typ report/report.pdf
```

`make_figures.py` reads only the shipped results, plus the few raw arrays it needs, which are included. It
redraws the six figures and writes the numbers behind them, with their sources, to `report/figures/tables.md`.
The report needs Typst and the Libertinus Serif font.

The re-implementation runs from its own folder, in the order given in its `AUDIT_REPORT.md`.

## Results

Each run writes to `experiments/<pwNN_name>/results/<run_id>/`. The main file is `results.json`, next to an audit
record (`AUDIT.json`, `audit_recompute.json`) and a list of every file with its hash (`MANIFEST.json`). Only the
small aggregate files are shipped, and they are the ones the report uses. Model weights and per-run arrays stay
out, except the 14 arrays the figures read.

## Tests

```
uv run pytest
```

The tests need no external data. They use the shipped plan files and the pw01 table of legal states.
`tests/crosscheck/dlib_crosscheck.py` is not part of pytest. It compares our MIG and DCI with
disentanglement_lib in a separate environment, and its docstring says how.
