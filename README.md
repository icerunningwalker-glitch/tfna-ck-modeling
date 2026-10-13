# tFNA–CK computational modeling

Reproducible code for pair-preserving tFNA sequence design, sequence-model benchmarking,
virtual screening and reference-carrier skin transport. The project includes preserved
inputs, fitted models, example results and evaluation records.

The publication release is `tfna_ck_publication_20261013`. The scientific model snapshots
remain the historical MLP (20261001) and the separate seven-algorithm benchmark (20261003).
The competition track is **赛道四：递送载体设计** (Track 4: delivery carrier design).

## Quick start

Cached sequence inference needs Python and NumPy:

```sh
python -m pip install -r requirements-inference.txt
python predict.py --output-dir runs/predict
```

For the CPU demonstration, reference transport and saved-model evaluation:

```sh
python -m pip install -r requirements-verified.txt
python verify_submission.py
python demo.py --output-dir runs/demo
python verify_submission.py --models
```

Each output directory must be new. The default prediction produces 84 real saved sequence
designs in UTF-8 `results.csv`. The demonstration re-scores the saved design pool, selects
candidates, runs an explicitly synthetic docking software fixture, and computes reference
carrier transport. Synthetic docking stays in its own folder; the final scientific table
contains the sequence-model predictions.

## Environments

| Use | Dependency file | Recorded environment |
|---|---|---|
| Cached inference | `requirements-inference.txt` | Python 3.12.14, NumPy 2.3.5 |
| Publication CPU replay and transport | `requirements-verified.txt` | Windows x86-64, Python 3.13.9; versions recorded in `logs/publication/` |
| Historical packaging environment | `requirements.txt` | Linux x86-64, Python 3.12.14; original saved estimator version scikit-learn 1.9.0 |
| New HyenaDNA encoding or full training | `requirements-full.txt` | Pinned encoder dependencies; optional offline TabPFN checkpoint |

The default example uses CPU. New encoding uses the frozen local HyenaDNA checkpoint.
TabPFN V2 weights are acquired separately under the provider's terms and supplied with
`--tabpfn-checkpoint`. Original training hardware and elapsed time were not retained;
the current review records its actual environment and elapsed time. Saved scikit-learn
1.9.0 estimators were also replayed under 1.7.2 with numerical comparisons and version
warnings recorded. The historical packaging environment and current replay environment are recorded separately.

## Workflows

### Pair-preserving sequence optimization

```sh
python design.py --output-dir runs/design_replay
python design.py --generate --output-dir runs/new_design
python predict.py --input my_candidates.csv --cached-only --output-dir runs/custom
python train.py historical-mlp --output-dir runs/historical_training
```

Four 63-nt strands define six complementary 20-nt edges. The historical frozen encoder
provides 128-dimensional local sequence features; the trained MLP estimates a local
docking-site count. The six-edge sum is a dimensionless screening proxy, ordered higher
first. The output retains its model identity and mutation counts. One paired-site edit
changes both complementary strands. WT, Opt1, Opt2 and Opt3 have 0, 3, 1 and 4 paired-site
changes, corresponding to 0, 6, 2 and 8 changed nucleotides.

The stored pool contains 1,854 constructs, including four comparators; 84 selected
constructs provide the default replay. New generation is explicit. `--cached-only`
requires every input sequence to have a saved feature; `--fresh` computes features
from the pinned local checkpoint. Model scope and dataset provenance are in
`models/MODEL_CARD.md` and `data/DATA_PROVENANCE.md`.

### Model benchmark

```sh
python src/ai_optimization/benchmark/verify_all7.py
python src/ai_optimization/benchmark/evaluate_models.py
python train.py benchmark --help
```

The preserved benchmark contains 123 local sequences, labels 0/1/2, five outer folds,
and three family-grouped inner folds. Preprocessing and PCA are fitted inside the
training folds. The saved fold membership is the reproduction reference. Seven
algorithms and two baselines produce 1,107 unique model/sample OOF records; repeated
training-fit predictions remain separately labelled diagnostics. See the module README
for commands using independent result, model, log and input directories.

### Reference-carrier transport

```sh
python transport.py --output-dir runs/transport
python src/transport/verify_results.py --generated-dir runs/transport
```

This finite-volume model connects a follicular reservoir, tissue diffusion, cellular
uptake and degradation under the preserved TH21 inputs and 15-minute application.
It outputs intact-carrier mass, concentration, uptake and retention, plus mass-balance
and supply-cutoff checks. `summary.dose_ug` uses a per-area mass baseline: its unit is
µg/cm²; for a 1 cm² modeled area it has the same numerical value as total µg.
The 2 CK/carrier design assumption supports a separate cargo-equivalent conversion.
The implemented output tables describe the carrier states. Parameter sources are
recorded in `data/transport/` and `logs/transport/Data_Sources.csv`.

### Virtual screening

```sh
python screen.py --manifest my_data/ligands.csv --receptor my_data/receptor.pdbqt --vina tools/vina --output-dir runs/real_screen --reference-id REF --track "赛道四：递送载体设计" --execute
python predict.py --source screening --screening-run runs/real_screen --output-dir runs/screen_export
```

The real workflow uses prepared receptor/ligand inputs and an externally installed Vina.
The eight overlapping search boxes are ranking observations under the recorded docking
protocol. Structure files from a completed real run are copied with relative references
into the final candidate table. Input templates and chemical checks are included.

## Repository layout and submission fields

The repository root is the competition `project/` directory: `data/`, `src/`, `models/`,
`results/`, `logs/`, `config/`, `notebooks/`, and the command entrypoints live here.
`config/result_schema.json` specifies identifiers, track, sequences/structure, metric,
units, direction, model/run versions and notes. Ranking is local to a computational
module. `notebooks/reproduce.ipynb` executes the same root entrypoints.

`SUBMISSION_CHECKLIST.md` maps the attachment requirements to the supplied assets and
actual checks. `PUBLICATION_CHANGES.md` records the portable-code fixes.

## Skills and license

Reusable agent skills are published separately in
[scientific-agent-skills](https://github.com/icerunningwalker-glitch/scientific-agent-skills).
The DNA-origami generator has its own
[generate-cadnano](https://github.com/icerunningwalker-glitch/generate-cadnano) repository.

Original project code, documentation and owner-produced docking labels/designs are
released under MIT. The pretrained encoder and other third-party material retain their
own terms and attribution in `THIRD_PARTY_NOTICES.md`. Public literature attribution is
retained. The released files use relative/configurable paths and scientific identifiers.
