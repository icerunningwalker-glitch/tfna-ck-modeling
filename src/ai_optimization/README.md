# AI optimization source module

The module covers tFNA sequence constraints, frozen HyenaDNA features, historical MLP scoring, pair-preserving design/selection, and the separate current seven-model benchmark. Scientific inputs, models, saved results and logs remain distinct from source code.

| Contents | Location relative to project root |
|---|---|
| Source | src/ai_optimization/design/ and src/ai_optimization/benchmark/ |
| Inputs and feature caches | data/ai_optimization/design/ and data/ai_optimization/benchmark/ |
| MLP, encoder and saved fold models | models/ai_optimization/design/ and models/ai_optimization/benchmark/ |
| Saved scores and CV predictions | results/ai_optimization/design/ and results/ai_optimization/benchmark/ |
| Parameters, plans and validation | logs/ai_optimization/design/ and logs/ai_optimization/benchmark/ |

Default inputs are resolved from the source-file location. These explicit entry points work from any cwd; commands below assume the project root. Existing outputs are never overwritten.

```text
python src/ai_optimization/predict.py --output-dir results/ai_optimization/runs/predict --track YOUR_TRACK
python src/ai_optimization/design.py --output-dir results/ai_optimization/runs/design --track YOUR_TRACK
python src/ai_optimization/verify_package.py
python src/ai_optimization/design/predict.py --fasta data/ai_optimization/design/four_constructs/Opt3.fasta --output results/ai_optimization/runs/Opt3.json
python src/ai_optimization/design/mutation_design.py edit --id candidate --edits "E12:11T>G" --output results/ai_optimization/runs/candidate.fasta
```

predict.py scores the saved 84 constructs, exporting candidate_scores.csv, results.csv and run.json. It accepts --input for another four-strand CSV and --fresh for offline encoding. design.py scores the saved 1,854-construct pool and replays the historical diversity selection. Its optional --generate actually produces a new pool; this option was not run during reorganization.

```text
python src/ai_optimization/train.py historical-mlp --output-dir models/ai_optimization/design/runs/refit
python src/ai_optimization/train.py benchmark --check-inputs
python src/ai_optimization/train.py benchmark --output-dir results/ai_optimization/benchmark/runs/refit --skip-tabpfn
```

Historical training logs default to logs/ai_optimization/design/runs/<output-name>. New benchmark models and logs default to models/ai_optimization/benchmark/runs/<output-name> and logs/ai_optimization/benchmark/runs/<output-name>; --model-dir and --log-dir can set explicit new locations. A complete seven-model run requires a licensed user-provided local TabPFN V2 checkpoint via --tabpfn-checkpoint. Those weights are not redistributed. No training was performed in packaging or verification.

## Two separate model versions

The historical candidate MLP uses 128 encoder features, a variance filter, input scaling, 16 ReLU hidden units, target scaling and seed 20261001. It estimates local spatially deduplicated docking-site counts at S=-E_Vina >=8.0 kcal/mol. Its six-edge sum is a dimensionless screening proxy, not docking energy, measured CK loading, or globally collision-free occupancy. Cached inference uses a pure numerical NPZ export and NumPy. The original trusted joblib is retained for source identity and training reproduction.

The separate current benchmark uses seven models, five outer count-stratified folds with seed 20261003, three family-grouped inner tuning folds, and training-only scaling/PCA8. Saved OOF scores concern within-library development evaluation, not an external test. They must not replace the historical candidate-model scores. Only two sequences have target count 2, and related families can cross the outer split.

One paired-site change edits two complementary nucleotides. WT, Opt2, Opt1 and Opt3 have 0, 1, 3 and 4 paired-site changes. Historical scores are 3.034244, 3.943306, 4.935741 and 4.935741. Opt1/Opt3 tie because this local model gives zero contribution to their differing E24 domain; the tie is not an experimental loading result.

## Actual checks

Validation replays 84 constructs, scores the existing 1,854 pool, checks identical selected IDs/order/sequences, recomputes saved CV arithmetic and replays 35 saved non-TabPFN fold snapshots without fitting. TabPFN saved predictions are verified arithmetically; fresh TabPFN inference needs the separately licensed checkpoint. Reports are in logs/ai_optimization/.

The module hash manifest uses project-relative names. Personal identities, account data, hostnames, machine-specific paths and runtime caches are excluded. Upstream component attribution is preserved in project notices. Root-level installation instructions apply.
