# Seven-model benchmark: portable, anonymous reproduction

This module packages the actual five-fold benchmark completed on 2026-10-03. It contains the original scientific results, aligned input data and fitted non-TabPFN fold models. Packaging does not retrain, tune, dock or run MD.

## What is predicted

There are 123 unique local 20-nt DNA sequences. Labels are sampled tight docking-site counts under docking seed 104729, using Vina scores at or below -8.0 kcal/mol. Counts 0/1/2 occur 31/90/2 times. These labels are local docking proxies. Sequence/family IDs are anonymous scientific identifiers.

The frozen HyenaDNA representation has 128 features per sequence. `data/ai_optimization/benchmark/hyena_features.npz` preserves the original feature values, adds explicit ordered sequence IDs, and replaces its old mixed-seed label array with the actual single-seed labels used for this benchmark. This is an alignment/data-packaging change; saved predictions and fitted model parameters are unchanged. The cache reproduces this benchmark with the packaged fixed features. Encoding new sequences requires the compatible encoder supplied separately by the package's feature module.

## Saved protocol and results

- Seven predictors: MLP, SVR, Ridge, Random forest, ExtraTrees, GPR and TabPFN. Two baselines: training-median and one-hot Ridge.
- Outer five-fold shuffled cross-validation, stratified on exact 0/1/2 counts, seed 20261003. Test fold sizes: 25/25/25/24/24. Every sequence has exactly one OOF prediction per method.
- Three family-grouped inner folds select hyperparameters only inside each outer training set. The objective is minimum unweighted mean continuous nonnegative count MAE. Retraining reads the actual saved memberships in `training_group_CV_assignments.csv`, preserving fold allocation across family anonymization and dependency versions.
- Training-only variance filtering, scaling, fixed PCA with 8 components and target scaling are fitted inside the pipeline. One-hot Ridge uses canonical forward/reverse-complement 80-dimensional one-hot encoding and no PCA.
- Regression predictions are clipped to zero. ROC uses observed count >= 1; the confusion-matrix prediction threshold is 0.5. AUC describes discrimination, not calibrated probability.
- All sequences had already been analysed. Related families can cross outer folds. This is within-library development evaluation, not independent external or unseen-family validation. Inner group CV addresses a different, harder transfer task.
- Only two count-2 examples exist; they occupy two outer test folds. Fold variation should be inspected rather than interpreted as a large-sample uncertainty estimate.

The benchmark log directory contains `analysis_plan.json` with grids, seed, protocol and fold summaries, `training_only_parameters.json` with actual selected parameters, `training_CV_candidate_losses.csv` with all parameter-candidate losses, and `warnings.json` with convergence warnings. The five finally selected MLPs converged; a grid-search candidate reached its iteration limit. No further tuning was performed after inspecting OOF scores.

| Method | OOF AUC | MAE (count) | MSE (count squared) | RMSE (count) |
|---|---:|---:|---:|---:|
| MLP | 0.8973 | 0.2104 | 0.1279 | 0.3577 |
| SVR | 0.8447 | 0.2374 | 0.1489 | 0.3859 |
| Ridge | 0.8776 | 0.2635 | 0.1350 | 0.3674 |
| Random forest | 0.8661 | 0.2477 | 0.1547 | 0.3933 |
| ExtraTrees | 0.8713 | 0.2448 | 0.1426 | 0.3776 |
| GPR | 0.8678 | 0.2645 | 0.1389 | 0.3727 |
| TabPFN | 0.8531 | 0.2146 | 0.1521 | 0.3900 |

Pooled OOF metrics and unweighted averages of fold metrics are different quantities. No significance test was run. Training-fit predictions are diagnostic: each sequence has four in-sample fold-model predictions, and the combined train table averages those four values. These are not inner-CV predictions or independent validation.

## Run without training

Run commands from the package root. The current CPU publication review uses Windows x86-64, Python 3.13.9 and the root `requirements-verified.txt`; its actual package versions are recorded in `logs/publication/environment.json`. The earlier Linux/Python 3.12 packaging verification is retained as separate historical evidence below.

```bash
python -B src/ai_optimization/benchmark/verify_all7.py
python -B src/ai_optimization/benchmark/evaluate_models.py
python -B src/ai_optimization/benchmark/train_all7.py --check-inputs
python -B src/ai_optimization/benchmark/plot_all7.py --output-dir results/ai_optimization/benchmark/runs/new_plots
```

`verify_all7.py` uses only Python's standard library. It independently recalculates all saved metrics, checks OOF uniqueness, fold membership, training/test boundaries, family separation in inner folds and packaged input hashes. `evaluate_models.py` loads 35 trusted fitted snapshots and reproduces their OOF predictions without fitting. TabPFN inference is not executed by this script; its saved prediction arithmetic is still verified. Model deserialization requires compatible NumPy/scikit-learn/joblib versions. Never load an untrusted joblib file.

`plot_all7.py` produces the existing seven-model ROC/confusion layout and separate MSE/RMSE charts as 600-dpi PNG, SVG and preview PNG. It uses installed Arial when available; otherwise it explicitly reports a DejaVu Sans fallback. Supply `--font-file path/to/arial.ttf` to use your own Arial font. No font file is redistributed.

Optional validation output: `--output new_validation.json`. Existing output destinations are not overwritten. Existing saved scientific results remain unchanged.

## Explicit retraining on another machine

Training was not rerun while building this package. The portable training entry preserves the fixed seven-model protocol:

```bash
python -B src/ai_optimization/benchmark/train_all7.py --output-dir results/ai_optimization/benchmark/runs/new_all7 --tabpfn-checkpoint path/to/tabpfn-v2-regressor.ckpt
python -B src/ai_optimization/benchmark/verify_all7.py --results-dir results/ai_optimization/benchmark/runs/new_all7 --log-dir logs/ai_optimization/benchmark/runs/new_all7
python -B src/ai_optimization/benchmark/evaluate_models.py --results-dir results/ai_optimization/benchmark/runs/new_all7 --model-dir models/ai_optimization/benchmark/runs/new_all7
```

TabPFN weights are obtained separately from the [official V2 regressor repository](https://huggingface.co/Prior-Labs/TabPFN-v2-reg), under its provider license. The following explicit download restores `tabpfn-v2-regressor.ckpt` at revision `4972a65a1b30806315c6f92499959ffbfc69a673` and verifies SHA256 `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736` (44,390,977 bytes):

```sh
python src/ai_optimization/fetch_weights.py tabpfn --output-dir external_weights/tabpfn_v2
python src/ai_optimization/benchmark/train_all7.py --output-dir results/ai_optimization/benchmark/runs/new_all7 --tabpfn-checkpoint external_weights/tabpfn_v2/tabpfn-v2-regressor.ckpt
```

The current official file identity was checked on 2026-10-13. The historical run did not record its checkpoint hash, so this fixed file supports a new reproducible V2 run; historical numerical equality is assessed against the supplied saved predictions. The training entry itself remains offline. A missing local checkpoint raises an error before creating a run.

An explicitly reduced six-model run is possible with `--skip-tabpfn`; it must not be called a reproduction of the seven-model benchmark:

```bash
python -B src/ai_optimization/benchmark/train_all7.py --skip-tabpfn --output-dir results/ai_optimization/benchmark/runs/new_six_models
python -B src/ai_optimization/benchmark/verify_all7.py --results-dir results/ai_optimization/benchmark/runs/new_six_models --log-dir logs/ai_optimization/benchmark/runs/new_six_models
```

The saved plotting entry expects the full seven-model comparison. The optional reduced run is for training reuse and is not substituted into the published saved figures. `--data-dir` accepts the same input schema/order and fixed saved fold assignments; this entry reproduces this specific 123-sequence benchmark, not an arbitrary new dataset search.

New training writes timestamped `training_events.jsonl` to its log directory, together with runtime package versions, CPU configuration, seed, warnings, chosen parameters and elapsed seconds. Historical hardware and total training time were not retained in the original result files. Use `--data-dir`, `--results-dir`, `--log-dir` and `--model-dir` when checking a run whose directories differ from the defaults.

## Dependencies, attribution and gaps

The original benchmark report records scikit-learn 1.9.0. Historical packaging verification used Linux x86-64, Python 3.12.14, NumPy 2.5.2, scikit-learn 1.9.0, joblib 1.6.0, PyTorch 2.11.0+cu128 and TabPFN 9.0.0. `logs/ai_optimization/benchmark/environment_verification.json` preserves those observed versions; that inventory does not establish every dependency version from the original training date. The root `requirements.txt` retains the historical packaging environment pins.

The current Windows CPU publication review uses Python 3.13.9 and the root `requirements-verified.txt`, including scikit-learn 1.7.2. Actual versions are recorded in `logs/publication/environment.json`, and `logs/publication/review.json` records the numerical replay checks. Saved scikit-learn 1.9.0 snapshots were replayed under 1.7.2 with version warnings and numerical comparisons. Fresh model training and TabPFN checkpoint inference were not performed in this current review.

Built with PriorLabs-TabPFN. Official source: https://github.com/PriorLabs/TabPFN . scikit-learn source: https://github.com/scikit-learn/scikit-learn . Third-party package/checkpoint licenses still apply. This module does not grant a new license to any third-party model or checkpoint.

The benchmark does not include raw docking pose files, absolute source paths, experiment screenshots, document authors, names, contact details, machine identifiers, Python bytecode or hidden credentials. The TabPFN V2 checkpoint is not packaged. The pinned HyenaDNA checkpoint is included at `models/ai_optimization/design/checkpoint/` under its applicable third-party terms. Retained data are the scientific sequence inputs, anonymous identifiers, count labels, cached features and reproducibility outputs.
