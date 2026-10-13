# tFNA sequence design and historical MLP inference

This folder is portable. Run commands from the package root with `python`; outputs use a new directory and never overwrite bundled inputs or frozen assets. No project-private paths, names, account identifiers, or runtime logs are needed.

## What this module reproduces

- Four existing constructs: WT, Opt1, Opt2, Opt3; each has four 63-nt strands.
- Six antiparallel, complementary 20-bp edges; linker positions 1, 22, 43 are excluded from encoder inputs.
- Frozen HyenaDNA tiny, revision `e8c1effa8673814e257e627d2e1eda9ea5a373f6`: mean over 20 real-base embeddings and average of forward/reverse-complement orientations, yielding 128 features.
- The **historical candidate-scoring MLP**: variance filter, input standardization, 16-unit ReLU hidden layer, output layer, inverse target standardization, nonnegative clipping. This is the frozen model used for the provided 84 scores, not the later seven-model five-fold benchmark's fitted models.
- Pair-preserving candidate generation and score/diversity selection, provided as optional commands. Saved candidate pools and scientific outputs retain their original values; new runs use explicit output paths.

One paired-site change modifies two nucleotides, one in each complementary strand. Opt2, Opt1, Opt3 have 1, 3, 4 paired-site changes versus WT, corresponding to 2, 6, 8 total nucleotide substitutions.

## Minimal example and checks

```text
python src/ai_optimization/design/mutation_design.py validate --fasta data/ai_optimization/design/four_constructs/Opt3.fasta
python src/ai_optimization/design/predict.py --fasta data/ai_optimization/design/four_constructs/Opt3.fasta --cached-only --output results/ai_optimization/design/runs/Opt3.json
python src/ai_optimization/design/smoke_check.py --output results/ai_optimization/design/runs/replay_check.json
```

Cached replay requires NumPy only. `--cached-only` rejects uncached inputs before loading an encoder; the default uses cached features and starts the local offline encoder for missing sequences. `--fresh` ignores the cache and recomputes every input feature. `FrozenMLP` reads a numeric NPZ snapshot. The original joblib file is included for source identity and optional sklearn reproduction; load it only from a trusted source.

Fresh offline HyenaDNA inference for an uncached sequence also requires Torch, Transformers, tokenizers, safetensors, and the bundled checkpoint. It does not download anything:

```text
python src/ai_optimization/design/predict.py --fasta data/ai_optimization/design/four_constructs/Opt3.fasta --fresh --output results/ai_optimization/design/runs/Opt3_fresh.json
```

## Design, scoring, and selection

Explicit edits use the first-oriented edge domain, with 1-based edge positions. Complementary edits are automatic:

```text
python src/ai_optimization/design/mutation_design.py edit --id candidate --edits "E12:11T>G" --output results/ai_optimization/design/runs/candidate.fasta
python src/ai_optimization/design/predict.py --fasta results/ai_optimization/design/runs/candidate.fasta --output results/ai_optimization/design/runs/candidate_score.json
```

Score the saved pool or a custom CSV containing candidate_id and strand_1c/strand_2c/strand_3c/strand_4c:

```text
python src/ai_optimization/design/score_candidates.py --input data/ai_optimization/design/historical_candidate_pool.csv --cached-only --output results/ai_optimization/design/runs/pool_scores.csv
python src/ai_optimization/design/select_candidates.py --input results/ai_optimization/design/runs/pool_scores.csv --output results/ai_optimization/design/runs/selected_84.csv
```

Optional historical generation enumerates single paired substitutions and samples 250 within-edge plus 250 cross-edge candidates per burden 2, 3, 4, with seed 20261002. The previous pool is supplied as sequence hashes to preserve exclusions without requiring old directories. Each rejected construct is logged in the newly specified output directory. The supplied historical pool can be replayed directly; full-size generation is a separate run:

```text
python src/ai_optimization/design/generate_candidates.py --output-dir results/ai_optimization/design/runs/new_pool
```

`--seed`, `--per-pattern`, `--max-attempts`, and `--prior-pool-hashes` are explicit options. Changing these changes the pool. Multi-site sampling is not exhaustive. Historical selection returns the four saved comparators plus 20 per burden; burdens 2-4 include 10 within-edge and 10 cross-edge constructs, using score ordering and a minimum paired-sequence distance of 2 before fallback. Diversity is enforced separately in each selected group; a shortage invokes a score-order fallback. These 84 entries are a stratified selection rather than the first 84 of the globally sorted pool. Selection requires generation metadata (`kind`, `generation_pattern`, `paired_site_mutations`) and the four comparator IDs.

## Original full-data MLP training entry

The original model was selected in a historical seven-model comparison. Its original final-fit source was `train_shuffled.py`; this module reproduces the MLP-specific final-fit procedure, with the same 123 training vectors and labels, binary-presence stratification, seed 20261001, five tuning folds, integer-count MAE, grid hidden units 8/16 and alpha 1/10/100, and a full-data refit. Historical selected values were 16 hidden units and alpha 1. This is not a new evaluation protocol and does not rerun seven-model selection.

```text
python src/ai_optimization/design/train_historical_mlp.py --output-dir models/ai_optimization/design/runs/retrained_historical_mlp
python src/ai_optimization/design/predict.py --fasta data/ai_optimization/design/four_constructs/Opt3.fasta --model-dir models/ai_optimization/design/runs/retrained_historical_mlp --output results/ai_optimization/design/runs/Opt3_retrained.json
```

Training requires the recorded sklearn/NumPy environment for closest numerical reproduction; it was not executed during packaging. Version changes can affect fitted results. Bundled `model/` remains frozen; retrained assets go to the explicit new directory. The training score is used for tuning, not an independent test-performance claim.

## Scientific meaning

The target is a local count of spatially deduplicated docking representatives with `S=-E_Vina >= 8.0 kcal/mol`. Counts derive from sampled local docking. A continuous local estimate is not a docking energy or binding probability. Summing six local estimates produces a **dimensionless screening proxy**, not a measured number of CK molecules on a complete tFNA. The sum does not resolve cross-edge clashes or global duplication. Training-sequence overlaps are reported and four-construct predictions are not independent external validation.

WT, Opt1, Opt2, Opt3 historical scores are 3.034244, 4.935741, 3.943306, 4.935741. Opt1 and Opt3 tie because this local model gives zero contribution to their differing E24 region. These scores must not be relabeled as outputs of the later benchmark model.

## Dependencies and traceability

Cached replay: Python and NumPy. Original historical training: NumPy 2.5.2, scikit-learn 1.9.0, joblib 1.6.0, threadpoolctl. Original fresh encoding: Python 3.12.14, Torch 2.11.0, Transformers 4.36.2, tokenizers 0.15.2, safetensors 0.8.0. CUDA is unnecessary for this CPU encoder. Newer versions have not been assumed to reproduce all training behavior.

Paths are resolved relative to the package, so the commands also run when another directory is current. `data/ai_optimization/design/` contains the reference, labels, cached features and saved sequences. `models/ai_optimization/design/` contains the frozen model and `checkpoint/source_manifest.json`. `results/ai_optimization/design/` contains historical scored and selected tables. `logs/ai_optimization/design/` contains prior validation records. Choose a new explicit output directory for each run. Upstream copyright and component licenses are retained in the package notices.

## Restore or verify the frozen encoder

The checkpoint is already bundled. Verify its fixed manifest without network access:

```sh
python src/ai_optimization/fetch_weights.py hyenadna --verify-only --output-dir models/ai_optimization/design/checkpoint
```

To restore a separate copy, use a new directory and then the low-level inference entry:

```sh
python src/ai_optimization/fetch_weights.py hyenadna --output-dir external_weights/hyenadna
python src/ai_optimization/design/predict.py --fasta data/ai_optimization/design/four_constructs/Opt3.fasta --fresh --checkpoint external_weights/hyenadna --output results/ai_optimization/design/runs/Opt3_restored.json
```

The downloader verifies every file against the pinned revision and bundled SHA256 manifest. Provider license and attribution are recorded in THIRD_PARTY_NOTICES.md.
