# Model card

## Historical candidate scorer

Version: `historical_frozen_MLP_16_hidden_20261001`.

Input: a local 20-nt A/C/G/T sequence. Encoder: frozen `LongSafari/hyenadna-tiny-1k-seqlen-hf`, revision `e8c1effa8673814e257e627d2e1eda9ea5a373f6`, two layers, 128 channels. Mean-pool the 20 actual bases in each forward/RC orientation and average the two vectors; no extra token or padding. The pretrained encoder was not fine-tuned.

The project-trained head is variance filtering, input StandardScaler and a ReLU MLP (16 hidden units), with output target StandardScaler inverted at prediction. Solver `lbfgs`, alpha 1, maximum iterations 2000, tolerance 1e-5, seed 20261001. Training has 123 distinct local sequences (reverse complements canonicalized); count labels 0/1/2 occur 31/90/2 times. Historical model selection used five outer/five inner presence-stratified folds and integer-count MAE; the packaged training entry reproduces the MLP final search/full refit, not the original entire seven-model selection. Frozen original joblib SHA256: `7f84908119a43d0083772759f842c72ce69a77ede80f6602e879d0a5e47c57ae`.

Bundled NPZ exports the same fitted numerical transforms and weights to permit NumPy-only inference. The original joblib remains for identity and optional sklearn use; only load trusted pickle/joblib sources.

Output: nonnegative continuous predicted count of sampled, spatially deduplicated local docking representatives passing `-E_Vina >= 8.0 kcal/mol`. For a complete construct, six local estimates are summed into a dimensionless screening proxy. No energy, binding probability or measured occupancy is returned. Opt2/Opt1/Opt3 have 1/3/4 paired changes (2/6/8 nucleotides). WT/Opt1/Opt2/Opt3 scores are approximately 3.034244/4.935741/3.943306/4.935741; the equal Opt1/Opt3 sum reflects zero predicted contribution from their differing E24 region.

## Current comparison

Version: `benchmark_single_seed_5fold_20261003`; seven regressors MLP/SVR/Ridge/Random forest/ExtraTrees/GPR/TabPFN, plus training-median and one-hot Ridge baselines. Same 123 sequences and cached 128 features; labels rebuilt from uniform docking seed 104729. Five shuffled count-stratified outer folds (seed 20261003), test sizes 25/25/25/24/24; three family-grouped inner folds select parameters by continuous nonnegative MAE. Input variance filter/scaler/PCA8 and target scaler are fitted on each training set only.

OOF predictions contain one held-out prediction per sequence/model, not external experiments. TabPFN uses pretrained V2 inference with fitted training context; no gradient fine-tuning. The TabPFN weight must be obtained separately. Saved results permit metric checking without that weight. Metrics and all folds are retained; MLP was not selected by deleting competing models.

## Limits and intended use

For prioritizing candidates in this four-strand topology. Local docking labels depend on sampling, pose clustering and preparation. There are only two count-2 sequences; related sequence families can cross outer folds. Historical four-construct predictions overlap training sequences and are not independent validation. The six-edge sum ignores global competition/clashes and is not whole-tFNA loading. No uncertainty interval has been validated for individual new candidates. No biological efficacy, safety or delivery outcome is inferred.

No retraining, docking or MD was run while creating this package. Cached candidate scores and saved evaluation results were actually recomputed; checks are supplied. Full retraining is available as code but not retested here. Original training hardware/runtime and exact pretrained-data acquisition timestamp were not completely recorded.
