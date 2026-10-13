"""Reproduce the historical MLP full-data fit; this is not the new benchmark."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from common import DATA_ROOT, MODEL_ROOT, RESULTS_ROOT, LOGS_ROOT, ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for retrained assets; bundled frozen assets are never overwritten.")
    parser.add_argument("--log-dir", type=Path,
                        help="New parameter-log directory; defaults to logs/ai_optimization/design/runs/<output-name>.")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new empty output directory.")
    log_dir = args.log_dir.resolve() if args.log_dir else LOGS_ROOT / "runs" / args.output_dir.name
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError("Use a new empty log directory.")
    import joblib
    from sklearn import __version__ as sklearn_version
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    path = DATA_ROOT / "historical_training_features.npz"
    with np.load(path, allow_pickle=False) as archive:
        features = archive["X"].astype(np.float64)
        target = archive["y"].astype(int)
    with (DATA_ROOT / "sequence_count_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = np.asarray([int(row["target_tight_site_count_S_ge_8p0"]) for row in rows])
    if features.shape != (123, 128) or not np.array_equal(target, expected):
        raise ValueError("Historical feature/label identity mismatch.")
    seed = 20261001
    splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(features, target >= 1))
    pipeline = Pipeline([("variance", VarianceThreshold()), ("scale", StandardScaler()),
                         ("model", MLPRegressor(solver="lbfgs", max_iter=2000, tol=1e-5, random_state=seed))])
    estimator = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())

    def integer_count_score(model, x, y):
        prediction = np.floor(np.maximum(model.predict(x), 0) + .5).astype(int)
        return -mean_absolute_error(y, prediction)

    grid = {"regressor__model__hidden_layer_sizes": [(8,), (16,)],
            "regressor__model__alpha": [1, 10, 100]}
    search = GridSearchCV(estimator, grid, scoring=integer_count_score,
                          cv=splits, n_jobs=1, error_score="raise")
    with threadpool_limits(limits=1):
        search.fit(features, target)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((MODEL_ROOT / "historical_mlp_metadata.json").read_text())
    payload = {key: metadata[key] for key in ("model_name", "features", "encoder", "checkpoint_revision", "target", "integer_rule", "pooling", "scope")}
    payload["estimator"] = search.best_estimator_
    joblib.dump(payload, args.output_dir / "hyena_count.joblib")
    fitted = search.best_estimator_
    pipe = fitted.regressor_
    mlp = pipe.named_steps["model"]
    scale = pipe.named_steps["scale"]
    np.savez_compressed(args.output_dir / "historical_mlp_weights.npz",
                        variance_mask=pipe.named_steps["variance"].get_support(),
                        input_mean=scale.mean_, input_scale=scale.scale_,
                        weight_0=mlp.coefs_[0], bias_0=mlp.intercepts_[0],
                        weight_1=mlp.coefs_[1], bias_1=mlp.intercepts_[1],
                        target_mean=fitted.transformer_.mean_, target_scale=fitted.transformer_.scale_)
    metadata["model_version"] = "retrained_historical_MLP_full_fit"
    metadata["fitted_hidden_layer_sizes"] = list(mlp.hidden_layer_sizes)
    metadata["fitted_alpha"] = mlp.alpha
    metadata["model_joblib_sha256"] = hashlib.sha256((args.output_dir / "hyena_count.joblib").read_bytes()).hexdigest()
    write_json(args.output_dir / "historical_mlp_metadata.json", metadata)
    report = {"status": "COMPLETE", "model_version": "retrained_historical_MLP_full_fit",
              "seed": seed, "training_sequences": len(target), "feature_dimensions": 128,
              "stratification": "binary presence: target >= 1", "folds": 5,
              "tuning_metric": "integer-count MAE after nonnegative clipping and half-up rounding",
              "best_parameters": search.best_params_, "best_internal_tuning_MAE": -float(search.best_score_),
              "sklearn": sklearn_version, "numpy": np.__version__,
              "source_features_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "historical_selected_parameters": {"hidden_layer_sizes": [16], "alpha": 1},
              "scope": "Original MLP full-fit procedure after historical seven-model selection; the reported tuning score is not a new held-out performance estimate.",
              "not_the_new_seven_model_five_fold_benchmark": True,
              "bitwise_identity_across_dependency_versions_guaranteed": False}
    write_json(log_dir / "training_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
