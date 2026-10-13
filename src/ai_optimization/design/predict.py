"""Predict historical local-region MLP scores from cached or fresh embeddings."""
from pathlib import Path
import argparse
import json
import numpy as np
from common import DATA_ROOT, MODEL_ROOT, RESULTS_ROOT, LOGS_ROOT, ROOT, CONFIG, EDGES, domains, load_fasta, normalize, validate, write_json


class FrozenMLP:
    """Numeric snapshot of the historical sklearn pipeline; no pickle execution."""
    def __init__(self, model_dir=None):
        model_dir = Path(model_dir) if model_dir else MODEL_ROOT
        self.weights = np.load(model_dir / "historical_mlp_weights.npz", allow_pickle=False)
        self.metadata = json.loads((model_dir / "historical_mlp_metadata.json").read_text())
        self.revision = self.metadata["checkpoint_revision"]
        self.version = self.metadata["model_version"]

    def predict(self, features):
        x = np.asarray(features)
        if x.ndim != 2 or x.shape[1] != 128 or not np.isfinite(x).all():
            raise ValueError("Expected a finite [n,128] feature matrix.")
        w = self.weights
        x = x[:, w["variance_mask"]].copy()
        # Preserve sklearn's float32 input transformations before the float64 MLP.
        x -= w["input_mean"].astype(x.dtype)
        x /= w["input_scale"].astype(x.dtype)
        hidden = np.maximum(x @ w["weight_0"] + w["bias_0"], 0)
        output = hidden @ w["weight_1"] + w["bias_1"]
        output = output * w["target_scale"] + w["target_mean"]
        return np.maximum(output.ravel(), 0)


class FeatureProvider:
    def __init__(self, cache=None, fresh=False, checkpoint=None, expected_revision=None, cached_only=False):
        self.cache = {}
        self.fresh = fresh
        self.cached_only = cached_only
        if fresh and cached_only:
            raise ValueError("Fresh encoding and cached-only inference are mutually exclusive.")
        self.checkpoint = checkpoint
        self.encoder = None
        self.expected_revision = expected_revision or json.loads((MODEL_ROOT / "historical_mlp_metadata.json").read_text())["checkpoint_revision"]
        if cache is not None and not fresh:
            with np.load(cache, allow_pickle=False) as archive:
                x = archive["X"]
                sequences = archive["sequences"].tolist()
                if x.shape != (len(sequences), 128):
                    raise ValueError("Cache shape mismatch.")
                self.cache = {s: row for s, row in zip(sequences, x)}

    def featurize(self, sequences):
        sequences = [normalize(s) for s in sequences]
        missing = sorted(set(s for s in sequences if s not in self.cache))
        if missing and self.cached_only:
            raise ValueError(f"Cache lacks {len(missing)} local sequences; allow offline encoding or supply a matching cache.")
        if missing:
            # Fresh inference is local and offline; it never downloads a checkpoint.
            from hyena_features import FrozenHyena
            if self.encoder is None:
                self.encoder = FrozenHyena(checkpoint=self.checkpoint)
                if self.encoder.revision != self.expected_revision:
                    raise ValueError("Encoder revision differs from the trained predictor.")
            for start in range(0, len(missing), 128):
                batch = missing[start:start + 128]
                for sequence, vector in zip(batch, self.encoder.featurize(batch)):
                    self.cache[sequence] = vector
        return np.asarray([self.cache[s] for s in sequences], dtype=np.float32)


def predict_tfna(strands, provider, model):
    errors = validate(strands)
    if errors:
        raise ValueError("Sequence constraints failed: " + ", ".join(errors))
    ds = domains(strands)
    features = provider.featurize([ds[e] for e in EDGES])
    predictions = model.predict(features)
    training = set()
    import csv
    from common import DATA_ROOT, MODEL_ROOT, RESULTS_ROOT, LOGS_ROOT, reverse_complement
    with (DATA_ROOT / "sequence_count_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        training = {min(row["sequence_20nt"], reverse_complement(row["sequence_20nt"])) for row in csv.DictReader(handle)}
    regional = []
    for edge, value in zip(EDGES, predictions):
        v = CONFIG["edges"][edge]
        regional.append({"edge": edge, "sequence_20nt": ds[edge],
                         "strand_a": v["strand_a"], "start_a_1based": v["start_a"],
                         "strand_b": v["strand_b"], "start_b_1based": v["start_b"],
                         "unrounded_local_count_estimate": float(value),
                         "rounded_local_count": int(np.floor(value + .5)),
                         "exact_or_RC_input_in_training": min(ds[edge], reverse_complement(ds[edge])) in training})
    return {"model_version": model.version,
            "checkpoint_revision": model.revision, "all_six_edges_verified": True,
            "region_predictions": regional,
            "sum_unrounded_local_estimates_PROXY": float(predictions.sum()),
            "score_unit": "dimensionless screening proxy",
            "whole_tetrahedron_loading_validated": False,
            "scope": "Sum of six local docking-site estimates, not CK energy, simultaneous loading, or an independent test result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fasta", type=Path)
    inputs.add_argument("--sequence")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path, default=DATA_ROOT / "local_feature_cache.npz")
    encoding = parser.add_mutually_exclusive_group()
    encoding.add_argument("--fresh", action="store_true", help="Ignore the bundled cache and compute offline embeddings.")
    encoding.add_argument("--cached-only", action="store_true", help="Require every sequence in the cache; never start an encoder.")
    parser.add_argument("--checkpoint", type=Path, help="Local checkpoint; defaults to models/ai_optimization/design/checkpoint/.")
    parser.add_argument("--model-dir", type=Path, help="Numeric MLP assets; defaults to models/ai_optimization/design/.")
    args = parser.parse_args()
    model = FrozenMLP(args.model_dir)
    provider = FeatureProvider(args.cache, args.fresh, args.checkpoint, model.revision, args.cached_only)
    if args.fasta:
        result = predict_tfna(load_fasta(args.fasta), provider, model)
    else:
        sequence = normalize(args.sequence)
        value = float(model.predict(provider.featurize([sequence]))[0])
        result = {"sequence_20nt": sequence, "unrounded_local_count_estimate": value,
                  "rounded_local_count": int(np.floor(value + .5)),
                  "target": "local docking-site count at S=-E_Vina >= 8.0 kcal/mol",
                  "model_version": model.version}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
