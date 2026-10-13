"""Check all 84 existing inputs and replay saved scores; no generation/training."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from common import DATA_ROOT, MODEL_ROOT, RESULTS_ROOT, LOGS_ROOT, ROOT, WT, domains, load_fasta, mutation_record, validate, write_json
from predict import FrozenMLP, FeatureProvider, predict_tfna


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fresh-opt3", action="store_true", help="Also verify the six Opt3 regions using the offline encoder.")
    args = parser.parse_args()
    model = FrozenMLP()
    provider = FeatureProvider(DATA_ROOT / "local_feature_cache.npz")
    with np.load(DATA_ROOT / "local_feature_cache.npz", allow_pickle=False) as archive:
        predictions = model.predict(archive["X"])
        local_error = float(np.max(np.abs(predictions - archive["predictions"])))
    with (RESULTS_ROOT / "historical_selected_84.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checks = []
    for row in rows:
        strands = load_fasta(DATA_ROOT / "candidates_84" / (row["candidate_id"] + ".fasta"))
        if any(strands[name] != row["strand_" + name] for name in WT):
            raise ValueError("FASTA/table mismatch: " + row["candidate_id"])
        errors = validate(strands)
        if errors or len(mutation_record(strands)) != int(row["paired_site_mutations"]):
            raise ValueError("Sequence or paired-mutation mismatch: " + row["candidate_id"])
        result = predict_tfna(strands, provider, model)
        error = abs(result["sum_unrounded_local_estimates_PROXY"] - float(row["score_PROXY"]))
        checks.append({"candidate_id": row["candidate_id"],
                       "score_PROXY": result["sum_unrounded_local_estimates_PROXY"], "absolute_error": error})
    maximum_score_error = max(row["absolute_error"] for row in checks)
    if local_error > 1e-10 or maximum_score_error > 1e-9:
        raise ValueError("Numeric export differs from historical predictions.")
    report = {"status": "PASS", "constructs": len(rows), "unique_constructs": len({row["candidate_id"] for row in rows}),
              "all_fasta_and_tables_agree": True, "all_sequence_constraints_pass": True,
              "all_paired_mutation_counts_agree": True, "cached_local_prediction_max_absolute_error": local_error,
              "construct_score_max_absolute_error": maximum_score_error,
              "original_model_sha256": model.metadata["historical_joblib_sha256"],
              "no_new_generation_training_docking_or_MD": True, "checks": checks}
    if args.fresh_opt3:
        fresh = FeatureProvider(fresh=True)
        strands = load_fasta(DATA_ROOT / "four_constructs" / "Opt3.fasta")
        fresh_result = predict_tfna(strands, fresh, model)
        stored = next(row["score_PROXY"] for row in checks if row["candidate_id"] == "Opt3")
        error = abs(fresh_result["sum_unrounded_local_estimates_PROXY"] - stored)
        if error > 1e-5:
            raise ValueError("Fresh offline encoder differs from historical Opt3 score.")
        report["fresh_Opt3_encoder"] = {"status": "PASS", "edges": 6, "score_PROXY": fresh_result["sum_unrounded_local_estimates_PROXY"],
                                      "absolute_error": error}
    if args.output:
        write_json(args.output, report)
    summary = {key: value for key, value in report.items() if key != "checks"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
