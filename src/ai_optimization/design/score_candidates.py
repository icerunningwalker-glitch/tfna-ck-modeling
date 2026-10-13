"""Score an existing candidate table with the historical frozen MLP."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from common import DATA_ROOT, MODEL_ROOT, RESULTS_ROOT, LOGS_ROOT, ROOT, WT, EDGES, domains, mutation_record, validate, write_table
from predict import FrozenMLP, FeatureProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DATA_ROOT / "local_feature_cache.npz")
    encoding = parser.add_mutually_exclusive_group()
    encoding.add_argument("--fresh", action="store_true", help="Ignore the cache and recompute offline embeddings.")
    encoding.add_argument("--cached-only", action="store_true", help="Require all local sequences in the supplied cache.")
    parser.add_argument("--checkpoint", type=Path, help="Optional local pinned encoder checkpoint.")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing output: " + str(args.output))
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty candidate table.")
    ids = [row["candidate_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique.")
    sequence_rows = []
    for row in rows:
        strands = {name: row["strand_" + name] for name in WT}
        errors = validate(strands)
        if errors:
            raise ValueError(row["candidate_id"] + ": " + ", ".join(errors))
        mutations = mutation_record(strands)
        if "paired_site_mutations" in row and int(row["paired_site_mutations"]) != len(mutations):
            raise ValueError(row["candidate_id"] + ": paired-site count differs from the actual strands.")
        row["paired_site_mutations"] = len(mutations)
        ds = domains(strands)
        for edge in EDGES:
            row[edge + "_20nt"] = ds[edge]
        sequence_rows.extend(ds[e] for e in EDGES)
    model = FrozenMLP(args.model_dir)
    provider = FeatureProvider(args.cache, args.fresh, checkpoint=args.checkpoint,
                               expected_revision=model.revision, cached_only=args.cached_only)
    unique = sorted(set(sequence_rows))
    values = dict(zip(unique, model.predict(provider.featurize(unique)).tolist()))
    reference = sum(values.get(s, 0) for s in domains(WT).values())
    # Ensure the parent baseline exists even if a custom table omits WT edges.
    missing_reference = [s for s in domains(WT).values() if s not in values]
    if missing_reference:
        values.update(zip(missing_reference, model.predict(provider.featurize(missing_reference)).tolist()))
        reference = sum(values[s] for s in domains(WT).values())
    for row in rows:
        for edge in EDGES:
            row[edge + "_prediction"] = values[row[edge + "_20nt"]]
        row["score_PROXY"] = sum(row[edge + "_prediction"] for edge in EDGES)
        row["delta_vs_WT"] = row["score_PROXY"] - reference
        row["model_version"] = model.version
    write_table(args.output, rows)
    print(json.dumps({"status": "PASS", "constructs": len(rows), "unique_region_sequences": len(unique),
                      "score_unit": "dimensionless screening proxy", "new_training": False}))


if __name__ == "__main__":
    main()
