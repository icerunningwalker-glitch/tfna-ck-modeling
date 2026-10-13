"""Recompute candidate scores and export an anonymous submission table."""
from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys

from layout import SOURCE_ROOT as ROOT, DATA_ROOT, RESULTS_ROOT, LOGS_ROOT, PROJECT_ROOT
VERSION = "portable_20261004"


def export_results(scored_path, output, track):
    with Path(scored_path).open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    records.sort(key=lambda row: (-float(row["score_PROXY"]), row["candidate_id"]))
    fields = ["rank", "candidate_id", "track", "strand_1c", "strand_2c", "strand_3c", "strand_4c",
              "paired_site_substitutions", "total_nucleotide_substitutions", "predicted_score",
              "score_unit", "delta_vs_reference", "model_version", "run_version", "structure_file", "notes"]
    with Path(output).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(records, 1):
            paired = int(row["paired_site_mutations"])
            writer.writerow(dict(rank=rank, candidate_id=row["candidate_id"], track=track,
                                 **{name: row[name] for name in fields if name.startswith("strand_")},
                                 paired_site_substitutions=paired, total_nucleotide_substitutions=2*paired,
                                 predicted_score=f'{float(row["score_PROXY"]):.12g}', score_unit="dimensionless proxy",
                                 delta_vs_reference=f'{float(row["delta_vs_WT"]):.12g}',
                                 model_version=row["model_version"], run_version=VERSION, structure_file="",
                                 notes="Sum of six local predictions; not measured loading or docking energy. No 3D structure used in this sequence-only export."))
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RESULTS_ROOT/"design/historical_selected_84.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT/"runs/replay")
    parser.add_argument("--track", default='赛道四：递送载体设计', help="Replace with the official competition track.")
    parser.add_argument("--cached-only", action="store_true", help="Require existing cached sequence features.")
    parser.add_argument("--fresh", action="store_true", help="Encode sequences with offline HyenaDNA rather than cached features.")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing results are preserved.")
    args.output_dir.mkdir(parents=True)
    scored = args.output_dir.resolve()/"candidate_scores.csv"
    command = [sys.executable, str(ROOT/"design/score_candidates.py"), "--input", str(args.input.resolve()), "--output", str(scored)]
    if args.fresh and args.cached_only:
        parser.error("--fresh and --cached-only are mutually exclusive.")
    if args.fresh:
        command.append("--fresh")
    if args.cached_only:
        command.append("--cached-only")
    command.insert(1, "-B")
    subprocess.run(command, check=True)
    count = export_results(scored, args.output_dir/"results.csv", args.track)
    report = {"status": "PASS", "candidates": count, "model_version": "historical_frozen_MLP_16_hidden_20261001",
              "run_version": VERSION, "new_training": False, "new_docking": False, "new_MD": False,
              "official_track_confirmed": args.track != "UNSPECIFIED"}
    (args.output_dir/"run.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
