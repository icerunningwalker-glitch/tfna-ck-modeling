"""Verify the unified handoff, scientific results and separation of synthetic tests."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import tempfile
from src.submission_runtime import ROOT, FIELDS, dispatch, read_rows, screening_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hashes", action="store_true", help="Only for a deliberately changed working copy.")
    parser.add_argument("--models", action="store_true", help="Also reload 35 trusted fitted sklearn snapshots; requires full core environment.")
    parser.add_argument("--output", type=Path, help="Optional new verification record, never overwrites an existing file.")
    args = parser.parse_args()
    checks = []
    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
        if not passed:
            raise ValueError(name)
    if not args.skip_hashes:
        manifest = json.loads((ROOT/"manifest.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            p = ROOT/item["path"]
            check("hash "+item["path"], p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==item["sha256"])
    for path in ROOT.rglob("*.py"):
        if not {"__pycache__", "runs", "generated"}.intersection(path.relative_to(ROOT).parts):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=path.name)
    check("Python source parses", True)
    reference = read_rows(ROOT/"results/results.csv")
    check("84 unique standard candidate rows", len(reference)==84 and len({r["candidate_key"] for r in reference})==84)
    check("Submission columns match schema", all(list(row)==FIELDS for row in reference))
    check("No synthetic row in scientific results", all(row["evidence_type"]=="model_prediction" for row in reference))
    check("Four 63-nt DNA strands per AI candidate", all(len(row[key])==63 and not set(row[key])-set("ACGT") for row in reference for key in ("strand_1c", "strand_2c", "strand_3c", "strand_4c")))
    check("Paired versus nucleotide changes", all(int(row["total_nucleotide_substitutions"])==2*int(row["paired_site_substitutions"]) for row in reference))
    check("No invented sequence-only structures", all(row["structure_file"]=="" for row in reference))
    for row in reference:
        float(row["prediction_value"])
    with tempfile.TemporaryDirectory(prefix="submission_check_") as temporary:
        output = Path(temporary)/"predictions"
        dispatch("predict.py", ["--output-dir", output])
        actual = read_rows(output/"results.csv")
        # A submitter may fill track metadata; candidate content and predictions must remain reproducible.
        strip_track = lambda rows: [{k:v for k,v in row.items() if k!="track"} for row in rows]
        check("Root inference reproduces final candidate content", strip_track(actual)==strip_track(reference))
        try:
            screening_candidates(ROOT/"results/virtual_screening/synthetic_example", Path(temporary), "test")
        except ValueError:
            check("Synthetic screening rejected by final exporter", True)
        else:
            check("Synthetic screening rejected by final exporter", False)
    dispatch("src/ai_optimization/design/smoke_check.py", [])
    dispatch("src/ai_optimization/benchmark/verify_all7.py", [])
    dispatch("src/virtual_screening/self_test.py", [])
    if args.models:
        dispatch("src/ai_optimization/benchmark/evaluate_models.py", [])
    physical = json.loads((ROOT/"results/transport/summary.json").read_text(encoding="utf-8"))["checks"]
    check("Saved reference mass conservation", physical["mass_relative"]<1e-7)
    check("Saved reference nonnegative states", physical["min_state"]>-1e-7)
    check("Saved reference supply stops after removal", physical["input_after_removal"]<1e-8)
    report = {"status": "PASS", "checks": checks, "actual_inference": True, "model_snapshots_loaded": args.models,
              "synthetic_excluded": True, "new_training": False, "new_real_docking": False,
              "transport_check": "saved physical checks; full fresh simulation is recorded separately"}
    if args.output:
        if args.output.exists():
            raise FileExistsError("Choose a new verification output file.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key:value for key,value in report.items() if key!="checks"}))


if __name__ == "__main__":
    main()
