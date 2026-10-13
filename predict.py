"""One entry generates the standard final candidate table from real module outputs."""
from pathlib import Path
import argparse
from src.submission_runtime import default_track, dispatch, new_output, ai_candidates, screening_candidates, write_submission, write_run_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["ai", "screening", "both"], default="ai")
    parser.add_argument("--input", type=Path, help="Optional AI candidate CSV.")
    parser.add_argument("--screening-run", type=Path, help="Completed real screening run; synthetic demos are rejected.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/predict"))
    parser.add_argument("--track", default=default_track())
    parser.add_argument("--cached-only", action="store_true", help="Require saved features for all AI sequences.")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.source in ("screening", "both") and args.screening_run is None:
        parser.error("--screening-run is required for real docking candidates.")
    if args.fresh and args.cached_only:
        parser.error("--fresh and --cached-only are mutually exclusive.")
    output = new_output(args.output_dir)
    entries = []
    rows = []
    if args.source in ("ai", "both"):
        command = ["--output-dir", output/"ai", "--track", args.track]
        if args.input:
            command += ["--input", args.input.resolve()]
        if args.fresh:
            command.append("--fresh")
        if args.cached_only:
            command.append("--cached-only")
        entries.append(dispatch("src/ai_optimization/predict.py", command))
        rows += ai_candidates(output/"ai/results.csv", args.track)
    if args.source in ("screening", "both"):
        rows += screening_candidates(args.screening_run, output, args.track)
    write_submission(output, rows)
    write_run_record(output, entries, len(rows))
    print("PASS: results.csv generated; candidate count =", len(rows))


if __name__ == "__main__":
    main()
