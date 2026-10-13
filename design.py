"""Design/replay, optimize and export the standard tFNA candidate list."""
from pathlib import Path
import argparse
from src.submission_runtime import default_track, dispatch, new_output, ai_candidates, write_submission, write_run_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/design"))
    parser.add_argument("--generate", action="store_true", help="Explicitly create a new sampled pool, rather than replay the saved pool.")
    parser.add_argument("--track", default=default_track())
    args = parser.parse_args()
    out = new_output(args.output_dir)
    command = ["--output-dir", out/"ai", "--track", args.track]
    if args.generate:
        command.append("--generate")
    entry = dispatch("src/ai_optimization/design.py", command)
    rows = ai_candidates(out/"ai/results.csv", args.track)
    write_submission(out, rows)
    write_run_record(out, [entry], len(rows), "new_design" if args.generate else "saved_design_optimization_replay")
    print("PASS: results.csv generated; candidate count =", len(rows))


if __name__ == "__main__":
    main()
