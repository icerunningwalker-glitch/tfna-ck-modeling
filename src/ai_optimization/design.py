"""Replay score/diversity optimization of the saved candidate pool; optionally generate a new pool."""
from pathlib import Path
import argparse
import subprocess
import sys
from predict import ROOT, export_results
from layout import DATA_ROOT, RESULTS_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT/"runs/design_demo")
    parser.add_argument("--generate", action="store_true", help="Generate candidates anew with the recorded fixed seed; optional and slower.")
    parser.add_argument("--track", default='赛道四：递送载体设计')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory.")
    args.output_dir.mkdir(parents=True)
    out = args.output_dir.resolve()
    pool = DATA_ROOT/"design/historical_candidate_pool.csv"
    if args.generate:
        subprocess.run([sys.executable, "-B", str(ROOT/"design/generate_candidates.py"), "--output-dir", str(out/"generated")], check=True)
        pool = out/"generated/candidate_pool.csv"
    subprocess.run([sys.executable, "-B", str(ROOT/"design/score_candidates.py"), "--input", str(pool), "--output", str(out/"pool_scores.csv")], check=True)
    subprocess.run([sys.executable, "-B", str(ROOT/"design/select_candidates.py"), "--input", str(out/"pool_scores.csv"), "--output", str(out/"selected_candidates.csv")], check=True)
    export_results(out/"selected_candidates.csv", out/"results.csv", args.track)
    print("PASS: generated results.csv with 84 constructs.")


if __name__ == "__main__":
    main()
