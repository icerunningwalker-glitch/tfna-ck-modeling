"""Explicit training entry. Historical candidate scoring and newer comparison are separate versions."""
import argparse
import sys
from src.submission_runtime import dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["historical-mlp", "benchmark"])
    args = parser.parse_args(sys.argv[1:2])
    script = "src/ai_optimization/design/train_historical_mlp.py" if args.mode=="historical-mlp" else "src/ai_optimization/benchmark/train_all7.py"
    dispatch(script, sys.argv[2:])


if __name__ == "__main__":
    main()
