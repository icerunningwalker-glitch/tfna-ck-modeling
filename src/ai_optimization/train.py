"""Explicit entry to historical final-MLP training or the separate current benchmark."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["historical-mlp", "benchmark"])
    args, extra = parser.parse_known_args()
    script = ROOT/("design/train_historical_mlp.py" if args.mode == "historical-mlp" else "benchmark/train_all7.py")
    subprocess.run([sys.executable, "-B", str(script), *extra], check=True)


if __name__ == "__main__":
    main()
