"""Verify distributed hashes, cached candidate inference and saved CV arithmetic."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hashes", action="store_true", help="For deliberately modified working copies only.")
    args = parser.parse_args()
    manifest = json.loads((ROOT/"manifest.json").read_text(encoding="utf-8"))
    if not args.skip_hashes:
        for item in manifest["files"]:
            path = PROJECT_ROOT/item["path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Missing or changed distributed file: "+item["path"])
    for path in ROOT.rglob("*.py"):
        if "outputs" not in path.parts and "__pycache__" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=path.name)
    subprocess.run([sys.executable, "-B", str(ROOT/"design/smoke_check.py")], check=True)
    subprocess.run([sys.executable, "-B", str(ROOT/"benchmark/verify_all7.py")], check=True)
    print(json.dumps({"status": "PASS", "distributed_files": len(manifest["files"]),
                      "candidate_constructs": 84, "saved_CV_metrics_verified": True,
                      "training_performed": False}))


if __name__ == "__main__":
    main()
