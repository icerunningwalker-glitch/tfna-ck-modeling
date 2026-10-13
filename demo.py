"""Runnable cross-module demonstration; synthetic docking stays outside scientific results.csv."""
from pathlib import Path
import argparse
import json
from src.submission_runtime import ROOT, dispatch, new_output, write_run_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/demo"))
    parser.add_argument("--transport", choices=["run", "saved"], default="run")
    args = parser.parse_args()
    out = new_output(args.output_dir)
    entries = [dispatch("design.py", ["--output-dir", out/"optimization"]),
               dispatch("src/virtual_screening/demo.py", ["--output-dir", out/"synthetic_screening"])]
    if args.transport == "run":
        entries.append(dispatch("transport.py", ["--output-dir", out/"transport"]))
    else:
        saved = json.loads((ROOT/"results/transport/summary.json").read_text(encoding="utf-8"))
        (out/"saved_transport_summary.json").write_text(json.dumps({"mode": "saved_result_review_not_new_simulation", "summary": saved}, indent=2)+"\n", encoding="utf-8")
    # The standard scientific table contains only the actual AI predictions, never mock docking rows.
    (out/"results.csv").write_bytes((out/"optimization/results.csv").read_bytes())
    write_run_record(out, entries, 84, "cross_module_software_demo")
    print("PASS: demo complete; synthetic screening outputs are separately labelled.")


if __name__ == "__main__":
    main()
