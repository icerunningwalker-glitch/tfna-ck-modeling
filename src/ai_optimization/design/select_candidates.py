"""Replay historical top-score/diversity selection from a scored candidate pool."""
from pathlib import Path
import argparse
import csv
import json
from common import DATA_ROOT, MODEL_ROOT, RESULTS_ROOT, LOGS_ROOT, EDGES, write_table


def choose(pool, count):
    chosen = []
    joined = lambda row: "".join(row[e + "_20nt"] for e in EDGES)
    for row in pool:
        if all(sum(a != b for a, b in zip(joined(row), joined(other))) >= 2 for other in chosen):
            chosen.append(row)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        used = {row["candidate_id"] for row in chosen}
        chosen.extend(row for row in pool if row["candidate_id"] not in used)
    if len(chosen) < count:
        raise ValueError("Insufficient eligible candidates for the requested group.")
    return chosen[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["score_PROXY"] = float(row["score_PROXY"])
        row["paired_site_mutations"] = int(row["paired_site_mutations"])
    rows.sort(key=lambda row: (-row["score_PROXY"], row["paired_site_mutations"], row["candidate_id"]))
    selected = [next(row.copy() for row in rows if row["candidate_id"] == name)
                for name in ("WT", "Opt1", "Opt2", "Opt3")]
    for burden in (1, 2, 3, 4):
        candidates = [row for row in rows if row["kind"] == "candidate" and row["paired_site_mutations"] == burden]
        if burden == 1:
            chosen = choose(candidates, 20)
        else:
            chosen = choose([row for row in candidates if row["generation_pattern"] == "within-edge"], 10)
            chosen += choose([row for row in candidates if row["generation_pattern"] == "cross-edge"], 10)
        for index, row in enumerate(sorted(chosen, key=lambda r: (-r["score_PROXY"], r["candidate_id"])), 1):
            row = row.copy()
            row["source_pool_id"] = row["candidate_id"]
            row["candidate_id"] = f"P{burden}_{index:02}"
            selected.append(row)
    for index, row in enumerate(selected, 1):
        row.setdefault("source_pool_id", row["candidate_id"])
        row["selection_order"] = index
    write_table(args.output, selected)
    print(json.dumps({"status": "PASS", "selected": len(selected),
                      "selection": "20 per burden; burdens 2-4 have 10 within-edge and 10 cross-edge designs",
                      "scope": "Historical selection replay; no new docking or MD."}))


if __name__ == "__main__":
    main()
