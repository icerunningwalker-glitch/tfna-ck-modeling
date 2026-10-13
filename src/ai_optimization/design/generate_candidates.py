"""Optional historical pair-preserving generation, with explicit output paths."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
from common import DATA_ROOT, MODEL_ROOT, RESULTS_ROOT, LOGS_ROOT, ROOT, WT, EDGES, domains, apply_domains, mutation_record, sequence_key, validate, write_table, write_json, load_fasta


def make_row(candidate_id, changes, pattern, history):
    strands = apply_domains(changes)
    ds = domains(strands)
    errors = validate(strands)
    return {"candidate_id": candidate_id, "design_class": "paired-site mutation",
            "paired_site_mutations": len(mutation_record(strands)),
            "changed_edges": ";".join(e for e in EDGES if ds[e] != domains(WT)[e]),
            "mutations": ";".join(mutation_record(strands)),
            "constraint_status": "FAIL" if errors else "PASS",
            "constraint_errors": ";".join(errors),
            **{"strand_" + name: sequence for name, sequence in strands.items()},
            **{e + "_20nt": ds[e] for e in EDGES},
            "kind": "candidate", "generation_pattern": pattern,
            "previous_candidate_pool_overlap": hashlib.sha256(sequence_key(strands).encode()).hexdigest() in history}


def row_hash(row):
    return hashlib.sha256("".join(row["strand_" + name] for name in WT).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--per-pattern", type=int, default=250,
                        help="Accepted within-edge and cross-edge designs for each burden 2,3,4.")
    parser.add_argument("--max-attempts", type=int, default=25000)
    parser.add_argument("--prior-pool-hashes", type=Path, default=DATA_ROOT / "previous_pool_sequence_hashes.txt")
    args = parser.parse_args()
    if args.per_pattern < 1 or args.max_attempts < args.per_pattern:
        parser.error("Require 1 <= per-pattern <= max-attempts.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new empty output directory.")
    history = set(args.prior_pool_hashes.read_text().splitlines())
    rows = []
    for name in ("WT", "Opt1", "Opt2", "Opt3"):
        strands = load_fasta(DATA_ROOT / "four_constructs" / (name + ".fasta"))
        row = make_row(name, domains(strands), "existing", history)
        if row["constraint_status"] != "PASS":
            raise ValueError("An existing construct failed sequence validation.")
        row.update(kind="existing", design_class="existing comparator", previous_candidate_pool_overlap=True)
        rows.append(row)
    seen = {row_hash(row) for row in rows}
    reference = domains(WT)
    rejected = []
    singles = 0
    for edge in EDGES:
        for index, base in enumerate(reference[edge]):
            for replacement in "ACGT":
                if replacement == base:
                    continue
                row = make_row(f"M1_{edge}_{index + 1:02}_{replacement}",
                               {edge: reference[edge][:index] + replacement + reference[edge][index + 1:]},
                               "within-edge", history)
                key = row_hash(row)
                if key in seen:
                    continue
                if row["constraint_status"] != "PASS":
                    rejected.append(row)
                    continue
                seen.add(key)
                rows.append(row)
                singles += 1
    rng = random.Random(args.seed)
    attempts = {}
    accepted = {}
    for burden in (2, 3, 4):
        for pattern in ("within-edge", "cross-edge"):
            obtained = tried = 0
            while obtained < args.per_pattern and tried < args.max_attempts:
                tried += 1
                if pattern == "within-edge":
                    edge = rng.choice(EDGES)
                    positions = [(edge, i) for i in rng.sample(range(20), burden)]
                else:
                    positions = [(EDGES[j // 20], j % 20) for j in rng.sample(range(120), burden)]
                    if len({edge for edge, index in positions}) < 2:
                        continue
                changed = {edge: list(reference[edge]) for edge, index in positions}
                for edge, index in positions:
                    changed[edge][index] = rng.choice([base for base in "ACGT" if base != reference[edge][index]])
                row = make_row(f'M{burden}_{"W" if pattern == "within-edge" else "X"}_{obtained + 1:03}',
                               {edge: "".join(sequence) for edge, sequence in changed.items()}, pattern, history)
                key = row_hash(row)
                if key in seen or key in history:
                    continue
                if row["constraint_status"] != "PASS":
                    row["candidate_id"] = f"REJECT_M{burden}_{pattern}_{tried:05}"
                    rejected.append(row)
                    continue
                seen.add(key)
                rows.append(row)
                obtained += 1
            if obtained != args.per_pattern:
                raise RuntimeError(f"Sampling exhausted for {burden}/{pattern}: {obtained} accepted.")
            attempts[f"{burden}_{pattern}"] = tried
            accepted[f"{burden}_{pattern}"] = obtained
    write_table(args.output_dir / "candidate_pool.csv", rows)
    if rejected:
        write_table(args.output_dir / "rejected_candidates.csv", rejected)
    report = {"seed": args.seed, "reference": "WT", "mutation_unit": "paired site",
              "actual_nucleotide_changes_per_paired_site": 2,
              "single_site_all_360_enumerated": True, "accepted_nonreference_single_sites": singles,
              "accepted_multi_site_patterns": accepted, "attempts": attempts,
              "total_accepted": len(rows), "rejected_records": len(rejected),
              "multi_site_enumeration_exhaustive": False, "new_training": False,
              "new_docking": False, "new_MD": False}
    write_json(args.output_dir / "generation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
