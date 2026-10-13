"""Validate or construct pair-preserving edits; coordinates are 1-based."""
from pathlib import Path
import argparse
import json
import re
from common import DATA_ROOT, MODEL_ROOT, RESULTS_ROOT, LOGS_ROOT, ROOT, WT, domains, apply_domains, load_fasta, mutation_record, validate, write_json


def apply_edits(edits):
    changed = {}
    reference = domains(WT)
    seen = set()
    for edit in edits:
        match = re.fullmatch(r"(E[1-4][1-4]):([0-9]+)([ACGT])>([ACGT])", edit)
        if match is None:
            raise ValueError("Use an edit such as E12:11T>G.")
        edge, position, old, new = match.groups()
        position = int(position)
        if edge not in reference or not 1 <= position <= 20:
            raise ValueError("Unknown edge or position outside 1..20: " + edit)
        if (edge, position) in seen:
            raise ValueError("Duplicate edited position: " + edit)
        seen.add((edge, position))
        if reference[edge][position - 1] != old or old == new:
            raise ValueError("Reference base mismatch or unchanged base: " + edit)
        changed.setdefault(edge, list(reference[edge]))[position - 1] = new
    return apply_domains({edge: "".join(sequence) for edge, sequence in changed.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate")
    check.add_argument("--fasta", type=Path, required=True)
    check.add_argument("--output", type=Path)
    build = commands.add_parser("edit")
    build.add_argument("--edits", nargs="+", required=True, help="Each edit uses the first-oriented edge sequence.")
    build.add_argument("--id", default="candidate")
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "edit":
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.id):
            parser.error("Use a simple alphanumeric candidate ID.")
        strands = apply_edits(args.edits)
    else:
        strands = load_fasta(args.fasta)
    errors = validate(strands)
    mutations = mutation_record(strands)
    nucleotide_changes = sum(sum(a != b for a, b in zip(WT[name], sequence))
                             for name, sequence in strands.items())
    result = {"status": "FAIL" if errors else "PASS", "errors": errors,
              "paired_site_mutations": len(mutations), "actual_nucleotide_changes": nucleotide_changes,
              "mutations": mutations, "linker_positions_1based_preserved": [1, 22, 43]}
    if args.command == "edit":
        if errors:
            parser.error("Constraints failed: " + ", ".join(errors))
        if args.output.exists():
            raise FileExistsError("Preserve existing output: " + str(args.output))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(f">{args.id}_{name}\n{sequence}\n" for name, sequence in strands.items()), encoding="ascii")
    elif args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
