"""Portable, pair-preserving tFNA topology and sequence validation."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import re

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data" / "ai_optimization" / "design"
MODEL_ROOT = ROOT / "models" / "ai_optimization" / "design"
RESULTS_ROOT = ROOT / "results" / "ai_optimization" / "design"
LOGS_ROOT = ROOT / "logs" / "ai_optimization" / "design"
CONFIG = json.loads((DATA_ROOT / "reference.json").read_text(encoding="utf-8"))
WT = CONFIG["strands"]
EDGES = tuple(CONFIG["edges"])


def reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def normalize(sequence):
    sequence = sequence.strip().upper()
    if len(sequence) != 20 or any(base not in "ACGT" for base in sequence):
        raise ValueError("Supply exactly 20 A/C/G/T bases.")
    return sequence


def load_fasta(path):
    strands = {}
    current = None
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            match = re.search(r"(?:^|_)([1-4]c)$", line[1:].split()[0], re.I)
            if not match:
                raise ValueError("FASTA headers must end in 1c, 2c, 3c or 4c.")
            current = match.group(1).lower()
            if current in strands:
                raise ValueError("Duplicate strand: " + current)
            strands[current] = ""
        elif current is None:
            raise ValueError("Sequence precedes FASTA header.")
        else:
            strands[current] += line.upper()
    if set(strands) != set(WT):
        raise ValueError("Supply all four strands.")
    for name, sequence in strands.items():
        if len(sequence) != 63 or set(sequence) - set("ACGT"):
            raise ValueError(name + " must contain exactly 63 A/C/G/T bases.")
    return strands


def domains(strands):
    return {edge: strands[v["strand_a"]][v["start_a"] - 1:v["start_a"] + 19]
            for edge, v in CONFIG["edges"].items()}


def longest_common_substring(first, second):
    previous = [0] * (len(second) + 1)
    best = 0
    for char_a in first:
        current = [0]
        for index, char_b in enumerate(second, 1):
            value = previous[index - 1] + 1 if char_a == char_b else 0
            current.append(value)
            best = max(best, value)
        previous = current
    return best


def maximum_homopolymer(sequence):
    best = run = 0
    previous = None
    for base in sequence:
        run = run + 1 if base == previous else 1
        previous = base
        best = max(best, run)
    return best


def validate(strands):
    errors = []
    if set(strands) != set(WT):
        return ["Supply exactly 1c, 2c, 3c, 4c."]
    if any(len(s) != 63 or set(s) - set("ACGT") for s in strands.values()):
        return ["Each strand must contain exactly 63 uppercase A/C/G/T bases."]
    constraints = CONFIG["sequence_constraints"]
    ds = domains(strands)
    for name, sequence in strands.items():
        if any(sequence[i - 1] != WT[name][i - 1] for i in (1, 22, 43)):
            errors.append("linker_changed_" + name)
        if maximum_homopolymer(sequence) > constraints["max_homopolymer"]:
            errors.append("homopolymer_" + name)
    for edge, v in CONFIG["edges"].items():
        partner = strands[v["strand_b"]][v["start_b"] - 1:v["start_b"] + 19]
        if partner != reverse_complement(ds[edge]):
            errors.append("pairing_" + edge)
        gc = sum(base in "GC" for base in ds[edge]) / 20
        if not constraints["gc_fraction"][0] <= gc <= constraints["gc_fraction"][1]:
            errors.append("GC_" + edge)
    for name in strands:
        strand_domains = [ds[e] if v["strand_a"] == name else reverse_complement(ds[e])
                          for e, v in CONFIG["edges"].items()
                          if name in (v["strand_a"], v["strand_b"])]
        for a, b in itertools.combinations(strand_domains, 2):
            if sum(x == y for x, y in zip(a, b)) / 20 > constraints["max_within_strand_domain_identity"]:
                errors.append("domain_identity_" + name)
    for a, b in itertools.combinations(ds.values(), 2):
        if max(longest_common_substring(a, b), longest_common_substring(a, reverse_complement(b))) > constraints["max_unintended_complement_run"]:
            errors.append("unintended_complement")
    return sorted(set(errors))


def apply_domains(changes):
    strands = {name: list(sequence) for name, sequence in WT.items()}
    for edge, sequence in changes.items():
        v = CONFIG["edges"][edge]
        for strand, start, value in ((v["strand_a"], v["start_a"], sequence),
                                     (v["strand_b"], v["start_b"], reverse_complement(sequence))):
            strands[strand][start - 1:start + 19] = value
    return {name: "".join(sequence) for name, sequence in strands.items()}


def mutation_record(strands):
    reference = domains(WT)
    current = domains(strands)
    return [f"{edge}:{i + 1}{a}>{b}" for edge in EDGES
            for i, (a, b) in enumerate(zip(reference[edge], current[edge])) if a != b]


def sequence_key(strands):
    return "".join(strands[name] for name in WT)


def write_json(path, result):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve existing output: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_table(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve existing output: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
