"""Shared command dispatch, final-result schema and anonymous run records."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = "tfna_ck_publication_20261013"
FIELDS = ["candidate_key", "candidate_id", "track", "module", "rank", "candidate_type",
          "candidate_sequence", "strand_1c", "strand_2c", "strand_3c", "strand_4c", "smiles",
          "structure_file", "prediction_metric", "prediction_value", "prediction_unit", "score_direction",
          "paired_site_substitutions", "total_nucleotide_substitutions", "hit_box_fraction",
          "model_version", "run_version", "evidence_type", "notes"]


def default_track():
    return json.loads((ROOT/"config/submission.json").read_text(encoding="utf-8"))["track"]


def new_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("Choose a new output directory; existing results are preserved.")
    path.mkdir(parents=True)
    return path


def dispatch(relative_script, arguments):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    started = time.monotonic()
    completed = subprocess.run([sys.executable, "-B", str(ROOT/relative_script), *map(str, arguments)],
                               cwd=ROOT, env=env, text=True, encoding="utf-8", capture_output=True)
    # Preserve readable console output without saving local paths from tracebacks to distributed logs.
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError("Module command failed: "+relative_script)
    return {"entry": relative_script, "exit_code": completed.returncode,
            "elapsed_seconds": round(time.monotonic()-started, 6)}


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ai_candidates(path, track):
    rows = []
    for source in read_rows(path):
        strands = {key: source[key] for key in ("strand_1c", "strand_2c", "strand_3c", "strand_4c")}
        if any(len(sequence) != 63 or set(sequence)-set("ACGT") for sequence in strands.values()):
            raise ValueError("Invalid four-strand candidate sequence.")
        rows.append(dict(candidate_key="ai:"+source["candidate_id"], candidate_id=source["candidate_id"],
                         track=track, module="ai_optimization", rank=source["rank"], candidate_type="four_strand_tFNA",
                         candidate_sequence=";".join(strands.values()), **strands, smiles="", structure_file="",
                         prediction_metric="six_edge_local_count_sum_PROXY", prediction_value=source["predicted_score"],
                         prediction_unit="dimensionless", score_direction="higher", paired_site_substitutions=source["paired_site_substitutions"],
                         total_nucleotide_substitutions=source["total_nucleotide_substitutions"], hit_box_fraction="",
                         model_version=source["model_version"], run_version=VERSION, evidence_type="model_prediction",
                         notes="Sequence-only local proxy; no measured loading, docking energy or per-candidate transport performance."))
    return rows


def screening_candidates(run_directory, output, track):
    run = Path(run_directory).resolve()
    for directory in (run, run.parent):
        if (directory/"SYNTHETIC_EXAMPLE.txt").exists():
            raise ValueError("Synthetic screening is excluded from the scientific submission table.")
    summary = json.loads((run/"ranking_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") not in {"completed", "completed_with_failures"}:
        raise ValueError("Screening run is not terminal.")
    source_path = run/"results/results.csv"
    rows = []
    for source in read_rows(source_path):
        text = " ".join(str(source.get(k, "")) for k in ("track", "notes", "evidence_type")).lower()
        if any(token in text for token in ("synthetic", "mock", "toy")) or source.get("scientific_candidate", "true").lower()=="false":
            raise ValueError("Synthetic screening rows cannot enter results.csv.")
        structure = source.get("structure_file", "")
        if structure:
            relative = Path(structure)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Structure references must stay inside the source result folder.")
            source_structure = (run/"results"/relative).resolve()
            if not source_structure.is_relative_to((run/"results").resolve()):
                raise ValueError("Structure references must not escape through a symlink.")
            if not source_structure.is_file():
                raise FileNotFoundError("Missing candidate structure.")
            destination = output/"structures"/relative.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source_structure, destination)
            structure = "structures/"+relative.name
        rows.append(dict(candidate_key="screen:"+source["candidate_id"], candidate_id=source["candidate_id"], track=track,
                         module="virtual_screening", rank=source["rank"], candidate_type="ligand", candidate_sequence="",
                         **{key: "" for key in ("strand_1c", "strand_2c", "strand_3c", "strand_4c")},
                         smiles=source.get("smiles", ""), structure_file=structure, prediction_metric="best_vina_score",
                         prediction_value=source["best_vina_score_kcal_mol"], prediction_unit="kcal/mol", score_direction="lower",
                         paired_site_substitutions="", total_nucleotide_substitutions="",
                         hit_box_fraction=source.get("predicted_hit_box_fraction_8", ""), model_version=source["model_version"],
                         run_version=VERSION, evidence_type="docking_prediction", notes=source.get("notes", "")))
    return rows


def write_submission(output, rows):
    keys = [row["candidate_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Candidate keys must be unique.")
    with (output/"results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output/"result_schema.json").write_text((ROOT/"config/result_schema.json").read_text(encoding="utf-8"), encoding="utf-8")


def write_run_record(output, entries, candidate_count=None, scope="inference"):
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "torch", "transformers", "tabpfn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    record = {"status": "PASS", "run_version": VERSION, "scope": scope, "entries": entries,
              "candidate_count": candidate_count, "python": platform.python_version(), "operating_system": platform.system(),
              "architecture": platform.machine(), "packages": versions, "hostname_recorded": False,
              "account_recorded": False, "local_paths_recorded": False}
    (output/"run.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    return record
