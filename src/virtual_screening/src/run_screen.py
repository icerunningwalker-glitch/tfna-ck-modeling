#!/usr/bin/env python3
"""Resume a fixed WT Vina acquisition screen using only the standard library.

Each ligand has eight search boxes. Box counts are screening proxies, not counts
of independent binding sites, simultaneous occupancy, or experimental loading.
``runs.csv`` is an append-only attempt journal; ``current_runs.csv`` is the latest
one-row-per-job snapshot. A preflight limit does not change the saved protocol.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
BOX_COUNT = 8
NUM_MODES = 20
ENERGY_RANGE = 4
PROGRESS_INTERVAL = 15.0
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
LOG_SCORE = re.compile(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", re.MULTILINE)
POSE_SCORE = re.compile(r"^REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", re.MULTILINE)
COORDINATES = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
JOB_FIELDS = ("run_id", "candidate_id", "ligand_id", "name", "ligand", "heavy_atoms", "site_id", "receptor", *COORDINATES)
RUN_FIELDS = ("run_id", "candidate_id", "ligand_id", "name", "site_id", "status", "attempt", "best_score", "exit_code", "elapsed_seconds", "started_at", "finished_at", "pose_file", "log_file", "stdout_file", "error", "protocol_sha256")
NOTICE = "Strong-box counts refer to eight possibly overlapping search boxes; they are not independent binding-site counts or loading stoichiometry."


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_with_retry(temporary, path, attempts=8, delay_seconds=0.1):
    """Keep atomic replacement despite brief Windows reader/sharing locks."""
    if attempts < 1 or delay_seconds < 0:
        raise ValueError("Replacement attempts must be positive and delay nonnegative")
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except OSError as exc:
            retryable = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in {5, 32, 33}
            if not retryable or attempt == attempts - 1:
                raise
            # Readers can briefly omit FILE_SHARE_DELETE. Do not remove the
            # previous valid file or silently ignore a persistent write error.
            time.sleep(min(delay_seconds * (2 ** attempt), 1.0))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path, fields, rows):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: required CSV fields: {', '.join(required)}")
        return list(reader)


def resolved_file(value, relative_to):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(relative_to) / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty input: {path}")
    return path


def check_identifier(value, label):
    if not IDENTIFIER.fullmatch(value or "") or value in {".", ".."} or value.endswith("."):
        raise ValueError(f"Unsafe {label}: {value!r}")
    devices = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if value.split(".", 1)[0].upper() in devices:
        raise ValueError(f"Reserved Windows {label}: {value!r}")


def pdbqt_atom_heavy(line):
    """Check finite fixed-column coordinates and classify a PDBQT atom type."""
    try:
        coordinates = [float(line[start:start + 8]) for start in (30, 38, 46)]
    except ValueError as exc:
        raise ValueError("PDBQT atom has malformed coordinate columns") from exc
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("PDBQT atom has nonfinite coordinates")
    atom_type = line[77:].strip() if len(line) > 77 else ""
    if not atom_type or len(atom_type.split()) != 1:
        raise ValueError("PDBQT atom has missing/malformed AutoDock atom type")
    return atom_type.upper() not in {"H", "HD", "HS"}


def validate_ligand_records(lines, expected_heavy_atoms=None):
    roots, ends, heavy_atoms = 0, 0, 0
    branch_stack = []
    for line in lines:
        record = line.split(maxsplit=1)[0] if line.strip() else ""
        if record in {"ATOM", "HETATM"}:
            if roots != 1 or (ends and not branch_stack):
                raise ValueError("PDBQT atom outside a ROOT or BRANCH section")
            heavy_atoms += int(pdbqt_atom_heavy(line))
        elif record == "ROOT":
            if roots or ends:
                raise ValueError("Duplicate or misplaced PDBQT ROOT")
            roots += 1
        elif record == "ENDROOT":
            if roots != 1 or ends:
                raise ValueError("PDBQT ENDROOT without an open ROOT")
            ends += 1
        elif record == "BRANCH":
            if roots != 1 or ends != 1 or len(line.split()) != 3:
                raise ValueError("Malformed PDBQT BRANCH")
            branch_stack.append(tuple(line.split()[1:]))
        elif record == "ENDBRANCH":
            if not branch_stack or branch_stack.pop() != tuple(line.split()[1:]):
                raise ValueError("Unbalanced PDBQT BRANCH/ENDBRANCH")
    if roots != 1 or ends != 1 or branch_stack:
        raise ValueError("PDBQT requires one ROOT/ENDROOT pair and balanced branches")
    if heavy_atoms < 1:
        raise ValueError("PDBQT has no heavy atoms")
    if expected_heavy_atoms is not None and heavy_atoms != expected_heavy_atoms:
        raise ValueError(f"Pose heavy-atom count {heavy_atoms} differs from ligand count {expected_heavy_atoms}")
    return heavy_atoms


def ligand_heavy_atoms(path):
    return validate_ligand_records(Path(path).read_text(encoding="utf-8", errors="replace").splitlines())


def validate_pose_file(path, expected_heavy_atoms, log_score):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    models, current = [], None
    for line in lines:
        if line.startswith("MODEL"):
            if current is not None:
                raise ValueError("Nested pose MODEL records")
            current = []
        elif line.startswith("ENDMDL"):
            if current is None:
                raise ValueError("Pose ENDMDL without MODEL")
            models.append(current)
            current = None
        elif current is not None:
            current.append(line)
        elif line.startswith(("ATOM", "HETATM", "ROOT", "ENDROOT", "BRANCH", "ENDBRANCH")):
            raise ValueError("Pose ligand records outside MODEL/ENDMDL")
    if current is not None or not models:
        raise ValueError("Pose file has incomplete or absent MODEL/ENDMDL records")
    scores = []
    for model in models:
        validate_ligand_records(model, expected_heavy_atoms)
        matches = [POSE_SCORE.match(line) for line in model if POSE_SCORE.match(line)]
        if len(matches) != 1:
            raise ValueError("Each pose MODEL requires one REMARK VINA RESULT score")
        score = float(matches[0].group(1))
        if not math.isfinite(score):
            raise ValueError("Pose score is not finite")
        scores.append(score)
    # Vina 1.1.2 prints table scores to one decimal and pose remarks to three.
    if not math.isfinite(log_score) or abs(scores[0] - log_score) > 0.051:
        raise ValueError("First pose REMARK VINA score does not match Vina log score")
    return {"pose_count": len(models), "heavy_atoms_per_pose": expected_heavy_atoms, "first_pose_score": scores[0]}


def build_protocol(args):
    if args.workers < 1 or args.cpu_per_job < 1 or args.exhaustiveness < 1:
        raise ValueError("workers, cpu-per-job, exhaustiveness and expected-ligands must be positive")
    if args.limit_jobs is not None and args.limit_jobs < 1:
        raise ValueError("limit-jobs must be positive")
    ligand_path = resolved_file(args.ligand_manifest, Path.cwd())
    site_path = resolved_file(args.site_jobs, Path.cwd())
    vina_path = resolved_file(args.vina, Path.cwd())
    ligands = read_csv(ligand_path, ("identifier", "name", "pdbqt", "status"))
    if args.expected_ligands is None:
        args.expected_ligands = len(ligands)
    if args.expected_ligands < 1:
        raise ValueError("expected-ligands must be positive")
    if len(ligands) != args.expected_ligands:
        raise ValueError(f"Expected {args.expected_ligands} ligand rows; found {len(ligands)}")
    identifiers, library = [], []
    for row in ligands:
        identifier = row["identifier"].strip()
        check_identifier(identifier, "ligand identifier")
        if row["status"].strip() not in {"prepared", "reused"}:
            raise ValueError(f"Ligand {identifier} is not prepared/reused: {row['status']!r}")
        ligand = resolved_file(row["pdbqt"], ligand_path.parent)
        identifiers.append(identifier.casefold())
        library.append({"identifier": identifier, "name": row["name"], "pdbqt": str(ligand), "sha256": file_hash(ligand), "heavy_atoms": ligand_heavy_atoms(ligand), "status": row["status"].strip()})
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate ligand identifiers (case-insensitive)")
    sites = read_csv(site_path, ("candidate_id", "site_id", "receptor", *COORDINATES))
    if len(sites) != BOX_COUNT or any(row["candidate_id"].strip() != "WT" for row in sites):
        raise ValueError("Site manifest must contain exactly eight WT search boxes")
    site_ids, receptors, boxes = [], {}, []
    for row in sites:
        site_id = row["site_id"].strip()
        check_identifier(site_id, "site identifier")
        site_ids.append(site_id.casefold())
        receptor = resolved_file(row["receptor"], site_path.parent)
        if str(receptor) not in receptors:
            receptors[str(receptor)] = file_hash(receptor)
        box = {"candidate_id": "WT", "site_id": site_id, "receptor": str(receptor)}
        for key in COORDINATES:
            number = float(row[key])
            if not math.isfinite(number) or (key.startswith("size_") and number <= 0):
                raise ValueError(f"Invalid {key} for box {site_id}: {row[key]!r}")
            box[key] = format(number, ".15g")
        boxes.append(box)
    if len(set(site_ids)) != BOX_COUNT:
        raise ValueError("Duplicate site identifiers (case-insensitive)")
    if len(receptors) != 1:
        raise ValueError("All eight search boxes must use the same receptor file")
    identity = {
        "schema_version": SCHEMA_VERSION, "candidate_id": "WT", "box_count": BOX_COUNT,
        "ligand_count": len(library), "seed": args.seed, "cpu_per_job": args.cpu_per_job,
        "exhaustiveness": args.exhaustiveness, "num_modes": NUM_MODES, "energy_range": ENERGY_RANGE,
        "vina": {"path": str(vina_path), "sha256": file_hash(vina_path)},
        "ligand_manifest": {"path": str(ligand_path), "sha256": file_hash(ligand_path)},
        "site_manifest": {"path": str(site_path), "sha256": file_hash(site_path)},
        "library_sha256": canonical_hash(library), "receptor_sha256": receptors,
        "library": library, "boxes": boxes,
    }
    jobs = [{"run_id": f"WT_{ligand['identifier']}_{box['site_id']}_s{args.seed}",
             "ligand_id": ligand["identifier"], "name": ligand["name"], "ligand": ligand["pdbqt"], "heavy_atoms": str(ligand["heavy_atoms"]), **box}
            for ligand in library for box in boxes]
    if len({job["run_id"].casefold() for job in jobs}) != len(jobs):
        raise ValueError("Run ID collision: choose unambiguous ligand and box identifiers")
    identity["jobs_sha256"] = canonical_hash(jobs)
    return {"identity": identity, "protocol_sha256": canonical_hash(identity), "created_at": utc_now(), "interpretation": NOTICE}, jobs


def pid_alive(pid):
    """Read liveness; never use os.kill on Windows because it can terminate a PID."""
    if not isinstance(pid, int) or pid < 1:
        return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        kernel.GetExitCodeProcess.restype = ctypes.c_int
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel.CloseHandle.restype = ctypes.c_int
        handle = kernel.OpenProcess(0x1000, 0, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # Access denied/unknown: conservatively live.
        try:
            exit_code = ctypes.c_ulong()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class ProcessLock:
    """OS-held advisory lock, released on crashes, with readable PID metadata."""

    def __init__(self, path):
        self.path, self.handle, self.metadata = Path(path), None, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            handle.write(b" ")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            try:
                handle.seek(1)
                owner = json.loads(handle.read().decode("utf-8").strip())
            except (ValueError, OSError):
                owner = {}
            handle.close()
            pid = owner.get("pid")
            raise RuntimeError(f"Another batch owns {self.path}; PID={pid}, live={pid_alive(pid)}") from exc
        self.handle = handle
        self.metadata = {"pid": os.getpid(), "active": True, "acquired_at": utc_now(), "token": uuid.uuid4().hex}
        self._write_metadata()
        return self

    def _write_metadata(self):
        self.handle.seek(1)
        self.handle.truncate()
        self.handle.write(json.dumps(self.metadata, sort_keys=True).encode("utf-8"))
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def __exit__(self, exc_type, exc, traceback):
        self.metadata.update(active=False, released_at=utc_now())
        try:
            self._write_metadata()
        finally:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def batch_is_active(output_dir):
    path = Path(output_dir) / "batch.lock"
    try:
        with path.open("rb") as handle:
            handle.seek(1)
            metadata = json.loads(handle.read().decode("utf-8"))
        return metadata.get("active") is True and pid_alive(metadata.get("pid"))
    except (OSError, ValueError):
        return False


def ensure_protocol(output_dir, protocol, jobs):
    path, jobs_path = output_dir / "protocol.json", output_dir / "jobs.csv"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError("Refusing changed resumed protocol: hashes, boxes or Vina parameters differ. Use a new output directory.")
        saved_jobs = read_csv(jobs_path, JOB_FIELDS)
        if saved_jobs != jobs or canonical_hash(saved_jobs) != saved["identity"]["jobs_sha256"]:
            raise ValueError("Saved jobs.csv does not match the immutable protocol")
        return saved
    if jobs_path.exists() or (output_dir / "raw").exists() or (output_dir / "runs.csv").exists():
        raise ValueError("Refusing existing run artifacts without protocol.json")
    atomic_json(path, protocol)
    atomic_csv(jobs_path, JOB_FIELDS, jobs)
    return protocol


def job_directory(output_dir, job):
    return Path(output_dir) / "raw" / job["ligand_id"] / job["site_id"]


def read_status(output_dir, job, protocol_sha256, validate_cache=True):
    path = job_directory(output_dir, job) / "status.json"
    if not path.is_file():
        return None
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Unreadable job status (preserved): {path}") from exc
    if status.get("protocol_sha256") != protocol_sha256 or any(status.get(key) != job[key] for key in ("run_id", "ligand_id", "site_id")):
        raise ValueError(f"Job status does not match protocol: {path}")
    if status.get("status") not in {"success", "failed", "running"}:
        raise ValueError(f"Unknown job status: {path}")
    if status["status"] == "success":
        score = status.get("best_score")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"Successful cache has invalid score: {path}")
        if validate_cache:
            pose, log = Path(status.get("pose_file", "")), Path(status.get("log_file", ""))
            if not pose.is_file() or pose.stat().st_size == 0 or not log.is_file():
                raise ValueError(f"Successful cache has missing artifacts (preserved): {path}")
            match = LOG_SCORE.search(log.read_text(encoding="utf-8", errors="replace"))
            if match is None or abs(float(match.group(1)) - score) > 0.000001:
                raise ValueError(f"Successful cache score does not match its log (preserved): {path}")
            validate_pose_file(pose, int(job["heavy_atoms"]), score)
            if status.get("pose_sha256") != file_hash(pose) or status.get("log_sha256") != file_hash(log):
                raise ValueError(f"Successful cache artifact hashes changed (preserved): {path}")
    return status


def load_statuses(output_dir, jobs, protocol_sha256):
    return {job["run_id"]: status for job in jobs if (status := read_status(output_dir, job, protocol_sha256)) is not None}


def vina_command(job, settings, pose, log):
    return [str(settings.vina), "--receptor", job["receptor"], "--ligand", job["ligand"],
            *[item for key in COORDINATES for item in ("--" + key, str(job[key]))],
            "--exhaustiveness", str(settings.exhaustiveness), "--num_modes", str(NUM_MODES),
            "--energy_range", str(ENERGY_RANGE), "--seed", str(settings.seed),
            "--cpu", str(settings.cpu_per_job), "--out", str(pose), "--log", str(log)]


def run_job(job, settings, protocol_sha256, subprocess_runner=None):
    prior = read_status(settings.output_dir, job, protocol_sha256)
    if prior and prior["status"] == "success":
        return prior
    directory = job_directory(settings.output_dir, job)
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        attempt_dir = directory / f"attempt_{attempt:04d}"
        try:
            attempt_dir.mkdir()
            break
        except FileExistsError:
            attempt += 1
    pose, log, stdout = (attempt_dir / name for name in ("poses.pdbqt", "vina.log", "stdout.log"))
    status = {key: job[key] for key in ("run_id", "candidate_id", "ligand_id", "name", "site_id")}
    status.update(status="running", attempt=attempt, best_score=None, exit_code=None,
                  elapsed_seconds=0.0, started_at=utc_now(), finished_at=None,
                  pose_file=str(pose.resolve()), log_file=str(log.resolve()), stdout_file=str(stdout.resolve()),
                  error="", protocol_sha256=protocol_sha256,
                  seed=settings.seed, exhaustiveness=settings.exhaustiveness, cpu_per_job=settings.cpu_per_job,
                  num_modes=NUM_MODES, energy_range=ENERGY_RANGE)
    command = vina_command(job, settings, pose, log)
    status["command"] = command
    atomic_json(attempt_dir / "status.json", status)
    atomic_json(directory / "status.json", status)
    started = time.monotonic()
    try:
        result = (subprocess_runner or subprocess.run)(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        output_text = result.stdout or ""
        stdout.write_text(output_text, encoding="utf-8")
        log_text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else output_text
        if not log.is_file():
            log.write_text(log_text, encoding="utf-8")
        match = LOG_SCORE.search(log_text)
        score = float(match.group(1)) if match is not None else None
        successful = result.returncode == 0 and pose.is_file() and pose.stat().st_size > 0 and score is not None and math.isfinite(score)
        if successful:
            status.update(validate_pose_file(pose, int(job["heavy_atoms"]), score))
            status.update(pose_sha256=file_hash(pose), log_sha256=file_hash(log))
        status.update(status="success" if successful else "failed", exit_code=result.returncode,
                      best_score=score if successful else None,
                      error="" if successful else f"Vina did not produce a valid pose and score (exit={result.returncode}). {log_text[-2000:]}")
    except Exception as exc:
        status.update(status="failed", best_score=None, error=f"{type(exc).__name__}: {exc}")
        if not stdout.exists():
            stdout.write_text(status["error"] + "\n", encoding="utf-8")
    status.update(elapsed_seconds=round(time.monotonic() - started, 3), finished_at=utc_now())
    atomic_json(attempt_dir / "status.json", status)
    atomic_json(directory / "status.json", status)
    return status


def append_run(output_dir, status):
    path = Path(output_dir) / "runs.csv"
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(status)
        handle.flush()
        os.fsync(handle.fileno())


def progress_record(jobs, statuses, protocol, state, started=None, initial_terminal=0, selected_jobs=None):
    successful = sum(row["status"] == "success" for row in statuses.values())
    failed = sum(row["status"] == "failed" for row in statuses.values())
    recorded_running = sum(row["status"] == "running" for row in statuses.values())
    total, terminal = len(jobs), successful + failed
    elapsed = max(0.0, time.monotonic() - started) if started is not None else None
    newly_terminal = max(0, terminal - initial_terminal)
    rate = newly_terminal / elapsed if elapsed and newly_terminal else None
    remaining = total - terminal
    if terminal == total and state != "running":
        state = "completed_with_failures" if failed else "completed"
    return {"schema_version": SCHEMA_VERSION, "protocol_sha256": protocol["protocol_sha256"],
            "updated_at": utc_now(), "status": state, "total_jobs": total,
            "success": successful, "failed": failed, "terminal_jobs": terminal,
            "pending": total - len(statuses), "recorded_running": recorded_running,
            "remaining_nonterminal": remaining, "selected_jobs": selected_jobs,
            "elapsed_seconds_this_session": round(elapsed, 3) if elapsed is not None else None,
            "jobs_per_second_this_session": rate, "jobs_per_hour_this_session": rate * 3600 if rate is not None else None,
            "eta_seconds": remaining / rate if rate else None, "interpretation": NOTICE}


def summarize(output_dir, jobs, protocol, statuses=None, state="snapshot", started=None, initial_terminal=0, selected_jobs=None, progress_filename="progress.json"):
    output_dir = Path(output_dir)
    if statuses is None:
        statuses = load_statuses(output_dir, jobs, protocol["protocol_sha256"])
    unique_pairs = {(job["ligand_id"].casefold(), job["site_id"].casefold()) for job in jobs}
    if len(unique_pairs) != len(jobs) or len({job["run_id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate jobs cannot be summarized")
    grouped, current_rows, failures = {}, [], []
    for job in jobs:
        status = statuses.get(job["run_id"])
        current = {key: job[key] for key in ("run_id", "candidate_id", "ligand_id", "name", "site_id")}
        current.update(status or {"status": "pending", "protocol_sha256": protocol["protocol_sha256"]})
        current_rows.append(current)
        grouped.setdefault(job["ligand_id"], []).append((job, status))
        if not status or status["status"] != "success":
            failures.append({**current, "reason": "not_attempted" if not status else "interrupted_or_running" if status["status"] == "running" else "failed"})
    coverage = []
    for ligand_id, rows in grouped.items():
        if len(rows) != BOX_COUNT:
            raise ValueError(f"Ligand {ligand_id} does not have exactly eight box jobs")
        scores = [status["best_score"] for _, status in rows if status and status["status"] == "success"]
        successful_sites = [job["site_id"] for job, status in rows if status and status["status"] == "success"]
        missing_sites = [job["site_id"] for job, status in rows if not status or status["status"] != "success"]
        failed_sites = [job["site_id"] for job, status in rows if status and status["status"] == "failed"]
        row = {"ligand_id": ligand_id, "name": rows[0][0]["name"], "coverage": f"{len(scores)}/{BOX_COUNT}",
               "successful_boxes": len(scores), "expected_boxes": BOX_COUNT, "failed_boxes": len(failed_sites),
               "complete": len(scores) == BOX_COUNT, "successful_box_ids": ";".join(successful_sites),
               "missing_box_ids": ";".join(missing_sites), "failed_box_ids": ";".join(failed_sites),
               "strong_box_count_le_minus8": sum(score <= -8.0 for score in scores),
               "strong_box_fraction_le_minus8": sum(score <= -8.0 for score in scores) / BOX_COUNT,
               "strong_box_count_le_minus8_5": sum(score <= -8.5 for score in scores),
               "strong_box_count_le_minus9": sum(score <= -9.0 for score in scores),
               "best_score": min(scores) if scores else None,
               "mean_score": sum(scores) / len(scores) if scores else None, "interpretation": NOTICE}
        coverage.append(row)
    complete = [row for row in coverage if row["complete"]]
    primary = sorted(complete, key=lambda row: (-row["strong_box_count_le_minus8"], row["best_score"], row["mean_score"], row["ligand_id"]))
    secondary = sorted(complete, key=lambda row: (row["best_score"], row["mean_score"], row["ligand_id"]))
    coverage.sort(key=lambda row: row["ligand_id"])
    fields = list(coverage[0]) if coverage else []
    atomic_csv(output_dir / "current_runs.csv", RUN_FIELDS, current_rows)
    atomic_csv(output_dir / "failures.csv", (*RUN_FIELDS, "reason"), failures)
    atomic_csv(output_dir / "ligand_coverage.csv", fields, coverage)
    atomic_csv(output_dir / "ranking_primary_complete.csv", ("rank", *fields), [{"rank": rank, **row} for rank, row in enumerate(primary, 1)])
    atomic_csv(output_dir / "ranking_best_score_complete.csv", ("rank", *fields), [{"rank": rank, **row} for rank, row in enumerate(secondary, 1)])
    progress = progress_record(jobs, statuses, protocol, state, started, initial_terminal, selected_jobs)
    if started is None:
        # A heartbeat can preserve the runner's measured throughput. A snapshot
        # alone cannot estimate throughput from a single set of status files.
        try:
            previous = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        if previous.get("protocol_sha256") == protocol["protocol_sha256"]:
            for key in ("elapsed_seconds_this_session", "jobs_per_second_this_session", "jobs_per_hour_this_session", "completed_this_session", "cached_success_at_start"):
                if key in previous:
                    progress[key] = previous[key]
            rate = progress["jobs_per_second_this_session"]
            progress["eta_seconds"] = progress["remaining_nonterminal"] / rate if rate else None
    progress.update(complete_ligands=len(complete), incomplete_ligands=len(coverage) - len(complete),
                    ranking_primary="strong-box-count(score<=-8)/8 descending, then best score ascending, then mean score ascending",
                    ranking_secondary="best score ascending, then mean score ascending")
    atomic_json(output_dir / progress_filename, progress)
    if progress["terminal_jobs"] == len(jobs) and state != "running":
        atomic_json(output_dir / "completed.json", {**progress, "completed_at": utc_now()})
    return progress


def execute(args, protocol, jobs):
    statuses = load_statuses(args.output_dir, jobs, protocol["protocol_sha256"])
    pending = [job for job in jobs if not statuses.get(job["run_id"]) or statuses[job["run_id"]]["status"] != "success"]
    selected = pending[:args.limit_jobs] if args.limit_jobs is not None else pending
    started, completed_this_session = time.monotonic(), 0
    initial_success = sum(row["status"] == "success" for row in statuses.values())
    initial_terminal = sum(row["status"] in {"success", "failed"} for row in statuses.values())
    if selected:
        (Path(args.output_dir) / "completed.json").unlink(missing_ok=True)

    def update(state):
        progress = progress_record(jobs, statuses, protocol, state, started, initial_terminal, len(selected))
        elapsed = max(time.monotonic() - started, 0.000001)
        rate = completed_this_session / elapsed if completed_this_session else None
        progress.update(completed_this_session=completed_this_session,
                        cached_success_at_start=initial_success, jobs_per_second_this_session=rate,
                        jobs_per_hour_this_session=rate * 3600 if rate else None,
                        eta_seconds=(len(selected) - completed_this_session) / rate if rate else None)
        atomic_json(Path(args.output_dir) / "progress.json", progress)
        return progress

    update("running" if selected else "snapshot")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        job_iterator, active = iter(selected), {}

        def submit_next():
            job = next(job_iterator, None)
            if job is not None:
                # Do not count a retried failure as completed while it is active.
                statuses[job["run_id"]] = {**(statuses.get(job["run_id"]) or job), "status": "running"}
                active[pool.submit(run_job, job, args, protocol["protocol_sha256"])] = job
                return True
            return False

        for _ in range(min(args.workers, len(selected))):
            submit_next()
        next_update = time.monotonic() + PROGRESS_INTERVAL
        while active:
            done, _ = wait(active, timeout=max(0.0, next_update - time.monotonic()), return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    status = future.result()
                except Exception as exc:
                    # A worker failure must not cancel the remaining ligands.
                    try:
                        status = read_status(args.output_dir, job, protocol["protocol_sha256"])
                    except (OSError, ValueError):
                        status = None
                    if not status or status["status"] != "success":
                        status = {**(status or job), "status": "failed", "best_score": None,
                                  "finished_at": utc_now(), "error": f"Worker {type(exc).__name__}: {exc}",
                                  "protocol_sha256": protocol["protocol_sha256"]}
                        directory = job_directory(args.output_dir, job)
                        atomic_json(directory / "status.json", status)
                        if status.get("attempt"):
                            atomic_json(directory / f"attempt_{status['attempt']:04d}" / "status.json", status)
                statuses[job["run_id"]] = status
                append_run(args.output_dir, status)
                completed_this_session += 1
                update("running")
                if completed_this_session == 1 or completed_this_session % 25 == 0 or completed_this_session == len(selected):
                    print(f"session={completed_this_session}/{len(selected)} success={sum(r['status'] == 'success' for r in statuses.values())}/{len(jobs)} failed={sum(r['status'] == 'failed' for r in statuses.values())} run={job['run_id']}", flush=True)
                submit_next()
            if time.monotonic() >= next_update:
                update("running")
                next_update = time.monotonic() + PROGRESS_INTERVAL
    with ProcessLock(Path(args.output_dir) / "summary.lock"):
        result = summarize(args.output_dir, jobs, protocol, statuses, "partial" if len(selected) < len(pending) else "finished", started, initial_terminal, len(selected))
    elapsed = max(time.monotonic() - started, 0.000001)
    rate = completed_this_session / elapsed if completed_this_session else None
    result.update(completed_this_session=completed_this_session, cached_success_at_start=initial_success,
                  jobs_per_second_this_session=rate, jobs_per_hour_this_session=rate * 3600 if rate else None,
                  eta_seconds=0 if not result["remaining_nonterminal"] else result["remaining_nonterminal"] / rate if rate else None)
    atomic_json(Path(args.output_dir) / "progress.json", result)
    if result["terminal_jobs"] == len(jobs):
        atomic_json(Path(args.output_dir) / "completed.json", {**result, "completed_at": utc_now()})
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand-manifest", type=Path)
    parser.add_argument("--site-jobs", type=Path)
    parser.add_argument("--vina", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--cpu-per-job", type=int, default=4)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--expected-ligands", type=int, help="Optional count guard; otherwise infer from manifest")
    parser.add_argument("--limit-jobs", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    supplied = (args.ligand_manifest, args.site_jobs, args.vina)
    if not args.summarize_only and not all(supplied):
        parser.error("--ligand-manifest, --site-jobs and --vina are required for execution")
    if args.summarize_only and any(supplied) and not all(supplied):
        parser.error("For summarize-only, supply all three input paths or none")
    return args


def main(argv=None):
    args = arguments(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        with ProcessLock(args.output_dir / "summary.lock"):
            protocol = json.loads((args.output_dir / "protocol.json").read_text(encoding="utf-8"))
            if canonical_hash(protocol["identity"]) != protocol["protocol_sha256"]:
                raise ValueError("Invalid protocol.json identity hash")
            jobs = read_csv(args.output_dir / "jobs.csv", JOB_FIELDS)
            if canonical_hash(jobs) != protocol["identity"]["jobs_sha256"]:
                raise ValueError("Invalid saved jobs.csv hash")
            if args.ligand_manifest:
                current, current_jobs = build_protocol(args)
                ensure_protocol(args.output_dir, current, current_jobs)
            active = batch_is_active(args.output_dir)
            result = summarize(args.output_dir, jobs, protocol, state="running" if active else "snapshot", progress_filename="summary_progress.json" if active else "progress.json")
    else:
        with ProcessLock(args.output_dir / "batch.lock"):
            protocol, jobs = build_protocol(args)
            protocol = ensure_protocol(args.output_dir, protocol, jobs)
            args.vina = Path(protocol["identity"]["vina"]["path"])
            result = execute(args, protocol, jobs)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 2 if result["status"] == "completed_with_failures" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
