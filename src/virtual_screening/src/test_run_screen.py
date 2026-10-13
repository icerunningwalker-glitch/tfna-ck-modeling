"""Unit tests use synthetic files and a mocked subprocess; no Vina is launched."""
import argparse
import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import call, patch

import run_screen as screen


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vina_runner_test_")
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.receptor = self.root / "WT.pdbqt"
        self.receptor.write_text("ATOM synthetic receptor\n", encoding="utf-8")
        self.vina = self.root / "mock_vina.exe"
        self.vina.write_text("not an executable; tests always mock subprocess\n", encoding="utf-8")
        self.ligand_manifest = self.root / "ligands.csv"
        self.site_manifest = self.root / "boxes.csv"
        self.ligands = []
        for ligand_id in ("A", "B", "C"):
            path = self.root / f"{ligand_id}.pdbqt"
            path.write_text("ROOT\n" + self.atom_line(1) + "\n" + self.atom_line(2, "OA") + "\n" + self.atom_line(3, "HD") + "\nENDROOT\nTORSDOF 0\n", encoding="utf-8")
            self.ligands.append({"identifier": ligand_id, "name": f"name {ligand_id}", "pdbqt": str(path), "status": "prepared"})
        self.boxes = [{"candidate_id": "WT", "site_id": f"box{i}", "receptor": str(self.receptor),
                       "center_x": str(i), "center_y": "0", "center_z": "0", "size_x": "58", "size_y": "71", "size_z": "61"}
                      for i in range(1, 9)]
        self.write_csv(self.ligand_manifest, self.ligands)
        self.write_csv(self.site_manifest, self.boxes)
        self.args = argparse.Namespace(ligand_manifest=self.ligand_manifest, site_jobs=self.site_manifest,
                                       vina=self.vina, output_dir=self.output, workers=2, cpu_per_job=4,
                                       exhaustiveness=8, seed=104729, expected_ligands=3, limit_jobs=None)
        self.protocol, self.jobs = screen.build_protocol(self.args)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def atom_line(serial, atom_type="C", x=0.0):
        name = f"C{serial}"
        return f"ATOM  {serial:5d} {name:<4} LIG A   1    {x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}    {0.0:6.3f} {atom_type}"

    @staticmethod
    def success_runner(command, **kwargs):
        pose = Path(command[command.index("--out") + 1])
        log = Path(command[command.index("--log") + 1])
        pose.write_text("MODEL 1\nREMARK VINA RESULT: -8.600 0.0 0.0\nROOT\n" + ScreenTests.atom_line(1) + "\n" + ScreenTests.atom_line(2, "OA") + "\nENDROOT\nTORSDOF 0\nENDMDL\n", encoding="utf-8")
        log.write_text("mode | affinity | dist\n    1    -8.6    0.0    0.0\n", encoding="utf-8")
        return argparse.Namespace(returncode=0, stdout="mock success\n")

    @staticmethod
    def failure_runner(command, **kwargs):
        return argparse.Namespace(returncode=1, stdout="mock failure\n")

    def statuses(self, scores, failed=()):
        result = {}
        for job in self.jobs:
            index = int(job["site_id"].removeprefix("box")) - 1
            values = scores.get(job["ligand_id"], [])
            if index < len(values):
                result[job["run_id"]] = {**job, "status": "success", "best_score": values[index], "protocol_sha256": self.protocol["protocol_sha256"]}
            if (job["ligand_id"], job["site_id"]) in failed:
                result[job["run_id"]] = {**job, "status": "failed", "best_score": None, "error": "mock failure", "protocol_sha256": self.protocol["protocol_sha256"]}
        return result

    def test_atomic_json_retries_transient_permission_error(self):
        path = self.output / "progress.json"
        path.write_text('{"status": "old"}\n', encoding="utf-8")
        value = {"status": "running", "completed": 2, "note": "合成测试"}
        real_replace = os.replace
        attempts = []

        def temporarily_locked(source, target):
            attempts.append((source, target))
            if len(attempts) <= 2:
                raise PermissionError("synthetic temporary file lock")
            real_replace(source, target)

        with patch.object(screen.os, "replace", side_effect=temporarily_locked) as replace, patch.object(screen.time, "sleep") as sleep:
            screen.atomic_json(path, value)
        self.assertEqual(replace.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(0.1), call(0.2)])
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), value)
        self.assertEqual(list(self.output.glob("*.tmp")), [])

    def test_replace_retry_exhaustion_preserves_original_and_raises(self):
        temporary = self.output / "progress.json.tmp"
        path = self.output / "progress.json"
        temporary.write_text("new contents", encoding="utf-8")
        path.write_text("old contents", encoding="utf-8")
        error = PermissionError("synthetic persistent file lock")
        with patch.object(screen.os, "replace", side_effect=error) as replace, patch.object(screen.time, "sleep") as sleep:
            with self.assertRaises(PermissionError) as caught:
                screen.replace_with_retry(temporary, path, attempts=3, delay_seconds=0.1)
        self.assertIs(caught.exception, error)
        self.assertEqual(replace.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(0.1), call(0.2)])
        self.assertEqual(path.read_text(encoding="utf-8"), "old contents")
        self.assertEqual(temporary.read_text(encoding="utf-8"), "new contents")

    def test_replace_retries_documented_windows_lock_errors(self):
        real_replace = os.replace
        for winerror in (5, 32, 33):
            with self.subTest(winerror=winerror):
                temporary = self.output / "progress.json.tmp"
                path = self.output / "progress.json"
                temporary.write_text("new contents", encoding="utf-8")
                error = OSError("synthetic Windows file lock")
                error.winerror = winerror
                attempts = []

                def temporarily_locked(source, target):
                    attempts.append((source, target))
                    if len(attempts) == 1:
                        raise error
                    real_replace(source, target)

                with patch.object(screen.os, "replace", side_effect=temporarily_locked) as replace, patch.object(screen.time, "sleep") as sleep:
                    screen.replace_with_retry(temporary, path, attempts=3, delay_seconds=0.1)
                self.assertEqual(replace.call_count, 2)
                sleep.assert_called_once_with(0.1)
                self.assertEqual(path.read_text(encoding="utf-8"), "new contents")
                self.assertFalse(temporary.exists())

    def test_replace_propagates_unrelated_errors_without_retry(self):
        temporary = self.output / "progress.json.tmp"
        path = self.output / "progress.json"
        for error in (OSError("synthetic unrelated I/O error"), ValueError("synthetic invalid path")):
            with self.subTest(error=type(error).__name__):
                with patch.object(screen.os, "replace", side_effect=error) as replace, patch.object(screen.time, "sleep") as sleep:
                    with self.assertRaises(type(error)) as caught:
                        screen.replace_with_retry(temporary, path, attempts=3, delay_seconds=0.1)
                self.assertIs(caught.exception, error)
                replace.assert_called_once_with(temporary, path)
                sleep.assert_not_called()

    def test_atomic_csv_uses_shared_retry_helper(self):
        path = self.output / "progress.csv"
        rows = [{"status": "running", "completed": 2, "note": "合成测试"}]
        real_replace = os.replace
        attempts = []

        def temporarily_locked(source, target):
            attempts.append((source, target))
            if len(attempts) == 1:
                raise PermissionError("synthetic temporary CSV lock")
            real_replace(source, target)

        with patch.object(screen, "replace_with_retry", wraps=screen.replace_with_retry) as retry, patch.object(screen.os, "replace", side_effect=temporarily_locked) as replace, patch.object(screen.time, "sleep") as sleep:
            screen.atomic_csv(path, ("status", "completed", "note"), rows)
        retry.assert_called_once()
        self.assertEqual(Path(retry.call_args.args[1]), path)
        self.assertEqual(replace.call_count, 2)
        sleep.assert_called_once_with(0.1)
        self.assertEqual(screen.read_csv(path, ("status", "completed", "note")), [{"status": "running", "completed": "2", "note": "合成测试"}])
        self.assertEqual(list(self.output.glob("*.tmp")), [])

    def test_protocol_inputs_and_hashes_are_fixed(self):
        self.assertEqual(len(self.jobs), 24)
        identity = self.protocol["identity"]
        self.assertEqual(identity["num_modes"], 20)
        self.assertEqual(identity["energy_range"], 4)
        self.assertEqual(identity["box_count"], 8)
        self.assertEqual(len(identity["library"]), 3)
        screen.ensure_protocol(self.output, self.protocol, self.jobs)
        second, jobs = screen.build_protocol(self.args)
        screen.ensure_protocol(self.output, second, jobs)
        self.args.workers = 5  # Scheduler concurrency may change on resume.
        same, _ = screen.build_protocol(self.args)
        self.assertEqual(same["protocol_sha256"], second["protocol_sha256"])
        self.args.exhaustiveness = 9
        changed, changed_jobs = screen.build_protocol(self.args)
        with self.assertRaisesRegex(ValueError, "changed resumed protocol"):
            screen.ensure_protocol(self.output, changed, changed_jobs)
        self.args.exhaustiveness = 8
        self.receptor.write_text("changed receptor\n", encoding="utf-8")
        changed, changed_jobs = screen.build_protocol(self.args)
        with self.assertRaisesRegex(ValueError, "changed resumed protocol"):
            screen.ensure_protocol(self.output, changed, changed_jobs)

    def test_duplicate_ligands_boxes_and_bad_coordinates_rejected(self):
        self.ligands[1]["identifier"] = "a"
        self.write_csv(self.ligand_manifest, self.ligands)
        with self.assertRaisesRegex(ValueError, "Duplicate ligand"):
            screen.build_protocol(self.args)
        self.ligands[1]["identifier"] = "B"
        self.write_csv(self.ligand_manifest, self.ligands)
        self.boxes[1]["site_id"] = "BOX1"
        self.write_csv(self.site_manifest, self.boxes)
        with self.assertRaisesRegex(ValueError, "Duplicate site"):
            screen.build_protocol(self.args)
        self.boxes[1]["site_id"] = "box2"
        self.boxes[0]["size_x"] = "nan"
        self.write_csv(self.site_manifest, self.boxes)
        with self.assertRaisesRegex(ValueError, "Invalid size_x"):
            screen.build_protocol(self.args)

    def test_failed_attempt_preserved_success_cache_never_overwritten(self):
        job, sha = self.jobs[0], self.protocol["protocol_sha256"]
        failed = screen.run_job(job, self.args, sha, self.failure_runner)
        self.assertEqual(failed["status"], "failed")
        failed_path = screen.job_directory(self.output, job) / "attempt_0001" / "status.json"
        failed_bytes = failed_path.read_bytes()
        succeeded = screen.run_job(job, self.args, sha, self.success_runner)
        self.assertEqual(succeeded["attempt"], 2)
        self.assertEqual(succeeded["status"], "success")
        self.assertEqual(succeeded["best_score"], -8.6)
        self.assertEqual(failed_path.read_bytes(), failed_bytes)
        pointer = screen.job_directory(self.output, job) / "status.json"
        original = pointer.read_bytes()
        def forbidden(*args, **kwargs):
            raise AssertionError("successful job was launched again")
        resumed = screen.run_job(job, self.args, sha, forbidden)
        self.assertEqual(resumed, succeeded)
        self.assertEqual(pointer.read_bytes(), original)
        command = succeeded["command"]
        for flag, value in (("--cpu", "4"), ("--seed", "104729"), ("--num_modes", "20"), ("--energy_range", "4"), ("--exhaustiveness", "8")):
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_subprocess_exception_is_a_persisted_failure(self):
        def missing_executable(*args, **kwargs):
            raise OSError("synthetic missing executable")
        result = screen.run_job(self.jobs[0], self.args, self.protocol["protocol_sha256"], missing_executable)
        self.assertEqual(result["status"], "failed")
        self.assertIn("OSError", result["error"])
        self.assertEqual(screen.read_status(self.output, self.jobs[0], self.protocol["protocol_sha256"]), result)

    def test_invalid_success_cache_is_preserved_and_refused(self):
        result = screen.run_job(self.jobs[0], self.args, self.protocol["protocol_sha256"], self.success_runner)
        pointer = screen.job_directory(self.output, self.jobs[0]) / "status.json"
        original = pointer.read_bytes()
        Path(result["pose_file"]).unlink()
        with self.assertRaisesRegex(ValueError, "missing artifacts"):
            screen.run_job(self.jobs[0], self.args, self.protocol["protocol_sha256"], self.success_runner)
        self.assertEqual(pointer.read_bytes(), original)

    def test_strict_pose_gate_rejects_missing_atoms_mismatched_score_and_nan(self):
        job, sha = self.jobs[0], self.protocol["protocol_sha256"]
        def malformed(command, **kwargs):
            result = self.success_runner(command, **kwargs)
            pose = Path(command[command.index("--out") + 1])
            pose.write_text("MODEL 1\nREMARK VINA RESULT: -8.6 0.0 0.0\nROOT\n" + self.atom_line(1) + "\nENDROOT\nENDMDL\n", encoding="utf-8")
            return result
        result = screen.run_job(job, self.args, sha, malformed)
        self.assertEqual(result["status"], "failed")
        self.assertIn("heavy-atom count", result["error"])
        pose = self.root / "validation_pose.pdbqt"
        valid = "MODEL 1\nREMARK VINA RESULT: -8.612 0 0\nROOT\n" + self.atom_line(1) + "\n" + self.atom_line(2, "OA") + "\nENDROOT\nENDMDL\n"
        pose.write_text(valid, encoding="utf-8")
        self.assertEqual(screen.validate_pose_file(pose, 2, -8.6)["heavy_atoms_per_pose"], 2)
        with self.assertRaisesRegex(ValueError, "does not match"):
            screen.validate_pose_file(pose, 2, -8.2)
        pose.write_text(valid.replace(self.atom_line(1), self.atom_line(1, x=float("nan"))), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            screen.validate_pose_file(pose, 2, -8.6)
        pose.write_text(valid.replace("ENDROOT\n", ""), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ROOT/ENDROOT"):
            screen.validate_pose_file(pose, 2, -8.6)

    def test_active_summary_does_not_overwrite_runtime_progress_or_complete(self):
        screen.ensure_protocol(self.output, self.protocol, self.jobs)
        runtime = {"protocol_sha256": self.protocol["protocol_sha256"], "status": "running", "jobs_per_second_this_session": 1.25, "eta_seconds": 19.2}
        screen.atomic_json(self.output / "progress.json", runtime)
        original = (self.output / "progress.json").read_bytes()
        with screen.ProcessLock(self.output / "batch.lock"), redirect_stdout(io.StringIO()):
            self.assertEqual(screen.main(["--output-dir", str(self.output), "--summarize-only"]), 0)
        self.assertEqual((self.output / "progress.json").read_bytes(), original)
        snapshot = json.loads((self.output / "summary_progress.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["status"], "running")
        self.assertEqual(snapshot["jobs_per_second_this_session"], 1.25)
        self.assertFalse((self.output / "completed.json").exists())

    def test_incomplete_ligands_excluded_and_rankings_differ(self):
        statuses = self.statuses({"A": [-8.0, -8.5, -9.0, -8.2, -7, -7, -7, -7],
                                  "B": [-12.0, -7, -7, -7, -7, -7, -7, -7],
                                  "C": [-20.0]}, failed={("C", "box2")})
        result = screen.summarize(self.output, self.jobs, self.protocol, statuses)
        primary = screen.read_csv(self.output / "ranking_primary_complete.csv", ("ligand_id",))
        secondary = screen.read_csv(self.output / "ranking_best_score_complete.csv", ("ligand_id",))
        self.assertEqual([row["ligand_id"] for row in primary], ["A", "B"])
        self.assertEqual([row["ligand_id"] for row in secondary], ["B", "A"])
        self.assertEqual(primary[0]["strong_box_count_le_minus8"], "4")
        self.assertEqual(primary[0]["strong_box_count_le_minus8_5"], "2")
        self.assertEqual(primary[0]["strong_box_count_le_minus9"], "1")
        coverage = screen.read_csv(self.output / "ligand_coverage.csv", ("coverage",))
        self.assertEqual(coverage[2]["coverage"], "1/8")
        self.assertIn("box8", coverage[2]["missing_box_ids"])
        failures = screen.read_csv(self.output / "failures.csv", ("reason",))
        self.assertEqual(len(failures), 7)
        self.assertEqual({row["reason"] for row in failures}, {"failed", "not_attempted"})
        self.assertEqual(result["success"], 17)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["pending"], 6)
        self.assertEqual(result["complete_ligands"], 2)
        self.assertFalse((self.output / "completed.json").exists())

    def test_primary_ties_use_best_then_mean_score(self):
        statuses = self.statuses({"A": [-9, -8, -7, -7, -7, -7, -7, -7],
                                  "B": [-9, -8, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5],
                                  "C": [-10, -8, -6, -6, -6, -6, -6, -6]})
        screen.summarize(self.output, self.jobs, self.protocol, statuses)
        primary = screen.read_csv(self.output / "ranking_primary_complete.csv", ("ligand_id",))
        self.assertEqual([row["ligand_id"] for row in primary], ["C", "B", "A"])

    def test_duplicate_jobs_cannot_be_summarized(self):
        with self.assertRaisesRegex(ValueError, "Duplicate jobs"):
            screen.summarize(self.output, self.jobs + [self.jobs[0]], self.protocol, {})

    def test_complete_with_failures_is_terminal_and_coverage_explicit(self):
        statuses = self.statuses({"A": [-8] * 8, "B": [-7] * 8, "C": [-9] * 8}, failed={("C", "box8")})
        result = screen.summarize(self.output, self.jobs, self.protocol, statuses)
        self.assertEqual(result["status"], "completed_with_failures")
        complete = json.loads((self.output / "completed.json").read_text(encoding="utf-8"))
        self.assertEqual(complete["terminal_jobs"], 24)
        self.assertEqual(complete["incomplete_ligands"], 1)

    def test_limited_mock_execution_resume_failed_only_and_summary_cli(self):
        screen.ensure_protocol(self.output, self.protocol, self.jobs)
        self.args.limit_jobs = 2
        with patch.object(screen.subprocess, "run", self.success_runner), redirect_stdout(io.StringIO()):
            first = screen.execute(self.args, self.protocol, self.jobs)
        self.assertEqual(first["success"], 2)
        self.assertEqual(first["status"], "partial")
        self.assertFalse((self.output / "completed.json").exists())
        self.args.limit_jobs = None
        calls = []
        def mixed(command, **kwargs):
            calls.append(command)
            if Path(command[command.index("--ligand") + 1]).stem == "B":
                return self.failure_runner(command, **kwargs)
            return self.success_runner(command, **kwargs)
        with patch.object(screen.subprocess, "run", mixed), redirect_stdout(io.StringIO()):
            second = screen.execute(self.args, self.protocol, self.jobs)
        self.assertEqual(len(calls), 22)
        self.assertEqual(second["success"], 16)
        self.assertEqual(second["failed"], 8)
        self.assertEqual(second["status"], "completed_with_failures")
        with redirect_stdout(io.StringIO()):
            exit_code = screen.main(["--output-dir", str(self.output), "--summarize-only"])
        self.assertEqual(exit_code, 2)
        with patch.object(screen.subprocess, "run", self.success_runner), redirect_stdout(io.StringIO()):
            final = screen.execute(self.args, self.protocol, self.jobs)
        self.assertEqual(final["success"], 24)
        self.assertEqual(final["completed_this_session"], 8)
        self.assertEqual(final["status"], "completed")
        history = screen.read_csv(self.output / "runs.csv", ("run_id", "attempt"))
        current = screen.read_csv(self.output / "current_runs.csv", ("run_id", "attempt"))
        self.assertEqual(len(history), 32)
        self.assertEqual(len(current), 24)
        b_rows = [row for row in history if row["ligand_id"] == "B"]
        self.assertEqual({row["attempt"] for row in b_rows}, {"1", "2"})

    def test_live_process_lock_refuses_second_batch_and_releases(self):
        self.assertTrue(screen.pid_alive(os.getpid()))
        self.assertFalse(screen.pid_alive(-1))
        path = self.output / "batch.lock"
        with screen.ProcessLock(path):
            self.assertTrue(screen.batch_is_active(self.output))
            with self.assertRaisesRegex(RuntimeError, "live=True"):
                with screen.ProcessLock(path):
                    self.fail("duplicate lock acquired")
        self.assertFalse(screen.batch_is_active(self.output))
        with screen.ProcessLock(path):
            self.assertTrue(screen.batch_is_active(self.output))

    def test_unexpected_worker_failure_does_not_cancel_other_jobs(self):
        screen.ensure_protocol(self.output, self.protocol, self.jobs)
        self.args.limit_jobs = 3
        original = screen.run_job
        def worker(job, settings, protocol_sha256):
            if job["site_id"] == "box1":
                raise RuntimeError("mock worker exception")
            return original(job, settings, protocol_sha256, self.success_runner)
        with patch.object(screen, "run_job", worker), redirect_stdout(io.StringIO()):
            result = screen.execute(self.args, self.protocol, self.jobs)
        self.assertEqual(result["completed_this_session"], 3)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["success"], 2)
        stored = screen.read_status(self.output, self.jobs[0], self.protocol["protocol_sha256"])
        self.assertEqual(stored["status"], "failed")
        self.assertIn("mock worker exception", stored["error"])


if __name__ == "__main__":
    unittest.main()
