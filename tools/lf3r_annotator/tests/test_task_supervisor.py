from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from tools.lf3r_annotator.task_supervisor import TmuxJobSupervisor, TmuxSupervisorError


class TmuxJobSupervisorTest(unittest.TestCase):
    def test_tmux_job_persists_and_finishes_without_server_process(self) -> None:
        if not TmuxJobSupervisor(Path.cwd()).available:
            self.skipTest("tmux is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "project_env.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            command = [
                "bash",
                "-c",
                "printf 'supervisor-ok\\n'; exit 0",
            ]
            job = {
                "job_id": "supervisor-test",
                "job_type": "test",
                "status": "queued",
                "submitted_at": "2026-08-28T00:00:00+00:00",
            }
            supervisor = TmuxJobSupervisor(root)
            finished = []
            supervisor.register_handler("test", lambda loaded: None)
            supervisor.submit(
                job,
                command,
                root / "logs" / "job.log",
                interpreter="bash",
                on_finished=lambda item, return_code, reason: finished.append(
                    (item["job_id"], return_code, reason)
                ),
            )
            for _ in range(80):
                record_path = root / job["job_record_path"]
                if record_path.is_file():
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    if record.get("status") in {"complete", "failed"}:
                        break
                time.sleep(0.05)
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertTrue(record["persistent"])
            self.assertEqual(record["interpreter"], "bash")
            self.assertEqual(record["argv"], command)
            self.assertEqual(record["working_directory"], str(root))
            self.assertTrue(record["finished_at_path"])
            self.assertEqual(record["return_code"], 0)
            self.assertTrue((root / "logs" / "job.log").is_file())
            self.assertIn("supervisor-ok", (root / "logs" / "job.log").read_text())
            self.assertEqual(finished[-1][1], 0)
            for _ in range(40):
                if not supervisor.monitors:
                    break
                time.sleep(0.05)
            self.assertFalse(supervisor.monitors)

    def test_recover_reattaches_recorded_active_session(self) -> None:
        if not TmuxJobSupervisor(Path.cwd()).available:
            self.skipTest("tmux is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "project_env.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            job = {
                "job_id": "recover-test",
                "job_type": "test",
                "status": "queued",
                "submitted_at": "2026-08-28T00:00:00+00:00",
            }
            first = TmuxJobSupervisor(root)
            first.register_handler("test", lambda loaded: None)
            first.submit(
                job,
                ["bash", "-c", "sleep 1; printf recovered > recovered.txt"],
                root / "logs" / "recover.log",
                interpreter="bash",
                on_finished=lambda item, return_code, reason: item.update(
                    status="complete" if return_code == 0 else "failed"
                ),
            )
            second = TmuxJobSupervisor(root)
            finished = []

            def finish(item, return_code, reason):
                item["status"] = "complete" if return_code == 0 else "failed"
                finished.append(return_code)

            second.register_handler("test", lambda loaded: None, on_finished=finish)
            second.recover()
            recovered = second.get("recover-test")
            self.assertTrue(recovered.get("reattached_at"))
            self.assertIn(recovered["tmux_state"], {"reattached", "running", "exited"})
            record_path = root / job["job_record_path"]
            for _ in range(80):
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if record.get("return_code") == 0:
                    break
                time.sleep(0.05)
            self.assertEqual(record["return_code"], 0)
            # A previous server instance may still be finishing the same
            # persisted job while this supervisor reattaches. Wait for the
            # reattached supervisor's callback instead of coupling the
            # assertion to monitor scheduling.
            for _ in range(40):
                if 0 in finished:
                    break
                time.sleep(0.05)
            self.assertIn(0, finished)
            for _ in range(40):
                if not second.monitors:
                    break
                time.sleep(0.05)
            self.assertFalse(second.monitors)

    def test_recover_marks_disappeared_session_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "logs" / "annotator_jobs" / "orphan-test"
            registry.mkdir(parents=True)
            record_path = registry / "job.json"
            record_path.write_text(
                json.dumps(
                    {
                        "job_id": "orphan-test",
                        "job_type": "test",
                        "status": "running",
                        "tmux_session": "lf3r-annotator-orphan-test",
                        "submitted_at": "2026-08-28T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            supervisor = TmuxJobSupervisor(root, tmux_binary="tmux")
            supervisor.register_handler("test", lambda loaded: None)
            supervisor.recover()
            recovered = supervisor.get("orphan-test")
            self.assertEqual(recovered["status"], "failed")
            self.assertEqual(recovered["tmux_state"], "missing")
            self.assertTrue(recovered["finished_at"])

    def test_missing_tmux_does_not_fallback_to_background_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            supervisor = TmuxJobSupervisor(Path(directory), tmux_binary="tmux-not-installed")
            self.assertFalse(supervisor.available)
            with self.assertRaises(TmuxSupervisorError):
                supervisor.submit(
                    {
                        "job_id": "no-tmux",
                        "job_type": "test",
                        "status": "queued",
                    },
                    ["true"],
                    Path(directory) / "job.log",
                )


if __name__ == "__main__":
    unittest.main()
