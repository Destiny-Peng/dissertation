from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from indexed_task_supervisor import IndexedTmuxJobSupervisor  # noqa: E402


class IndexedTmuxJobSupervisorTest(unittest.TestCase):
    def test_recover_reads_only_indexed_or_live_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_root = root / "logs" / "annotator_jobs"

            first = IndexedTmuxJobSupervisor(root, tmux_binary="/bin/echo")
            active_dir = registry_root / "active-job"
            active_dir.mkdir(parents=True)
            active = {
                "job_id": "active-job",
                "job_type": "test",
                "status": "running",
                "tmux_session": "lf3r-annotator-active-job",
                "submitted_at": "2026-10-02T00:00:00+00:00",
            }
            first.persist(active)

            legacy_dir = registry_root / "legacy-terminal"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "job.json").write_text(
                json.dumps(
                    {
                        "job_id": "legacy-terminal",
                        "job_type": "test",
                        "status": "complete",
                    }
                ),
                encoding="utf-8",
            )

            second = IndexedTmuxJobSupervisor(root, tmux_binary="/bin/echo")
            second.register_handler("test", lambda _job: None)
            with (
                mock.patch.object(
                    second,
                    "_active_tmux_sessions",
                    return_value={"lf3r-annotator-active-job"},
                ),
                mock.patch.object(second, "_start_monitor"),
            ):
                stats = second.recover()

            self.assertEqual(stats["recovered"], 1)
            self.assertEqual(second.get("active-job")["status"], "running")
            with self.assertRaises(KeyError):
                second.get("legacy-terminal")

    def test_terminal_jobs_are_lazy_loaded_from_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = IndexedTmuxJobSupervisor(root, tmux_binary="/bin/echo")
            job_dir = root / "logs" / "annotator_jobs" / "done-job"
            job_dir.mkdir(parents=True)
            first.persist(
                {
                    "job_id": "done-job",
                    "job_type": "test",
                    "status": "complete",
                    "submitted_at": "2026-10-02T00:00:00+00:00",
                }
            )

            second = IndexedTmuxJobSupervisor(root, tmux_binary="/bin/echo")
            second.register_handler("test", lambda _job: None)
            with mock.patch.object(second, "_active_tmux_sessions", return_value=set()):
                stats = second.recover()

            self.assertEqual(stats["indexed_active"], 0)
            self.assertNotIn("done-job", second.jobs)
            listed = second.list(job_type="test")
            self.assertEqual([item["job_id"] for item in listed], ["done-job"])
            self.assertEqual(second.get("done-job")["status"], "complete")


if __name__ == "__main__":
    unittest.main()
