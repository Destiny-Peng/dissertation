#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import non_analysis_tools as tools  # noqa: E402


class FakeTmux:
    def __init__(self) -> None:
        self.handler = None
        self.submitted = None

    def register_handler(self, job_type, on_loaded, on_poll=None, on_finished=None):
        self.handler = (job_type, on_loaded, on_poll, on_finished)

    def submit(self, job, command, log_path, **kwargs):
        self.submitted = (job, command, log_path, kwargs)
        result = dict(job)
        result["status"] = "running"
        return result

    def get(self, job_id):
        if not self.submitted or self.submitted[0]["job_id"] != job_id:
            raise KeyError(job_id)
        return dict(self.submitted[0])

    def list(self, job_type=None, status=None):
        if not self.submitted:
            return []
        job = dict(self.submitted[0])
        if job_type and job.get("job_type") != job_type:
            return []
        if status and job.get("status") != status:
            return []
        return [job]


class NonAnalysisToolTests(unittest.TestCase):
    def test_gpu_status_is_one_shot_and_structured(self):
        output = "0, NVIDIA RTX, GPU-1, 8192, 2048, 6144, 25, 10, 61, 80, 140\n"
        completed = subprocess.CompletedProcess(["nvidia-smi"], 0, stdout=output, stderr="")
        with mock.patch.object(tools.shutil, "which", return_value="/usr/bin/nvidia-smi"), mock.patch.object(
            tools.subprocess, "run", return_value=completed
        ) as run:
            payload = tools.gpu_status()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["gpus"][0]["index"], 0)
        self.assertAlmostEqual(payload["gpus"][0]["memory_free_fraction"], 0.75)
        run.assert_called_once()

    def test_export_command_reuses_existing_script(self):
        command = tools.export_command(
            {
                "outcomes": ["failure", "recovered_success"],
                "dataset_role": "libero_10",
                "review_status": "complete",
                "dry_run": True,
            }
        )
        self.assertIn("tools/export_failure_cases.py", command[1])
        self.assertEqual(command.count("--outcome"), 2)
        self.assertIn("--dry-run", command)

    def test_safe_train_command_maps_visible_parameters(self):
        command = tools.safe_train_command(
            {
                "dataset_dir": "outputs/safe_training/datasets/example",
                "model": "lstm",
                "gpu": "2",
                "epochs": 7,
                "batch_size": 32,
                "hidden_dim": 128,
                "seed": "4",
                "logs_root": "outputs/safe_training/logs/example",
                "normalize": True,
            }
        )
        self.assertIn("tools/safe_training/run_safe_training.py", command[1])
        self.assertEqual(command[command.index("--model") + 1], "lstm")
        self.assertEqual(command[command.index("--gpu") + 1], "2")
        self.assertIn("--normalize", command)

    def test_robo_sweep_requires_rollout(self):
        with self.assertRaises(ValueError):
            tools.robo_interval_sweep_command({})

    def test_manifest_rebuild_keeps_default_roots_and_appends_extras(self):
        command = tools.rebuild_manifest_command(
            {"extra_scan_roots": ["tools", "outputs/openvla_libero"]}
        )
        roots = [
            Path(command[index + 1]).resolve()
            for index, token in enumerate(command[:-1])
            if token == "--scan-root"
        ]
        expected_defaults = [path.resolve() for path in tools.DEFAULT_MANIFEST_SCAN_ROOTS]
        for path in expected_defaults:
            self.assertIn(path, roots)
        self.assertIn((tools.PROJECT_ROOT / "tools").resolve(), roots)
        self.assertEqual(len(roots), len(set(roots)))
        self.assertEqual(roots[: len(expected_defaults)], expected_defaults)

    def test_manifest_rebuild_rejects_missing_extra_root(self):
        with self.assertRaisesRegex(ValueError, "scan root does not exist"):
            tools.rebuild_manifest_command(
                {"extra_scan_roots": ["outputs/definitely-not-a-real-rollout-root"]}
            )

    def test_service_submits_persistent_project_tool_job(self):
        tmux = FakeTmux()
        service = tools.NonAnalysisToolService(tools.PROJECT_ROOT, tmux)
        job = service.submit("validate_variants", {})
        self.assertEqual(job["job_type"], tools.TOOL_JOB_TYPE)
        self.assertEqual(job["action"], "validate_variants")
        self.assertEqual(tmux.handler[0], tools.TOOL_JOB_TYPE)
        self.assertIn("--check-only", tmux.submitted[1])


if __name__ == "__main__":
    unittest.main()
