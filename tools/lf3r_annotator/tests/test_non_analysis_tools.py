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

    def submit_async(self, job, command, log_path, **kwargs):
        self.submitted = (job, command, log_path, kwargs)
        result = dict(job)
        result["status"] = "queued"
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
            {"extra_scan_roots": ["tools"]}
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

    def test_manifest_batch_transcode_command_uses_only_manifest_paths(self):
        manifest = tools.PROJECT_ROOT / "datasets" / "lf3r_failure_rollouts" / "v1" / "manifest.jsonl"
        if not manifest.is_file():
            self.skipTest("project manifest is unavailable")
        command = tools.transcode_manifest_videos_command(
            {"manifest_paths": [str(manifest.relative_to(tools.PROJECT_ROOT))]}
        )
        self.assertIn("transcode_manifest_videos_h264.py", command[1])
        self.assertEqual(command.count("--manifest"), 1)
        self.assertNotIn("camera_video_paths", " ".join(command))

    def test_manifest_batch_transcode_requires_existing_jsonl(self):
        with self.assertRaisesRegex(ValueError, "manifest does not exist"):
            tools.transcode_manifest_videos_command(
                {"manifest_paths": ["datasets/lf3r_failure_rollouts/v1/not-real.jsonl"]}
            )
        with self.assertRaisesRegex(ValueError, "manifest_paths must be a non-empty list"):
            tools.transcode_manifest_videos_command({"manifest_paths": []})

    def test_manifest_rebuild_is_serialized(self):
        tmux = FakeTmux()
        service = tools.NonAnalysisToolService(tools.PROJECT_ROOT, tmux)
        service.submit("rebuild_manifest", {})
        with self.assertRaisesRegex(ValueError, "another manifest rebuild is already active"):
            service.submit("rebuild_manifest", {})

    def test_batch_transcode_is_serialized(self):
        tmux = FakeTmux()
        service = tools.NonAnalysisToolService(tools.PROJECT_ROOT, tmux)
        manifest = tools.PROJECT_ROOT / "datasets" / "lf3r_failure_rollouts" / "v1" / "manifest.jsonl"
        if not manifest.is_file():
            self.skipTest("project manifest is unavailable")
        service.submit(
            "transcode_manifest_videos",
            {"manifest_paths": [str(manifest.relative_to(tools.PROJECT_ROOT))]},
        )
        with self.assertRaisesRegex(ValueError, "another H.264 transcode job is already active"):
            service.submit(
                "transcode_manifest_videos",
                {"manifest_paths": [str(manifest.relative_to(tools.PROJECT_ROOT))]},
            )

    def test_service_submits_persistent_project_tool_job(self):
        tmux = FakeTmux()
        service = tools.NonAnalysisToolService(tools.PROJECT_ROOT, tmux)
        job = service.submit(
            "validate_variants",
            {},
            client_request_id="web:validate_variants:test123",
        )
        self.assertEqual(job["job_type"], tools.TOOL_JOB_TYPE)
        self.assertEqual(job["action"], "validate_variants")
        self.assertEqual(job["client_request_id"], "web:validate_variants:test123")
        self.assertEqual(tmux.submitted[0]["client_request_id"], "web:validate_variants:test123")
        self.assertEqual(tmux.handler[0], tools.TOOL_JOB_TYPE)
        self.assertIn("--check-only", tmux.submitted[1])
        self.assertEqual(job["status"], "queued")

    def test_runs_ui_exposes_external_rollout_rescan_and_auto_refresh(self):
        source = (HERE / "static" / "runs-layout.js").read_text(encoding="utf-8")
        self.assertIn("rebuildManifestExtraRoots", source)
        self.assertIn("extra_scan_roots", source)
        self.assertIn("Default scan roots · always included", source)
        self.assertIn("Rebuild manifest + refresh", source)
        self.assertIn("await loadRollouts(preferredId)", source)
        self.assertIn("await loadRolloutOptions()", source)
        self.assertIn("lf3r.runs.extraManifestScanRoots", source)
        self.assertIn("Manifest rebuild failed:", source)
        self.assertIn("client_request_id", source)
        self.assertIn("recoverToolSubmission", source)
        self.assertIn("AbortController", source)
        self.assertIn("Project-tool submission timed out", source)
        self.assertIn("Manifest rebuild running · ", source)
        self.assertIn("another manifest rebuild is already active", (HERE / "non_analysis_tools.py").read_text(encoding="utf-8"))
        self.assertIn("batchManifestTranscodeRun", source)
        self.assertIn("batchManifestTranscodeManifests", source)
        self.assertIn("batchManifestTranscodeSelectAll", source)
        self.assertIn("batchManifestTranscodeSelectNone", source)
        self.assertIn("selectedBatchManifestPaths", source)
        self.assertIn("loadBatchManifestOptions", source)
        self.assertIn("manifest_paths: manifests", source)
        self.assertIn("transcode_manifest_videos", source)
        self.assertIn("canonical <code>video_path</code>", source)
        self.assertIn("BATCH_H264_SUMMARY", source)
        manifest_support = (HERE / "static" / "manifest-support-v2.js").read_text(encoding="utf-8")
        self.assertIn("var playbackPath = record.video_path;", manifest_support)


if __name__ == "__main__":
    unittest.main()
