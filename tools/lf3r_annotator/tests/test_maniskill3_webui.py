from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

import build_maniskill3_manifest as maniskill_manifest  # noqa: E402
from webui_rollout import WebUIRolloutGenerationService  # noqa: E402


class _Coordinator:
    def acquire(self, *_args, **_kwargs):
        raise AssertionError("ManiSkill3 generation must not acquire the LIBERO manifest writer")

    def release(self, *_args, **_kwargs):
        raise AssertionError("ManiSkill3 generation must not release the LIBERO manifest writer")


class _Tmux:
    def __init__(self) -> None:
        self.handler = None
        self.submitted = None

    def register_handler(self, job_type, on_loaded, on_poll=None, on_finished=None):
        self.handler = (job_type, on_loaded, on_poll, on_finished)

    def submit(
        self,
        job,
        command,
        log_path,
        interpreter=None,
        environment=None,
        on_poll=None,
        on_finished=None,
    ):
        self.submitted = {
            "job": dict(job),
            "command": list(command),
            "log_path": Path(log_path),
            "interpreter": interpreter,
            "environment": dict(environment or {}),
            "on_poll": on_poll,
            "on_finished": on_finished,
        }
        job["status"] = "running"
        return dict(job)

    def list(self, *_args, **_kwargs):
        return []


class ManiSkill3WebUIContractTest(unittest.TestCase):
    def test_new_python_modules_parse(self) -> None:
        for name in [
            "webui_rollout.py",
            "generate_maniskill3_success.py",
            "build_maniskill3_manifest.py",
        ]:
            ast.parse((TOOL_ROOT / name).read_text(encoding="utf-8"), filename=name)

    def test_frontend_module_is_loaded_and_preserves_libero_default(self) -> None:
        workspace = (TOOL_ROOT / "static/workspace.js").read_text(encoding="utf-8")
        frontend = (TOOL_ROOT / "static/runs/maniskill3.js").read_text(encoding="utf-8")
        self.assertIn('/static/runs/maniskill3.js', workspace)
        self.assertIn('value="libero" selected', frontend)
        self.assertIn('value="maniskill3"', frontend)
        self.assertIn('rolloutGenerationRenderWidth', frontend)
        self.assertIn('rolloutGenerationRenderHeight', frontend)
        self.assertIn('generator: "maniskill3"', frontend)
        self.assertIn('/api/rollouts/generate', frontend)

    def test_maniskill_command_receives_resolution_and_project_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            manifest = project_root / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("", encoding="utf-8")

            python = project_root / "conda_envs/LF3R-maniskill3/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            runner = project_root / "tools/lf3r_annotator/generate_maniskill3_success.py"
            runner.parent.mkdir(parents=True)
            runner.touch()

            tmux = _Tmux()
            service = WebUIRolloutGenerationService(
                project_root,
                manifest,
                _Coordinator(),
                tmux,
            )
            job = service.start(
                {
                    "generator": "maniskill3",
                    "gpu": "2",
                    "maniskill_env_id": "PickCube-v1",
                    "trials": 3,
                    "seed": 11,
                    "run_label": "web",
                    "render_width": 640,
                    "render_height": 384,
                }
            )

            self.assertEqual(job["generator"], "maniskill3")
            self.assertEqual(job["render_width"], 640)
            self.assertEqual(job["render_height"], 384)
            self.assertEqual(job["expected_rollouts"], 3)
            self.assertEqual(
                job["manifest_path"],
                "datasets/lf3r_failure_rollouts/v1/maniskill3_manifest.jsonl",
            )
            self.assertFalse(job["manifest_rebuilt"])
            self.assertIsNotNone(tmux.submitted)

            submitted = tmux.submitted
            command = submitted["command"]
            self.assertEqual(command[0], str(python))
            self.assertIn("--env-id", command)
            self.assertEqual(command[command.index("--env-id") + 1], "PickCube-v1")
            self.assertEqual(command[command.index("--render-width") + 1], "640")
            self.assertEqual(command[command.index("--render-height") + 1], "384")
            self.assertEqual(command[command.index("--num-traj") + 1], "3")

            environment = submitted["environment"]
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(
                environment["MS_ASSET_DIR"],
                str(project_root / "datasets/maniskill3"),
            )
            self.assertEqual(
                environment["SAPIEN_CACHE_DIR"],
                str(project_root / "cache/sapien"),
            )

    def test_standalone_manifest_records_video_and_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            video_dir = (
                project_root
                / "outputs/maniskill3/lf3r-maniskill3-pickcube-v1-test"
                / "PickCube-v1"
                / "motionplanning"
            )
            video_dir.mkdir(parents=True)
            (video_dir / "0.mp4").touch()
            (video_dir / "trajectory.h5").touch()
            (video_dir / "trajectory.json").write_text("{}", encoding="utf-8")
            output = (
                project_root
                / "datasets/lf3r_failure_rollouts/v1/maniskill3_manifest.jsonl"
            )

            with mock.patch.object(
                maniskill_manifest,
                "_probe_video",
                return_value={
                    "total_frames": 80,
                    "fps": 20.0,
                    "duration_seconds": 4.0,
                    "render_width": 640,
                    "render_height": 384,
                },
            ):
                summary = maniskill_manifest.build_manifest(
                    project_root=project_root,
                    scan_root=project_root / "outputs/maniskill3",
                    output=output,
                )

            self.assertEqual(summary["total_rollouts"], 1)
            row = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(row["task_suite"], "maniskill3")
            self.assertEqual(row["task_key"], "PickCube-v1")
            self.assertEqual(row["ground_truth_outcome"], "success")
            self.assertEqual(row["camera_video_paths"]["cam_high"], str((video_dir / "0.mp4").relative_to(project_root)))
            self.assertEqual(row["trajectory_path"], str((video_dir / "trajectory.h5").relative_to(project_root)))
            self.assertEqual(row["trajectory_group"], "traj_0")
            self.assertEqual(row["render_width"], 640)
            self.assertEqual(row["render_height"], 384)


if __name__ == "__main__":
    unittest.main()
