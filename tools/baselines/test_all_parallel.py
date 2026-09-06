from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


RUNNER_PATH = Path(__file__).with_name("run_lf3r_baseline.py")
if str(RUNNER_PATH.parent) not in sys.path:
    sys.path.insert(0, str(RUNNER_PATH.parent))
SPEC = importlib.util.spec_from_file_location("lf3r_all_parallel_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class AllBaselineParallelRunnerTest(unittest.TestCase):
    def _records(self, root: Path, count: int = 4) -> list[dict]:
        records = []
        for index in range(count):
            video = root / f"rollout-{index}.mp4"
            video.write_bytes(b"video")
            csv_path = root / f"rollout-{index}.csv"
            csv_path.write_text("action/timestep,feature\n0,1\n", encoding="utf-8")
            records.append({
                "id": f"r{index}",
                "video_path": str(video),
                "csv_path": str(csv_path),
                "task_description": "test task",
                "total_frames": 20,
            })
        return records

    def _args(self, root: Path, baseline: str) -> argparse.Namespace:
        return argparse.Namespace(
            baseline=baseline,
            start_index=0,
            end_index=4,
            limit=None,
            parallel_workers=2,
            worker_spec=["0:0:2", "1:2:4"],
            gpu="0,1",
            dry_run=False,
            data_root=root,
            render_video=False,
            continue_on_error=True,
            vllm_free_memory_fraction=0.8,
            procvlm_window_size=4,
            procvlm_max_sampled_frames=None,
            procvlm_max_new_tokens=32,
            dtype="bf16",
            tensor_parallel_size=1,
            rynn_num_frames=16,
            rynn_num_steps=16,
            rynn_evaluation_interval=4,
            rynn_batch_size=1,
            rynn_max_image_side=448,
            rynn_max_new_tokens=128,
            robot_description="robot",
            camera_description="camera",
            robo_frame_interval=4,
            robo_batch_size=1,
            robo_eval_mode="forward",
            goal_image=root / "goal.png",
            validate_environment=False,
        )

    def _fake_persistent_run(self, baseline: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self._records(root)
            (root / "goal.png").write_bytes(b"goal")
            args = self._args(root, baseline)
            plan = runner.build_worker_plan(records, args)
            config = {"python": Path(sys.executable), "repo": root}
            run_root = root / "run"
            run_root.mkdir()
            raw_root = run_root / "raw"
            logs = root / "logs"
            logs.mkdir()
            metadata = {"baseline": baseline, "status": "planning"}
            active = 0
            peak = 0
            visible: list[str] = []
            lock = threading.Lock()
            original_memory = runner.resolve_vllm_memory_budget
            original_streamed = runner.run_streamed

            def fake_memory(gpu: str, _fraction: float):
                return 0.5, {"scope": "free_gpu_memory", "gpu": gpu}

            def fake_streamed(command, _cwd, env, _log_path, **kwargs):
                nonlocal active, peak
                jobs_path = Path(command[command.index("--jobs-output-file") + 1])
                plan_path = Path(command[command.index("--jobs-file") + 1])
                specs = runner.load_jsonl(plan_path)
                with lock:
                    active += 1
                    peak = max(peak, active)
                    visible.append(env["CUDA_VISIBLE_DEVICES"])
                time.sleep(0.04)
                runner.write_jsonl_atomic(
                    jobs_path,
                    [
                        {
                            **spec,
                            "status": "complete",
                            "return_code": 0,
                            "inference_seconds": 0.04,
                            "raw_output_files": [],
                        }
                        for spec in specs
                    ],
                )
                callback = kwargs.get("progress_callback")
                if callback:
                    callback()
                with lock:
                    active -= 1
                return 0

            runner.resolve_vllm_memory_budget = fake_memory
            runner.run_streamed = fake_streamed
            try:
                result = runner.run_persistent_parallel(
                    args=args,
                    config=config,
                    model_path=root / "model",
                    run_root=run_root,
                    raw_root=raw_root,
                    log_path=logs / "run.log",
                    metadata_path=run_root / "run.json",
                    jobs_path=run_root / "jobs.jsonl",
                    commands_path=run_root / "commands.jsonl",
                    env={},
                    metadata=metadata,
                    plan=plan,
                )
            finally:
                runner.resolve_vllm_memory_budget = original_memory
                runner.run_streamed = original_streamed
            self.assertEqual(result, 0)
            final = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "complete")
            self.assertEqual(final["completed_jobs"], 4)
            self.assertEqual(len(final["worker_progress"]), 2)
            self.assertTrue(all(item["status"] == "complete" for item in final["worker_progress"]))
            self.assertGreaterEqual(peak, 2)
            self.assertEqual(set(visible), {"0", "1"})
            jobs = runner.load_jsonl(run_root / "jobs.jsonl")
            self.assertEqual(len(jobs), 4)
            self.assertEqual({job["worker_index"] for job in jobs}, {0, 1})
            commands = runner.load_jsonl(run_root / "commands.jsonl")
            self.assertEqual(len(commands), 4)

    def test_missing_worker_jobs_file_is_aggregated_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self._records(root)
            (root / "goal.png").write_bytes(b"goal")
            args = self._args(root, "robo_dopamine")
            args.parallel_workers = 1
            args.worker_spec = ["0:0:4"]
            args.gpu = "0"
            plan = runner.build_worker_plan(records, args)
            config = {"python": Path(sys.executable), "repo": root}
            run_root = root / "run"
            run_root.mkdir()
            logs = root / "logs"
            logs.mkdir()
            metadata = {"baseline": "robo_dopamine", "status": "planning"}
            original_memory = runner.resolve_vllm_memory_budget
            original_streamed = runner.run_streamed

            def fake_memory(_gpu: str, _fraction: float):
                return 0.5, {"scope": "free_gpu_memory", "gpu": _gpu}

            def fake_streamed(_command, _cwd, _env, _log_path, **kwargs):
                # Model initialization fails before the child creates its jobs file.
                callback = kwargs.get("progress_callback")
                if callback:
                    callback()
                return 70

            runner.resolve_vllm_memory_budget = fake_memory
            runner.run_streamed = fake_streamed
            try:
                result = runner.run_persistent_parallel(
                    args=args,
                    config=config,
                    model_path=root / "model",
                    run_root=run_root,
                    raw_root=run_root / "raw",
                    log_path=logs / "run.log",
                    metadata_path=run_root / "run.json",
                    jobs_path=run_root / "jobs.jsonl",
                    commands_path=run_root / "commands.jsonl",
                    env={},
                    metadata=metadata,
                    plan=plan,
                )
            finally:
                runner.resolve_vllm_memory_budget = original_memory
                runner.run_streamed = original_streamed

            self.assertEqual(result, 1)
            final = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "complete_with_errors")
            self.assertEqual(final["completed_jobs"], 0)
            self.assertEqual(final["failed_jobs"], 4)
            self.assertEqual(final["pending_jobs"], 0)
            self.assertEqual(final["worker_progress"][0]["status"], "failed")
            self.assertEqual(final["worker_progress"][0]["failed_jobs"], 4)
            jobs = runner.load_jsonl(run_root / "jobs.jsonl")
            self.assertEqual(len(jobs), 4)
            self.assertTrue(all(job["status"] == "failed" for job in jobs))
            self.assertTrue(all(job["return_code"] == 70 for job in jobs))
            self.assertNotIn("FileNotFoundError", (logs / "run.log").read_text(encoding="utf-8"))

    def test_procvlm_persistent_workers_share_aggregate_run(self):
        self._fake_persistent_run("procvlm")

    def test_robo_dopamine_persistent_workers_share_aggregate_run(self):
        self._fake_persistent_run("robo_dopamine")

    def test_safe_workers_run_concurrently_and_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self._records(root)
            args = self._args(root, "safe")
            plan = runner.build_worker_plan(records, args)
            config = {"python": Path(sys.executable), "repo": root}
            run_root = root / "run"
            run_root.mkdir()
            logs = root / "logs"
            logs.mkdir()
            metadata = {"baseline": "safe", "status": "planning"}
            active = 0
            peak = 0
            lock = threading.Lock()
            original_streamed = runner.run_streamed

            def fake_streamed(_command, _cwd, _env, _log_path, **_kwargs):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.03)
                with lock:
                    active -= 1
                return 0

            runner.run_streamed = fake_streamed
            try:
                result = runner.run_simple_parallel(
                    args=args,
                    config=config,
                    model_path=None,
                    run_root=run_root,
                    raw_root=run_root / "raw",
                    log_path=logs / "run.log",
                    metadata_path=run_root / "run.json",
                    jobs_path=run_root / "jobs.jsonl",
                    commands_path=run_root / "commands.jsonl",
                    metadata=metadata,
                    plan=plan,
                )
            finally:
                runner.run_streamed = original_streamed
            self.assertEqual(result, 0)
            final = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "complete")
            self.assertEqual(final["completed_jobs"], 4)
            self.assertGreaterEqual(peak, 2)
            self.assertEqual(len(runner.load_jsonl(run_root / "jobs.jsonl")), 4)


if __name__ == "__main__":
    unittest.main()
