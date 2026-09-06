from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path


RUNNER_PATH = Path(__file__).with_name("run_lf3r_baseline.py")
SPEC = importlib.util.spec_from_file_location("lf3r_parallel_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RynnParallelRunnerTest(unittest.TestCase):
    def _args(self, root: Path, **overrides):
        values = {
            "start_index": 0,
            "end_index": 8,
            "limit": None,
            "parallel_workers": 1,
            "worker_spec": [],
            "gpu": "0",
            "dry_run": False,
            "data_root": root,
            "rynn_evaluation_interval": 4,
            "rynn_num_frames": 16,
            "rynn_batch_size": 1,
            "rynn_max_image_side": 448,
            "rynn_max_new_tokens": 128,
            "robot_description": "robot",
            "camera_description": "camera",
            "render_video": False,
            "baseline": "rynnvalue",
            "continue_on_error": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def _records(self, root: Path, count: int = 8):
        records = []
        for index in range(count):
            video = root / f"r{index}.mp4"
            video.write_bytes(b"video")
            records.append({
                "id": f"r{index}",
                "video_path": str(video),
                "task_description": "test task",
                "total_frames": 20,
            })
        return records

    def test_auto_split_is_contiguous_right_open_and_reuses_gpu_ids(self):
        records = [{"id": f"r{index}"} for index in range(10)]
        args = self._args(Path("."), start_index=2, end_index=9, parallel_workers=3, gpu="0,1")
        plan = runner.build_rynn_worker_plan(records, args)
        self.assertEqual(
            [(item["start_index"], item["end_index"], item["gpu"]) for item in plan["workers"]],
            [(2, 5, "0"), (5, 7, "1"), (7, 9, "0")],
        )
        self.assertEqual(plan["unique_count"], 7)
        self.assertEqual(plan["overlaps"], [])
        self.assertEqual(plan["gaps"], [])

    def test_manual_overlap_and_gap_are_reported_and_first_owner_wins(self):
        records = [{"id": f"r{index}"} for index in range(8)]
        args = self._args(
            Path("."),
            end_index=8,
            parallel_workers=3,
            worker_spec=["0:0:3", "0:2:5", "1:6:8"],
        )
        plan = runner.build_rynn_worker_plan(records, args)
        self.assertEqual(plan["overlaps"], [2])
        self.assertEqual(plan["gaps"], [5])
        self.assertEqual(plan["unique_count"], 7)
        self.assertEqual(plan["workers"][1]["duplicate_count"], 1)

    def test_fake_workers_run_concurrently_with_independent_cuda_visibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self._records(root, 6)
            args = self._args(
                root,
                end_index=6,
                parallel_workers=2,
                worker_spec=["0:0:3", "1:3:6"],
            )
            plan = runner.build_rynn_worker_plan(records, args)
            active = 0
            peak = 0
            lock = threading.Lock()
            visible = []

            runner.make_execution_environment = lambda _config, gpu: {
                "CUDA_VISIBLE_DEVICES": str(gpu)
            }
            runner.command_for = lambda _args, _config, record, _job_dir, _model, _fraction: (
                ["fake", record["id"]], root
            )

            def fake_run(command, _cwd, env, _log_path, **_kwargs):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                    visible.append(env["CUDA_VISIBLE_DEVICES"])
                time.sleep(0.03)
                with lock:
                    active -= 1
                return 0

            runner.run_streamed = fake_run
            run_root = root / "run"
            run_root.mkdir()
            log_root = root / "logs"
            log_root.mkdir()
            metadata = {"baseline": "rynnvalue", "status": "planning"}
            result = runner.run_rynn_parallel(
                args=args,
                config={"repo": root},
                scope_records=records,
                model_path=root,
                run_root=run_root,
                raw_root=run_root / "raw",
                log_path=log_root / "run.log",
                metadata_path=run_root / "run.json",
                jobs_path=run_root / "jobs.jsonl",
                commands_path=run_root / "commands.jsonl",
                metadata=metadata,
                plan=plan,
            )
            self.assertEqual(result, 0)
            final = json.loads((run_root / "run.json").read_text())
            self.assertEqual(final["status"], "complete")
            self.assertEqual(final["completed_jobs"], 6)
            self.assertGreaterEqual(peak, 2)
            self.assertEqual(set(visible), {"0", "1"})
            jobs = [json.loads(line) for line in (run_root / "jobs.jsonl").read_text().splitlines()]
            self.assertEqual(len(jobs), 6)
            self.assertEqual({job["worker_index"] for job in jobs}, {0, 1})

    def test_one_failed_rollout_does_not_stop_other_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self._records(root, 6)
            args = self._args(root, end_index=6, parallel_workers=2, worker_spec=["0:0:3", "1:3:6"])
            plan = runner.build_rynn_worker_plan(records, args)
            runner.make_execution_environment = lambda _config, gpu: {"CUDA_VISIBLE_DEVICES": str(gpu)}
            runner.command_for = lambda _args, _config, record, _job_dir, _model, _fraction: (["fake", record["id"]], root)
            runner.run_streamed = lambda command, _cwd, _env, _log_path, **_kwargs: 1 if command[1] == "r1" else 0
            run_root = root / "run"
            run_root.mkdir()
            log_root = root / "logs"
            log_root.mkdir()
            metadata = {"baseline": "rynnvalue", "status": "planning"}
            result = runner.run_rynn_parallel(
                args=args,
                config={"repo": root},
                scope_records=records,
                model_path=root,
                run_root=run_root,
                raw_root=run_root / "raw",
                log_path=log_root / "run.log",
                metadata_path=run_root / "run.json",
                jobs_path=run_root / "jobs.jsonl",
                commands_path=run_root / "commands.jsonl",
                metadata=metadata,
                plan=plan,
            )
            self.assertEqual(result, 1)
            final = json.loads((run_root / "run.json").read_text())
            self.assertEqual(final["status"], "complete_with_errors")
            self.assertEqual(final["completed_jobs"], 5)
            self.assertEqual(final["failed_jobs"], 1)


if __name__ == "__main__":
    unittest.main()
