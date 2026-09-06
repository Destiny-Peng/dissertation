#!/usr/bin/env python3
"""Contract tests for the persistent Robo-Dopamine worker without a model."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from procvlm_worker import FATAL_EXIT_CODE, load_jsonl
from robo_dopamine_persistent_worker import run_persistent_jobs


def make_jobs(root: Path, count: int) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for index in range(count):
        rollout_id = f"rollout-{index}"
        output_dir = root / "raw" / rollout_id
        jobs.append(
            {
                "job_index": index,
                "rollout_id": rollout_id,
                "video_path": str(root / f"{rollout_id}.mp4"),
                "task": f"task {index}",
                "raw_output_dir": str(output_dir),
                "goal_image": str(root / "goal.png"),
                "frame_interval": 4,
                "batch_size": 1,
                "eval_mode": "forward",
                "cwd": str(root),
                "argv": ["robo_dopamine_persistent_worker.py"],
                "shell_preview": "robo_dopamine_persistent_worker.py",
                "vllm_memory_budget": {
                    "scope": "free_gpu_memory",
                    "requested_free_fraction": 0.8,
                    "resolved_total_fraction": 0.55,
                },
                "execution_scope": "persistent_robo_dopamine_worker",
            }
        )
    return jobs


def make_args(root: Path) -> argparse.Namespace:
    model_path = root / "model"
    model_path.mkdir(exist_ok=True)
    return argparse.Namespace(
        repo=root,
        model_path=model_path,
        goal_image=root / "goal.png",
        frame_interval=4,
        batch_size=1,
        eval_mode="forward",
        render_video=False,
        vllm_total_memory_fraction=0.55,
    )


def write_raw(job: dict[str, object]) -> dict[str, str]:
    output_dir = Path(str(job["raw_output_dir"]))
    official_output = output_dir / "fake_official_run"
    official_output.mkdir(parents=True, exist_ok=True)
    prediction = official_output / "pred_vllm.json"
    prediction.write_text(json.dumps([{"pred": "<score>+25%</score>"}]), encoding="utf-8")
    worker_result = output_dir / "worker_result.json"
    worker_result.write_text(json.dumps({"raw_model_output": str(prediction)}), encoding="utf-8")
    return {
        "official_output_dir": str(official_output),
        "raw_model_output": str(prediction),
        "worker_result_path": str(worker_result),
    }


def run_worker(root: Path, jobs: list[dict[str, object]], initialize, infer):
    jobs_path = root / "jobs.jsonl"
    progress_path = root / "progress.jsonl"
    state_path = root / "state.json"
    return_code = run_persistent_jobs(
        jobs,
        make_args(root),
        jobs_output_path=jobs_path,
        progress_path=progress_path,
        state_path=state_path,
        engine_memory_budget={
            "scope": "free_gpu_memory",
            "requested_free_fraction": 0.8,
            "resolved_total_fraction": 0.55,
        },
        initialize_model=initialize,
        infer_one=infer,
    )
    return return_code, jobs_path, progress_path, state_path


def test_one_engine_is_reused() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-persistent-reuse-") as temporary:
        root = Path(temporary)
        (root / "goal.png").touch()
        jobs = make_jobs(root, 3)
        init_calls: list[object] = []
        infer_calls: list[tuple[str, object]] = []
        model = object()

        def initialize(args):
            init_calls.append(args)
            return model

        def infer(job, args, initialized):
            infer_calls.append((str(job["rollout_id"]), initialized))
            return write_raw(job)

        return_code, jobs_path, progress_path, state_path = run_worker(
            root, jobs, initialize, infer
        )
        assert return_code == 0
        assert len(init_calls) == 1
        assert [item[0] for item in infer_calls] == [
            "rollout-0",
            "rollout-1",
            "rollout-2",
        ]
        assert all(item[1] is model for item in infer_calls)
        results = load_jsonl(jobs_path)
        assert [result["status"] for result in results] == ["complete"] * 3
        assert all(Path(result["raw_model_output"]).is_file() for result in results)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["status"] == "complete"
        assert state["completed_jobs"] == 3 and state["failed_jobs"] == 0
        progress = load_jsonl(progress_path)
        assert sum(event.get("event") == "engine_initialized" for event in progress) == 1
        assert sum(event.get("event") == "rollout_finished" for event in progress) == 3


def test_recoverable_error_does_not_stop_later_jobs() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-persistent-recoverable-") as temporary:
        root = Path(temporary)
        (root / "goal.png").touch()
        jobs = make_jobs(root, 3)
        init_calls: list[object] = []
        infer_ids: list[str] = []

        def initialize(args):
            init_calls.append(args)
            return object()

        def infer(job, args, model):
            rollout_id = str(job["rollout_id"])
            infer_ids.append(rollout_id)
            if rollout_id == "rollout-1":
                raise ValueError("recoverable parser failure")
            return write_raw(job)

        return_code, jobs_path, _, state_path = run_worker(
            root, jobs, initialize, infer
        )
        assert return_code == 0
        assert len(init_calls) == 1
        assert infer_ids == ["rollout-0", "rollout-1", "rollout-2"]
        results = {result["rollout_id"]: result for result in load_jsonl(jobs_path)}
        assert results["rollout-0"]["status"] == "complete"
        assert results["rollout-1"]["status"] == "failed"
        assert results["rollout-2"]["status"] == "complete"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["status"] == "complete"
        assert state["completed_jobs"] == 2 and state["failed_jobs"] == 1


def test_fatal_error_saves_progress_and_resume_skips_completed_jobs() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-persistent-fatal-") as temporary:
        root = Path(temporary)
        (root / "goal.png").touch()
        jobs = make_jobs(root, 3)
        init_calls: list[object] = []
        first_pass_ids: list[str] = []

        def initialize(args):
            init_calls.append(args)
            return object()

        def fatal_infer(job, args, model):
            rollout_id = str(job["rollout_id"])
            first_pass_ids.append(rollout_id)
            if rollout_id == "rollout-1":
                raise RuntimeError("EngineCore encountered an issue")
            return write_raw(job)

        return_code, jobs_path, progress_path, state_path = run_worker(
            root, jobs, initialize, fatal_infer
        )
        assert return_code == FATAL_EXIT_CODE
        partial = {result["rollout_id"]: result for result in load_jsonl(jobs_path)}
        assert partial["rollout-0"]["status"] == "complete"
        assert partial["rollout-1"]["status"] == "interrupted"
        assert "rollout-2" not in partial
        fatal_state = json.loads(state_path.read_text(encoding="utf-8"))
        assert fatal_state["status"] == "fatal_engine_failure"
        assert fatal_state["pending_jobs"] == 2
        assert any(
            event.get("event") == "fatal_engine_failure"
            for event in load_jsonl(progress_path)
        )

        resumed_ids: list[str] = []

        def resume_infer(job, args, model):
            resumed_ids.append(str(job["rollout_id"]))
            return write_raw(job)

        resumed_code, _, _, resumed_state_path = run_worker(
            root, jobs, initialize, resume_infer
        )
        assert resumed_code == 0
        assert len(init_calls) == 2
        assert resumed_ids == ["rollout-1", "rollout-2"]
        final = {result["rollout_id"]: result for result in load_jsonl(jobs_path)}
        assert set(final) == {"rollout-0", "rollout-1", "rollout-2"}
        assert all(result["status"] == "complete" for result in final.values())
        state = json.loads(resumed_state_path.read_text(encoding="utf-8"))
        assert state["status"] == "complete"
        assert state["completed_jobs"] == 3 and state["pending_jobs"] == 0
        progress = load_jsonl(progress_path)
        assert sum(event.get("event") == "engine_initialized" for event in progress) == 2


def test_fatal_initialization_failure_saves_pending_state() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-persistent-init-fatal-") as temporary:
        root = Path(temporary)
        (root / "goal.png").touch()
        jobs = make_jobs(root, 2)

        def initialize(args):
            raise RuntimeError("CUDA out of memory while creating engine")

        def infer_never_called(job, args, model):
            raise AssertionError("inference must not run after init failure")

        return_code, jobs_path, progress_path, state_path = run_worker(
            root, jobs, initialize, infer_never_called
        )
        assert return_code == FATAL_EXIT_CODE
        assert not jobs_path.exists()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["status"] == "fatal_engine_failure"
        assert state["pending_jobs"] == 2
        events = load_jsonl(progress_path)
        assert events[-1]["event"] == "fatal_engine_failure"
        assert events[-1]["phase"] == "initialization"


def main() -> None:
    test_one_engine_is_reused()
    test_recoverable_error_does_not_stop_later_jobs()
    test_fatal_error_saves_progress_and_resume_skips_completed_jobs()
    test_fatal_initialization_failure_saves_pending_state()
    print("ROBODOPAMINE_PERSISTENT_WORKER_TESTS_OK")


if __name__ == "__main__":
    main()
