#!/usr/bin/env python3
"""Contract tests for the persistent ProcVLM worker without loading a model."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from procvlm_worker import FATAL_EXIT_CODE, load_jsonl, run_persistent_jobs


def make_jobs(root: Path, count: int) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for index in range(count):
        rollout_id = f"rollout-{index}"
        output_dir = root / "raw" / rollout_id
        jobs.append({
            "job_index": index,
            "rollout_id": rollout_id,
            "video_path": str(root / f"{rollout_id}.mp4"),
            "task": f"task {index}",
            "raw_output_dir": str(output_dir),
            "output_path": str(output_dir / "procvlm_raw.jsonl"),
            "cwd": str(root),
            "argv": ["procvlm_worker.py"],
            "shell_preview": "procvlm_worker.py",
            "vllm_memory_budget": {
                "scope": "free_gpu_memory",
                "requested_free_fraction": 0.8,
                "resolved_total_fraction": 0.55,
            },
            "execution_scope": "persistent_procvlm_worker",
        })
    return jobs


def make_args(root: Path) -> argparse.Namespace:
    model_path = root / "model"
    model_path.mkdir(exist_ok=True)
    return argparse.Namespace(
        model_path=model_path,
        torch_dtype="bf16",
        vllm_total_memory_fraction=0.55,
        window_size=4,
        max_sampled_frames=None,
        max_new_tokens=128,
        tp=1,
    )


def write_raw(job: dict[str, object]) -> None:
    output_path = Path(str(job["output_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_index = int(job["job_index"]) * 10
    record = {
        "video_path": job["video_path"],
        "task": job["task"],
        "sample_index": 0,
        "frame_index": frame_index,
        "timestamp_sec": 0.5,
        "window_frame_indices": [frame_index, frame_index + 1, frame_index + 2, frame_index + 3],
        "progress": 0.25,
        "parsed_progress": {"progress": 0.25},
        "progress_source": "test",
        "reasoning": "test reasoning",
        "model_output": "test model output",
    }
    output_path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def run_worker(
    root: Path,
    jobs: list[dict[str, object]],
    initialize_engine,
    infer_one,
) -> tuple[int, Path, Path, Path]:
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
        initialize_engine=initialize_engine,
        infer_one=infer_one,
    )
    return return_code, jobs_path, progress_path, state_path


def test_one_engine_is_reused() -> None:
    with tempfile.TemporaryDirectory(prefix="procvlm-persistent-reuse-") as temporary:
        root = Path(temporary)
        jobs = make_jobs(root, 3)
        init_calls: list[object] = []
        infer_calls: list[tuple[str, object, dict[str, object]]] = []
        engine = object()

        def initialize(model_path, tp, engine_kwargs):
            init_calls.append((model_path, tp, engine_kwargs))
            return engine

        def infer(job, args, bundle, engine_kwargs):
            infer_calls.append((str(job["rollout_id"]), bundle, engine_kwargs))
            write_raw(job)

        return_code, jobs_path, progress_path, state_path = run_worker(
            root, jobs, initialize, infer
        )
        assert return_code == 0
        assert len(init_calls) == 1
        assert [call[0] for call in infer_calls] == ["rollout-0", "rollout-1", "rollout-2"]
        assert all(call[1] is engine for call in infer_calls)
        assert all(call[2]["gpu_memory_utilization"] == 0.55 for call in infer_calls)
        results = load_jsonl(jobs_path)
        assert [result["status"] for result in results] == ["complete"] * 3
        assert all("inference_seconds" in result for result in results)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["status"] == "complete"
        assert state["completed_jobs"] == 3 and state["failed_jobs"] == 0
        progress = load_jsonl(progress_path)
        assert sum(event.get("event") == "engine_initialized" for event in progress) == 1
        assert sum(event.get("event") == "rollout_finished" for event in progress) == 3
        raw = json.loads(Path(results[0]["raw_output_files"][0]).read_text(encoding="utf-8"))
        assert raw["frame_index"] == 0
        assert raw["window_frame_indices"] == [0, 1, 2, 3]


def test_recoverable_error_does_not_stop_later_jobs() -> None:
    with tempfile.TemporaryDirectory(prefix="procvlm-persistent-recoverable-") as temporary:
        root = Path(temporary)
        jobs = make_jobs(root, 3)
        init_calls: list[object] = []
        infer_ids: list[str] = []

        def initialize(model_path, tp, engine_kwargs):
            init_calls.append(engine_kwargs)
            return object()

        def infer(job, args, bundle, engine_kwargs):
            rollout_id = str(job["rollout_id"])
            infer_ids.append(rollout_id)
            if rollout_id == "rollout-1":
                raise ValueError("recoverable parser failure")
            write_raw(job)

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
    with tempfile.TemporaryDirectory(prefix="procvlm-persistent-fatal-") as temporary:
        root = Path(temporary)
        jobs = make_jobs(root, 3)
        init_calls: list[object] = []
        first_pass_ids: list[str] = []

        def initialize(model_path, tp, engine_kwargs):
            init_calls.append(engine_kwargs)
            return object()

        def fatal_infer(job, args, bundle, engine_kwargs):
            rollout_id = str(job["rollout_id"])
            first_pass_ids.append(rollout_id)
            if rollout_id == "rollout-1":
                raise RuntimeError("EngineCore encountered an issue")
            write_raw(job)

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
        assert any(event.get("event") == "fatal_engine_failure" for event in load_jsonl(progress_path))

        resumed_ids: list[str] = []

        def resume_infer(job, args, bundle, engine_kwargs):
            resumed_ids.append(str(job["rollout_id"]))
            write_raw(job)

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
    with tempfile.TemporaryDirectory(prefix="procvlm-persistent-init-fatal-") as temporary:
        root = Path(temporary)
        jobs = make_jobs(root, 2)

        def initialize(model_path, tp, engine_kwargs):
            raise RuntimeError("CUDA out of memory while creating engine")

        def infer_never_called(job, args, bundle, engine_kwargs):
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
    print("PROCVLM_PERSISTENT_WORKER_TESTS_OK")


if __name__ == "__main__":
    main()
