#!/usr/bin/env python3
"""Run ProcVLM rollouts sequentially with one persistent vLLM engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


FATAL_EXIT_CODE = 70
TERMINAL_JOB_STATUSES = {"complete", "failed"}
VLLM_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
}


def iso_now() -> str:
    return datetime.now().astimezone().isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rendered = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def upsert_job(path: Path, job_result: dict[str, Any]) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    rollout_id = job_result["rollout_id"]
    for index, record in enumerate(records):
        if record.get("rollout_id") == rollout_id:
            records[index] = job_result
            break
    else:
        records.append(job_result)
    write_jsonl_atomic(path, records)
    return records


def job_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "completed_jobs": sum(record.get("status") == "complete" for record in records),
        "failed_jobs": sum(record.get("status") == "failed" for record in records),
        "interrupted_jobs": sum(record.get("status") == "interrupted" for record in records),
    }


def is_fatal_engine_failure(error: BaseException) -> bool:
    """Classify failures that make the persistent engine unsafe to reuse."""
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return True
    try:
        import torch

        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass

    error_name = type(error).__name__
    if error_name in {"EngineDeadError", "AsyncEngineDeadError"}:
        return True

    details = "\n".join(
        (
            error_name,
            str(error),
            traceback.format_exc(),
        )
    ).lower()
    fatal_markers = (
        "enginecore encountered an issue",
        "engine core died",
        "engine dead",
        "engine is dead",
        "engine process died",
        "worker process died",
        "worker died",
        "cuda out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "outofmemoryerror",
    )
    return any(marker in details for marker in fatal_markers)


def cleanup_after_recoverable_failure() -> None:
    """Release transient Python/CUDA allocations before reusing the engine."""
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Cleanup is best effort; the original inference error is authoritative.
        pass


def initialize_vllm_engine(model_path: str, tp: int, engine_kwargs: dict[str, Any]) -> Any:
    """Initialize the upstream cached vLLM engine and processor exactly once."""
    from core.backends.vllm import get_vllm

    return get_vllm(model_path, tp, dict(engine_kwargs))


def infer_rollout(
    job: dict[str, Any],
    args: argparse.Namespace,
    engine_bundle: Any,
    engine_kwargs: dict[str, Any],
) -> Any:
    """Run one rollout through the already-initialized upstream cache."""
    del engine_bundle
    from evqa.inference import infer_progress_from_video

    max_sampled_frames = (
        512 if args.max_sampled_frames is None else args.max_sampled_frames
    )
    return infer_progress_from_video(
        video_path=job["video_path"],
        task=job["task"],
        model_path=str(args.model_path),
        output_path=job["output_path"],
        window_size=args.window_size,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        max_sampled_frames=max_sampled_frames,
        tp=args.tp,
        engine_kwargs=engine_kwargs,
        show_progress=True,
    )


def refresh_state(
    state_path: Path,
    *,
    status: str,
    total_jobs: int,
    records: list[dict[str, Any]],
    engine_initialization_seconds: float | None,
    engine_memory_budget: dict[str, Any],
    last_rollout_id: str | None = None,
    fatal_error: dict[str, Any] | None = None,
) -> None:
    counts = job_counts(records)
    state = {
        "schema_version": 1,
        "status": status,
        "updated_at": iso_now(),
        "total_jobs": total_jobs,
        **counts,
        "pending_jobs": total_jobs - counts["completed_jobs"] - counts["failed_jobs"],
        "engine_initialization_seconds": engine_initialization_seconds,
        "engine_memory_budget": engine_memory_budget,
        "last_rollout_id": last_rollout_id,
    }
    if fatal_error is not None:
        state["fatal_error"] = fatal_error
    atomic_json(state_path, state)


def command_metadata(job: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "job_index",
        "rollout_id",
        "cwd",
        "argv",
        "shell_preview",
        "vllm_memory_budget",
        "execution_scope",
    )
    return {key: job[key] for key in keys if key in job}


def run_persistent_jobs(
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    jobs_output_path: Path,
    progress_path: Path,
    state_path: Path,
    engine_memory_budget: dict[str, Any],
    initialize_engine: Callable[[str, int, dict[str, Any]], Any] = initialize_vllm_engine,
    infer_one: Callable[[dict[str, Any], argparse.Namespace, Any, dict[str, Any]], Any] = infer_rollout,
) -> int:
    """Initialize one engine, then process pending jobs in manifest order."""
    existing = load_jsonl(jobs_output_path)
    existing_by_id = {record.get("rollout_id"): record for record in existing}
    pending = [
        job
        for job in jobs
        if existing_by_id.get(job.get("rollout_id"), {}).get("status") not in TERMINAL_JOB_STATUSES
    ]

    engine_kwargs = {
        "dtype": VLLM_DTYPE_ALIASES.get(str(args.torch_dtype).lower(), args.torch_dtype),
        "gpu_memory_utilization": args.vllm_total_memory_fraction,
    }
    if not pending:
        refresh_state(
            state_path,
            status="no_pending_jobs",
            total_jobs=len(jobs),
            records=existing,
            engine_initialization_seconds=None,
            engine_memory_budget=engine_memory_budget,
        )
        append_jsonl(
            progress_path,
            {
                "event": "worker_complete",
                "status": "no_pending_jobs",
                "at": iso_now(),
                **job_counts(existing),
            },
        )
        return 0

    append_jsonl(
        progress_path,
        {
            "event": "engine_initialization_started",
            "at": iso_now(),
            "pending_jobs": len(pending),
            "total_jobs": len(jobs),
            "memory_budget": engine_memory_budget,
        },
    )
    init_started = time.perf_counter()
    try:
        engine_bundle = initialize_engine(
            str(args.model_path), args.tp, engine_kwargs
        )
    except BaseException as error:
        init_seconds = time.perf_counter() - init_started
        fatal = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        refresh_state(
            state_path,
            status="fatal_engine_failure",
            total_jobs=len(jobs),
            records=existing,
            engine_initialization_seconds=init_seconds,
            engine_memory_budget=engine_memory_budget,
            fatal_error=fatal,
        )
        append_jsonl(
            progress_path,
            {
                "event": "fatal_engine_failure",
                "phase": "initialization",
                "at": iso_now(),
                "initialization_seconds": init_seconds,
                "memory_budget": engine_memory_budget,
                **fatal,
            },
        )
        return FATAL_EXIT_CODE

    init_seconds = time.perf_counter() - init_started
    append_jsonl(
        progress_path,
        {
            "event": "engine_initialized",
            "at": iso_now(),
            "initialization_seconds": init_seconds,
            "memory_budget": engine_memory_budget,
            "engine_reused_for_jobs": len(pending),
        },
    )
    refresh_state(
        state_path,
        status="running",
        total_jobs=len(jobs),
        records=existing,
        engine_initialization_seconds=init_seconds,
        engine_memory_budget=engine_memory_budget,
    )

    for job in pending:
        rollout_id = str(job["rollout_id"])
        output_dir = Path(job["raw_output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        started_at = iso_now()
        started = time.perf_counter()
        append_jsonl(
            progress_path,
            {
                "event": "rollout_started",
                "at": started_at,
                "job_index": job.get("job_index"),
                "rollout_id": rollout_id,
            },
        )
        try:
            infer_one(job, args, engine_bundle, engine_kwargs)
            output_path = Path(job.get("output_path", ""))
            if not output_path.is_file():
                raise RuntimeError(f"ProcVLM produced no raw output file: {output_path}")
            return_code = 0
            status = "complete"
            error_details: dict[str, Any] = {}
        except BaseException as error:
            return_code = FATAL_EXIT_CODE if is_fatal_engine_failure(error) else 1
            status = "interrupted" if return_code == FATAL_EXIT_CODE else "failed"
            error_details = {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }

        elapsed = time.perf_counter() - started
        raw_files = sorted(
            str(path) for path in output_dir.rglob("*") if path.is_file()
        )
        result = {
            **command_metadata(job),
            "status": status,
            "return_code": return_code,
            "started_at": started_at,
            "completed_at": iso_now(),
            "inference_seconds": elapsed,
            "raw_output_dir": str(output_dir),
            "raw_output_files": raw_files,
        }
        result.update(error_details)
        if status == "interrupted":
            result["resume_required"] = True
        records = upsert_job(jobs_output_path, result)
        append_jsonl(
            progress_path,
            {
                "event": "rollout_finished",
                "at": result["completed_at"],
                "job_index": job.get("job_index"),
                "rollout_id": rollout_id,
                "status": status,
                "inference_seconds": elapsed,
                **error_details,
            },
        )
        refresh_state(
            state_path,
            status="fatal_engine_failure" if status == "interrupted" else "running",
            total_jobs=len(jobs),
            records=records,
            engine_initialization_seconds=init_seconds,
            engine_memory_budget=engine_memory_budget,
            last_rollout_id=rollout_id,
            fatal_error=error_details if status == "interrupted" else None,
        )
        if status == "interrupted":
            append_jsonl(
                progress_path,
                {
                    "event": "fatal_engine_failure",
                    "phase": "rollout",
                    "at": iso_now(),
                    "rollout_id": rollout_id,
                    "memory_budget": engine_memory_budget,
                    **error_details,
                },
            )
            return FATAL_EXIT_CODE
        if status == "failed":
            cleanup_after_recoverable_failure()

    records = load_jsonl(jobs_output_path)
    counts = job_counts(records)
    refresh_state(
        state_path,
        status="complete",
        total_jobs=len(jobs),
        records=records,
        engine_initialization_seconds=init_seconds,
        engine_memory_budget=engine_memory_budget,
        last_rollout_id=records[-1].get("rollout_id") if records else None,
    )
    append_jsonl(
        progress_path,
        {
            "event": "worker_complete",
            "status": "complete",
            "at": iso_now(),
            "engine_initialization_seconds": init_seconds,
            **counts,
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--jobs-output-file", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--vllm-total-memory-fraction", type=float, default=None)
    parser.add_argument("--vllm-free-memory-fraction", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--memory-budget-json", required=True)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--max-sampled-frames", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--torch-dtype", default="bf16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("window_size", "max_new_tokens", "tp"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_sampled_frames is not None and args.max_sampled_frames < 1:
        parser.error("--max-sampled-frames must be positive when provided")
    if args.dry_run:
        if args.vllm_free_memory_fraction is not None and not 0.0 < args.vllm_free_memory_fraction <= 1.0:
            parser.error("--vllm-free-memory-fraction must be in (0, 1]")
    else:
        if args.vllm_total_memory_fraction is None:
            parser.error("--vllm-total-memory-fraction is required unless --dry-run is used")
        if not 0.0 < args.vllm_total_memory_fraction <= 1.0:
            parser.error("--vllm-total-memory-fraction must be in (0, 1]")
    try:
        args.memory_budget = json.loads(args.memory_budget_json)
    except json.JSONDecodeError as error:
        parser.error(f"--memory-budget-json is invalid JSON: {error}")
    if not isinstance(args.memory_budget, dict):
        parser.error("--memory-budget-json must contain an object")
    return args


def main() -> int:
    args = parse_args()
    args.jobs_file = args.jobs_file.expanduser().resolve()
    args.jobs_output_file = args.jobs_output_file.expanduser().resolve()
    args.progress_file = args.progress_file.expanduser().resolve()
    args.state_file = args.state_file.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    jobs = load_jsonl(args.jobs_file)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "jobs_file": str(args.jobs_file),
            "jobs": len(jobs),
            "requested_free_memory_fraction": args.vllm_free_memory_fraction,
        }, ensure_ascii=False))
        return 0
    if not jobs:
        raise ValueError(f"No ProcVLM jobs found in {args.jobs_file}")
    return run_persistent_jobs(
        jobs,
        args,
        jobs_output_path=args.jobs_output_file,
        progress_path=args.progress_file,
        state_path=args.state_file,
        engine_memory_budget=args.memory_budget,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
