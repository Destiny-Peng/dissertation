#!/usr/bin/env python3
"""Run ProcVLM rollouts with persistent vLLM or official PyTorch/LoRA inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


def is_lora_adapter_path(model_path: Path) -> bool:
    return model_path.is_dir() and (model_path / "adapter_config.json").is_file()


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


def infer_procedure_rollout(
    job: dict[str, Any],
    args: argparse.Namespace,
    engine_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run external ProcVLM procedure tracking without changing upstream model code."""
    from evqa.inference import (
        build_uniform_indices,
        build_window_values,
        extract_frames_moviepy,
        extract_progress,
        extract_reasoning_text,
        load_prompt_template,
        run_batch_progress_inference,
        write_jsonl,
    )
    from procvlm_procedure_state import (
        StatefulProcedureTracker,
        build_stateful_history_prompt,
        load_procedure,
        normalize_text,
        parse_remaining_actions,
        task_history_text,
    )

    mode = str(args.procedure_mode)
    if mode not in {"tracker_only", "stateful_history"}:
        raise ValueError(
            "procedure inference requires tracker_only/stateful_history mode, "
            f"got {mode!r}"
        )
    config_value = job.get("procedure_config") or args.procedure_config
    if not config_value:
        raise ValueError(f"--procedure-config is required for procedure mode {mode}")
    procedure_path = Path(config_value).expanduser().resolve()
    procedure = load_procedure(procedure_path)
    job_task = str(job["task"])
    if normalize_text(procedure.task) != normalize_text(job_task):
        raise ValueError(
            "procedure config task does not match rollout task: "
            f"{procedure.task!r} != {job_task!r}"
        )

    video_path = Path(job["video_path"]).expanduser().resolve()
    output_path = Path(job["output_path"]).expanduser().resolve()
    frame_dir = output_path.parent / "_procedure_frames"
    all_frame_paths, fps = extract_frames_moviepy(str(video_path), frame_dir)
    source_frames = len(all_frame_paths)
    max_sampled_frames = 512 if args.max_sampled_frames is None else int(args.max_sampled_frames)
    sampled_count = min(source_frames, max(max_sampled_frames, 1))
    selected_indices = build_uniform_indices(source_frames, sampled_count)
    if not selected_indices:
        raise RuntimeError("no frames selected for procedure inference")

    source_frame_indices = list(range(source_frames))
    official_prompt = load_prompt_template(job_task)
    tracker = StatefulProcedureTracker(
        procedure,
        support_threshold=args.tracker_support_threshold,
        window_size=args.tracker_window_size,
        max_forward_jump=args.tracker_max_forward_jump,
    )
    last_progress = 0.0
    records: list[dict[str, Any]] = []

    def window_for(frame_index: int) -> tuple[list[str], list[int]]:
        window_paths = build_window_values(
            all_frame_paths,
            frame_index,
            max(1, args.window_size),
            args.frame_stride,
        )
        window_indices = [
            int(value)
            for value in build_window_values(
                source_frame_indices,
                frame_index,
                max(1, args.window_size),
                args.frame_stride,
            )
        ]
        return window_paths, window_indices

    def run_items(items: list[dict[str, Any]]) -> list[str]:
        answers = run_batch_progress_inference(
            batch_items=items,
            model_path=str(args.model_path),
            max_new_tokens=args.max_new_tokens,
            enable_value_head=getattr(args, "enable_value_head", False),
            use_lora=getattr(args, "use_lora", False),
            torch_dtype=args.torch_dtype,
            tp=args.tp,
            engine_kwargs=engine_kwargs,
        )
        return [str(answer) for answer in answers]

    def append_record(
        *,
        sample_index: int,
        frame_index: int,
        window_indices: list[int],
        answer: str,
        prompt_history: str,
    ) -> None:
        nonlocal last_progress
        parsed = parse_remaining_actions(answer, procedure)
        tracker_row = tracker.update(int(frame_index), parsed)
        parsed_progress = extract_progress(answer)
        if parsed_progress is not None:
            last_progress = parsed_progress
        records.append({
            "video_path": str(video_path),
            "task": job["task"],
            "sample_index": sample_index,
            "frame_index": int(frame_index),
            "timestamp_sec": float(frame_index) / max(float(fps), 1e-6),
            "window_frame_indices": window_indices,
            "frame_stride": int(args.frame_stride),
            "procedure_mode": mode,
            "procedure_task_id": procedure.task_id,
            "procedure_config": str(procedure_path),
            "raw_reasoning": extract_reasoning_text(answer),
            "reasoning": extract_reasoning_text(answer),
            "model_output": answer,
            "parsed_actions": list(parsed.parsed_actions),
            "canonical_remaining_ids": list(parsed.remaining_ids),
            "parsed_remaining_ids": list(parsed.remaining_ids),
            "parse_valid": bool(parsed.parse_valid),
            "parse_source": parsed.source,
            "parse_errors": list(parsed.errors),
            "observed_stage": parsed.observed_stage,
            "observed_state": parsed.observed_stage,
            "persistent_stage": tracker_row["persistent_stage"],
            "persistent_state": tracker_row["persistent_stage"],
            "confirmed_stage": tracker_row["persistent_stage"],
            "transition_support": tracker_row["transition_support"],
            "state_update": tracker_row["state_update"],
            "transition_event": tracker_row["transition_event"],
            "transition_events": tracker_row["transition_events"],
            "task_history_text": prompt_history,
            "next_task_history_text": task_history_text(
                procedure, tracker_row["persistent_stage"]
            ),
            "progress": float(last_progress),
            "parsed_progress": parsed_progress,
            "progress_source": "model" if parsed_progress is not None else "previous",
        })

    try:
        if mode == "tracker_only":
            batch_items: list[dict[str, Any]] = []
            windows: list[list[int]] = []
            for frame_index in selected_indices:
                window_paths, window_indices = window_for(frame_index)
                windows.append(window_indices)
                batch_items.append({
                    "image": window_paths,
                    "conversations": [{"from": "human", "value": official_prompt}],
                })
            answers = run_items(batch_items)
            if len(answers) != len(selected_indices):
                raise RuntimeError(
                    f"ProcVLM returned {len(answers)} answers for "
                    f"{len(selected_indices)} tracker-only items"
                )
            for sample_index, (frame_index, window_indices, answer) in enumerate(
                zip(selected_indices, windows, answers)
            ):
                append_record(
                    sample_index=sample_index,
                    frame_index=int(frame_index),
                    window_indices=window_indices,
                    answer=answer,
                    prompt_history="",
                )
        else:
            for sample_index, frame_index in enumerate(selected_indices):
                prompt_history = task_history_text(procedure, tracker.persistent_stage)
                prompt = build_stateful_history_prompt(job_task, prompt_history)
                window_paths, window_indices = window_for(frame_index)
                answers = run_items([{
                    "image": window_paths,
                    "conversations": [{"from": "human", "value": prompt}],
                }])
                if len(answers) != 1:
                    raise RuntimeError(
                        f"ProcVLM returned {len(answers)} answers for one stateful-history item"
                    )
                append_record(
                    sample_index=sample_index,
                    frame_index=int(frame_index),
                    window_indices=window_indices,
                    answer=answers[0],
                    prompt_history=prompt_history,
                )

        write_jsonl(records, output_path)
        return records
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def infer_rollout(
    job: dict[str, Any],
    args: argparse.Namespace,
    engine_bundle: Any,
    engine_kwargs: dict[str, Any],
) -> Any:
    """Run one rollout through the already-initialized upstream cache."""
    del engine_bundle
    if getattr(args, "procedure_mode", "baseline") != "baseline":
        return infer_procedure_rollout(job, args, engine_kwargs)

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
        frame_stride=getattr(args, "frame_stride", 1),
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        max_sampled_frames=max_sampled_frames,
        tp=args.tp,
        engine_kwargs=engine_kwargs,
        show_progress=True,
        enable_value_head=getattr(args, "enable_value_head", False),
        use_lora=getattr(args, "use_lora", False),
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
            "use_lora": bool(getattr(args, "use_lora", False)),
        },
    )
    init_started = time.perf_counter()
    try:
        if getattr(args, "enable_value_head", False) or getattr(args, "use_lora", False):
            from evqa.model import load_procvlm

            engine_bundle = load_procvlm(str(args.model_path), "cuda:0", args.torch_dtype)
        else:
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
            "use_lora": bool(getattr(args, "use_lora", False)),
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
            "use_lora": bool(getattr(args, "use_lora", False)),
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
            "use_lora": bool(getattr(args, "use_lora", False)),
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
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-sampled-frames", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--procedure-mode",
        choices=("baseline", "canonical", "stateful"),
        default="baseline",
    )
    parser.add_argument("--procedure-config", type=Path, default=None)
    parser.add_argument("--tracker-decision-interval-frames", type=int, default=3)
    parser.add_argument("--tracker-forward-votes", type=int, default=3)
    parser.add_argument("--tracker-forward-window", type=int, default=4)
    parser.add_argument("--tracker-forward-min-span-sec", type=float, default=0.2)
    parser.add_argument("--tracker-completion-votes", type=int, default=4)
    parser.add_argument("--tracker-completion-window", type=int, default=5)
    parser.add_argument("--tracker-completion-min-span-sec", type=float, default=0.3)
    parser.add_argument("--tracker-candidate-timeout-sec", type=float, default=0.5)
    parser.add_argument("--tracker-max-forward-jump", type=int, default=1)
    parser.add_argument("--torch-dtype", default="bf16")
    parser.add_argument("--enable-value-head", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in (
        "window_size",
        "frame_stride",
        "max_new_tokens",
        "tp",
        "tracker_decision_interval_frames",
        "tracker_forward_votes",
        "tracker_forward_window",
        "tracker_completion_votes",
        "tracker_completion_window",
        "tracker_max_forward_jump",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.tracker_forward_votes > args.tracker_forward_window:
        parser.error("--tracker-forward-votes cannot exceed --tracker-forward-window")
    if args.tracker_completion_votes > args.tracker_completion_window:
        parser.error("--tracker-completion-votes cannot exceed --tracker-completion-window")
    if args.tracker_max_forward_jump != 1:
        parser.error("V1 requires --tracker-max-forward-jump 1")
    for name in (
        "tracker_forward_min_span_sec",
        "tracker_completion_min_span_sec",
        "tracker_candidate_timeout_sec",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.tracker_candidate_timeout_sec <= 0:
        parser.error("--tracker-candidate-timeout-sec must be positive")
    if args.max_sampled_frames is not None and args.max_sampled_frames < 1:
        parser.error("--max-sampled-frames must be positive when provided")
    if args.procedure_mode != "baseline" and args.procedure_config is None:
        parser.error("--procedure-config is required for canonical/stateful mode")
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
    if args.procedure_config is not None:
        args.procedure_config = args.procedure_config.expanduser().resolve()
        if not args.procedure_config.is_file():
            raise FileNotFoundError(f"procedure config not found: {args.procedure_config}")
    adapter_checkpoint = is_lora_adapter_path(args.model_path)
    if args.use_lora and not adapter_checkpoint:
        raise ValueError(
            "--use-lora requires --model-path to be a LoRA adapter directory containing adapter_config.json"
        )
    # The top-level runner predates the explicit LoRA flag. An adapter path is
    # unambiguous, so preserve direct runner compatibility by enabling the same
    # official inference mode automatically when such a checkpoint is supplied.
    args.use_lora = bool(args.use_lora or adapter_checkpoint)

    jobs = load_jsonl(args.jobs_file)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "jobs_file": str(args.jobs_file),
            "jobs": len(jobs),
            "requested_free_memory_fraction": args.vllm_free_memory_fraction,
            "use_lora": args.use_lora,
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
