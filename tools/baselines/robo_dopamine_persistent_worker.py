#!/usr/bin/env python3
"""Run Robo-Dopamine rollouts with one persistent GRMInference/vLLM engine."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from procvlm_worker import (
    FATAL_EXIT_CODE,
    TERMINAL_JOB_STATUSES,
    append_jsonl,
    atomic_json,
    cleanup_after_recoverable_failure,
    is_fatal_engine_failure,
    job_counts,
    load_jsonl,
    upsert_job,
)
from robo_dopamine_multi_perspective import (
    FUSED_EVAL_MODE,
    PERSPECTIVE_MODES,
    fuse_prediction_files,
    plot_progress_curves,
    resolve_eval_modes,
    summarize_incremental_noise,
    write_progress_csv,
)


def initialize_robo_model(args: argparse.Namespace) -> Any:
    """Import the official module and construct GRMInference exactly once."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    repo = str(args.repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import examples.inference as official

    # The official constructor hard-codes 0.9. Patch only its module-local LLM
    # symbol so the runner's free-memory-derived budget remains authoritative.
    official_llm = getattr(official, "LLM", None)
    if official_llm is None:
        raise RuntimeError("Robo-Dopamine examples.inference has no LLM symbol")

    def bounded_llm(*model_args: Any, **model_kwargs: Any) -> Any:
        model_kwargs["gpu_memory_utilization"] = args.vllm_total_memory_fraction
        return official_llm(*model_args, **model_kwargs)

    official.LLM = bounded_llm
    return official.GRMInference(str(args.model_path.resolve()))


def official_source_revision(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _run_official_mode(
    *,
    model: Any,
    video: str,
    goal_image: str,
    output_dir: Path,
    task: str,
    frame_interval: int,
    batch_size: int,
    eval_mode: str,
    render_video: bool,
) -> tuple[Path, float]:
    started = time.perf_counter()
    output = model.run_pipeline(
        cam_high_path=video,
        cam_left_path=video,
        cam_right_path=video,
        out_root=str(output_dir),
        task=task,
        frame_interval=frame_interval,
        batch_size=batch_size,
        goal_image=goal_image,
        eval_mode=eval_mode,
        visualize=render_video,
    )
    output_path = Path(output).resolve()
    prediction = output_path / "pred_vllm.json"
    if not prediction.is_file():
        raise RuntimeError(
            f"Robo-Dopamine {eval_mode} mode did not write raw predictions: {prediction}"
        )
    return prediction, time.perf_counter() - started


def infer_rollout(
    job: dict[str, Any], args: argparse.Namespace, model: Any
) -> dict[str, Any]:
    """Run one rollout through the already-initialized GRMInference object."""
    output_dir = Path(job["raw_output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video = str(Path(job["video_path"]).resolve())
    goal_image = str(Path(job.get("goal_image", args.goal_image)).resolve())
    frame_interval = int(job.get("frame_interval", args.frame_interval))
    batch_size = int(job.get("batch_size", args.batch_size))
    eval_mode = str(job.get("eval_mode", args.eval_mode))
    requested_modes = job.get("eval_modes") or getattr(args, "eval_modes", None)
    eval_modes = resolve_eval_modes(eval_mode, requested_modes)
    if not eval_modes:
        eval_modes = [eval_mode]
    render_video = bool(job.get("render_video", args.render_video))
    task = str(job["task"])

    mode_predictions: dict[str, Path] = {}
    mode_seconds: dict[str, float] = {}
    for mode in eval_modes:
        prediction, elapsed = _run_official_mode(
            model=model,
            video=video,
            goal_image=goal_image,
            output_dir=output_dir,
            task=task,
            frame_interval=frame_interval,
            batch_size=batch_size,
            eval_mode=mode,
            render_video=render_video,
        )
        mode_predictions[mode] = prediction
        mode_seconds[mode] = elapsed

    multi_perspective = len(eval_modes) > 1
    fused_path: Path | None = None
    fusion_metadata: dict[str, Any] = {}
    if multi_perspective:
        if set(eval_modes) != set(PERSPECTIVE_MODES):
            raise ValueError(
                "Multi-perspective Robo-Dopamine output requires exactly "
                f"{', '.join(PERSPECTIVE_MODES)}"
            )
        fused_root = output_dir / "multi_perspective"
        fused_path = fused_root / "fused_progress.json"
        fusion_metadata = fuse_prediction_files(
            mode_predictions,
            fused_path,
            metadata={
                "checkpoint": str(args.model_path.resolve()),
                "eval_modes": list(PERSPECTIVE_MODES),
                "frame_interval": frame_interval,
                "goal_image": goal_image,
                "fusion_rule": "arithmetic_mean_of_official_progress",
                "source_commit": official_source_revision(args.repo),
            },
        )
        write_progress_csv(fused_root / "progress_curves.csv", mode_predictions, fused_path)
        fusion_metadata["incremental_noise"] = summarize_incremental_noise(
            mode_predictions["incremental"]
        )
        plot_path = fused_root / "progress_curves.png"
        if plot_progress_curves(
            plot_path,
            mode_predictions,
            fused_path,
            title=f"Robo-Dopamine multi-perspective: {job['rollout_id']}",
        ):
            fusion_metadata["plot_path"] = str(plot_path)
        else:
            fusion_metadata["plot_path"] = None
        (fused_root / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "checkpoint": str(args.model_path.resolve()),
                    "eval_modes": list(PERSPECTIVE_MODES),
                    "frame_interval": frame_interval,
                    "goal_image": goal_image,
                    "fusion_rule": "arithmetic_mean_of_official_progress",
                    "source_commit": official_source_revision(args.repo),
                    "prediction_paths": {mode: str(path) for mode, path in mode_predictions.items()},
                    "fused_path": str(fused_path),
                    "mode_seconds": mode_seconds,
                    "fusion": fusion_metadata,
                },
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

    # Keep the legacy single-mode contract unchanged. For a multi-mode job,
    # raw_model_output points to the fused curve so existing readers show the
    # fused result, while every official per-mode file remains available.
    raw_model_output = fused_path or mode_predictions[eval_modes[0]]
    official_output_dir = mode_predictions[eval_modes[0]].parent
    result_path = output_dir / "worker_result.json"
    result = {
        "schema_version": 2 if multi_perspective else 1,
        "baseline": "robo_dopamine",
        "official_output_dir": str(official_output_dir),
        "raw_model_output": str(raw_model_output),
        "eval_modes": eval_modes,
        "frame_interval": frame_interval,
        "batch_size": batch_size,
        "goal_image": goal_image,
        "checkpoint": str(args.model_path.resolve()),
        "source_commit": official_source_revision(args.repo),
        "video_path": video,
        "task": task,
        "eval_mode": eval_mode,
    }
    if multi_perspective:
        result.update(
            {
                "multi_perspective": True,
                "fusion_rule": "arithmetic_mean_of_official_progress",
                "fused_model_output": str(fused_path),
                "perspective_outputs": {
                    mode: {
                        "official_output_dir": str(path.parent),
                        "raw_model_output": str(path),
                        "seconds": mode_seconds[mode],
                    }
                    for mode, path in mode_predictions.items()
                },
                "mode_seconds": mode_seconds,
                "fusion": fusion_metadata,
            }
        )
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for mode, prediction in mode_predictions.items():
        print(f"Raw Robo-Dopamine {mode} output: {prediction}", flush=True)
    if fused_path is not None:
        print(f"Fused Robo-Dopamine output: {fused_path}", flush=True)
    return {
        "official_output_dir": str(official_output_dir),
        "raw_model_output": str(raw_model_output),
        "worker_result_path": str(result_path),
        "eval_modes": eval_modes,
        "perspective_outputs": result.get("perspective_outputs"),
        "fused_model_output": str(fused_path) if fused_path else None,
        "mode_seconds": mode_seconds,
        "fusion": fusion_metadata,
    }


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
        "updated_at": datetime_now(),
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


def datetime_now() -> str:
    # Kept local to make this worker easy to exercise without importing the
    # baseline runner (which would otherwise initialize project-level paths).
    from datetime import datetime

    return datetime.now().astimezone().isoformat()


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


def find_prediction(output_dir: Path) -> Path | None:
    predictions = sorted(path for path in output_dir.rglob("pred_vllm.json") if path.is_file())
    return predictions[-1] if predictions else None


def run_persistent_jobs(
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    jobs_output_path: Path,
    progress_path: Path,
    state_path: Path,
    engine_memory_budget: dict[str, Any],
    initialize_model: Callable[[argparse.Namespace], Any] = initialize_robo_model,
    infer_one: Callable[[dict[str, Any], argparse.Namespace, Any], Any] = infer_rollout,
) -> int:
    """Initialize one GRMInference, then process pending jobs in plan order."""
    existing = load_jsonl(jobs_output_path)
    existing_by_id = {record.get("rollout_id"): record for record in existing}
    pending = [
        job
        for job in jobs
        if existing_by_id.get(job.get("rollout_id"), {}).get("status")
        not in TERMINAL_JOB_STATUSES
    ]

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
                "at": datetime_now(),
                **job_counts(existing),
            },
        )
        return 0

    append_jsonl(
        progress_path,
        {
            "event": "engine_initialization_started",
            "at": datetime_now(),
            "pending_jobs": len(pending),
            "total_jobs": len(jobs),
            "memory_budget": engine_memory_budget,
        },
    )
    init_started = time.perf_counter()
    try:
        model = initialize_model(args)
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
                "at": datetime_now(),
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
            "at": datetime_now(),
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
        output_dir = Path(job["raw_output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime_now()
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
        result_metadata: dict[str, Any] = {}
        try:
            returned = infer_one(job, args, model)
            if isinstance(returned, dict):
                result_metadata = {
                    key: value
                    for key, value in returned.items()
                    if key in {"official_output_dir", "raw_model_output", "worker_result_path", "eval_modes", "perspective_outputs", "fused_model_output", "mode_seconds", "fusion"}
                }
            prediction_value = result_metadata.get("raw_model_output")
            prediction = (
                Path(str(prediction_value)).resolve()
                if prediction_value
                else find_prediction(output_dir)
            )
            if prediction is None or not prediction.is_file():
                raise RuntimeError(
                    f"Robo-Dopamine produced no pred_vllm.json under {output_dir}"
                )
            result_metadata.setdefault("raw_model_output", str(prediction))
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
        raw_files = sorted(str(path) for path in output_dir.rglob("*") if path.is_file())
        result = {
            **command_metadata(job),
            **result_metadata,
            "status": status,
            "return_code": return_code,
            "started_at": started_at,
            "completed_at": datetime_now(),
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
                    "at": datetime_now(),
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
            "at": datetime_now(),
            "engine_initialization_seconds": init_seconds,
            **counts,
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--jobs-output-file", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--goal-image", type=Path, required=True)
    parser.add_argument("--frame-interval", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--eval-mode",
        choices=(FUSED_EVAL_MODE, "forward", "incremental", "backward"),
        default=FUSED_EVAL_MODE,
        help="Wrapper mode; fused runs the official incremental, forward, and backward perspectives",
    )
    parser.add_argument(
        "--eval-modes",
        nargs="+",
        choices=PERSPECTIVE_MODES,
        default=None,
        help="Official modes to run with one persistent GRM; all three are required for fusion",
    )
    parser.add_argument("--vllm-total-memory-fraction", type=float, default=None)
    parser.add_argument("--vllm-free-memory-fraction", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--memory-budget-json", required=True)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.frame_interval < 1:
        parser.error("--frame-interval must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.eval_modes:
        if len(set(args.eval_modes)) != len(args.eval_modes):
            parser.error("--eval-modes must not contain duplicates")
        if len(args.eval_modes) > 1 and set(args.eval_modes) != set(PERSPECTIVE_MODES):
            parser.error("multi-perspective mode requires incremental, forward, and backward")
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
    args.repo = args.repo.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.jobs_file = args.jobs_file.expanduser().resolve()
    args.jobs_output_file = args.jobs_output_file.expanduser().resolve()
    args.progress_file = args.progress_file.expanduser().resolve()
    args.state_file = args.state_file.expanduser().resolve()
    args.goal_image = args.goal_image.expanduser().resolve()
    jobs = load_jsonl(args.jobs_file)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "jobs_file": str(args.jobs_file),
                    "jobs": len(jobs),
                    "requested_free_memory_fraction": args.vllm_free_memory_fraction,
                },
                ensure_ascii=False,
            )
        )
        return 0
    for required in (args.repo / "examples/inference.py", args.model_path, args.goal_image):
        if not required.exists():
            raise FileNotFoundError(required)
    if not jobs:
        raise ValueError(f"No Robo-Dopamine jobs found in {args.jobs_file}")
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
