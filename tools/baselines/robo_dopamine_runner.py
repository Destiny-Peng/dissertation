#!/usr/bin/env python3
"""Runner-side orchestration for the persistent Robo-Dopamine worker."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from run_lf3r_baseline import (
    BASELINES,
    PROJECT_ROOT,
    ROBODOPAMINE_FATAL_EXIT_CODE,
    TOOLS_ROOT,
    append_jsonl,
    atomic_json,
    file_sha256,
    iso_now,
    load_jsonl,
    log_line,
    make_execution_environment,
    resolve_record_path,
    resolve_vllm_memory_budget,
    run_streamed,
    timestamp,
    validate_static,
    write_jsonl_atomic,
)
from robo_dopamine_multi_perspective import (
    FUSED_EVAL_MODE,
    PERSPECTIVE_MODES,
    resolve_eval_modes,
)


def build_job_specs(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    config: dict[str, Any],
    raw_root: Path,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    default_goal = config["repo"] / "examples/blank_goal.png"
    for index, record in enumerate(records):
        rollout_id = str(record.get("rollout_id", record.get("id", "")))
        if not rollout_id:
            raise ValueError("Robo-Dopamine job is missing rollout ID")
        video_value = record.get("video_path")
        if video_value is None:
            raise ValueError(f"Robo-Dopamine job {rollout_id} is missing video path")
        video = resolve_record_path(str(video_value), args.data_root)
        if not video.is_file():
            raise FileNotFoundError(f"Input for {rollout_id} does not exist: {video}")
        task = record.get("task", record.get("task_description"))
        if task is None:
            raise ValueError(f"Robo-Dopamine job {rollout_id} is missing task description")

        goal_value = record.get("goal_image")
        if goal_value is None:
            goal_value = args.goal_image or default_goal
        goal = Path(goal_value).expanduser()
        goal = goal.resolve() if goal.is_absolute() else resolve_record_path(str(goal), args.data_root)
        if not goal.is_file():
            raise FileNotFoundError(f"Goal image for {rollout_id} does not exist: {goal}")

        output_value = record.get("raw_output_dir", raw_root / rollout_id)
        output_dir = Path(output_value).expanduser()
        output_dir = output_dir.resolve() if output_dir.is_absolute() else (raw_root / output_dir).resolve()
        eval_mode = str(record.get("eval_mode", args.robo_eval_mode))
        eval_modes = resolve_eval_modes(
            eval_mode,
            record.get("eval_modes") or getattr(args, "robo_eval_modes", None),
        )
        specs.append({
            "job_index": int(record.get("job_index", index)),
            "rollout_id": rollout_id,
            "video_path": str(video),
            "task": str(task),
            "raw_output_dir": str(output_dir),
            "goal_image": str(goal),
            "frame_interval": int(record.get("frame_interval", args.robo_frame_interval)),
            "batch_size": int(record.get("batch_size", args.robo_batch_size)),
            "eval_mode": eval_mode,
            "eval_modes": eval_modes,
            "render_video": bool(record.get("render_video", args.render_video)),
        })
    return specs


def build_worker_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    model_path: Path,
    plan_path: Path,
    jobs_path: Path,
    progress_path: Path,
    state_path: Path,
    memory_budget: dict[str, Any],
    vllm_total_memory_fraction: float | None,
    *,
    resume: bool,
) -> list[str]:
    goal_image = args.goal_image or config["repo"] / "examples/blank_goal.png"
    command = [
        str(config["python"]),
        str(TOOLS_ROOT / "robo_dopamine_persistent_worker.py"),
        "--repo", str(config["repo"]),
        "--model-path", str(model_path),
        "--jobs-file", str(plan_path),
        "--jobs-output-file", str(jobs_path),
        "--progress-file", str(progress_path),
        "--state-file", str(state_path),
        "--goal-image", str(Path(goal_image).resolve()),
        "--frame-interval", str(args.robo_frame_interval),
        "--batch-size", str(args.robo_batch_size),
        "--eval-mode", args.robo_eval_mode,
        "--memory-budget-json", json.dumps(memory_budget, ensure_ascii=False, separators=(",", ":")),
    ]
    requested_modes = getattr(args, "robo_eval_modes", None)
    if requested_modes:
        command.extend(["--eval-modes", *requested_modes])
    elif args.robo_eval_mode == FUSED_EVAL_MODE:
        command.extend(["--eval-modes", *PERSPECTIVE_MODES])
    if vllm_total_memory_fraction is None:
        command.extend([
            "--dry-run",
            "--vllm-free-memory-fraction",
            str(args.vllm_free_memory_fraction),
        ])
    else:
        command.extend([
            "--vllm-total-memory-fraction",
            str(vllm_total_memory_fraction),
        ])
    if args.render_video:
        command.append("--render-video")
    if resume:
        command.append("--resume")
    return command


def job_statuses(jobs_path: Path) -> dict[str, dict[str, Any]]:
    if not jobs_path.is_file():
        return {}
    return {
        str(record["rollout_id"]): record
        for record in load_jsonl(jobs_path)
        if record.get("rollout_id") is not None
    }


def finalize_run(
    *,
    metadata: dict[str, Any],
    metadata_path: Path,
    jobs_path: Path,
    progress_path: Path,
    state_path: Path,
    log_path: Path,
    total_jobs: int,
    worker_return_code: int,
) -> int:
    jobs = load_jsonl(jobs_path) if jobs_path.is_file() else []
    counts = {
        "completed_jobs": sum(job.get("status") == "complete" for job in jobs),
        "failed_jobs": sum(job.get("status") == "failed" for job in jobs),
        "interrupted_jobs": sum(job.get("status") == "interrupted" for job in jobs),
    }
    progress = load_jsonl(progress_path) if progress_path.is_file() else []
    initialized = [
        event for event in progress if event.get("event") == "engine_initialized"
    ]
    fatal_events = [
        event for event in progress if event.get("event") == "fatal_engine_failure"
    ]
    if initialized:
        latest = initialized[-1]
        metadata["robo_dopamine_engine_initialization_seconds"] = latest.get(
            "initialization_seconds"
        )
        metadata["robo_dopamine_engine_status"] = "initialized"
    if fatal_events:
        metadata["robo_dopamine_engine_status"] = "fatal_engine_failure"
        metadata["robo_dopamine_fatal_error"] = fatal_events[-1]
    metadata.update(counts)
    metadata["pending_jobs"] = total_jobs - counts["completed_jobs"] - counts["failed_jobs"]
    metadata["robo_dopamine_inference_seconds"] = {
        str(job["rollout_id"]): job["inference_seconds"]
        for job in jobs
        if "inference_seconds" in job
    }
    metadata["robo_dopamine_worker_return_code"] = worker_return_code
    if state_path.is_file():
        metadata["robo_dopamine_last_state"] = json.loads(
            state_path.read_text(encoding="utf-8")
        )

    fatal = worker_return_code == ROBODOPAMINE_FATAL_EXIT_CODE or bool(fatal_events)
    all_jobs_terminal = counts["completed_jobs"] + counts["failed_jobs"] == total_jobs
    if fatal:
        status = "fatal_engine_failure"
    elif not all_jobs_terminal:
        status = "interrupted"
    elif counts["failed_jobs"]:
        status = "complete_with_errors"
    else:
        status = "complete"
    metadata.update(status=status, completed_at=iso_now())
    atomic_json(metadata_path, metadata)
    log_line(
        log_path,
        f"END status={status} completed={counts['completed_jobs']} failed={counts['failed_jobs']} interrupted={counts['interrupted_jobs']}",
    )
    print(f"Run metadata: {metadata_path}")
    print(f"Timestamped log: {log_path}")
    if status == "complete":
        return 0
    if status == "complete_with_errors":
        return 1
    return ROBODOPAMINE_FATAL_EXIT_CODE if fatal else (worker_return_code or 1)


def run_persistent(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    model_path: Path,
    run_root: Path,
    raw_root: Path,
    log_path: Path,
    metadata_path: Path,
    jobs_path: Path,
    commands_path: Path,
    env: dict[str, str],
    metadata: dict[str, Any],
    resume: bool = False,
) -> int:
    plan_path = run_root / "robo_dopamine_jobs.jsonl"
    progress_path = run_root / "robo_dopamine_progress.jsonl"
    state_path = run_root / "robo_dopamine_state.json"
    worker_path = TOOLS_ROOT / "robo_dopamine_persistent_worker.py"
    if not worker_path.is_file():
        raise FileNotFoundError(worker_path)
    plan_records = load_jsonl(plan_path) if resume else records
    specs = build_job_specs(plan_records, args, config, raw_root)
    if not specs:
        raise ValueError("No Robo-Dopamine jobs were selected")

    if resume:
        existing_status = job_statuses(jobs_path)
        if all(
            existing_status.get(spec["rollout_id"], {}).get("status")
            in {"complete", "failed"}
            for spec in specs
        ):
            completed_jobs = sum(
                item.get("status") == "complete" for item in existing_status.values()
            )
            failed_jobs = sum(
                item.get("status") == "failed" for item in existing_status.values()
            )
            atomic_json(
                state_path,
                {
                    "schema_version": 1,
                    "status": "no_pending_jobs",
                    "updated_at": iso_now(),
                    "total_jobs": len(specs),
                    "completed_jobs": completed_jobs,
                    "failed_jobs": failed_jobs,
                    "interrupted_jobs": 0,
                    "pending_jobs": 0,
                    "engine_initialization_seconds": metadata.get(
                        "robo_dopamine_engine_initialization_seconds"
                    ),
                    "engine_memory_budget": metadata.get(
                        "robo_dopamine_engine_memory_budget"
                    ),
                },
            )
            append_jsonl(
                progress_path,
                {
                    "event": "worker_complete",
                    "status": "no_pending_jobs",
                    "at": iso_now(),
                    "completed_jobs": completed_jobs,
                    "failed_jobs": failed_jobs,
                },
            )
            return finalize_run(
                metadata=metadata,
                metadata_path=metadata_path,
                jobs_path=jobs_path,
                progress_path=progress_path,
                state_path=state_path,
                log_path=log_path,
                total_jobs=len(specs),
                worker_return_code=0,
            )

    if args.dry_run:
        memory_budget = {
            "scope": "free_gpu_memory",
            "requested_free_fraction": args.vllm_free_memory_fraction,
            "resolved_total_fraction": None,
            "resolution": "deferred_until_execution",
        }
        vllm_total_memory_fraction = None
    else:
        try:
            vllm_total_memory_fraction, memory_budget = resolve_vllm_memory_budget(
                args.gpu, args.vllm_free_memory_fraction
            )
        except ValueError as error:
            metadata.update(status="memory_check_failed", error=str(error), completed_at=iso_now())
            atomic_json(metadata_path, metadata)
            log_line(log_path, f"MEMORY_CHECK_FAILED {error}")
            return 2
        log_line(log_path, "VLLM_MEMORY " + json.dumps(memory_budget, ensure_ascii=False))

    command = build_worker_command(
        args,
        config,
        model_path,
        plan_path,
        jobs_path,
        progress_path,
        state_path,
        memory_budget,
        vllm_total_memory_fraction,
        resume=resume,
    )
    existing_jobs = job_statuses(jobs_path)
    existing_commands = load_jsonl(commands_path) if commands_path.is_file() else []
    decorated_specs: list[dict[str, Any]] = []
    command_records: list[dict[str, Any]] = []
    for spec in specs:
        rollout_id = spec["rollout_id"]
        prior_attempts = [
            int(item.get("attempt", 1))
            for item in existing_commands
            if item.get("rollout_id") == rollout_id
        ]
        attempt = max(prior_attempts, default=0) + 1
        decorated = {
            **spec,
            "cwd": str(config["repo"]),
            "argv": command,
            "shell_preview": shlex.join(command),
            "vllm_memory_budget": memory_budget,
            "execution_scope": "persistent_robo_dopamine_worker",
        }
        decorated_specs.append(decorated)
        if args.dry_run or existing_jobs.get(rollout_id, {}).get("status") not in {"complete", "failed"}:
            command_records.append({**decorated, "attempt": attempt})
    write_jsonl_atomic(plan_path, decorated_specs)
    for command_record in command_records:
        append_jsonl(commands_path, command_record)
    metadata.update(
        robo_dopamine_persistent_worker=True,
        robo_dopamine_job_plan=str(plan_path),
        robo_dopamine_progress_file=str(progress_path),
        robo_dopamine_state_file=str(state_path),
        robo_dopamine_engine_command=command,
        robo_dopamine_engine_memory_budget=memory_budget,
        selected_rollouts=len(specs),
    )
    memory_history = list(metadata.get("robo_dopamine_memory_budgets", []))
    memory_history.append(memory_budget)
    metadata["robo_dopamine_memory_budgets"] = memory_history

    if args.dry_run:
        planned = [{**spec, "status": "planned"} for spec in decorated_specs]
        write_jsonl_atomic(jobs_path, planned)
        atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "dry_run",
                "updated_at": iso_now(),
                "total_jobs": len(specs),
                "completed_jobs": 0,
                "failed_jobs": 0,
                "interrupted_jobs": 0,
                "pending_jobs": len(specs),
                "engine_initialization_seconds": None,
                "engine_memory_budget": memory_budget,
            },
        )
        metadata.update(
            status="dry_run_complete",
            completed_jobs=0,
            failed_jobs=0,
            interrupted_jobs=0,
            pending_jobs=len(specs),
            completed_at=iso_now(),
        )
        atomic_json(metadata_path, metadata)
        log_line(log_path, "END status=dry_run_complete completed=0 failed=0")
        print(f"Run metadata: {metadata_path}")
        print(f"Timestamped log: {log_path}")
        return 0

    metadata["status"] = "running"
    atomic_json(metadata_path, metadata)
    log_line(
        log_path,
        f"PERSISTENT_ROBO_DOPAMINE_START jobs={len(specs)} resume={resume} engine_reuse=true",
    )
    worker_return_code = run_streamed(command, config["repo"], env, log_path)
    return finalize_run(
        metadata=metadata,
        metadata_path=metadata_path,
        jobs_path=jobs_path,
        progress_path=progress_path,
        state_path=state_path,
        log_path=log_path,
        total_jobs=len(specs),
        worker_return_code=worker_return_code,
    )


def resume_run(args: argparse.Namespace) -> int:
    """Resume a Robo-Dopamine run from its atomic per-rollout progress files."""
    run_root = args.resume_run.expanduser().resolve()
    metadata_path = run_root / "run.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Robo-Dopamine run metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("baseline") != "robo_dopamine":
        raise ValueError(f"--resume-run is only supported for Robo-Dopamine runs: {run_root}")
    if metadata.get("dry_run"):
        raise ValueError(f"Cannot resume a dry-run plan: {run_root}")

    stored_arguments = metadata.get("arguments", {})
    manifest_value = metadata.get("manifest")
    if not manifest_value:
        raise ValueError(f"Robo-Dopamine run metadata has no manifest: {metadata_path}")
    args.manifest = Path(manifest_value).expanduser().resolve()
    args.data_root = Path(metadata.get("data_root", PROJECT_ROOT)).expanduser().resolve()
    output_root = Path(metadata.get("output_root", run_root)).expanduser().resolve()
    if output_root != run_root:
        raise ValueError(f"Run metadata output_root does not match --resume-run: {output_root}")
    args.output_dir = run_root.parent
    fallback_log_dir = args.logs_dir or PROJECT_ROOT / "logs/baselines"
    args.logs_dir = Path(
        metadata.get("log_path", fallback_log_dir / f"robo_dopamine_resume_{timestamp()}.log")
    ).expanduser().resolve().parent
    args.model_path = Path(
        args.model_path
        or metadata.get("model_path")
        or BASELINES["robo_dopamine"]["checkpoint"]
    ).expanduser().resolve()
    args.gpu = args.gpu or stored_arguments.get("gpu") or "0"
    args.vllm_free_memory_fraction = (
        args.vllm_free_memory_fraction
        if args.vllm_free_memory_fraction is not None
        else metadata.get("vllm_requested_free_fraction")
        or stored_arguments.get("vllm_free_memory_fraction")
        or 0.80
    )
    args.robo_frame_interval = int(stored_arguments.get("robo_frame_interval", 4))
    args.robo_batch_size = int(stored_arguments.get("robo_batch_size", 1))
    args.robo_eval_mode = str(stored_arguments.get("robo_eval_mode", FUSED_EVAL_MODE))
    stored_modes = stored_arguments.get("robo_eval_modes")
    args.robo_eval_modes = list(stored_modes) if stored_modes else None
    stored_goal = stored_arguments.get("goal_image")
    args.goal_image = Path(stored_goal).expanduser().resolve() if stored_goal else None
    args.render_video = bool(stored_arguments.get("render_video", False))
    args.dry_run = False

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {args.manifest}")
    recorded_hash = metadata.get("manifest_sha256")
    if recorded_hash and file_sha256(args.manifest) != recorded_hash:
        raise ValueError(f"Manifest changed since the original run: {args.manifest}")

    config = BASELINES["robo_dopamine"]
    validate_static(config, args.model_path)
    plan_path = Path(
        metadata.get("robo_dopamine_job_plan", run_root / "robo_dopamine_jobs.jsonl")
    ).expanduser().resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Robo-Dopamine job plan does not exist: {plan_path}")
    records = load_jsonl(plan_path)
    if not records:
        raise ValueError(f"Robo-Dopamine job plan is empty: {plan_path}")

    log_path = Path(
        metadata.get(
            "log_path", fallback_log_dir / f"robo_dopamine_resume_{timestamp()}.log"
        )
    ).expanduser().resolve()
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = run_root / "jobs.jsonl"
    commands_path = run_root / "commands.jsonl"
    raw_root = run_root / "raw"
    env = make_execution_environment(config, args.gpu)
    if args.validate_environment:
        check = [str(config["python"]), "-c", config["imports"]]
        log_line(log_path, "ENVIRONMENT_CHECK " + json.dumps(check))
        return_code = run_streamed(check, config["repo"], env, log_path)
        if return_code:
            metadata.update(
                status="environment_validation_failed",
                failed_jobs=metadata.get("failed_jobs", 0) + 1,
                completed_at=iso_now(),
            )
            atomic_json(metadata_path, metadata)
            return return_code

    metadata["resume_count"] = int(metadata.get("resume_count", 0)) + 1
    metadata["last_resume_at"] = iso_now()
    metadata["last_resume_overrides"] = {
        "gpu": args.gpu,
        "requested_free_memory_fraction": args.vllm_free_memory_fraction,
    }
    metadata["status"] = "resuming"
    atomic_json(metadata_path, metadata)
    return run_persistent(
        args=args,
        config=config,
        records=records,
        model_path=args.model_path,
        run_root=run_root,
        raw_root=raw_root,
        log_path=log_path,
        metadata_path=metadata_path,
        jobs_path=jobs_path,
        commands_path=commands_path,
        env=env,
        metadata=metadata,
        resume=True,
    )
