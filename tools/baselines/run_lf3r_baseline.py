#!/usr/bin/env python3
"""Plan or execute existing baseline methods over an LF3R manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
VLLM_BASELINES = {"procvlm", "robo_dopamine"}
PERSISTENT_BASELINES = VLLM_BASELINES | {"densereward"}
PROCVLM_FATAL_EXIT_CODE = 70
ROBODOPAMINE_FATAL_EXIT_CODE = 70

BASELINES = {
    "safe": {
        "python": PROJECT_ROOT / "conda_envs/LF3R-safe/bin/python",
        "repo": PROJECT_ROOT / "repos/SAFE",
        "entrypoint": TOOLS_ROOT / "safe_worker.py",
        "checkpoint": None,
        "imports": "import pandas, torch; from failure_prob.data.openvla import compute_hand_crafted_metrics",
    },
    "procvlm": {
        "python": PROJECT_ROOT / "repos/ProcVLM/.venv/bin/python",
        "repo": PROJECT_ROOT / "repos/ProcVLM",
        "entrypoint": PROJECT_ROOT / "repos/ProcVLM/evqa/inference.py",
        "checkpoint": PROJECT_ROOT / "checkpoints/ProcVLM-2B",
        "imports": "import PIL, moviepy, vllm; import evqa.inference",
    },
    "rynnvalue": {
        "python": PROJECT_ROOT / "repos/RynnValue/.venv/bin/python",
        "repo": PROJECT_ROOT / "repos/RynnValue",
        "entrypoint": TOOLS_ROOT / "rynnvalue_worker.py",
        "checkpoint": PROJECT_ROOT / "checkpoints/RynnValue-4B",
        "imports": "import torch, transformers, imageio; print(torch.__version__, transformers.__version__)",
    },
    "robo_dopamine": {
        "python": PROJECT_ROOT / "conda_envs/LF3R-robo-dopamine/bin/python",
        "repo": PROJECT_ROOT / "repos/Robo-Dopamine",
        "entrypoint": TOOLS_ROOT / "robo_dopamine_persistent_worker.py",
        "checkpoint": PROJECT_ROOT / "checkpoints/Robo-Dopamine-GRM-2.0-4B-Preview",
        "imports": "import cv2, transformers, vllm; print(cv2.__version__, transformers.__version__)",
    },
    "densereward": {
        "python": PROJECT_ROOT / "conda_envs/LF3R-densereward/bin/python",
        "repo": PROJECT_ROOT,
        "entrypoint": TOOLS_ROOT / "densereward_worker.py",
        "checkpoint": PROJECT_ROOT / "checkpoints/densereward-3frame-thinking",
        "imports": "import av, PIL, qwen_vl_utils, torch, transformers; print(torch.__version__, transformers.__version__)",
    },
}


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return records


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    """Read an optional worker status file before the worker has written it."""
    if not path.is_file():
        return []
    return load_jsonl(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_jsonl_atomic(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rendered = "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values)
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def log_line(
    log_path: Path,
    message: str,
    *,
    console: bool = True,
    lock: threading.Lock | None = None,
) -> None:
    rendered = f"[{iso_now()}] {message}"
    if lock is None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    else:
        with lock:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
    if console:
        print(rendered, flush=True)


def resolve_record_path(value: str, data_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


def approximate_num_steps_for_interval(total_frames: int, interval: int) -> int:
    """Return the official sampler count that approximates a fixed frame interval.

    RynnValue's official CLI uniformly distributes num_steps endpoints from
    frame 0 through frame T - 1. Choosing ceil((T - 1) / K) + 1 gives the
    same number of points as 0, K, 2K, ... plus the final frame, while retaining
    the upstream prefix-uniform implementation.
    """
    if total_frames < 1:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if interval < 1:
        raise ValueError(f"evaluation interval must be positive, got {interval}")
    if total_frames == 1:
        return 1
    return min(total_frames, ((total_frames - 1) + interval - 1) // interval + 1)


def selected_gpu_ids(gpu: str) -> list[int]:
    parts = [part.strip() for part in str(gpu).split(",")]
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"--gpu must be a comma-separated list of CUDA device indices: {gpu}")
    # Repeated IDs are intentional when several rollout workers share one GPU.
    return [int(part) for part in parts]


def query_gpu_memory(gpu: str) -> list[dict[str, int]]:
    requested = selected_gpu_ids(gpu)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"Unable to query GPU memory with nvidia-smi: {error}") from error

    rows: dict[int, dict[str, int]] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not all(field.isdigit() for field in fields):
            continue
        index, total, free = (int(field) for field in fields)
        rows[index] = {"gpu": index, "total_mib": total, "free_mib": free}
    missing = [str(index) for index in requested if index not in rows]
    if missing:
        raise ValueError(f"nvidia-smi did not report requested GPU indices: {missing}")
    return [rows[index] for index in requested]


def convert_free_memory_fraction(
    free_memory_fraction: float, snapshots: list[dict[str, int]]
) -> tuple[float, float]:
    if not 0.0 < free_memory_fraction <= 1.0:
        raise ValueError("free_memory_fraction must be in (0, 1]")
    if not snapshots:
        raise ValueError("No GPU memory snapshots were provided")
    if any(
        row["total_mib"] <= 0
        or row["free_mib"] < 0
        or row["free_mib"] > row["total_mib"]
        for row in snapshots
    ):
        raise ValueError("GPU memory snapshots contain invalid values")
    free_total_ratio = min(row["free_mib"] / row["total_mib"] for row in snapshots)
    # Flooring avoids turning a measurement into a slightly larger request
    # because of decimal rounding. The requested fraction itself leaves the
    # remaining free-memory fraction as headroom for external workloads.
    effective = math.floor(free_memory_fraction * free_total_ratio * 1_000_000) / 1_000_000
    if effective <= 0.0:
        raise ValueError("Resolved vLLM memory fraction is non-positive")
    return effective, free_total_ratio


def resolve_vllm_memory_budget(gpu: str, free_memory_fraction: float) -> tuple[float, dict[str, Any]]:
    """Convert a free-memory target to vLLM total-memory fraction.

    vLLM gpu_memory_utilization is relative to total visible memory. The
    user-facing runner value is instead relative to currently free memory, so
    the smallest free/total ratio is used for tensor-parallel GPU selections.
    """
    snapshots = query_gpu_memory(gpu)
    effective, free_total_ratio = convert_free_memory_fraction(free_memory_fraction, snapshots)
    budget = {
        "scope": "free_gpu_memory",
        "requested_free_fraction": free_memory_fraction,
        "resolved_total_fraction": effective,
        "selected_gpus": snapshots,
        "minimum_free_total_ratio": free_total_ratio,
    }
    return effective, budget


def build_procvlm_job_specs(
    records: list[dict[str, Any]], args: argparse.Namespace, raw_root: Path
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        rollout_id = str(record.get("rollout_id", record.get("id", "")))
        if not rollout_id:
            raise ValueError("ProcVLM job is missing rollout ID")
        video_value = record.get("video_path")
        if video_value is None:
            raise ValueError(f"ProcVLM job {rollout_id} is missing video path")
        video_path = Path(video_value).expanduser()
        video = video_path.resolve() if video_path.is_absolute() else resolve_record_path(str(video_value), args.data_root)
        if not video.is_file():
            raise FileNotFoundError(f"Input for {rollout_id} does not exist: {video}")
        task = record.get("task", record.get("task_description"))
        if task is None:
            raise ValueError(f"ProcVLM job {rollout_id} is missing task description")
        output_dir = Path(record.get("raw_output_dir", raw_root / rollout_id)).expanduser().resolve()
        output_value = record.get("output_path", output_dir / "procvlm_raw.jsonl")
        output_path = Path(output_value).expanduser().resolve()
        specs.append({
            "job_index": int(record.get("job_index", index)),
            "rollout_id": rollout_id,
            "video_path": str(video),
            "task": str(task),
            "raw_output_dir": str(output_dir),
            "output_path": str(output_path),
        })
    return specs


def build_procvlm_worker_command(
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
    command = [
        str(config["python"]),
        str(TOOLS_ROOT / "procvlm_worker.py"),
        "--model-path", str(model_path),
        "--jobs-file", str(plan_path),
        "--jobs-output-file", str(jobs_path),
        "--progress-file", str(progress_path),
        "--state-file", str(state_path),
        "--memory-budget-json", json.dumps(memory_budget, ensure_ascii=False, separators=(",", ":")),
        "--window-size", str(args.procvlm_window_size),
        "--max-new-tokens", str(args.procvlm_max_new_tokens),
        "--torch-dtype", args.dtype,
        "--tp", str(args.tensor_parallel_size),
    ]
    if vllm_total_memory_fraction is None:
        command.extend(["--dry-run", "--vllm-free-memory-fraction", str(args.vllm_free_memory_fraction)])
    else:
        command.extend(["--vllm-total-memory-fraction", str(vllm_total_memory_fraction)])
    if args.procvlm_max_sampled_frames is not None:
        command.extend(["--max-sampled-frames", str(args.procvlm_max_sampled_frames)])
    if resume:
        command.append("--resume")
    return command


def build_densereward_job_specs(
    records: list[dict[str, Any]], args: argparse.Namespace, raw_root: Path
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        rollout_id = str(record.get("rollout_id", record.get("id", "")))
        if not rollout_id:
            raise ValueError("DenseReward job is missing rollout ID")
        video_value = record.get("video_path")
        if video_value is None:
            raise ValueError(f"DenseReward job {rollout_id} is missing video path")
        video_path = Path(video_value).expanduser()
        video = (
            video_path.resolve()
            if video_path.is_absolute()
            else resolve_record_path(str(video_value), args.data_root)
        )
        if not video.is_file():
            raise FileNotFoundError(f"Input for {rollout_id} does not exist: {video}")
        task = record.get("task", record.get("task_description"))
        if task is None:
            raise ValueError(f"DenseReward job {rollout_id} is missing task description")
        output_dir = Path(record.get("raw_output_dir", raw_root / rollout_id)).expanduser().resolve()
        specs.append({
            "job_index": int(record.get("job_index", index)),
            "rollout_id": rollout_id,
            "video_path": str(video),
            "task": str(task),
            "raw_output_dir": str(output_dir),
            "output_path": str(output_dir / "densereward_raw.jsonl"),
            "frame_interval": int(args.densereward_frame_interval),
            "max_new_tokens": int(args.densereward_max_new_tokens),
        })
    return specs


def build_densereward_worker_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    model_path: Path,
    plan_path: Path,
    jobs_path: Path,
    progress_path: Path,
    state_path: Path,
    *,
    resume: bool,
) -> list[str]:
    command = [
        str(config["python"]),
        str(TOOLS_ROOT / "densereward_worker.py"),
        "--model-path", str(model_path),
        "--jobs-file", str(plan_path),
        "--jobs-output-file", str(jobs_path),
        "--progress-file", str(progress_path),
        "--state-file", str(state_path),
        "--frame-interval", str(args.densereward_frame_interval),
        "--max-new-tokens", str(args.densereward_max_new_tokens),
    ]
    if resume:
        command.append("--resume")
    return command


def procvlm_job_statuses(jobs_path: Path) -> dict[str, dict[str, Any]]:
    if not jobs_path.is_file():
        return {}
    return {
        str(record["rollout_id"]): record
        for record in load_jsonl(jobs_path)
        if record.get("rollout_id") is not None
    }


def finalize_procvlm_persistent_run(
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
    engine_initialized = [
        event for event in progress if event.get("event") == "engine_initialized"
    ]
    fatal_events = [
        event for event in progress if event.get("event") == "fatal_engine_failure"
    ]
    if engine_initialized:
        latest_engine = engine_initialized[-1]
        metadata["procvlm_engine_initialization_seconds"] = latest_engine.get("initialization_seconds")
        metadata["procvlm_engine_status"] = "initialized"
    if fatal_events:
        metadata["procvlm_engine_status"] = "fatal_engine_failure"
        metadata["procvlm_fatal_error"] = fatal_events[-1]
    metadata.update(counts)
    metadata["pending_jobs"] = total_jobs - counts["completed_jobs"] - counts["failed_jobs"]
    metadata["procvlm_inference_seconds"] = {
        str(job["rollout_id"]): job["inference_seconds"]
        for job in jobs
        if "inference_seconds" in job
    }
    metadata["procvlm_worker_return_code"] = worker_return_code
    if state_path.is_file():
        metadata["procvlm_last_state"] = json.loads(state_path.read_text(encoding="utf-8"))

    fatal = worker_return_code == PROCVLM_FATAL_EXIT_CODE or bool(fatal_events)
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
    log_line(log_path, f"END status={status} completed={counts['completed_jobs']} failed={counts['failed_jobs']} interrupted={counts['interrupted_jobs']}")
    print(f"Run metadata: {metadata_path}")
    print(f"Timestamped log: {log_path}")
    if status == "complete":
        return 0
    if status == "complete_with_errors":
        return 1
    return PROCVLM_FATAL_EXIT_CODE if fatal else (worker_return_code or 1)


def run_procvlm_persistent(
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
    plan_path = run_root / "procvlm_jobs.jsonl"
    progress_path = run_root / "procvlm_progress.jsonl"
    state_path = run_root / "procvlm_state.json"
    if not (TOOLS_ROOT / "procvlm_worker.py").is_file():
        raise FileNotFoundError(TOOLS_ROOT / "procvlm_worker.py")
    plan_records = load_jsonl(plan_path) if resume else records
    specs = build_procvlm_job_specs(plan_records, args, raw_root)
    if not specs:
        raise ValueError("No ProcVLM jobs were selected")

    if resume:
        existing_status = procvlm_job_statuses(jobs_path)
        if all(existing_status.get(spec["rollout_id"], {}).get("status") in {"complete", "failed"} for spec in specs):
            completed_jobs = sum(item.get("status") == "complete" for item in existing_status.values())
            failed_jobs = sum(item.get("status") == "failed" for item in existing_status.values())
            atomic_json(state_path, {
                "schema_version": 1,
                "status": "no_pending_jobs",
                "updated_at": iso_now(),
                "total_jobs": len(specs),
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "interrupted_jobs": 0,
                "pending_jobs": 0,
                "engine_initialization_seconds": metadata.get("procvlm_engine_initialization_seconds"),
                "engine_memory_budget": metadata.get("procvlm_engine_memory_budget"),
            })
            append_jsonl(progress_path, {
                "event": "worker_complete",
                "status": "no_pending_jobs",
                "at": iso_now(),
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
            })
            return finalize_procvlm_persistent_run(
                metadata=metadata, metadata_path=metadata_path, jobs_path=jobs_path,
                progress_path=progress_path, state_path=state_path, log_path=log_path,
                total_jobs=len(specs), worker_return_code=0,
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
            vllm_total_memory_fraction, memory_budget = resolve_vllm_memory_budget(args.gpu, args.vllm_free_memory_fraction)
        except ValueError as error:
            metadata.update(status="memory_check_failed", error=str(error), completed_at=iso_now())
            atomic_json(metadata_path, metadata)
            log_line(log_path, f"MEMORY_CHECK_FAILED {error}")
            return 2
        log_line(log_path, "VLLM_MEMORY " + json.dumps(memory_budget, ensure_ascii=False))

    command = build_procvlm_worker_command(
        args, config, model_path, plan_path, jobs_path, progress_path, state_path,
        memory_budget, vllm_total_memory_fraction, resume=resume,
    )
    existing_jobs = procvlm_job_statuses(jobs_path)
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
            "execution_scope": "persistent_procvlm_worker",
        }
        decorated_specs.append(decorated)
        if args.dry_run or existing_jobs.get(rollout_id, {}).get("status") not in {"complete", "failed"}:
            command_records.append({**decorated, "attempt": attempt})
    write_jsonl_atomic(plan_path, decorated_specs)
    for command_record in command_records:
        append_jsonl(commands_path, command_record)
    metadata.update(
        procvlm_persistent_worker=True,
        procvlm_job_plan=str(plan_path),
        procvlm_progress_file=str(progress_path),
        procvlm_state_file=str(state_path),
        procvlm_engine_command=command,
        procvlm_engine_memory_budget=memory_budget,
        selected_rollouts=len(specs),
    )
    memory_history = list(metadata.get("procvlm_memory_budgets", []))
    memory_history.append(memory_budget)
    metadata["procvlm_memory_budgets"] = memory_history

    if args.dry_run:
        planned = [
            {**command_record, "status": "planned", "raw_output_dir": spec["raw_output_dir"]}
            for spec, command_record in zip(specs, command_records)
        ]
        write_jsonl_atomic(jobs_path, planned)
        atomic_json(state_path, {
            "schema_version": 1, "status": "dry_run", "updated_at": iso_now(),
            "total_jobs": len(specs), "completed_jobs": 0, "failed_jobs": 0,
            "interrupted_jobs": 0, "pending_jobs": len(specs),
            "engine_initialization_seconds": None, "engine_memory_budget": memory_budget,
        })
        metadata.update(status="dry_run_complete", completed_jobs=0, failed_jobs=0, interrupted_jobs=0, pending_jobs=len(specs), completed_at=iso_now())
        atomic_json(metadata_path, metadata)
        log_line(log_path, "END status=dry_run_complete completed=0 failed=0")
        print(f"Run metadata: {metadata_path}")
        print(f"Timestamped log: {log_path}")
        return 0

    metadata["status"] = "running"
    atomic_json(metadata_path, metadata)
    log_line(log_path, f"PERSISTENT_PROCVLM_START jobs={len(specs)} resume={resume} engine_reuse=true")
    worker_return_code = run_streamed(command, config["repo"], env, log_path)
    return finalize_procvlm_persistent_run(
        metadata=metadata, metadata_path=metadata_path, jobs_path=jobs_path,
        progress_path=progress_path, state_path=state_path, log_path=log_path,
        total_jobs=len(specs), worker_return_code=worker_return_code,
    )



def make_execution_environment(config: dict[str, Any], gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = str(config["repo"]) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def resume_procvlm_run(args: argparse.Namespace) -> int:
    """Resume a ProcVLM run from its atomic per-rollout progress files."""
    run_root = args.resume_run.expanduser().resolve()
    metadata_path = run_root / "run.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"ProcVLM run metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("baseline") != "procvlm":
        raise ValueError(f"--resume-run is only supported for ProcVLM runs: {run_root}")
    if metadata.get("dry_run"):
        raise ValueError(f"Cannot resume a dry-run plan: {run_root}")

    stored_arguments = metadata.get("arguments", {})
    manifest_value = metadata.get("manifest")
    if not manifest_value:
        raise ValueError(f"ProcVLM run metadata has no manifest: {metadata_path}")
    args.manifest = Path(manifest_value).expanduser().resolve()
    args.data_root = Path(metadata.get("data_root", PROJECT_ROOT)).expanduser().resolve()
    output_root = Path(metadata.get("output_root", run_root)).expanduser().resolve()
    if output_root != run_root:
        raise ValueError(f"Run metadata output_root does not match --resume-run: {output_root}")
    args.output_dir = run_root.parent
    fallback_log_dir = args.logs_dir or PROJECT_ROOT / "logs/baselines"
    args.logs_dir = Path(metadata.get("log_path", fallback_log_dir / f"procvlm_resume_{timestamp()}.log")).expanduser().resolve().parent
    args.model_path = Path(
        args.model_path or metadata.get("model_path") or BASELINES["procvlm"]["checkpoint"]
    ).expanduser().resolve()
    args.gpu = args.gpu or stored_arguments.get("gpu") or "0"
    args.vllm_free_memory_fraction = (
        args.vllm_free_memory_fraction
        if args.vllm_free_memory_fraction is not None
        else metadata.get("vllm_requested_free_fraction")
        or stored_arguments.get("vllm_free_memory_fraction")
        or 0.80
    )
    args.procvlm_window_size = int(stored_arguments.get("procvlm_window_size", 4))
    stored_max_frames = stored_arguments.get("procvlm_max_sampled_frames")
    args.procvlm_max_sampled_frames = (
        int(stored_max_frames) if stored_max_frames not in (None, "None") else None
    )
    args.procvlm_max_new_tokens = int(stored_arguments.get("procvlm_max_new_tokens", 4096))
    args.dtype = str(stored_arguments.get("dtype", "bf16"))
    args.tensor_parallel_size = int(stored_arguments.get("tensor_parallel_size", 1))
    args.dry_run = False

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {args.manifest}")
    recorded_hash = metadata.get("manifest_sha256")
    if recorded_hash and file_sha256(args.manifest) != recorded_hash:
        raise ValueError(f"Manifest changed since the original run: {args.manifest}")

    config = BASELINES["procvlm"]
    validate_static(config, args.model_path)
    plan_path = Path(
        metadata.get("procvlm_job_plan", run_root / "procvlm_jobs.jsonl")
    ).expanduser().resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"ProcVLM job plan does not exist: {plan_path}")
    records = load_jsonl(plan_path)
    if not records:
        raise ValueError(f"ProcVLM job plan is empty: {plan_path}")

    log_path = Path(metadata.get("log_path", fallback_log_dir / f"procvlm_resume_{timestamp()}.log")).expanduser().resolve()
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

    resume_count = int(metadata.get("resume_count", 0)) + 1
    metadata["resume_count"] = resume_count
    metadata["last_resume_at"] = iso_now()
    metadata["last_resume_overrides"] = {
        "gpu": args.gpu,
        "requested_free_memory_fraction": args.vllm_free_memory_fraction,
    }
    metadata["status"] = "resuming"
    atomic_json(metadata_path, metadata)
    return run_procvlm_persistent(
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


def filter_records(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply manifest filters before any positional range selection."""
    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest has duplicate rollout IDs")
    selected = records
    if args.partition != "all":
        selected = [record for record in selected if record.get("analysis_partition") == args.partition]
    if args.dataset_role:
        selected = [record for record in selected if record.get("dataset_role") == args.dataset_role]
    if args.rollout_id:
        requested = set(args.rollout_id)
        missing = sorted(requested - set(ids))
        if missing:
            raise ValueError(f"Requested rollout IDs not in manifest: {missing}")
        selected = [record for record in selected if record["id"] in requested]
    return selected


def selection_bounds(args: argparse.Namespace, record_count: int) -> tuple[int, int]:
    start = int(getattr(args, "start_index", 0) or 0)
    end_value = getattr(args, "end_index", None)
    limit = getattr(args, "limit", None)
    if start < 0:
        raise ValueError("--start-index must be non-negative")
    if end_value is not None and limit is not None:
        raise ValueError("--end-index cannot be combined with --limit")
    if end_value is not None:
        end = int(end_value)
        if end < start:
            raise ValueError("--end-index must be greater than or equal to --start-index")
    elif limit is not None:
        end = start + int(limit)
    else:
        end = record_count
    return start, min(end, record_count)


def select_records(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = filter_records(args, records)
    start, end = selection_bounds(args, len(selected))
    selected = selected[start:end]
    if not selected:
        raise ValueError("No rollouts matched the requested filters")
    return selected


def parse_worker_spec(value: str) -> dict[str, int | str]:
    """Parse one baseline worker assignment: GPU:START:END."""
    parts = str(value).strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid --worker-spec {value!r}; expected GPU:START:END")
    gpu, start_text, end_text = (part.strip() for part in parts)
    if not gpu.isdigit():
        raise ValueError(f"Worker GPU must be one numeric CUDA device index: {gpu!r}")
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError as error:
        raise ValueError(f"Worker range must be integer-valued: {value!r}") from error
    if start < 0 or end <= start:
        raise ValueError(f"Worker range must satisfy 0 <= start < end: {value!r}")
    return {"gpu": gpu, "start_index": start, "end_index": end}


def build_worker_plan(
    scope_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a scope-relative, right-open worker plan for any baseline."""
    total_start, total_end = selection_bounds(args, len(scope_records))
    if total_end <= total_start:
        raise ValueError("The requested total range selects no rollouts")

    raw_specs = list(getattr(args, "worker_spec", []) or [])
    if raw_specs:
        if getattr(args, "parallel_workers", 1) not in (1, len(raw_specs)):
            raise ValueError(
                "--parallel-workers must be 1 or equal the number of --worker-spec values"
            )
        workers = [parse_worker_spec(value) for value in raw_specs]
    else:
        worker_count = int(getattr(args, "parallel_workers", 1) or 1)
        if worker_count < 1:
            raise ValueError("--parallel-workers must be positive")
        gpu_ids = selected_gpu_ids(getattr(args, "gpu", "0") or "0")
        workers = []
        total = total_end - total_start
        base, remainder = divmod(total, worker_count)
        cursor = total_start
        for worker_index in range(worker_count):
            width = base + (1 if worker_index < remainder else 0)
            workers.append({
                "gpu": str(gpu_ids[worker_index % len(gpu_ids)]),
                "start_index": cursor,
                "end_index": cursor + width,
            })
            cursor += width

    assignments: list[dict[str, Any]] = []
    index_owners: dict[int, int] = {}
    overlaps: set[int] = set()
    gaps: set[int] = set(range(total_start, total_end))
    for worker_index, worker in enumerate(workers):
        start = int(worker["start_index"])
        end = int(worker["end_index"])
        if start < total_start or end > total_end or end <= start:
            raise ValueError(
                f"Worker {worker_index} range [{start},{end}) must be inside "
                f"total range [{total_start},{total_end})"
            )
        worker_items: list[dict[str, Any]] = []
        duplicate_items: list[dict[str, Any]] = []
        for scope_index in range(start, end):
            gaps.discard(scope_index)
            owner = index_owners.get(scope_index)
            if owner is not None:
                overlaps.add(scope_index)
                duplicate_items.append({
                    "scope_index": scope_index,
                    "record": scope_records[scope_index],
                    "owner_worker_index": owner,
                })
                continue
            index_owners[scope_index] = worker_index
            worker_items.append({
                "scope_index": scope_index,
                "record": scope_records[scope_index],
            })
        assignments.append({
            "worker_index": worker_index,
            "gpu": str(worker["gpu"]),
            "start_index": start,
            "end_index": end,
            "requested_count": end - start,
            "unique_count": len(worker_items),
            "duplicate_count": len(duplicate_items),
            "items": worker_items,
            "duplicates": duplicate_items,
        })

    unique_items = [
        item
        for assignment in assignments
        for item in assignment["items"]
    ]
    return {
        "total_start_index": total_start,
        "total_end_index": total_end,
        "total_count": total_end - total_start,
        "workers": assignments,
        "unique_items": unique_items,
        "unique_count": len(unique_items),
        "overlaps": sorted(overlaps),
        "gaps": sorted(gaps),
    }


def build_rynn_worker_plan(
    scope_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Backward-compatible name for the generic worker planner."""
    return build_worker_plan(scope_records, args)


def validate_static(config: dict[str, Any], model_path: Path | None) -> None:
    required = [config["python"], config["repo"], config["entrypoint"]]
    if model_path is not None:
        required.append(model_path)
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Required baseline paths are missing: {missing}")
    if not os.access(config["python"], os.X_OK):
        raise PermissionError(f"Environment Python is not executable: {config['python']}")


def command_for(
    args: argparse.Namespace,
    config: dict[str, Any],
    record: dict[str, Any],
    job_dir: Path,
    model_path: Path | None,
    vllm_total_memory_fraction: float | None = None,
) -> tuple[list[str], Path]:
    python = str(config["python"])
    task = str(record["task_description"])
    vllm_memory = (
        args.vllm_free_memory_fraction
        if vllm_total_memory_fraction is None
        else vllm_total_memory_fraction
    )
    if args.baseline == "safe":
        input_csv = resolve_record_path(record["csv_path"], args.data_root)
        command = [
            python,
            str(TOOLS_ROOT / "safe_worker.py"),
            "--input-csv", str(input_csv),
            "--output-csv", str(job_dir / "safe_features.csv"),
            "--rollout-id", record["id"],
        ]
        return command, config["repo"]

    video = resolve_record_path(record["video_path"], args.data_root)
    if args.baseline == "procvlm":
        command = [
            python, "-m", "evqa.inference",
            "--video_path", str(video),
            "--task", task,
            "--model_path", str(model_path),
            "--output_path", str(job_dir / "procvlm_raw.jsonl"),
            "--window_size", str(args.procvlm_window_size),
            "--max_new_tokens", str(args.procvlm_max_new_tokens),
            "--torch_dtype", args.dtype,
            "--tp", str(args.tensor_parallel_size),
            "--gpu_memory_utilization", str(vllm_memory),
        ]
        if args.procvlm_max_sampled_frames is not None:
            command.extend(["--max_sampled_frames", str(args.procvlm_max_sampled_frames)])
        return command, config["repo"]

    if args.baseline == "rynnvalue":
        total_frames_value = record.get("total_frames")
        if total_frames_value is None:
            raise ValueError(f"RynnValue rollout {record['id']} is missing total_frames")
        try:
            total_frames = int(total_frames_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"RynnValue rollout {record['id']} has invalid total_frames: {total_frames_value!r}"
            ) from error
        evaluation_steps = approximate_num_steps_for_interval(
            total_frames, args.rynn_evaluation_interval
        )
        command = [
            python,
            str(TOOLS_ROOT / "rynnvalue_worker.py"),
            "--repo", str(config["repo"]),
            "--model-path", str(model_path),
            "--video-path", str(video),
            "--instruction", task,
            "--output-dir", str(job_dir),
            "--num-frames", str(args.rynn_num_frames),
            "--num-steps", str(evaluation_steps),
            "--evaluation-interval", str(args.rynn_evaluation_interval),
            "--batch-size", str(args.rynn_batch_size),
            "--max-image-side", str(args.rynn_max_image_side),
            "--max-new-tokens", str(args.rynn_max_new_tokens),
            "--robot-description", args.robot_description,
            "--camera-description", args.camera_description,
        ]
        if args.render_video:
            command.append("--render-video")
        return command, config["repo"] / "rynn_infer"

    if args.baseline == "robo_dopamine":
        raise AssertionError("Robo-Dopamine uses the persistent worker runner")
    raise AssertionError(args.baseline)


def run_streamed(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    *,
    log_lock: threading.Lock | None = None,
    progress_callback: Callable[[], None] | None = None,
) -> int:
    log_line(
        log_path,
        "COMMAND " + json.dumps(command, ensure_ascii=False),
        lock=log_lock,
    )
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if log_lock is None:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
        else:
            with log_lock:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
        print(line, end="", flush=True)
        if progress_callback is not None:
            progress_callback()
    return_code = process.wait()
    if progress_callback is not None:
        progress_callback()
    return return_code



def _public_worker_progress(worker: dict[str, Any]) -> dict[str, Any]:
    return {
        key: worker.get(key)
        for key in (
            "worker_index", "gpu", "start_index", "end_index", "requested_count",
            "unique_count", "duplicate_count", "completed_jobs", "failed_jobs",
            "pending_jobs", "status", "started_at", "completed_at", "error",
        )
    }


def run_rynn_parallel(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    scope_records: list[dict[str, Any]],
    model_path: Path,
    run_root: Path,
    raw_root: Path,
    log_path: Path,
    metadata_path: Path,
    jobs_path: Path,
    commands_path: Path,
    metadata: dict[str, Any],
    plan: dict[str, Any],
) -> int:
    """Run RynnValue rollout shards concurrently while preserving one run root."""
    state_lock = threading.RLock()
    file_lock = threading.Lock()
    worker_progress: list[dict[str, Any]] = []
    for assignment in plan["workers"]:
        worker_progress.append({
            "worker_index": assignment["worker_index"],
            "gpu": assignment["gpu"],
            "start_index": assignment["start_index"],
            "end_index": assignment["end_index"],
            "requested_count": assignment["requested_count"],
            "unique_count": assignment["unique_count"],
            "duplicate_count": assignment["duplicate_count"],
            "completed_jobs": 0,
            "failed_jobs": 0,
            "pending_jobs": assignment["unique_count"],
            "status": "queued" if not args.dry_run else "planned",
            "started_at": None,
            "completed_at": None,
            "error": None,
        })

    metadata.update({
        "selection_start_index": plan["total_start_index"],
        "selection_end_index": plan["total_end_index"],
        "selection_range": f"[{plan['total_start_index']},{plan['total_end_index']})",
        "requested_rollouts": plan["total_count"],
        "selected_rollouts": plan["total_count"],
        "unique_selected_rollouts": plan["unique_count"],
        "parallel_workers": len(plan["workers"]),
        "worker_assignments": [
            {
                key: assignment[key]
                for key in (
                    "worker_index", "gpu", "start_index", "end_index",
                    "requested_count", "unique_count", "duplicate_count",
                )
            }
            for assignment in plan["workers"]
        ],
        "worker_progress": worker_progress,
        "overlaps": plan["overlaps"],
        "gaps": plan["gaps"],
        "duplicate_assignments": sum(item["duplicate_count"] for item in plan["workers"]),
        "completed_jobs": 0,
        "failed_jobs": 0,
        "pending_jobs": plan["unique_count"],
    })
    atomic_json(metadata_path, metadata)
    if plan["overlaps"]:
        log_line(log_path, f"WARNING overlapping worker assignments at {plan['overlaps']}")
    if plan["gaps"]:
        log_line(log_path, f"WARNING uncovered worker assignment indices at {plan['gaps']}")

    def save_progress() -> None:
        with state_lock:
            metadata["worker_progress"] = [
                _public_worker_progress(worker) for worker in worker_progress
            ]
            metadata["completed_jobs"] = sum(
                int(worker.get("completed_jobs") or 0) for worker in worker_progress
            )
            metadata["failed_jobs"] = sum(
                int(worker.get("failed_jobs") or 0) for worker in worker_progress
            )
            metadata["pending_jobs"] = (
                plan["unique_count"] - metadata["completed_jobs"] - metadata["failed_jobs"]
            )
            atomic_json(metadata_path, metadata)

    def append_job(payload: dict[str, Any]) -> None:
        with file_lock:
            append_jsonl(jobs_path, payload)

    def append_command(payload: dict[str, Any]) -> None:
        with file_lock:
            append_jsonl(commands_path, payload)

    def run_worker(assignment: dict[str, Any]) -> None:
        worker_index = int(assignment["worker_index"])
        worker = worker_progress[worker_index]
        try:
            worker_environment = make_execution_environment(config, str(assignment["gpu"]))
            with state_lock:
                worker["status"] = "running"
                worker["started_at"] = iso_now()
            save_progress()
            for duplicate in assignment["duplicates"]:
                record = duplicate["record"]
                duplicate_job = {
                    "job_index": int(duplicate["scope_index"]),
                    "scope_index": int(duplicate["scope_index"]),
                    "rollout_id": record["id"],
                    "worker_index": worker_index,
                    "worker_gpu": str(assignment["gpu"]),
                    "worker_start_index": assignment["start_index"],
                    "worker_end_index": assignment["end_index"],
                    "status": "duplicate_assignment",
                    "return_code": None,
                    "duplicate_of_worker_index": duplicate["owner_worker_index"],
                    "completed_at": iso_now(),
                }
                append_job(duplicate_job)

            for item in assignment["items"]:
                scope_index = int(item["scope_index"])
                record = item["record"]
                rollout_id = str(record["id"])
                job_dir = raw_root / rollout_id
                started_at = iso_now()
                command_record: dict[str, Any] = {
                    "job_index": scope_index,
                    "scope_index": scope_index,
                    "rollout_id": rollout_id,
                    "worker_index": worker_index,
                    "worker_gpu": str(assignment["gpu"]),
                    "worker_start_index": assignment["start_index"],
                    "worker_end_index": assignment["end_index"],
                }
                try:
                    command, cwd = command_for(
                        args, config, record, job_dir, model_path, None
                    )
                    required_input = resolve_record_path(
                        record["video_path"], args.data_root
                    )
                    if not required_input.is_file():
                        raise FileNotFoundError(
                            f"Input for {rollout_id} does not exist: {required_input}"
                        )
                    command_record.update({
                        "cwd": str(cwd),
                        "argv": command,
                        "shell_preview": shlex.join(command),
                        "sampling": {
                            "total_frames": int(record["total_frames"]),
                            "target_interval": args.rynn_evaluation_interval,
                            "approximate_num_steps": approximate_num_steps_for_interval(
                                int(record["total_frames"]), args.rynn_evaluation_interval
                            ),
                            "endpoint_sampler": "official_uniform_prefix",
                        },
                    })
                    append_command(command_record)
                    if args.dry_run:
                        append_job({
                            **command_record,
                            "status": "planned",
                            "raw_output_dir": str(job_dir),
                            "started_at": started_at,
                            "completed_at": iso_now(),
                        })
                        with state_lock:
                            worker["completed_jobs"] += 1
                            worker["pending_jobs"] -= 1
                        save_progress()
                        log_line(
                            log_path,
                            f"PLAN worker={worker_index} index={scope_index} rollout={rollout_id}",
                            lock=file_lock,
                        )
                        continue

                    job_dir.mkdir(parents=True, exist_ok=False)
                    log_line(
                        log_path,
                        f"RUN worker={worker_index} gpu={assignment['gpu']} "
                        f"index={scope_index} rollout={rollout_id}",
                        lock=file_lock,
                    )
                    return_code = run_streamed(
                        command,
                        cwd,
                        worker_environment,
                        log_path,
                        log_lock=file_lock,
                    )
                    error_message = None
                except Exception as error:
                    return_code = None
                    error_message = str(error)
                    log_line(
                        log_path,
                        f"ERROR worker={worker_index} rollout={rollout_id} {error_message}",
                        lock=file_lock,
                    )
                raw_files = (
                    sorted(str(path) for path in job_dir.rglob("*") if path.is_file())
                    if job_dir.is_dir() else []
                )
                status = "complete" if return_code == 0 else "failed"
                job_result = {
                    **command_record,
                    "status": status,
                    "return_code": return_code,
                    "error": error_message,
                    "started_at": started_at,
                    "completed_at": iso_now(),
                    "raw_output_dir": str(job_dir),
                    "raw_output_files": raw_files,
                }
                append_job(job_result)
                with state_lock:
                    worker["pending_jobs"] -= 1
                    if status == "complete":
                        worker["completed_jobs"] += 1
                    else:
                        worker["failed_jobs"] += 1
                save_progress()
        except Exception as error:
            with state_lock:
                worker["status"] = "failed"
                worker["error"] = str(error)
            save_progress()
            log_line(
                log_path,
                f"WORKER_FAILED worker={worker_index} error={error}",
                lock=file_lock,
            )
            return
        with state_lock:
            worker["status"] = "complete_with_errors" if worker["failed_jobs"] else "complete"
            worker["completed_at"] = iso_now()
        save_progress()

    metadata["status"] = "dry_run" if args.dry_run else "running"
    atomic_json(metadata_path, metadata)
    log_line(
        log_path,
        f"RYNNVALUE_START range=[{plan['total_start_index']},{plan['total_end_index']}) "
        f"workers={len(plan['workers'])} unique={plan['unique_count']} dry_run={args.dry_run}",
    )
    if args.dry_run:
        for assignment in plan["workers"]:
            run_worker(assignment)
    else:
        with ThreadPoolExecutor(
            max_workers=max(1, len(plan["workers"])),
            thread_name_prefix="rynnvalue-worker",
        ) as executor:
            futures = [executor.submit(run_worker, assignment) for assignment in plan["workers"]]
            for future in as_completed(futures):
                future.result()

    with state_lock:
        metadata["worker_progress"] = [
            _public_worker_progress(worker) for worker in worker_progress
        ]
        metadata["completed_jobs"] = sum(
            int(worker.get("completed_jobs") or 0) for worker in worker_progress
        )
        metadata["failed_jobs"] = sum(
            int(worker.get("failed_jobs") or 0) for worker in worker_progress
        )
        metadata["pending_jobs"] = (
            plan["unique_count"] - metadata["completed_jobs"] - metadata["failed_jobs"]
        )
        worker_failures = any(worker.get("status") == "failed" for worker in worker_progress)
        if args.dry_run:
            metadata["status"] = "dry_run_complete"
        elif metadata["failed_jobs"] or metadata["pending_jobs"] or plan["gaps"] or worker_failures:
            metadata["status"] = "complete_with_errors"
        else:
            metadata["status"] = "complete"
        metadata["completed_at"] = iso_now()
        atomic_json(metadata_path, metadata)
    log_line(
        log_path,
        f"END status={metadata['status']} completed={metadata['completed_jobs']} "
        f"failed={metadata['failed_jobs']} duplicates={metadata['duplicate_assignments']}",
    )
    print(f"Run metadata: {metadata_path}")
    print(f"Timestamped log: {log_path}")
    return 0 if (
        not metadata["failed_jobs"]
        and not metadata["pending_jobs"]
        and not plan["gaps"]
        and not any(worker.get("status") == "failed" for worker in worker_progress)
    ) else 1



def _parallel_progress_records(plan: dict[str, Any], *, planned: bool = False) -> list[dict[str, Any]]:
    return [
        {
            "worker_index": assignment["worker_index"],
            "gpu": assignment["gpu"],
            "start_index": assignment["start_index"],
            "end_index": assignment["end_index"],
            "requested_count": assignment["requested_count"],
            "unique_count": assignment["unique_count"],
            "duplicate_count": assignment["duplicate_count"],
            "completed_jobs": 0,
            "failed_jobs": 0,
            "pending_jobs": assignment["unique_count"],
            "status": "planned" if planned else "queued",
            "started_at": None,
            "completed_at": None,
            "error": None,
        }
        for assignment in plan["workers"]
    ]


def _parallel_metadata_update(
    metadata: dict[str, Any],
    plan: dict[str, Any],
    worker_progress: list[dict[str, Any]],
) -> None:
    metadata.update({
        "selection_start_index": plan["total_start_index"],
        "selection_end_index": plan["total_end_index"],
        "selection_range": f"[{plan['total_start_index']},{plan['total_end_index']})",
        "requested_rollouts": plan["total_count"],
        "selected_rollouts": plan["total_count"],
        "unique_selected_rollouts": plan["unique_count"],
        "parallel_workers": len(plan["workers"]),
        "worker_assignments": [
            {
                key: assignment[key]
                for key in (
                    "worker_index", "gpu", "start_index", "end_index",
                    "requested_count", "unique_count", "duplicate_count",
                )
            }
            for assignment in plan["workers"]
        ],
        "worker_progress": worker_progress,
        "overlaps": plan["overlaps"],
        "gaps": plan["gaps"],
        "duplicate_assignments": sum(
            item["duplicate_count"] for item in plan["workers"]
        ),
        "completed_jobs": 0,
        "failed_jobs": 0,
        "pending_jobs": plan["unique_count"],
    })


def _parallel_job_count(records: list[dict[str, Any]]) -> tuple[int, int, int]:
    completed = sum(record.get("status") == "complete" for record in records)
    failed = sum(
        record.get("status") in {"failed", "interrupted", "fatal_engine_failure"}
        for record in records
    )
    duplicates = sum(record.get("status") == "duplicate_assignment" for record in records)
    return completed, failed, duplicates


def run_simple_parallel(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    model_path: Path | None,
    run_root: Path,
    raw_root: Path,
    log_path: Path,
    metadata_path: Path,
    jobs_path: Path,
    commands_path: Path,
    metadata: dict[str, Any],
    plan: dict[str, Any],
) -> int:
    """Run non-persistent per-rollout workers concurrently (currently SAFE)."""
    state_lock = threading.RLock()
    file_lock = threading.Lock()
    worker_progress = _parallel_progress_records(plan, planned=args.dry_run)
    _parallel_metadata_update(metadata, plan, worker_progress)
    atomic_json(metadata_path, metadata)

    def save_progress() -> None:
        with state_lock:
            metadata["worker_progress"] = [
                _public_worker_progress(worker) for worker in worker_progress
            ]
            metadata["completed_jobs"] = sum(
                int(worker.get("completed_jobs") or 0) for worker in worker_progress
            )
            metadata["failed_jobs"] = sum(
                int(worker.get("failed_jobs") or 0) for worker in worker_progress
            )
            metadata["pending_jobs"] = (
                plan["unique_count"] - metadata["completed_jobs"] - metadata["failed_jobs"]
            )
            atomic_json(metadata_path, metadata)

    def append_job(record: dict[str, Any]) -> None:
        with file_lock:
            append_jsonl(jobs_path, record)

    def append_command(record: dict[str, Any]) -> None:
        with file_lock:
            append_jsonl(commands_path, record)

    def run_worker(assignment: dict[str, Any]) -> None:
        worker_index = int(assignment["worker_index"])
        worker = worker_progress[worker_index]
        try:
            worker_environment = make_execution_environment(
                config, str(assignment["gpu"])
            )
            with state_lock:
                worker["status"] = "running"
                worker["started_at"] = iso_now()
            save_progress()
            for duplicate in assignment["duplicates"]:
                record = duplicate["record"]
                append_job({
                    "job_index": int(duplicate["scope_index"]),
                    "scope_index": int(duplicate["scope_index"]),
                    "rollout_id": record["id"],
                    "worker_index": worker_index,
                    "worker_gpu": str(assignment["gpu"]),
                    "worker_start_index": assignment["start_index"],
                    "worker_end_index": assignment["end_index"],
                    "status": "duplicate_assignment",
                    "return_code": None,
                    "duplicate_of_worker_index": duplicate["owner_worker_index"],
                    "completed_at": iso_now(),
                })

            for item in assignment["items"]:
                scope_index = int(item["scope_index"])
                record = item["record"]
                rollout_id = str(record["id"])
                job_dir = raw_root / rollout_id
                started_at = iso_now()
                command_record: dict[str, Any] = {
                    "job_index": scope_index,
                    "scope_index": scope_index,
                    "rollout_id": rollout_id,
                    "worker_index": worker_index,
                    "worker_gpu": str(assignment["gpu"]),
                    "worker_start_index": assignment["start_index"],
                    "worker_end_index": assignment["end_index"],
                }
                try:
                    command, cwd = command_for(
                        args, config, record, job_dir, model_path, None
                    )
                    input_key = "csv_path" if args.baseline == "safe" else "video_path"
                    required_input = resolve_record_path(record[input_key], args.data_root)
                    if not required_input.is_file():
                        raise FileNotFoundError(
                            f"Input for {rollout_id} does not exist: {required_input}"
                        )
                    command_record.update({
                        "cwd": str(cwd),
                        "argv": command,
                        "shell_preview": shlex.join(command),
                        "execution_scope": "parallel_rollout_worker",
                    })
                    append_command(command_record)
                    if args.dry_run:
                        append_job({
                            **command_record,
                            "status": "planned",
                            "raw_output_dir": str(job_dir),
                            "started_at": started_at,
                            "completed_at": iso_now(),
                        })
                        with state_lock:
                            worker["completed_jobs"] += 1
                            worker["pending_jobs"] -= 1
                        save_progress()
                        continue

                    job_dir.mkdir(parents=True, exist_ok=False)
                    log_line(
                        log_path,
                        f"RUN worker={worker_index} gpu={assignment['gpu']} "
                        f"index={scope_index} rollout={rollout_id}",
                        lock=file_lock,
                    )
                    return_code = run_streamed(
                        command,
                        cwd,
                        worker_environment,
                        log_path,
                        log_lock=file_lock,
                    )
                    error_message = None
                except Exception as error:
                    return_code = None
                    error_message = str(error)
                    log_line(
                        log_path,
                        f"ERROR worker={worker_index} rollout={rollout_id} {error_message}",
                        lock=file_lock,
                    )
                raw_files = (
                    sorted(str(path) for path in job_dir.rglob("*") if path.is_file())
                    if job_dir.is_dir() else []
                )
                status = "complete" if return_code == 0 else "failed"
                append_job({
                    **command_record,
                    "status": status,
                    "return_code": return_code,
                    "error": error_message,
                    "started_at": started_at,
                    "completed_at": iso_now(),
                    "raw_output_dir": str(job_dir),
                    "raw_output_files": raw_files,
                })
                with state_lock:
                    worker["pending_jobs"] -= 1
                    if status == "complete":
                        worker["completed_jobs"] += 1
                    else:
                        worker["failed_jobs"] += 1
                save_progress()
        except Exception as error:
            with state_lock:
                worker["status"] = "failed"
                worker["error"] = str(error)
            save_progress()
            log_line(
                log_path,
                f"WORKER_FAILED worker={worker_index} error={error}",
                lock=file_lock,
            )
            return
        with state_lock:
            worker["status"] = (
                "complete_with_errors" if worker["failed_jobs"] else "complete"
            )
            worker["completed_at"] = iso_now()
        save_progress()

    metadata["status"] = "dry_run" if args.dry_run else "running"
    atomic_json(metadata_path, metadata)
    log_line(
        log_path,
        f"{args.baseline.upper()}_START range=[{plan['total_start_index']},{plan['total_end_index']}) "
        f"workers={len(plan['workers'])} unique={plan['unique_count']} dry_run={args.dry_run}",
    )
    if args.dry_run:
        for assignment in plan["workers"]:
            run_worker(assignment)
    else:
        with ThreadPoolExecutor(
            max_workers=max(1, len(plan["workers"])),
            thread_name_prefix=f"{args.baseline}-worker",
        ) as executor:
            futures = [executor.submit(run_worker, assignment) for assignment in plan["workers"]]
            for future in as_completed(futures):
                future.result()

    with state_lock:
        metadata["worker_progress"] = [
            _public_worker_progress(worker) for worker in worker_progress
        ]
        metadata["completed_jobs"] = sum(
            int(worker.get("completed_jobs") or 0) for worker in worker_progress
        )
        metadata["failed_jobs"] = sum(
            int(worker.get("failed_jobs") or 0) for worker in worker_progress
        )
        metadata["pending_jobs"] = (
            plan["unique_count"] - metadata["completed_jobs"] - metadata["failed_jobs"]
        )
        worker_failures = any(worker.get("status") == "failed" for worker in worker_progress)
        if args.dry_run:
            metadata["status"] = "dry_run_complete"
        elif metadata["failed_jobs"] or metadata["pending_jobs"] or plan["gaps"] or worker_failures:
            metadata["status"] = "complete_with_errors"
        else:
            metadata["status"] = "complete"
        metadata["completed_at"] = iso_now()
        atomic_json(metadata_path, metadata)
    log_line(
        log_path,
        f"END status={metadata['status']} completed={metadata['completed_jobs']} "
        f"failed={metadata['failed_jobs']} duplicates={metadata['duplicate_assignments']}",
    )
    print(f"Run metadata: {metadata_path}")
    print(f"Timestamped log: {log_path}")
    return 0 if (
        not metadata["failed_jobs"]
        and not metadata["pending_jobs"]
        and not plan["gaps"]
        and not any(worker.get("status") == "failed" for worker in worker_progress)
    ) else 1


def _persistent_worker_record(
    record: dict[str, Any],
    scope_index: int,
    raw_root: Path,
    baseline: str,
) -> dict[str, Any]:
    value = dict(record)
    rollout_id = str(record["id"])
    output_dir = raw_root / rollout_id
    value.update({
        "job_index": scope_index,
        "rollout_id": rollout_id,
        "raw_output_dir": str(output_dir),
    })
    if baseline == "procvlm":
        value["output_path"] = str(output_dir / "procvlm_raw.jsonl")
    elif baseline == "densereward":
        value["output_path"] = str(output_dir / "densereward_raw.jsonl")
    return value


def run_persistent_parallel(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    model_path: Path,
    run_root: Path,
    raw_root: Path,
    log_path: Path,
    metadata_path: Path,
    jobs_path: Path,
    commands_path: Path,
    env: dict[str, str],
    metadata: dict[str, Any],
    plan: dict[str, Any],
) -> int:
    """Launch one independent persistent model worker per rollout shard."""
    del env  # Every shard receives its own CUDA_VISIBLE_DEVICES environment.
    state_lock = threading.RLock()
    file_lock = threading.Lock()
    worker_progress = _parallel_progress_records(plan, planned=args.dry_run)
    _parallel_metadata_update(metadata, plan, worker_progress)
    worker_contexts: list[dict[str, Any]] = []
    aggregate_commands: list[dict[str, Any]] = []

    for assignment in plan["workers"]:
        worker_index = int(assignment["worker_index"])
        worker_root = run_root / "workers" / f"worker-{worker_index}"
        worker_root.mkdir(parents=True, exist_ok=True)
        worker_plan_path = worker_root / f"{args.baseline}_jobs.jsonl"
        worker_jobs_path = worker_root / "jobs.jsonl"
        worker_progress_path = worker_root / "progress.jsonl"
        worker_state_path = worker_root / "state.json"
        worker_records = [
            _persistent_worker_record(
                item["record"], int(item["scope_index"]), raw_root, args.baseline
            )
            for item in assignment["items"]
        ]
        if args.baseline == "procvlm":
            specs = build_procvlm_job_specs(worker_records, args, raw_root)
        elif args.baseline == "robo_dopamine":
            from robo_dopamine_runner import build_job_specs

            specs = build_job_specs(worker_records, args, config, raw_root)
        elif args.baseline == "densereward":
            specs = build_densereward_job_specs(worker_records, args, raw_root)
        else:
            raise ValueError(
                f"Persistent parallel workers are not supported for {args.baseline}"
            )

        setup_error: str | None = None
        if args.dry_run and args.baseline in VLLM_BASELINES:
            memory_budget: dict[str, Any] = {
                "scope": "free_gpu_memory",
                "requested_free_fraction": args.vllm_free_memory_fraction,
                "resolved_total_fraction": None,
                "resolution": "deferred_until_execution",
            }
            total_memory_fraction = None
        elif args.dry_run:
            memory_budget = {
                "scope": "not_applicable",
                "requested_free_fraction": args.vllm_free_memory_fraction,
                "resolution": "not_used",
                "description": "DenseReward uses Transformers device_map=auto; no vLLM memory conversion is applied",
            }
            total_memory_fraction = None
        elif args.baseline in VLLM_BASELINES:
            try:
                total_memory_fraction, memory_budget = resolve_vllm_memory_budget(
                    str(assignment["gpu"]), args.vllm_free_memory_fraction
                )
            except ValueError as error:
                total_memory_fraction = None
                memory_budget = {
                    "scope": "free_gpu_memory",
                    "requested_free_fraction": args.vllm_free_memory_fraction,
                    "error": str(error),
                }
                setup_error = str(error)
        else:
            total_memory_fraction = None
            memory_budget = {
                "scope": "not_applicable",
                "requested_free_fraction": args.vllm_free_memory_fraction,
                "resolution": "not_used",
                "description": "DenseReward uses Transformers device_map=auto; no vLLM memory conversion is applied",
            }

        if args.baseline == "procvlm":
            command = build_procvlm_worker_command(
                args,
                config,
                model_path,
                worker_plan_path,
                worker_jobs_path,
                worker_progress_path,
                worker_state_path,
                memory_budget,
                total_memory_fraction,
                resume=False,
            )
        elif args.baseline == "robo_dopamine":
            from robo_dopamine_runner import build_worker_command

            command = build_worker_command(
                args,
                config,
                model_path,
                worker_plan_path,
                worker_jobs_path,
                worker_progress_path,
                worker_state_path,
                memory_budget,
                total_memory_fraction,
                resume=False,
            )
        else:
            command = build_densereward_worker_command(
                args,
                config,
                model_path,
                worker_plan_path,
                worker_jobs_path,
                worker_progress_path,
                worker_state_path,
                resume=False,
            )
        decorated_specs = []
        for spec in specs:
            decorated = {
                **spec,
                "worker_index": worker_index,
                "worker_gpu": str(assignment["gpu"]),
                "worker_start_index": assignment["start_index"],
                "worker_end_index": assignment["end_index"],
                "cwd": str(config["repo"]),
                "argv": command,
                "shell_preview": shlex.join(command),
                "vllm_memory_budget": memory_budget,
                "execution_scope": f"persistent_{args.baseline}_worker",
            }
            decorated_specs.append(decorated)
            aggregate_commands.append({
                key: decorated[key]
                for key in (
                    "job_index", "rollout_id", "worker_index", "worker_gpu",
                    "worker_start_index", "worker_end_index", "cwd", "argv",
                    "shell_preview", "vllm_memory_budget", "execution_scope",
                )
            })
        write_jsonl_atomic(worker_plan_path, decorated_specs)
        worker_contexts.append({
            "assignment": assignment,
            "worker_index": worker_index,
            "worker_root": worker_root,
            "plan_path": worker_plan_path,
            "jobs_path": worker_jobs_path,
            "progress_path": worker_progress_path,
            "state_path": worker_state_path,
            "specs": decorated_specs,
            "command": command,
            "memory_budget": memory_budget,
            "setup_error": setup_error,
            "return_code": None,
            "worker_jobs": [],
        })

    metadata["persistent_worker_runs"] = [
        {
            "worker_index": context["worker_index"],
            "gpu": context["assignment"]["gpu"],
            "plan_path": str(context["plan_path"]),
            "jobs_path": str(context["jobs_path"]),
            "progress_path": str(context["progress_path"]),
            "state_path": str(context["state_path"]),
            "argv": context["command"],
            "memory_budget": context["memory_budget"],
            "setup_error": context["setup_error"],
        }
        for context in worker_contexts
    ]
    metadata["worker_memory_budgets"] = [
        context["memory_budget"] for context in worker_contexts
    ]
    for command_record in aggregate_commands:
        append_jsonl(commands_path, command_record)
    for assignment in plan["workers"]:
        for duplicate in assignment["duplicates"]:
            append_jsonl(commands_path, {
                "job_index": int(duplicate["scope_index"]),
                "scope_index": int(duplicate["scope_index"]),
                "rollout_id": duplicate["record"]["id"],
                "worker_index": int(assignment["worker_index"]),
                "worker_gpu": str(assignment["gpu"]),
                "worker_start_index": assignment["start_index"],
                "worker_end_index": assignment["end_index"],
                "status": "duplicate_assignment",
            })
    atomic_json(metadata_path, metadata)

    def sync_worker(context: dict[str, Any], *, save: bool = True) -> None:
        worker_index = context["worker_index"]
        jobs = load_jsonl_if_exists(context["jobs_path"])
        completed, failed, _duplicates = _parallel_job_count(jobs)
        worker = worker_progress[worker_index]
        with state_lock:
            worker["completed_jobs"] = completed
            worker["failed_jobs"] = failed
            worker["pending_jobs"] = max(
                int(context["assignment"]["unique_count"]) - completed - failed,
                0,
            )
        if save:
            save_progress()

    def save_progress() -> None:
        with state_lock:
            metadata["worker_progress"] = [
                _public_worker_progress(worker) for worker in worker_progress
            ]
            metadata["completed_jobs"] = sum(
                int(worker.get("completed_jobs") or 0) for worker in worker_progress
            )
            metadata["failed_jobs"] = sum(
                int(worker.get("failed_jobs") or 0) for worker in worker_progress
            )
            metadata["pending_jobs"] = (
                plan["unique_count"] - metadata["completed_jobs"] - metadata["failed_jobs"]
            )
            atomic_json(metadata_path, metadata)

    def run_worker(context: dict[str, Any]) -> None:
        worker_index = context["worker_index"]
        worker = worker_progress[worker_index]
        assignment = context["assignment"]
        with state_lock:
            worker["status"] = "running"
            worker["started_at"] = iso_now()
        save_progress()
        if context["setup_error"]:
            context["return_code"] = 2
            context["worker_jobs"] = []
            with state_lock:
                worker["status"] = "failed"
                worker["error"] = context["setup_error"]
                worker["failed_jobs"] = assignment["unique_count"]
                worker["pending_jobs"] = 0
                worker["completed_at"] = iso_now()
            log_line(
                log_path,
                f"WORKER_SETUP_FAILED worker={worker_index} gpu={assignment['gpu']} "
                f"error={context['setup_error']}",
                lock=file_lock,
            )
            save_progress()
            return
        child_env = make_execution_environment(config, str(assignment["gpu"]))
        try:
            return_code = run_streamed(
                context["command"],
                config["repo"],
                child_env,
                log_path,
                log_lock=file_lock,
                progress_callback=lambda: sync_worker(context),
            )
            context["return_code"] = return_code
        except Exception as error:
            context["return_code"] = None
            with state_lock:
                worker["error"] = str(error)
            log_line(
                log_path,
                f"WORKER_FAILED worker={worker_index} error={error}",
                lock=file_lock,
            )
        sync_worker(context, save=False)
        context["worker_jobs"] = load_jsonl_if_exists(context["jobs_path"])
        with state_lock:
            # A fatal child exit can leave jobs absent from the child jobs file.
            # They are synthesized as failed aggregate records below; reflect the
            # same terminal accounting in this worker progress snapshot.
            unrecorded = max(
                int(assignment["unique_count"])
                - int(worker["completed_jobs"])
                - int(worker["failed_jobs"]),
                0,
            )
            if context["return_code"] != 0 and unrecorded:
                worker["failed_jobs"] += unrecorded
                worker["pending_jobs"] = 0
            if context["return_code"] == 0 and worker["failed_jobs"] == 0 and worker["pending_jobs"] == 0:
                worker["status"] = "complete"
            elif context["return_code"] != 0:
                worker["status"] = "failed"
            else:
                worker["status"] = "complete_with_errors"
            worker["completed_at"] = iso_now()
        save_progress()

    metadata["status"] = "dry_run" if args.dry_run else "running"
    atomic_json(metadata_path, metadata)
    log_line(
        log_path,
        f"{args.baseline.upper()}_START range=[{plan['total_start_index']},{plan['total_end_index']}) "
        f"workers={len(plan['workers'])} unique={plan['unique_count']} dry_run={args.dry_run}",
    )
    if args.dry_run:
        for context in worker_contexts:
            worker = worker_progress[context["worker_index"]]
            for spec in context["specs"]:
                append_jsonl(jobs_path, {
                    **spec,
                    "status": "planned",
                    "raw_output_dir": spec["raw_output_dir"],
                })
            worker["status"] = "planned"
            worker["completed_at"] = iso_now()
        for assignment in plan["workers"]:
            for duplicate in assignment["duplicates"]:
                append_jsonl(jobs_path, {
                    "job_index": int(duplicate["scope_index"]),
                    "scope_index": int(duplicate["scope_index"]),
                    "rollout_id": duplicate["record"]["id"],
                    "worker_index": int(assignment["worker_index"]),
                    "worker_gpu": str(assignment["gpu"]),
                    "worker_start_index": assignment["start_index"],
                    "worker_end_index": assignment["end_index"],
                    "status": "duplicate_assignment",
                    "return_code": None,
                    "duplicate_of_worker_index": duplicate["owner_worker_index"],
                    "completed_at": iso_now(),
                })
        save_progress()
    else:
        with ThreadPoolExecutor(
            max_workers=max(1, len(worker_contexts)),
            thread_name_prefix=f"{args.baseline}-persistent-worker",
        ) as executor:
            futures = [executor.submit(run_worker, context) for context in worker_contexts]
            for future in as_completed(futures):
                future.result()

    aggregate_by_id: dict[str, dict[str, Any]] = {}
    for context in worker_contexts:
        assignment = context["assignment"]
        raw_by_id = {
            str(record.get("rollout_id")): record
            for record in context["worker_jobs"]
            if record.get("rollout_id") is not None
        }
        for spec in context["specs"]:
            rollout_id = str(spec["rollout_id"])
            result = dict(raw_by_id.get(rollout_id, {}))
            if args.dry_run:
                result = {
                    **spec,
                    "status": "planned",
                    "return_code": None,
                    "raw_output_dir": spec["raw_output_dir"],
                    "raw_output_files": [],
                }
            elif not result or result.get("status") not in {"complete", "failed", "interrupted", "fatal_engine_failure"}:
                result = {
                    **spec,
                    "status": "failed",
                    "return_code": context["return_code"] or 1,
                    "error": context["setup_error"] or (
                        f"worker exited with code {context['return_code']} before recording this rollout"
                    ),
                    "completed_at": iso_now(),
                    "raw_output_dir": spec["raw_output_dir"],
                    "raw_output_files": [],
                }
            result.update({
                "job_index": spec["job_index"],
                "scope_index": spec["job_index"],
                "rollout_id": rollout_id,
                "worker_index": context["worker_index"],
                "worker_gpu": str(assignment["gpu"]),
                "worker_start_index": assignment["start_index"],
                "worker_end_index": assignment["end_index"],
            })
            aggregate_by_id[rollout_id] = result

    aggregate_jobs = list(aggregate_by_id.values())
    for assignment in plan["workers"]:
        for duplicate in assignment["duplicates"]:
            aggregate_jobs.append({
                "job_index": int(duplicate["scope_index"]),
                "scope_index": int(duplicate["scope_index"]),
                "rollout_id": duplicate["record"]["id"],
                "worker_index": int(assignment["worker_index"]),
                "worker_gpu": str(assignment["gpu"]),
                "worker_start_index": assignment["start_index"],
                "worker_end_index": assignment["end_index"],
                "status": "duplicate_assignment",
                "return_code": None,
                "duplicate_of_worker_index": duplicate["owner_worker_index"],
                "completed_at": iso_now(),
            })
    aggregate_jobs.sort(key=lambda record: (int(record.get("job_index", 0)), str(record.get("status", ""))))
    write_jsonl_atomic(jobs_path, aggregate_jobs)
    completed, failed, _duplicates = _parallel_job_count(aggregate_jobs)
    worker_failures = any(context["return_code"] not in (0, None) for context in worker_contexts)
    with state_lock:
        metadata["completed_jobs"] = completed
        metadata["failed_jobs"] = failed
        metadata["pending_jobs"] = max(plan["unique_count"] - completed - failed, 0)
        metadata["worker_progress"] = [
            _public_worker_progress(worker) for worker in worker_progress
        ]
        if args.dry_run:
            metadata["status"] = "dry_run_complete"
        elif failed or metadata["pending_jobs"] or plan["gaps"] or worker_failures:
            metadata["status"] = "complete_with_errors"
        else:
            metadata["status"] = "complete"
        metadata["completed_at"] = iso_now()
        atomic_json(metadata_path, metadata)
    log_line(
        log_path,
        f"END status={metadata['status']} completed={completed} failed={failed} "
        f"duplicates={metadata['duplicate_assignments']}",
    )
    print(f"Run metadata: {metadata_path}")
    print(f"Timestamped log: {log_path}")
    if args.dry_run:
        return 0
    return 0 if not failed and not metadata["pending_jobs"] and not plan["gaps"] and not worker_failures else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=sorted(BASELINES))
    parser.add_argument("--manifest", type=Path, default=None, help="Input LF3R JSONL manifest")
    parser.add_argument("--data-root", type=Path, default=None, help="Root for relative video/csv paths")
    parser.add_argument("--output-dir", type=Path, default=None, help="Parent directory for timestamped run output")
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None, help="Override the baseline checkpoint")
    parser.add_argument("--partition", choices=("natural_observation", "controlled_analysis", "all"), default="natural_observation")
    parser.add_argument("--dataset-role", default=None)
    parser.add_argument("--rollout-id", action="append", default=[])
    parser.add_argument("--start-index", type=int, default=0, help="Scope-relative start of the right-open rollout range")
    parser.add_argument("--end-index", type=int, default=None, help="Scope-relative exclusive end of the rollout range")
    parser.add_argument("--limit", type=int, default=None, help="Compatibility alternative to --end-index")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Independent rollout workers for any baseline")
    parser.add_argument(
        "--worker-spec",
        action="append",
        default=[],
        metavar="GPU:START:END",
        help="Baseline worker assignment; repeat for explicit scope-relative right-open ranges",
    )
    parser.add_argument("--gpu", default=None, help="CUDA device list for automatic worker assignment")
    parser.add_argument("--resume-run", type=Path, default=None, help="Resume an interrupted ProcVLM or Robo-Dopamine run directory; its saved plan is authoritative")
    parser.add_argument("--dry-run", action="store_true", help="Validate and record commands without running models")
    parser.add_argument("--validate-environment", action="store_true", help="Import core packages in the selected existing environment")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--render-video", action="store_true", help="Also render optional baseline visualization videos")

    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--vllm-free-memory-fraction",
        "--vllm-gpu-memory-utilization",
        dest="vllm_free_memory_fraction",
        type=float,
        default=None,
        help="Target fraction of currently free GPU memory; converted to vLLM total-memory fraction",
    )
    parser.add_argument("--procvlm-window-size", type=int, default=4)
    parser.add_argument(
        "--procvlm-max-sampled-frames",
        type=int,
        default=None,
        help="Optional ProcVLM frame cap; omitted to use its upstream default",
    )
    parser.add_argument("--procvlm-max-new-tokens", type=int, default=4096)

    parser.add_argument(
        "--rynn-num-frames",
        type=int,
        default=16,
        help="Uniformly sampled frames per prefix; zero/all-frame mode is not supported",
    )
    parser.add_argument(
        "--rynn-num-steps",
        type=int,
        default=16,
        help="Deprecated compatibility value; per-rollout num_steps is computed from --rynn-evaluation-interval",
    )
    parser.add_argument(
        "--rynn-evaluation-interval",
        type=int,
        default=8,
        help="Target source-frame interval; converted to approximate official prefix endpoint count per rollout",
    )
    parser.add_argument(
        "--rynn-batch-size",
        type=int,
        default=1,
        help="RynnValue prefix batch size; choose based on available GPU memory",
    )
    parser.add_argument("--rynn-max-image-side", type=int, default=448)
    parser.add_argument("--rynn-max-new-tokens", type=int, default=128)
    parser.add_argument("--robot-description", default="A Franka Panda 7-DoF robot arm with a parallel-jaw gripper.")
    parser.add_argument("--camera-description", default="A fixed third-person agent-view RGB camera observing the robot workspace.")

    parser.add_argument("--robo-frame-interval", type=int, default=4)
    parser.add_argument("--robo-batch-size", type=int, default=1)
    parser.add_argument(
        "--densereward-frame-interval",
        type=int,
        default=1,
        help="Evaluate one DenseReward 3-frame window every N source frames; the first current frame is 2",
    )
    parser.add_argument(
        "--densereward-max-new-tokens",
        type=int,
        default=32,
        help="Maximum generated tokens for each official DenseReward response",
    )
    parser.add_argument(
        "--robo-eval-mode",
        choices=("fused", "forward", "incremental", "backward"),
        default="fused",
        help="Robo-Dopamine mode; fused runs incremental, forward, and backward then averages progress",
    )
    parser.add_argument(
        "--robo-eval-modes",
        nargs="+",
        choices=("incremental", "forward", "backward"),
        default=None,
        help="Run official Robo-Dopamine perspectives in one persistent model instance; pass all three for fusion",
    )
    parser.add_argument("--goal-image", type=Path, default=None)
    args = parser.parse_args()
    if args.resume_run is None:
        if args.manifest is None:
            parser.error("--manifest is required unless --resume-run is used")
        if args.output_dir is None:
            parser.error("--output-dir is required unless --resume-run is used")
        args.data_root = args.data_root or PROJECT_ROOT
        args.logs_dir = args.logs_dir or PROJECT_ROOT / "logs/baselines"
        args.gpu = args.gpu or "0"
        args.vllm_free_memory_fraction = (
            0.80 if args.vllm_free_memory_fraction is None else args.vllm_free_memory_fraction
        )
    else:
        if args.baseline not in {"procvlm", "robo_dopamine"}:
            parser.error("--resume-run currently supports only --baseline procvlm or robo_dopamine")
        if args.dry_run:
            parser.error("--resume-run cannot be combined with --dry-run")
    if args.vllm_free_memory_fraction is not None and not 0.0 < args.vllm_free_memory_fraction <= 1.0:
        parser.error("--vllm-free-memory-fraction must be in (0, 1]")
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < 0:
        parser.error("--end-index must be non-negative")
    if args.end_index is not None and args.limit is not None:
        parser.error("--end-index cannot be combined with --limit")
    if args.end_index is not None and args.end_index < args.start_index:
        parser.error("--end-index must be greater than or equal to --start-index")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.parallel_workers < 1:
        parser.error("--parallel-workers must be positive")
    if args.procvlm_max_sampled_frames is not None and args.procvlm_max_sampled_frames < 1:
        parser.error("--procvlm-max-sampled-frames must be positive when provided")
    for name in (
        "procvlm_window_size", "procvlm_max_new_tokens",
        "tensor_parallel_size", "rynn_num_frames", "rynn_num_steps",
        "rynn_evaluation_interval", "rynn_batch_size",
        "rynn_max_image_side", "rynn_max_new_tokens", "robo_frame_interval", "robo_batch_size",
        "densereward_frame_interval", "densereward_max_new_tokens",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.robo_eval_modes:
        if len(set(args.robo_eval_modes)) != len(args.robo_eval_modes):
            parser.error("--robo-eval-modes must not contain duplicates")
        if len(args.robo_eval_modes) > 1 and set(args.robo_eval_modes) != {"incremental", "forward", "backward"}:
            parser.error("multi-perspective Robo-Dopamine requires incremental, forward, and backward")
    if args.baseline == "rynnvalue":
        if args.rynn_num_frames < 1:
            parser.error("--rynn-num-frames must be positive; zero/all-frame mode is not supported")
        if args.rynn_num_steps < 1:
            parser.error("--rynn-num-steps must be positive; it is deprecated and no longer controls sampling")
        if args.rynn_evaluation_interval < 1:
            parser.error("--rynn-evaluation-interval must be positive")
        if args.rynn_batch_size < 1:
            parser.error("--rynn-batch-size must be positive")
    if args.baseline == "densereward":
        if args.densereward_frame_interval < 1:
            parser.error("--densereward-frame-interval must be positive")
        if args.densereward_max_new_tokens < 1:
            parser.error("--densereward-max-new-tokens must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.resume_run is not None:
        if args.baseline == "robo_dopamine":
            from robo_dopamine_runner import resume_run

            return resume_run(args)
        return resume_procvlm_run(args)
    args.manifest = args.manifest.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.logs_dir = args.logs_dir.expanduser().resolve()
    if args.goal_image is not None:
        args.goal_image = args.goal_image.expanduser().resolve()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {args.manifest}")

    config = BASELINES[args.baseline]
    model_path = args.model_path.expanduser().resolve() if args.model_path else config["checkpoint"]
    validate_static(config, model_path)
    manifest_records = load_jsonl(args.manifest)
    worker_plan = None
    scope_records: list[dict[str, Any]] | None = None
    if args.baseline in {"rynnvalue", "densereward"} or args.worker_spec or args.parallel_workers != 1:
        scope_records = filter_records(args, manifest_records)
        if not scope_records:
            raise ValueError("No rollouts matched the requested filters")
        selection_start, selection_end = selection_bounds(args, len(scope_records))
        records = scope_records[selection_start:selection_end]
        if not records:
            raise ValueError("No rollouts matched the requested filters")
        worker_plan = build_worker_plan(scope_records, args)
    else:
        records = select_records(args, manifest_records)

    run_stamp = timestamp()
    run_root = args.output_dir / f"{args.baseline}_{run_stamp}"
    raw_root = run_root / "raw"
    log_path = args.logs_dir / f"{args.baseline}_{run_stamp}.log"
    metadata_path = run_root / "run.json"
    jobs_path = run_root / "jobs.jsonl"
    commands_path = run_root / "commands.jsonl"
    run_root.mkdir(parents=True, exist_ok=False)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": 1,
        "status": "planning",
        "created_at": iso_now(),
        "baseline": args.baseline,
        "dry_run": args.dry_run,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "data_root": str(args.data_root),
        "output_root": str(run_root),
        "log_path": str(log_path),
        "selected_rollouts": len(records),
        "environment_python": str(config["python"]),
        "baseline_repo": str(config["repo"]),
        "baseline_revision": git_revision(config["repo"]),
        "model_path": str(model_path) if model_path else None,
        "vllm_memory_scope": "free_gpu_memory" if args.baseline in VLLM_BASELINES else None,
        "vllm_requested_free_fraction": (
            args.vllm_free_memory_fraction if args.baseline in VLLM_BASELINES else None
        ),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "resume_run"},
        "completed_jobs": 0,
        "failed_jobs": 0,
    }
    atomic_json(metadata_path, metadata)
    log_line(log_path, f"START baseline={args.baseline} rollouts={len(records)} dry_run={args.dry_run}")

    environment_gpu = args.gpu
    if worker_plan and worker_plan["workers"]:
        environment_gpu = str(worker_plan["workers"][0]["gpu"])
    env = make_execution_environment(config, environment_gpu)

    if args.validate_environment:
        check = [str(config["python"]), "-c", config["imports"]]
        log_line(log_path, "ENVIRONMENT_CHECK " + json.dumps(check))
        return_code = run_streamed(check, config["repo"], env, log_path)
        if return_code:
            metadata.update(status="environment_validation_failed", failed_jobs=1, completed_at=iso_now())
            atomic_json(metadata_path, metadata)
            return return_code

    metadata["status"] = "dry_run" if args.dry_run else "running"
    atomic_json(metadata_path, metadata)
    if worker_plan is not None and args.baseline == "safe":
        return run_simple_parallel(
            args=args,
            config=config,
            model_path=model_path,
            run_root=run_root,
            raw_root=raw_root,
            log_path=log_path,
            metadata_path=metadata_path,
            jobs_path=jobs_path,
            commands_path=commands_path,
            metadata=metadata,
            plan=worker_plan,
        )
    if worker_plan is not None and args.baseline in PERSISTENT_BASELINES:
        return run_persistent_parallel(
            args=args,
            config=config,
            model_path=model_path,
            run_root=run_root,
            raw_root=raw_root,
            log_path=log_path,
            metadata_path=metadata_path,
            jobs_path=jobs_path,
            commands_path=commands_path,
            env=env,
            metadata=metadata,
            plan=worker_plan,
        )
    if args.baseline == "procvlm":
        return run_procvlm_persistent(
            args=args,
            config=config,
            records=records,
            model_path=model_path,
            run_root=run_root,
            raw_root=raw_root,
            log_path=log_path,
            metadata_path=metadata_path,
            jobs_path=jobs_path,
            commands_path=commands_path,
            env=env,
            metadata=metadata,
        )
    if args.baseline == "robo_dopamine":
        from robo_dopamine_runner import run_persistent

        return run_persistent(
            args=args,
            config=config,
            records=records,
            model_path=model_path,
            run_root=run_root,
            raw_root=raw_root,
            log_path=log_path,
            metadata_path=metadata_path,
            jobs_path=jobs_path,
            commands_path=commands_path,
            env=env,
            metadata=metadata,
        )
    if args.baseline == "rynnvalue":
        assert worker_plan is not None
        assert scope_records is not None
        return run_rynn_parallel(
            args=args,
            config=config,
            scope_records=scope_records,
            model_path=model_path,
            run_root=run_root,
            raw_root=raw_root,
            log_path=log_path,
            metadata_path=metadata_path,
            jobs_path=jobs_path,
            commands_path=commands_path,
            metadata=metadata,
            plan=worker_plan,
        )
    for index, record in enumerate(records):
        job_dir = raw_root / record["id"]
        vllm_total_memory_fraction = None
        vllm_memory_budget = None
        if args.baseline in VLLM_BASELINES:
            if args.dry_run:
                vllm_memory_budget = {
                    "scope": "free_gpu_memory",
                    "requested_free_fraction": args.vllm_free_memory_fraction,
                    "resolved_total_fraction": None,
                    "resolution": "deferred_until_execution",
                }
            else:
                try:
                    vllm_total_memory_fraction, vllm_memory_budget = resolve_vllm_memory_budget(
                        args.gpu, args.vllm_free_memory_fraction
                    )
                except ValueError as error:
                    metadata.update(status="memory_check_failed", error=str(error), completed_at=iso_now())
                    atomic_json(metadata_path, metadata)
                    log_line(log_path, f"MEMORY_CHECK_FAILED {error}")
                    return 2
                log_line(log_path, "VLLM_MEMORY " + json.dumps(vllm_memory_budget, ensure_ascii=False))
        command, cwd = command_for(
            args, config, record, job_dir, model_path, vllm_total_memory_fraction
        )
        required_input = resolve_record_path(
            record["csv_path"] if args.baseline == "safe" else record["video_path"],
            args.data_root,
        )
        if not required_input.is_file():
            raise FileNotFoundError(f"Input for {record['id']} does not exist: {required_input}")
        command_record = {
            "job_index": index,
            "rollout_id": record["id"],
            "cwd": str(cwd),
            "argv": command,
            "shell_preview": shlex.join(command),
        }
        if vllm_memory_budget is not None:
            command_record["vllm_memory_budget"] = vllm_memory_budget
        if args.baseline == "rynnvalue":
            command_record["sampling"] = {
                "total_frames": int(record["total_frames"]),
                "target_interval": args.rynn_evaluation_interval,
                "approximate_num_steps": int(command[command.index("--num-steps") + 1]),
                "endpoint_sampler": "official_uniform_prefix",
            }
        append_jsonl(commands_path, command_record)
        if args.dry_run:
            append_jsonl(jobs_path, {**command_record, "status": "planned", "raw_output_dir": str(job_dir)})
            log_line(log_path, f"PLAN {index + 1}/{len(records)} rollout={record['id']}")
            continue

        job_dir.mkdir(parents=True, exist_ok=False)
        started_at = iso_now()
        log_line(log_path, f"RUN {index + 1}/{len(records)} rollout={record['id']}")
        return_code = run_streamed(command, cwd, env, log_path)
        raw_files = sorted(str(path) for path in job_dir.rglob("*") if path.is_file())
        job_result = {
            **command_record,
            "status": "complete" if return_code == 0 else "failed",
            "return_code": return_code,
            "started_at": started_at,
            "completed_at": iso_now(),
            "raw_output_dir": str(job_dir),
            "raw_output_files": raw_files,
        }
        append_jsonl(jobs_path, job_result)
        if return_code == 0:
            metadata["completed_jobs"] += 1
        else:
            metadata["failed_jobs"] += 1
        atomic_json(metadata_path, metadata)
        if return_code and not args.continue_on_error:
            metadata.update(status="failed", completed_at=iso_now())
            atomic_json(metadata_path, metadata)
            log_line(log_path, f"STOP return_code={return_code}")
            return return_code

    metadata["status"] = "dry_run_complete" if args.dry_run else ("complete_with_errors" if metadata["failed_jobs"] else "complete")
    metadata["completed_at"] = iso_now()
    atomic_json(metadata_path, metadata)
    log_line(log_path, f"END status={metadata['status']} completed={metadata['completed_jobs']} failed={metadata['failed_jobs']}")
    print(f"Run metadata: {metadata_path}")
    print(f"Timestamped log: {log_path}")
    return 0 if not metadata["failed_jobs"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, PermissionError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
