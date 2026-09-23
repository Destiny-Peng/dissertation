#!/usr/bin/env python3
"""Bounded non-Analysis tools exposed by the LF3R WebUI.

The helpers in this module deliberately reuse existing project scripts instead
of reimplementing their behavior in the browser. Long-running actions are
submitted through the annotator's persistent tmux supervisor; GPU inspection
is one-shot and read-only.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_JOB_TYPE = "project_tool"
DEFAULT_MANIFEST_SCAN_ROOTS = (
    PROJECT_ROOT / "outputs/openvla_libero",
    PROJECT_ROOT / "outputs/openvla_libero_spatial_native",
)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"path must stay inside the project: {value}") from exc
    return resolved


def gpu_status() -> dict[str, Any]:
    """Return one-shot NVIDIA GPU status; never start or mutate GPU workloads."""
    binary = shutil.which("nvidia-smi")
    if not binary:
        return {"available": False, "error": "nvidia-smi is unavailable", "gpus": []}
    query = (
        "index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,"
        "utilization.memory,temperature.gpu,power.draw,power.limit"
    )
    try:
        completed = subprocess.run(
            [binary, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc), "gpus": []}
    rows = []
    for raw in csv.reader(completed.stdout.splitlines()):
        if len(raw) != 11:
            continue
        values = [item.strip() for item in raw]

        def number(text: str) -> float | None:
            try:
                return float(text)
            except (TypeError, ValueError):
                return None

        total = number(values[3])
        free = number(values[5])
        rows.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "uuid": values[2],
                "memory_total_mib": total,
                "memory_used_mib": number(values[4]),
                "memory_free_mib": free,
                "memory_free_fraction": (free / total) if total and free is not None else None,
                "gpu_utilization_percent": number(values[6]),
                "memory_utilization_percent": number(values[7]),
                "temperature_c": number(values[8]),
                "power_draw_w": number(values[9]),
                "power_limit_w": number(values[10]),
            }
        )
    return {
        "available": True,
        "error": None,
        "queried_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpus": rows,
    }


def export_command(payload: dict[str, Any]) -> list[str]:
    command = ["/usr/bin/python3", str(PROJECT_ROOT / "tools/export_failure_cases.py")]
    for outcome in payload.get("outcomes") or ["failure"]:
        if outcome not in {"success", "failure", "recovered_success", "uncertain"}:
            raise ValueError(f"invalid outcome: {outcome}")
        command.extend(["--outcome", outcome])
    role = str(payload.get("dataset_role") or "libero_10")
    if role not in {"libero_10", "libero_spatial", "controlled_analysis", "all"}:
        raise ValueError("invalid dataset_role")
    command.extend(["--dataset-role", role])
    review = str(payload.get("review_status") or "complete")
    if review not in {"complete", "in_progress", "unreviewed", "all"}:
        raise ValueError("invalid review_status")
    command.extend(["--review-status", review])
    if payload.get("output_dir"):
        command.extend(["--output-dir", str(project_path(str(payload["output_dir"])))])
    if bool(payload.get("dry_run", False)):
        command.append("--dry-run")
    return command


def safe_prepare_command(payload: dict[str, Any]) -> list[str]:
    python = Path(
        os.environ.get(
            "LF3R_SAFE_PYTHON", PROJECT_ROOT / "conda_envs/LF3R-safe/bin/python"
        )
    )
    output = project_path(
        str(payload.get("output") or "outputs/safe_training/datasets/web_prepare")
    )
    role = str(payload.get("dataset_role") or "libero_10")
    partition = str(payload.get("partition") or "natural_observation")
    if role not in {"all", "libero_10", "libero_spatial", "controlled_analysis"}:
        raise ValueError("invalid SAFE dataset_role")
    if partition not in {"all", "natural_observation", "controlled_analysis"}:
        raise ValueError("invalid SAFE partition")
    command = [
        str(python),
        str(PROJECT_ROOT / "tools/safe_training/prepare_dataset.py"),
        "--output",
        str(output),
        "--dataset-role",
        role,
        "--partition",
        partition,
    ]
    run_name = str(payload.get("run_name") or "").strip()
    if run_name:
        command.extend(["--run-name", run_name])
    for rollout_id in payload.get("rollout_ids") or []:
        command.extend(["--rollout-id", str(rollout_id)])
    return command


def safe_train_command(payload: dict[str, Any]) -> list[str]:
    python = Path(
        os.environ.get(
            "LF3R_SAFE_PYTHON", PROJECT_ROOT / "conda_envs/LF3R-safe/bin/python"
        )
    )
    model = str(payload.get("model") or "mlp")
    if model not in {"mlp", "lstm"}:
        raise ValueError("SAFE model must be mlp or lstm")
    dataset = project_path(
        str(payload.get("dataset_dir") or "outputs/safe_training/datasets/web_prepare")
    )
    logs = project_path(
        str(payload.get("logs_root") or "outputs/safe_training/logs/web")
    )
    command = [
        str(python),
        str(PROJECT_ROOT / "tools/safe_training/run_safe_training.py"),
        "--dataset-dir",
        str(dataset),
        "--model",
        model,
        "--gpu",
        str(payload.get("gpu") or "0"),
        "--logs-root",
        str(logs),
        "--epochs",
        str(int(payload.get("epochs") or 1000)),
        "--batch-size",
        str(int(payload.get("batch_size") or 512)),
        "--hidden-dim",
        str(int(payload.get("hidden_dim") or 256)),
        "--seed",
        str(payload.get("seed") or "0"),
        "--token-idx-rel",
        str(payload.get("token_idx_rel") or "1.0"),
    ]
    if bool(payload.get("normalize", False)):
        command.append("--normalize")
    return command


def safe_validate_checkpoint_command(payload: dict[str, Any]) -> list[str]:
    python = Path(
        os.environ.get(
            "LF3R_SAFE_PYTHON", PROJECT_ROOT / "conda_envs/LF3R-safe/bin/python"
        )
    )
    model = str(payload.get("model") or "mlp")
    if model not in {"mlp", "lstm"}:
        raise ValueError("SAFE model must be mlp or lstm")
    if not payload.get("dataset_dir") or not payload.get("checkpoint"):
        raise ValueError("SAFE validation requires dataset_dir and checkpoint")
    dataset = project_path(str(payload["dataset_dir"]))
    checkpoint = project_path(str(payload["checkpoint"]))
    output = project_path(
        str(payload.get("output") or "outputs/safe_training/validation/web_scores.json")
    )
    return [
        str(python),
        str(PROJECT_ROOT / "tools/safe_training/validate_safe_checkpoint.py"),
        "--dataset-dir",
        str(dataset),
        "--checkpoint",
        str(checkpoint),
        "--model",
        model,
        "--gpu",
        str(payload.get("gpu") or "0"),
        "--output",
        str(output),
        "--batch-size",
        str(int(payload.get("batch_size") or 16)),
        "--hidden-dim",
        str(int(payload.get("hidden_dim") or 256)),
    ]


def robo_interval_sweep_command(payload: dict[str, Any]) -> list[str]:
    output_dir = project_path(
        str(payload.get("output_dir") or "outputs/baselines/robo_interval_sweeps")
    )
    memory = float(payload.get("memory_utilization") or 0.6)
    if not 0.0 < memory <= 1.0:
        raise ValueError("memory_utilization must be in (0, 1]")
    command = [
        "/usr/bin/python3",
        str(PROJECT_ROOT / "tools/baselines/run_robo_dopamine_interval_sanity.py"),
        "--manifest",
        str(PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"),
        "--data-root",
        str(PROJECT_ROOT),
        "--output-dir",
        str(output_dir),
        "--gpu",
        str(payload.get("gpu") or "0"),
        "--vllm-free-memory-fraction",
        str(memory),
        "--batch-size",
        str(int(payload.get("batch_size") or 1)),
    ]
    rollout_ids = payload.get("rollout_ids") or []
    if not rollout_ids:
        raise ValueError("at least one rollout_id is required")
    for rollout_id in rollout_ids:
        command.extend(["--rollout-id", str(rollout_id)])
    intervals = payload.get("intervals") or [2, 5, 10]
    for interval in intervals:
        interval = int(interval)
        if interval < 1:
            raise ValueError("frame intervals must be positive")
        command.extend(["--frame-interval", str(interval)])
    if payload.get("model_path"):
        command.extend(["--model-path", str(project_path(str(payload["model_path"])))])
    if payload.get("goal_image"):
        command.extend(["--goal-image", str(project_path(str(payload["goal_image"])))])
    return command


def validate_instruction_variants_command(_: dict[str, Any]) -> list[str]:
    return [
        "/usr/bin/python3",
        str(PROJECT_ROOT / "tools/prepare_libero10_instruction_variants.py"),
        "--check-only",
    ]


def rebuild_manifest_command(payload: dict[str, Any]) -> list[str]:
    command = [
        "/usr/bin/python3",
        str(PROJECT_ROOT / "tools/lf3r_annotator/build_manifest.py"),
    ]
    roots: list[Path] = list(DEFAULT_MANIFEST_SCAN_ROOTS)
    raw_extra = payload.get("extra_scan_roots") or []
    if isinstance(raw_extra, str):
        raw_extra = [
            value.strip()
            for value in raw_extra.replace(",", "\n").splitlines()
            if value.strip()
        ]
    if not isinstance(raw_extra, list):
        raise ValueError("extra_scan_roots must be a list of project-local directories")

    seen = {path.resolve() for path in roots}
    for value in raw_extra:
        text = str(value).strip()
        if not text:
            continue
        path = project_path(text)
        if not path.is_dir():
            raise ValueError(f"scan root does not exist or is not a directory: {text}")
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)

    for root in roots:
        command.extend(["--scan-root", str(root)])
    return command


def transcode_video_command(payload: dict[str, Any]) -> list[str]:
    value = str(payload.get("video_path") or "").strip()
    if not value:
        raise ValueError("video_path is required")
    video = project_path(value)
    if video.suffix.lower() != ".mp4":
        raise ValueError("video_path must point to an .mp4 file")
    if not video.is_file():
        raise ValueError(f"video does not exist: {value}")
    backup = video.with_name(video.name[:-4] + ".orig.mp4")
    if backup.exists():
        raise ValueError(
            "original backup already exists; refusing to overwrite: "
            + str(backup.relative_to(PROJECT_ROOT))
        )
    script = PROJECT_ROOT / "tools/lf3r_annotator/transcode_video_h264.sh"
    return ["/usr/bin/bash", str(script), str(video)]


def transcode_manifest_videos_command(payload: dict[str, Any]) -> list[str]:
    raw_manifests = payload.get("manifest_paths") or []
    if not isinstance(raw_manifests, list) or not raw_manifests:
        raise ValueError("manifest_paths must be a non-empty list")
    if len(raw_manifests) > 64:
        raise ValueError("too many manifest paths")

    manifests: list[Path] = []
    seen: set[Path] = set()
    for value in raw_manifests:
        text = str(value).strip()
        if not text:
            continue
        path = project_path(text)
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"manifest must be a .jsonl file: {text}")
        if not path.is_file():
            raise ValueError(f"manifest does not exist: {text}")
        if path in seen:
            continue
        seen.add(path)
        manifests.append(path)

    if not manifests:
        raise ValueError("no valid manifest paths were provided")

    command = [
        "/usr/bin/python3",
        str(PROJECT_ROOT / "tools/lf3r_annotator/transcode_manifest_videos_h264.py"),
        "--project-root",
        str(PROJECT_ROOT),
    ]
    for path in manifests:
        command.extend(["--manifest", str(path)])
    return command


def baseline_pipeline_validation_command(payload: dict[str, Any]) -> list[str]:
    command = [
        "/usr/bin/python3",
        str(PROJECT_ROOT / "tools/baselines/validate_pipeline.py"),
    ]
    if bool(payload.get("check_environments", True)):
        command.append("--check-environments")
    return command


TOOL_BUILDERS = {
    "export_cases": export_command,
    "safe_prepare": safe_prepare_command,
    "safe_train": safe_train_command,
    "safe_validate": safe_validate_checkpoint_command,
    "robo_interval_sweep": robo_interval_sweep_command,
    "validate_variants": validate_instruction_variants_command,
    "rebuild_manifest": rebuild_manifest_command,
    "transcode_video": transcode_video_command,
    "transcode_manifest_videos": transcode_manifest_videos_command,
    "validate_baselines": baseline_pipeline_validation_command,
}

TOOL_LABELS = {
    "export_cases": "Export rollout package",
    "safe_prepare": "Prepare SAFE dataset",
    "safe_train": "Train SAFE detector",
    "safe_validate": "Validate SAFE checkpoint",
    "robo_interval_sweep": "Robo-Dopamine interval sweep",
    "validate_variants": "Validate instruction variants",
    "rebuild_manifest": "Import / rescan rollout manifest",
    "transcode_video": "Transcode selected video to H.264",
    "transcode_manifest_videos": "Transcode manifest videos to H.264",
    "validate_baselines": "Validate baseline pipelines",
}


class NonAnalysisToolService:
    """Persistent wrappers around existing non-Analysis project scripts."""

    ACTIVE_STATUSES = {"queued", "running"}

    def __init__(self, project_root: Path, tmux: Any) -> None:
        self.project_root = project_root.resolve()
        self.tmux = tmux
        self.lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.tmux.register_handler(
            TOOL_JOB_TYPE,
            self._on_loaded,
            on_finished=self._on_finished,
        )

    def _on_loaded(self, job: dict[str, Any]) -> None:
        job.setdefault(
            "tool_label",
            TOOL_LABELS.get(str(job.get("action")), "Project tool"),
        )
        with self.lock:
            self.jobs[str(job["job_id"])] = job

    def _on_finished(
        self,
        job: dict[str, Any],
        return_code: int | None,
        reason: str | None,
    ) -> None:
        if reason:
            job["status"] = "failed"
            job["error"] = reason
        elif return_code == 0:
            job["status"] = "complete"
        else:
            job["status"] = "failed"
            job["error"] = f"command exited with status {return_code}"
        with self.lock:
            self.jobs[str(job["job_id"])] = job

    def _active_conflict(self, action: str) -> dict[str, Any] | None:
        with self.lock:
            jobs = [dict(job) for job in self.jobs.values()]
        for existing in jobs:
            existing_action = str(existing.get("action") or "")
            status = str(existing.get("status") or "")
            if status not in self.ACTIVE_STATUSES:
                continue
            if (
                action in {"transcode_video", "transcode_manifest_videos"}
                and existing_action in {"transcode_video", "transcode_manifest_videos"}
            ):
                return existing
            if action == "rebuild_manifest" and existing_action == "rebuild_manifest":
                return existing
        return None

    def submit(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        client_request_id: str | None = None,
    ) -> dict[str, Any]:
        if action not in TOOL_BUILDERS:
            raise ValueError(f"unknown project tool action: {action}")

        conflict = self._active_conflict(action)
        if conflict is not None:
            if action == "rebuild_manifest":
                raise ValueError(
                    "another manifest rebuild is already active: "
                    + str(conflict.get("job_id") or "unknown")
                )
            raise ValueError(
                "another H.264 transcode job is already active: "
                + str(conflict.get("job_id") or "unknown")
            )

        payload = dict(payload or {})
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        job_id = f"tool-{action}-{stamp}-{uuid.uuid4().hex[:8]}"
        log_path = self.project_root / "logs/annotator_tools" / f"{job_id}.log"
        job = {
            "job_id": job_id,
            "job_type": TOOL_JOB_TYPE,
            "action": action,
            "tool_label": TOOL_LABELS[action],
            "status": "queued",
            "supervisor_status": "preparing",
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "log_path": str(log_path.relative_to(self.project_root)),
            "request": payload,
        }
        if client_request_id:
            job["client_request_id"] = str(client_request_id)

        with self.lock:
            self.jobs[job_id] = job

        threading.Thread(
            target=self._prepare_and_launch,
            args=(job_id, payload, log_path),
            name=f"lf3r-project-tool-{job_id}",
            daemon=True,
        ).start()
        return dict(job)

    def _prepare_and_launch(
        self,
        job_id: str,
        payload: dict[str, Any],
        log_path: Path,
    ) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            return

        try:
            command = TOOL_BUILDERS[str(job["action"])](payload)
            with self.lock:
                current = self.jobs.get(job_id)
                if current is None:
                    return
                current["argv"] = [str(item) for item in command]
                current["supervisor_status"] = "launch_pending"

            launched = self.tmux.submit_async(
                current,
                command,
                log_path,
                interpreter=command[0],
                on_finished=self._on_finished,
            )
            with self.lock:
                current = self.jobs.get(job_id)
                if current is not None:
                    current.update(launched)
        except Exception as error:
            with self.lock:
                current = self.jobs.get(job_id)
                if current is None:
                    return
                current["status"] = "failed"
                current["supervisor_status"] = "prepare_failed"
                current["error"] = str(error)
                current["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return dict(job)

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            jobs = [dict(job) for job in self.jobs.values()]
        if status:
            jobs = [job for job in jobs if job.get("status") == status]
        jobs.sort(
            key=lambda job: str(
                job.get("submitted_at")
                or job.get("tmux_created_at")
                or job.get("job_id")
            ),
            reverse=True,
        )
        return jobs

    def log(self, job_id: str, tail: int = 240) -> dict[str, Any]:
        job = self.get(job_id)
        count = max(1, min(int(tail), 2000))
        path = self.project_root / str(job["log_path"])
        if not path.is_file():
            return {"job_id": job_id, "lines": [], "text": ""}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]
        return {"job_id": job_id, "lines": lines, "text": "\n".join(lines)}
