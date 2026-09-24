"""Shared LF3R backend primitives with no dependency on server.py."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any


ROLLOUT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,160}$")
INSTRUCTION_VARIANT_CONDITIONS = ("full_instruction", "subtask_a", "subtask_b")
DYNAMIC_SCOPE_RE = re.compile(r"^[A-Za-z0-9._-]{1,160}$")
RESERVED_RUN_SCOPES = {"all", "controlled_analysis"}
PRIMARY_CAMERA_ORDER = ("cam_high", "cam_wrist", "cam_left_wrist", "cam_right_wrist")


class ValidationError(ValueError):
    pass


class JobConflictError(RuntimeError):
    pass


class AnalysisEnvironmentError(RuntimeError):
    pass


class JobCoordinator:
    """Coordinate only manifest writes; independent compute may overlap."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_jobs: dict[str, str] = {}

    def acquire(self, job_id: str, role: str = "compute") -> None:
        with self.lock:
            if role == "manifest_writer" and any(
                active_role == "manifest_writer"
                for active_role in self.active_jobs.values()
            ):
                raise JobConflictError(
                    "Another rollout-generation job is already rebuilding the manifest"
                )
            self.active_jobs[job_id] = role

    def restore(self, job_id: str, role: str = "compute") -> None:
        with self.lock:
            self.active_jobs[job_id] = role

    def release(self, job_id: str) -> None:
        with self.lock:
            self.active_jobs.pop(job_id, None)

    def active(self) -> dict[str, str]:
        with self.lock:
            return dict(self.active_jobs)


def is_controlled_record(record: dict[str, Any]) -> bool:
    return (
        record.get("analysis_partition") == "controlled_analysis"
        or record.get("source_kind") == "controlled_injected"
    )


# Compatibility alias retained for existing call sites.
_is_controlled_record = is_controlled_record


def record_camera_video_paths(record: dict[str, Any]) -> dict[str, str]:
    raw = record.get("camera_video_paths")
    if not isinstance(raw, dict) or not raw:
        raise ValidationError(
            "Manifest record must define a non-empty camera_video_paths mapping: "
            + str(record.get("id") or "<unknown>")
        )
    paths: dict[str, str] = {}
    for camera, value in raw.items():
        key = str(camera or "").strip()
        if not key.startswith("cam_"):
            raise ValidationError(
                "Camera keys must use the cam_* schema: " + repr(camera)
            )
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(
                "Camera path must be a non-empty string for " + key
            )
        paths[key] = value.strip()
    return paths


def primary_camera_video_path(record: dict[str, Any]) -> tuple[str, str]:
    paths = record_camera_video_paths(record)
    for camera in PRIMARY_CAMERA_ORDER:
        if camera in paths:
            return camera, paths[camera]
    camera = sorted(paths)[0]
    return camera, paths[camera]


def validate_run_scope(scope: Any) -> str:
    value = str(scope or "all").strip()
    if value in RESERVED_RUN_SCOPES:
        return value
    if not DYNAMIC_SCOPE_RE.fullmatch(value):
        raise ValidationError(
            "scope must be 'all', 'controlled_analysis', or a task_suite name "
            "present in the loaded manifests"
        )
    return value


def validate_instruction_condition(condition: Any) -> str:
    value = str(condition or "full_instruction")
    if value not in INSTRUCTION_VARIANT_CONDITIONS:
        raise ValidationError(
            "instruction_condition must be one of: "
            + ", ".join(INSTRUCTION_VARIANT_CONDITIONS)
        )
    return value


def record_matches_scope(record: dict[str, Any], scope: str) -> bool:
    scope = validate_run_scope(scope)
    if scope == "all":
        return True
    if scope == "controlled_analysis":
        return is_controlled_record(record)
    return (
        str(record.get("task_suite") or "") == scope
        and not is_controlled_record(record)
    )


def select_scope_records(
    records: list[dict[str, Any]],
    scope: str,
) -> list[dict[str, Any]]:
    scope = validate_run_scope(scope)
    return [record for record in records if record_matches_scope(record, scope)]


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    records = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            rollout_id = record.get("id")
            if (
                not isinstance(rollout_id, str)
                or not ROLLOUT_ID_RE.fullmatch(rollout_id)
            ):
                raise ValidationError(
                    f"Invalid rollout id at manifest line {line_number}"
                )
            if rollout_id in seen:
                raise ValidationError(f"Duplicate rollout id: {rollout_id}")
            seen.add(rollout_id)
            records.append(record)
    return records


def run_rollout_ids(run_path: Path) -> set[str]:
    """Return rollout IDs whose worker records are already complete."""
    result: set[str] = set()
    paths = [run_path / "jobs.jsonl"]
    workers_root = run_path / "workers"
    if workers_root.is_dir():
        paths.extend(sorted(workers_root.glob("worker-*/jobs.jsonl")))

    for jobs_path in paths:
        if not jobs_path.is_file():
            continue
        try:
            with jobs_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("status") != "complete":
                        continue
                    rollout_id = row.get("rollout_id", row.get("id"))
                    if (
                        isinstance(rollout_id, str)
                        and ROLLOUT_ID_RE.fullmatch(rollout_id)
                    ):
                        result.add(rollout_id)
        except OSError:
            continue
    return result


def atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
