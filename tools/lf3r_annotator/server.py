#!/usr/bin/env python3
"""Local-only LF3R rollout annotation server.

Uses only the Python standard library. The server binds to loopback by default,
serves videos with byte-range support, and atomically persists annotations under
the LF3R project root.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import datetime as dt
import fcntl
import json
import math
import mimetypes
import os
import re
import signal
import subprocess
import sys
import time
import tempfile
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from task_supervisor import TmuxJobSupervisor, TmuxSupervisorError


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLLOUT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,160}$")
FAILURE_TYPES = {
    "none_success",
    "grasp_failure",
    "placement_failure",
    "dropped_object",
    "wrong_object",
    "collision",
    "timeout_no_progress",
    "control_error",
    "observation_error",
    "other",
}
OUTCOME_LABELS = {"success", "failure", "recovered_success", "uncertain"}
REVIEW_STATUSES = {"unreviewed", "in_progress", "complete"}
BASELINE_METHODS = ("safe", "procvlm", "rynnvalue", "robo_dopamine", "densereward")
ANALYSIS_BASELINE_METHODS = ("safe", "procvlm", "rynnvalue", "robo_dopamine")
BASELINE_LABELS = {
    "safe": "SAFE",
    "procvlm": "ProcVLM",
    "rynnvalue": "RynnValue",
    "robo_dopamine": "Robo-Dopamine",
    "densereward": "DenseReward",
}
BASELINE_RUN_STATUSES = {"complete", "complete_with_errors"}
INSTRUCTION_VARIANT_CONDITIONS = ("full_instruction", "subtask_a", "subtask_b")
INSTRUCTION_VARIANT_LABELS = {
    "full_instruction": "Full instruction",
    "subtask_a": "A",
    "subtask_b": "B",
}
RUN_SCOPES = (
    "all",
    "natural_observation",
    "primary_natural",
    "reference_natural",
    "controlled_analysis",
)
RUN_SCOPE_LABELS = {
    "all": "All manifest rollouts",
    "natural_observation": "All natural observations",
    "primary_natural": "Primary natural",
    "reference_natural": "Reference natural",
    "controlled_analysis": "Controlled analysis",
}
VLLM_BASELINE_METHODS = {"procvlm", "robo_dopamine"}
BASELINE_METHOD_OPTION_FIELDS = {
    "safe": {"render_video", "validate_environment", "dry_run"},
    "procvlm": {
        "model_path", "dtype", "tensor_parallel_size", "procvlm_window_size",
        "procvlm_max_sampled_frames", "procvlm_max_new_tokens",
        "render_video", "validate_environment", "dry_run",
    },
    "rynnvalue": {
        "model_path", "rynn_num_frames", "rynn_num_steps", "rynn_evaluation_interval", "rynn_batch_size",
        "rynn_max_image_side", "rynn_max_new_tokens", "robot_description",
        "camera_description", "render_video", "validate_environment", "dry_run",
    },
    "robo_dopamine": {
        "model_path", "dtype", "tensor_parallel_size", "robo_frame_interval",
        "robo_batch_size", "robo_eval_mode", "goal_image",
        "render_video", "validate_environment", "dry_run",
    },
    "densereward": {
        "model_path", "densereward_frame_interval", "densereward_max_new_tokens",
        "validate_environment", "dry_run",
    },
}
BASELINE_ADVANCED_FIELDS = {
    "model_path",
    "dtype",
    "tensor_parallel_size",
    "procvlm_window_size",
    "procvlm_max_sampled_frames",
    "procvlm_max_new_tokens",
    "rynn_num_frames",
    "rynn_num_steps",
    "rynn_evaluation_interval",
    "rynn_batch_size",
    "rynn_max_image_side",
    "rynn_max_new_tokens",
    "robot_description",
    "camera_description",
    "robo_frame_interval",
    "robo_batch_size",
    "robo_eval_mode",
    "goal_image",
    "densereward_frame_interval",
    "densereward_max_new_tokens",
    "render_video",
    "validate_environment",
    "dry_run",
}

DEFAULT_SETTINGS = {
    "background_color": "#0b0d10",
    "surface_color": "#11151a",
    "surface_raised_color": "#171c22",
    "control_color": "#0f1318",
    "text_color": "#f4f6f7",
    "muted_color": "#98a3ad",
    "accent_color": "#67d9b5",
    "font_scale": 1.0,
    "review_font_scale": 1.0,
    "analysis_font_scale": 1.0,
    "control_font_scale": 1.0,
    "density": "comfortable",
}
SETTING_COLOR_FIELDS = {
    "background_color",
    "surface_color",
    "surface_raised_color",
    "control_color",
    "text_color",
    "muted_color",
    "accent_color",
}
SETTING_DENSITIES = {"compact", "comfortable", "spacious"}
SETTING_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

ANALYSIS_TABLE_FILES = {
    "method_coverage": "method_coverage.csv",
    "summary_by_method_signal_outcome": "summary_by_method_signal_outcome.csv",
    "summary_by_method_outcome": "summary_by_method_outcome.csv",
    "onset_signal_statistics": "onset_signal_statistics.csv",
    "clean_background_summary": "clean_background_summary.csv",
}
ANALYSIS_LOCALIZATION_TABLE_FILES = {
    "summary": "localization_summary.csv",
    "thresholds": "localization_thresholds.csv",
    "by_failure_type": "localization_by_failure_type.csv",
}
CHANGEPOINT_TABLE_FILES = {
    "summary": "changepoint_summary.csv",
    "reference_summary": "changepoint_reference_summary.csv",
    "by_failure_type": "changepoint_by_failure_type.csv",
    "scales": "changepoint_scales.csv",
    "method_coverage": "method_coverage.csv",
}
EVENT_TRIGGERED_TABLE_FILES = {
    "curves": "event_triggered_curves.csv",
    "change_scores": "event_triggered_change_scores.csv",
    "separation": "event_triggered_separation.csv",
    "summary": "event_triggered_summary.csv",
    "peak_events": "event_triggered_peak_events.csv",
    "controls": "event_triggered_controls.csv",
    "method_coverage": "method_coverage.csv",
}

# These allowlists are deliberately kept server-side. The Analysis page can
# browse high-cardinality CSV/JSONL artifacts without turning the generic file
# server into an arbitrary-file reader.
ANALYSIS_DETAIL_KINDS = {
    "changepoint_summary": ("change_point", "changepoint_summary.csv"),
    "changepoint_failure_types": ("change_point", "changepoint_by_failure_type.csv"),
    "changepoint_events": ("change_point", "changepoint_event_metrics.jsonl"),
    "changepoint_comparison": ("change_point", "comparison_with_full_136_20260827.csv"),
    "localization_summary": ("change_point", "localization_summary.csv"),
    "localization_thresholds": ("change_point", "localization_thresholds.csv"),
    "localization_failure_types": ("change_point", "localization_by_failure_type.csv"),
    "localization_events": ("change_point", "localization_event_metrics.jsonl"),
    "event_triggered_summary": ("event_triggered", "event_triggered_summary.csv"),
    "event_triggered_peaks": ("event_triggered", "event_triggered_peak_events.csv"),
    "event_triggered_curves": ("event_triggered", "event_triggered_curves.csv"),
    "event_triggered_change_scores": ("event_triggered", "event_triggered_change_scores.csv"),
    "legacy_anomalies": ("legacy", "event_metrics.jsonl"),
}
ANALYSIS_DETAIL_FILTERS = {
    "method", "signal", "feature", "scale", "threshold", "outcome",
    "failure_type", "task",
}
ANALYSIS_DETAIL_SORTS = {
    "score", "event_score", "localization_error", "peak_distance", "recall",
    "false_alarm", "f1", "auroc", "rollout", "task",
}
ANALYSIS_ARTIFACT_NAMES = {
    "REPORT.md",
    "metadata.json",
    "method_coverage.csv",
    "summary_by_method_signal_outcome.csv",
    "summary_by_method_outcome.csv",
    "onset_signal_statistics.csv",
    "clean_background_summary.csv",
    "event_metrics.jsonl",
    "localization_summary.csv",
    "localization_thresholds.csv",
    "localization_by_failure_type.csv",
    "localization_event_metrics.jsonl",
    "changepoint_summary.csv",
    "changepoint_reference_summary.csv",
    "changepoint_by_failure_type.csv",
    "changepoint_scales.csv",
    "changepoint_event_metrics.jsonl",
    "comparison_with_full_136_20260827.csv",
    "event_triggered_curves.csv",
    "event_triggered_change_scores.csv",
    "event_triggered_separation.csv",
    "event_triggered_summary.csv",
    "event_triggered_peak_events.csv",
    "event_triggered_controls.csv",
}
CHANGEPOINT_EVENT_FIELDS = (
    "method",
    "signal",
    "scale_frames",
    "feature",
    "feature_label",
    "threshold",
    "threshold_value",
    "rollout_id",
    "event_index",
    "event_type",
    "outcome_group",
    "failure_type",
    "event_frame",
    "observable_onset_frame",
    "recovery_frame",
    "terminal_failure_frame",
    "next_event_frame",
    "event_window_low",
    "event_window_high_exclusive",
    "onset_candidate_frame",
    "onset_candidate_distance_frames",
    "onset_feature_value",
    "onset_score",
    "peak_frame",
    "peak_feature_value",
    "peak_score",
    "peak_signed_offset_frames",
    "peak_distance_frames",
    "first_exceedance_frame",
    "first_exceedance_signed_offset_frames",
    "first_exceedance_absolute_error_frames",
    "feature_valid",
    "hit",
    "miss",
    "early_peak",
    "early_first_exceedance",
    "onset_near_unusual",
    "direction_after_unusual",
    "failure_direction_consistent",
    "native_samples_only",
    "clean_success_false_alarm_rate",
    "same_rollout_non_onset_false_alarm_rate",
    "clean_pseudo_events",
    "false_alarm_count",
    "false_alarm_rate",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "f1",
    "auroc",
    "average_precision",
    "task_suite",
    "task_id",
    "task_description",
    "source_kind",
    "analysis_partition",
    "dataset_role",
)
ANALYSIS_LOCALIZATION_EVENT_FIELDS = (
    "method",
    "signal",
    "threshold",
    "threshold_value",
    "rollout_id",
    "event_index",
    "event_type",
    "outcome_group",
    "failure_type",
    "event_frame",
    "observable_onset_frame",
    "recovery_frame",
    "terminal_failure_frame",
    "next_event_frame",
    "window_low",
    "window_high_exclusive",
    "pre_center",
    "robust_scale",
    "event_score",
    "score_peak_frame",
    "score_peak_offset_frames",
    "first_alarm_frame",
    "first_post_onset_frame",
    "signed_lead_lag_frames",
    "post_onset_delay_frames",
    "localization_error_frames",
    "absolute_localization_error_frames",
    "hit",
    "miss",
    "early_alarm",
    "native_samples_only",
    "clean_pseudo_event_count",
    "hit_within_4_frames",
    "hit_within_8_frames",
    "hit_within_16_frames",
    "hit_within_30_frames",
    "task_suite",
    "task_id",
    "task_description",
    "source_kind",
    "analysis_partition",
    "dataset_role",
)
ANALYSIS_EVENT_FIELDS = (
    "rollout_id",
    "event_index",
    "event_type",
    "outcome_group",
    "failure_type",
    "is_background",
    "event_frame",
    "recovery_frame",
    "terminal_failure_frame",
    "next_event_frame",
    "recovery_offset_frames",
    "method",
    "signal",
    "normalized_response_magnitude",
    "normalized_failure_oriented_response",
    "post_event_persistence_fraction",
    "post_event_persistence_median_oriented",
    "recovery_rebound_normalized",
    "recovery_fraction_toward_baseline",
    "clean_background_response_q95",
    "clean_background_response_percentile",
    "native_direction",
    "pre_window_requested_frames",
    "post_window_requested_frames",
    "pre_sample_count",
    "post_sample_count",
    "persistence_sample_count",
    "post_window_fallback",
    "pre_center",
    "pre_mad",
    "raw_pre_scale",
    "robust_pre_scale",
    "robust_scale_source",
    "post_center",
    "recovery_sample_frame",
    "recovery_frame_nearest_sample_error",
    "recovery_value",
    "recovery_rebound_signed_change",
    "recovery_rebound_toward_baseline",
    "clean_background_pseudo_event_count",
    "clean_background_response_median",
)
ANALYSIS_EVENT_ARRAY_FIELDS = {
    "pre_sample_frames",
    "post_sample_frames",
    "persistence_sample_frames",
}
ANALYSIS_RECORD_FIELDS = (
    "task_suite",
    "task_id",
    "task_description",
    "source_kind",
    "analysis_partition",
    "dataset_role",
)


class ValidationError(ValueError):
    pass


class JobConflictError(RuntimeError):
    pass


class AnalysisEnvironmentError(RuntimeError):
    pass


class JobCoordinator:
    """Coordinate only manifest writes; independent baseline jobs may overlap."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_jobs: dict[str, str] = {}

    def acquire(self, job_id: str, role: str = "compute") -> None:
        with self.lock:
            if role == "manifest_writer" and self.active_jobs:
                raise JobConflictError(
                    "Rollout generation is exclusive because it rebuilds the manifest"
                )
            if any(active_role == "manifest_writer" for active_role in self.active_jobs.values()):
                raise JobConflictError(
                    "A rollout-generation job is rebuilding the manifest"
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


def validate_run_scope(scope: Any) -> str:
    value = str(scope or "natural_observation")
    if value not in RUN_SCOPES:
        raise ValidationError("scope must be one of: " + ", ".join(RUN_SCOPES))
    return value


def record_matches_scope(record: dict[str, Any], scope: str) -> bool:
    if scope == "all":
        return True
    if scope in {"primary_natural", "reference_natural"}:
        return record.get("dataset_role") == scope
    return record.get("analysis_partition") == scope


def select_scope_records(records: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
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
            if not isinstance(rollout_id, str) or not ROLLOUT_ID_RE.fullmatch(rollout_id):
                raise ValidationError(f"Invalid rollout id at manifest line {line_number}")
            if rollout_id in seen:
                raise ValidationError(f"Duplicate rollout id: {rollout_id}")
            seen.add(rollout_id)
            records.append(record)
    return records


def run_rollout_ids(run_path: Path) -> set[str]:
    jobs_path = run_path / "jobs.jsonl"
    if not jobs_path.is_file():
        return set()
    result: set[str] = set()
    try:
        with jobs_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "complete":
                    continue
                rollout_id = row.get("rollout_id", row.get("id"))
                if isinstance(rollout_id, str) and ROLLOUT_ID_RE.fullmatch(rollout_id):
                    result.add(rollout_id)
    except (OSError, json.JSONDecodeError):
        return set()
    return result


def atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


class SettingsStore:
    """Read and atomically persist project-wide annotator appearance settings."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.path = self.project_root / "config" / "lf3r_annotator.json"

    def _validate(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValidationError("Settings must be a JSON object")
        expected = set(DEFAULT_SETTINGS)
        actual = set(payload)
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            raise ValidationError("Settings fields are invalid (" + "; ".join(details) + ")")
        clean: dict[str, Any] = {}
        for field in SETTING_COLOR_FIELDS:
            value = payload[field]
            if not isinstance(value, str) or not SETTING_COLOR_RE.fullmatch(value):
                raise ValidationError(f"{field} must be a six-digit hex color")
            clean[field] = value.lower()
        def validate_scale(field: str, minimum: float, maximum: float) -> float:
            value = payload[field]
            if isinstance(value, bool):
                raise ValidationError(
                    f"{field} must be a number from {minimum:.2f} to {maximum:.2f} in 0.05 steps"
                )
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValidationError(
                    f"{field} must be a number from {minimum:.2f} to {maximum:.2f} in 0.05 steps"
                ) from error
            step_position = (value - minimum) / 0.05
            if (
                not math.isfinite(value)
                or not minimum <= value <= maximum
                or abs(step_position - round(step_position)) > 1e-8
            ):
                raise ValidationError(
                    f"{field} must be a number from {minimum:.2f} to {maximum:.2f} in 0.05 steps"
                )
            return round(value, 2)

        clean["font_scale"] = validate_scale("font_scale", 0.75, 1.60)
        for field in ("review_font_scale", "analysis_font_scale", "control_font_scale"):
            clean[field] = validate_scale(field, 0.85, 1.30)
        density = payload["density"]
        if not isinstance(density, str) or density not in SETTING_DENSITIES:
            raise ValidationError("density must be compact, comfortable, or spacious")
        clean["density"] = density
        return clean

    def read(self) -> tuple[dict[str, Any], str | None]:
        if not self.path.exists():
            return dict(DEFAULT_SETTINGS), None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                optional_font_fields = {
                    "review_font_scale", "analysis_font_scale", "control_font_scale"
                }
                missing = set(DEFAULT_SETTINGS) - set(payload)
                if missing and missing.issubset(optional_font_fields):
                    payload = {**DEFAULT_SETTINGS, **payload}
            settings = self._validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError):
            return dict(DEFAULT_SETTINGS), None
        updated_at = dt.datetime.fromtimestamp(
            self.path.stat().st_mtime, tz=dt.timezone.utc
        ).isoformat()
        return settings, updated_at

    def response(self) -> dict[str, Any]:
        settings, updated_at = self.read()
        return {
            "settings": settings,
            "defaults": dict(DEFAULT_SETTINGS),
            "updated_at": updated_at,
        }

    def write(self, payload: Any) -> dict[str, Any]:
        settings = self._validate(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=".lf3r_annotator_settings.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(settings, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return settings


class AnalysisService:
    """Expose the primary local change-point snapshot and legacy global comparison."""

    def __init__(self, project_root: Path, manifest_path: Path, annotation_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.annotation_root = annotation_root.resolve()
        self.analysis_root = self.project_root / "outputs" / "baseline_signal_analysis"

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(path)

    @staticmethod
    def _coerce(value: Any) -> Any:
        if isinstance(value, (list, dict, tuple)):
            return None
        if value in (None, ""):
            return None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
            try:
                number = float(value)
            except ValueError:
                return value
            if not math.isfinite(number):
                return None
            return int(number) if number.is_integer() else number
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _read_csv(self, path: Path) -> list[dict[str, Any]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return [
                {key: self._coerce(value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]

    def _manifest(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    records[row["id"]] = row
        return records

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _iso_mtime(path: Path) -> str:
        return dt.datetime.fromtimestamp(
            path.stat().st_mtime, tz=dt.timezone.utc
        ).isoformat()

    def _latest_snapshot(self) -> tuple[Path, dict[str, Any]] | None:
        if not self.analysis_root.is_dir():
            return None
        candidates = []
        required = ("metadata.json", "event_metrics.jsonl", *ANALYSIS_TABLE_FILES.values())
        for metadata_path in self.analysis_root.glob("*/metadata.json"):
            directory = metadata_path.parent
            if not all((directory / name).is_file() for name in required):
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            candidates.append((metadata_path.stat().st_mtime, directory, metadata))
        if not candidates:
            return None
        _, directory, metadata = max(candidates, key=lambda item: item[0])
        return directory, metadata

    def _latest_annotation_update(self) -> str | None:
        records_dir = self.annotation_root / "records"
        latest = None
        if not records_dir.is_dir():
            return None
        for path in records_dir.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8")).get("updated_at")
            except (OSError, json.JSONDecodeError):
                continue
            if value and (latest is None or str(value) > latest):
                latest = str(value)
        return latest

    def _event_metrics(self, path: Path, manifest: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("is_background"):
                    continue
                rollout_id = raw.get("rollout_id")
                row = {field: self._coerce(raw.get(field)) for field in ANALYSIS_EVENT_FIELDS}
                for field, value in raw.items():
                    if field in row or field in ANALYSIS_EVENT_ARRAY_FIELDS or isinstance(value, (list, dict, tuple)):
                        continue
                    row[field] = self._coerce(value)
                row["rollout_id"] = rollout_id
                record = manifest.get(str(rollout_id), {})
                row.update({field: record.get(field) for field in ANALYSIS_RECORD_FIELDS})
                rows.append(row)
        return rows

    def _localization_event_metrics(
        self,
        path: Path,
        manifest: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                row = {
                    field: self._coerce(raw.get(field))
                    for field in ANALYSIS_LOCALIZATION_EVENT_FIELDS
                    if field in raw
                }
                for field, value in raw.items():
                    if field in row or isinstance(value, (list, dict, tuple)):
                        continue
                    row[field] = self._coerce(value)
                rollout_id = raw.get("rollout_id")
                row["rollout_id"] = rollout_id
                record = manifest.get(str(rollout_id), {})
                for field in ANALYSIS_RECORD_FIELDS:
                    if row.get(field) is None and record.get(field) is not None:
                        row[field] = record.get(field)
                rows.append(row)
        return rows

    def _latest_event_triggered_snapshot(self) -> tuple[Path, dict[str, Any]] | None:
        if not self.analysis_root.is_dir():
            return None
        required = (
            "metadata.json",
            *EVENT_TRIGGERED_TABLE_FILES.values(),
        )
        candidates = []
        for metadata_path in self.analysis_root.glob("*/metadata.json"):
            directory = metadata_path.parent
            if not all((directory / name).is_file() for name in required):
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("analysis") != "lf3r_event_triggered_signal_analysis":
                continue
            candidates.append((metadata_path.stat().st_mtime, directory, metadata))
        if not candidates:
            return None
        _, directory, metadata = max(candidates, key=lambda item: item[0])
        return directory, metadata

    def _event_triggered_response(self) -> dict[str, Any]:
        selected = self._latest_event_triggered_snapshot()
        if selected is None:
            return {
                "available": False,
                "message": (
                    "No complete event-triggered signal snapshot found under "
                    "outputs/baseline_signal_analysis."
                ),
            }
        directory, metadata = selected
        metadata_path = directory / "metadata.json"
        manifest = self._manifest()
        generated_at = str(
            metadata.get("generated_at")
            or metadata.get("completed_at")
            or self._iso_mtime(metadata_path)
        )
        current_manifest_hash = self._sha256(self.manifest_path)
        snapshot_manifest_hash = metadata.get("manifest_sha256")
        counts = metadata.get("counts") or {}
        snapshot_rollout_count = int(
            counts.get("rollouts")
            or len(metadata.get("rollouts") or [])
            or 0
        )
        latest_annotation_update = self._latest_annotation_update()
        selection_path = None
        if metadata.get("selection"):
            selection_path = Path(str(metadata["selection"]))
            if not selection_path.is_absolute():
                selection_path = self.project_root / selection_path
        stale = bool(
            snapshot_manifest_hash
            and snapshot_manifest_hash != current_manifest_hash
        )
        if latest_annotation_update and latest_annotation_update > generated_at:
            stale = True
        tables = {
            name: self._read_csv(directory / filename)
            for name, filename in EVENT_TRIGGERED_TABLE_FILES.items()
        }
        return {
            "available": True,
            "source": {
                "directory": self._relative(directory),
                "metadata": self._relative(metadata_path),
                "generated_at": generated_at,
                "selection": (
                    self._relative(selection_path) if selection_path else None
                ),
                "selection_count": snapshot_rollout_count,
                "plots": [
                    self._relative(directory / str(plot))
                    for plot in (metadata.get("plots") or [])
                    if (directory / str(plot)).is_file()
                ],
            },
            "freshness": {
                "stale": stale,
                "manifest_matches": snapshot_manifest_hash == current_manifest_hash,
                "snapshot_manifest_sha256": snapshot_manifest_hash,
                "current_manifest_sha256": current_manifest_hash,
                "snapshot_rollout_count": snapshot_rollout_count,
                "current_rollout_count": len(manifest),
                "latest_annotation_update": latest_annotation_update,
            },
            "parameters": metadata.get("parameters") or {},
            "methods": metadata.get("methods") or list(ANALYSIS_BASELINE_METHODS),
            "signals_by_method": metadata.get("signals_by_method") or {},
            "event_group_counts": metadata.get("event_group_counts") or {},
            "counts": counts,
            "native_sampling": metadata.get("native_sampling") or {},
            "source_runs": metadata.get("source_runs") or {},
            "method_coverage": tables["method_coverage"],
            "curves": tables["curves"],
            "change_scores": tables["change_scores"],
            "separation": tables["separation"],
            "summary": tables["summary"],
            "peak_events": tables["peak_events"],
            "controls": tables["controls"],
        }

    def _latest_change_point_snapshot(self) -> tuple[Path, dict[str, Any]] | None:
        if not self.analysis_root.is_dir():
            return None
        required = (
            "metadata.json",
            "changepoint_event_metrics.jsonl",
            *CHANGEPOINT_TABLE_FILES.values(),
        )
        candidates = []
        for metadata_path in self.analysis_root.glob("*/metadata.json"):
            directory = metadata_path.parent
            if not all((directory / name).is_file() for name in required):
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("analysis") != "lf3r_baseline_change_points":
                continue
            candidates.append((metadata_path.stat().st_mtime, directory, metadata))
        if not candidates:
            return None
        _, directory, metadata = max(candidates, key=lambda item: item[0])
        return directory, metadata

    def _change_point_event_metrics(
        self,
        path: Path,
        manifest: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                row = {
                    field: self._coerce(raw.get(field))
                    for field in CHANGEPOINT_EVENT_FIELDS
                    if field in raw
                }
                rollout_id = raw.get("rollout_id")
                row["rollout_id"] = rollout_id
                record = manifest.get(str(rollout_id), {})
                for field in ANALYSIS_RECORD_FIELDS:
                    if row.get(field) is None and record.get(field) is not None:
                        row[field] = record.get(field)
                rows.append(row)
        return rows

    def _change_point_response(self) -> dict[str, Any]:
        selected = self._latest_change_point_snapshot()
        if selected is None:
            return {
                "available": False,
                "message": "No complete local change-point snapshot found under outputs/baseline_signal_analysis.",
            }
        directory, metadata = selected
        manifest = self._manifest()
        metadata_path = directory / "metadata.json"
        generated_at = str(
            metadata.get("generated_at")
            or metadata.get("completed_at")
            or self._iso_mtime(metadata_path)
        )
        current_manifest_hash = self._sha256(self.manifest_path)
        snapshot_manifest_hash = metadata.get("manifest_sha256")
        counts = metadata.get("counts") or {}
        snapshot_rollout_count = int(
            counts.get("rollouts")
            or len(metadata.get("rollouts") or [])
            or 0
        )
        latest_annotation_update = self._latest_annotation_update()
        selection_path = None
        if metadata.get("selection"):
            selection_path = Path(str(metadata["selection"]))
            if not selection_path.is_absolute():
                selection_path = self.project_root / selection_path
        stale = bool(snapshot_manifest_hash and snapshot_manifest_hash != current_manifest_hash)
        if latest_annotation_update and latest_annotation_update > generated_at:
            stale = True
        tables = {
            name: self._read_csv(directory / filename)
            for name, filename in CHANGEPOINT_TABLE_FILES.items()
        }
        events = self._change_point_event_metrics(
            directory / "changepoint_event_metrics.jsonl",
            manifest,
        )
        localization_tables = {
            "summary": (
                self._read_csv(directory / "localization_summary.csv")
                if (directory / "localization_summary.csv").is_file()
                else tables["summary"]
            ),
            "thresholds": (
                self._read_csv(directory / "localization_thresholds.csv")
                if (directory / "localization_thresholds.csv").is_file()
                else tables["scales"]
            ),
            "by_failure_type": (
                self._read_csv(directory / "localization_by_failure_type.csv")
                if (directory / "localization_by_failure_type.csv").is_file()
                else tables["by_failure_type"]
            ),
        }
        comparison_path = directory / "comparison_with_full_136_20260827.csv"
        comparison = self._read_csv(comparison_path) if comparison_path.is_file() else []
        return {
            "available": True,
            "source": {
                "directory": self._relative(directory),
                "metadata": self._relative(metadata_path),
                "generated_at": generated_at,
                "selection": self._relative(selection_path) if selection_path else None,
                "selection_count": snapshot_rollout_count,
                "comparison": self._relative(comparison_path) if comparison_path.is_file() else None,
            },
            "freshness": {
                "stale": stale,
                "manifest_matches": snapshot_manifest_hash == current_manifest_hash,
                "snapshot_manifest_sha256": snapshot_manifest_hash,
                "current_manifest_sha256": current_manifest_hash,
                "snapshot_rollout_count": snapshot_rollout_count,
                "current_rollout_count": len(manifest),
                "latest_annotation_update": latest_annotation_update,
            },
            "parameters": {
                "pre_window_frames": metadata.get("pre_window_frames"),
                "post_window_frames": metadata.get("post_window_frames"),
                "min_samples_per_side": metadata.get("min_samples_per_side"),
                "reference_exclusion_radius_frames": metadata.get("reference_exclusion_radius_frames"),
                "frame_coordinate": metadata.get("frame_coordinate"),
                "native_sampling_preserved": metadata.get("native_sampling_preserved"),
                "threshold_calibration": metadata.get("threshold_calibration"),
            },
            "methods": metadata.get("methods") or list(ANALYSIS_BASELINE_METHODS),
            "features": metadata.get("features") or [],
            "feature_labels": metadata.get("feature_labels") or {},
            "local_scales_frames": metadata.get("local_scales_frames") or [],
            "thresholds": metadata.get("thresholds") or [],
            "tolerances_frames": metadata.get("tolerances_frames") or [],
            "direction_used_after_detection": metadata.get("direction_used_after_detection") or {},
            "counts": counts,
            "method_coverage": tables["method_coverage"],
            "summary": tables["summary"],
            "reference_summary": tables["reference_summary"],
            "by_failure_type": tables["by_failure_type"],
            "scales": tables["scales"],
            "localization_available": True,
            "localization_summary": localization_tables["summary"],
            "localization_thresholds": localization_tables["thresholds"],
            "localization_by_failure_type": localization_tables["by_failure_type"],
            "localization_event_metrics": events,
            "localization": {
                "available": True,
                "summary": localization_tables["summary"],
                "thresholds": localization_tables["thresholds"],
                "by_failure_type": localization_tables["by_failure_type"],
                "event_metrics": events,
            },
            "event_metrics": events,
            "comparison": comparison,
        }

    @staticmethod
    def _query_value(query: dict[str, Any], key: str, default: str = "") -> str:
        value = query.get(key, default)
        if isinstance(value, list):
            value = value[0] if value else default
        return str(value) if value is not None else default

    @staticmethod
    def _same_detail_value(value: Any, wanted: str) -> bool:
        if wanted in {"", "all"}:
            return True
        if value is None:
            return False
        left = str(value)
        right = str(wanted)
        if left == right:
            return True
        try:
            return float(left) == float(right)
        except (TypeError, ValueError):
            return left.lower() == right.lower()

    def _latest_for_detail_kind(
        self, kind: str
    ) -> tuple[str, Path, dict[str, Any]] | None:
        snapshot_type, _filename = ANALYSIS_DETAIL_KINDS[kind]
        if snapshot_type == "change_point":
            selected = self._latest_change_point_snapshot()
        elif snapshot_type == "event_triggered":
            selected = self._latest_event_triggered_snapshot()
        else:
            selected = self._latest_snapshot()
        if selected is None:
            return None
        directory, metadata = selected
        return snapshot_type, directory, metadata

    def _detail_rows(
        self,
        kind: str,
        directory: Path,
        manifest: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        _snapshot_type, filename = ANALYSIS_DETAIL_KINDS[kind]
        path = directory / filename
        if not path.is_file():
            return []
        if filename.endswith(".csv"):
            return self._read_csv(path)
        if kind == "changepoint_events":
            return self._change_point_event_metrics(path, manifest)
        if kind == "localization_events":
            return self._localization_event_metrics(path, manifest)
        return self._event_metrics(path, manifest)

    @staticmethod
    def _detail_matches(row: dict[str, Any], filters: dict[str, str]) -> bool:
        for field, wanted in filters.items():
            if wanted in {"", "all"}:
                continue
            if field == "task":
                values = (
                    row.get("task_id"),
                    row.get("task"),
                    row.get("task_description"),
                )
            elif field == "outcome":
                values = (row.get("outcome_group"), row.get("event_group"), row.get("outcome"))
            elif field == "failure_type":
                values = (row.get("failure_type"), row.get("failure"))
            elif field == "scale":
                values = (row.get("scale_frames"), row.get("scale"))
            else:
                values = (row.get(field),)
            if not any(AnalysisService._same_detail_value(value, wanted) for value in values):
                return False
        return True

    @staticmethod
    def _detail_sort_value(row: dict[str, Any], sort: str) -> Any:
        aliases = {
            "score": (
                "event_score", "onset_score", "peak_score",
                "normalized_response_magnitude", "strongest_separation_z",
            ),
            "event_score": ("event_score", "onset_score", "peak_score"),
            "localization_error": (
                "absolute_localization_error_frames",
                "localization_error_frames",
                "first_exceedance_absolute_error_frames",
                "peak_distance_frames",
            ),
            "peak_distance": ("peak_distance_frames", "median_peak_distance_frames"),
            "recall": ("recall", "event_hit_rate"),
            "false_alarm": (
                "false_alarm_rate", "clean_success_false_alarm_rate",
                "clean_success_false_alarm_count",
            ),
            "f1": ("f1",),
            "auroc": ("auroc",),
            "rollout": ("rollout_id",),
            "task": ("task_id", "task_description"),
        }
        for field in aliases.get(sort, ()):
            value = row.get(field)
            if value not in (None, ""):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return str(value).lower()
        return float("-inf") if sort != "rollout" else ""

    @staticmethod
    def _detail_columns(rows: list[dict[str, Any]]) -> list[str]:
        preferred = [
            "method", "signal", "feature", "scale_frames", "threshold",
            "outcome_group", "failure_type", "task_suite", "task_id",
            "rollout_id", "event_frame", "observable_onset_frame",
            "event_score", "recall", "false_alarm_rate",
            "clean_success_false_alarm_rate",
            "absolute_localization_error_frames", "f1", "auroc",
        ]
        keys = set()
        for row in rows:
            keys.update(
                key for key, value in row.items()
                if not isinstance(value, (list, dict, tuple))
            )
        return [key for key in preferred if key in keys] + sorted(
            keys.difference(preferred)
        )

    @staticmethod
    def _dashboard_rows(
        rows: list[dict[str, Any]],
        *,
        limit: int = 64,
        feature: str = "level",
        scale: str = "16",
        threshold: str = "q95",
        outcome: str = "all_events",
    ) -> list[dict[str, Any]]:
        if len(rows) <= limit:
            return rows
        exact = [
            row for row in rows
            if AnalysisService._same_detail_value(row.get("feature"), feature)
            and AnalysisService._same_detail_value(row.get("scale_frames"), scale)
            and AnalysisService._same_detail_value(
                row.get("threshold") or row.get("threshold_name"), threshold
            )
            and (
                outcome == "all_events"
                or AnalysisService._same_detail_value(row.get("outcome_group"), outcome)
            )
        ]
        return exact[:limit] or rows[:limit]

    def _live_summary(self) -> dict[str, Any]:
        records = self._manifest()
        outcomes = {
            "clean_success": 0,
            "recovered_success": 0,
            "terminal_failure": 0,
            "uncertain": 0,
        }
        by_partition: dict[str, int] = {}
        annotated = 0
        failure_events = 0
        observable_events = 0
        for record in records.values():
            partition = str(record.get("analysis_partition") or "unknown")
            by_partition[partition] = by_partition.get(partition, 0) + 1
            annotation_path = self.annotation_root / "records" / f"{record['id']}.json"
            annotation: dict[str, Any] = {}
            try:
                if annotation_path.is_file():
                    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                annotation = {}
            if annotation:
                annotated += 1
            label = str(annotation.get("outcome_label") or record.get("ground_truth_outcome") or "uncertain")
            if label in {"success", "clean_success"}:
                outcome = "clean_success"
            elif label == "recovered_success":
                outcome = "recovered_success"
            elif label in {"failure", "terminal_failure"}:
                outcome = "terminal_failure"
            else:
                outcome = "uncertain"
            outcomes[outcome] += 1
            events = annotation.get("failure_events") or []
            failure_events += len(events)
            observable_events += sum(
                1 for event in events
                if event.get("observable_onset_frame") is not None
            )
        resolved = outcomes["clean_success"] + outcomes["recovered_success"] + outcomes["terminal_failure"]
        return {
            "rollouts": len(records),
            "annotated": annotated,
            "resolved": resolved,
            "resolved_success_rate": (
                (outcomes["clean_success"] + outcomes["recovered_success"]) / resolved
                if resolved else None
            ),
            "outcomes": outcomes,
            "failure_events": failure_events,
            "observable_onset_events": observable_events,
            "by_partition": by_partition,
        }

    def _artifact_links(self) -> list[dict[str, Any]]:
        selected = [
            ("change_point", self._latest_change_point_snapshot()),
            ("event_triggered", self._latest_event_triggered_snapshot()),
            ("legacy", self._latest_snapshot()),
        ]
        links: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source_type, result in selected:
            if result is None:
                continue
            directory, _metadata = result
            for name in sorted(ANALYSIS_ARTIFACT_NAMES):
                if name in seen:
                    continue
                candidate = directory / name
                if candidate.is_file():
                    seen.add(name)
                    links.append({
                        "name": name,
                        "source": source_type,
                        "path": self._relative(candidate),
                        "url": "/api/analysis/artifacts/" + name,
                    })
        return links

    def _compact_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact = copy.deepcopy(payload)
        compact["dashboard"] = True
        compact["default_scope"] = "primary_natural"
        compact["available_tabs"] = [
            "overview", "comparison", "failures", "events", "signals", "archive"
        ]
        compact["artifact_links"] = self._artifact_links()
        compact["live"] = self._live_summary()
        detail_counts: dict[str, int] = {}
        for kind, (snapshot_type, _filename) in ANALYSIS_DETAIL_KINDS.items():
            if snapshot_type == "change_point":
                node = compact.get("change_point") or {}
                count_key = {
                    "changepoint_summary": "summary",
                    "changepoint_failure_types": "by_failure_type",
                    "changepoint_events": "event_metrics",
                    "localization_summary": "localization_summary",
                    "localization_thresholds": "localization_thresholds",
                    "localization_failure_types": "localization_by_failure_type",
                    "localization_events": "localization_event_metrics",
                    "changepoint_comparison": "comparison",
                }.get(kind)
            elif snapshot_type == "event_triggered":
                node = compact.get("event_triggered") or {}
                count_key = {
                    "event_triggered_summary": "summary",
                    "event_triggered_peaks": "peak_events",
                    "event_triggered_curves": "curves",
                    "event_triggered_change_scores": "change_scores",
                }.get(kind)
            else:
                node = compact
                count_key = "event_metrics"
            value = node.get(count_key, []) if count_key else []
            detail_counts[kind] = len(value) if isinstance(value, list) else 0
        compact["detail_counts"] = detail_counts

        change_point = compact.get("change_point")
        if isinstance(change_point, dict):
            for key in ("summary", "reference_summary", "by_failure_type", "scales",
                        "localization_summary", "localization_thresholds",
                        "localization_by_failure_type"):
                if isinstance(change_point.get(key), list):
                    change_point[key] = self._dashboard_rows(change_point[key])
            localization = change_point.get("localization")
            if isinstance(localization, dict):
                for key in ("summary", "thresholds", "by_failure_type"):
                    if isinstance(localization.get(key), list):
                        localization[key] = self._dashboard_rows(localization[key])
            for key in ("event_metrics", "localization_event_metrics"):
                if isinstance(change_point.get(key), list) and len(change_point[key]) > 64:
                    change_point.pop(key, None)
            if isinstance(localization, dict) and isinstance(localization.get("event_metrics"), list):
                if len(localization["event_metrics"]) > 64:
                    localization.pop("event_metrics", None)

        for key in ("event_metrics", "localization_event_metrics"):
            if isinstance(compact.get(key), list) and len(compact[key]) > 64:
                compact.pop(key, None)
        event_triggered = compact.get("event_triggered")
        if isinstance(event_triggered, dict):
            for key in ("summary", "peak_events"):
                if isinstance(event_triggered.get(key), list):
                    event_triggered[key] = event_triggered[key][:64]
            for key in ("curves", "change_scores", "separation", "controls"):
                if isinstance(event_triggered.get(key), list) and len(event_triggered[key]) > 240:
                    event_triggered.pop(key, None)
        for key in (
            "summary_by_method_signal_outcome",
            "summary_by_method_outcome",
            "onset_signal_statistics",
            "clean_background_summary",
        ):
            if isinstance(compact.get(key), list) and len(compact[key]) > 64:
                compact[key] = compact[key][:64]
        compact["compatibility"] = {
            "full_endpoint": "/api/analysis?view=full",
            "note": "The full compatibility response contains event-level arrays; dashboard mode keeps them behind /api/analysis/details.",
        }
        return compact

    def details(self, query: dict[str, Any]) -> dict[str, Any]:
        allowed = {"kind", "page", "page_size", "sort"} | ANALYSIS_DETAIL_FILTERS
        unknown = set(query) - allowed
        if unknown:
            raise ValidationError("Unknown analysis details field(s): " + ", ".join(sorted(unknown)))
        kind = self._query_value(query, "kind", "changepoint_summary")
        if kind not in ANALYSIS_DETAIL_KINDS:
            raise ValidationError("Unsupported analysis details kind")
        try:
            page = int(self._query_value(query, "page", "1"))
            page_size = int(self._query_value(query, "page_size", "25"))
        except ValueError as exc:
            raise ValidationError("page and page_size must be integers") from exc
        if page < 1:
            raise ValidationError("page must be at least 1")
        if page_size < 1 or page_size > 100:
            raise ValidationError("page_size must be between 1 and 100")
        sort = self._query_value(query, "sort", "score")
        if sort not in ANALYSIS_DETAIL_SORTS:
            raise ValidationError("Unsupported analysis details sort")
        filters = {
            field: self._query_value(query, field)
            for field in ANALYSIS_DETAIL_FILTERS
            if self._query_value(query, field) not in {"", "all"}
        }
        selected = self._latest_for_detail_kind(kind)
        if selected is None:
            return {
                "available": False,
                "kind": kind,
                "page": page,
                "page_size": page_size,
                "total": 0,
                "page_count": 0,
                "items": [],
                "columns": [],
                "message": "No compatible analysis snapshot is available.",
            }
        snapshot_type, directory, metadata = selected
        rows = self._detail_rows(kind, directory, self._manifest())
        rows = [row for row in rows if self._detail_matches(row, filters)]
        reverse = sort not in {"rollout", "task"}
        rows.sort(
            key=lambda row: self._detail_sort_value(row, sort),
            reverse=reverse,
        )
        total = len(rows)
        start = (page - 1) * page_size
        items = rows[start:start + page_size]
        source = {
            "directory": self._relative(directory),
            "generated_at": metadata.get("generated_at") or self._iso_mtime(directory / "metadata.json"),
            "kind": kind,
            "snapshot_type": snapshot_type,
        }
        return {
            "available": bool((directory / ANALYSIS_DETAIL_KINDS[kind][1]).is_file()),
            "kind": kind,
            "source": source,
            "page": page,
            "page_size": page_size,
            "page_count": math.ceil(total / page_size) if total else 0,
            "total": total,
            "filters": filters,
            "sort": sort,
            "columns": self._detail_columns(items or rows[:1]),
            "items": items,
        }

    def artifact_path(self, name: str) -> Path:
        if name not in ANALYSIS_ARTIFACT_NAMES or "/" in name or "\\" in name:
            raise ValidationError("Unsupported analysis artifact")
        for _source_type, result in (
            ("change_point", self._latest_change_point_snapshot()),
            ("event_triggered", self._latest_event_triggered_snapshot()),
            ("legacy", self._latest_snapshot()),
        ):
            if result is None:
                continue
            directory, _metadata = result
            candidate = (directory / name).resolve()
            try:
                candidate.relative_to(directory.resolve())
            except ValueError as exc:
                raise ValidationError("Analysis artifact escapes its snapshot") from exc
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(name)

    def _full_response(self) -> dict[str, Any]:
        change_point = self._change_point_response()
        event_triggered = self._event_triggered_response()
        selected = self._latest_snapshot()
        if selected is None:
            has_analysis_artifact = bool(
                change_point.get("available") or event_triggered.get("available")
            )
            return {
                "available": has_analysis_artifact,
                "temporal_available": False,
                "message": (
                    None
                    if has_analysis_artifact
                    else "No complete baseline analysis snapshot found under outputs/baseline_signal_analysis."
                ),
                "source": None,
                "freshness": {},
                "parameters": {},
                "methods": list(ANALYSIS_BASELINE_METHODS),
                "orientation": {},
                "signal_units": {},
                "method_coverage": [],
                "summary_by_method_signal_outcome": [],
                "summary_by_method_outcome": [],
                "onset_signal_statistics": [],
                "clean_background_summary": [],
                "event_metrics": [],
                "localization_available": bool(change_point.get("available")),
                "localization_summary": change_point.get("localization_summary", []),
                "localization_thresholds": change_point.get("localization_thresholds", []),
                "localization_by_failure_type": change_point.get("localization_by_failure_type", []),
                "localization_event_metrics": change_point.get("localization_event_metrics", []),
                "localization": change_point.get("localization", {
                    "available": False,
                    "summary": [],
                    "thresholds": [],
                    "by_failure_type": [],
                    "event_metrics": [],
                }),
                "change_point_available": bool(change_point.get("available")),
                "change_point": change_point,
                "event_triggered_available": bool(event_triggered.get("available")),
                "event_triggered": event_triggered,
                "primary_analysis_type": (
                    "change_point"
                    if change_point.get("available")
                    else ("event_triggered" if event_triggered.get("available") else None)
                ),
                "primary_analysis_source": (
                    change_point.get("source")
                    if change_point.get("available")
                    else (event_triggered.get("source") if event_triggered.get("available") else None)
                ),
                "legacy_temporal_available": False,
            }
        directory, metadata = selected
        manifest = self._manifest()
        metadata_path = directory / "metadata.json"
        generated_at = str(
            metadata.get("generated_at")
            or metadata.get("completed_at")
            or self._iso_mtime(metadata_path)
        )
        current_manifest_hash = self._sha256(self.manifest_path)
        snapshot_manifest_hash = metadata.get("manifest_sha256")
        snapshot_rollout_count = int(
            (metadata.get("counts") or {}).get("rollouts")
            or len(metadata.get("rollouts") or [])
            or 0
        )
        latest_annotation_update = self._latest_annotation_update()
        selection_path = None
        if metadata.get("selection"):
            selection_path = Path(str(metadata["selection"]))
            if not selection_path.is_absolute():
                selection_path = self.project_root / selection_path
        stale = bool(snapshot_manifest_hash and snapshot_manifest_hash != current_manifest_hash)
        if latest_annotation_update and latest_annotation_update > generated_at:
            stale = True
        tables = {
            name: self._read_csv(directory / filename)
            for name, filename in ANALYSIS_TABLE_FILES.items()
        }
        localization_tables = {
            name: self._read_csv(directory / filename)
            if (directory / filename).is_file()
            else []
            for name, filename in ANALYSIS_LOCALIZATION_TABLE_FILES.items()
        }
        localization_events = self._localization_event_metrics(
            directory / "localization_event_metrics.jsonl",
            manifest,
        )
        localization_available = bool(
            localization_events
            or localization_tables["summary"]
            or localization_tables["thresholds"]
        )
        return {
            "available": True,
            "temporal_available": True,
            "source": {
                "directory": self._relative(directory),
                "metadata": self._relative(metadata_path),
                "generated_at": generated_at,
                "selection": self._relative(selection_path) if selection_path else None,
                "selection_count": snapshot_rollout_count,
            },
            "freshness": {
                "stale": stale,
                "manifest_matches": snapshot_manifest_hash == current_manifest_hash,
                "snapshot_manifest_sha256": snapshot_manifest_hash,
                "current_manifest_sha256": current_manifest_hash,
                "snapshot_rollout_count": snapshot_rollout_count,
                "current_rollout_count": len(manifest),
                "latest_annotation_update": latest_annotation_update,
            },
            "parameters": {
                "pre_window_frames": metadata.get("pre_window_frames"),
                "post_window_frames": metadata.get("post_window_frames"),
                "background_stride_frames": metadata.get("background_stride_frames"),
                "frame_coordinate": metadata.get("frame_coordinate"),
                "native_sampling_preserved": metadata.get("native_sampling_preserved"),
            },
            "methods": metadata.get("methods") or list(ANALYSIS_BASELINE_METHODS),
            "orientation": metadata.get("orientation") or {},
            "signal_units": metadata.get("signal_units") or {},
            "method_coverage": tables["method_coverage"],
            "summary_by_method_signal_outcome": tables["summary_by_method_signal_outcome"],
            "summary_by_method_outcome": tables["summary_by_method_outcome"],
            "onset_signal_statistics": tables["onset_signal_statistics"],
            "clean_background_summary": tables["clean_background_summary"],
            "event_metrics": self._event_metrics(directory / "event_metrics.jsonl", manifest),
            "localization_available": localization_available,
            "localization_summary": localization_tables["summary"],
            "localization_thresholds": localization_tables["thresholds"],
            "localization_by_failure_type": localization_tables["by_failure_type"],
            "localization_event_metrics": localization_events,
            "localization": {
                "available": localization_available,
                "summary": localization_tables["summary"],
                "thresholds": localization_tables["thresholds"],
                "by_failure_type": localization_tables["by_failure_type"],
                "event_metrics": localization_events,
            },
            "change_point_available": bool(change_point.get("available")),
            "change_point": change_point,
            "event_triggered_available": bool(event_triggered.get("available")),
            "event_triggered": event_triggered,
            "primary_analysis_type": (
                "change_point"
                if change_point.get("available")
                else "legacy_temporal"
            ),
            "primary_analysis_source": (
                change_point.get("source")
                if change_point.get("available")
                else {
                    "directory": self._relative(directory),
                    "metadata": self._relative(metadata_path),
                    "generated_at": generated_at,
                    "selection_count": snapshot_rollout_count,
                }
            ),
            "legacy_temporal_available": True,
        }


    def response(self, compact: bool = True) -> dict[str, Any]:
        payload = self._full_response()
        return self._compact_response(payload) if compact else payload



class AnnotationStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.records_dir = self.root / "records"
        self.events_dir = self.root / "events"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, rollout_id: str) -> Path:
        if not ROLLOUT_ID_RE.fullmatch(rollout_id):
            raise ValidationError("Invalid rollout id")
        return self.records_dir / f"{rollout_id}.json"

    def read(self, rollout_id: str) -> dict[str, Any] | None:
        path = self.path_for(rollout_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write(self, rollout: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        rollout_id = rollout["id"]
        clean = validate_annotation(payload, rollout)
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        previous = self.read(rollout_id)
        record = {
            "schema_version": 2,
            "rollout_id": rollout_id,
            "updated_at": now,
            "created_at": previous.get("created_at", now) if previous else now,
            **clean,
        }
        target = self.path_for(rollout_id)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{rollout_id}.", suffix=".tmp", dir=self.records_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        self._append_event(record)
        return record

    def _append_event(self, record: dict[str, Any]) -> None:
        day = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
        path = self.events_dir / f"annotation_events_{day}.jsonl"
        event = {
            "timestamp": record["updated_at"],
            "rollout_id": record["rollout_id"],
            "review_status": record["review_status"],
            "annotator": record["annotator"],
            "outcome_label": record["outcome_label"],
            "failure_event_count": len(record["failure_events"]),
        }
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def optional_frame(value: Any, name: str, total_frames: int) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer frame or null")
    if value < 0 or value >= total_frames:
        raise ValidationError(f"{name} must be between 0 and {total_frames - 1}")
    return value


def validate_failure_event(
    value: Any, index: int, total_frames: int, default_failure_type: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"failure_events[{index}] must be an object")
    failure_type = value.get("failure_type", default_failure_type)
    if failure_type not in FAILURE_TYPES:
        raise ValidationError(f"failure_events[{index}].failure_type is invalid")
    prefix = f"failure_events[{index}]"
    causal = optional_frame(
        value.get("causal_onset_frame"), f"{prefix}.causal_onset_frame", total_frames
    )
    observable = optional_frame(
        value.get("observable_onset_frame"),
        f"{prefix}.observable_onset_frame",
        total_frames,
    )
    terminal = optional_frame(
        value.get("terminal_failure_frame"),
        f"{prefix}.terminal_failure_frame",
        total_frames,
    )
    recovery = optional_frame(
        value.get("recovery_frame"), f"{prefix}.recovery_frame", total_frames
    )
    if causal is not None and observable is not None and causal > observable:
        raise ValidationError(f"{prefix}: causal onset cannot be after observable onset")
    onset = observable if observable is not None else causal
    if onset is not None and terminal is not None and onset > terminal:
        raise ValidationError(f"{prefix}: onset cannot be after terminal failure")
    if onset is not None and recovery is not None and onset > recovery:
        raise ValidationError(f"{prefix}: onset cannot be after recovery")
    if terminal is not None and recovery is not None:
        raise ValidationError(f"{prefix}: terminal failure and recovery are mutually exclusive")
    notes = str(value.get("notes", ""))
    if len(notes) > 1000:
        raise ValidationError(f"{prefix}.notes must be at most 1000 characters")
    return {
        "failure_type": failure_type,
        "causal_onset_frame": causal,
        "observable_onset_frame": observable,
        "terminal_failure_frame": terminal,
        "recovery_frame": recovery,
        "notes": notes,
    }


def validate_annotation(payload: Any, rollout: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError("Annotation must be a JSON object")
    total_frames = int(rollout["total_frames"])
    annotator = str(payload.get("annotator", "")).strip()
    if not annotator or len(annotator) > 100:
        raise ValidationError("annotator is required and must be at most 100 characters")
    review_status = payload.get("review_status", "in_progress")
    if review_status not in REVIEW_STATUSES:
        raise ValidationError("Invalid review_status")
    outcome_label = payload.get("outcome_label", "uncertain")
    if outcome_label not in OUTCOME_LABELS:
        raise ValidationError("Invalid outcome_label")
    failure_type = payload.get("failure_type", "other")
    if failure_type not in FAILURE_TYPES:
        raise ValidationError("Invalid failure_type")
    confidence = payload.get("confidence")
    if confidence in ("", None):
        confidence = None
    elif isinstance(confidence, bool) or not isinstance(confidence, int) or not 1 <= confidence <= 5:
        raise ValidationError("confidence must be an integer from 1 to 5 or null")
    raw_events = payload.get("failure_events")
    if raw_events is None:
        legacy_values = {
            "failure_type": failure_type,
            "causal_onset_frame": payload.get("causal_onset_frame"),
            "observable_onset_frame": payload.get("observable_onset_frame"),
            "terminal_failure_frame": payload.get("terminal_failure_frame"),
            "recovery_frame": payload.get("recovery_frame"),
            "notes": "",
        }
        has_legacy_boundary = any(
            legacy_values[name] not in (None, "")
            for name in (
                "causal_onset_frame",
                "observable_onset_frame",
                "terminal_failure_frame",
                "recovery_frame",
            )
        )
        raw_events = [legacy_values] if has_legacy_boundary else []
    if not isinstance(raw_events, list):
        raise ValidationError("failure_events must be an array")
    failure_events = [
        validate_failure_event(event, index, total_frames, failure_type)
        for index, event in enumerate(raw_events)
    ]
    if outcome_label == "recovered_success" and not failure_events:
        raise ValidationError("recovered_success requires at least one failure event")
    if outcome_label == "recovered_success" and any(
        event["terminal_failure_frame"] is not None for event in failure_events
    ):
        raise ValidationError("recovered_success events cannot be terminal failures")
    first_event = failure_events[0] if failure_events else {}
    notes = str(payload.get("notes", ""))
    if len(notes) > 5000:
        raise ValidationError("notes must be at most 5000 characters")
    return {
        "annotator": annotator,
        "review_status": review_status,
        "outcome_label": outcome_label,
        "failure_type": first_event.get("failure_type", failure_type),
        "confidence": confidence,
        "failure_events": failure_events,
        "causal_onset_frame": first_event.get("causal_onset_frame"),
        "observable_onset_frame": first_event.get("observable_onset_frame"),
        "terminal_failure_frame": first_event.get("terminal_failure_frame"),
        "recovery_frame": first_event.get("recovery_frame"),
        "notes": notes,
    }


class BaselineService:
    """Read existing baseline outputs and optionally launch one bounded rollout."""

    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        coordinator: JobCoordinator | None = None,
        tmux: TmuxJobSupervisor | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.baseline_root = self.project_root / "outputs" / "baselines"
        self.web_output_root = self.baseline_root / "web_runs"
        self.web_logs_root = self.project_root / "logs" / "baselines" / "web_runs"
        self.jobs: dict[str, dict[str, Any]] = {}
        self.jobs_lock = threading.Lock()
        self.coordinator = coordinator or JobCoordinator()
        self.tmux = tmux or TmuxJobSupervisor(self.project_root)
        self.tmux.register_handler(
            "baseline",
            self._on_job_loaded,
            self._on_job_poll,
            self._on_job_finished,
        )

    def _project_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        resolved = path.resolve() if path.is_absolute() else (self.project_root / path).resolve()
        try:
            resolved.relative_to(self.project_root)
        except ValueError as exc:
            raise ValidationError("Baseline path escapes project root") from exc
        return resolved

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(path)

    
    def _number(self, value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _run_candidates(self, method: str) -> list[tuple[Path, dict[str, Any]]]:
        candidates: list[tuple[tuple[str, int, str], Path, dict[str, Any]]] = []
        if not self.baseline_root.is_dir():
            return []
        baseline_root = self.baseline_root.resolve()
        for path in self.baseline_root.rglob("run.json"):
            try:
                path.parent.resolve().relative_to(baseline_root)
            except ValueError:
                continue
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("baseline") != method or metadata.get("status") not in BASELINE_RUN_STATUSES:
                continue
            try:
                selected = int(metadata.get("selected_rollouts") or 0)
            except (TypeError, ValueError):
                continue
            key = (str(metadata.get("completed_at") or metadata.get("created_at") or ""), selected, str(path))
            candidates.append((key, path.parent, metadata))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [(path, metadata) for _, path, metadata in candidates]

    def _explicit_run_candidate(
        self,
        method: str,
        run_root: Any,
        allowed_conditions: set[str],
    ) -> tuple[Path, dict[str, Any]]:
        if not isinstance(run_root, (str, Path)) or not str(run_root).strip():
            raise ValidationError(f"{method} run selection must be a project-relative run path")
        path = self._project_path(str(run_root))
        try:
            path.relative_to(self.baseline_root.resolve())
        except ValueError as exc:
            raise ValidationError(f"Selected {method} run must be inside outputs/baselines") from exc
        metadata_path = path / "run.json"
        if not metadata_path.is_file():
            raise ValidationError(f"Selected {method} run.json is missing")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError(f"Selected {method} run metadata is invalid") from exc
        if metadata.get("baseline") != method:
            raise ValidationError(f"Selected run is not a {method} run")
        if metadata.get("status") not in BASELINE_RUN_STATUSES:
            raise ValidationError(f"Selected {method} run is not complete")
        condition = self._run_instruction_condition(metadata)
        if condition not in allowed_conditions:
            allowed = ", ".join(sorted(allowed_conditions))
            raise ValidationError(
                f"Selected {method} run is for instruction condition {condition!r}; expected {allowed}"
            )
        return path, metadata

    @staticmethod
    def _metadata_count(metadata: dict[str, Any], field: str) -> int:
        try:
            return int(metadata.get(field) or 0)
        except (TypeError, ValueError):
            return 0

    def _run_instruction_condition(self, metadata: dict[str, Any]) -> str:
        explicit = metadata.get("instruction_variant") or metadata.get("instruction_condition")
        if explicit in INSTRUCTION_VARIANT_CONDITIONS:
            return str(explicit)
        manifest = metadata.get("manifest")
        if manifest:
            try:
                resolved = self._project_path(str(manifest))
            except (ValidationError, OSError):
                resolved = None
            if resolved == self.manifest_path:
                return "full_instruction"
            if resolved is not None:
                path_text = resolved.as_posix()
                for condition in ("subtask_a", "subtask_b", "full_instruction"):
                    if f"instruction_variants/libero_10/{condition}" in path_text:
                        return condition
        return "unknown"

    def _run_summary(self, run_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "baseline": metadata.get("baseline"),
            "method": metadata.get("baseline"),
            "status": metadata.get("status"),
            "selected_rollouts": self._metadata_count(metadata, "selected_rollouts"),
            "completed_jobs": self._metadata_count(metadata, "completed_jobs"),
            "failed_jobs": self._metadata_count(metadata, "failed_jobs"),
            "run_root": self._relative(run_path),
            "completed_at": metadata.get("completed_at"),
            "instruction_condition": self._run_instruction_condition(metadata),
        }

    def _pack(
        self,
        method: str,
        run_summary: dict[str, Any],
        samples: list[dict[str, Any]],
        raw_files: list[Path],
        raw_frames: list[int],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not samples:
            raise ValueError(f"{method} produced no usable samples")
        total_frames = int(extra.pop("_total_frames")) if extra and "_total_frames" in extra else None
        out_of_range = []
        if total_frames is not None:
            out_of_range = [frame for frame in raw_frames if frame < 0 or frame >= total_frames]
        frames = [int(sample["frame"]) for sample in samples]
        validation_status = "warning" if out_of_range else "ok"
        validation_message = (
            f"{len(out_of_range)} raw sample frame(s) were clipped only for display alignment"
            if out_of_range
            else "Raw output parsed and frame indices are within the rollout bounds"
        )
        result = {
            "available": True,
            "method": method,
            "label": BASELINE_LABELS[method],
            "run": run_summary,
            "samples": samples,
            "sample_count": len(samples),
            "raw_files": [self._relative(path) for path in raw_files],
            "validation": {
                "status": validation_status,
                "message": validation_message,
                "raw_sample_count": len(raw_frames),
                "analysis_frame_min": min(frames),
                "analysis_frame_max": max(frames),
                "out_of_range_raw_frames": out_of_range,
            },
        }
        if extra:
            result.update(extra)
        return result

    def _read_safe(self, run_path: Path, rollout: dict[str, Any], run_summary: dict[str, Any]) -> dict[str, Any]:
        path = run_path / "raw" / rollout["id"] / "safe_features.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        first_timestep = int(rollout.get("first_environment_timestep") or 0)
        signals = []
        raw_frames = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_frame = int(row["action_timestep"]) if row.get("action_timestep") not in (None, "") else len(signals) + first_timestep
                frame = raw_frame - first_timestep
                values = {}
                for name, value in row.items():
                    if name == "action_timestep" or name.endswith("_rmean"):
                        continue
                    number = self._number(value)
                    if number is not None:
                        values[name] = number
                if values:
                    raw_frames.append(frame)
                    signals.append({"frame": frame, "raw_frame": raw_frame, "signals": values})
        return self._pack("safe", run_summary, signals, [path], raw_frames, {"_total_frames": int(rollout["total_frames"]), "kind": "numeric_handcrafted_features"})

    def _read_procvlm(self, run_path: Path, rollout: dict[str, Any], run_summary: dict[str, Any]) -> dict[str, Any]:
        path = run_path / "raw" / rollout["id"] / "procvlm_raw.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        total_frames = int(rollout["total_frames"])
        samples = []
        raw_frames = []
        for row in self._read_jsonl(path):
            raw_frame = int(row["frame_index"])
            value = self._number(row.get("progress"))
            if value is None:
                continue
            frame = min(max(raw_frame, 0), total_frames - 1)
            sample = {"frame": frame, "raw_frame": raw_frame, "signals": {"progress": value}}
            for field in ("model_output", "reasoning"):
                if row.get(field):
                    sample[field] = str(row[field])[:12000]
            samples.append(sample)
            raw_frames.append(raw_frame)
        return self._pack("procvlm", run_summary, samples, [path], raw_frames, {"_total_frames": total_frames, "kind": "model_text_and_progress"})

    def _read_rynnvalue(self, run_path: Path, rollout: dict[str, Any], run_summary: dict[str, Any]) -> dict[str, Any]:
        paths = sorted((run_path / "raw" / rollout["id"]).glob("*/raw_model_outputs.json"))
        if not paths:
            raise FileNotFoundError(run_path / "raw" / rollout["id"] / "*/raw_model_outputs.json")
        if len(paths) != 1:
            raise ValueError(f"Expected one RynnValue output, found {len(paths)}")
        path = paths[0]
        total_frames = int(rollout["total_frames"])
        raw = json.loads(path.read_text(encoding="utf-8"))
        values = raw["values"]
        indices = raw["sampled_indices"]
        samples = []
        raw_frames = []
        parsed = raw.get("parsed_analysis")
        analysis_text = raw.get("analysis_text")
        for index, value in zip(indices, values):
            number = self._number(value)
            if number is None:
                continue
            raw_frame = int(index)
            frame = min(max(raw_frame, 0), total_frames - 1)
            samples.append({
                "frame": frame,
                "raw_frame": raw_frame,
                "signals": {"value": number},
                "analysis_text": str(analysis_text)[:12000] if analysis_text else None,
                "parsed_analysis": parsed,
            })
            raw_frames.append(raw_frame)
        return self._pack("rynnvalue", run_summary, samples, [path], raw_frames, {"_total_frames": int(rollout["total_frames"]), "kind": "value_head_and_analysis"})

    def _read_robo_dopamine(self, run_path: Path, rollout: dict[str, Any], run_summary: dict[str, Any]) -> dict[str, Any]:
        result_path = run_path / "raw" / rollout["id"] / "worker_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        prediction_path = self._project_path(result["raw_model_output"])
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        rows = json.loads(prediction_path.read_text(encoding="utf-8"))
        total_frames = int(rollout["total_frames"])
        samples = []
        raw_frames = []
        perspective_paths = []
        for perspective in (result.get("perspective_outputs") or {}).values():
            if isinstance(perspective, dict) and perspective.get("raw_model_output"):
                candidate = self._project_path(str(perspective["raw_model_output"]))
                if candidate.is_file():
                    perspective_paths.append(candidate)
        for row in rows:
            image = row.get("image") or []
            frame_match = re.search(r"frame_([0-9]+)[.]png", str(image[5])) if len(image) > 5 else None
            if frame_match is None:
                frame_match = re.search(r"af_([0-9]+)$", str(row.get("id", "")))
            if frame_match is None:
                continue
            raw_frame = int(frame_match.group(1))
            progress = self._number(row.get("progress"))
            hop = self._number(row.get("hop"))
            signals = {}
            if progress is not None:
                signals["progress"] = progress
            if hop is not None:
                signals["hop"] = hop
            for mode, value in (row.get("component_progress") or {}).items():
                number = self._number(value)
                if number is not None:
                    signals[f"{mode}_progress"] = number
            for mode, value in (row.get("component_hop") or {}).items():
                number = self._number(value)
                if number is not None:
                    signals[f"{mode}_hop"] = number
            if not signals:
                continue
            samples.append({
                "frame": min(max(raw_frame, 0), total_frames - 1),
                "raw_frame": raw_frame,
                "signals": signals,
                "pred": str(row.get("pred", ""))[:12000],
            })
            raw_frames.append(raw_frame)
        files = [result_path, prediction_path, *perspective_paths]
        return self._pack("robo_dopamine", run_summary, samples, files, raw_frames, {"_total_frames": total_frames, "kind": "model_scores_and_progress"})

    
    def _read_densereward(self, run_path: Path, rollout: dict[str, Any], run_summary: dict[str, Any]) -> dict[str, Any]:
        output_dir = run_path / "raw" / rollout["id"]
        path = output_dir / "densereward_raw.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        total_frames = int(rollout["total_frames"])
        samples = []
        raw_frames = []
        for row in self._read_jsonl(path):
            if row.get("frame_index") in (None, ""):
                continue
            raw_frame = int(row["frame_index"])
            reward = self._number(row.get("reward"))
            if reward is None:
                continue
            sample = {
                "frame": min(max(raw_frame, 0), total_frames - 1),
                "raw_frame": raw_frame,
                "signals": {"reward": reward},
                "model_output": str(row.get("raw_text", ""))[:12000],
                "reasoning": str(row.get("reason", ""))[:2000],
                "sampled_frame_indices": [int(item) for item in (row.get("sampled_frame_indices") or [])],
            }
            samples.append(sample)
            raw_frames.append(raw_frame)
        worker_result = output_dir / "worker_result.json"
        files = [path, worker_result] if worker_result.is_file() else [path]
        return self._pack(
            "densereward",
            run_summary,
            samples,
            files,
            raw_frames,
            {"_total_frames": total_frames, "kind": "three_frame_reward_score"},
        )

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def _read_method(self, method: str, run_path: Path, rollout: dict[str, Any], run_summary: dict[str, Any]) -> dict[str, Any]:
        if method == "safe":
            return self._read_safe(run_path, rollout, run_summary)
        if method == "procvlm":
            return self._read_procvlm(run_path, rollout, run_summary)
        if method == "rynnvalue":
            return self._read_rynnvalue(run_path, rollout, run_summary)
        if method == "robo_dopamine":
            return self._read_robo_dopamine(run_path, rollout, run_summary)
        if method == "densereward":
            return self._read_densereward(run_path, rollout, run_summary)
        raise ValidationError("Unknown baseline method")

    def evaluation(
        self,
        rollout: dict[str, Any],
        *,
        condition: str = "full_instruction",
        source_rollout_id: str | None = None,
        variant_available: bool = True,
        unavailable_reason: str | None = None,
        run_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if condition not in INSTRUCTION_VARIANT_CONDITIONS:
            raise ValidationError(
                "condition must be one of: " + ", ".join(INSTRUCTION_VARIANT_CONDITIONS)
            )
        methods = {}
        allowed_conditions = (
            {condition, "unknown"} if condition == "full_instruction" else {condition}
        )
        for method in BASELINE_METHODS:
            requested_run = (run_overrides or {}).get(method)
            explicit_run = None
            if variant_available and requested_run:
                explicit_run = self._explicit_run_candidate(
                    method,
                    requested_run,
                    allowed_conditions,
                )
                candidates = [explicit_run]
            else:
                candidates = self._run_candidates(method) if variant_available else []
            errors = []
            selected = None
            for run_path, metadata in candidates:
                run_summary = self._run_summary(run_path, metadata)
                if run_summary["instruction_condition"] not in allowed_conditions:
                    continue
                try:
                    selected = self._read_method(method, run_path, rollout, run_summary)
                    break
                except FileNotFoundError:
                    continue
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    errors.append(str(error))
            selection = {
                "mode": "explicit" if explicit_run else "automatic",
                "requested_run_root": (
                    self._relative(explicit_run[0]) if explicit_run else None
                ),
            }
            if selected is None:
                compatible_runs = [
                    self._run_summary(run_path, metadata)
                    for run_path, metadata in candidates
                    if self._run_instruction_condition(metadata) in allowed_conditions
                ]
                run = compatible_runs[0] if compatible_runs else None
                message = unavailable_reason or (
                    errors[-1] if errors else "No usable raw output for this rollout"
                )
                methods[method] = {
                    "available": False,
                    "method": method,
                    "label": BASELINE_LABELS[method],
                    "run": run,
                    "samples": [],
                    "sample_count": 0,
                    "raw_files": [],
                    "validation": {
                        "status": "missing",
                        "message": (
                            "Selected run has no usable output for this rollout"
                            if explicit_run and not errors
                            else message
                        ),
                        "raw_sample_count": 0,
                        "out_of_range_raw_frames": [],
                    },
                    "run_selection": selection,
                }
            else:
                selected["run_selection"] = selection
                methods[method] = selected
        return {
            "rollout_id": rollout["id"],
            "source_rollout_id": source_rollout_id or rollout["id"],
            "condition": condition,
            "condition_label": INSTRUCTION_VARIANT_LABELS[condition],
            "variant_available": variant_available,
            "method_order": list(BASELINE_METHODS),
            "methods": methods,
            "available_methods": [
                method for method, result in methods.items() if result["available"]
            ],
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }

    def _manifest_records(self) -> list[dict[str, Any]]:
        return load_manifest_records(self.manifest_path)

    def list_runs(self, scope: Any = "natural_observation") -> list[dict[str, Any]]:
        scope = validate_run_scope(scope)
        selected_ids = {record["id"] for record in select_scope_records(self._manifest_records(), scope)}
        summaries = []
        for method in BASELINE_METHODS:
            for run_path, metadata in self._run_candidates(method):
                run_ids = run_rollout_ids(run_path)
                missing = selected_ids - run_ids if run_ids else selected_ids
                summary = self._run_summary(run_path, metadata)
                summary.update({
                    "created_at": metadata.get("created_at"),
                    "manifest_sha256": metadata.get("manifest_sha256"),
                    "run_rollout_count": len(run_ids),
                    "run_rollout_ids": sorted(run_ids),
                    "selected_scope_rollouts": len(selected_ids),
                    "compatible": bool(selected_ids) and not missing and summary["selected_rollouts"] >= len(selected_ids),
                    "partial_compatible": method == "rynnvalue" and bool(selected_ids.intersection(run_ids)),
                    "missing_rollouts": len(missing),
                    "scope": scope,
                })
                summaries.append(summary)
        summaries.sort(key=lambda item: (str(item.get("baseline")), str(item.get("completed_at") or item.get("created_at") or "")), reverse=True)
        return summaries

    def _validate_gpu(self, value: Any) -> str:
        gpu = str(value or "").strip()
        if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", gpu):
            raise ValidationError("gpu must be a comma-separated list of CUDA device indices")
        return gpu

    @staticmethod
    def _integer(value: Any, name: str, minimum: int = 0, maximum: int = 1000000) -> int:
        if isinstance(value, bool):
            raise ValidationError(f"{name} must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValidationError(f"{name} must be an integer")
        try:
            number = int(value)
        except (TypeError, ValueError) as error:
            raise ValidationError(f"{name} must be an integer") from error
        if number < minimum or number > maximum:
            raise ValidationError(f"{name} must be between {minimum} and {maximum}")
        return number

    @staticmethod
    def _positive_integer(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValidationError(f"{name} must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValidationError(f"{name} must be an integer")
        try:
            number = int(value)
        except (TypeError, ValueError) as error:
            raise ValidationError(f"{name} must be an integer") from error
        if number < 1:
            raise ValidationError(f"{name} must be positive")
        return number

    def _validate_options(self, baseline: str, raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValidationError("options must be a JSON object")
        unknown = set(raw) - BASELINE_ADVANCED_FIELDS
        if unknown:
            raise ValidationError("Unknown baseline option(s): " + ", ".join(sorted(unknown)))
        unsupported = set(raw) - BASELINE_METHOD_OPTION_FIELDS[baseline]
        if unsupported:
            raise ValidationError(
                f"Options not supported by {baseline}: " + ", ".join(sorted(unsupported))
            )
        options = dict(raw)
        integer_fields = {
            "tensor_parallel_size": (1, 32),
            "procvlm_window_size": (1, 4096),
            "procvlm_max_sampled_frames": (1, 1000000),
            "procvlm_max_new_tokens": (1, 1000000),
            "rynn_num_frames": (1, 1000000),
            "rynn_num_steps": (1, 1000000),
            "rynn_evaluation_interval": (1, 1000000),
            "rynn_max_image_side": (1, 8192),
            "rynn_max_new_tokens": (1, 1000000),
            "robo_frame_interval": (1, 1000000),
            "robo_batch_size": (1, 4096),
            "densereward_frame_interval": (1, 1000000),
            "densereward_max_new_tokens": (1, 1000000),
        }
        for name, (minimum, maximum) in integer_fields.items():
            if name in options and options[name] is not None:
                options[name] = self._integer(options[name], name, minimum, maximum)
        if "rynn_batch_size" in options and options["rynn_batch_size"] is not None:
            options["rynn_batch_size"] = self._positive_integer(options["rynn_batch_size"], "rynn_batch_size")
        for name in ("dtype", "robo_eval_mode"):
            if name in options and options[name] is not None:
                value = str(options[name]).strip()
                if not value or len(value) > 80:
                    raise ValidationError(f"{name} must be a non-empty short string")
                options[name] = value
        if "robo_eval_mode" in options and options["robo_eval_mode"] not in {"fused", "forward", "incremental", "backward"}:
            raise ValidationError("robo_eval_mode must be fused, forward, incremental, or backward")
        for name in ("robot_description", "camera_description"):
            if name in options and options[name] is not None:
                value = str(options[name])
                if len(value) > 4000:
                    raise ValidationError(f"{name} is too long")
                options[name] = value
        for name in ("model_path", "goal_image"):
            if name in options and options[name] not in (None, ""):
                resolved = self._project_path(str(options[name]))
                if not resolved.is_file():
                    raise ValidationError(f"{name} does not exist inside the project: {options[name]}")
                options[name] = str(resolved)
        for name in ("render_video", "validate_environment", "dry_run"):
            if name in options and not isinstance(options[name], bool):
                raise ValidationError(f"{name} must be boolean")
        return options

    @staticmethod
    def _worker_gpu(value: Any) -> str:
        gpu = str(value if value is not None else "").strip()
        if not re.fullmatch(r"[0-9]+", gpu):
            raise ValidationError("worker gpu must be one numeric CUDA device index")
        return gpu

    def _worker_plan(
        self,
        baseline: str,
        total_start: int,
        total_end: int,
        gpu: str,
        parallel_workers: Any,
        raw_workers: Any,
    ) -> dict[str, Any]:
        """Validate or create scope-relative worker assignments for every method."""
        del baseline  # All methods use the same rollout-level assignment contract.
        if raw_workers is not None:
            if not isinstance(raw_workers, list) or not raw_workers:
                raise ValidationError("workers must be a non-empty array")
            requested_count = self._integer(
                parallel_workers if parallel_workers is not None else len(raw_workers),
                "parallel_workers",
                1,
            )
            if requested_count not in (1, len(raw_workers)):
                raise ValidationError(
                    "parallel_workers must be 1 or equal the number of workers"
                )
            assignments: list[dict[str, Any]] = []
            for index, raw in enumerate(raw_workers):
                if not isinstance(raw, dict):
                    raise ValidationError(f"workers[{index}] must be an object")
                unknown = set(raw) - {"gpu", "start_index", "end_index"}
                if unknown:
                    raise ValidationError(
                        f"Unknown workers[{index}] field(s): {', '.join(sorted(unknown))}"
                    )
                worker_gpu = self._worker_gpu(raw.get("gpu"))
                start = self._integer(
                    raw.get("start_index"), f"workers[{index}].start_index", 0
                )
                end = self._integer(
                    raw.get("end_index"), f"workers[{index}].end_index", 0
                )
                if start < total_start or end > total_end or end <= start:
                    raise ValidationError(
                        f"workers[{index}] range [{start},{end}) must be inside "
                        f"[{total_start},{total_end})"
                    )
                assignments.append({
                    "worker_index": index,
                    "gpu": worker_gpu,
                    "start_index": start,
                    "end_index": end,
                    "requested_count": end - start,
                    "unique_count": 0,
                    "duplicate_count": 0,
                })
        else:
            count = self._integer(parallel_workers, "parallel_workers", 1)
            gpu_ids = [part for part in gpu.split(",") if part]
            if not gpu_ids:
                raise ValidationError("gpu must contain at least one device index")
            total = total_end - total_start
            base, remainder = divmod(total, count)
            cursor = total_start
            assignments = []
            for index in range(count):
                width = base + (1 if index < remainder else 0)
                if width <= 0:
                    raise ValidationError(
                        "parallel_workers cannot exceed the number of selected rollouts"
                    )
                assignments.append({
                    "worker_index": index,
                    "gpu": gpu_ids[index % len(gpu_ids)],
                    "start_index": cursor,
                    "end_index": cursor + width,
                    "requested_count": width,
                    "unique_count": width,
                    "duplicate_count": 0,
                })
                cursor += width

        ownership: dict[int, int] = {}
        overlaps: set[int] = set()
        for assignment in assignments:
            for scope_index in range(
                assignment["start_index"], assignment["end_index"]
            ):
                owner = ownership.get(scope_index)
                if owner is None:
                    ownership[scope_index] = assignment["worker_index"]
                else:
                    overlaps.add(scope_index)
                    assignment["duplicate_count"] += 1
        for assignment in assignments:
            assignment["unique_count"] = (
                assignment["requested_count"] - assignment["duplicate_count"]
            )
        gaps = sorted(set(range(total_start, total_end)) - set(ownership))
        worker_progress = [
            {
                **assignment,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "pending_jobs": assignment["unique_count"],
                "status": "queued",
                "started_at": None,
                "completed_at": None,
                "error": None,
            }
            for assignment in assignments
        ]
        return {
            "parallel_workers": len(assignments),
            "worker_assignments": assignments,
            "worker_progress": worker_progress,
            "overlaps": sorted(overlaps),
            "gaps": gaps,
            "unique_selected_rollouts": len(ownership),
        }

    def _baseline_command(
        self,
        baseline: str,
        scope: str,
        gpu: str,
        utilization: float,
        run_parent: Path,
        options: dict[str, Any],
        start_index: int = 0,
        limit: int | None = None,
        end_index: int | None = None,
        worker_assignments: list[dict[str, Any]] | None = None,
        parallel_workers: int = 1,
    ) -> list[str]:
        runner = self.project_root / "tools" / "baselines" / "run_lf3r_baseline.py"
        if not runner.is_file():
            raise ValidationError("Baseline runner is not installed")
        command = [
            sys.executable, str(runner),
            "--baseline", baseline,
            "--manifest", str(self.manifest_path),
            "--data-root", str(self.project_root),
            "--output-dir", str(run_parent),
            "--logs-dir", str(self.web_logs_root),
            "--gpu", gpu,
            "--vllm-free-memory-fraction", str(utilization),
            "--continue-on-error",
        ]
        if scope in {"primary_natural", "reference_natural"}:
            command.extend(["--partition", "natural_observation", "--dataset-role", scope])
        else:
            command.extend(["--partition", scope])
        if end_index is not None:
            command.extend(["--start-index", str(start_index), "--end-index", str(end_index)])
        else:
            if start_index:
                command.extend(["--start-index", str(start_index)])
            if limit is not None:
                command.extend(["--limit", str(limit)])
        if parallel_workers > 1:
            command.extend(["--parallel-workers", str(parallel_workers)])
        if worker_assignments:
            for assignment in worker_assignments:
                command.extend([
                    "--worker-spec",
                    f"{assignment['gpu']}:{assignment['start_index']}:{assignment['end_index']}",
                ])
        if baseline == "robo_dopamine" and "robo_eval_mode" not in options:
            # The web single-rollout endpoint has no advanced-options payload.
            # Make its default explicit and keep batch/API callers consistent.
            options = {**options, "robo_eval_mode": "fused"}
        flag_values = {
            "model_path": "--model-path",
            "dtype": "--dtype",
            "tensor_parallel_size": "--tensor-parallel-size",
            "procvlm_window_size": "--procvlm-window-size",
            "procvlm_max_sampled_frames": "--procvlm-max-sampled-frames",
            "procvlm_max_new_tokens": "--procvlm-max-new-tokens",
            "rynn_num_frames": "--rynn-num-frames",
            "rynn_num_steps": "--rynn-num-steps",
            "rynn_evaluation_interval": "--rynn-evaluation-interval",
            "rynn_batch_size": "--rynn-batch-size",
            "rynn_max_image_side": "--rynn-max-image-side",
            "rynn_max_new_tokens": "--rynn-max-new-tokens",
            "robot_description": "--robot-description",
            "camera_description": "--camera-description",
            "robo_frame_interval": "--robo-frame-interval",
            "robo_batch_size": "--robo-batch-size",
            "robo_eval_mode": "--robo-eval-mode",
            "goal_image": "--goal-image",
            "densereward_frame_interval": "--densereward-frame-interval",
            "densereward_max_new_tokens": "--densereward-max-new-tokens",
        }
        for name, flag in flag_values.items():
            if name in options and options[name] not in (None, ""):
                command.extend([flag, str(options[name])])
        if options.get("render_video"):
            command.append("--render-video")
        if options.get("dry_run"):
            command.append("--dry-run")
        if options.get("validate_environment"):
            command.append("--validate-environment")
        return command

    def _new_job(
        self,
        command: list[str],
        baseline: str,
        scope: str,
        selected_count: int,
        gpu: str,
        utilization: float,
        run_parent: Path,
        job_id: str,
        rollout_id: str | None = None,
        *,
        parallel_workers: int = 1,
        worker_assignments: list[dict[str, Any]] | None = None,
        worker_progress: list[dict[str, Any]] | None = None,
        unique_selected_rollouts: int | None = None,
        overlaps: list[int] | None = None,
        gaps: list[int] | None = None,
    ) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "job_type": "baseline",
            "baseline_mode": "rollout" if rollout_id else "batch",
            "baseline": baseline,
            "rollout_id": rollout_id,
            "scope": scope,
            "selected_rollouts": selected_count,
            "unique_selected_rollouts": (
                selected_count if unique_selected_rollouts is None else unique_selected_rollouts
            ),
            "completed_jobs": 0,
            "failed_jobs": 0,
            "pending_jobs": (
                selected_count if unique_selected_rollouts is None else unique_selected_rollouts
            ),
            "parallel_workers": parallel_workers,
            "worker_assignments": worker_assignments or [],
            "worker_progress": worker_progress or [],
            "overlaps": overlaps or [],
            "gaps": gaps or [],
            "command": command,
            "gpu": gpu,
            "memory_utilization": utilization,
            "memory_scope": "not_applicable" if baseline == "densereward" else "free_gpu_memory",
            "run_parent": self._relative(run_parent),
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "run_root": None,
            "log_path": self._relative(self.web_logs_root / f"{job_id}.log"),
            "error": None,
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "interpreter": str(sys.executable),
        }

    def start_run(self, rollout: dict[str, Any], baseline: str, gpu: str, memory_utilization: Any = 0.80) -> dict[str, Any]:
        if baseline not in BASELINE_METHODS:
            raise ValidationError("Invalid baseline method")
        gpu = self._validate_gpu(gpu)
        if isinstance(memory_utilization, bool):
            raise ValidationError("memory_utilization must be a number")
        try:
            utilization = float(memory_utilization)
        except (TypeError, ValueError) as error:
            raise ValidationError("memory_utilization must be a number") from error
        if not math.isfinite(utilization) or not 0.0 < utilization <= 1.0:
            raise ValidationError("memory_utilization must be in (0, 1]")
        self.web_output_root.mkdir(parents=True, exist_ok=True)
        self.web_logs_root.mkdir(parents=True, exist_ok=True)
        job_id = baseline + "-" + uuid.uuid4().hex[:12]
        self.coordinator.acquire(job_id, "baseline")
        try:
            command = self._baseline_command(
                baseline, "all", gpu, utilization, self.web_output_root, {},
                start_index=0, limit=None,
            )
            command.extend(["--rollout-id", rollout["id"]])
            job = self._new_job(
                command, baseline, "all", 1, gpu, utilization,
                self.web_output_root, job_id, rollout["id"],
            )
            with self.jobs_lock:
                self.jobs[job_id] = job
            self.tmux.submit(
                job,
                command,
                self.web_logs_root / f"{job_id}.log",
                interpreter=str(command[0]),
                on_poll=self._on_job_poll,
                on_finished=self._on_job_finished,
            )
        except Exception:
            with self.jobs_lock:
                self.jobs.pop(job_id, None)
            self.coordinator.release(job_id)
            raise
        return dict(job)

    def start_batch(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValidationError("Batch request must be a JSON object")
        allowed_fields = {
            "baseline", "scope", "gpu", "memory_utilization", "start_index", "end_index",
            "limit", "parallel_workers", "workers", "options",
        }
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            raise ValidationError("Unknown batch field(s): " + ", ".join(sorted(unknown_fields)))
        baseline = str(payload.get("baseline", ""))
        if baseline not in BASELINE_METHODS:
            raise ValidationError("Invalid baseline method")
        scope = validate_run_scope(payload.get("scope"))
        records = select_scope_records(self._manifest_records(), scope)
        if not records:
            raise ValidationError(f"No rollouts matched scope {scope}")
        gpu = self._validate_gpu(payload.get("gpu", "0"))
        if isinstance(payload.get("memory_utilization", 0.80), bool):
            raise ValidationError("memory_utilization must be a number")
        try:
            utilization = float(payload.get("memory_utilization", 0.80))
        except (TypeError, ValueError) as error:
            raise ValidationError("memory_utilization must be a number") from error
        if not math.isfinite(utilization) or not 0.0 < utilization <= 1.0:
            raise ValidationError("memory_utilization must be in (0, 1]")

        start_index = self._integer(payload.get("start_index", 0), "start_index", 0)
        end_value = payload.get("end_index")
        limit_value = payload.get("limit")
        if end_value not in (None, "") and limit_value not in (None, ""):
            raise ValidationError("end_index cannot be combined with limit")
        end_index = None
        limit = None
        if end_value not in (None, ""):
            end_index = self._integer(end_value, "end_index", 0)
            if end_index < start_index:
                raise ValidationError("end_index must be greater than or equal to start_index")
            total_end = min(end_index, len(records))
        elif limit_value not in (None, ""):
            limit = self._integer(limit_value, "limit", 1)
            total_end = min(start_index + limit, len(records))
        else:
            limit = None
            total_end = len(records)
        if total_end <= start_index:
            raise ValidationError("The requested range selected no rollouts")

        raw_workers = payload.get("workers")
        requested_parallel = payload.get("parallel_workers")
        if requested_parallel is None:
            requested_parallel = len(raw_workers) if raw_workers is not None else 1
        worker_plan = self._worker_plan(
            baseline,
            start_index,
            total_end,
            gpu,
            requested_parallel,
            raw_workers,
        )
        # Legacy requests without worker rows retain the original single
        # persistent-engine behavior. Explicit rows or parallel_workers opt
        # into the aggregate worker runner for every baseline method.
        use_worker_plan = (
            baseline in {"rynnvalue", "densereward"}
            or raw_workers is not None
            or worker_plan["parallel_workers"] > 1
        )
        options = self._validate_options(baseline, payload.get("options"))
        if baseline == "robo_dopamine":
            options.setdefault("robo_eval_mode", "fused")
        selected_count = total_end - start_index
        self.web_output_root.mkdir(parents=True, exist_ok=True)
        self.web_logs_root.mkdir(parents=True, exist_ok=True)
        job_id = baseline + "-batch-" + uuid.uuid4().hex[:12]
        self.coordinator.acquire(job_id, "baseline")
        try:
            run_parent = self.web_output_root / job_id
            run_parent.mkdir(parents=True, exist_ok=False)
            command = self._baseline_command(
                baseline,
                scope,
                gpu,
                utilization,
                run_parent,
                options,
                start_index,
                limit if (end_index is None and not use_worker_plan) else None,
                end_index=(total_end if (use_worker_plan or end_value not in (None, "")) else None),
                worker_assignments=(
                    worker_plan["worker_assignments"] if use_worker_plan else None
                ),
                parallel_workers=(worker_plan["parallel_workers"] if use_worker_plan else 1),
            )
            job = self._new_job(
                command,
                baseline,
                scope,
                selected_count,
                gpu,
                utilization,
                run_parent,
                job_id,
                parallel_workers=(worker_plan["parallel_workers"] if use_worker_plan else 1),
                worker_assignments=(worker_plan["worker_assignments"] if use_worker_plan else None),
                worker_progress=(worker_plan["worker_progress"] if use_worker_plan else None),
                unique_selected_rollouts=(worker_plan["unique_selected_rollouts"] if use_worker_plan else selected_count),
                overlaps=(worker_plan["overlaps"] if use_worker_plan else None),
                gaps=(worker_plan["gaps"] if use_worker_plan else None),
            )
            job["options"] = options
            job["start_index"] = start_index
            job["end_index"] = total_end
            job["limit"] = limit if (end_index is None and not use_worker_plan) else None
            job["requested_rollouts"] = selected_count
            job["selection_range"] = f"[{start_index},{total_end})"
            warnings: list[str] = []
            if use_worker_plan and worker_plan["overlaps"]:
                warnings.append(
                    f"overlap at {len(worker_plan['overlaps'])} scope index(es); first worker owns duplicates"
                )
            if use_worker_plan and worker_plan["gaps"]:
                warnings.append(
                    f"gap at {len(worker_plan['gaps'])} scope index(es); aggregate run is partial"
                )
            gpu_use = {}
            for assignment in worker_plan["worker_assignments"]:
                gpu_use[assignment["gpu"]] = gpu_use.get(assignment["gpu"], 0) + 1
            repeated_gpus = sorted(gpu_id for gpu_id, count in gpu_use.items() if count > 1)
            if repeated_gpus:
                warnings.append(
                    "GPU reused by multiple workers: " + ", ".join(repeated_gpus)
                )
            job["warnings"] = warnings
            with self.jobs_lock:
                self.jobs[job_id] = job
            self.tmux.submit(
                job,
                command,
                self.web_logs_root / f"{job_id}.log",
                interpreter=str(command[0]),
                on_poll=self._on_job_poll,
                on_finished=self._on_job_finished,
            )
        except Exception:
            with self.jobs_lock:
                self.jobs.pop(job_id, None)
            self.coordinator.release(job_id)
            raise
        return dict(job)

    def _find_job_run(self, job: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
        parent_value = job.get("run_parent")
        if parent_value:
            try:
                parent = self._project_path(str(parent_value))
            except ValidationError:
                parent = self.web_output_root
        else:
            parent = self.web_output_root
        candidates = []
        if parent.is_dir():
            for metadata_path in parent.glob("*/run.json"):
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                candidates.append((metadata_path.stat().st_mtime, metadata_path.parent, metadata))
        if not candidates:
            return None, None
        _, run_path, metadata = max(candidates, key=lambda item: item[0])
        return run_path, metadata

    def _refresh_progress(self, job_id: str) -> None:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            run_path, metadata = self._find_job_run(job)
            if run_path is None or metadata is None:
                return
            job["run_root"] = self._relative(run_path)
            for field in (
                "selected_rollouts", "unique_selected_rollouts", "completed_jobs",
                "failed_jobs", "pending_jobs", "parallel_workers", "duplicate_assignments",
            ):
                if field in metadata and metadata[field] is not None:
                    try:
                        job[field] = int(metadata[field])
                    except (TypeError, ValueError):
                        pass
            for field in ("worker_assignments", "worker_progress", "overlaps", "gaps"):
                if field in metadata:
                    job[field] = metadata[field]
            job["run_status"] = metadata.get("status")

    def _on_job_loaded(self, job: dict[str, Any]) -> None:
        job_id = str(job["job_id"])
        with self.jobs_lock:
            self.jobs[job_id] = job
        if job.get("status") in {"queued", "running"}:
            self.coordinator.restore(job_id, "baseline")
        self._refresh_progress(job_id)

    def _on_job_poll(self, job: dict[str, Any]) -> None:
        self._refresh_progress(str(job["job_id"]))

    def _on_job_finished(
        self,
        job: dict[str, Any],
        return_code: int | None,
        reason: str | None,
    ) -> None:
        job_id = str(job["job_id"])
        try:
            self._refresh_progress(job_id)
            with self.jobs_lock:
                if reason:
                    job["status"] = "failed"
                    job["error"] = reason
                elif job.get("run_status") == "complete_with_errors":
                    job["status"] = "complete_with_errors"
                    job["error"] = job.get("error") or "Aggregate run completed with partial coverage or worker errors"
                elif return_code == 0:
                    job["status"] = "complete"
                    job["error"] = None
                elif return_code == 75:
                    job["status"] = "memory_blocked"
                    job["error"] = "Baseline runner refused to start because the memory gate did not pass"
                else:
                    job["status"] = "failed"
                    job["error"] = job.get("error") or f"Baseline runner exited with code {return_code}"
                job["return_code"] = return_code
                job["finished_at"] = job.get("finished_at") or dt.datetime.now(dt.timezone.utc).isoformat()
        finally:
            self.coordinator.release(job_id)

    def job(self, job_id: str) -> dict[str, Any]:
        with self.jobs_lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
        self._refresh_progress(job_id)
        with self.jobs_lock:
            return dict(self.jobs[job_id])

    def list_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        return self.tmux.list("baseline", status)

    def log(self, job_id: str, tail: Any = 200) -> dict[str, Any]:
        job = self.job(job_id)
        try:
            count = self._integer(tail, "tail", 1, 2000)
        except ValidationError:
            count = 200
        path = self._project_path(job["log_path"])
        if not path.is_file():
            return {"job_id": job_id, "lines": [], "text": ""}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]
        return {"job_id": job_id, "lines": lines, "text": "\n".join(lines)}


class AnalysisJobService:
    """Run the existing temporal-analysis script on selected completed outputs."""

    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        annotation_root: Path,
        baselines: BaselineService,
        coordinator: JobCoordinator,
        tmux: TmuxJobSupervisor,
        analysis_python: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.annotation_root = annotation_root.resolve()
        self.baselines = baselines
        self.coordinator = coordinator
        self.tmux = tmux
        self.analysis_root = self.project_root / "outputs" / "baseline_signal_analysis"
        self.log_root = self.project_root / "logs" / "baselines" / "analysis_web"
        configured_python = (
            analysis_python
            if analysis_python is not None
            else self.project_root / "conda_envs" / "LF3R-ananlyse" / "bin" / "python"
        )
        # Preserve the venv entrypoint symlink. Resolving it points at uv's
        # cache interpreter and bypasses the venv's site-packages.
        configured_python = Path(configured_python).expanduser()
        if not configured_python.is_absolute():
            configured_python = self.project_root / configured_python
        self.analysis_python = Path(os.path.abspath(configured_python))
        self.jobs: dict[str, dict[str, Any]] = {}
        self.jobs_lock = threading.Lock()
        self.tmux.register_handler(
            "analysis",
            self._on_job_loaded,
            self._on_job_poll,
            self._on_job_finished,
        )

    def _project_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        resolved = path.resolve() if path.is_absolute() else (self.project_root / path).resolve()
        try:
            resolved.relative_to(self.project_root)
        except ValueError as exc:
            raise ValidationError("Analysis path escapes project root") from exc
        return resolved

    def _relative(self, path: Path) -> str:
        try:
            return str(Path(os.path.abspath(path)).relative_to(self.project_root))
        except ValueError:
            return str(path)

    def environment_status(self) -> dict[str, Any]:
        path = self.analysis_python
        status: dict[str, Any] = {
            "path": self._relative(path),
            "absolute_path": str(path),
            "exists": path.is_file(),
            "executable": path.is_file() and os.access(path, os.X_OK),
            "ready": False,
            "dependencies": ["matplotlib", "numpy", "pandas"],
            "error": None,
        }
        if not status["executable"]:
            status["error"] = "Analysis environment Python executable is missing"
            return status
        try:
            result = subprocess.run(
                [str(path), "-c", "import matplotlib, numpy, pandas"],
                cwd=str(self.project_root),
                env={**os.environ, "MPLBACKEND": "Agg"},
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            status["error"] = str(error)
            return status
        if result.returncode != 0:
            status["error"] = (
                result.stderr or result.stdout or "Analysis dependency import failed"
            ).strip()[-2000:]
            return status
        status["ready"] = True
        return status

    def require_environment(self) -> None:
        status = self.environment_status()
        if not status["ready"]:
            raise AnalysisEnvironmentError(
                "Analysis environment unavailable: "
                + str(status.get("error") or status["path"])
            )

    @staticmethod
    def _integer(value: Any, name: str, minimum: int = 1, maximum: int = 1000000) -> int:
        if isinstance(value, bool):
            raise ValidationError(f"{name} must be an integer")
        try:
            number = int(value)
        except (TypeError, ValueError) as error:
            raise ValidationError(f"{name} must be an integer") from error
        if number < minimum or number > maximum:
            raise ValidationError(f"{name} must be between {minimum} and {maximum}")
        return number

    def _manifest_records(self) -> list[dict[str, Any]]:
        return load_manifest_records(self.manifest_path)

    def _validate_runs(
        self,
        raw_runs: Any,
        selected_ids: set[str],
        allow_partial_coverage: bool = False,
    ) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
        if not isinstance(raw_runs, dict):
            raise ValidationError("runs must be an object containing all baseline methods")
        missing_methods = [method for method in ANALYSIS_BASELINE_METHODS if not raw_runs.get(method)]
        if missing_methods:
            raise ValidationError("Missing analysis run(s): " + ", ".join(missing_methods))
        validated: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
        for method in ANALYSIS_BASELINE_METHODS:
            raw_value = raw_runs[method]
            raw_values = raw_value if isinstance(raw_value, list) else [raw_value]
            if method != "rynnvalue" and len(raw_values) != 1:
                raise ValidationError(f"{method} accepts one run root")
            if not raw_values or any(not isinstance(value, (str, Path)) for value in raw_values):
                raise ValidationError(f"{method} run must be a path or, for RynnValue, a list of paths")
            method_runs: list[tuple[Path, dict[str, Any]]] = []
            seen_ids: set[str] = set()
            duplicate_ids: set[str] = set()
            for raw_path in raw_values:
                run_path = self._project_path(str(raw_path))
                try:
                    run_path.relative_to(self.baselines.baseline_root.resolve())
                except ValueError as exc:
                    raise ValidationError(f"{method} run must be inside outputs/baselines") from exc
                metadata_path = run_path / "run.json"
                if not metadata_path.is_file():
                    raise ValidationError(f"{method} run.json is missing")
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ValidationError(f"Invalid {method} run metadata") from error
                if metadata.get("baseline") != method:
                    raise ValidationError(f"Selected run is not a {method} run")
                if metadata.get("status") not in BASELINE_RUN_STATUSES:
                    raise ValidationError(f"{method} run is not complete")
                run_ids = run_rollout_ids(run_path)
                duplicate_ids.update(seen_ids.intersection(run_ids))
                seen_ids.update(run_ids)
                try:
                    int(metadata.get("selected_rollouts") or 0)
                except (TypeError, ValueError) as error:
                    raise ValidationError(f"{method} run has an invalid selected_rollouts count") from error
                method_runs.append((run_path, metadata))
            if duplicate_ids:
                raise ValidationError(
                    f"Duplicate {method} rollout IDs across selected runs: {sorted(duplicate_ids)[:5]}"
                )
            missing = sorted(selected_ids - seen_ids)
            if missing and not allow_partial_coverage:
                raise ValidationError(f"{method} run does not cover {len(missing)} selected rollout(s)")
            if not allow_partial_coverage and sum(
                int(metadata.get("selected_rollouts") or 0)
                for _, metadata in method_runs
            ) < len(selected_ids):
                raise ValidationError(f"{method} run selected fewer rollouts than the requested scope")
            validated[method] = method_runs
        return validated

    def start_run(self, payload: Any) -> dict[str, Any]:
        self.require_environment()
        if not isinstance(payload, dict):
            raise ValidationError("Analysis request must be a JSON object")
        allowed_fields = {
            "scope", "runs", "pre_window_frames", "post_window_frames",
            "background_stride_frames", "output_label", "allow_partial_coverage",
        }
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            raise ValidationError("Unknown analysis field(s): " + ", ".join(sorted(unknown_fields)))
        scope = validate_run_scope(payload.get("scope"))
        records = self._manifest_records()
        records = select_scope_records(records, scope)
        if not records:
            raise ValidationError(f"No rollouts matched scope {scope}")
        selected_ids = {record["id"] for record in records}
        raw_runs = payload.get("runs")
        allow_partial_coverage = bool(payload.get("allow_partial_coverage", False))
        if isinstance(raw_runs, dict) and isinstance(raw_runs.get("rynnvalue"), list) and len(raw_runs["rynnvalue"]) > 1:
            allow_partial_coverage = True
        validated_runs = self._validate_runs(
            raw_runs,
            selected_ids,
            allow_partial_coverage=allow_partial_coverage,
        )
        pre_window = self._integer(payload.get("pre_window_frames", 60), "pre_window_frames")
        post_window = self._integer(payload.get("post_window_frames", 60), "post_window_frames")
        stride = self._integer(payload.get("background_stride_frames", 30), "background_stride_frames")
        label = str(payload.get("output_label") or "web_analysis").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", label):
            raise ValidationError("output_label must contain only letters, numbers, dot, underscore, or hyphen")
        script = self.project_root / "tools" / "analyze_baseline_temporal_signals.py"
        if not script.is_file():
            raise ValidationError("Temporal analysis script is not installed")
        job_id = "analysis-" + uuid.uuid4().hex[:12]
        workspace = self.analysis_root / ".web_jobs" / job_id
        output_temp = workspace / "output"
        output_final = self.analysis_root / (
            "web_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            + "_" + label + "_" + job_id[-8:]
        )
        selection_path = workspace / "selection.json"
        selection_doc = {
            "schema_version": 1,
            "scope": scope,
            "selection": [{"id": record["id"]} for record in records],
        }
        command = [
            str(self.analysis_python), str(script),
            "--selection", str(selection_path),
            "--manifest", str(self.manifest_path),
            "--annotations-dir", str(self.annotation_root / "records"),
            "--output-dir", str(output_temp),
            "--safe-run", str(validated_runs["safe"][0][0]),
            "--procvlm-run", str(validated_runs["procvlm"][0][0]),
            "--robo-dopamine-run", str(validated_runs["robo_dopamine"][0][0]),
            "--pre-window-frames", str(pre_window),
            "--post-window-frames", str(post_window),
            "--background-stride-frames", str(stride),
        ]
        for run_path, _metadata in validated_runs["rynnvalue"]:
            command.extend(["--rynnvalue-run", str(run_path)])
        self.analysis_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        (self.analysis_root / ".web_jobs").mkdir(parents=True, exist_ok=True)
        self.coordinator.acquire(job_id, "analysis")
        try:
            workspace.mkdir(parents=True, exist_ok=False)
            atomic_json_write(selection_path, selection_doc)
            job = {
                "job_id": job_id,
                "job_type": "analysis",
                "status": "queued",
                "scope": scope,
                "selected_rollouts": len(records),
                "allow_partial_coverage": allow_partial_coverage,
                "runs": {
                    method: (
                        [self._relative(pair[0]) for pair in pairs]
                        if len(pairs) > 1
                        else self._relative(pairs[0][0])
                    )
                    for method, pairs in validated_runs.items()
                },
                "run_source_counts": {
                    method: len(pairs) for method, pairs in validated_runs.items()
                },
                "parameters": {
                    "pre_window_frames": pre_window,
                    "post_window_frames": post_window,
                    "background_stride_frames": stride,
                },
                "command": command,
                "output_dir": self._relative(output_final),
                "output_temp": self._relative(output_temp),
                "selection_path": self._relative(selection_path),
                "log_path": self._relative(self.log_root / f"{job_id}.log"),
                "started_at": None,
                "finished_at": None,
                "return_code": None,
                "error": None,
                "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "interpreter": str(self.analysis_python),
            }
            with self.jobs_lock:
                self.jobs[job_id] = job
            self.tmux.submit(
                job,
                command,
                self.log_root / f"{job_id}.log",
                interpreter=str(self.analysis_python),
                environment={"MPLBACKEND": "Agg"},
                on_poll=self._on_job_poll,
                on_finished=self._on_job_finished,
            )
        except Exception:
            with self.jobs_lock:
                self.jobs.pop(job_id, None)
            self.coordinator.release(job_id)
            raise
        return dict(job)

    def _on_job_loaded(self, job: dict[str, Any]) -> None:
        job_id = str(job["job_id"])
        with self.jobs_lock:
            self.jobs[job_id] = job
        if job.get("status") in {"queued", "running"}:
            self.coordinator.restore(job_id, "analysis")

    def _on_job_poll(self, job: dict[str, Any]) -> None:
        job["last_polled_at"] = dt.datetime.now(dt.timezone.utc).isoformat()

    def _on_job_finished(
        self,
        job: dict[str, Any],
        return_code: int | None,
        reason: str | None,
    ) -> None:
        job_id = str(job["job_id"])
        error = reason
        try:
            if error is None and return_code == 0:
                output_temp = self._project_path(str(job["output_temp"]))
                output_final = self._project_path(str(job["output_dir"]))
                required = ("metadata.json", "event_metrics.jsonl", *ANALYSIS_TABLE_FILES.values())
                if not all((output_temp / name).is_file() for name in required):
                    raise OSError("Temporal analysis completed without all required artifacts")
                if output_final.exists():
                    raise OSError(f"Analysis output already exists: {output_final}")
                os.replace(output_temp, output_final)
                job["status"] = "complete"
            else:
                job["status"] = "failed"
                error = error or f"Temporal analysis exited with code {return_code}"
        except Exception as exc:
            job["status"] = "failed"
            error = str(exc)
        finally:
            with self.jobs_lock:
                job["return_code"] = return_code
                job["finished_at"] = job.get("finished_at") or dt.datetime.now(dt.timezone.utc).isoformat()
                job["error"] = error
            self.coordinator.release(job_id)

    def list_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        return self.tmux.list("analysis", status)

    def job(self, job_id: str) -> dict[str, Any]:
        with self.jobs_lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return dict(self.jobs[job_id])

    def log(self, job_id: str, tail: Any = 200) -> dict[str, Any]:
        job = self.job(job_id)
        try:
            count = self._integer(tail, "tail", 1, 2000)
        except ValidationError:
            count = 200
        path = self._project_path(job["log_path"])
        if not path.is_file():
            return {"job_id": job_id, "lines": [], "text": ""}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]
        return {"job_id": job_id, "lines": lines, "text": "\n".join(lines)}


GENERATION_CONFIGS = {
    "libero_10": {
        "label": "LIBERO-10",
        "script_name": "generate_libero10_natural.sh",
        "output_root": "outputs/openvla_libero",
        "run_prefix": "lf3r-data-natural-libero10-",
        "max_task": 9,
        "render_resolution": 256,
        "policy_resolution": 224,
        "record_resolution": 224,
    },
    "libero_spatial": {
        "label": "LIBERO-Spatial",
        "script_name": "generate_libero_spatial_native.sh",
        "output_root": "outputs/openvla_libero_spatial_native",
        "run_prefix": "lf3r-data-natural-libero-spatial-256-",
        "max_task": 9,
        "render_resolution": 256,
        "policy_resolution": 224,
        "record_resolution": 256,
    },
}
GENERATION_RUN_PREFIX = GENERATION_CONFIGS["libero_10"]["run_prefix"]
GENERATION_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
GENERATION_VIDEO_RE = re.compile(r"^task[0-9]+--ep[0-9]+--succ[01]\.mp4$")


class RolloutGenerationService:
    """Run an existing natural OpenVLA/LIBERO generator from the web UI."""

    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        coordinator: JobCoordinator,
        tmux: TmuxJobSupervisor,
    ) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        script_root = self.project_root / "tools" / "lf3r_annotator"
        self.scripts = {
            suite: script_root / str(config["script_name"])
            for suite, config in GENERATION_CONFIGS.items()
        }
        # Preserve these attributes for existing callers and test adapters.
        self.script = self.scripts["libero_10"]
        self.output_roots = {
            suite: self.project_root / str(config["output_root"])
            for suite, config in GENERATION_CONFIGS.items()
        }
        self.output_root = self.output_roots["libero_10"]
        self.log_root = self.project_root / "logs" / "rollouts" / "web_runs"
        self.jobs: dict[str, dict[str, Any]] = {}
        self.jobs_lock = threading.Lock()
        self.coordinator = coordinator
        self.tmux = tmux
        self.tmux.register_handler(
            "rollout_generation",
            self._on_job_loaded,
            self._on_job_poll,
            self._on_job_finished,
        )

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(path)

    @staticmethod
    def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
        if isinstance(value, bool):
            raise ValidationError(f"{name} must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValidationError(f"{name} must be an integer")
        try:
            number = int(value)
        except (TypeError, ValueError) as error:
            raise ValidationError(f"{name} must be an integer") from error
        if number < minimum or number > maximum:
            raise ValidationError(f"{name} must be between {minimum} and {maximum}")
        return number

    @staticmethod
    def _gpu(value: Any) -> str:
        gpu = str(value if value is not None else "0").strip()
        if not re.fullmatch(r"[0-9]+", gpu):
            raise ValidationError("gpu must be one numeric CUDA device index")
        return gpu

    @staticmethod
    def _task_suite(value: Any) -> str:
        suite = str(value or "libero_10").strip()
        if suite not in GENERATION_CONFIGS:
            raise ValidationError(
                "task_suite must be one of: " + ", ".join(GENERATION_CONFIGS)
            )
        return suite

    @staticmethod
    def _run_label(value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValidationError("run_label must be a string")
        label = value.strip()
        if label and not GENERATION_LABEL_RE.fullmatch(label):
            raise ValidationError(
                "run_label must contain only letters, numbers, dot, underscore, or hyphen"
            )
        return label

    def _output_root(self, task_suite: str) -> Path:
        # Keep the historical output_root override working for LIBERO-10 callers.
        return self.output_root if task_suite == "libero_10" else self.output_roots[task_suite]

    def _new_run_note(self, label: str, task_suite: str = "libero_10") -> str:
        config = GENERATION_CONFIGS[task_suite]
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = f"-{label}" if label else ""
        candidate = f"{config['run_prefix']}{stamp}{suffix}"
        output_dir = self._output_root(task_suite) / candidate
        if output_dir.exists():
            candidate = f"{candidate}-{uuid.uuid4().hex[:8]}"
        return candidate

    def _count_generated(self, run_note: str, task_suite: str = "libero_10") -> int:
        directory = self._output_root(task_suite) / run_note / task_suite
        if not directory.is_dir():
            return 0
        return sum(
            1 for path in directory.glob("*.mp4") if GENERATION_VIDEO_RE.fullmatch(path.name)
        )

    def _manifest_signature(self) -> str | None:
        if not self.manifest_path.is_file():
            return None
        digest = hashlib.sha256()
        with self.manifest_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _job_snapshot(self, job_id: str) -> dict[str, Any]:
        with self.jobs_lock:
            return dict(self.jobs[job_id])

    def start(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValidationError("Rollout generation request must be a JSON object")
        allowed = {
            "task_suite", "gpu", "task_start", "task_end", "trials", "seed",
            "run_label", "log_safe_features",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValidationError("Unknown rollout-generation field(s): " + ", ".join(sorted(unknown)))

        log_safe_features = payload.get("log_safe_features", True)
        if not isinstance(log_safe_features, bool):
            raise ValidationError("log_safe_features must be boolean")

        task_suite = self._task_suite(payload.get("task_suite", "libero_10"))
        config = GENERATION_CONFIGS[task_suite]
        gpu = self._gpu(payload.get("gpu", "0"))
        max_task = int(config["max_task"])
        task_start = self._integer(payload.get("task_start", 0), "task_start", 0, max_task)
        task_end = self._integer(payload.get("task_end", 3), "task_end", 0, max_task)
        if task_start > task_end:
            raise ValidationError("task_start must not be greater than task_end")
        trials = self._integer(payload.get("trials", 1), "trials", 1, 50)
        seed = self._integer(payload.get("seed", 7), "seed", 0, 2147483647)
        label = self._run_label(payload.get("run_label", ""))
        script = self.script if task_suite == "libero_10" else self.scripts[task_suite]
        if not script.is_file():
            raise ValidationError(f"{config['label']} natural rollout generator is not installed")

        output_root = self._output_root(task_suite)
        output_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        run_note = self._new_run_note(label, task_suite)
        output_dir = output_root / run_note
        if output_dir.exists():
            raise ValidationError("Generated run directory already exists; retry with a new label")

        job_id = "rollout-" + uuid.uuid4().hex[:12]
        command = [
            "bash",
            str(script),
            gpu,
            str(task_start),
            str(task_end),
            str(trials),
            str(seed),
            run_note,
        ]
        if log_safe_features:
            command.append("--log-safe-features")
        expected = (task_end - task_start + 1) * trials
        job = {
            "job_id": job_id,
            "job_type": "rollout_generation",
            "status": "queued",
            "task_suite": task_suite,
            "suite_label": config["label"],
            "gpu": gpu,
            "task_start": task_start,
            "task_end": task_end,
            "trials": trials,
            "seed": seed,
            "run_label": label,
            "log_safe_features": log_safe_features,
            "run_note": run_note,
            "requested_rollouts": expected,
            "expected_rollouts": expected,
            "completed_rollouts": 0,
            "render_resolution": config["render_resolution"],
            "policy_resolution": config["policy_resolution"],
            "record_resolution": config["record_resolution"],
            "generator_script": self._relative(script),
            "output_root": self._relative(output_root),
            "run_root": self._relative(output_dir),
            "output_dir": self._relative(output_dir),
            "log_path": self._relative(self.log_root / f"{job_id}.log"),
            "command": command,
            "memory_gate": {
                "kind": "memory_only",
                "minimum_free_mib": 30720,
                "maximum_used_fraction": 0.50,
                "gpu_utilization": "informational",
            },
            "manifest_rebuilt": False,
            "manifest_before": self._manifest_signature(),
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "error": None,
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "interpreter": "bash",
        }
        self.coordinator.acquire(job_id, "manifest_writer")
        try:
            with self.jobs_lock:
                self.jobs[job_id] = job
            self.tmux.submit(
                job,
                command,
                self.log_root / f"{job_id}.log",
                interpreter="bash",
                environment={
                    "MPLBACKEND": "Agg",
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                },
                on_poll=self._on_job_poll,
                on_finished=self._on_job_finished,
            )
        except Exception:
            with self.jobs_lock:
                self.jobs.pop(job_id, None)
            self.coordinator.release(job_id)
            raise
        return dict(job)

    def _on_job_loaded(self, job: dict[str, Any]) -> None:
        job_id = str(job["job_id"])
        with self.jobs_lock:
            self.jobs[job_id] = job
        if job.get("status") in {"queued", "running"}:
            self.coordinator.restore(job_id, "manifest_writer")

    def _on_job_poll(self, job: dict[str, Any]) -> None:
        with self.jobs_lock:
            task_suite = self._task_suite(job.get("task_suite", "libero_10"))
            job["completed_rollouts"] = self._count_generated(str(job["run_note"]), task_suite)

    def _on_job_finished(
        self,
        job: dict[str, Any],
        return_code: int | None,
        reason: str | None,
    ) -> None:
        job_id = str(job["job_id"])
        error = reason
        try:
            task_suite = self._task_suite(job.get("task_suite", "libero_10"))
            completed = self._count_generated(str(job["run_note"]), task_suite)
            manifest_after = self._manifest_signature()
            with self.jobs_lock:
                job["completed_rollouts"] = completed
                job["manifest_after"] = manifest_after
                job["manifest_rebuilt"] = (
                    return_code == 0
                    and manifest_after is not None
                    and manifest_after != job.get("manifest_before")
                )
                if error:
                    job["status"] = "failed"
                elif return_code == 75:
                    job["status"] = "memory_blocked"
                    error = "Generator refused to start because the memory-only GPU gate did not pass"
                elif return_code == 0:
                    job["status"] = "complete"
                else:
                    job["status"] = "failed"
                    error = f"Rollout generator exited with code {return_code}"
                job["error"] = error
                job["return_code"] = return_code
                job["finished_at"] = job.get("finished_at") or dt.datetime.now(dt.timezone.utc).isoformat()
        finally:
            self.coordinator.release(job_id)

    def job(self, job_id: str) -> dict[str, Any]:
        with self.jobs_lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return dict(self.jobs[job_id])

    def list_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        return self.tmux.list("rollout_generation", status)

    def log(self, job_id: str, tail: Any = 200) -> dict[str, Any]:
        job = self.job(job_id)
        try:
            count = self._integer(tail, "tail", 1, 2000)
        except ValidationError:
            count = 200
        path = self.project_root / job["log_path"]
        if not path.is_file():
            return {"job_id": job_id, "lines": [], "text": ""}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]
        return {"job_id": job_id, "lines": lines, "text": "\n".join(lines)}


class LF3RApplication:
    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        annotation_root: Path,
        analysis_python: Path | None = None,
        tmux_binary: str | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.static_dir = (Path(__file__).resolve().parent / "static").resolve()
        self.instruction_variant_manifest_path = (
            self.project_root
            / "tools"
            / "lf3r_annotator"
            / "instruction_variants"
            / "libero_10_v1"
            / "manifest.jsonl"
        )
        self._instruction_variant_index: dict[str, dict[str, dict[str, Any]]] | None = None
        self.store = AnnotationStore(annotation_root)
        self.settings = SettingsStore(self.project_root)
        self.analysis = AnalysisService(self.project_root, self.manifest_path, annotation_root)
        self.job_coordinator = JobCoordinator()
        self.tmux = TmuxJobSupervisor(self.project_root, tmux_binary=tmux_binary)
        self.baselines = BaselineService(
            self.project_root, self.manifest_path, self.job_coordinator, self.tmux
        )
        configured_analysis_python = (
            Path(os.environ["LF3R_ANALYSIS_PYTHON"]).expanduser()
            if analysis_python is None and os.environ.get("LF3R_ANALYSIS_PYTHON")
            else analysis_python
        )
        self.analysis_jobs = AnalysisJobService(
            self.project_root,
            self.manifest_path,
            annotation_root,
            self.baselines,
            self.job_coordinator,
            self.tmux,
            analysis_python=configured_analysis_python,
        )
        self.rollout_jobs = RolloutGenerationService(
            self.project_root, self.manifest_path, self.job_coordinator, self.tmux
        )
        self.tmux.recover()

    def load_instruction_variant_records(self) -> dict[str, dict[str, dict[str, Any]]]:
        if self._instruction_variant_index is not None:
            return self._instruction_variant_index
        path = self.instruction_variant_manifest_path
        if not path.is_file():
            self._instruction_variant_index = {}
            return self._instruction_variant_index
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for row in load_manifest_records(path):
            source_id = row.get("source_rollout_id") or row.get("source_id")
            condition = row.get("instruction_variant") or row.get("condition")
            if not isinstance(source_id, str) or not ROLLOUT_ID_RE.fullmatch(source_id):
                raise ValidationError("Invalid source rollout id in instruction variant manifest")
            if condition not in {"subtask_a", "subtask_b"}:
                continue
            if row.get("id") != source_id + "--" + condition:
                raise ValidationError("Instruction variant id does not match its source and condition")
            if condition in grouped.setdefault(source_id, {}):
                raise ValidationError("Duplicate instruction variant for " + source_id + ": " + condition)
            grouped[source_id][condition] = row
        self._instruction_variant_index = grouped
        return grouped

    def instruction_variant_options(self, record: dict[str, Any]) -> dict[str, dict[str, Any]]:
        source_id = str(record["id"])
        options: dict[str, dict[str, Any]] = {
            "full_instruction": {
                "id": source_id,
                "condition": "full_instruction",
                "label": INSTRUCTION_VARIANT_LABELS["full_instruction"],
                "instruction": record.get("task_description", ""),
                "instruction_type": "original_full_instruction",
                "available": True,
                "counterfactual": False,
            }
        }
        variants = self.load_instruction_variant_records().get(source_id, {})
        for condition in ("subtask_a", "subtask_b"):
            row = variants.get(condition)
            if row is None:
                continue
            if row.get("video_path") != record.get("video_path"):
                raise ValidationError("Instruction variant video does not match source rollout: " + source_id)
            options[condition] = {
                "id": row["id"],
                "condition": condition,
                "label": row.get("subtask_label") or INSTRUCTION_VARIANT_LABELS[condition],
                "instruction": row.get("task_description") or row.get("instruction", ""),
                "instruction_type": row.get("instruction_type", "counterfactual_single_subtask"),
                "subtask_label": row.get("subtask_label"),
                "subtask_subject": row.get("subtask_subject"),
                "subtask_target": row.get("subtask_target"),
                "available": True,
                "counterfactual": True,
            }
        return options

    def instruction_variant_for(
        self,
        record: dict[str, Any],
        condition: str,
    ) -> dict[str, Any] | None:
        if condition not in INSTRUCTION_VARIANT_CONDITIONS:
            raise ValidationError(
                "condition must be one of: " + ", ".join(INSTRUCTION_VARIANT_CONDITIONS)
            )
        if condition == "full_instruction":
            variant = dict(record)
            variant.update({
                "condition": "full_instruction",
                "instruction_variant": "full_instruction",
                "instruction_type": "original_full_instruction",
                "instruction": record.get("task_description", ""),
                "original_full_instruction": record.get("task_description", ""),
                "source_rollout_id": record["id"],
            })
            return variant
        row = self.load_instruction_variant_records().get(record["id"], {}).get(condition)
        if row is None:
            return None
        if row.get("video_path") != record.get("video_path"):
            raise ValidationError("Instruction variant video does not match source rollout: " + record["id"])
        return dict(row)

    def load_rollouts(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                rollout_id = record.get("id")
                if not isinstance(rollout_id, str) or not ROLLOUT_ID_RE.fullmatch(rollout_id):
                    raise ValidationError(f"Invalid rollout id at manifest line {line_number}")
                if rollout_id in seen:
                    raise ValidationError(f"Duplicate rollout id: {rollout_id}")
                seen.add(rollout_id)
                self.resolve_project_file(record["video_path"], ".mp4")
                records.append(record)
        return records

    def rollout_map(self) -> dict[str, dict[str, Any]]:
        return {record["id"]: record for record in self.load_rollouts()}

    def resolve_project_file(self, relative: str, suffix: str | None = None) -> Path:
        path = (self.project_root / relative).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValidationError("Manifest path escapes project root") from exc
        if suffix and path.suffix.lower() != suffix:
            raise ValidationError(f"Expected a {suffix} file")
        return path


class LF3RHandler(BaseHTTPRequestHandler):
    app: LF3RApplication
    server_version = "LF3RAnnotator/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        super().log_message(fmt, *args)

    def json_response(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json_error(self, status: int, message: str) -> None:
        self.json_response(status, {"error": message})

    def file_response(self, path: Path) -> None:
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", 'attachment; filename="' + path.name + '"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            if path == "/api/health":
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "manifest": str(self.app.manifest_path.relative_to(self.app.project_root)),
                        "rollouts": len(self.app.load_rollouts()),
                        "tmux": {
                            "available": self.app.tmux.available,
                            "persistent": True,
                            "persistent_jobs": len(self.app.tmux.list()),
                            "error": None if self.app.tmux.available else "tmux is required for persistent annotator jobs",
                        },
                        "analysis_environment": self.app.analysis_jobs.environment_status(),
                    },
                )
                return
            if path == "/api/settings":
                self.json_response(HTTPStatus.OK, self.app.settings.response())
                return
            if path == "/api/analysis":
                requested_view = (query.get("view", ["dashboard"])[0] or "dashboard").lower()
                compact = requested_view not in {"full", "legacy", "compatibility"}
                if query.get("include_details", [""])[0] in {"1", "true", "yes"}:
                    compact = False
                self.json_response(
                    HTTPStatus.OK,
                    {"analysis": self.app.analysis.response(compact=compact)},
                )
                return
            if path == "/api/analysis/details":
                detail_query = {
                    key: values[0] if values else ""
                    for key, values in query.items()
                }
                self.json_response(
                    HTTPStatus.OK,
                    self.app.analysis.details(detail_query),
                )
                return
            if path.startswith("/api/analysis/artifacts/"):
                name = path[len("/api/analysis/artifacts/"):]
                try:
                    artifact = self.app.analysis.artifact_path(name)
                except FileNotFoundError:
                    self.json_error(HTTPStatus.NOT_FOUND, "Analysis artifact is unavailable")
                    return
                self.file_response(artifact)
                return
            if path == "/api/rollouts":
                records = []
                for record in self.app.load_rollouts():
                    annotation = self.app.store.read(record["id"])
                    enriched = {
                        **record,
                        "annotation": annotation,
                        "annotation_status": annotation["review_status"] if annotation else "unreviewed",
                    }
                    options = self.app.instruction_variant_options(record)
                    enriched["instruction_variants"] = options
                    enriched["instruction_variant_conditions"] = list(options)
                    if self.app.instruction_variant_manifest_path.is_file():
                        enriched["instruction_variant_manifest"] = str(
                            self.app.instruction_variant_manifest_path.relative_to(self.app.project_root)
                        )
                    records.append(enriched)
                self.json_response(HTTPStatus.OK, {"rollouts": records})
                return
            if path == "/api/baselines/runs":
                scope = query.get("scope", ["natural_observation"])[0]
                self.json_response(
                    HTTPStatus.OK,
                    {"scope": scope, "runs": self.app.baselines.list_runs(scope)},
                )
                return
            if path == "/api/jobs":
                job_type = query.get("job_type", [None])[0] or None
                status = query.get("status", [None])[0] or None
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "jobs": self.app.tmux.list(job_type=job_type, status=status),
                        "tmux_available": self.app.tmux.available,
                    },
                )
                return
            if path.startswith("/api/rollout-jobs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[3] == "log":
                    job_id = parts[2]
                    tail = query.get("tail", ["200"])[0]
                    try:
                        log = self.app.rollout_jobs.log(job_id, tail)
                    except KeyError:
                        self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout-generation job")
                        return
                    self.json_response(HTTPStatus.OK, {"log": log})
                    return
                if len(parts) != 3:
                    self.json_error(HTTPStatus.NOT_FOUND, "Not found")
                    return
                job_id = parts[2]
                try:
                    job = self.app.rollout_jobs.job(job_id)
                except KeyError:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout-generation job")
                    return
                self.json_response(HTTPStatus.OK, {"job": job})
                return
            if path.startswith("/api/baseline-jobs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[3] == "log":
                    job_id = parts[2]
                    tail = query.get("tail", ["200"])[0]
                    try:
                        log = self.app.baselines.log(job_id, tail)
                    except KeyError:
                        self.json_error(HTTPStatus.NOT_FOUND, "Unknown baseline job")
                        return
                    self.json_response(HTTPStatus.OK, {"log": log})
                    return
                if len(parts) != 3:
                    self.json_error(HTTPStatus.NOT_FOUND, "Not found")
                    return
                job_id = parts[2]
                try:
                    job = self.app.baselines.job(job_id)
                except KeyError:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown baseline job")
                    return
                self.json_response(HTTPStatus.OK, {"job": job})
                return
            if path.startswith("/api/analysis-jobs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[3] == "log":
                    job_id = parts[2]
                    tail = query.get("tail", ["200"])[0]
                    try:
                        log = self.app.analysis_jobs.log(job_id, tail)
                    except KeyError:
                        self.json_error(HTTPStatus.NOT_FOUND, "Unknown analysis job")
                        return
                    self.json_response(HTTPStatus.OK, {"log": log})
                    return
                if len(parts) != 3:
                    self.json_error(HTTPStatus.NOT_FOUND, "Not found")
                    return
                job_id = parts[2]
                try:
                    job = self.app.analysis_jobs.job(job_id)
                except KeyError:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown analysis job")
                    return
                self.json_response(HTTPStatus.OK, {"job": job})
                return
            if path.startswith("/api/baselines/"):
                rollout_id = path.rsplit("/", 1)[-1]
                rollout = self.app.rollout_map().get(rollout_id)
                if not rollout:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                    return
                condition = query.get("condition", ["full_instruction"])[0] or "full_instruction"
                run_overrides = {
                    method: query.get("run_" + method, [""])[0]
                    for method in BASELINE_METHODS
                    if query.get("run_" + method, [""])[0]
                }
                variant = self.app.instruction_variant_for(rollout, condition)
                if variant is None:
                    evaluation = self.app.baselines.evaluation(
                        rollout,
                        condition=condition,
                        source_rollout_id=rollout_id,
                        variant_available=False,
                        unavailable_reason=(
                            "No prepared " + INSTRUCTION_VARIANT_LABELS[condition]
                            + " variant exists for this rollout."
                        ),
                        run_overrides=run_overrides,
                    )
                    evaluation.update({
                        "variant_id": None,
                        "instruction": None,
                        "instruction_type": None,
                    })
                else:
                    evaluation = self.app.baselines.evaluation(
                        variant,
                        condition=condition,
                        source_rollout_id=rollout_id,
                        variant_available=True,
                        run_overrides=run_overrides,
                    )
                    evaluation.update({
                        "variant_id": variant["id"],
                        "instruction": variant.get("task_description") or variant.get("instruction", ""),
                        "instruction_type": variant.get("instruction_type"),
                    })
                self.json_response(HTTPStatus.OK, {"evaluation": evaluation})
                return
            if path.startswith("/api/annotations/"):
                rollout_id = path.rsplit("/", 1)[-1]
                if rollout_id not in self.app.rollout_map():
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                    return
                self.json_response(
                    HTTPStatus.OK,
                    {"annotation": self.app.store.read(rollout_id)},
                )
                return
            if path.startswith("/api/videos/"):
                rollout_id = path.rsplit("/", 1)[-1]
                rollout = self.app.rollout_map().get(rollout_id)
                if not rollout:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                    return
                video = self.app.resolve_project_file(rollout["video_path"], ".mp4")
                self.serve_video(video)
                return
            if path == "/":
                self.serve_static(self.app.static_dir / "index.html")
                return
            if path.startswith("/static/"):
                relative = path[len("/static/") :]
                target = (self.app.static_dir / relative).resolve()
                try:
                    target.relative_to(self.app.static_dir)
                except ValueError:
                    self.json_error(HTTPStatus.FORBIDDEN, "Invalid static path")
                    return
                self.serve_static(target)
                return
            self.json_error(HTTPStatus.NOT_FOUND, "Not found")
        except JobConflictError as exc:
            self.json_error(HTTPStatus.CONFLICT, str(exc))
        except TmuxSupervisorError as exc:
            self.json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        except (ValidationError, json.JSONDecodeError) as exc:
            self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
        except OSError as exc:
            self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_PUT(self) -> None:
        try:
            path = unquote(urlparse(self.path).path)
            if path != "/api/settings":
                self.json_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.json_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                return
            if length <= 0 or length > 100_000:
                self.json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size")
                return
            payload = json.loads(self.rfile.read(length))
            settings = self.app.settings.write(payload)
            self.json_response(
                HTTPStatus.OK,
                {
                    "settings": settings,
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
        except (ValidationError, json.JSONDecodeError) as exc:
            self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
        except OSError as exc:
            self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        try:
            path = unquote(urlparse(self.path).path)
            if path in {"/api/baselines/run-batch", "/api/analysis/run", "/api/rollouts/generate"}:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.json_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                    return
                if length <= 0 or length > 100_000:
                    self.json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size")
                    return
                payload = json.loads(self.rfile.read(length))
                if path == "/api/baselines/run-batch":
                    job = self.app.baselines.start_batch(payload)
                elif path == "/api/analysis/run":
                    job = self.app.analysis_jobs.start_run(payload)
                else:
                    job = self.app.rollout_jobs.start(payload)
                self.json_response(HTTPStatus.ACCEPTED, {"job": job})
                return
            if path.startswith("/api/baselines/run/"):
                rollout_id = path.rsplit("/", 1)[-1]
                rollout = self.app.rollout_map().get(rollout_id)
                if not rollout:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.json_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                    return
                if length > 100_000:
                    self.json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size")
                    return
                payload = json.loads(self.rfile.read(length)) if length else {}
                job = self.app.baselines.start_run(
                    rollout,
                    str(payload.get("baseline", "")),
                    str(payload.get("gpu", "0")),
                    payload.get("memory_utilization", 0.80),
                )
                self.json_response(HTTPStatus.ACCEPTED, {"job": job})
                return
            if not path.startswith("/api/annotations/"):
                self.json_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            rollout_id = path.rsplit("/", 1)[-1]
            rollout = self.app.rollout_map().get(rollout_id)
            if not rollout:
                self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.json_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                return
            if length <= 0 or length > 1_000_000:
                self.json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size")
                return
            payload = json.loads(self.rfile.read(length))
            record = self.app.store.write(rollout, payload)
            self.json_response(HTTPStatus.OK, {"annotation": record})
        except JobConflictError as exc:
            self.json_error(HTTPStatus.CONFLICT, str(exc))
        except AnalysisEnvironmentError as exc:
            self.json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        except TmuxSupervisorError as exc:
            self.json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        except (ValidationError, json.JSONDecodeError) as exc:
            self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
        except OSError as exc:
            self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def serve_static(self, path: Path) -> None:
        if not path.is_file():
            self.json_error(HTTPStatus.NOT_FOUND, "Static file not found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def serve_video(self, path: Path) -> None:
        if not path.is_file():
            self.json_error(HTTPStatus.NOT_FOUND, "Video not found")
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            left, right = match.groups()
            if left:
                start = int(left)
                end = int(right) if right else end
            elif right:
                count = int(right)
                start = max(size - count, 0)
            if start >= size or start > end:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def make_handler(app: LF3RApplication) -> type[LF3RHandler]:
    class BoundHandler(LF3RHandler):
        pass

    BoundHandler.app = app
    return BoundHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the LF3R annotation tool locally")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; keep loopback for SSH forwarding")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--annotations", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    manifest = args.manifest or project_root / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
    annotations = args.annotations or project_root / "annotations/failure_annotations/v1"
    app = LF3RApplication(project_root, manifest, annotations)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"LF3R annotator: http://{args.host}:{server.server_port}")
    print(f"Manifest: {manifest}")
    print(f"Annotations: {annotations}")

    def stop_server(_signum: int, _frame: Any) -> None:
        print("Shutdown requested; stopping LF3R annotator...")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
