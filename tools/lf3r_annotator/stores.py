"""Persistent settings and annotation stores for the LF3R WebUI."""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from backend_core import ROLLOUT_ID_RE, ValidationError


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
