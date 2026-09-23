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
import sqlite3
import statistics
import subprocess
import sys
import time
import tempfile
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlparse
from task_supervisor import TmuxJobSupervisor, TmuxSupervisorError
from analysis_constants import (
    ANALYSIS_ARTIFACT_NAMES,
    ANALYSIS_BASELINE_METHODS,
    ANALYSIS_DETAIL_FILTERS,
    ANALYSIS_DETAIL_KINDS,
    ANALYSIS_DETAIL_SORTS,
    ANALYSIS_EVENT_ARRAY_FIELDS,
    ANALYSIS_EVENT_FIELDS,
    ANALYSIS_LOCALIZATION_EVENT_FIELDS,
    ANALYSIS_LOCALIZATION_TABLE_FILES,
    ANALYSIS_RECORD_FIELDS,
    ANALYSIS_TABLE_FILES,
    CHANGEPOINT_EVENT_FIELDS,
    CHANGEPOINT_TABLE_FILES,
    EVENT_TRIGGERED_TABLE_FILES,
    ROBO_HOP_EXTENDED_FILES,
    ROBO_HOP_REQUIRED_FILES,
    ROBO_HOP_TABLE_FILES,
    ROBO_LABEL_LOSS_REQUIRED_FILES,
    ROBO_LOCALIZATION_HEAD_REQUIRED_FILES,
)
from analysis_jobs import AnalysisJobService
from analysis_service import AnalysisService
from backend_core import (
    AnalysisEnvironmentError,
    INSTRUCTION_VARIANT_CONDITIONS,
    JobConflictError,
    JobCoordinator,
    ROLLOUT_ID_RE,
    RESERVED_RUN_SCOPES,
    ValidationError,
    _is_controlled_record,
    atomic_json_write,
    load_manifest_records,
    record_matches_scope,
    run_rollout_ids,
    select_scope_records,
    validate_instruction_condition,
    validate_run_scope,
)
from baseline_constants import (
    BASELINE_ADVANCED_FIELDS,
    BASELINE_LABELS,
    BASELINE_METHOD_OPTION_FIELDS,
    BASELINE_METHODS,
    BASELINE_RESULT_FILTERS,
    BASELINE_RUN_STATUSES,
    INSTRUCTION_VARIANT_LABELS,
)
from baseline_index import BaselineRunIndex
from baseline_service import BaselineService
from rollout_service import RolloutGenerationService
from stores import (
    AnnotationStore,
    DEFAULT_SETTINGS,
    FAILURE_TYPES,
    OUTCOME_LABELS,
    REVIEW_STATUSES,
    SETTING_COLOR_FIELDS,
    SETTING_COLOR_RE,
    SETTING_DENSITIES,
    SettingsStore,
    optional_frame,
    validate_annotation,
    validate_failure_event,
)


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTROL_PREFIX_REQUEST_RE = re.compile(
    rb"^(?P<prefix>[\x00-\x1f]{1,32})"
    rb"(?P<request>(?:GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH) [^\r\n]+ HTTP/1\.[01]\r?\n)$"
)
class LF3RApplication:
    coordinator_class = JobCoordinator
    baseline_service_class = BaselineService
    project_tool_service_class = None

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
        self.job_coordinator = self.coordinator_class()
        self.tmux = TmuxJobSupervisor(self.project_root, tmux_binary=tmux_binary)
        self.baselines = self.baseline_service_class(
            self.project_root,
            self.manifest_path,
            self.job_coordinator,
            self.tmux,
            annotation_store=self.store,
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
        if self.project_tool_service_class is not None:
            self.project_tools = self.project_tool_service_class(
                self.project_root,
                self.tmux,
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
    protocol_version = "HTTP/1.0"
    CLIENT_DISCONNECT_ERRORS = (
        BrokenPipeError,
        ConnectionResetError,
        ConnectionAbortedError,
    )

    def handle(self) -> None:
        try:
            super().handle()
        except self.CLIENT_DISCONNECT_ERRORS:
            # Browsers routinely cancel stale fetch/video requests during
            # navigation and manifest refresh. The response can no longer be
            # delivered, but this is not a server-side failure.
            self.close_connection = True

    def _safe_end_headers(self) -> bool:
        self.close_connection = True
        try:
            self.end_headers()
        except self.CLIENT_DISCONNECT_ERRORS:
            return False
        return True

    def _safe_write(self, data: bytes) -> bool:
        try:
            self.wfile.write(data)
        except self.CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True
            return False
        return True

    @staticmethod
    def _sanitize_raw_requestline(raw: bytes) -> tuple[bytes, int]:
        match = CONTROL_PREFIX_REQUEST_RE.fullmatch(raw)
        if match is None:
            return raw, 0
        prefix = match.group("prefix")
        return match.group("request"), len(prefix)

    def parse_request(self) -> bool:
        sanitized, stripped = self._sanitize_raw_requestline(self.raw_requestline)
        if stripped:
            self.raw_requestline = sanitized
            method = sanitized.split(b" ", 1)[0].decode("ascii", errors="replace")
            self.log_error(
                "sanitized %d leading control byte(s) before HTTP method %s",
                stripped,
                method,
            )
        return super().parse_request()

    def log_message(self, fmt: str, *args: Any) -> None:
        super().log_message(fmt, *args)

    def json_response(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        if not self._safe_end_headers():
            return
        self._safe_write(body)

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
        self.send_header("Connection", "close")
        if not self._safe_end_headers():
            return
        self._safe_write(body)

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
            if path == "/api/analysis/localization/presets":
                self.json_response(
                    HTTPStatus.OK,
                    {"presets": self.app.analysis_jobs.localization_presets()},
                )
                return
            if path == "/api/analysis/localization":
                self.json_response(
                    HTTPStatus.OK,
                    {"localization": self.app.analysis_jobs.localization_results()},
                )
                return
            if path == "/api/analysis/localization/challenge-sets":
                self.json_response(
                    HTTPStatus.OK,
                    {"challenge_sets": self.app.analysis_jobs.localization_challenge_sets()},
                )
                return
            if path.startswith("/api/analysis/localization/result/"):
                run_name = path.rsplit("/", 1)[-1]
                try:
                    result = self.app.analysis_jobs.localization_result_detail(run_name)
                except FileNotFoundError:
                    self.json_error(HTTPStatus.NOT_FOUND, "Localization result not found")
                    return
                self.json_response(HTTPStatus.OK, {"result": result})
                return
            if path.startswith("/api/analysis/localization/challenge/"):
                run_name = path.rsplit("/", 1)[-1]
                self.json_response(
                    HTTPStatus.OK,
                    {"challenge": self.app.analysis_jobs.localization_challenge_data(run_name)},
                )
                return
            if path.startswith("/api/analysis/localization/artifacts/"):
                relative = path[len("/api/analysis/localization/artifacts/"):]
                parts = relative.split("/", 1)
                if len(parts) != 2:
                    self.json_error(HTTPStatus.NOT_FOUND, "Localization artifact not found")
                    return
                try:
                    artifact = self.app.analysis_jobs.localization_artifact(parts[0], parts[1])
                except FileNotFoundError:
                    self.json_error(HTTPStatus.NOT_FOUND, "Localization artifact not found")
                    return
                self.file_response(artifact)
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
            if path == "/api/analysis/robo-localization-head":
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "localization_head": (
                            self.app.analysis.robo_localization_head_response()
                        )
                    },
                )
                return
            if path.startswith("/api/analysis/robo-localization-head/artifacts/"):
                name = path.rsplit("/", 1)[-1]
                try:
                    artifact = self.app.analysis.robo_localization_head_artifact_path(name)
                except FileNotFoundError:
                    self.json_error(
                        HTTPStatus.NOT_FOUND,
                        "Localization-head artifact is unavailable",
                    )
                    return
                self.file_response(artifact)
                return
            if path == "/api/analysis/robo-label-loss":
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "label_loss": (
                            self.app.analysis.robo_label_loss_response()
                        )
                    },
                )
                return
            if path.startswith("/api/analysis/robo-label-loss/artifacts/"):
                name = path.rsplit("/", 1)[-1]
                try:
                    artifact = self.app.analysis.robo_label_loss_artifact_path(name)
                except FileNotFoundError:
                    self.json_error(
                        HTTPStatus.NOT_FOUND,
                        "Label/loss ablation artifact is unavailable",
                    )
                    return
                self.file_response(artifact)
                return
            if path == "/api/analysis/robo-hop":
                self.json_response(
                    HTTPStatus.OK,
                    {"robo_hop": self.app.analysis.robo_hop_response()},
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
            if path == "/api/baselines/result-coverage":
                baseline = query.get("baseline", [""])[0]
                scope = query.get("scope", ["libero_10"])[0]
                condition = query.get("condition", ["full_instruction"])[0]
                self.json_response(
                    HTTPStatus.OK,
                    self.app.baselines.result_coverage(
                        baseline,
                        scope,
                        condition,
                    ),
                )
                return
            if path == "/api/baselines/runs":
                scope = query.get("scope", ["libero_10"])[0]
                condition = query.get("condition", ["full_instruction"])[0]
                rescan = str(query.get("rescan", ["0"])[0]).lower() in {
                    "1", "true", "yes"
                }
                index_refresh = (
                    self.app.baselines.rebuild_run_index()
                    if rescan
                    else None
                )
                payload = {
                    "scope": scope,
                    "condition": condition,
                    "runs": self.app.baselines.list_runs(scope, condition),
                }
                if index_refresh is not None:
                    payload["index_refresh"] = index_refresh
                self.json_response(HTTPStatus.OK, payload)
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
                camera = str(query.get("camera", [""])[0] or "").strip()
                if camera:
                    camera_paths = rollout.get("camera_video_paths")
                    value = camera_paths.get(camera) if isinstance(camera_paths, dict) else None
                    if not isinstance(value, str) or not value:
                        self.json_error(HTTPStatus.NOT_FOUND, "Camera video is unavailable")
                        return
                    video = self.app.resolve_project_file(value, ".mp4")
                    if not video.is_file():
                        self.json_error(HTTPStatus.NOT_FOUND, "Camera video file is unavailable")
                        return
                else:
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
        except subprocess.SubprocessError as exc:
            self.json_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Subprocess check failed: " + str(exc),
            )
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
            if path in {
                "/api/analysis/localization/run",
                "/api/analysis/localization/presets/save",
                "/api/analysis/localization/presets/delete",
                "/api/analysis/localization/challenge/save",
            }:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.json_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                    return
                if length <= 0 or length > 1_000_000:
                    self.json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size")
                    return
                payload = json.loads(self.rfile.read(length))
                if path == "/api/analysis/localization/run":
                    job = self.app.analysis_jobs.start_localization_spec_run(payload)
                    self.json_response(HTTPStatus.ACCEPTED, {"job": job})
                elif path == "/api/analysis/localization/presets/save":
                    preset = self.app.analysis_jobs.save_localization_preset(payload)
                    self.json_response(HTTPStatus.OK, {"preset": preset})
                elif path == "/api/analysis/localization/presets/delete":
                    result = self.app.analysis_jobs.delete_localization_preset(payload)
                    self.json_response(HTTPStatus.OK, result)
                else:
                    manifest = self.app.analysis_jobs.save_localization_challenge_set(payload)
                    self.json_response(HTTPStatus.OK, {"challenge_set": manifest})
                return
            if path.startswith("/api/baselines/posthoc-localization/"):
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
                if length <= 0 or length > 100_000:
                    self.json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size")
                    return
                payload = json.loads(self.rfile.read(length))
                result = self.app.baselines.run_posthoc_localization(
                    rollout,
                    payload,
                )
                self.json_response(HTTPStatus.OK, {"posthoc_localization": result})
                return
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
                    payload.get(
                        "instruction_condition",
                        payload.get("condition", "full_instruction"),
                    ),
                    options=payload.get("options"),
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
        self.send_header("Connection", "close")
        if not self._safe_end_headers():
            return
        self._safe_write(body)

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
        self.send_header("Connection", "close")
        if not self._safe_end_headers():
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                if not self._safe_write(chunk):
                    break
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
