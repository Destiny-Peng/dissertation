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
        self.robo_hop_root = (
            self.project_root / "outputs" / "robo_dopamine_incremental_hop"
        )
        self.robo_localization_head_root = (
            self.project_root / "outputs" / "robo_dopamine_localization_head"
        )
        self.robo_label_loss_root = (
            self.project_root / "outputs" / "robo_dopamine_label_loss_ablation"
        )
        self.localization_root = self.project_root / "outputs" / "robo_localization"
        self.localization_preset_root = (
            self.project_root / "config" / "robo_localization_presets"
        )
        self.localization_challenge_root = (
            self.project_root / "config" / "robo_localization_challenge_sets"
        )
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
        robo_python = os.environ.get("LF3R_ROBODOPAMINE_PYTHON")
        if robo_python:
            self.robo_python = Path(os.path.abspath(Path(robo_python).expanduser()))
        else:
            self.robo_python = (
                self.project_root / "conda_envs" / "LF3R-robo-dopamine" / "bin" / "python"
            )
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

    def _validate_robo_hop_run(
        self,
        raw_run: Any,
        selected_ids: set[str],
    ) -> tuple[Path, dict[str, Any], set[str]]:
        run_path, metadata = self.baselines._explicit_run_candidate(
            "robo_dopamine",
            raw_run,
            {"full_instruction", "unknown"},
        )
        if metadata.get("status") not in BASELINE_RUN_STATUSES:
            raise ValidationError(
                "Fused-hop failure analysis requires a completed Robo-Dopamine run"
            )
        run_ids, signal_ids_by_mode, _four_signal_ids = (
            self.baselines._run_inventory(run_path, "robo_dopamine")
        )
        overlap = selected_ids.intersection(run_ids)
        fused_ids = overlap.intersection(
            signal_ids_by_mode.get("fused", set())
        )
        if not fused_ids:
            raise ValidationError(
                "Selected Robo-Dopamine run has no rollout in this scope with a "
                "saved fused hop signal."
            )
        return run_path, metadata, fused_ids

    def start_robo_hop_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.require_environment()
        allowed_fields = {
            "analysis_kind", "scope", "runs", "task_cv", "output_label",
            "cpu_limit",
        }
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            raise ValidationError(
                "Unknown Robo-Dopamine hop-analysis field(s): "
                + ", ".join(sorted(unknown_fields))
            )
        scope = validate_run_scope(payload.get("scope"))
        records = select_scope_records(self._manifest_records(), scope)
        if not records:
            raise ValidationError(f"No rollouts matched scope {scope}")
        selected_ids = {record["id"] for record in records}
        raw_runs = payload.get("runs")
        if not isinstance(raw_runs, dict):
            raise ValidationError("runs must be an object")
        run_path, _metadata, available_ids = self._validate_robo_hop_run(
            raw_runs.get("robo_dopamine"),
            selected_ids,
        )
        available_records = [
            record for record in records if record["id"] in available_ids
        ]
        task_cv = payload.get("task_cv", False)
        if not isinstance(task_cv, bool):
            raise ValidationError("task_cv must be boolean")
        cpu_limit = self._integer(
            payload.get("cpu_limit", 4),
            "cpu_limit",
            1,
            16,
        )
        label = str(payload.get("output_label") or "web_robo_hop").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", label):
            raise ValidationError(
                "output_label must contain only letters, numbers, dot, underscore, or hyphen"
            )
        script = (
            self.project_root / "tools" / "analyze_robo_dopamine_incremental_hop.py"
        )
        if not script.is_file():
            raise ValidationError(
                "Robo-Dopamine hop-comparison analysis script is not installed"
            )

        job_id = "analysis-hop-" + uuid.uuid4().hex[:12]
        workspace = self.robo_hop_root / ".web_jobs" / job_id
        output_temp = workspace / "output"
        output_final = self.robo_hop_root / (
            "web_"
            + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            + "_"
            + label
            + "_"
            + job_id[-8:]
        )
        selection_path = workspace / "selection.json"
        selection_doc = {
            "schema_version": 1,
            "scope": scope,
            "requested_rollouts": len(records),
            "available_fused_rollouts": len(available_records),
            "selection": [{"id": record["id"]} for record in available_records],
        }
        command = [
            str(self.analysis_python),
            str(script),
            "--run-root",
            str(run_path),
            "--selection",
            str(selection_path),
            "--manifest",
            str(self.manifest_path),
            "--annotations",
            str(self.annotation_root / "records"),
            "--output-dir",
            str(output_temp),
            "--cpu-limit",
            str(cpu_limit),
        ]
        if task_cv:
            command.append("--task-cv")

        self.robo_hop_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        (self.robo_hop_root / ".web_jobs").mkdir(parents=True, exist_ok=True)
        self.coordinator.acquire(job_id, "analysis")
        try:
            workspace.mkdir(parents=True, exist_ok=False)
            atomic_json_write(selection_path, selection_doc)
            job = {
                "job_id": job_id,
                "job_type": "analysis",
                "analysis_kind": "robo_hop_comparison",
                "status": "queued",
                "scope": scope,
                "requested_rollouts": len(records),
                "selected_rollouts": len(available_records),
                "fused_coverage": (
                    len(available_records) / len(records) if records else 0.0
                ),
                "runs": {"robo_dopamine": self._relative(run_path)},
                "parameters": {
                    "task_cv": task_cv,
                    "cpu_limit": cpu_limit,
                    "nice_target": 10,
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
                environment={
                    "MPLBACKEND": "Agg",
                    "OMP_NUM_THREADS": str(cpu_limit),
                    "OMP_THREAD_LIMIT": str(cpu_limit),
                    "OPENBLAS_NUM_THREADS": str(cpu_limit),
                    "MKL_NUM_THREADS": str(cpu_limit),
                    "NUMEXPR_NUM_THREADS": str(cpu_limit),
                    "BLIS_NUM_THREADS": str(cpu_limit),
                    "VECLIB_MAXIMUM_THREADS": str(cpu_limit),
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

    def start_robo_localization_head_run(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        allowed_fields = {
            "analysis_kind", "runs", "output_label", "device", "repeats",
            "epochs", "patience", "learning_rate", "weight_decay", "grad_clip",
            "batch_size", "success_ratios",
        }
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            raise ValidationError(
                "Unknown localization-head field(s): " + ", ".join(sorted(unknown_fields))
            )
        if not self.robo_python.is_file() or not os.access(self.robo_python, os.X_OK):
            raise AnalysisEnvironmentError(
                "Robo-Dopamine PyTorch Python is unavailable: "
                + self._relative(self.robo_python)
            )
        # Do not synchronously import torch in the HTTP request path. On some
        # servers the first PyTorch/CUDA import is slow enough to outlive a
        # frontend request or subprocess timeout. The tmux training process is
        # the authoritative environment check; any import/CUDA error is kept in
        # its persistent log instead of dropping the POST connection.

        # BiLSTM evaluation now treats all completed Robo-Dopamine runs as a
        # result pool. The training process selects the latest usable fused
        # result independently for each rollout, so a newly rerun partial batch
        # is automatically merged with older rollout results.
        pool_root = self.baselines.baseline_root
        if not pool_root.is_dir():
            raise ValidationError("Baseline output pool does not exist")

        label = str(payload.get("output_label") or "web_bilstm_success_ablation").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", label):
            raise ValidationError(
                "output_label must contain only letters, numbers, dot, underscore, or hyphen"
            )
        device = str(payload.get("device") or "auto").strip()
        if not re.fullmatch(r"(auto|cpu|cuda(?::\d+)?)", device):
            raise ValidationError("device must be auto, cpu, cuda, or cuda:N")
        repeats = self._integer(payload.get("repeats", 5), "repeats", 1, 50)
        epochs = self._integer(payload.get("epochs", 300), "epochs", 1, 5000)
        patience = self._integer(payload.get("patience", 35), "patience", 1, 1000)
        batch_size = self._integer(payload.get("batch_size", 32), "batch_size", 1, 128)

        raw_success_ratios = payload.get("success_ratios", "0.5,1,2")
        if isinstance(raw_success_ratios, str):
            ratio_parts = [
                part.strip()
                for part in raw_success_ratios.split(",")
                if part.strip()
            ]
        elif isinstance(raw_success_ratios, list):
            ratio_parts = list(raw_success_ratios)
        else:
            raise ValidationError(
                "success_ratios must be a comma-separated string or array"
            )
        if not ratio_parts:
            raise ValidationError("success_ratios must contain at least one value")
        success_ratios: list[float] = []
        seen_ratios: set[float] = set()
        for raw_ratio in ratio_parts:
            try:
                ratio = float(raw_ratio)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "success_ratios must contain only numeric values"
                ) from exc
            if not math.isfinite(ratio) or ratio <= 0 or ratio > 20:
                raise ValidationError(
                    "each success ratio must be finite, > 0, and <= 20"
                )
            if ratio not in seen_ratios:
                seen_ratios.add(ratio)
                success_ratios.append(ratio)
        if len(success_ratios) > 16:
            raise ValidationError("success_ratios accepts at most 16 unique values")
        success_ratios.sort()
        success_ratios_arg = ",".join(f"{ratio:g}" for ratio in success_ratios)
        try:
            learning_rate = float(payload.get("learning_rate", 0.003))
            weight_decay = float(payload.get("weight_decay", 1e-4))
            grad_clip = float(payload.get("grad_clip", 5.0))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "learning_rate, weight_decay, and grad_clip must be numeric"
            ) from exc
        if learning_rate <= 0 or weight_decay < 0 or grad_clip <= 0:
            raise ValidationError(
                "learning_rate and grad_clip must be > 0; weight_decay must be >= 0"
            )

        script = self.project_root / "tools" / "train_robo_localization.py"
        if not script.is_file():
            raise ValidationError("Unified BiLSTM localization trainer is missing")

        job_id = "analysis-bilstm-" + uuid.uuid4().hex[:12]
        workspace = self.robo_localization_head_root / ".web_jobs" / job_id
        output_temp = workspace / "output"
        output_final = self.robo_localization_head_root / (
            "web_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            + "_" + label + "_" + job_id[-8:]
        )
        command = [
            str(self.robo_python),
            str(script),
            "--experiment", "success_negative",
            "--run-pool-root", str(pool_root),
            "--manifest", str(self.manifest_path),
            "--annotations", str(self.annotation_root / "records"),
            "--output-dir", str(output_temp),
            "--device", device,
            "--repeats", str(repeats),
            "--epochs", str(epochs),
            "--patience", str(patience),
            "--learning-rate", str(learning_rate),
            "--weight-decay", str(weight_decay),
            "--grad-clip", str(grad_clip),
            "--batch-size", str(batch_size),
            "--success-ratios", success_ratios_arg,
        ]
        self.robo_localization_head_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        (self.robo_localization_head_root / ".web_jobs").mkdir(
            parents=True, exist_ok=True
        )
        self.coordinator.acquire(job_id, "analysis")
        try:
            workspace.mkdir(parents=True, exist_ok=False)
            job = {
                "job_id": job_id,
                "job_type": "analysis",
                "analysis_kind": "robo_bilstm_success_ablation",
                "status": "queued",
                "selected_rollouts": 0,
                "runs": {
                    "robo_dopamine_pool": self._relative(pool_root)
                },
                "parameters": {
                    "device": device,
                    "repeats": repeats,
                    "epochs": epochs,
                    "patience": patience,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "grad_clip": grad_clip,
                    "batch_size": batch_size,
                    "success_ratios": success_ratios,
                    "result_selection": "latest_usable_fused_per_rollout",
                    "run_pool_root": self._relative(pool_root),
                },
                "command": command,
                "output_dir": self._relative(output_final),
                "output_temp": self._relative(output_temp),
                "log_path": self._relative(self.log_root / f"{job_id}.log"),
                "started_at": None,
                "finished_at": None,
                "return_code": None,
                "error": None,
                "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "interpreter": str(self.robo_python),
            }
            with self.jobs_lock:
                self.jobs[job_id] = job
            self.tmux.submit(
                job,
                command,
                self.log_root / f"{job_id}.log",
                interpreter=str(self.robo_python),
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

    def start_robo_label_loss_run(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        allowed_fields = {
            "analysis_kind", "output_label", "device", "repeats", "epochs",
            "patience", "learning_rate", "weight_decay", "grad_clip", "batch_size",
            "tau_event", "distance_weight", "ranking_weight", "ranking_margin",
            "run_asymmetric_if_soft_improves",
        }
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            raise ValidationError(
                "Unknown label/loss ablation field(s): "
                + ", ".join(sorted(unknown_fields))
            )
        if not self.robo_python.is_file() or not os.access(self.robo_python, os.X_OK):
            raise AnalysisEnvironmentError(
                "Robo-Dopamine PyTorch Python is unavailable: "
                + self._relative(self.robo_python)
            )
        pool_root = self.baselines.baseline_root
        if not pool_root.is_dir():
            raise ValidationError("Baseline output pool does not exist")

        label = str(
            payload.get("output_label") or "web_bilstm_label_loss"
        ).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", label):
            raise ValidationError(
                "output_label must contain only letters, numbers, dot, underscore, or hyphen"
            )
        device = str(payload.get("device") or "auto").strip()
        if not re.fullmatch(r"(auto|cpu|cuda(?::\d+)?)", device):
            raise ValidationError("device must be auto, cpu, cuda, or cuda:N")
        repeats = self._integer(payload.get("repeats", 5), "repeats", 1, 50)
        epochs = self._integer(payload.get("epochs", 300), "epochs", 1, 5000)
        patience = self._integer(payload.get("patience", 35), "patience", 1, 1000)
        batch_size = self._integer(payload.get("batch_size", 32), "batch_size", 1, 128)
        try:
            learning_rate = float(payload.get("learning_rate", 0.003))
            weight_decay = float(payload.get("weight_decay", 1e-4))
            grad_clip = float(payload.get("grad_clip", 5.0))
            tau_event = float(payload.get("tau_event", 20.0))
            distance_weight = float(payload.get("distance_weight", 1.0))
            ranking_weight = float(payload.get("ranking_weight", 1.0))
            ranking_margin = float(payload.get("ranking_margin", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Label/loss numeric parameters are invalid") from exc
        numeric = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "tau_event": tau_event,
            "distance_weight": distance_weight,
            "ranking_weight": ranking_weight,
            "ranking_margin": ranking_margin,
        }
        if any(not math.isfinite(value) for value in numeric.values()):
            raise ValidationError("Label/loss numeric parameters must be finite")
        if learning_rate <= 0 or grad_clip <= 0 or tau_event <= 0:
            raise ValidationError(
                "learning_rate, grad_clip, and tau_event must be > 0"
            )
        if weight_decay < 0 or distance_weight < 0 or ranking_weight < 0 or ranking_margin < 0:
            raise ValidationError(
                "weight_decay and loss weights/margin must be >= 0"
            )
        run_asymmetric = payload.get("run_asymmetric_if_soft_improves", True)
        if not isinstance(run_asymmetric, bool):
            raise ValidationError(
                "run_asymmetric_if_soft_improves must be boolean"
            )

        script = self.project_root / "tools" / "train_robo_localization.py"
        if not script.is_file():
            raise ValidationError("Unified BiLSTM localization trainer is missing")

        job_id = "analysis-label-loss-" + uuid.uuid4().hex[:12]
        workspace = self.robo_label_loss_root / ".web_jobs" / job_id
        output_temp = workspace / "output"
        output_final = self.robo_label_loss_root / (
            "web_"
            + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            + "_"
            + label
            + "_"
            + job_id[-8:]
        )
        command = [
            str(self.robo_python),
            str(script),
            "--experiment", "label_loss",
            "--run-pool-root", str(pool_root),
            "--manifest", str(self.manifest_path),
            "--annotations", str(self.annotation_root / "records"),
            "--output-dir", str(output_temp),
            "--device", device,
            "--repeats", str(repeats),
            "--epochs", str(epochs),
            "--patience", str(patience),
            "--learning-rate", str(learning_rate),
            "--weight-decay", str(weight_decay),
            "--grad-clip", str(grad_clip),
            "--batch-size", str(batch_size),
            "--tau-event", str(tau_event),
            "--distance-weight", str(distance_weight),
            "--ranking-weight", str(ranking_weight),
            "--ranking-margin", str(ranking_margin),
        ]
        command.append(
            "--run-asymmetric-if-soft-improves"
            if run_asymmetric
            else "--no-run-asymmetric-if-soft-improves"
        )

        self.robo_label_loss_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        (self.robo_label_loss_root / ".web_jobs").mkdir(
            parents=True, exist_ok=True
        )
        self.coordinator.acquire(job_id, "analysis")
        try:
            workspace.mkdir(parents=True, exist_ok=False)
            job = {
                "job_id": job_id,
                "job_type": "analysis",
                "analysis_kind": "robo_bilstm_label_loss_ablation",
                "status": "queued",
                "selected_rollouts": 0,
                "runs": {
                    "robo_dopamine_pool": self._relative(pool_root)
                },
                "parameters": {
                    "device": device,
                    "repeats": repeats,
                    "epochs": epochs,
                    "patience": patience,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "grad_clip": grad_clip,
                    "batch_size": batch_size,
                    "tau_event": tau_event,
                    "distance_weight": distance_weight,
                    "ranking_weight": ranking_weight,
                    "ranking_margin": ranking_margin,
                    "run_asymmetric_if_soft_improves": run_asymmetric,
                    "result_selection": "latest_usable_fused_per_rollout",
                    "run_pool_root": self._relative(pool_root),
                    "model": "tiny_bilstm_h16",
                    "training_population": "failure_only",
                },
                "command": command,
                "output_dir": self._relative(output_final),
                "output_temp": self._relative(output_temp),
                "log_path": self._relative(self.log_root / f"{job_id}.log"),
                "started_at": None,
                "finished_at": None,
                "return_code": None,
                "error": None,
                "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "interpreter": str(self.robo_python),
            }
            with self.jobs_lock:
                self.jobs[job_id] = job
            self.tmux.submit(
                job,
                command,
                self.log_root / f"{job_id}.log",
                interpreter=str(self.robo_python),
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

    @staticmethod
    def _localization_builtin_presets() -> dict[str, dict[str, Any]]:
        base = {
            "data": {
                "population": "failure_only", "success_ratio": 0.0,
                "challenge_set_name": "", "force_train_rollout_ids": [],
            },
            "target": {"kind": "hard", "sigma_pre": 3.0, "sigma_post": 3.0, "tau_event": 20.0},
            "model": {"hidden": 16},
            "loss": {"name": "bce", "distance_weight": 1.0, "ranking_weight": 1.0, "ranking_margin": 1.0},
            "training": {
                "device": "auto", "batch_size": 32, "parallel_workers": 4, "epochs": 300, "patience": 35,
                "learning_rate": 0.003, "weight_decay": 0.0001, "grad_clip": 5.0,
                "seed": 17, "train_fraction": 0.70, "val_fraction": 0.15,
            },
        }
        return {
            "bilstm_default": {
                "schema_version": 1, "name": "bilstm_default", "base": base,
                "sweep": [], "variants": [], "stages": [], "repeats": 5, "builtin": True,
            },
            "label_loss_default": {
                "schema_version": 1, "name": "label_loss_default", "base": base,
                "sweep": [], "repeats": 5, "builtin": True,
                "stages": [
                    {
                        "name": "label_selection",
                        "sweep": [],
                        "variants": [
                            {"name": "hard", "set": {
                                "target.kind": "hard", "target.sigma_pre": 3.0,
                                "target.sigma_post": 3.0, "target.tau_event": 20.0,
                            }},
                            {"name": "gaussian_sigma_1", "set": {
                                "target.kind": "gaussian", "target.sigma_pre": 1.0,
                                "target.sigma_post": 1.0, "target.tau_event": 20.0,
                            }},
                            {"name": "gaussian_sigma_2", "set": {
                                "target.kind": "gaussian", "target.sigma_pre": 2.0,
                                "target.sigma_post": 2.0, "target.tau_event": 20.0,
                            }},
                            {"name": "gaussian_sigma_3", "set": {
                                "target.kind": "gaussian", "target.sigma_pre": 3.0,
                                "target.sigma_post": 3.0, "target.tau_event": 20.0,
                            }},
                            {"name": "gaussian_sigma_5", "set": {
                                "target.kind": "gaussian", "target.sigma_pre": 5.0,
                                "target.sigma_post": 5.0, "target.tau_event": 20.0,
                            }},
                        ],
                        "select": {"metric": "in_interval_rate_mean", "mode": "max", "tie_breakers": [
                            {"metric": "mae_samples_mean", "mode": "min"},
                            {"metric": "mse_samples_mean", "mode": "min"},
                        ]},
                    },
                    {
                        "name": "loss_selection",
                        "variants": [],
                        "sweep": [{"path": "loss.name", "values": [
                            "bce", "temporal_softmax_ce", "temporal_softmax_ce_distance",
                            "temporal_softmax_ce_squared_distance", "temporal_softmax_ce_ranking",
                            "temporal_softmax_ce_distance_ranking",
                        ]}],
                        "select": {"metric": "in_interval_rate_mean", "mode": "max", "tie_breakers": [
                            {"metric": "mae_samples_mean", "mode": "min"},
                            {"metric": "mse_samples_mean", "mode": "min"},
                        ]},
                    },
                ],
            },
            "success_ratio_default": {
                "schema_version": 1, "name": "success_ratio_default",
                "base": {**base, "data": {"population": "failure_success", "success_ratio": 0.0}},
                "sweep": [
                    {"path": "data.success_ratio", "values": [0.0, 0.5, 1.0, 2.0]},
                    {"path": "model.hidden", "values": [16, 32]},
                ],
                "variants": [],
                "stages": [], "repeats": 5, "builtin": True,
            },
        }

    def localization_presets(self) -> list[dict[str, Any]]:
        presets = self._localization_builtin_presets()
        self.localization_preset_root.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.localization_preset_root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payload = dict(payload)
                payload["builtin"] = False
                presets[path.stem] = payload
        return [presets[name] for name in sorted(presets)]

    def save_localization_preset(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("spec"), dict):
            raise ValidationError("Preset request requires a spec object")
        name = str(payload["spec"].get("name") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
            raise ValidationError("Invalid preset name")
        if name in self._localization_builtin_presets():
            raise ValidationError("Built-in presets cannot be overwritten")
        self.localization_preset_root.mkdir(parents=True, exist_ok=True)
        path = self.localization_preset_root / f"{name}.json"
        overwrite = bool(payload.get("overwrite", False))
        if path.exists() and not overwrite:
            raise ValidationError("Preset already exists; set overwrite=true to replace it")
        spec = dict(payload["spec"])
        spec["schema_version"] = 1
        path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        return {**spec, "builtin": False}

    def delete_localization_preset(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValidationError("Preset delete request must be an object")
        name = str(payload.get("name") or "").strip()
        if name in self._localization_builtin_presets():
            raise ValidationError("Built-in presets cannot be deleted")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
            raise ValidationError("Invalid preset name")
        path = self.localization_preset_root / f"{name}.json"
        if not path.is_file():
            raise ValidationError("Preset does not exist")
        path.unlink()
        return {"name": name, "deleted": True}

    def localization_results(self) -> dict[str, Any]:
        if not self.localization_root.is_dir():
            return {"available": False, "runs": []}
        rows = []
        for metadata_path in self.localization_root.glob("*/metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("analysis") != "robo_localization_experiment":
                continue
            directory = metadata_path.parent
            rows.append({
                "run_id": directory.name,
                "directory": self._relative(directory),
                "name": metadata.get("name"),
                "generated_at": metadata.get("generated_at"),
                "configuration_count": metadata.get("configuration_count"),
                "training_run_count": metadata.get("training_run_count"),
                "checkpoint_count": metadata.get("checkpoint_count"),
                "challenge_available": (directory / "all_failure_predictions.csv").is_file(),
                "summary_url": "/api/analysis/localization/artifacts/" + directory.name + "/summary.csv",
                "manifest_url": "/api/analysis/localization/artifacts/" + directory.name + "/experiment_manifest.json",
                "all_failure_url": (
                    "/api/analysis/localization/artifacts/" + directory.name
                    + "/all_failure_predictions.csv"
                    if (directory / "all_failure_predictions.csv").is_file()
                    else None
                ),
            })
        rows.sort(key=lambda row: str(row.get("generated_at") or ""), reverse=True)
        return {"available": bool(rows), "runs": rows[:50]}

    def _localization_repeat_metrics(
        self,
        directory: Path,
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Recover every repeat's observed test metrics for each config.

        The primary source is per_rollout_predictions.csv so this also works for
        runs created before explicit per-repeat summaries were persisted.
        """
        predictions_path = directory / "per_rollout_predictions.csv"
        if not predictions_path.is_file():
            return {}

        record_meta: dict[tuple[str, str, int], dict[str, Any]] = {}
        records_path = directory / "training_records.json"
        if records_path.is_file():
            try:
                loaded = json.loads(records_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = []
            if isinstance(loaded, list):
                for record in loaded:
                    if not isinstance(record, dict):
                        continue
                    stage = str(record.get("stage") or "main")
                    config_id = str(record.get("config_id") or "")
                    try:
                        repeat = int(record.get("repeat"))
                    except (TypeError, ValueError):
                        continue
                    if config_id:
                        record_meta[(stage, config_id, repeat)] = record

        grouped: dict[tuple[str, str, int], dict[str, Any]] = {}
        try:
            with predictions_path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    stage = str(row.get("stage") or "main")
                    config_id = str(row.get("config_id") or "")
                    if not config_id:
                        continue
                    try:
                        repeat = int(float(str(row.get("repeat") or "")))
                        error = int(float(str(row.get("interval_error_samples") or "")))
                    except (TypeError, ValueError):
                        continue
                    key = (stage, config_id, repeat)
                    bucket = grouped.setdefault(
                        key,
                        {
                            "stage": stage,
                            "config_id": config_id,
                            "repeat": repeat,
                            "errors": [],
                            "first_event_hits": [],
                            "checkpoint": None,
                        },
                    )
                    bucket["errors"].append(error)
                    first_event = str(
                        row.get("first_event_in_interval") or ""
                    ).strip().lower()
                    if first_event in {"true", "1", "yes"}:
                        bucket["first_event_hits"].append(True)
                    elif first_event in {"false", "0", "no"}:
                        bucket["first_event_hits"].append(False)
                    checkpoint = str(row.get("checkpoint") or "").strip()
                    if checkpoint and not bucket["checkpoint"]:
                        bucket["checkpoint"] = checkpoint
        except OSError:
            return {}

        by_config: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for key, bucket in grouped.items():
            stage, config_id, repeat = key
            errors = list(bucket["errors"])
            if not errors:
                continue
            absolute = [abs(value) for value in errors]
            count = len(errors)
            first_hits = list(bucket["first_event_hits"])
            meta = record_meta.get(key, {})
            split = meta.get("split") if isinstance(meta.get("split"), dict) else {}
            summary = {
                "repeat": repeat,
                "seed": meta.get("seed"),
                "checkpoint": bucket["checkpoint"] or meta.get("checkpoint"),
                "test_n": count,
                "train_n": len(split.get("train") or []),
                "val_n": len(split.get("val") or []),
                "in_interval_rate": sum(value == 0 for value in errors) / count,
                "first_event_in_interval_rate": (
                    sum(first_hits) / len(first_hits) if first_hits else None
                ),
                "within_1": sum(value <= 1 for value in absolute) / count,
                "within_3": sum(value <= 3 for value in absolute) / count,
                "within_5": sum(value <= 5 for value in absolute) / count,
                "median_absolute_interval_error_samples": float(
                    statistics.median(absolute)
                ),
                "mae_samples": float(sum(absolute) / count),
                "mse_samples": float(
                    sum(value * value for value in errors) / count
                ),
                "before_interval_rate": sum(value < 0 for value in errors) / count,
                "after_interval_rate": sum(value > 0 for value in errors) / count,
                "best_epoch": meta.get("best_epoch"),
                "best_val_loss": meta.get("best_val_loss"),
                "effective_train_batch_size": meta.get(
                    "effective_train_batch_size"
                ),
                "optimizer_steps_per_epoch": meta.get(
                    "optimizer_steps_per_epoch"
                ),
                "derived_from": "per_rollout_predictions.csv",
            }
            by_config.setdefault((stage, config_id), []).append(summary)

        for repeats in by_config.values():
            repeats.sort(key=lambda row: int(row["repeat"]))
        return by_config

    def _localization_best_repeats(
        self,
        directory: Path,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        repeats_by_config = self._localization_repeat_metrics(directory)
        best: dict[tuple[str, str], dict[str, Any]] = {}
        for key, repeats in repeats_by_config.items():
            if not repeats:
                continue
            best[key] = min(
                repeats,
                key=lambda row: (
                    -float(row["in_interval_rate"]),
                    float(row["mae_samples"]),
                    float(row["mse_samples"]),
                    int(row["repeat"]),
                ),
            )
            best[key] = {
                **best[key],
                "selection": "test_in_interval_desc_mae_mse_asc",
            }
        return best

    def localization_result_detail(self, run_name: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
            raise ValidationError("Invalid localization run id")
        directory = self.localization_root / run_name
        metadata_path = directory / "metadata.json"
        summary_path = directory / "summary.csv"
        manifest_path = directory / "experiment_manifest.json"
        if not metadata_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(run_name)

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError("Localization metadata is unreadable") from exc

        manifest: dict[str, Any] = {"stages": []}
        if manifest_path.is_file():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest = loaded
            except (OSError, json.JSONDecodeError):
                pass

        repeat_metrics = self._localization_repeat_metrics(directory)
        best_repeats = self._localization_best_repeats(directory)

        best_by_stage: dict[str, str] = {}
        stage_meta: dict[str, dict[str, Any]] = {}
        for raw_stage in manifest.get("stages") or []:
            if not isinstance(raw_stage, dict):
                continue
            stage_name = str(raw_stage.get("stage") or "")
            if not stage_name:
                continue
            best_id = str(raw_stage.get("best_config_id") or "")
            if best_id:
                best_by_stage[stage_name] = best_id
            stage_meta[stage_name] = {
                "stage": stage_name,
                "best_config_id": best_id or None,
                "selector": raw_stage.get("selector") or {},
                "configuration_count": raw_stage.get("configuration_count"),
                "parallel_workers": raw_stage.get("parallel_workers"),
            }

        numeric_fields = {
            "repeat_n",
            "in_interval_rate_mean", "in_interval_rate_variance",
            "first_event_in_interval_rate_mean", "first_event_in_interval_rate_variance",
            "within_1_mean", "within_1_variance",
            "within_3_mean", "within_3_variance",
            "within_5_mean", "within_5_variance",
            "before_interval_rate_mean", "before_interval_rate_variance",
            "after_interval_rate_mean", "after_interval_rate_variance",
            "median_absolute_interval_error_samples_mean",
            "median_absolute_interval_error_samples_variance",
            "mae_samples_mean", "mae_samples_variance",
            "mse_samples_mean", "mse_samples_variance",
            "target.sigma_pre", "target.sigma_post", "target.tau_event",
            "model.hidden", "loss.distance_weight", "loss.ranking_weight",
            "loss.ranking_margin", "training.batch_size",
            "training.parallel_workers", "training.learning_rate",
            "training.weight_decay", "training.grad_clip", "training.seed",
            "training.epochs", "training.patience",
            "best_repeat", "best_repeat_seed", "best_repeat_test_n",
            "best_repeat_in_interval_rate",
            "best_repeat_first_event_in_interval_rate",
            "best_repeat_within_3",
            "best_repeat_median_absolute_interval_error_samples",
            "best_repeat_mae_samples", "best_repeat_mse_samples",
            "best_repeat_before_interval_rate", "best_repeat_after_interval_rate",
        }

        rows: list[dict[str, Any]] = []
        with summary_path.open("r", newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, Any] = dict(raw)
                for field in numeric_fields:
                    value = row.get(field)
                    if value in (None, ""):
                        row[field] = None
                        continue
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    row[field] = int(number) if number.is_integer() else number

                stage = str(row.get("stage") or "main")
                config_id = str(row.get("config_id") or "")
                target_kind = str(row.get("target.kind") or "")
                loss_name = str(row.get("loss.name") or "")
                hidden = row.get("model.hidden")
                sigma_pre = row.get("target.sigma_pre")
                sigma_post = row.get("target.sigma_post")
                label_parts = [config_id]
                if target_kind:
                    target_label = target_kind
                    if target_kind == "gaussian" and sigma_pre is not None and sigma_post is not None:
                        target_label += f" σ={sigma_pre}/{sigma_post}"
                    label_parts.append(target_label)
                if loss_name:
                    label_parts.append(loss_name)
                if hidden is not None:
                    label_parts.append(f"h{hidden}")
                row["label"] = " · ".join(part for part in label_parts if part)
                row["best"] = best_by_stage.get(stage) == config_id

                stored_repeat = row.get("best_repeat")
                if stored_repeat is not None:
                    row["best_repeat"] = {
                        "repeat": int(stored_repeat),
                        "seed": row.get("best_repeat_seed"),
                        "checkpoint": row.get("best_repeat_checkpoint") or None,
                        "test_n": row.get("best_repeat_test_n"),
                        "in_interval_rate": row.get("best_repeat_in_interval_rate"),
                        "first_event_in_interval_rate": row.get(
                            "best_repeat_first_event_in_interval_rate"
                        ),
                        "within_3": row.get("best_repeat_within_3"),
                        "median_absolute_interval_error_samples": row.get(
                            "best_repeat_median_absolute_interval_error_samples"
                        ),
                        "mae_samples": row.get("best_repeat_mae_samples"),
                        "mse_samples": row.get("best_repeat_mse_samples"),
                        "before_interval_rate": row.get(
                            "best_repeat_before_interval_rate"
                        ),
                        "after_interval_rate": row.get(
                            "best_repeat_after_interval_rate"
                        ),
                        "selection": (
                            row.get("best_repeat_selection")
                            or "test_in_interval_desc_mae_mse_asc"
                        ),
                        "derived_from": "summary.csv",
                    }
                else:
                    row["best_repeat"] = best_repeats.get((stage, config_id))
                row["repeats"] = repeat_metrics.get((stage, config_id), [])
                base_seed = row.get("training.seed")
                if base_seed is not None:
                    for repeat_row in row["repeats"]:
                        if repeat_row.get("seed") is None:
                            repeat_row["seed"] = int(base_seed) + int(repeat_row["repeat"])
                rows.append(row)

        stage_order: list[str] = []
        for raw_stage in manifest.get("stages") or []:
            if isinstance(raw_stage, dict):
                stage_name = str(raw_stage.get("stage") or "")
                if stage_name and stage_name not in stage_order:
                    stage_order.append(stage_name)
        for row in rows:
            stage_name = str(row.get("stage") or "main")
            if stage_name not in stage_order:
                stage_order.append(stage_name)

        stages = []
        for stage_name in stage_order:
            stage_rows = [row for row in rows if str(row.get("stage") or "main") == stage_name]
            meta = stage_meta.get(stage_name, {
                "stage": stage_name,
                "best_config_id": best_by_stage.get(stage_name),
                "selector": {},
                "configuration_count": len(stage_rows),
                "parallel_workers": None,
            })
            stages.append({**meta, "rows": stage_rows})

        return {
            "run_id": run_name,
            "name": metadata.get("name"),
            "generated_at": metadata.get("generated_at"),
            "configuration_count": metadata.get("configuration_count"),
            "training_run_count": metadata.get("training_run_count"),
            "checkpoint_count": metadata.get("checkpoint_count"),
            "all_failure_prediction_count": metadata.get("all_failure_prediction_count"),
            "challenge_available": (directory / "all_failure_predictions.csv").is_file(),
            "stages": stages,
        }

    def localization_challenge_sets(self) -> list[dict[str, Any]]:
        self.localization_challenge_root.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for path in sorted(self.localization_challenge_root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            rows.append({
                "name": str(payload.get("name") or path.stem),
                "source_run": payload.get("source_run"),
                "generated_at": payload.get("generated_at"),
                "criterion": payload.get("criterion"),
                "size": len(payload.get("rollout_ids") or []),
                "rollout_ids": list(payload.get("rollout_ids") or []),
                "config_ids": list(payload.get("config_ids") or []),
                "composition": payload.get("composition") or {},
            })
        rows.sort(key=lambda row: str(row.get("generated_at") or ""), reverse=True)
        return rows

    def _localization_annotation_meta(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        root = self.annotation_root / "records"
        if not root.is_dir():
            return result
        for path in root.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            rollout_id = str(payload.get("rollout_id") or "")
            if not rollout_id:
                continue
            events = payload.get("failure_events")
            if not isinstance(events, list):
                events = []
            failure_types = []
            for event in events:
                if not isinstance(event, dict):
                    continue
                failure_type = str(event.get("failure_type") or "").strip()
                if failure_type and failure_type not in failure_types:
                    failure_types.append(failure_type)
            primary = str(payload.get("failure_type") or "").strip()
            if primary and primary not in failure_types:
                failure_types.insert(0, primary)
            recovery_like = payload.get("recovery_frame") is not None or any(
                isinstance(event, dict) and event.get("recovery_frame") is not None
                for event in events
            )
            result[rollout_id] = {
                "primary_failure_type": primary or (failure_types[0] if failure_types else "unknown"),
                "failure_types": failure_types or ["unknown"],
                "annotated_event_count": len(events),
                "recovery_like": bool(recovery_like),
                "outcome_label": payload.get("outcome_label"),
            }
        return result

    def localization_challenge_data(self, run_name: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
            raise ValidationError("Invalid localization run id")
        directory = self.localization_root / run_name
        predictions_path = directory / "all_failure_predictions.csv"
        if not predictions_path.is_file():
            return {
                "available": False,
                "run_id": run_name,
                "reason": (
                    "This run predates all-failure inference/checkpoint saving. "
                    "Run the experiment again to build a challenge set."
                ),
                "configs": [],
                "rows": [],
            }

        best_ids: set[str] = set()
        default_config_id: str | None = None
        manifest_path = directory / "experiment_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                stages = manifest.get("stages") if isinstance(manifest, dict) else []
                if isinstance(stages, list):
                    for stage in stages:
                        if isinstance(stage, dict) and stage.get("best_config_id"):
                            best_ids.add(str(stage["best_config_id"]))
                    if stages and isinstance(stages[-1], dict):
                        value = stages[-1].get("best_config_id")
                        default_config_id = str(value) if value else None
            except (OSError, json.JSONDecodeError):
                pass

        configs: list[dict[str, Any]] = []
        summary_path = directory / "summary.csv"
        if summary_path.is_file():
            with summary_path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    config_id = str(row.get("config_id") or "")
                    if not config_id:
                        continue
                    target_kind = str(row.get("target.kind") or "")
                    sigma_pre = str(row.get("target.sigma_pre") or "")
                    sigma_post = str(row.get("target.sigma_post") or "")
                    loss_name = str(row.get("loss.name") or "")
                    hidden = str(row.get("model.hidden") or "")
                    parts = [str(row.get("stage") or config_id), config_id]
                    if target_kind:
                        target_label = target_kind
                        if target_kind == "gaussian" and sigma_pre and sigma_post:
                            target_label += f" σ={sigma_pre}/{sigma_post}"
                        parts.append(target_label)
                    if loss_name:
                        parts.append(loss_name)
                    if hidden:
                        parts.append("h" + hidden)
                    configs.append({
                        "config_id": config_id,
                        "stage": row.get("stage"),
                        "label": " · ".join(parts),
                        "best": config_id in best_ids,
                    })

        annotations = self._localization_annotation_meta()
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        with predictions_path.open("r", newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                config_id = str(raw.get("config_id") or "")
                rollout_id = str(raw.get("rollout_id") or "")
                if not config_id or not rollout_id:
                    continue
                grouped.setdefault((config_id, rollout_id), []).append(raw)

        rows: list[dict[str, Any]] = []
        repeat_counts: list[int] = []
        for (config_id, rollout_id), samples in grouped.items():
            errors = [abs(int(float(sample.get("interval_error_samples") or 0))) for sample in samples]
            in_interval = [1 if int(float(sample.get("interval_error_samples") or 0)) == 0 else 0 for sample in samples]
            within_3 = [1 if abs(int(float(sample.get("interval_error_samples") or 0))) <= 3 else 0 for sample in samples]
            train_samples = [
                sample for sample in samples
                if str(sample.get("seen_in_train") or "").lower() in {"1", "true", "yes"}
            ]
            forced_train_samples = [
                sample for sample in samples
                if str(sample.get("forced_into_train") or "").lower() in {"1", "true", "yes"}
            ]
            train_errors = [
                abs(int(float(sample.get("interval_error_samples") or 0)))
                for sample in train_samples
            ]
            train_in_interval = [
                1 if int(float(sample.get("interval_error_samples") or 0)) == 0 else 0
                for sample in train_samples
            ]
            train_within_3 = [
                1 if abs(int(float(sample.get("interval_error_samples") or 0))) <= 3 else 0
                for sample in train_samples
            ]
            first = samples[0]
            meta = annotations.get(rollout_id, {})
            repeat_count = len(samples)
            repeat_counts.append(repeat_count)
            rows.append({
                "config_id": config_id,
                "stage": first.get("stage"),
                "rollout_id": rollout_id,
                "repeat_count": repeat_count,
                "in_interval_success_rate": sum(in_interval) / repeat_count,
                "within_3_success_rate": sum(within_3) / repeat_count,
                "median_absolute_interval_error": float(statistics.median(errors)),
                "mean_absolute_interval_error": float(sum(errors) / repeat_count),
                "worst_absolute_interval_error": int(max(errors)),
                "train_repeat_count": len(train_samples),
                "train_exposure_rate": len(train_samples) / repeat_count,
                "forced_train_repeat_count": len(forced_train_samples),
                "forced_train_rate": len(forced_train_samples) / repeat_count,
                "train_in_interval_success_rate": (
                    sum(train_in_interval) / len(train_samples) if train_samples else None
                ),
                "train_within_3_success_rate": (
                    sum(train_within_3) / len(train_samples) if train_samples else None
                ),
                "train_mean_absolute_interval_error": (
                    float(sum(train_errors) / len(train_errors)) if train_errors else None
                ),
                "task_key": first.get("task_key"),
                "task_id": first.get("task_id"),
                "event_count": int(float(first.get("event_count") or 0)),
                "multi_event": int(float(first.get("event_count") or 0)) > 1,
                "primary_failure_type": meta.get("primary_failure_type", "unknown"),
                "failure_types": meta.get("failure_types", ["unknown"]),
                "recovery_like": bool(meta.get("recovery_like", False)),
                "all_repeats_failed_interval": sum(in_interval) == 0,
                "all_repeats_failed_within_3": sum(within_3) == 0,
            })

        rows.sort(key=lambda row: (
            float(row["within_3_success_rate"]),
            float(row["in_interval_success_rate"]),
            -float(row["median_absolute_interval_error"]),
            -float(row["mean_absolute_interval_error"]),
            -float(row["worst_absolute_interval_error"]),
            str(row["rollout_id"]),
        ))
        return {
            "available": True,
            "run_id": run_name,
            "default_config_id": default_config_id,
            "configs": configs,
            "rows": rows,
            "repeat_count": max(repeat_counts) if repeat_counts else 0,
        }

    def save_localization_challenge_set(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValidationError("Challenge-set request must be an object")
        name = str(payload.get("name") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
            raise ValidationError("Invalid challenge-set name")
        source_run = str(payload.get("source_run") or "").strip()
        config_ids = payload.get("config_ids")
        rollout_ids = payload.get("rollout_ids")
        criterion = str(payload.get("criterion") or "outside_interval")
        if criterion not in {"outside_interval", "outside_3"}:
            raise ValidationError("Unsupported challenge criterion")
        if not isinstance(config_ids, list) or not config_ids:
            raise ValidationError("At least one localization config must be selected")
        if not isinstance(rollout_ids, list) or not rollout_ids:
            raise ValidationError("At least one rollout must be selected")
        if any(not isinstance(value, str) or not value for value in config_ids + rollout_ids):
            raise ValidationError("Config and rollout ids must be strings")

        challenge = self.localization_challenge_data(source_run)
        if not challenge.get("available"):
            raise ValidationError(str(challenge.get("reason") or "Challenge data unavailable"))
        rows_by_key = {
            (str(row["config_id"]), str(row["rollout_id"])): row
            for row in challenge["rows"]
        }
        available_configs = {str(row["config_id"]) for row in challenge["configs"]}
        if any(config_id not in available_configs for config_id in config_ids):
            raise ValidationError("Unknown config id in challenge selection")

        ranked_rows: list[dict[str, Any]] = []
        all_rollout_ids = sorted({
            str(row["rollout_id"])
            for row in challenge["rows"]
            if str(row["config_id"]) in config_ids
        })
        for rollout_id in all_rollout_ids:
            per_config_rank = [
                rows_by_key.get((config_id, rollout_id))
                for config_id in config_ids
            ]
            if any(row is None for row in per_config_rank):
                continue
            concrete_rows = [row for row in per_config_rank if row is not None]
            persistent = all(
                bool(row[
                    "all_repeats_failed_within_3"
                    if criterion == "outside_3"
                    else "all_repeats_failed_interval"
                ])
                for row in concrete_rows
            )
            first_rank = concrete_rows[0]
            ranked_rows.append({
                "rollout_id": rollout_id,
                "persistent_across_selected_configs": persistent,
                "repeat_count": min(int(row["repeat_count"]) for row in concrete_rows),
                "in_interval_success_rate": sum(
                    float(row["in_interval_success_rate"]) for row in concrete_rows
                ) / len(concrete_rows),
                "within_3_success_rate": sum(
                    float(row["within_3_success_rate"]) for row in concrete_rows
                ) / len(concrete_rows),
                "median_absolute_interval_error": sum(
                    float(row["median_absolute_interval_error"]) for row in concrete_rows
                ) / len(concrete_rows),
                "mean_absolute_interval_error": sum(
                    float(row["mean_absolute_interval_error"]) for row in concrete_rows
                ) / len(concrete_rows),
                "worst_absolute_interval_error": max(
                    float(row["worst_absolute_interval_error"]) for row in concrete_rows
                ),
                "train_exposure_rate": sum(
                    float(row["train_exposure_rate"]) for row in concrete_rows
                ) / len(concrete_rows),
                "forced_train_rate": sum(
                    float(row.get("forced_train_rate") or 0.0) for row in concrete_rows
                ) / len(concrete_rows),
                "task_id": first_rank.get("task_id"),
                "primary_failure_type": first_rank.get("primary_failure_type"),
                "multi_event": first_rank.get("multi_event"),
                "recovery_like": first_rank.get("recovery_like"),
            })
        ranked_rows.sort(key=lambda row: (
            float(row["within_3_success_rate"]),
            float(row["in_interval_success_rate"]),
            -float(row["median_absolute_interval_error"]),
            -float(row["mean_absolute_interval_error"]),
            -float(row["worst_absolute_interval_error"]),
            str(row["rollout_id"]),
        ))
        for rank, row in enumerate(ranked_rows, start=1):
            row["rank"] = rank

        entries: list[dict[str, Any]] = []
        for rollout_id in rollout_ids:
            per_config = []
            for config_id in config_ids:
                row = rows_by_key.get((config_id, rollout_id))
                if row is None:
                    raise ValidationError(
                        f"Rollout {rollout_id} has no full-failure result for {config_id}"
                    )
                per_config.append(row)
            entries.append({
                "rollout_id": rollout_id,
                "task_id": per_config[0].get("task_id"),
                "task_key": per_config[0].get("task_key"),
                "primary_failure_type": per_config[0].get("primary_failure_type"),
                "failure_types": per_config[0].get("failure_types"),
                "multi_event": per_config[0].get("multi_event"),
                "recovery_like": per_config[0].get("recovery_like"),
                "persistent_across_selected_configs": all(
                    bool(row[
                        "all_repeats_failed_within_3"
                        if criterion == "outside_3"
                        else "all_repeats_failed_interval"
                    ])
                    for row in per_config
                ),
                "per_config": per_config,
            })

        task_counts: dict[str, int] = {}
        failure_counts: dict[str, int] = {}
        multi_event_n = 0
        recovery_like_n = 0
        for entry in entries:
            task = str(entry.get("task_id"))
            task_counts[task] = task_counts.get(task, 0) + 1
            failure_type = str(entry.get("primary_failure_type") or "unknown")
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
            multi_event_n += 1 if entry.get("multi_event") else 0
            recovery_like_n += 1 if entry.get("recovery_like") else 0

        manifest = {
            "schema_version": 1,
            "name": name,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "purpose": "diagnostic_localization_challenge_set",
            "source_run": source_run,
            "config_ids": list(config_ids),
            "criterion": criterion,
            "rollout_ids": list(rollout_ids),
            "entries": entries,
            "ranked_hard_cases_csv": f"{name}_ranked_hard_cases.csv",
            "challenge_csv": f"{name}.csv",
            "composition": {
                "size": len(entries),
                "tasks": task_counts,
                "failure_types": failure_counts,
                "multi_event": multi_event_n,
                "recovery_like": recovery_like_n,
            },
        }
        self.localization_challenge_root.mkdir(parents=True, exist_ok=True)
        path = self.localization_challenge_root / f"{name}.json"
        overwrite = bool(payload.get("overwrite", False))
        if path.exists() and not overwrite:
            raise ValidationError("Challenge set already exists; set overwrite=true to replace it")
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        csv_path = self.localization_challenge_root / f"{name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "rollout_id", "task_id", "primary_failure_type",
                "multi_event", "recovery_like", "persistent_across_selected_configs",
            ])
            writer.writeheader()
            for entry in entries:
                writer.writerow({
                    key: entry.get(key)
                    for key in writer.fieldnames or []
                })

        ranked_path = self.localization_challenge_root / f"{name}_ranked_hard_cases.csv"
        ranked_fields = [
            "rank", "rollout_id", "persistent_across_selected_configs", "repeat_count",
            "in_interval_success_rate", "within_3_success_rate",
            "median_absolute_interval_error", "mean_absolute_interval_error",
            "worst_absolute_interval_error", "train_exposure_rate", "forced_train_rate",
            "task_id", "primary_failure_type", "multi_event", "recovery_like",
        ]
        with ranked_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ranked_fields)
            writer.writeheader()
            for row in ranked_rows:
                writer.writerow({key: row.get(key) for key in ranked_fields})
        return manifest

    def localization_artifact(self, run_name: str, artifact_name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name) or not re.fullmatch(r"[A-Za-z0-9._-]+", artifact_name):
            raise FileNotFoundError(artifact_name)
        path = (self.localization_root / run_name / artifact_name).resolve()
        try:
            path.relative_to(self.localization_root.resolve())
        except ValueError as exc:
            raise FileNotFoundError(artifact_name) from exc
        if not path.is_file():
            raise FileNotFoundError(artifact_name)
        return path

    def start_localization_spec_run(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("spec"), dict):
            raise ValidationError("Localization run requires a spec object")
        spec = dict(payload["spec"])
        name = str(spec.get("name") or "localization_experiment").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
            raise ValidationError("Invalid experiment name")
        if not self.robo_python.is_file() or not os.access(self.robo_python, os.X_OK):
            raise AnalysisEnvironmentError(
                "Robo-Dopamine PyTorch Python is unavailable: "
                + self._relative(self.robo_python)
            )
        pool_root = self.baselines.baseline_root
        if not pool_root.is_dir():
            raise ValidationError("Baseline output pool does not exist")
        script = self.project_root / "tools" / "train_robo_localization.py"
        if not script.is_file():
            raise ValidationError("Localization trainer is missing")
        job_id = "analysis-localization-" + uuid.uuid4().hex[:12]
        self.localization_root.mkdir(parents=True, exist_ok=True)
        workspace = self.localization_root / ".web_jobs" / job_id
        output_temp = workspace / "output"
        output_final = self.localization_root / (
            dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            + "_" + name + "_" + job_id[-8:]
        )
        workspace.mkdir(parents=True, exist_ok=False)
        spec_path = workspace / "experiment_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        command = [
            str(self.robo_python), str(script),
            "--spec", str(spec_path),
            "--run-pool-root", str(pool_root),
            "--manifest", str(self.manifest_path),
            "--annotations", str(self.annotation_root / "records"),
            "--output-dir", str(output_temp),
        ]
        self.localization_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.coordinator.acquire(job_id, "analysis")
        try:
            job = {
                "job_id": job_id,
                "job_type": "analysis",
                "analysis_kind": "robo_localization_experiment",
                "status": "queued",
                "parameters": {"experiment_name": name, "spec": spec},
                "command": command,
                "output_dir": self._relative(output_final),
                "output_temp": self._relative(output_temp),
                "spec_path": self._relative(spec_path),
                "log_path": self._relative(self.log_root / f"{job_id}.log"),
                "started_at": None, "finished_at": None, "return_code": None, "error": None,
                "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "interpreter": str(self.robo_python),
            }
            with self.jobs_lock:
                self.jobs[job_id] = job
            self.tmux.submit(
                job, command, self.log_root / f"{job_id}.log",
                interpreter=str(self.robo_python),
                environment={"MPLBACKEND": "Agg"},
                on_poll=self._on_job_poll, on_finished=self._on_job_finished,
            )
        except Exception:
            with self.jobs_lock:
                self.jobs.pop(job_id, None)
            self.coordinator.release(job_id)
            raise
        return dict(job)

    def start_run(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValidationError("Analysis request must be a JSON object")
        if payload.get("analysis_kind") == "robo_localization_experiment":
            return self.start_localization_spec_run(payload)
        if payload.get("analysis_kind") == "robo_bilstm_success_ablation":
            return self.start_robo_localization_head_run(payload)
        if payload.get("analysis_kind") == "robo_bilstm_label_loss_ablation":
            return self.start_robo_label_loss_run(payload)
        if payload.get("analysis_kind") in {
            "robo_incremental_hop",
            "robo_hop_comparison",
        }:
            return self.start_robo_hop_run(payload)
        self.require_environment()
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
                if job.get("analysis_kind") == "robo_localization_experiment":
                    required = (
                        "metadata.json", "config.json", "experiment_manifest.json",
                        "training_records.json", "summary.csv", "per_rollout_predictions.csv",
                        "all_failure_predictions.csv",
                    )
                    missing_message = "Localization experiment completed without all required artifacts"
                elif job.get("analysis_kind") == "robo_bilstm_success_ablation":
                    required = ROBO_LOCALIZATION_HEAD_REQUIRED_FILES
                    missing_message = (
                        "BiLSTM localization-head training completed without all required artifacts"
                    )
                elif job.get("analysis_kind") == "robo_bilstm_label_loss_ablation":
                    required = ROBO_LABEL_LOSS_REQUIRED_FILES
                    missing_message = (
                        "BiLSTM label/loss ablation completed without all required artifacts"
                    )
                elif job.get("analysis_kind") in {
                    "robo_incremental_hop",
                    "robo_hop_comparison",
                }:
                    required = ROBO_HOP_REQUIRED_FILES
                    if job.get("analysis_kind") == "robo_hop_comparison":
                        required = (
                            *required,
                            *ROBO_HOP_EXTENDED_FILES,
                        )
                    missing_message = (
                        "Robo-Dopamine hop comparison completed without all required artifacts"
                    )
                else:
                    required = (
                        "metadata.json",
                        "event_metrics.jsonl",
                        *ANALYSIS_TABLE_FILES.values(),
                    )
                    missing_message = (
                        "Temporal analysis completed without all required artifacts"
                    )
                if not all((output_temp / name).is_file() for name in required):
                    raise OSError(missing_message)
                if (
                    job.get("analysis_kind") == "robo_localization_experiment"
                    and not (output_temp / "checkpoints").is_dir()
                ):
                    raise OSError("Localization experiment completed without checkpoints")
                if output_final.exists():
                    raise OSError(f"Analysis output already exists: {output_final}")
                os.replace(output_temp, output_final)
                job["status"] = "complete"
            else:
                job["status"] = "failed"
                if job.get("analysis_kind") == "robo_localization_experiment":
                    label = "Localization experiment"
                elif job.get("analysis_kind") == "robo_bilstm_success_ablation":
                    label = "BiLSTM localization-head training"
                elif job.get("analysis_kind") == "robo_bilstm_label_loss_ablation":
                    label = "BiLSTM label/loss ablation"
                else:
                    label = (
                        "Robo-Dopamine hop comparison"
                        if job.get("analysis_kind") in {
                            "robo_incremental_hop",
                            "robo_hop_comparison",
                        }
                        else "Temporal analysis"
                    )
                error = error or f"{label} exited with code {return_code}"
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
