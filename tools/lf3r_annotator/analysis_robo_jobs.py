"""Robo-Dopamine analysis job launchers."""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import signal
import time
import uuid
from pathlib import Path
from typing import Any

from backend_core import (
    AnalysisEnvironmentError,
    ValidationError,
    atomic_json_write,
    select_scope_records,
    validate_run_scope,
)
from baseline_constants import BASELINE_RUN_STATUSES


class AnalysisRoboJobsMixin:
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

