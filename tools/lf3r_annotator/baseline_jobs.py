"""Baseline launch planning, command construction, and job lifecycle."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from backend_core import (
    ValidationError,
    validate_instruction_condition,
    validate_run_scope,
)
from baseline_constants import (
    BASELINE_ADVANCED_FIELDS,
    BASELINE_METHOD_OPTION_FIELDS,
    BASELINE_METHODS,
    BASELINE_RUN_STATUSES,
)


class BaselineJobsMixin:
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
            "procvlm_frame_stride": (1, 1000000),
            "procvlm_max_sampled_frames": (1, 1000000),
            "procvlm_max_new_tokens": (1, 1000000),
            "procvlm_tracker_support_threshold": (1, 1000000),
            "procvlm_tracker_window_size": (1, 1000000),
            "procvlm_tracker_max_forward_jump": (1, 1),
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
        for name in ("dtype", "robo_eval_mode", "robo_camera_mode", "procvlm_procedure_mode"):
            if name in options and options[name] is not None:
                value = str(options[name]).strip()
                if not value or len(value) > 80:
                    raise ValidationError(f"{name} must be a non-empty short string")
                options[name] = value
        if "procvlm_procedure_mode" in options and options["procvlm_procedure_mode"] not in {"baseline", "tracker_only", "stateful_history"}:
            raise ValidationError("procvlm_procedure_mode must be baseline, tracker_only, or stateful_history")
        if baseline == "procvlm" and options.get("procvlm_procedure_mode", "baseline") == "baseline":
            for name in (
                "procvlm_procedure_config",
                "procvlm_tracker_support_threshold",
                "procvlm_tracker_window_size",
                "procvlm_tracker_max_forward_jump",
            ):
                options.pop(name, None)
        if "robo_eval_mode" in options and options["robo_eval_mode"] not in {"fused", "forward", "incremental", "backward"}:
            raise ValidationError("robo_eval_mode must be fused, forward, incremental, or backward")
        if "robo_camera_mode" in options and options["robo_camera_mode"] not in {"auto", "single_view", "multi_view"}:
            raise ValidationError("robo_camera_mode must be auto, single_view, or multi_view")
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
        if "procvlm_procedure_config" in options and options["procvlm_procedure_config"] not in (None, ""):
            resolved = self._project_path(str(options["procvlm_procedure_config"]))
            if not resolved.is_file():
                raise ValidationError(
                    f"procvlm_procedure_config does not exist inside the project: {options['procvlm_procedure_config']}"
                )
            options["procvlm_procedure_config"] = str(resolved)
        if options.get("procvlm_tracker_support_threshold", 7) > options.get("procvlm_tracker_window_size", 9):
            raise ValidationError("procvlm_tracker_support_threshold cannot exceed procvlm_tracker_window_size")
        if options.get("procvlm_procedure_mode", "baseline") != "baseline" and not options.get("procvlm_procedure_config"):
            raise ValidationError("tracker_only/stateful_history ProcVLM requires procvlm_procedure_config")
        for name in ("render_video", "validate_environment", "dry_run", "procvlm_enable_value_head"):
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
        manifest_path: Path | None = None,
        instruction_condition: str = "full_instruction",
        rollout_ids: list[str] | None = None,
    ) -> list[str]:
        runner = self.project_root / "tools" / "baselines" / "run_lf3r_baseline.py"
        if not runner.is_file():
            raise ValidationError("Baseline runner is not installed")
        instruction_condition = validate_instruction_condition(instruction_condition)
        selected_manifest = manifest_path or self._manifest_for_condition(instruction_condition)
        command = [
            sys.executable, str(runner),
            "--baseline", baseline,
            "--manifest", str(selected_manifest),
            "--instruction-condition", instruction_condition,
            "--data-root", str(self.project_root),
            "--output-dir", str(run_parent),
            "--logs-dir", str(self.web_logs_root),
            "--gpu", gpu,
            "--vllm-free-memory-fraction", str(utilization),
            "--continue-on-error",
        ]
        if rollout_ids is not None:
            # Explicit rollout IDs define the authoritative pre-range selection.
            # This is used both for instruction variants and for the
            # missing-valid-result batch filter, so positional worker ranges
            # remain relative to the filtered rollout list.
            command.extend(["--partition", "all"])
            for rollout_id in rollout_ids:
                command.extend(["--rollout-id", str(rollout_id)])
        elif scope in {"all", "controlled_analysis"}:
            command.extend(["--partition", scope])
        else:
            command.extend(["--partition", "natural_observation", "--task-suite", scope])
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
            # Preserve the default when no explicit mode is supplied.
            # Make its default explicit and keep batch/API callers consistent.
            options = {**options, "robo_eval_mode": "fused"}
        flag_values = {
            "model_path": "--model-path",
            "dtype": "--dtype",
            "tensor_parallel_size": "--tensor-parallel-size",
            "procvlm_window_size": "--procvlm-window-size",
            "procvlm_frame_stride": "--procvlm-frame-stride",
            "procvlm_max_sampled_frames": "--procvlm-max-sampled-frames",
            "procvlm_max_new_tokens": "--procvlm-max-new-tokens",
            "procvlm_procedure_mode": "--procvlm-procedure-mode",
            "procvlm_procedure_config": "--procvlm-procedure-config",
            "procvlm_tracker_support_threshold": "--procvlm-tracker-support-threshold",
            "procvlm_tracker_window_size": "--procvlm-tracker-window-size",
            "procvlm_tracker_max_forward_jump": "--procvlm-tracker-max-forward-jump",
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
            "robo_camera_mode": "--robo-camera-mode",
            "goal_image": "--goal-image",
            "robo_localization_ckpt": "--robo-localization-ckpt",
            "densereward_frame_interval": "--densereward-frame-interval",
            "densereward_max_new_tokens": "--densereward-max-new-tokens",
        }
        for name, flag in flag_values.items():
            if name in options and options[name] not in (None, ""):
                command.extend([flag, str(options[name])])
        if options.get("procvlm_enable_value_head"):
            command.append("--procvlm-enable-value-head")
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
        instruction_condition: str = "full_instruction",
        manifest_path: Path | None = None,
        variant_rollout_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "job_type": "baseline",
            "baseline_mode": "rollout" if rollout_id else "batch",
            "baseline": baseline,
            "rollout_id": rollout_id,
            "variant_rollout_id": variant_rollout_id,
            "instruction_condition": instruction_condition,
            "manifest": self._relative(manifest_path) if manifest_path else self._relative(self.manifest_path),
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

    def start_run(
        self,
        rollout: dict[str, Any],
        baseline: str,
        gpu: str,
        memory_utilization: Any = 0.80,
        instruction_condition: Any = "full_instruction",
        options: Any = None,
    ) -> dict[str, Any]:
        if baseline not in BASELINE_METHODS:
            raise ValidationError("Invalid baseline method")
        instruction_condition = validate_instruction_condition(instruction_condition)
        options = self._validate_options(baseline, options)
        run_rollout = rollout
        if instruction_condition != "full_instruction":
            run_rollout = self._variant_record_for_source(
                str(rollout["id"]), instruction_condition
            )
        manifest_path = self._manifest_for_condition(instruction_condition)
        output_parent = (
            self.web_output_root
            if instruction_condition == "full_instruction"
            else self.variant_output_root / instruction_condition
        )
        gpu = self._validate_gpu(gpu)
        if isinstance(memory_utilization, bool):
            raise ValidationError("memory_utilization must be a number")
        try:
            utilization = float(memory_utilization)
        except (TypeError, ValueError) as error:
            raise ValidationError("memory_utilization must be a number") from error
        if not math.isfinite(utilization) or not 0.0 < utilization <= 1.0:
            raise ValidationError("memory_utilization must be in (0, 1]")
        output_parent.mkdir(parents=True, exist_ok=True)
        self.web_logs_root.mkdir(parents=True, exist_ok=True)
        job_id = baseline + "-" + uuid.uuid4().hex[:12]
        self.coordinator.acquire(job_id, "baseline")
        try:
            command = self._baseline_command(
                baseline,
                "all",
                gpu,
                utilization,
                output_parent,
                options,
                start_index=0,
                limit=None,
                manifest_path=manifest_path,
                instruction_condition=instruction_condition,
            )
            command.extend(["--rollout-id", run_rollout["id"]])
            job = self._new_job(
                command,
                baseline,
                "all",
                1,
                gpu,
                utilization,
                output_parent,
                job_id,
                str(rollout["id"]),
                instruction_condition=instruction_condition,
                manifest_path=manifest_path,
                variant_rollout_id=(
                    str(run_rollout["id"])
                    if run_rollout["id"] != rollout["id"]
                    else None
                ),
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
            "limit", "parallel_workers", "workers", "options", "instruction_condition",
            "result_filter",
        }
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            raise ValidationError("Unknown batch field(s): " + ", ".join(sorted(unknown_fields)))
        baseline = str(payload.get("baseline", ""))
        if baseline not in BASELINE_METHODS:
            raise ValidationError("Invalid baseline method")
        instruction_condition = validate_instruction_condition(
            payload.get("instruction_condition", "full_instruction")
        )
        scope = validate_run_scope(payload.get("scope"))
        result_filter = self._validate_result_filter(payload.get("result_filter", "all"))
        records = self._condition_records(instruction_condition, scope)
        scope_record_count = len(records)
        if not records:
            raise ValidationError(f"No rollouts matched scope {scope}")
        valid_result_ids: set[str] = set()
        incomplete_source_ids: list[str] = []
        complete_annotation_count = scope_record_count
        if result_filter == "missing_valid":
            records, incomplete_source_ids = self._complete_annotation_records(
                records,
                instruction_condition,
            )
            complete_annotation_count = len(records)
            if not records:
                raise ValidationError(
                    f"No complete annotations matched scope {scope} for "
                    f"{instruction_condition}"
                )
            valid_result_ids = self._valid_result_rollout_ids(
                baseline,
                instruction_condition,
                records,
            )
            records = [
                record
                for record in records
                if str(record["id"]) not in valid_result_ids
            ]
            if not records:
                raise ValidationError(
                    f"Every complete annotation matched by scope {scope} already has "
                    f"a valid {baseline} result for {instruction_condition}"
                )
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
        output_parent = (
            self.web_output_root
            if instruction_condition == "full_instruction"
            else self.variant_output_root / instruction_condition / "web_runs"
        )
        output_parent.mkdir(parents=True, exist_ok=True)
        self.web_logs_root.mkdir(parents=True, exist_ok=True)
        job_id = baseline + "-batch-" + uuid.uuid4().hex[:12]
        self.coordinator.acquire(job_id, "baseline")
        try:
            run_parent = output_parent / job_id
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
                manifest_path=self._manifest_for_condition(instruction_condition),
                instruction_condition=instruction_condition,
                rollout_ids=(
                    [str(record["id"]) for record in records]
                    if (
                        instruction_condition != "full_instruction"
                        or result_filter == "missing_valid"
                    )
                    else None
                ),
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
                instruction_condition=instruction_condition,
                manifest_path=self._manifest_for_condition(instruction_condition),
            )
            job["options"] = options
            job["result_filter"] = result_filter
            job["scope_rollouts_before_result_filter"] = scope_record_count
            job["complete_annotation_rollouts"] = complete_annotation_count
            job["incomplete_annotation_rollouts_skipped"] = (
                len(incomplete_source_ids) if result_filter == "missing_valid" else 0
            )
            job["valid_result_rollouts_skipped"] = (
                len(valid_result_ids) if result_filter == "missing_valid" else 0
            )
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
            self.run_index.upsert(run_path, metadata)
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
            run_path, metadata = self._find_job_run(job)
            if (
                run_path is not None
                and metadata is not None
                and str(metadata.get("status") or "") in BASELINE_RUN_STATUSES
                and str(metadata.get("baseline") or "") in BASELINE_METHODS
            ):
                self._run_inventory(run_path, str(metadata["baseline"]))
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
