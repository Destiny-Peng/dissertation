#!/usr/bin/env python3
"""WebUI baseline service with explicit extensions over the core baseline service."""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import server


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _latest_job_statuses(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return latest
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        rollout_id = row.get("rollout_id")
        if rollout_id not in (None, ""):
            latest[str(rollout_id)] = row
    return latest


def _remove_cli_pairs(command: list[str], flags: set[str]) -> list[str]:
    cleaned: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token in flags:
            index += 2
            continue
        cleaned.append(token)
        index += 1
    return cleaned


class WebUIBaselineService(server.BaselineService):
    """Baseline behavior required by the current WebUI, without monkey-patching."""

    def _validate_options(self, baseline: str, raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise server.ValidationError("options must be a JSON object")

        unknown = set(raw) - server.BASELINE_ADVANCED_FIELDS
        if unknown:
            raise server.ValidationError(
                "Unknown baseline option(s): " + ", ".join(sorted(unknown))
            )
        unsupported = set(raw) - server.BASELINE_METHOD_OPTION_FIELDS[baseline]
        if unsupported:
            raise server.ValidationError(
                f"Options not supported by {baseline}: " + ", ".join(sorted(unsupported))
            )

        options = dict(raw)
        integer_fields = {
            "tensor_parallel_size": (1, 32),
            "procvlm_window_size": (1, 4096),
            "procvlm_frame_stride": (1, 1_000_000),
            "procvlm_max_sampled_frames": (1, 1_000_000),
            "procvlm_max_new_tokens": (1, 1_000_000),
            "procvlm_tracker_support_threshold": (1, 1_000_000),
            "procvlm_tracker_window_size": (1, 1_000_000),
            "procvlm_tracker_max_forward_jump": (1, 1),
            "rynn_num_frames": (1, 1_000_000),
            "rynn_num_steps": (1, 1_000_000),
            "rynn_evaluation_interval": (1, 1_000_000),
            "rynn_max_image_side": (1, 8192),
            "rynn_max_new_tokens": (1, 1_000_000),
            "robo_frame_interval": (1, 1_000_000),
            "robo_batch_size": (1, 4096),
            "densereward_frame_interval": (1, 1_000_000),
            "densereward_max_new_tokens": (1, 1_000_000),
        }
        for name, (minimum, maximum) in integer_fields.items():
            if name in options and options[name] is not None:
                options[name] = self._integer(options[name], name, minimum, maximum)

        if "rynn_batch_size" in options and options["rynn_batch_size"] is not None:
            options["rynn_batch_size"] = self._positive_integer(
                options["rynn_batch_size"], "rynn_batch_size"
            )

        for name in ("dtype", "robo_eval_mode", "procvlm_procedure_mode"):
            if name in options and options[name] is not None:
                value = str(options[name]).strip()
                if not value or len(value) > 80:
                    raise server.ValidationError(f"{name} must be a non-empty short string")
                options[name] = value

        if (
            "procvlm_procedure_mode" in options
            and options["procvlm_procedure_mode"]
            not in {"baseline", "tracker_only", "stateful_history"}
        ):
            raise server.ValidationError(
                "procvlm_procedure_mode must be baseline, tracker_only, or stateful_history"
            )

        if baseline == "procvlm" and options.get("procvlm_procedure_mode", "baseline") == "baseline":
            for name in (
                "procvlm_procedure_config",
                "procvlm_tracker_support_threshold",
                "procvlm_tracker_window_size",
                "procvlm_tracker_max_forward_jump",
            ):
                options.pop(name, None)

        if (
            "robo_eval_mode" in options
            and options["robo_eval_mode"]
            not in {"fused", "forward", "incremental", "backward"}
        ):
            raise server.ValidationError(
                "robo_eval_mode must be fused, forward, incremental, or backward"
            )

        for name in ("robot_description", "camera_description"):
            if name in options and options[name] is not None:
                value = str(options[name])
                if len(value) > 4000:
                    raise server.ValidationError(f"{name} is too long")
                options[name] = value

        for name in (
            "render_video",
            "validate_environment",
            "dry_run",
            "procvlm_enable_value_head",
            "procvlm_use_lora",
        ):
            if name in options and not isinstance(options[name], bool):
                raise server.ValidationError(f"{name} must be boolean")

        model_path: Path | None = None
        is_procvlm_adapter = False
        if "model_path" in options and options["model_path"] not in (None, ""):
            model_path = self._project_path(str(options["model_path"]))
            if not model_path.exists():
                raise server.ValidationError(
                    f"model_path does not exist inside the project: {options['model_path']}"
                )
            options["model_path"] = str(model_path)
            is_procvlm_adapter = bool(
                baseline == "procvlm"
                and model_path.is_dir()
                and (model_path / "adapter_config.json").is_file()
            )

        if baseline == "procvlm":
            use_lora = bool(options.get("procvlm_use_lora", False))
            if use_lora:
                if model_path is None:
                    raise server.ValidationError(
                        "ProcVLM One-shot LoRA mode requires model_path to point to "
                        "the saved LoRA checkpoint directory"
                    )
                if not is_procvlm_adapter:
                    raise server.ValidationError(
                        "ProcVLM One-shot LoRA mode requires a checkpoint directory "
                        "containing adapter_config.json"
                    )
            elif is_procvlm_adapter:
                raise server.ValidationError(
                    "model_path is a ProcVLM LoRA adapter checkpoint; "
                    "select Inference mode = One-shot LoRA"
                )

        if (
            "procvlm_procedure_config" in options
            and options["procvlm_procedure_config"] not in (None, "")
        ):
            resolved = self._project_path(str(options["procvlm_procedure_config"]))
            if not resolved.is_file():
                raise server.ValidationError(
                    "procvlm_procedure_config does not exist inside the project: "
                    + str(options["procvlm_procedure_config"])
                )
            options["procvlm_procedure_config"] = str(resolved)

        if options.get("procvlm_tracker_support_threshold", 7) > options.get(
            "procvlm_tracker_window_size", 9
        ):
            raise server.ValidationError(
                "procvlm_tracker_support_threshold cannot exceed procvlm_tracker_window_size"
            )
        if (
            baseline == "procvlm"
            and options.get("procvlm_procedure_mode", "baseline") != "baseline"
            and not options.get("procvlm_procedure_config")
        ):
            raise server.ValidationError(
                "tracker_only/stateful_history ProcVLM requires procvlm_procedure_config"
            )

        if "goal_image" in options and options["goal_image"] not in (None, ""):
            resolved = self._project_path(str(options["goal_image"]))
            if not resolved.is_file():
                raise server.ValidationError(
                    f"goal_image does not exist inside the project: {options['goal_image']}"
                )
            options["goal_image"] = str(resolved)

        if (
            "robo_localization_ckpt" in options
            and options["robo_localization_ckpt"] not in (None, "")
        ):
            if baseline != "robo_dopamine":
                raise server.ValidationError(
                    "robo_localization_ckpt is only valid for Robo-Dopamine"
                )
            resolved = self._project_path(str(options["robo_localization_ckpt"]))
            if not resolved.is_file():
                raise server.ValidationError(
                    "robo_localization_ckpt does not exist inside the project: "
                    + str(options["robo_localization_ckpt"])
                )
            if resolved.suffix.lower() not in {".pt", ".pth"}:
                raise server.ValidationError(
                    "robo_localization_ckpt must be a .pt or .pth checkpoint"
                )
            if options.get("robo_eval_mode", "fused") != "fused":
                raise server.ValidationError(
                    "robo_localization_ckpt requires Robo-Dopamine eval mode = fused"
                )
            options["robo_localization_ckpt"] = str(resolved)

        return options

    def _baseline_command(
        self,
        baseline: str,
        scope: str,
        gpu: str,
        utilization: float,
        run_parent: Path,
        options: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> list[str]:
        forwarded = dict(options)
        forwarded.pop("procvlm_use_lora", None)
        command = super()._baseline_command(
            baseline,
            scope,
            gpu,
            utilization,
            run_parent,
            forwarded,
            *args,
            **kwargs,
        )

        instruction_condition = kwargs.get("instruction_condition")
        if instruction_condition is None and len(args) > 6:
            instruction_condition = args[6]
        instruction_condition = str(instruction_condition or "full_instruction")
        rollout_ids = kwargs.get("rollout_ids")
        if rollout_ids is None and len(args) > 7:
            rollout_ids = args[7]

        if (
            instruction_condition != "full_instruction"
            or scope in server.RESERVED_RUN_SCOPES
            or rollout_ids is not None
        ):
            return command

        selected = self._condition_records("full_instruction", scope)
        command = _remove_cli_pairs(
            command,
            {"--partition", "--task-suite", "--rollout-id"},
        )
        command.extend(["--partition", "all"])
        for record in selected:
            command.extend(["--rollout-id", str(record["id"])])
        return command

    def _run_candidates(self, method: str) -> list[tuple[Path, dict[str, Any]]]:
        return self._indexed_run_candidates(method, server.BASELINE_READABLE_RUN_STATUSES)

    def _explicit_run_candidate(
        self,
        method: str,
        run_root: Any,
        allowed_conditions: set[str],
    ) -> tuple[Path, dict[str, Any]]:
        if not isinstance(run_root, (str, Path)) or not str(run_root).strip():
            raise server.ValidationError(
                f"{method} run selection must be a project-relative run path"
            )
        path = self._project_path(str(run_root))
        try:
            path.relative_to(self.baseline_root.resolve())
        except ValueError as exc:
            raise server.ValidationError(
                f"Selected {method} run must be inside outputs/baselines"
            ) from exc
        metadata_path = path / "run.json"
        if not metadata_path.is_file():
            raise server.ValidationError(f"Selected {method} run.json is missing")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise server.ValidationError(
                f"Selected {method} run metadata is invalid"
            ) from exc
        if metadata.get("baseline") != method:
            raise server.ValidationError(f"Selected run is not a {method} run")
        if metadata.get("status") not in server.BASELINE_READABLE_RUN_STATUSES:
            raise server.ValidationError(
                f"Selected {method} run has no readable completed rollout results"
            )
        condition = self._run_instruction_condition(metadata)
        if condition not in allowed_conditions:
            allowed = ", ".join(sorted(allowed_conditions))
            raise server.ValidationError(
                f"Selected {method} run is for instruction condition {condition!r}; "
                f"expected {allowed}"
            )
        return path, metadata

    def _read_method(
        self,
        method: str,
        run_path: Path,
        rollout: dict[str, Any],
        run_summary: dict[str, Any],
    ) -> dict[str, Any]:
        if run_summary.get("status") == "running":
            rollout_id = str(rollout.get("id") or "")
            if rollout_id not in server.run_rollout_ids(run_path):
                raise FileNotFoundError(run_path / "raw" / rollout_id)
        return super()._read_method(method, run_path, rollout, run_summary)

    def _refresh_progress(self, job_id: str) -> None:
        super()._refresh_progress(job_id)
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            run_root_value = job.get("run_root")
            baseline = str(job.get("baseline") or "")
            total = int(
                job.get("unique_selected_rollouts")
                or job.get("selected_rollouts")
                or 0
            )
        if not run_root_value:
            return
        try:
            run_root = self._project_path(str(run_root_value))
        except server.ValidationError:
            return

        live: dict[str, Any] | None = None
        source = None
        if baseline == "procvlm":
            live = _read_json(run_root / "procvlm_state.json")
            if live:
                source = "procvlm_state"

        if not live:
            latest = _latest_job_statuses(run_root / "jobs.jsonl")
            if latest:
                completed = sum(
                    row.get("status") == "complete" for row in latest.values()
                )
                failed = sum(
                    row.get("status")
                    in {"failed", "interrupted", "fatal_engine_failure"}
                    for row in latest.values()
                )
                live = {
                    "completed_jobs": completed,
                    "failed_jobs": failed,
                    "pending_jobs": max(total - completed - failed, 0),
                }
                source = "jobs_jsonl"

        if not live:
            return
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            for field in ("completed_jobs", "failed_jobs", "pending_jobs"):
                value = live.get(field)
                if value is None:
                    continue
                try:
                    job[field] = int(value)
                except (TypeError, ValueError):
                    continue
            job["live_progress_source"] = source
            job["live_progress_updated_at"] = dt.datetime.now(
                dt.timezone.utc
            ).isoformat()
            live_status = live.get("status")
            if live_status not in (None, ""):
                job["live_run_status"] = str(live_status)

    def _atomic_update_run_metadata(self, job: dict[str, Any]) -> None:
        run_root_value = job.get("run_root")
        if not run_root_value:
            return
        run_root = Path(str(run_root_value))
        if not run_root.is_absolute():
            run_root = self.project_root / run_root
        metadata_path = run_root / "run.json"
        payload = _read_json(metadata_path)
        if payload is None:
            return
        payload.update(
            {
                "status": "cancelled",
                "cancelled_at": job.get("cancelled_at"),
                "completed_at": job.get("cancelled_at"),
                "completed_jobs": int(job.get("completed_jobs") or 0),
                "failed_jobs": int(job.get("failed_jobs") or 0),
                "pending_jobs": int(job.get("pending_jobs") or 0),
            }
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            dir=metadata_path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, metadata_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _on_job_finished(
        self,
        job: dict[str, Any],
        return_code: int | None,
        reason: str | None,
    ) -> None:
        if not job.get("cancel_requested_at"):
            return super()._on_job_finished(job, return_code, reason)

        job_id = str(job["job_id"])
        try:
            self._refresh_progress(job_id)
            cancelled_at = dt.datetime.now(dt.timezone.utc).isoformat()
            with self.jobs_lock:
                job["status"] = "cancelled"
                job["tmux_state"] = "cancelled"
                job["cancelled_at"] = cancelled_at
                job["finished_at"] = cancelled_at
                job["return_code"] = return_code
                job["error"] = None
                job["run_status"] = "cancelled"
            self._atomic_update_run_metadata(job)
        finally:
            self.coordinator.release(job_id)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.get("status") not in {"queued", "running"}:
                raise server.ValidationError(
                    f"Baseline job is not active: {job.get('status') or 'unknown'}"
                )

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self.tmux.lock:
            shared = self.tmux.jobs.get(job_id)
            if shared is None:
                raise KeyError(job_id)
            if shared.get("cancel_requested_at"):
                return dict(shared)
            shared["cancel_requested_at"] = now
            shared["cancel_requested_by"] = "webui"
            shared["supervisor_status"] = "cancelling"
            self.tmux.persist(shared)
            session = str(shared.get("tmux_session") or "")

        if session and self.tmux.has_session(session):
            self.tmux._tmux(["send-keys", "-t", session, "C-c"], check=False)
            time.sleep(0.15)
            self.tmux._tmux(["kill-session", "-t", session], check=False)

        with self.tmux.lock:
            shared = self.tmux.jobs.get(job_id)
            if shared is not None:
                shared["cancel_signal_sent_at"] = dt.datetime.now(
                    dt.timezone.utc
                ).isoformat()
                self.tmux.persist(shared)
        self._refresh_progress(job_id)
        with self.jobs_lock:
            return dict(self.jobs[job_id])
