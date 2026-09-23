"""Baseline result readers and persistent run launch service."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from backend_core import (
    INSTRUCTION_VARIANT_CONDITIONS,
    JobCoordinator,
    ValidationError,
    load_manifest_records,
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
from stores import AnnotationStore
from task_supervisor import TmuxJobSupervisor


class BaselineService:
    """Read existing baseline outputs and optionally launch one bounded rollout."""

    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        coordinator: JobCoordinator | None = None,
        tmux: TmuxJobSupervisor | None = None,
        annotation_store: AnnotationStore | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.annotation_store = annotation_store
        self.baseline_root = self.project_root / "outputs" / "baselines"
        self.web_output_root = self.baseline_root / "web_runs"
        self.variant_manifest_path = (
            self.project_root
            / "tools"
            / "lf3r_annotator"
            / "instruction_variants"
            / "libero_10_v1"
            / "manifest.jsonl"
        )
        self.variant_output_root = (
            self.baseline_root / "instruction_variants" / "libero_10"
        )
        self.web_logs_root = self.project_root / "logs" / "baselines" / "web_runs"
        robo_python = os.environ.get("LF3R_ROBODOPAMINE_PYTHON")
        if robo_python:
            self.robo_python = Path(
                os.path.abspath(Path(robo_python).expanduser())
            )
        else:
            self.robo_python = (
                self.project_root
                / "conda_envs"
                / "LF3R-robo-dopamine"
                / "bin"
                / "python"
            )
        self.run_index = BaselineRunIndex(self.project_root, self.baseline_root)
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

    def _indexed_run_candidates(
        self,
        method: str,
        statuses: set[str],
    ) -> list[tuple[Path, dict[str, Any]]]:
        return self.run_index.candidates(method, statuses)

    def rebuild_run_index(self) -> dict[str, int]:
        result = self.run_index.rebuild()
        self.run_index.clear_rollout_cache()
        # Explicit rescans are intentionally allowed to warm the persistent
        # rollout/signal inventory once. Normal catalog reads never recurse
        # through outputs/baselines after the cache is populated.
        for method in BASELINE_METHODS:
            for run_path, _metadata in self._run_candidates(method):
                self._run_inventory(run_path, method)
        return result

    def _run_candidates(self, method: str) -> list[tuple[Path, dict[str, Any]]]:
        return self._indexed_run_candidates(method, BASELINE_RUN_STATUSES)

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
            "procedure_mode": metadata.get("procedure_mode") or metadata.get("arguments", {}).get("procvlm_procedure_mode"),
            "procedure_config": metadata.get("procedure_config") or metadata.get("arguments", {}).get("procvlm_procedure_config"),
        }

    @staticmethod
    def _posthoc_localization_summary(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        frame = payload.get("predicted_frame")
        if frame is None:
            return None
        return {
            key: payload.get(key)
            for key in (
                "generated_at",
                "checkpoint",
                "checkpoint_sha256",
                "checkpoint_stage",
                "checkpoint_config_id",
                "checkpoint_repeat",
                "checkpoint_seed",
                "predicted_index",
                "predicted_frame",
                "predicted_logit",
                "predicted_sigmoid",
                "frame_count",
                "source_prediction",
            )
        } | {"output_path": str(path)}

    def _posthoc_localization_history(
        self,
        run_path: Path,
        rollout_id: str,
    ) -> list[dict[str, Any]]:
        root = run_path / "raw" / rollout_id / "posthoc_localization"
        if not root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in root.glob("*.json"):
            summary = self._posthoc_localization_summary(path)
            if summary is not None:
                rows.append(summary)
        rows.sort(
            key=lambda row: str(row.get("generated_at") or ""),
            reverse=True,
        )
        return rows

    def run_posthoc_localization(
        self,
        rollout: dict[str, Any],
        payload: Any,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValidationError("Post-hoc localization request must be an object")
        run_root = str(payload.get("run_root") or "").strip()
        checkpoint_value = str(payload.get("checkpoint") or "").strip()
        scope = str(payload.get("scope") or "current").strip()
        if not run_root:
            raise ValidationError("run_root is required")
        if not checkpoint_value:
            raise ValidationError("checkpoint is required")
        if scope not in {"current", "all"}:
            raise ValidationError("scope must be current or all")

        run_path = self._project_path(run_root)
        try:
            run_path.relative_to(self.baseline_root)
        except ValueError as error:
            raise ValidationError("run_root must be inside outputs/baselines") from error
        if not run_path.is_dir():
            raise ValidationError("Selected Robo-Dopamine run does not exist")

        checkpoint = self._project_path(checkpoint_value)
        if not checkpoint.is_file():
            raise ValidationError("Localization checkpoint does not exist")
        if checkpoint.suffix.lower() not in {".pt", ".pth"}:
            raise ValidationError("Localization checkpoint must be .pt or .pth")

        python = self.robo_python
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValidationError(
                "Robo-Dopamine Python is unavailable: " + str(python)
            )
        script = self.project_root / "tools" / "baselines" / "posthoc_robo_localization.py"
        if not script.is_file():
            raise ValidationError("Post-hoc localization helper is not installed")

        if scope == "current":
            source_worker_result = str(
                payload.get("source_worker_result") or ""
            ).strip()
            if source_worker_result:
                candidate = self._project_path(source_worker_result)
                try:
                    candidate.relative_to(run_path)
                except ValueError as error:
                    raise ValidationError(
                        "source_worker_result must belong to the selected run"
                    ) from error
                worker_results = [candidate]
            else:
                worker_results = [
                    run_path / "raw" / str(rollout["id"]) / "worker_result.json"
                ]
        else:
            worker_results = sorted(
                path
                for path in (run_path / "raw").glob("*/worker_result.json")
                if path.is_file()
            )
        worker_results = [path for path in worker_results if path.is_file()]
        if not worker_results:
            raise ValidationError(
                "No saved Robo-Dopamine worker_result.json was found for this selection"
            )

        command = [
            str(python),
            str(script),
            "--project-root",
            str(self.project_root),
            "--checkpoint",
            str(checkpoint),
        ]
        for path in worker_results:
            command.extend(["--worker-result", str(path)])

        try:
            completed = subprocess.run(
                command,
                cwd=str(self.project_root),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ValidationError(
                "Post-hoc localization process failed to start: " + str(error)
            ) from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise ValidationError(
                "Post-hoc localization failed: " + detail[-4000:]
            )
        try:
            result = json.loads(completed.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as error:
            raise ValidationError(
                "Post-hoc localization returned invalid JSON"
            ) from error
        if not isinstance(result, dict):
            raise ValidationError("Post-hoc localization returned invalid payload")
        return result

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
            for field in ("model_output", "reasoning", "raw_reasoning"):
                if row.get(field):
                    sample[field] = str(row[field])[:12000]
            for field in (
                "procedure_mode", "procedure_task_id", "parsed_actions",
                "canonical_remaining_ids", "parsed_remaining_ids", "parse_valid",
                "parse_source", "parse_errors", "observed_stage", "observed_state",
                "persistent_stage", "persistent_state", "confirmed_stage",
                "transition_support", "state_update", "state_updates",
                "transition_event", "transition_events", "tracker_window_counts",
                "task_history_text", "next_task_history_text",
            ):
                if field in row:
                    sample[field] = row[field]
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
        relative_values = raw.get("relative_values")
        if not isinstance(relative_values, list):
            relative_values = []
            # Be tolerant of a raw file produced by an intermediate wrapper
            # that retained only the complete per-prefix rows.  The Review
            # timeline uses the last native relative slot for each prefix.
            for row in raw.get("relative_values_by_prefix") or []:
                if not isinstance(row, list):
                    row = [row]
                selected = None
                for item in reversed(row):
                    selected = self._number(item)
                    if selected is not None:
                        break
                relative_values.append(selected)
        relative_indices = raw.get("relative_sampled_indices")
        if not isinstance(relative_indices, list):
            relative_indices = list(indices) if relative_values else []
        parsed = raw.get("parsed_analysis")
        analysis_text = raw.get("analysis_text")
        samples = []
        raw_frames = []
        relative_sample_count = 0
        for position, (index, value) in enumerate(zip(indices, values)):
            number = self._number(value)
            if number is None:
                continue
            raw_frame = int(index)
            frame = min(max(raw_frame, 0), total_frames - 1)
            signals = {"value": number}
            relative_number = (
                self._number(relative_values[position])
                if position < len(relative_values)
                else None
            )
            if relative_number is not None:
                signals["relative_value"] = relative_number
                relative_sample_count += 1
            samples.append({
                "frame": frame,
                "raw_frame": raw_frame,
                "signals": signals,
                "analysis_text": str(analysis_text)[:12000] if analysis_text else None,
                "parsed_analysis": parsed,
            })
            raw_frames.append(raw_frame)
        relative_output = {
            "available": relative_sample_count > 0,
            "signal": "relative_value",
            "official_head": "<relative_value>",
            "semantics": "signed temporal displacement between consecutive sampled observations",
            "aligned_to": "sampled_indices; last native relative slot per prefix",
            "complete_slot_rows_preserved": isinstance(raw.get("relative_values_by_prefix"), list),
            "prefix_count": len(relative_values),
            "aligned_prefix_count": len(relative_indices),
            "finite_aligned_sample_count": relative_sample_count,
        }
        return self._pack(
            "rynnvalue",
            run_summary,
            samples,
            [path],
            raw_frames,
            {
                "_total_frames": int(rollout["total_frames"]),
                "kind": "value_and_relative_heads_and_analysis",
                "relative_output": relative_output,
            },
        )

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
        extra: dict[str, Any] = {
            "_total_frames": total_frames,
            "kind": "model_scores_and_progress",
        }
        posthoc_localizations = self._posthoc_localization_history(
            run_path,
            str(rollout["id"]),
        )
        if posthoc_localizations:
            extra["posthoc_localizations"] = posthoc_localizations
            latest_posthoc = dict(posthoc_localizations[0])
            predicted_frame = latest_posthoc.get("predicted_frame")
            if predicted_frame is not None:
                latest_posthoc["predicted_frame"] = min(
                    max(int(predicted_frame), 0),
                    total_frames - 1,
                )
                extra["localization_prediction"] = latest_posthoc

        localization = result.get("localization_prediction")
        if (
            "localization_prediction" not in extra
            and isinstance(localization, dict)
        ):
            localization = dict(localization)
            output_value = localization.get("output_path")
            if output_value:
                localization_path = self._project_path(str(output_value))
                if localization_path.is_file():
                    files.append(localization_path)
                    try:
                        detailed = json.loads(
                            localization_path.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError):
                        detailed = None
                    if isinstance(detailed, dict):
                        localization.update({
                            key: detailed.get(key)
                            for key in (
                                "predicted_index",
                                "predicted_frame",
                                "predicted_logit",
                                "predicted_sigmoid",
                                "checkpoint",
                                "checkpoint_stage",
                                "checkpoint_config_id",
                                "checkpoint_repeat",
                                "checkpoint_seed",
                                "frame_count",
                            )
                            if detailed.get(key) is not None
                        })
            predicted_frame = localization.get("predicted_frame")
            if predicted_frame is not None:
                localization["predicted_frame"] = min(
                    max(int(predicted_frame), 0),
                    total_frames - 1,
                )
                extra["localization_prediction"] = localization
        return self._pack(
            "robo_dopamine",
            run_summary,
            samples,
            files,
            raw_frames,
            extra,
        )

    
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

    def _manifest_for_condition(self, condition: str) -> Path:
        condition = validate_instruction_condition(condition)
        if condition == "full_instruction":
            return self.manifest_path
        if not self.variant_manifest_path.is_file():
            raise ValidationError(
                "Instruction-variant manifest is unavailable; run "
                "tools/prepare_libero10_instruction_variants.py first"
            )
        return self.variant_manifest_path

    def _condition_records(self, condition: str, scope: str) -> list[dict[str, Any]]:
        condition = validate_instruction_condition(condition)
        source_records = select_scope_records(self._manifest_records(), scope)
        if condition == "full_instruction":
            return source_records

        variant_records = load_manifest_records(self._manifest_for_condition(condition))
        by_source: dict[str, dict[str, Any]] = {}
        for record in variant_records:
            row_condition = record.get("instruction_variant") or record.get("condition")
            if row_condition != condition:
                continue
            source_id = record.get("source_rollout_id") or record.get("source_id")
            if isinstance(source_id, str):
                by_source[source_id] = record
        return [
            by_source[record["id"]]
            for record in source_records
            if record["id"] in by_source
        ]

    @staticmethod
    def _validate_result_filter(value: Any) -> str:
        result_filter = str(value or "all").strip()
        if result_filter not in BASELINE_RESULT_FILTERS:
            raise ValidationError(
                "result_filter must be one of: " + ", ".join(sorted(BASELINE_RESULT_FILTERS))
            )
        return result_filter

    @staticmethod
    def _source_rollout_id(record: dict[str, Any], condition: str) -> str:
        if condition == "full_instruction":
            return str(record["id"])
        return str(
            record.get("source_rollout_id")
            or record.get("source_id")
            or record["id"]
        )

    def _complete_annotation_records(
        self,
        records: list[dict[str, Any]],
        condition: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Keep only source rollouts whose annotation review_status is complete."""
        condition = validate_instruction_condition(condition)
        if self.annotation_store is None:
            raise ValidationError("Annotation store is unavailable for result filtering")
        complete: list[dict[str, Any]] = []
        incomplete_source_ids: list[str] = []
        for record in records:
            source_id = self._source_rollout_id(record, condition)
            annotation = self.annotation_store.read(source_id)
            if annotation and annotation.get("review_status") == "complete":
                complete.append(record)
            else:
                incomplete_source_ids.append(source_id)
        return complete, incomplete_source_ids

    def _valid_result_rollout_ids(
        self,
        baseline: str,
        condition: str,
        records: list[dict[str, Any]],
    ) -> set[str]:
        """Return record IDs with at least one parseable completed baseline output."""
        if baseline not in BASELINE_METHODS:
            raise ValidationError("Invalid baseline method")
        condition = validate_instruction_condition(condition)
        allowed_conditions = (
            {condition, "unknown"} if condition == "full_instruction" else {condition}
        )
        remaining = {str(record["id"]): record for record in records}
        valid: set[str] = set()
        for run_path, metadata in self._run_candidates(baseline):
            if self._run_instruction_condition(metadata) not in allowed_conditions:
                continue
            completed_ids = run_rollout_ids(run_path)
            if not completed_ids:
                continue
            run_summary = self._run_summary(run_path, metadata)
            for rollout_id in list(remaining):
                if rollout_id not in completed_ids:
                    continue
                record = remaining[rollout_id]
                try:
                    self._read_method(
                        baseline,
                        run_path,
                        record,
                        dict(run_summary),
                    )
                except (
                    FileNotFoundError,
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    json.JSONDecodeError,
                ):
                    continue
                valid.add(rollout_id)
                remaining.pop(rollout_id, None)
            if not remaining:
                break
        return valid

    def result_coverage(
        self,
        baseline: Any,
        scope: Any = "libero_10",
        condition: Any = "full_instruction",
    ) -> dict[str, Any]:
        baseline = str(baseline or "")
        if baseline not in BASELINE_METHODS:
            raise ValidationError("Invalid baseline method")
        scope = validate_run_scope(scope)
        condition = validate_instruction_condition(condition)
        scope_records = self._condition_records(condition, scope)
        records, incomplete_source_ids = self._complete_annotation_records(
            scope_records,
            condition,
        )
        valid_ids = self._valid_result_rollout_ids(baseline, condition, records)
        valid_source_ids = [
            self._source_rollout_id(record, condition)
            for record in records
            if str(record["id"]) in valid_ids
        ]
        missing_source_ids = [
            self._source_rollout_id(record, condition)
            for record in records
            if str(record["id"]) not in valid_ids
        ]
        return {
            "baseline": baseline,
            "scope": scope,
            "condition": condition,
            "scope_rollouts": len(scope_records),
            "matched_rollouts": len(records),
            "complete_annotation_rollouts": len(records),
            "incomplete_annotation_rollouts": len(incomplete_source_ids),
            "incomplete_source_rollout_ids": incomplete_source_ids,
            "valid_result_rollouts": len(valid_source_ids),
            "missing_valid_result_rollouts": len(missing_source_ids),
            "valid_source_rollout_ids": valid_source_ids,
            "missing_source_rollout_ids": missing_source_ids,
        }

    def _variant_record_for_source(
        self,
        source_rollout_id: str,
        condition: str,
    ) -> dict[str, Any]:
        records = self._condition_records(condition, "all")
        for record in records:
            source_id = record.get("source_rollout_id") or record.get("source_id")
            if source_id == source_rollout_id:
                return record
        raise ValidationError(
            f"No prepared {INSTRUCTION_VARIANT_LABELS[condition]} variant exists "
            f"for rollout {source_rollout_id}"
        )

    ROBO_HOP_SIGNAL_MODES = ("incremental", "forward", "backward", "fused")

    def _resolve_robo_signal_prediction(
        self,
        run_path: Path,
        rollout_id: str,
        signal_mode: str,
    ) -> Path | None:
        """Resolve one saved Robo-Dopamine hop source without changing its semantics."""
        if signal_mode not in self.ROBO_HOP_SIGNAL_MODES:
            return None
        worker_dir = run_path / "raw" / rollout_id
        result_path = worker_dir / "worker_result.json"
        if not result_path.is_file():
            return None
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(result, dict):
            return None

        recorded: Any = None
        perspectives = result.get("perspective_outputs")
        if signal_mode != "fused" and isinstance(perspectives, dict):
            mode_output = perspectives.get(signal_mode)
            if isinstance(mode_output, dict):
                recorded = mode_output.get("raw_model_output")

        fusion = result.get("fusion")
        if (
            recorded in (None, "")
            and signal_mode != "fused"
            and isinstance(fusion, dict)
        ):
            source_paths = fusion.get("source_prediction_paths")
            if isinstance(source_paths, dict):
                recorded = source_paths.get(signal_mode)

        eval_mode = str(result.get("eval_mode") or "").lower()
        eval_modes = [
            str(value).lower()
            for value in (result.get("eval_modes") or [])
        ]
        if (
            recorded in (None, "")
            and signal_mode != "fused"
            and (eval_mode == signal_mode or eval_modes == [signal_mode])
        ):
            recorded = result.get("raw_model_output")

        if signal_mode == "fused" and recorded in (None, ""):
            if result.get("fused_model_output"):
                recorded = result["fused_model_output"]
            elif isinstance(fusion, dict) and fusion.get("output_path"):
                recorded = fusion["output_path"]
            elif (
                result.get("multi_perspective")
                or eval_mode == "fused"
                or set(eval_modes) >= {"incremental", "forward", "backward"}
            ) and result.get("raw_model_output"):
                recorded = result["raw_model_output"]

        if recorded in (None, ""):
            metadata_path = worker_dir / "multi_perspective" / "metadata.json"
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metadata = None
                if isinstance(metadata, dict):
                    if signal_mode == "fused":
                        recorded = metadata.get("fused_path")
                    else:
                        prediction_paths = metadata.get("prediction_paths")
                        if isinstance(prediction_paths, dict):
                            recorded = prediction_paths.get(signal_mode)

        candidates: list[Path] = []
        if recorded not in (None, ""):
            path = Path(str(recorded)).expanduser()
            if path.is_absolute():
                candidates.append(path)
                parts = path.parts
                for anchor in ("outputs", "datasets", "annotations", "tools", "repos"):
                    if anchor in parts:
                        index = parts.index(anchor)
                        candidates.append(self.project_root.joinpath(*parts[index:]))
                        break
            else:
                candidates.extend(
                    (
                        worker_dir / path,
                        run_path / path,
                        self.project_root / path,
                    )
                )

        candidates.extend(
            path
            for path in worker_dir.rglob("pred_vllm.json")
            if signal_mode in path.parts
        )

        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                resolved.relative_to(self.project_root)
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                return resolved
        return None

    def _robo_run_signal_ids(
        self,
        run_path: Path,
        run_ids: set[str],
        signal_mode: str,
    ) -> set[str]:
        return {
            rollout_id
            for rollout_id in run_ids
            if self._resolve_robo_signal_prediction(
                run_path, rollout_id, signal_mode
            ) is not None
        }

    def _scan_robo_run_four_signal_ids(
        self,
        run_path: Path,
        run_ids: set[str],
    ) -> tuple[set[str], dict[str, set[str]]]:
        by_mode = {
            mode: self._robo_run_signal_ids(run_path, run_ids, mode)
            for mode in self.ROBO_HOP_SIGNAL_MODES
        }
        common = set(run_ids)
        for mode in self.ROBO_HOP_SIGNAL_MODES:
            common.intersection_update(by_mode[mode])
        return common, by_mode

    def _run_inventory(
        self,
        run_path: Path,
        method: str,
    ) -> tuple[set[str], dict[str, set[str]], set[str]]:
        """Read run membership from cache; probe only fused Robo output once."""
        cached = self.run_index.cached_rollout_details(run_path)
        if cached is not None:
            run_ids = set(cached["rollout_ids"])
            if method != "robo_dopamine":
                return run_ids, {}, set()
            raw_signal_ids = cached.get("robo_signal_ids")
            if isinstance(raw_signal_ids, dict) and "fused" in raw_signal_ids:
                fused_ids = {
                    str(value)
                    for value in raw_signal_ids.get("fused", [])
                    if isinstance(value, str)
                }
                return run_ids, {"fused": fused_ids}, fused_ids

        run_ids = run_rollout_ids(run_path)
        by_mode: dict[str, set[str]] = {}
        fused_ids: set[str] = set()
        if method == "robo_dopamine":
            fused_ids = self._robo_run_signal_ids(
                run_path, run_ids, "fused"
            )
            by_mode = {"fused": fused_ids}
        self.run_index.store_rollout_details(
            run_path,
            run_ids,
            by_mode if method == "robo_dopamine" else None,
        )
        return run_ids, by_mode, fused_ids

    def _robo_run_four_signal_ids(
        self,
        run_path: Path,
        run_ids: set[str],
    ) -> tuple[set[str], dict[str, set[str]]]:
        """Legacy explicit four-signal probe; not used by fused-only catalog."""
        return self._scan_robo_run_four_signal_ids(run_path, run_ids)

    def _robo_run_incremental_ids(
        self,
        run_path: Path,
        run_ids: set[str],
    ) -> set[str]:
        """Legacy explicit incremental probe; not used by fused-only catalog."""
        return self._robo_run_signal_ids(run_path, run_ids, "incremental")

    def list_runs(
        self,
        scope: Any = "libero_10",
        condition: Any = "full_instruction",
    ) -> list[dict[str, Any]]:
        scope = validate_run_scope(scope)
        condition = validate_instruction_condition(condition)
        selected_ids = {
            record["id"] for record in self._condition_records(condition, scope)
        }
        variant_by_id: dict[str, dict[str, Any]] = {}
        if condition != "full_instruction":
            variant_by_id = {
                record["id"]: record
                for record in load_manifest_records(self._manifest_for_condition(condition))
            }
        summaries = []
        for method in BASELINE_METHODS:
            for run_path, metadata in self._run_candidates(method):
                run_condition = self._run_instruction_condition(metadata)
                if condition == "full_instruction":
                    if run_condition not in {"full_instruction", "unknown"}:
                        continue
                elif run_condition != condition:
                    continue
                run_ids, signal_ids_by_mode, fused_inventory_ids = self._run_inventory(
                    run_path, method
                )
                missing = selected_ids - run_ids if run_ids else selected_ids
                source_ids = {
                    (
                        variant_by_id[rollout_id].get("source_rollout_id")
                        or variant_by_id[rollout_id].get("source_id")
                        or rollout_id
                    )
                    for rollout_id in run_ids
                    if rollout_id in variant_by_id
                }
                if condition == "full_instruction":
                    source_ids = set(run_ids)
                summary = self._run_summary(run_path, metadata)
                fused_ids: set[str] = set()
                if method == "robo_dopamine":
                    fused_ids = signal_ids_by_mode.get(
                        "fused", fused_inventory_ids
                    )
                summary.update({
                    "created_at": metadata.get("created_at"),
                    "manifest_sha256": metadata.get("manifest_sha256"),
                    "run_rollout_count": len(run_ids),
                    "run_rollout_ids": sorted(run_ids),
                    "run_source_rollout_ids": sorted(source_ids),
                    "selected_scope_rollouts": len(selected_ids),
                    "compatible": bool(selected_ids)
                    and not missing
                    and summary["selected_rollouts"] >= len(selected_ids),
                    "partial_compatible": bool(selected_ids.intersection(run_ids)),
                    "missing_rollouts": len(missing),
                    "scope": scope,
                    "instruction_condition": run_condition,
                    "fused_rollout_count": (
                        len(fused_ids) if method == "robo_dopamine" else None
                    ),
                    "fused_scope_rollout_count": (
                        len(selected_ids.intersection(fused_ids))
                        if method == "robo_dopamine"
                        else None
                    ),
                    "fused_missing_rollouts": (
                        len(selected_ids - fused_ids)
                        if method == "robo_dopamine"
                        else None
                    ),
                    "fused_scope_coverage": (
                        (
                            len(selected_ids.intersection(fused_ids))
                            / len(selected_ids)
                        )
                        if method == "robo_dopamine" and selected_ids
                        else None
                    ),
                    "fused_compatible": (
                        bool(selected_ids)
                        and selected_ids.issubset(fused_ids)
                        if method == "robo_dopamine"
                        else None
                    ),
                    "hop_signal_rollout_counts": (
                        {"fused": len(fused_ids)}
                        if method == "robo_dopamine"
                        else None
                    ),
                })
                summaries.append(summary)
        summaries.sort(
            key=lambda item: (
                str(item.get("baseline")),
                str(item.get("completed_at") or item.get("created_at") or ""),
            ),
            reverse=True,
        )
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
        for name in ("dtype", "robo_eval_mode", "procvlm_procedure_mode"):
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
        elif scope in {"libero_10", "libero_spatial"}:
            command.extend(["--partition", "natural_observation", "--task-suite", scope])
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
