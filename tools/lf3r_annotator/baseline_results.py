"""Baseline run discovery, result reading, coverage, and catalogs."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from backend_core import (
    INSTRUCTION_VARIANT_CONDITIONS,
    ValidationError,
    load_manifest_records,
    run_rollout_ids,
    select_scope_records,
    validate_instruction_condition,
    validate_run_scope,
)
from baseline_constants import (
    BASELINE_LABELS,
    BASELINE_METHODS,
    BASELINE_RESULT_FILTERS,
    BASELINE_RUN_STATUSES,
    INSTRUCTION_VARIANT_LABELS,
)


class BaselineResultsMixin:
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

