"""Localization experiment presets, artifacts, and challenge-set workflows."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from backend_core import AnalysisEnvironmentError, ValidationError


class AnalysisLocalizationMixin:
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

