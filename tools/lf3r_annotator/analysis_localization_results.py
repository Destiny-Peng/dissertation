"""Localization presets and result inspection."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

from backend_core import ValidationError


class AnalysisLocalizationResultsMixin:
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

