"""Execute arbitrary Localization Lab experiment specs."""

from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from robo_incremental_hop.io import (
    build_base_records,
    ensure_within_project,
    load_manifest,
    project_relative,
    resolve_project_path,
)

from . import core, data, losses, metrics, specs, targets


def log(message: str) -> None:
    timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _positive_weight(dataset: Mapping[str, Mapping[str, Any]], rollout_ids: Sequence[str]) -> float:
    support = np.concatenate([
        np.asarray(dataset[rollout_id]["hard_support"], dtype=np.float32)
        for rollout_id in rollout_ids
    ])
    positive = float(np.sum(support > 0.5))
    negative = float(np.sum(support <= 0.5))
    if positive <= 0:
        raise ValueError("failure training split contains no positive target support")
    return min(20.0, max(1.0, negative / positive))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, output)
    else:
        output[prefix] = value


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "in_interval_rate",
        "first_event_in_interval_rate",
        "within_1",
        "within_3",
        "within_5",
        "before_interval_rate",
        "after_interval_rate",
        "median_absolute_interval_error_samples",
        "mae_samples",
        "mse_samples",
    ]
    result: dict[str, Any] = {"repeat_n": len(rows)}
    for name in metric_names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        result[f"{name}_mean"] = float(np.mean(values)) if values else None
        result[f"{name}_variance"] = float(np.var(values)) if values else None
    return result


def _selector_key(row: Mapping[str, Any], selector: Mapping[str, Any]) -> tuple[float, ...]:
    fields = [{
        "metric": selector.get("metric", "in_interval_rate_mean"),
        "mode": selector.get("mode", "max"),
    }, *selector.get("tie_breakers", [])]
    key = []
    for item in fields:
        metric = str(item.get("metric"))
        mode = str(item.get("mode", "max"))
        value = row.get(metric)
        number = float(value) if value is not None else (-math.inf if mode == "max" else math.inf)
        key.append(-number if mode == "max" else number)
    return tuple(key)


def _run_configuration(
    *,
    config: Mapping[str, Any],
    config_id: str,
    stage_name: str,
    repeats: int,
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    target_config = config["target"]
    training = config["training"]
    data_config = config["data"]
    loss_config = config["loss"]
    base_failure = data.build_failure_dataset(
        signals,
        events,
        no_event_failures,
        tau_event=float(target_config.get("tau_event", 20.0)),
    )
    if len(base_failure) < 4:
        raise ValueError(f"configuration {config_id} has fewer than 4 eligible failure rollouts")
    failure_dataset = targets.apply_labels(base_failure, target_config)
    success_dataset = data.build_success_dataset(signals, clean_rollouts)
    device = core.resolve_device(str(training.get("device", "auto")))

    per_repeat: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    seed0 = int(training.get("seed", 17))
    for repeat in range(repeats):
        seed = seed0 + repeat
        split = core.rollout_split(
            base_failure,
            seed=seed,
            train_fraction=float(training.get("train_fraction", 0.70)),
            val_fraction=float(training.get("val_fraction", 0.15)),
        )
        failure_train_ids = list(split["train"])
        mean, std = core.standardization_stats(base_failure, failure_train_ids)
        pos_weight = _positive_weight(base_failure, failure_train_ids)
        success_ids: list[str] = []
        if str(data_config.get("population", "failure_only")) == "failure_success":
            success_ids = data.select_success_rollouts(
                success_dataset,
                base_failure,
                failure_train_ids,
                ratio=float(data_config.get("success_ratio", 0.0)),
                seed=seed * 100 + 31,
            )
        combined = {**failure_dataset}
        for rollout_id in success_ids:
            combined[rollout_id] = dict(success_dataset[rollout_id])
        train_ids = [*failure_train_ids, *success_ids]

        def row_loss(
            logits: torch.Tensor,
            row: Mapping[str, Any],
            labels: torch.Tensor,
            pos_weight_tensor: torch.Tensor,
        ) -> torch.Tensor:
            return losses.loss_value(
                logits,
                row,
                labels,
                config=loss_config,
                pos_weight=pos_weight_tensor,
            )

        model, train_meta = core.train_bilstm(
            dataset=combined,
            train_ids=train_ids,
            val_ids=split["val"],
            mean=mean,
            std=std,
            hidden=int(config["model"].get("hidden", 16)),
            pos_weight=pos_weight,
            seed=seed,
            epochs=int(training.get("epochs", 300)),
            patience=int(training.get("patience", 35)),
            learning_rate=float(training.get("learning_rate", 0.003)),
            weight_decay=float(training.get("weight_decay", 0.0001)),
            grad_clip=float(training.get("grad_clip", 5.0)),
            batch_size=int(training.get("batch_size", 32)),
            device=device,
            row_loss_fn=row_loss,
            progress_label=f"{stage_name}/{config_id}/repeat{repeat}",
            logger=log,
        )
        logits_by_rollout = core.batched_logits(
            model,
            failure_dataset,
            split["test"],
            mean,
            std,
            device,
            int(training.get("batch_size", 32)),
        )
        repeat_predictions = []
        for rollout_id in split["test"]:
            logits = logits_by_rollout[rollout_id]
            prediction = int(torch.argmax(logits).item())
            row = metrics.prediction_row(
                failure_dataset,
                rollout_id,
                prediction,
                float(logits[prediction].item()),
            )
            row.update({
                "stage": stage_name,
                "config_id": config_id,
                "repeat": repeat,
            })
            repeat_predictions.append(row)
        metric_row = metrics.summary(repeat_predictions)
        metric_row.update({
            "stage": stage_name,
            "config_id": config_id,
            "repeat": repeat,
            "failure_train_n": len(failure_train_ids),
            "success_train_n": len(success_ids),
            "val_n": len(split["val"]),
            "test_n": len(split["test"]),
            "best_epoch": train_meta["best_epoch"],
            "best_val_loss": train_meta["best_val_loss"],
            "effective_train_batch_size": train_meta["effective_train_batch_size"],
            "optimizer_steps_per_epoch": train_meta["optimizer_steps_per_epoch"],
        })
        per_repeat.append(metric_row)
        predictions.extend(repeat_predictions)
        records.append({
            "stage": stage_name,
            "config_id": config_id,
            "repeat": repeat,
            "split": split,
            "failure_train_ids": failure_train_ids,
            "success_train_ids": success_ids,
        })
    summary = _aggregate(per_repeat)
    flat: dict[str, Any] = {}
    _flatten("", config, flat)
    summary.update({"stage": stage_name, "config_id": config_id, **flat})
    return summary, predictions, records


def run_spec(
    *,
    spec: Mapping[str, Any],
    run_pool_root: str | Path,
    manifest_path: str | Path,
    annotation_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    normalized = specs.normalize_spec(spec)
    source_root = ensure_within_project(resolve_project_path(run_pool_root), "run pool root")
    manifest = ensure_within_project(resolve_project_path(manifest_path), "manifest")
    annotations = ensure_within_project(resolve_project_path(annotation_dir), "annotation directory")
    out = ensure_within_project(resolve_project_path(output_dir), "output directory")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    signals, events, no_event_failures, clean_rollouts, provenance = build_base_records(
        source_root,
        load_manifest(manifest),
        annotations,
        allowed_rollout_ids=None,
        signal_mode="fused",
        latest_per_rollout=True,
    )
    if not signals:
        raise ValueError("no usable fused Robo-Dopamine signals were found")

    all_summary: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    stage_manifest: list[dict[str, Any]] = []
    inherited = copy.deepcopy(normalized["base"])

    stage_specs = normalized["stages"] or [{
        "name": "main",
        "base": {},
        "sweep": normalized["sweep"],
        "select": {
            "metric": "in_interval_rate_mean",
            "mode": "max",
            "tie_breakers": [
                {"metric": "mae_samples_mean", "mode": "min"},
                {"metric": "mse_samples_mean", "mode": "min"},
            ],
        },
    }]

    for stage_index, stage in enumerate(stage_specs):
        stage_name = str(stage["name"])
        stage_base = specs.deep_merge(inherited, stage.get("base", {}))
        configurations = specs.expand(stage_base, stage.get("sweep", []))
        log(f"stage={stage_name} configurations={len(configurations)} repeats={normalized['repeats']}")
        stage_rows = []
        configs_by_id: dict[str, dict[str, Any]] = {}
        for config_index, config in enumerate(configurations):
            specs.validate_config(config)
            config_id = f"s{stage_index + 1:02d}_c{config_index + 1:03d}"
            configs_by_id[config_id] = config
            summary, predictions, records = _run_configuration(
                config=config,
                config_id=config_id,
                stage_name=stage_name,
                repeats=int(normalized["repeats"]),
                signals=signals,
                events=events,
                no_event_failures=no_event_failures,
                clean_rollouts=clean_rollouts,
            )
            stage_rows.append(summary)
            all_summary.append(summary)
            all_predictions.extend(predictions)
            all_records.extend(records)
        selector = stage.get("select") or {}
        best_row = min(stage_rows, key=lambda row: _selector_key(row, selector))
        best_config = configs_by_id[str(best_row["config_id"])]
        inherited = copy.deepcopy(best_config)
        stage_manifest.append({
            "stage": stage_name,
            "configuration_count": len(configurations),
            "selector": selector,
            "best_config_id": best_row["config_id"],
            "best_config": best_config,
            "best_metrics": {
                key: value for key, value in best_row.items()
                if key.endswith("_mean") or key.endswith("_variance")
            },
        })
        log(f"stage={stage_name} best={best_row['config_id']}")

    (out / "config.json").write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "experiment_manifest.json").write_text(
        json.dumps({"stages": stage_manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "training_records.json").write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(out / "summary.csv", all_summary)
    _write_csv(out / "per_rollout_predictions.csv", all_predictions)
    metadata = {
        "schema_version": 1,
        "analysis": "robo_localization_experiment",
        "name": normalized["name"],
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_root": project_relative(source_root),
        "selection_mode": "latest_usable_fused_per_rollout",
        "configuration_count": len(all_summary),
        "training_run_count": len(all_records),
        "provenance": provenance,
        "outputs": [
            "config.json",
            "experiment_manifest.json",
            "training_records.json",
            "summary.csv",
            "per_rollout_predictions.csv",
        ],
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"complete output={project_relative(out)} configs={len(all_summary)}")
    return out
