"""Execute arbitrary Localization Lab experiment specs."""

from __future__ import annotations

import copy
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _best_repeat_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the descriptively best observed repeat for one configuration.

    Repeat ranking follows the Localization Lab's default localization metric:
    maximize test in-interval rate, then minimize test MAE and MSE.  The repeat
    index is the final deterministic tie-breaker.
    """
    eligible = [
        row for row in rows
        if row.get("repeat") is not None
        and row.get("in_interval_rate") is not None
        and row.get("mae_samples") is not None
        and row.get("mse_samples") is not None
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            -float(row["in_interval_rate"]),
            float(row["mae_samples"]),
            float(row["mse_samples"]),
            int(row["repeat"]),
        ),
    )


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
    checkpoint_root: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
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
    all_failure_predictions: list[dict[str, Any]] = []
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
        forced_train_ids = [
            rollout_id
            for rollout_id in data_config.get("force_train_rollout_ids", [])
            if rollout_id in base_failure
        ]
        if forced_train_ids:
            split = core.force_train_rollouts(split, forced_train_ids)
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
        checkpoint_dir = checkpoint_root / config_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"repeat_{repeat:02d}.pt"
        checkpoint_rel = checkpoint_path.relative_to(checkpoint_root.parent.parent).as_posix()
        torch.save(
            {
                "schema_version": 1,
                "stage": stage_name,
                "config_id": config_id,
                "repeat": repeat,
                "seed": seed,
                "config": copy.deepcopy(dict(config)),
                "split": copy.deepcopy(split),
                "failure_train_ids": list(failure_train_ids),
                "success_train_ids": list(success_ids),
                "forced_train_ids": list(split.get("forced_train", [])),
                "normalization_mean": torch.from_numpy(np.asarray(mean, dtype=np.float32).copy()),
                "normalization_std": torch.from_numpy(np.asarray(std, dtype=np.float32).copy()),
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "train_meta": copy.deepcopy(train_meta),
            },
            checkpoint_path,
        )

        all_rollout_ids = sorted(failure_dataset)
        logits_by_rollout = core.batched_logits(
            model,
            failure_dataset,
            all_rollout_ids,
            mean,
            std,
            device,
            int(training.get("batch_size", 32)),
        )
        train_set = set(split["train"])
        val_set = set(split["val"])
        test_set = set(split["test"])
        repeat_all_predictions: list[dict[str, Any]] = []
        for rollout_id in all_rollout_ids:
            logits = logits_by_rollout[rollout_id]
            prediction = int(torch.argmax(logits).item())
            if rollout_id in train_set:
                split_role = "train"
            elif rollout_id in val_set:
                split_role = "val"
            elif rollout_id in test_set:
                split_role = "test"
            else:
                split_role = "unknown"
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
                "split_role": split_role,
                "seen_in_train": split_role == "train",
                "forced_into_train": rollout_id in set(split.get("forced_train", [])),
                "checkpoint": checkpoint_rel,
            })
            repeat_all_predictions.append(row)
        repeat_predictions = [
            row for row in repeat_all_predictions if row["split_role"] == "test"
        ]
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
            "seed": seed,
            "checkpoint": checkpoint_rel,
        })
        per_repeat.append(metric_row)
        predictions.extend(repeat_predictions)
        all_failure_predictions.extend(repeat_all_predictions)
        records.append({
            "stage": stage_name,
            "config_id": config_id,
            "repeat": repeat,
            "split": split,
            "failure_train_ids": failure_train_ids,
            "success_train_ids": success_ids,
            "forced_train_ids": list(split.get("forced_train", [])),
            "checkpoint": checkpoint_rel,
        })
    summary = _aggregate(per_repeat)
    best_repeat = _best_repeat_row(per_repeat)
    if best_repeat is not None:
        summary.update({
            "best_repeat": int(best_repeat["repeat"]),
            "best_repeat_seed": int(best_repeat["seed"]),
            "best_repeat_checkpoint": best_repeat["checkpoint"],
            "best_repeat_test_n": int(best_repeat.get("n") or 0),
            "best_repeat_in_interval_rate": best_repeat.get("in_interval_rate"),
            "best_repeat_first_event_in_interval_rate": best_repeat.get(
                "first_event_in_interval_rate"
            ),
            "best_repeat_within_3": best_repeat.get("within_3"),
            "best_repeat_median_absolute_interval_error_samples": best_repeat.get(
                "median_absolute_interval_error_samples"
            ),
            "best_repeat_mae_samples": best_repeat.get("mae_samples"),
            "best_repeat_mse_samples": best_repeat.get("mse_samples"),
            "best_repeat_before_interval_rate": best_repeat.get(
                "before_interval_rate"
            ),
            "best_repeat_after_interval_rate": best_repeat.get(
                "after_interval_rate"
            ),
            "best_repeat_selection": "test_in_interval_desc_mae_mse_asc",
        })
    flat: dict[str, Any] = {}
    _flatten("", config, flat)
    summary.update({"stage": stage_name, "config_id": config_id, **flat})
    return summary, predictions, all_failure_predictions, records


def _run_configuration_concurrent(
    *,
    config: Mapping[str, Any],
    config_id: str,
    stage_name: str,
    repeats: int,
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
    checkpoint_root: Path,
    use_cuda_stream: bool,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Run one configuration, optionally on its own CUDA stream.

    Worker threads share one CUDA context, so this avoids the large overhead of
    spawning one process/context per tiny BiLSTM while still allowing kernels
    from independent configurations to overlap.
    """
    training = config["training"]
    device = core.resolve_device(str(training.get("device", "auto")))
    if not use_cuda_stream or device.type != "cuda":
        return _run_configuration(
            config=config,
            config_id=config_id,
            stage_name=stage_name,
            repeats=repeats,
            signals=signals,
            events=events,
            no_event_failures=no_event_failures,
            clean_rollouts=clean_rollouts,
            checkpoint_root=checkpoint_root,
        )

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.device(device), torch.cuda.stream(stream):
        result = _run_configuration(
            config=config,
            config_id=config_id,
            stage_name=stage_name,
            repeats=repeats,
            signals=signals,
            events=events,
            no_event_failures=no_event_failures,
            clean_rollouts=clean_rollouts,
            checkpoint_root=checkpoint_root,
        )
    stream.synchronize()
    return result


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
    all_failure_predictions: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    stage_manifest: list[dict[str, Any]] = []
    inherited = copy.deepcopy(normalized["base"])

    stage_specs = normalized["stages"] or [{
        "name": "main",
        "base": {},
        "sweep": normalized["sweep"],
        "variants": normalized["variants"],
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
        configurations = specs.expand(
            stage_base,
            stage.get("sweep", []),
            stage.get("variants", []),
        )
        requested_workers = int(
            stage_base.get("training", {}).get("parallel_workers", 4)
        )
        effective_workers = max(1, min(requested_workers, len(configurations)))
        log(
            f"stage={stage_name} configurations={len(configurations)} "
            f"repeats={normalized['repeats']} workers={effective_workers}"
        )

        configs_by_id: dict[str, dict[str, Any]] = {}
        jobs: list[tuple[int, str, dict[str, Any]]] = []
        for config_index, config in enumerate(configurations):
            specs.validate_config(config)
            config_id = f"s{stage_index + 1:02d}_c{config_index + 1:03d}"
            configs_by_id[config_id] = config
            jobs.append((config_index, config_id, config))

        completed: list[
            tuple[
                dict[str, Any],
                list[dict[str, Any]],
                list[dict[str, Any]],
                list[dict[str, Any]],
            ] | None
        ] = [None] * len(jobs)

        if effective_workers == 1:
            for config_index, config_id, config in jobs:
                log(f"stage={stage_name} start={config_id}")
                completed[config_index] = _run_configuration_concurrent(
                    config=config,
                    config_id=config_id,
                    stage_name=stage_name,
                    repeats=int(normalized["repeats"]),
                    signals=signals,
                    events=events,
                    no_event_failures=no_event_failures,
                    clean_rollouts=clean_rollouts,
                    checkpoint_root=out / "checkpoints" / stage_name,
                    use_cuda_stream=False,
                )
                log(f"stage={stage_name} done={config_id}")
        else:
            with ThreadPoolExecutor(
                max_workers=effective_workers,
                thread_name_prefix=f"localization-{stage_index + 1}",
            ) as executor:
                futures = {
                    executor.submit(
                        _run_configuration_concurrent,
                        config=config,
                        config_id=config_id,
                        stage_name=stage_name,
                        repeats=int(normalized["repeats"]),
                        signals=signals,
                        events=events,
                        no_event_failures=no_event_failures,
                        clean_rollouts=clean_rollouts,
                        checkpoint_root=out / "checkpoints" / stage_name,
                        use_cuda_stream=True,
                    ): (config_index, config_id)
                    for config_index, config_id, config in jobs
                }
                for future in as_completed(futures):
                    config_index, config_id = futures[future]
                    completed[config_index] = future.result()
                    log(f"stage={stage_name} done={config_id}")

        stage_rows: list[dict[str, Any]] = []
        for result in completed:
            if result is None:
                raise RuntimeError(f"stage {stage_name} has an incomplete worker result")
            summary, predictions, challenge_predictions, records = result
            stage_rows.append(summary)
            all_summary.append(summary)
            all_predictions.extend(predictions)
            all_failure_predictions.extend(challenge_predictions)
            all_records.extend(records)
        selector = stage.get("select") or {}
        best_row = min(stage_rows, key=lambda row: _selector_key(row, selector))
        best_config = configs_by_id[str(best_row["config_id"])]
        inherited = copy.deepcopy(best_config)
        stage_manifest.append({
            "stage": stage_name,
            "configuration_count": len(configurations),
            "parallel_workers": effective_workers,
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
    _write_csv(out / "all_failure_predictions.csv", all_failure_predictions)
    metadata = {
        "schema_version": 1,
        "analysis": "robo_localization_experiment",
        "name": normalized["name"],
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_root": project_relative(source_root),
        "selection_mode": "latest_usable_fused_per_rollout",
        "configuration_count": len(all_summary),
        "training_run_count": len(all_records),
        "all_failure_prediction_count": len(all_failure_predictions),
        "checkpoint_count": len(all_records),
        "parallel_workers": int(
            normalized["base"].get("training", {}).get("parallel_workers", 4)
        ),
        "provenance": provenance,
        "outputs": [
            "config.json",
            "experiment_manifest.json",
            "training_records.json",
            "summary.csv",
            "per_rollout_predictions.csv",
            "all_failure_predictions.csv",
            "checkpoints/",
        ],
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"complete output={project_relative(out)} configs={len(all_summary)}")
    return out
