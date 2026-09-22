#!/usr/bin/env python3
"""BiLSTM-only failure-localization ablation on saved Robo-Dopamine fused signals.

Measure the effect of clean-success negative-data ratio while keeping the
BiLSTM, loss, failure split, normalization, and evaluation fixed.
Robo-Dopamine is never rerun. Splits are strictly rollout-level.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

print("[BiLSTM] process started; importing PyTorch...", flush=True)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print(f"[BiLSTM] PyTorch import complete: {torch.__version__}", flush=True)
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Run this probe with an LF3R environment that "
        "already provides torch, e.g. conda_envs/LF3R-robo-dopamine/bin/python."
    ) from exc

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from robo_localization_head import core as localization_core

# Compatibility names used by existing tests and downstream imports.
_onset_anchor = localization_core.onset_anchor
_sequence = localization_core.sequence_from_signal
rollout_split = localization_core.rollout_split
standardization_stats = localization_core.standardization_stats
TinyBiLSTM = localization_core.TinyBiLSTM
resolve_device = localization_core.resolve_device

from robo_incremental_hop.io import (
    PROJECT_ROOT,
    build_base_records,
    ensure_within_project,
    load_manifest,
    project_relative,
    resolve_project_path,
)

DEFAULT_MANIFEST = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "annotations/failure_annotations/v1/records"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/robo_dopamine_localization_head"
MODEL_NAMES = ("tiny_bilstm_h16", "tiny_bilstm_h32")
DEFAULT_SUCCESS_RATIOS = (0.5, 1.0, 2.0)


def parse_success_ratios(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
        if not raw_values:
            raise ValueError("--success-ratios must contain at least one positive ratio")
        try:
            parsed = [float(part) for part in raw_values]
        except ValueError as exc:
            raise ValueError(
                "--success-ratios must be a comma-separated list of numbers"
            ) from exc
    else:
        parsed = [float(item) for item in value]

    nonzero: list[float] = []
    seen: set[float] = set()
    for ratio in parsed:
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("--success-ratios values must be finite and > 0")
        if ratio > 20:
            raise ValueError("--success-ratios values must be <= 20")
        if ratio not in seen:
            seen.add(ratio)
            nonzero.append(ratio)
    if len(nonzero) > 16:
        raise ValueError("--success-ratios accepts at most 16 unique values")
    return (0.0, *sorted(nonzero))


def log(message: str) -> None:
    timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


METRIC_NAMES = (
    "in_interval_rate",
    "within_1",
    "within_3",
    "within_5",
    "before_interval_rate",
    "after_interval_rate",
    "median_absolute_interval_error_samples",
    "mae_samples",
    "mse_samples",
)


def _first_primary_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_rollout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in events:
        if str(raw.get("outcome") or "") != "terminal_failure":
            continue
        causal = raw.get("causal_onset_frame")
        observable = raw.get("observable_onset_frame")
        if causal is None or observable is None:
            continue
        causal_i, observable_i = int(causal), int(observable)
        if causal_i > observable_i:
            continue
        row = dict(raw)
        row["causal_onset_frame"] = causal_i
        row["observable_onset_frame"] = observable_i
        row["event_index"] = int(row.get("event_index") or 0)
        row["event_id"] = str(
            row.get("event_id")
            or f"{row['rollout_id']}::event{row['event_index']}"
        )
        by_rollout[str(row["rollout_id"])].append(row)
    return sorted(
        (
            min(
                rows,
                key=lambda row: (
                    int(row["causal_onset_frame"]),
                    int(row["observable_onset_frame"]),
                    int(row["event_index"]),
                ),
            )
            for rows in by_rollout.values()
        ),
        key=lambda row: str(row["rollout_id"]),
    )


def build_failure_dataset(
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for event in _first_primary_events(events):
        rollout_id = str(event["rollout_id"])
        signal = signals.get(rollout_id)
        if signal is None:
            continue
        frames = [int(value) for value in signal["frames"]]
        causal = _onset_anchor(frames, int(event["causal_onset_frame"]))
        observable = _onset_anchor(frames, int(event["observable_onset_frame"]))
        if causal is None or observable is None or causal > observable:
            continue
        labels = np.zeros(len(frames), dtype=np.float32)
        labels[causal : observable + 1] = 1.0
        result[rollout_id] = {
            "rollout_id": rollout_id,
            "kind": "failure",
            "event": event,
            "task_key": str(event.get("task_key") or signal.get("task_key") or ""),
            "task_id": event.get("task_id", signal.get("task_id")),
            "frames": np.asarray(frames, dtype=np.int64),
            "labels": labels,
            "causal_index": causal,
            "observable_index": observable,
            "sequence": _sequence(signal),
        }
    return result


def build_success_dataset(
    signals: Mapping[str, Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for clean in clean_rollouts:
        rollout_id = str(clean["rollout_id"])
        signal = signals.get(rollout_id)
        if signal is None:
            continue
        frames = np.asarray([int(value) for value in signal["frames"]], dtype=np.int64)
        sequence = _sequence(signal)
        result[rollout_id] = {
            "rollout_id": rollout_id,
            "kind": "clean_success",
            "task_key": str(clean.get("task_key") or signal.get("task_key") or ""),
            "task_id": clean.get("task_id", signal.get("task_id")),
            "frames": frames,
            "labels": np.zeros(len(sequence), dtype=np.float32),
            "sequence": sequence,
        }
    return result


def _task(dataset: Mapping[str, Mapping[str, Any]], rollout_id: str) -> str:
    return str(dataset[rollout_id].get("task_key") or "")


def task_splits(
    dataset: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    min_test_rollouts: int,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[str]] = defaultdict(list)
    for rollout_id in dataset:
        by_task[_task(dataset, rollout_id)].append(rollout_id)

    result: list[dict[str, Any]] = []
    for task, test_ids in sorted(by_task.items()):
        if len(test_ids) < min_test_rollouts:
            continue
        remaining = sorted(set(dataset) - set(test_ids))
        if len(remaining) < 5:
            continue
        rng = random.Random(f"{seed}:{task}")
        rng.shuffle(remaining)
        val_n = max(1, min(len(remaining) - 2, int(round(0.20 * len(remaining)))))
        result.append(
            {
                "split_id": f"task_holdout::{task}",
                "kind": "task_held_out",
                "seed": seed,
                "held_out_task": task,
                "train": sorted(remaining[val_n:]),
                "val": sorted(remaining[:val_n]),
                "test": sorted(test_ids),
            }
        )
    return result


def success_ratio_label(ratio: float) -> str:
    return ("success_ratio_" + str(ratio).replace(".", "p")).replace("p0", "")


def build_success_ratio_subsets(
    success_dataset: Mapping[str, Mapping[str, Any]],
    failure_dataset: Mapping[str, Mapping[str, Any]],
    failure_train_ids: Sequence[str],
    *,
    seed: int,
    held_out_task: str | None = None,
    success_ratios: Sequence[float] = (0.0, *DEFAULT_SUCCESS_RATIOS),
) -> tuple[dict[float, list[str]], dict[str, Any]]:
    allowed = [
        rollout_id
        for rollout_id in sorted(success_dataset)
        if held_out_task is None
        or _task(success_dataset, rollout_id) != held_out_task
    ]
    train_tasks = {
        _task(failure_dataset, rollout_id)
        for rollout_id in failure_train_ids
    }
    same_task = [
        rollout_id
        for rollout_id in allowed
        if _task(success_dataset, rollout_id) in train_tasks
    ]
    fallback = [
        rollout_id
        for rollout_id in allowed
        if rollout_id not in set(same_task)
    ]

    rng = random.Random(seed)
    rng.shuffle(same_task)
    rng.shuffle(fallback)
    ordered = same_task + fallback

    subsets: dict[float, list[str]] = {}
    requested_counts: dict[str, int] = {}
    for ratio in success_ratios:
        requested = int(len(failure_train_ids) * ratio)
        requested_counts[str(ratio)] = requested
        subsets[ratio] = ordered[: min(requested, len(ordered))]

    return subsets, {
        "sampling": "nested_same_task_first_random_prefix",
        "seed": seed,
        "failure_train_n": len(failure_train_ids),
        "same_task_available_n": len(same_task),
        "fallback_available_n": len(fallback),
        "total_available_n": len(ordered),
        "requested_success_n_by_ratio": requested_counts,
    }


def positive_weight_from_failures(
    failure_dataset: Mapping[str, Mapping[str, Any]],
    failure_train_ids: Sequence[str],
) -> float:
    labels = np.concatenate(
        [
            np.asarray(failure_dataset[rollout_id]["labels"], dtype=np.float32)
            for rollout_id in failure_train_ids
        ]
    )
    positive = float(np.sum(labels > 0.5))
    negative = float(np.sum(labels <= 0.5))
    if positive <= 0:
        raise ValueError("failure training split contains no positive interval samples")
    return min(20.0, max(1.0, negative / positive))


def train_bilstm(
    *,
    hidden: int,
    failure_dataset: Mapping[str, Mapping[str, Any]],
    success_dataset: Mapping[str, Mapping[str, Any]],
    failure_train_ids: Sequence[str],
    success_train_ids: Sequence[str],
    val_ids: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    pos_weight: float,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    batch_size: int,
    device: torch.device,
    progress_label: str,
) -> tuple[localization_core.TinyBiLSTM, dict[str, Any]]:
    """Run success-negative training on the shared minibatch trainer."""
    overlap = set(failure_dataset) & set(success_dataset)
    if overlap:
        raise ValueError(f"failure/success rollout ID overlap: {sorted(overlap)[:3]}")
    dataset = {**failure_dataset, **success_dataset}
    train_ids = [*failure_train_ids, *success_train_ids]

    def row_loss(
        logits: torch.Tensor,
        _row: Mapping[str, Any],
        labels: torch.Tensor,
        pos_weight_tensor: torch.Tensor,
    ) -> torch.Tensor:
        per_timestep = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight_tensor,
            reduction="none",
        )
        weights = torch.where(
            labels > 0.5,
            pos_weight_tensor,
            torch.ones_like(labels),
        )
        return per_timestep.sum() / weights.sum().clamp_min(1e-12)

    model, training = localization_core.train_bilstm(
        dataset=dataset,
        train_ids=train_ids,
        val_ids=val_ids,
        mean=mean,
        std=std,
        hidden=hidden,
        pos_weight=pos_weight,
        seed=seed,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        batch_size=batch_size,
        device=device,
        row_loss_fn=row_loss,
        progress_label=progress_label,
        logger=log,
    )
    training.update(
        {
            "best_val_bce": training["best_val_loss"],
            "failure_train_n": len(failure_train_ids),
            "success_train_n": len(success_train_ids),
        }
    )
    return model, training

def interval_error(prediction: int, causal: int, observable: int) -> int:
    if prediction < causal:
        return prediction - causal
    if prediction > observable:
        return prediction - observable
    return 0


def prediction_row(
    dataset: Mapping[str, Mapping[str, Any]],
    rollout_id: str,
    prediction: int,
    score: float,
) -> dict[str, Any]:
    row = dataset[rollout_id]
    error = interval_error(
        prediction,
        int(row["causal_index"]),
        int(row["observable_index"]),
    )
    event = row["event"]
    frames = row["frames"]
    return {
        "rollout_id": rollout_id,
        "event_id": event["event_id"],
        "task_key": row["task_key"],
        "task_id": row["task_id"],
        "failure_type": event.get("failure_type"),
        "causal_index": int(row["causal_index"]),
        "observable_index": int(row["observable_index"]),
        "predicted_index": int(prediction),
        "predicted_frame": int(frames[prediction]),
        "score": float(score),
        "interval_error_samples": int(error),
        "in_interval": error == 0,
    }


def metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = [int(row["interval_error_samples"]) for row in rows]
    count = len(errors)
    if not count:
        return {key: None for key in METRIC_NAMES} | {"n": 0}
    absolute = [abs(value) for value in errors]
    result: dict[str, Any] = {
        "n": count,
        "in_interval_rate": sum(value == 0 for value in errors) / count,
        "before_interval_rate": sum(value < 0 for value in errors) / count,
        "after_interval_rate": sum(value > 0 for value in errors) / count,
        "median_absolute_interval_error_samples": float(statistics.median(absolute)),
        "mae_samples": float(sum(absolute) / count),
        "mse_samples": float(sum(value * value for value in errors) / count),
    }
    for window in (1, 3, 5):
        result[f"within_{window}"] = (
            sum(abs(value) <= window for value in errors) / count
        )
    return result


def evaluate_model(
    model: localization_core.TinyBiLSTM,
    dataset: Mapping[str, Mapping[str, Any]],
    test_ids: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    logits_by_rollout = localization_core.batched_logits(
        model,
        dataset,
        test_ids,
        mean,
        std,
        device,
        batch_size,
    )
    for rollout_id in test_ids:
        logits = logits_by_rollout[rollout_id]
        prediction = int(torch.argmax(logits).item())
        rows.append(
            prediction_row(
                dataset,
                rollout_id,
                prediction,
                float(logits[prediction].item()),
            )
        )
    return rows

def run_one_setting(
    *,
    success_ratio: float,
    model_name: str,
    failure_dataset: Mapping[str, Mapping[str, Any]],
    success_dataset: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, Any],
    success_train_ids: Sequence[str],
    requested_success_n: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    hidden = 16 if model_name.endswith("h16") else 32
    failure_train_ids = list(split["train"])

    # Shared across failure-only and failure+success for a clean ablation.
    mean, std = standardization_stats(failure_dataset, failure_train_ids)
    pos_weight = positive_weight_from_failures(
        failure_dataset, failure_train_ids
    )

    added_success_ids = list(success_train_ids)
    setting = success_ratio_label(success_ratio)
    model, training = train_bilstm(
        hidden=hidden,
        failure_dataset=failure_dataset,
        success_dataset=success_dataset,
        failure_train_ids=failure_train_ids,
        success_train_ids=added_success_ids,
        val_ids=split["val"],
        mean=mean,
        std=std,
        pos_weight=pos_weight,
        seed=seed,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        device=device,
        progress_label=(
            f"{split['split_id']} {model_name} {setting}"
        ),
    )
    rows = evaluate_model(
        model,
        failure_dataset,
        split["test"],
        mean,
        std,
        device,
        args.batch_size,
    )
    for row in rows:
        row.update(
            {
                "split_kind": split["kind"],
                "split_id": split["split_id"],
                "held_out_task": split.get("held_out_task"),
                "model": model_name,
                "training_setting": setting,
                "success_ratio": success_ratio,
                "requested_success_n": requested_success_n,
                "failure_train_n": len(failure_train_ids),
                "success_train_n": len(added_success_ids),
            }
        )
    metrics = metric_summary(rows)
    log(
        f"{split['split_id']} {model_name} {setting} complete "
        f"best_epoch={training['best_epoch']} "
        f"best_val_bce={training['best_val_bce']:.6f} "
        f"in_interval={metrics['in_interval_rate']:.3f} "
        f"before={metrics['before_interval_rate']:.3f} "
        f"after={metrics['after_interval_rate']:.3f}"
    )
    metrics.update(
        {
            "split_kind": split["kind"],
            "split_id": split["split_id"],
            "held_out_task": split.get("held_out_task"),
            "model": model_name,
            "training_setting": setting,
            "success_ratio": success_ratio,
            "requested_success_n": requested_success_n,
            "failure_train_n": len(failure_train_ids),
            "success_train_n": len(added_success_ids),
            "val_failure_n": len(split["val"]),
            "test_failure_n": len(split["test"]),
            "best_epoch": training["best_epoch"],
            "best_val_bce": training["best_val_bce"],
            "pos_weight": training["pos_weight"],
            "batch_size": training["batch_size"],
            "effective_train_batch_size": training["effective_train_batch_size"],
            "optimizer_steps_per_epoch": training["optimizer_steps_per_epoch"],
        }
    )
    return metrics, rows, training


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)

    result: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        out = {field: value for field, value in zip(group_fields, key)}
        out["repeat_n"] = len(group)
        for metric in METRIC_NAMES:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) not in (None, "")
            ]
            out[f"{metric}_mean"] = float(np.mean(values)) if values else None
            out[f"{metric}_variance"] = float(np.var(values)) if values else None
        for count_field in (
            "failure_train_n",
            "requested_success_n",
            "success_train_n",
            "test_failure_n",
        ):
            values = [
                float(row[count_field])
                for row in group
                if row.get(count_field) is not None
            ]
            out[f"{count_field}_mean"] = float(np.mean(values)) if values else None
        result.append(out)
    return result


def ablation_deltas(summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        str(row["model"]): row
        for row in summary
        if float(row.get("success_ratio") or 0.0) == 0.0
    }
    rows: list[dict[str, Any]] = []
    for current in summary:
        ratio = float(current.get("success_ratio") or 0.0)
        if ratio == 0.0:
            continue
        model = str(current["model"])
        base = baselines.get(model)
        if base is None:
            continue
        row: dict[str, Any] = {
            "model": model,
            "success_ratio": ratio,
            "success_train_n_mean": current.get("success_train_n_mean"),
        }
        for metric in METRIC_NAMES:
            base_value = base.get(f"{metric}_mean")
            new_value = current.get(f"{metric}_mean")
            row[f"{metric}_ratio0"] = base_value
            row[f"{metric}_current"] = new_value
            row[f"{metric}_delta_vs_ratio0"] = (
                float(new_value) - float(base_value)
                if base_value is not None and new_value is not None
                else None
            )
        rows.append(row)
    return rows

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def conclusion_text(
    random_summary: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
) -> str:
    ratios = sorted(
        {
            float(row.get("success_ratio") or 0.0)
            for row in random_summary
        }
    )
    ratio_text = ", ".join(f"{ratio:g}x" for ratio in ratios)
    lines = [
        "# Success-ratio BiLSTM ablation",
        "",
        "Only the amount of clean-success all-negative training data changes. "
        f"This run uses ratios {ratio_text} relative to the number of failure "
        "training rollouts in each repeat. Sampling is deterministic, nested, "
        "and same-task-first. Loss, normalization, model architecture, failure "
        "split, optimizer, and evaluation remain fixed.",
        "",
    ]
    for row in delta_rows:
        before_delta = row.get("before_interval_rate_delta_vs_ratio0")
        hit_delta = row.get("in_interval_rate_delta_vs_ratio0")
        mae_delta = row.get("mae_samples_delta_vs_ratio0")
        lines.append(
            f"- **{row['model']} @ {row['success_ratio']}x**: "
            f"before-interval Δ={before_delta:+.4f}, "
            f"in-interval Δ={hit_delta:+.4f}, MAE Δ={mae_delta:+.4f}."
            if None not in (before_delta, hit_delta, mae_delta)
            else (
                f"- **{row['model']} @ {row['success_ratio']}x**: "
                "insufficient paired summary values."
            )
        )
    lines.extend(
        [
            "",
            "The main diagnostic is whether before-interval error improves at a "
            "small success ratio and then degrades as success negatives dominate, "
            "or instead degrades immediately from the first non-zero ratio.",
            "",
        ]
    )
    return "\n".join(lines)

def analyze(args: argparse.Namespace) -> Path:
    if args.run_pool_root:
        source_root = ensure_within_project(
            resolve_project_path(args.run_pool_root), "run pool root"
        )
        latest_per_rollout = True
    elif args.run_root:
        source_root = ensure_within_project(
            resolve_project_path(args.run_root), "run root"
        )
        latest_per_rollout = False
    else:
        raise ValueError("Either --run-pool-root or --run-root is required")

    manifest_path = ensure_within_project(resolve_project_path(args.manifest), "manifest")
    annotation_dir = ensure_within_project(
        resolve_project_path(args.annotations), "annotation directory"
    )
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not annotation_dir.is_dir():
        raise FileNotFoundError(annotation_dir)

    output_dir = (
        ensure_within_project(resolve_project_path(args.output_dir), "output directory")
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    log(
        "Starting BiLSTM success-negative ablation "
        f"source_root={project_relative(source_root)} "
        f"selection={'latest-per-rollout' if latest_per_rollout else 'single-run'} "
        f"device={args.device} batch_size={args.batch_size} "
        f"repeats={args.repeats} epochs={args.epochs} patience={args.patience} "
        f"success_ratios={','.join(str(value) for value in args.success_ratios)}"
    )
    log("Loading saved fused Robo-Dopamine signals and annotations")
    manifest = load_manifest(manifest_path)
    signals, events, _no_event_failures, clean_rollouts, provenance = build_base_records(
        source_root,
        manifest,
        annotation_dir,
        allowed_rollout_ids=None,
        signal_mode="fused",
        latest_per_rollout=latest_per_rollout,
    )
    failure_dataset = build_failure_dataset(signals, events)
    success_dataset = build_success_dataset(signals, clean_rollouts)
    if len(failure_dataset) < 8:
        raise ValueError(
            "Need at least 8 eligible terminal-failure rollouts, "
            f"found {len(failure_dataset)}"
        )
    if not success_dataset:
        raise ValueError(
            "No clean-success rollout has a usable saved fused signal; "
            "the requested ablation cannot be run."
        )
    log(
        f"Dataset ready: failure_rollouts={len(failure_dataset)} "
        f"clean_success_rollouts={len(success_dataset)}"
    )

    random_splits = [
        rollout_split(
            failure_dataset,
            seed=args.seed + repeat,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
        for repeat in range(args.repeats)
    ]
    heldout_splits = task_splits(
        failure_dataset,
        seed=args.seed,
        min_test_rollouts=args.min_task_test_rollouts,
    )

    device = resolve_device(args.device)
    log(
        f"PyTorch device resolved to {device}; torch={torch.__version__}; "
        f"cuda_available={torch.cuda.is_available()}"
    )
    detailed_metrics: list[dict[str, Any]] = []
    task_metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []

    for split_index, split in enumerate(random_splits):
        success_subsets, sampling = build_success_ratio_subsets(
            success_dataset,
            failure_dataset,
            split["train"],
            seed=int(split["seed"]) * 100 + 31,
            success_ratios=args.success_ratios,
        )
        split_records.append(
            {
                **split,
                "success_subsets": {
                    str(ratio): ids for ratio, ids in success_subsets.items()
                },
                "success_sampling": sampling,
            }
        )
        log(
            f"{split['split_id']} prepared: failure_train={len(split['train'])} "
            f"val={len(split['val'])} test={len(split['test'])} "
            f"same_task_success={sampling['same_task_available_n']} "
            f"fallback_success={sampling['fallback_available_n']}"
        )
        for model_index, model_name in enumerate(MODEL_NAMES):
            shared_seed = int(split["seed"]) * 1000 + model_index
            for ratio in args.success_ratios:
                success_ids = success_subsets[ratio]
                requested_success_n = int(len(split["train"]) * ratio)
                setting = success_ratio_label(ratio)
                log(
                    f"{split['split_id']} {model_name} {setting} start "
                    f"failure_train={len(split['train'])} "
                    f"success_train={len(success_ids)}/{requested_success_n}"
                )
                metrics, rows, training = run_one_setting(
                    success_ratio=ratio,
                    model_name=model_name,
                    failure_dataset=failure_dataset,
                    success_dataset=success_dataset,
                    split=split,
                    success_train_ids=success_ids,
                    requested_success_n=requested_success_n,
                    seed=shared_seed,
                    args=args,
                    device=device,
                )
                detailed_metrics.append(metrics)
                predictions.extend(rows)
                training_records.append(
                    {
                        "split_id": split["split_id"],
                        "model": model_name,
                        "training_setting": setting,
                        "success_ratio": ratio,
                        "requested_success_n": requested_success_n,
                        "failure_train_ids": list(split["train"]),
                        "success_train_ids": success_ids,
                        "success_sampling": sampling,
                        **training,
                    }
                )

    task_split_records: list[dict[str, Any]] = []
    for split_index, split in enumerate(heldout_splits):
        success_subsets, sampling = build_success_ratio_subsets(
            success_dataset,
            failure_dataset,
            split["train"],
            seed=args.seed * 10000 + split_index * 100 + 31,
            held_out_task=str(split["held_out_task"]),
            success_ratios=args.success_ratios,
        )
        task_split_records.append(
            {
                **split,
                "success_subsets": {
                    str(ratio): ids for ratio, ids in success_subsets.items()
                },
                "success_sampling": sampling,
            }
        )
        log(
            f"{split['split_id']} prepared: failure_train={len(split['train'])} "
            f"val={len(split['val'])} test={len(split['test'])} "
            f"same_task_success={sampling['same_task_available_n']} "
            f"fallback_success={sampling['fallback_available_n']}"
        )
        for model_index, model_name in enumerate(MODEL_NAMES):
            shared_seed = args.seed * 10000 + split_index * 100 + model_index
            for ratio in args.success_ratios:
                success_ids = success_subsets[ratio]
                requested_success_n = int(len(split["train"]) * ratio)
                setting = success_ratio_label(ratio)
                log(
                    f"{split['split_id']} {model_name} {setting} start "
                    f"failure_train={len(split['train'])} "
                    f"success_train={len(success_ids)}/{requested_success_n}"
                )
                metrics, rows, training = run_one_setting(
                    success_ratio=ratio,
                    model_name=model_name,
                    failure_dataset=failure_dataset,
                    success_dataset=success_dataset,
                    split=split,
                    success_train_ids=success_ids,
                    requested_success_n=requested_success_n,
                    seed=shared_seed,
                    args=args,
                    device=device,
                )
                task_metrics.append(metrics)
                predictions.extend(rows)
                training_records.append(
                    {
                        "split_id": split["split_id"],
                        "model": model_name,
                        "training_setting": setting,
                        "success_ratio": ratio,
                        "requested_success_n": requested_success_n,
                        "failure_train_ids": list(split["train"]),
                        "success_train_ids": success_ids,
                        "success_sampling": sampling,
                        **training,
                    }
                )

    random_summary = aggregate_rows(
        detailed_metrics, ["model", "success_ratio", "training_setting"]
    )
    task_summary = (
        aggregate_rows(
            task_metrics, ["model", "success_ratio", "training_setting"]
        )
        if task_metrics
        else []
    )
    delta_rows = ablation_deltas(random_summary)

    write_csv(output_dir / "ablation_comparison.csv", random_summary)
    write_csv(output_dir / "ablation_delta.csv", delta_rows)
    write_csv(output_dir / "per_split_metrics.csv", detailed_metrics)
    write_csv(output_dir / "task_held_out_metrics.csv", task_metrics)
    write_csv(output_dir / "task_held_out_summary.csv", task_summary)
    write_csv(output_dir / "per_rollout_predictions.csv", predictions)
    write_json(output_dir / "training_records.json", training_records)
    write_json(
        output_dir / "split_manifest.json",
        {
            "schema_version": 3,
            "source_root": project_relative(source_root),
            "selection_mode": (
                "latest_usable_signal_per_rollout"
                if latest_per_rollout
                else "single_run_root"
            ),
            "failure_rollout_n": len(failure_dataset),
            "clean_success_rollout_n": len(success_dataset),
            "random_splits": split_records,
            "task_held_out_splits": task_split_records,
        },
    )
    (output_dir / "conclusion.md").write_text(
        conclusion_text(random_summary, delta_rows),
        encoding="utf-8",
    )
    write_json(
        output_dir / "metadata.json",
        {
            "schema_version": 3,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "analysis": "robo_dopamine_bilstm_success_negative_ablation",
            "analysis_mode": "saved-output PyTorch training; no Robo-Dopamine inference",
            "source_root": project_relative(source_root),
            "selection_mode": (
                "latest_usable_signal_per_rollout"
                if latest_per_rollout
                else "single_run_root"
            ),
            "manifest": project_relative(manifest_path),
            "annotation_dir": project_relative(annotation_dir),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "failure_rollout_n": len(failure_dataset),
            "clean_success_rollout_n": len(success_dataset),
            "models": {
                "tiny_bilstm_h16": "1-layer bidirectional LSTM, hidden=16 per direction",
                "tiny_bilstm_h32": "1-layer bidirectional LSTM, hidden=32 per direction",
            },
            "success_ratios": list(args.success_ratios),
            "training_settings": (
                "clean-success all-negative rollout count is "
                "int(failure_train_n * success_ratio)"
            ),
            "controls": {
                "normalization": (
                    "fit on failure-train sequences only and shared between both settings"
                ),
                "pos_weight": (
                    "computed from failure-train labels only and shared between both settings"
                ),
                "validation": "failure validation rollouts only",
                "test": "failure test rollouts only",
                "success_sampling": (
                    "deterministic nested random prefixes; same-task successes first, "
                    "then non-held-out fallback successes if required"
                ),
                "prediction": "rollout-global argmax of per-timestep BiLSTM score",
                "loss": "positive-class-weighted BCE via shared rollout-minibatch trainer",
            },
            "training": {
                "epochs_max": args.epochs,
                "patience": args.patience,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "batch_size": args.batch_size,
                "batching": (
                    "variable-length rollout mini-batches using "
                    "pack_padded_sequence; rollout-weighted mean loss"
                ),
                "repeats": args.repeats,
                "train_fraction": args.train_fraction,
                "val_fraction": args.val_fraction,
            },
            "provenance": provenance,
            "outputs": [
                "ablation_comparison.csv",
                "ablation_delta.csv",
                "per_split_metrics.csv",
                "task_held_out_metrics.csv",
                "task_held_out_summary.csv",
                "per_rollout_predictions.csv",
                "training_records.json",
                "split_manifest.json",
                "conclusion.md",
                "metadata.json",
            ],
        },
    )
    log(f"Wrote BiLSTM success-negative ablation to {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default=None,
        help=(
            "Legacy single completed Robo-Dopamine run. "
            "Use --run-pool-root for latest-per-rollout aggregation."
        ),
    )
    parser.add_argument(
        "--run-pool-root",
        default=None,
        help=(
            "Directory containing historical Robo-Dopamine runs. "
            "The latest usable fused result is selected independently per rollout."
        ),
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--min-task-test-rollouts", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of variable-length rollout sequences per optimizer step.",
    )
    parser.add_argument(
        "--success-ratios",
        default="0.5,1,2",
        help=(
            "Comma-separated positive success/failure ratios. "
            "The 0x failure-only baseline is always added automatically."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, cuda, cuda:0, ...",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.success_ratios = parse_success_ratios(args.success_ratios)
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.batch_size < 1 or args.batch_size > 128:
        raise ValueError("--batch-size must be between 1 and 128")
    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1")
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be >= 1")
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
