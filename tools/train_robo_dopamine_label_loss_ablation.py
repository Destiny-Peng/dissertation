#!/usr/bin/env python3
"""BiLSTM h16 label/loss ablation on saved Robo-Dopamine fused signals.

This runner never reruns Robo-Dopamine. It uses failure rollouts only, keeps
rollout-level splits fixed across configurations, supports multiple annotated
failure events per rollout with exponentially decaying event importance, and
turns eventless terminal failures into a pseudo [0, 0] failure interval.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

print("[BiLSTM label/loss] process started; importing PyTorch...", flush=True)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print(f"[BiLSTM label/loss] PyTorch import complete: {torch.__version__}", flush=True)
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Use conda_envs/LF3R-robo-dopamine/bin/python."
    ) from exc

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from robo_localization_head import core as localization_core

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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/robo_dopamine_label_loss_ablation"

SYMMETRIC_SIGMAS = (1.0, 2.0, 3.0, 5.0)
ASYMMETRIC_SIGMAS = ((3.0, 1.0), (2.0, 1.0), (1.0, 3.0))
LOSS_NAMES = (
    "bce",
    "temporal_softmax_ce",
    "temporal_softmax_ce_distance",
    "temporal_softmax_ce_squared_distance",
    "temporal_softmax_ce_ranking",
    "temporal_softmax_ce_distance_ranking",
)
METRIC_NAMES = (
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
)


def log(message: str) -> None:
    timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _onset_anchor(frames: Sequence[int], frame: int) -> int | None:
    return next(
        (index for index, value in enumerate(frames) if int(value) >= int(frame)),
        None,
    )


def _sequence(signal: Mapping[str, Any]) -> np.ndarray:
    progress = np.asarray(signal["progress"], dtype=np.float32)
    hops = np.asarray(signal["hops"], dtype=np.float32)
    if progress.ndim != 1 or hops.ndim != 1 or len(progress) != len(hops):
        raise ValueError("progress/hops must be same-length 1D arrays")
    if len(progress) == 0:
        raise ValueError("empty fused signal")
    return np.stack([progress, hops], axis=1)


def _distance_to_intervals(length: int, intervals: Sequence[tuple[int, int]]) -> np.ndarray:
    distances = np.full(length, np.inf, dtype=np.float32)
    indices = np.arange(length, dtype=np.float32)
    for causal, observable in intervals:
        left = np.maximum(float(causal) - indices, 0.0)
        right = np.maximum(indices - float(observable), 0.0)
        distances = np.minimum(distances, left + right)
    if not np.all(np.isfinite(distances)):
        raise ValueError("cannot compute interval distance without a valid interval")
    return distances


def build_failure_targets(
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    *,
    tau_event: float,
) -> dict[str, dict[str, Any]]:
    """Build rollout-level multi-event targets before choosing label shape."""
    if not math.isfinite(tau_event) or tau_event <= 0:
        raise ValueError("tau_event must be finite and > 0")

    by_rollout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in events:
        if str(raw.get("outcome") or "") != "terminal_failure":
            continue
        causal = raw.get("causal_onset_frame")
        observable = raw.get("observable_onset_frame")
        if causal is None or observable is None:
            continue
        try:
            causal_frame = int(causal)
            observable_frame = int(observable)
        except (TypeError, ValueError):
            continue
        if causal_frame > observable_frame:
            continue
        event = dict(raw)
        event["causal_onset_frame"] = causal_frame
        event["observable_onset_frame"] = observable_frame
        event["event_index"] = int(event.get("event_index") or 0)
        event["event_id"] = str(
            event.get("event_id")
            or f"{event['rollout_id']}::event{event['event_index']}"
        )
        event["target_source"] = "annotated_event"
        by_rollout[str(event["rollout_id"])].append(event)

    no_event_by_id = {
        str(row["rollout_id"]): dict(row)
        for row in no_event_failures
        if str(row.get("outcome") or "") == "terminal_failure"
    }

    rollout_ids = sorted(set(by_rollout) | set(no_event_by_id))
    result: dict[str, dict[str, Any]] = {}
    for rollout_id in rollout_ids:
        signal = signals.get(rollout_id)
        if signal is None:
            continue
        frames = [int(value) for value in signal["frames"]]
        if not frames:
            continue

        mapped_events: list[dict[str, Any]] = []
        for event in by_rollout.get(rollout_id, []):
            causal_index = _onset_anchor(frames, int(event["causal_onset_frame"]))
            observable_index = _onset_anchor(frames, int(event["observable_onset_frame"]))
            if (
                causal_index is None
                or observable_index is None
                or causal_index > observable_index
            ):
                continue
            mapped_events.append(
                {
                    **event,
                    "causal_index": int(causal_index),
                    "observable_index": int(observable_index),
                    "causal_native_frame": int(frames[causal_index]),
                    "observable_native_frame": int(frames[observable_index]),
                }
            )

        if not mapped_events and rollout_id in no_event_by_id:
            source = no_event_by_id[rollout_id]
            first_index = _onset_anchor(frames, 0)
            if first_index is None:
                first_index = 0
            mapped_events.append(
                {
                    "event_id": f"{rollout_id}::pseudo_no_event_frame0",
                    "rollout_id": rollout_id,
                    "event_index": -1,
                    "failure_type": source.get("failure_type") or "timeout_no_progress",
                    "causal_onset_frame": 0,
                    "observable_onset_frame": 0,
                    "causal_index": int(first_index),
                    "observable_index": int(first_index),
                    "causal_native_frame": int(frames[first_index]),
                    "observable_native_frame": int(frames[first_index]),
                    "target_source": "pseudo_no_event_frame0",
                    "task_key": source.get("task_key") or signal.get("task_key"),
                    "task_id": source.get("task_id", signal.get("task_id")),
                }
            )

        if not mapped_events:
            continue

        mapped_events.sort(
            key=lambda event: (
                int(event["causal_index"]),
                int(event["observable_index"]),
                int(event.get("event_index") or 0),
            )
        )
        first_causal = int(mapped_events[0]["causal_index"])
        for rank, event in enumerate(mapped_events):
            delta = int(event["causal_index"]) - first_causal
            event["event_rank"] = rank
            event["event_weight"] = float(math.exp(-float(delta) / tau_event))

        intervals = [
            (int(event["causal_index"]), int(event["observable_index"]))
            for event in mapped_events
        ]
        support = np.zeros(len(frames), dtype=np.float32)
        for causal, observable in intervals:
            support[causal : observable + 1] = 1.0

        first = mapped_events[0]
        task_key = str(first.get("task_key") or signal.get("task_key") or "")
        result[rollout_id] = {
            "rollout_id": rollout_id,
            "kind": "failure",
            "task_key": task_key,
            "task_id": first.get("task_id", signal.get("task_id")),
            "frames": np.asarray(frames, dtype=np.int64),
            "sequence": _sequence(signal),
            "events": mapped_events,
            "intervals": intervals,
            "first_interval": intervals[0],
            "hard_support": support,
            "distance_to_interval": _distance_to_intervals(len(frames), intervals),
            "event_count": len(mapped_events),
            "has_pseudo_event": any(
                event["target_source"] == "pseudo_no_event_frame0"
                for event in mapped_events
            ),
        }
    return result


def label_config_name(kind: str, sigma_pre: float | None = None, sigma_post: float | None = None) -> str:
    if kind == "hard":
        return "hard_weighted_interval"
    if sigma_pre is None or sigma_post is None:
        raise ValueError("Gaussian label requires sigma_pre and sigma_post")
    if sigma_pre == sigma_post:
        return f"gaussian_sigma_{sigma_pre:g}"
    return f"gaussian_pre_{sigma_pre:g}_post_{sigma_post:g}"


def make_labels(
    row: Mapping[str, Any],
    *,
    kind: str,
    sigma_pre: float | None = None,
    sigma_post: float | None = None,
) -> np.ndarray:
    length = len(row["frames"])
    labels = np.zeros(length, dtype=np.float32)
    if kind not in {"hard", "gaussian"}:
        raise ValueError(f"unknown label kind: {kind}")
    if kind == "gaussian":
        if (
            sigma_pre is None
            or sigma_post is None
            or sigma_pre <= 0
            or sigma_post <= 0
        ):
            raise ValueError("Gaussian sigmas must be > 0")

    indices = np.arange(length, dtype=np.float32)
    for event in row["events"]:
        causal = int(event["causal_index"])
        observable = int(event["observable_index"])
        weight = float(event["event_weight"])
        candidate = np.zeros(length, dtype=np.float32)
        if kind == "hard":
            candidate[causal : observable + 1] = weight
        else:
            before = indices < causal
            inside = (indices >= causal) & (indices <= observable)
            after = indices > observable
            candidate[inside] = weight
            candidate[before] = weight * np.exp(
                -((float(causal) - indices[before]) ** 2)
                / (2.0 * float(sigma_pre) ** 2)
            )
            candidate[after] = weight * np.exp(
                -((indices[after] - float(observable)) ** 2)
                / (2.0 * float(sigma_post) ** 2)
            )
        labels = np.maximum(labels, candidate)
    return labels.astype(np.float32)


def dataset_with_labels(
    base_dataset: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        rollout_id: {
            **dict(row),
            "labels": make_labels(
                row,
                kind=str(config["kind"]),
                sigma_pre=config.get("sigma_pre"),
                sigma_post=config.get("sigma_post"),
            ),
            "label_config": str(config["name"]),
        }
        for rollout_id, row in base_dataset.items()
    }


def _task(dataset: Mapping[str, Mapping[str, Any]], rollout_id: str) -> str:
    return str(dataset[rollout_id].get("task_key") or "")


def rollout_split(
    dataset: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, Any]:
    by_task: dict[str, list[str]] = defaultdict(list)
    for rollout_id in sorted(dataset):
        by_task[_task(dataset, rollout_id)].append(rollout_id)

    rng = random.Random(seed)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for task in sorted(by_task):
        ids = list(by_task[task])
        rng.shuffle(ids)
        count = len(ids)
        if count == 1:
            train.extend(ids)
            continue
        if count == 2:
            train.append(ids[0])
            test.append(ids[1])
            continue
        train_n = max(1, min(count - 2, int(round(count * train_fraction))))
        val_n = max(1, min(count - train_n - 1, int(round(count * val_fraction))))
        train.extend(ids[:train_n])
        val.extend(ids[train_n : train_n + val_n])
        test.extend(ids[train_n + val_n :])

    all_ids = set(dataset)
    train_set, val_set, test_set = set(train), set(val), set(test)
    train_set.update(all_ids - train_set - val_set - test_set)
    target_min = 1 if len(all_ids) < 12 else 2
    for target in (val_set, test_set):
        while len(target) < target_min and len(train_set) > target_min + 2:
            rollout_id = sorted(train_set)[0]
            train_set.remove(rollout_id)
            target.add(rollout_id)

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise AssertionError("rollout split overlap")
    if train_set | val_set | test_set != all_ids:
        raise AssertionError("rollout split does not cover dataset")
    return {
        "split_id": f"random_seed_{seed}",
        "kind": "rollout_random",
        "seed": seed,
        "train": sorted(train_set),
        "val": sorted(val_set),
        "test": sorted(test_set),
    }


def standardization_stats(
    dataset: Mapping[str, Mapping[str, Any]],
    rollout_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    sequences = [
        np.asarray(dataset[rollout_id]["sequence"], dtype=np.float32)
        for rollout_id in rollout_ids
    ]
    if not sequences:
        raise ValueError("empty rollout selection")
    merged = np.concatenate(sequences, axis=0)
    mean = merged.mean(axis=0, keepdims=True)
    std = merged.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def positive_weight_from_support(
    dataset: Mapping[str, Mapping[str, Any]],
    rollout_ids: Sequence[str],
) -> float:
    support = np.concatenate(
        [
            np.asarray(dataset[rollout_id]["hard_support"], dtype=np.float32)
            for rollout_id in rollout_ids
        ]
    )
    positive = float(np.sum(support > 0.5))
    negative = float(np.sum(support <= 0.5))
    if positive <= 0:
        raise ValueError("failure training split contains no positive interval samples")
    return min(20.0, max(1.0, negative / positive))


class TinyBiLSTM(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(2 * hidden, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lengths is None:
            encoded, _ = self.lstm(sequence)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_encoded, _ = self.lstm(packed)
            encoded, _ = nn.utils.rnn.pad_packed_sequence(
                packed_encoded,
                batch_first=True,
                total_length=sequence.shape[1],
            )
        return self.head(encoded).squeeze(-1)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_tensor_dataset(
    dataset: Mapping[str, Mapping[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, dict[str, torch.Tensor]]:
    prepared: dict[str, dict[str, torch.Tensor]] = {}
    for rollout_id, row in dataset.items():
        sequence = (
            np.asarray(row["sequence"], dtype=np.float32) - mean
        ) / std
        labels = np.asarray(row["labels"], dtype=np.float32)
        prepared[rollout_id] = {
            "sequence": torch.from_numpy(sequence),
            "labels": torch.from_numpy(labels),
        }
    return prepared


def collate_rollout_batch(
    prepared: Mapping[str, Mapping[str, torch.Tensor]],
    rollout_ids: Sequence[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rollout_ids:
        raise ValueError("cannot collate an empty rollout batch")
    sequences = [prepared[rollout_id]["sequence"] for rollout_id in rollout_ids]
    labels = [prepared[rollout_id]["labels"] for rollout_id in rollout_ids]
    lengths = torch.tensor(
        [int(sequence.shape[0]) for sequence in sequences],
        dtype=torch.long,
    )
    padded_sequence = nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0,
    )
    padded_labels = nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=0.0,
    )
    non_blocking = device.type == "cuda"
    return (
        padded_sequence.to(device, non_blocking=non_blocking),
        padded_labels.to(device, non_blocking=non_blocking),
        lengths.to(device, non_blocking=non_blocking),
    )


def rollout_batches(
    rollout_ids: Sequence[str],
    batch_size: int,
    *,
    rng: random.Random | None = None,
) -> list[list[str]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    ids = list(rollout_ids)
    if rng is not None:
        rng.shuffle(ids)
    return [
        ids[index : index + batch_size]
        for index in range(0, len(ids), batch_size)
    ]


def _logmeanexp(values: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(values, dim=0) - math.log(max(1, int(values.numel())))


def loss_value(
    logits: torch.Tensor,
    row: Mapping[str, Any],
    labels: torch.Tensor,
    *,
    loss_name: str,
    pos_weight: torch.Tensor,
    distance_weight: float,
    ranking_weight: float,
    ranking_margin: float,
) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]
    if labels.ndim == 2 and labels.shape[0] == 1:
        labels = labels[0]
    if logits.ndim != 1 or labels.ndim != 1:
        raise ValueError("loss_value expects one rollout of 1D logits/labels")
    if loss_name == "bce":
        return F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
            reduction="mean",
        )

    if loss_name not in LOSS_NAMES:
        raise ValueError(f"unknown loss: {loss_name}")

    total = labels.sum()
    if float(total.detach().cpu()) <= 0:
        raise ValueError("temporal-softmax target has zero mass")
    q = labels / total
    log_p = F.log_softmax(logits, dim=0)
    loss = -(q * log_p).sum()

    p = torch.softmax(logits, dim=0)
    distances = torch.as_tensor(
        np.asarray(row["distance_to_interval"], dtype=np.float32),
        dtype=logits.dtype,
        device=logits.device,
    )
    denom = float(max(1, len(distances) - 1))
    normalized_distance = distances / denom

    if loss_name in {
        "temporal_softmax_ce_distance",
        "temporal_softmax_ce_distance_ranking",
    }:
        loss = loss + distance_weight * (p * normalized_distance).sum()
    elif loss_name == "temporal_softmax_ce_squared_distance":
        loss = loss + distance_weight * (p * normalized_distance.square()).sum()

    if loss_name in {
        "temporal_softmax_ce_ranking",
        "temporal_softmax_ce_distance_ranking",
    }:
        interval_scores: list[torch.Tensor] = []
        interval_weights: list[float] = []
        for event in row["events"]:
            causal = int(event["causal_index"])
            observable = int(event["observable_index"])
            interval_scores.append(logits[causal : observable + 1].mean())
            interval_weights.append(float(event["event_weight"]))
        weights = torch.as_tensor(
            interval_weights,
            dtype=logits.dtype,
            device=logits.device,
        )
        positive_score = (
            torch.stack(interval_scores) * weights
        ).sum() / weights.sum().clamp_min(1e-12)

        support = torch.as_tensor(
            np.asarray(row["hard_support"], dtype=np.float32) > 0.5,
            device=logits.device,
        )
        background = logits[~support]
        if background.numel():
            negative_score = _logmeanexp(background)
            ranking = F.softplus(
                torch.as_tensor(
                    ranking_margin,
                    dtype=logits.dtype,
                    device=logits.device,
                )
                - (positive_score - negative_score)
            )
            loss = loss + ranking_weight * ranking
    return loss


def train_bilstm(
    *,
    dataset: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    pos_weight: float,
    loss_name: str,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    batch_size: int,
    distance_weight: float,
    ranking_weight: float,
    ranking_margin: float,
    device: torch.device,
    progress_label: str,
) -> tuple[localization_core.TinyBiLSTM, dict[str, Any]]:
    """Experiment adapter around the shared variable-length minibatch trainer."""

    def row_loss(
        logits: torch.Tensor,
        row: Mapping[str, Any],
        labels: torch.Tensor,
        pos_weight_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return loss_value(
            logits,
            row,
            labels,
            loss_name=loss_name,
            pos_weight=pos_weight_tensor,
            distance_weight=distance_weight,
            ranking_weight=ranking_weight,
            ranking_margin=ranking_margin,
        )

    model, training = localization_core.train_bilstm(
        dataset=dataset,
        train_ids=train_ids,
        val_ids=val_ids,
        mean=mean,
        std=std,
        hidden=16,
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
            "loss_name": loss_name,
            "failure_train_n": len(train_ids),
        }
    )
    return model, training

def interval_error(prediction: int, causal: int, observable: int) -> int:
    if prediction < causal:
        return prediction - causal
    if prediction > observable:
        return prediction - observable
    return 0


def nearest_interval_error(
    prediction: int,
    intervals: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    candidates = [
        (interval_error(prediction, causal, observable), index)
        for index, (causal, observable) in enumerate(intervals)
    ]
    return min(candidates, key=lambda item: (abs(item[0]), item[1]))


def prediction_row(
    dataset: Mapping[str, Mapping[str, Any]],
    rollout_id: str,
    prediction: int,
    score: float,
) -> dict[str, Any]:
    row = dataset[rollout_id]
    error, nearest_index = nearest_interval_error(prediction, row["intervals"])
    first_causal, first_observable = row["first_interval"]
    first_error = interval_error(prediction, first_causal, first_observable)
    frames = row["frames"]
    nearest_event = row["events"][nearest_index]
    return {
        "rollout_id": rollout_id,
        "task_key": row["task_key"],
        "task_id": row["task_id"],
        "event_count": int(row["event_count"]),
        "has_pseudo_event": bool(row["has_pseudo_event"]),
        "nearest_event_id": nearest_event["event_id"],
        "nearest_event_rank": int(nearest_event["event_rank"]),
        "nearest_event_weight": float(nearest_event["event_weight"]),
        "predicted_index": int(prediction),
        "predicted_frame": int(frames[prediction]),
        "score": float(score),
        "interval_error_samples": int(error),
        "first_event_interval_error_samples": int(first_error),
        "in_interval": error == 0,
        "first_event_in_interval": first_error == 0,
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
        "first_event_in_interval_rate": (
            sum(bool(row["first_event_in_interval"]) for row in rows) / count
        ),
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
        valid_logits = logits_by_rollout[rollout_id]
        prediction = int(torch.argmax(valid_logits).item())
        rows.append(
            prediction_row(
                dataset,
                rollout_id,
                prediction,
                float(valid_logits[prediction].item()),
            )
        )
    return rows

def run_setting(
    *,
    base_dataset: Mapping[str, Mapping[str, Any]],
    label_config: Mapping[str, Any],
    loss_name: str,
    split: Mapping[str, Any],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    dataset = dataset_with_labels(base_dataset, label_config)
    mean, std = standardization_stats(base_dataset, split["train"])
    pos_weight = positive_weight_from_support(base_dataset, split["train"])
    label_name = str(label_config["name"])
    progress_label = f"{split['split_id']} {label_name} {loss_name}"
    model, training = train_bilstm(
        dataset=dataset,
        train_ids=split["train"],
        val_ids=split["val"],
        mean=mean,
        std=std,
        pos_weight=pos_weight,
        loss_name=loss_name,
        seed=seed,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        distance_weight=args.distance_weight,
        ranking_weight=args.ranking_weight,
        ranking_margin=args.ranking_margin,
        device=device,
        progress_label=progress_label,
    )
    rows = evaluate_model(
        model,
        dataset,
        split["test"],
        mean,
        std,
        device,
        args.batch_size,
    )
    for row in rows:
        row.update(
            {
                "split_id": split["split_id"],
                "label_config": label_name,
                "loss": loss_name,
            }
        )
    metrics = metric_summary(rows)
    metrics.update(
        {
            "split_id": split["split_id"],
            "label_config": label_name,
            "loss": loss_name,
            "failure_train_n": len(split["train"]),
            "val_failure_n": len(split["val"]),
            "test_failure_n": len(split["test"]),
            "best_epoch": training["best_epoch"],
            "best_val_loss": training["best_val_loss"],
            "pos_weight": training["pos_weight"],
            "batch_size": training["batch_size"],
            "effective_train_batch_size": training["effective_train_batch_size"],
            "optimizer_steps_per_epoch": training["optimizer_steps_per_epoch"],
        }
    )
    log(
        f"{progress_label} complete "
        f"in_interval={metrics['in_interval_rate']:.3f} "
        f"first_event={metrics['first_event_in_interval_rate']:.3f} "
        f"mae={metrics['mae_samples']:.3f}"
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
        result.append(out)
    return result


def _summary_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    hit = float(row.get("in_interval_rate_mean") or 0.0)
    mae = float(row.get("mae_samples_mean") or math.inf)
    mse = float(row.get("mse_samples_mean") or math.inf)
    return (-hit, mae, mse)


def best_summary_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("no summary rows")
    return min(rows, key=_summary_key)


def soft_clearly_improves(
    hard: Mapping[str, Any],
    soft: Mapping[str, Any],
) -> bool:
    return (
        float(soft["in_interval_rate_mean"]) > float(hard["in_interval_rate_mean"])
        and float(soft["mae_samples_mean"]) < float(hard["mae_samples_mean"])
    )


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


def dataset_summary(
    dataset: Mapping[str, Mapping[str, Any]],
    *,
    tau_event: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_counts = Counter(int(row["event_count"]) for row in dataset.values())
    per_rollout = []
    annotated_event_n = 0
    pseudo_n = 0
    for rollout_id, row in sorted(dataset.items()):
        annotated = sum(
            event["target_source"] == "annotated_event"
            for event in row["events"]
        )
        pseudo = sum(
            event["target_source"] == "pseudo_no_event_frame0"
            for event in row["events"]
        )
        annotated_event_n += annotated
        pseudo_n += pseudo
        per_rollout.append(
            {
                "rollout_id": rollout_id,
                "task_key": row["task_key"],
                "event_count": row["event_count"],
                "annotated_event_n": annotated,
                "pseudo_event_n": pseudo,
                "event_indices": json.dumps(
                    [
                        [event["causal_index"], event["observable_index"]]
                        for event in row["events"]
                    ]
                ),
                "event_weights": json.dumps(
                    [event["event_weight"] for event in row["events"]]
                ),
            }
        )
    return (
        {
            "failure_rollout_n": len(dataset),
            "annotated_failure_event_n": annotated_event_n,
            "multi_event_rollout_n": sum(
                int(row["event_count"]) > 1 for row in dataset.values()
            ),
            "pseudo_no_event_frame0_rollout_n": pseudo_n,
            "events_per_rollout": {
                str(key): value for key, value in sorted(event_counts.items())
            },
            "tau_event_native_samples": tau_event,
            "event_weight": "exp(-(causal_index-first_causal_index)/tau_event)",
        },
        per_rollout,
    )


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

    manifest_path = ensure_within_project(
        resolve_project_path(args.manifest), "manifest"
    )
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
        ensure_within_project(
            resolve_project_path(args.output_dir), "output directory"
        )
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    log(
        "Starting BiLSTM h16 label/loss ablation "
        f"source={project_relative(source_root)} repeats={args.repeats} "
        f"tau_event={args.tau_event} batch_size={args.batch_size} "
        f"device={args.device}"
    )
    manifest = load_manifest(manifest_path)
    signals, events, no_event_failures, _clean, provenance = build_base_records(
        source_root,
        manifest,
        annotation_dir,
        allowed_rollout_ids=None,
        signal_mode="fused",
        latest_per_rollout=latest_per_rollout,
    )
    base_dataset = build_failure_targets(
        signals,
        events,
        no_event_failures,
        tau_event=args.tau_event,
    )
    if len(base_dataset) < 8:
        raise ValueError(
            "Need at least 8 eligible terminal-failure rollouts, "
            f"found {len(base_dataset)}"
        )

    target_summary, target_rows = dataset_summary(
        base_dataset, tau_event=args.tau_event
    )
    log(
        "Dataset ready: "
        f"failure_rollouts={target_summary['failure_rollout_n']} "
        f"annotated_events={target_summary['annotated_failure_event_n']} "
        f"multi_event_rollouts={target_summary['multi_event_rollout_n']} "
        f"pseudo_frame0={target_summary['pseudo_no_event_frame0_rollout_n']}"
    )

    splits = [
        rollout_split(
            base_dataset,
            seed=args.seed + repeat,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
        for repeat in range(args.repeats)
    ]
    device = resolve_device(args.device)

    hard_config = {
        "name": label_config_name("hard"),
        "kind": "hard",
        "sigma_pre": None,
        "sigma_post": None,
    }
    symmetric_configs = [
        {
            "name": label_config_name("gaussian", sigma, sigma),
            "kind": "gaussian",
            "sigma_pre": sigma,
            "sigma_post": sigma,
        }
        for sigma in SYMMETRIC_SIGMAS
    ]
    label_configs = [hard_config, *symmetric_configs]

    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, str], tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = {}

    def execute(config: Mapping[str, Any], loss_name: str, stage: str) -> None:
        label_name = str(config["name"])
        for split_index, split in enumerate(splits):
            key = (str(split["split_id"]), label_name, loss_name)
            if key in cache:
                metrics, rows, training = cache[key]
                reused = True
            else:
                shared_seed = int(split["seed"]) * 1000 + 16
                metrics, rows, training = run_setting(
                    base_dataset=base_dataset,
                    label_config=config,
                    loss_name=loss_name,
                    split=split,
                    seed=shared_seed,
                    args=args,
                    device=device,
                )
                cache[key] = (metrics, rows, training)
                reused = False
            metric_row = {**metrics, "stage": stage, "reused": reused}
            all_metrics.append(metric_row)
            for row in rows:
                all_predictions.append({**row, "stage": stage, "reused": reused})
            training_records.append(
                {
                    "stage": stage,
                    "split_id": split["split_id"],
                    "label_config": label_name,
                    "loss": loss_name,
                    "reused": reused,
                    "train_ids": list(split["train"]),
                    "val_ids": list(split["val"]),
                    "test_ids": list(split["test"]),
                    **training,
                }
            )

    # Step 1: hard + symmetric Gaussian labels, all with BCE.
    for config in label_configs:
        execute(config, "bce", "label")

    label_metrics = [
        row for row in all_metrics
        if row["stage"] == "label" and row["loss"] == "bce"
    ]
    label_summary = aggregate_rows(
        label_metrics, ["label_config", "loss"]
    )
    hard_summary = next(
        row for row in label_summary
        if row["label_config"] == hard_config["name"]
    )
    symmetric_summary = [
        row for row in label_summary
        if row["label_config"] != hard_config["name"]
    ]
    best_symmetric = best_summary_row(symmetric_summary)
    run_asymmetric = (
        args.run_asymmetric_if_soft_improves
        and soft_clearly_improves(hard_summary, best_symmetric)
    )

    asymmetric_configs: list[dict[str, Any]] = []
    if run_asymmetric:
        log(
            "Best symmetric soft label strictly improves both in-interval rate "
            "and MAE; running asymmetric Gaussian follow-up."
        )
        asymmetric_configs = [
            {
                "name": label_config_name("gaussian", pre, post),
                "kind": "gaussian",
                "sigma_pre": pre,
                "sigma_post": post,
            }
            for pre, post in ASYMMETRIC_SIGMAS
        ]
        for config in asymmetric_configs:
            execute(config, "bce", "label")
        label_metrics = [
            row for row in all_metrics
            if row["stage"] == "label" and row["loss"] == "bce"
        ]
        label_summary = aggregate_rows(
            label_metrics, ["label_config", "loss"]
        )
    else:
        log("Asymmetric Gaussian follow-up skipped by the improvement gate.")

    best_label_summary = best_summary_row(label_summary)
    best_label_name = str(best_label_summary["label_config"])
    config_by_name = {
        str(config["name"]): config
        for config in [*label_configs, *asymmetric_configs]
    }
    best_label_config = config_by_name[best_label_name]

    # Step 2: best label + loss ablation. BCE is already cached from step 1.
    for loss_name in LOSS_NAMES:
        execute(best_label_config, loss_name, "loss")
    loss_metrics = [
        row for row in all_metrics if row["stage"] == "loss"
    ]
    loss_summary = aggregate_rows(
        loss_metrics, ["label_config", "loss"]
    )
    best_loss_summary = best_summary_row(loss_summary)

    write_csv(output_dir / "label_ablation.csv", label_summary)
    write_csv(output_dir / "loss_ablation.csv", loss_summary)
    write_csv(output_dir / "per_split_metrics.csv", all_metrics)
    write_csv(output_dir / "per_rollout_predictions.csv", all_predictions)
    write_csv(output_dir / "dataset_targets.csv", target_rows)
    write_json(output_dir / "dataset_target_summary.json", target_summary)
    write_json(output_dir / "training_records.json", training_records)
    write_json(
        output_dir / "split_manifest.json",
        {
            "schema_version": 1,
            "failure_rollout_n": len(base_dataset),
            "random_splits": splits,
            "shared_across_all_configurations": True,
        },
    )

    best_payload = {
        "best_label": dict(best_label_summary),
        "best_loss": dict(best_loss_summary),
        "asymmetric_followup_ran": run_asymmetric,
        "asymmetric_gate": (
            "best symmetric soft label must strictly improve both "
            "mean in-interval rate and mean MAE versus hard weighted interval"
        ),
    }
    write_json(output_dir / "best_configuration.json", best_payload)
    (output_dir / "best_configuration.md").write_text(
        "\n".join(
            [
                "# Best BiLSTM label/loss configuration",
                "",
                f"- Best label: **{best_label_name}**",
                f"- Best loss: **{best_loss_summary['loss']}**",
                f"- Event decay tau: **{args.tau_event:g} native samples**",
                f"- Batch size: **{args.batch_size} rollouts**",
                f"- Asymmetric follow-up ran: **{run_asymmetric}**",
                "",
                "Selection order is highest mean in-interval rate, then lowest "
                "mean MAE, then lowest mean MSE.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        output_dir / "metadata.json",
        {
            "schema_version": 1,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "analysis": "robo_dopamine_bilstm_label_loss_ablation",
            "analysis_mode": "saved-output PyTorch training; no Robo-Dopamine inference",
            "source_root": project_relative(source_root),
            "selection_mode": (
                "latest_usable_signal_per_rollout"
                if latest_per_rollout
                else "single_run_root"
            ),
            "manifest": project_relative(manifest_path),
            "annotation_dir": project_relative(annotation_dir),
            "model": "tiny_bilstm_h16",
            "training_population": "terminal_failure only",
            "event_target_policy": {
                "multi_event": "all valid intervals retained",
                "event_weight": "exp(-(causal_index-first_causal_index)/tau_event)",
                "tau_event_native_samples": args.tau_event,
                "combine_events": "timestep-wise max",
                "no_event_terminal_failure": "pseudo event causal=observable=frame0",
            },
            "label_configs": [
                dict(config) for config in [*label_configs, *asymmetric_configs]
            ],
            "losses": list(LOSS_NAMES),
            "loss_parameters": {
                "distance_weight": args.distance_weight,
                "ranking_weight": args.ranking_weight,
                "ranking_margin": args.ranking_margin,
                "distance_normalization": "native-sample distance / (sequence_length-1)",
                "ranking": (
                    "event-weighted mean interval logit versus log-mean-exp "
                    "background logit"
                ),
            },
            "prediction": "global argmax timestep over full failed rollout",
            "evaluation": (
                "nearest valid interval; any interval counts as in-interval; "
                "first-event in-interval is reported separately"
            ),
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
                "seed": args.seed,
                "train_fraction": args.train_fraction,
                "val_fraction": args.val_fraction,
            },
            "reuse": (
                "step-2 BCE reuses the identical best-label BCE model/results "
                "from step 1; historical first-event hard baseline is not reused "
                "because target construction changed"
            ),
            "dataset_summary": target_summary,
            "best_configuration": best_payload,
            "provenance": provenance,
            "outputs": [
                "label_ablation.csv",
                "loss_ablation.csv",
                "per_split_metrics.csv",
                "per_rollout_predictions.csv",
                "dataset_targets.csv",
                "dataset_target_summary.json",
                "training_records.json",
                "split_manifest.json",
                "best_configuration.json",
                "best_configuration.md",
                "metadata.json",
            ],
        },
    )
    log(f"Wrote label/loss ablation to {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=None)
    parser.add_argument(
        "--run-pool-root",
        default=None,
        help=(
            "Historical Robo-Dopamine output pool. The latest usable fused "
            "result is selected independently per rollout."
        ),
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
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
        "--tau-event",
        type=float,
        default=20.0,
        help="Exponential event-decay constant in native Robo-Dopamine samples.",
    )
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument(
        "--run-asymmetric-if-soft-improves",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, cuda, cuda:0, ...",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1")
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be >= 1")
    if args.batch_size < 1 or args.batch_size > 128:
        raise ValueError("--batch-size must be between 1 and 128")
    if args.tau_event <= 0 or not math.isfinite(args.tau_event):
        raise ValueError("--tau-event must be finite and > 0")
    for name in ("distance_weight", "ranking_weight", "ranking_margin"):
        value = float(getattr(args, name))
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and >= 0")
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
