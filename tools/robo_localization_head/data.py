"""Dataset construction for the localization experiment builder."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .core import onset_anchor, sequence_from_signal, task_key


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


def build_failure_dataset(
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    *,
    tau_event: float,
) -> dict[str, dict[str, Any]]:
    if not math.isfinite(tau_event) or tau_event <= 0:
        raise ValueError("target.tau_event must be finite and > 0")

    by_rollout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in events:
        if str(raw.get("outcome") or "") != "terminal_failure":
            continue
        causal = raw.get("causal_onset_frame")
        observable = raw.get("observable_onset_frame")
        if causal is None or observable is None:
            continue
        causal_frame = int(causal)
        observable_frame = int(observable)
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

    result: dict[str, dict[str, Any]] = {}
    for rollout_id in sorted(set(by_rollout) | set(no_event_by_id)):
        signal = signals.get(rollout_id)
        if signal is None:
            continue
        frames = [int(value) for value in signal["frames"]]
        if not frames:
            continue
        mapped: list[dict[str, Any]] = []
        for event in by_rollout.get(rollout_id, []):
            causal = onset_anchor(frames, int(event["causal_onset_frame"]))
            observable = onset_anchor(frames, int(event["observable_onset_frame"]))
            if causal is None or observable is None or causal > observable:
                continue
            mapped.append({
                **event,
                "causal_index": int(causal),
                "observable_index": int(observable),
                "causal_native_frame": int(frames[causal]),
                "observable_native_frame": int(frames[observable]),
            })
        if not mapped and rollout_id in no_event_by_id:
            source = no_event_by_id[rollout_id]
            mapped.append({
                "event_id": f"{rollout_id}::pseudo_no_event_frame0",
                "rollout_id": rollout_id,
                "event_index": -1,
                "failure_type": source.get("failure_type") or "timeout_no_progress",
                "causal_onset_frame": 0,
                "observable_onset_frame": 0,
                "causal_index": 0,
                "observable_index": 0,
                "causal_native_frame": int(frames[0]),
                "observable_native_frame": int(frames[0]),
                "target_source": "pseudo_no_event_frame0",
                "task_key": source.get("task_key") or signal.get("task_key"),
                "task_id": source.get("task_id", signal.get("task_id")),
            })
        if not mapped:
            continue
        mapped.sort(key=lambda e: (
            int(e["causal_index"]),
            int(e["observable_index"]),
            int(e.get("event_index") or 0),
        ))
        first_causal = int(mapped[0]["causal_index"])
        for rank, event in enumerate(mapped):
            delta = int(event["causal_index"]) - first_causal
            event["event_rank"] = rank
            event["event_weight"] = float(math.exp(-float(delta) / tau_event))
        intervals = [
            (int(event["causal_index"]), int(event["observable_index"]))
            for event in mapped
        ]
        support = np.zeros(len(frames), dtype=np.float32)
        for causal, observable in intervals:
            support[causal:observable + 1] = 1.0
        first = mapped[0]
        result[rollout_id] = {
            "rollout_id": rollout_id,
            "kind": "failure",
            "task_key": str(first.get("task_key") or signal.get("task_key") or ""),
            "task_id": first.get("task_id", signal.get("task_id")),
            "frames": np.asarray(frames, dtype=np.int64),
            "sequence": sequence_from_signal(signal),
            "events": mapped,
            "intervals": intervals,
            "first_interval": intervals[0],
            "hard_support": support,
            "distance_to_interval": _distance_to_intervals(len(frames), intervals),
            "event_count": len(mapped),
            "has_pseudo_event": any(
                event["target_source"] == "pseudo_no_event_frame0" for event in mapped
            ),
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
        sequence = sequence_from_signal(signal)
        result[rollout_id] = {
            "rollout_id": rollout_id,
            "kind": "clean_success",
            "task_key": str(clean.get("task_key") or signal.get("task_key") or ""),
            "task_id": clean.get("task_id", signal.get("task_id")),
            "frames": np.asarray(signal["frames"], dtype=np.int64),
            "sequence": sequence,
            "labels": np.zeros(len(sequence), dtype=np.float32),
        }
    return result


def select_success_rollouts(
    success_dataset: Mapping[str, Mapping[str, Any]],
    failure_dataset: Mapping[str, Mapping[str, Any]],
    failure_train_ids: Sequence[str],
    *,
    ratio: float,
    seed: int,
) -> list[str]:
    if ratio <= 0:
        return []
    train_tasks = {task_key(failure_dataset, rollout_id) for rollout_id in failure_train_ids}
    same_task = [
        rollout_id for rollout_id in sorted(success_dataset)
        if task_key(success_dataset, rollout_id) in train_tasks
    ]
    same_task_set = set(same_task)
    fallback = [
        rollout_id for rollout_id in sorted(success_dataset)
        if rollout_id not in same_task_set
    ]
    rng = random.Random(seed)
    rng.shuffle(same_task)
    rng.shuffle(fallback)
    requested = int(len(failure_train_ids) * ratio)
    return (same_task + fallback)[:requested]
