"""Localization metrics shared by arbitrary experiment specs."""

from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence


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
    candidates = [
        (interval_error(prediction, causal, observable), index)
        for index, (causal, observable) in enumerate(row["intervals"])
    ]
    error, nearest_index = min(candidates, key=lambda item: (abs(item[0]), item[1]))
    first_causal, first_observable = row["first_interval"]
    first_error = interval_error(prediction, first_causal, first_observable)
    nearest = row["events"][nearest_index]
    return {
        "rollout_id": rollout_id,
        "task_key": row["task_key"],
        "task_id": row["task_id"],
        "event_count": int(row["event_count"]),
        "has_pseudo_event": bool(row["has_pseudo_event"]),
        "nearest_event_id": nearest["event_id"],
        "nearest_event_rank": int(nearest["event_rank"]),
        "nearest_event_weight": float(nearest["event_weight"]),
        "predicted_index": int(prediction),
        "predicted_frame": int(row["frames"][prediction]),
        "score": float(score),
        "interval_error_samples": int(error),
        "first_event_interval_error_samples": int(first_error),
        "in_interval": error == 0,
        "first_event_in_interval": first_error == 0,
    }


def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = [int(row["interval_error_samples"]) for row in rows]
    if not errors:
        return {"n": 0}
    absolute = [abs(value) for value in errors]
    result = {
        "n": len(errors),
        "in_interval_rate": sum(value == 0 for value in errors) / len(errors),
        "first_event_in_interval_rate": (
            sum(bool(row["first_event_in_interval"]) for row in rows) / len(errors)
        ),
        "before_interval_rate": sum(value < 0 for value in errors) / len(errors),
        "after_interval_rate": sum(value > 0 for value in errors) / len(errors),
        "median_absolute_interval_error_samples": float(statistics.median(absolute)),
        "mae_samples": float(sum(absolute) / len(errors)),
        "mse_samples": float(sum(value * value for value in errors) / len(errors)),
    }
    for window in (1, 3, 5):
        result[f"within_{window}"] = sum(abs(value) <= window for value in errors) / len(errors)
    return result
