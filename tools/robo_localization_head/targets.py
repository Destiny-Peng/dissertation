"""Target builders used by localization experiment specs."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np


def make_labels(row: Mapping[str, Any], config: Mapping[str, Any]) -> np.ndarray:
    kind = str(config.get("kind", "hard"))
    sigma_pre = float(config.get("sigma_pre", config.get("sigma", 3.0)))
    sigma_post = float(config.get("sigma_post", config.get("sigma", sigma_pre)))
    length = len(row["frames"])
    labels = np.zeros(length, dtype=np.float32)
    if kind not in {"hard", "gaussian"}:
        raise ValueError(f"unknown target.kind: {kind}")
    if kind == "gaussian" and (sigma_pre <= 0 or sigma_post <= 0):
        raise ValueError("Gaussian sigma must be > 0")

    indices = np.arange(length, dtype=np.float32)
    for event in row["events"]:
        causal = int(event["causal_index"])
        observable = int(event["observable_index"])
        weight = float(event["event_weight"])
        candidate = np.zeros(length, dtype=np.float32)
        if kind == "hard":
            candidate[causal:observable + 1] = weight
        else:
            before = indices < causal
            inside = (indices >= causal) & (indices <= observable)
            after = indices > observable
            candidate[inside] = weight
            candidate[before] = weight * np.exp(
                -((float(causal) - indices[before]) ** 2) / (2.0 * sigma_pre ** 2)
            )
            candidate[after] = weight * np.exp(
                -((indices[after] - float(observable)) ** 2) / (2.0 * sigma_post ** 2)
            )
        labels = np.maximum(labels, candidate)
    return labels.astype(np.float32)


def apply_labels(
    base_dataset: Mapping[str, Mapping[str, Any]],
    target_config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        rollout_id: {**dict(row), "labels": make_labels(row, target_config)}
        for rollout_id, row in base_dataset.items()
    }
