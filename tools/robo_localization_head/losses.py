"""Loss registry for the localization experiment builder."""

from __future__ import annotations

import math
from typing import Any, Mapping
import numpy as np
import torch
import torch.nn.functional as F

LOSS_NAMES = {
    "bce",
    "temporal_softmax_ce",
    "temporal_softmax_ce_distance",
    "temporal_softmax_ce_squared_distance",
    "temporal_softmax_ce_ranking",
    "temporal_softmax_ce_distance_ranking",
}


def _logmeanexp(values: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(values, dim=0) - math.log(max(1, int(values.numel())))


def loss_value(
    logits: torch.Tensor,
    row: Mapping[str, Any],
    labels: torch.Tensor,
    *,
    config: Mapping[str, Any],
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    name = str(config.get("name", "bce"))
    if name == "bce":
        return F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight, reduction="mean"
        )
    if name not in LOSS_NAMES:
        raise ValueError(f"unknown loss.name: {name}")
    total = labels.sum()
    if float(total.detach().cpu()) <= 0:
        raise ValueError("temporal-softmax loss cannot train all-negative success rollouts")
    q = labels / total
    loss = -(q * F.log_softmax(logits, dim=0)).sum()
    p = torch.softmax(logits, dim=0)
    distances = torch.as_tensor(
        np.asarray(row["distance_to_interval"], dtype=np.float32),
        dtype=logits.dtype,
        device=logits.device,
    )
    normalized_distance = distances / float(max(1, len(distances) - 1))
    distance_weight = float(config.get("distance_weight", 1.0))
    ranking_weight = float(config.get("ranking_weight", 1.0))
    ranking_margin = float(config.get("ranking_margin", 1.0))
    if name in {"temporal_softmax_ce_distance", "temporal_softmax_ce_distance_ranking"}:
        loss = loss + distance_weight * (p * normalized_distance).sum()
    elif name == "temporal_softmax_ce_squared_distance":
        loss = loss + distance_weight * (p * normalized_distance.square()).sum()
    if name in {"temporal_softmax_ce_ranking", "temporal_softmax_ce_distance_ranking"}:
        interval_scores = []
        interval_weights = []
        for event in row["events"]:
            causal = int(event["causal_index"])
            observable = int(event["observable_index"])
            interval_scores.append(logits[causal:observable + 1].mean())
            interval_weights.append(float(event["event_weight"]))
        weights = torch.as_tensor(interval_weights, dtype=logits.dtype, device=logits.device)
        positive_score = (torch.stack(interval_scores) * weights).sum() / weights.sum().clamp_min(1e-12)
        support = torch.as_tensor(
            np.asarray(row["hard_support"], dtype=np.float32) > 0.5,
            device=logits.device,
        )
        background = logits[~support]
        if background.numel():
            negative_score = _logmeanexp(background)
            loss = loss + ranking_weight * F.softplus(
                torch.as_tensor(ranking_margin, dtype=logits.dtype, device=logits.device)
                - (positive_score - negative_score)
            )
    return loss
