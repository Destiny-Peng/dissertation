"""Data-driven detector grids for fused-hop stagnation/regression phenotypes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .core import (
    CLEAN_FPR_CONSTRAINTS,
    CONSECUTIVE_NS,
    KOFM_MS,
    RECALL_SAMPLE_WINDOWS,
    REGRESSION_MIN_MS,
    STAGNATION_DELTAS,
    detector_mask,
    make_config,
)


def rolling_window_minimum(
    hops: Sequence[float],
    m: int,
) -> list[float | None]:
    if m < 1:
        raise ValueError("regression window m must be positive")
    values = [float(value) for value in hops]
    result: list[float | None] = [None] * len(values)
    for index in range(m - 1, len(values)):
        result[index] = min(values[index - m + 1 : index + 1])
    return result


def _event_post_indices(
    frames: Sequence[int],
    event: Mapping[str, Any],
) -> list[int]:
    onset = int(event["observable_onset_frame"])
    end = event.get("episode_end_frame")
    end_frame = int(end) if end is not None else None
    return [
        index
        for index, frame in enumerate(frames)
        if frame >= onset and (end_frame is None or frame < end_frame)
    ]


def _candidate_thresholds_for_m(
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
    *,
    m: int,
    max_clean_fpr: float,
) -> tuple[list[float], dict[str, Any]]:
    rolling_by_rollout = {
        rollout_id: rolling_window_minimum(signal["hops"], m)
        for rollout_id, signal in signals.items()
    }
    candidates: set[float] = set()

    def add_horizon_minima(
        rollout_id: str,
        indices: Sequence[int],
    ) -> None:
        rolling = rolling_by_rollout.get(rollout_id)
        if rolling is None:
            return
        horizons: list[int | None] = list(RECALL_SAMPLE_WINDOWS) + [None]
        for horizon in horizons:
            selected = indices if horizon is None else indices[:horizon]
            values = [
                float(rolling[index])
                for index in selected
                if rolling[index] is not None
            ]
            if values:
                threshold = min(values)
                if threshold < 0.0:
                    candidates.add(threshold)

    for event in events:
        rollout_id = str(event["rollout_id"])
        signal = signals.get(rollout_id)
        if signal is None:
            continue
        add_horizon_minima(
            rollout_id,
            _event_post_indices(signal["frames"], event),
        )

    for failure in no_event_failures:
        rollout_id = str(failure["rollout_id"])
        signal = signals.get(rollout_id)
        if signal is None:
            continue
        add_horizon_minima(
            rollout_id,
            list(range(len(signal["hops"]))),
        )

    clean_ids = [
        str(row["rollout_id"])
        for row in clean_rollouts
        if str(row["rollout_id"]) in signals
    ]
    retained: list[float] = []
    fpr_by_threshold: dict[float, float] = {}
    for threshold in sorted(candidates):
        config = make_config(
            "empirical_probe",
            "regression_window_min",
            m=m,
            theta=threshold,
        )
        positives = 0
        for rollout_id in clean_ids:
            if any(detector_mask(signals[rollout_id]["hops"], config)):
                positives += 1
        fpr = positives / len(clean_ids) if clean_ids else 0.0
        fpr_by_threshold[threshold] = fpr
        if fpr <= max_clean_fpr + 1e-12:
            retained.append(threshold)

    metadata = {
        "m": m,
        "raw_candidate_n": len(candidates),
        "retained_candidate_n": len(retained),
        "max_clean_fpr_filter": max_clean_fpr,
        "retained_min": min(retained) if retained else None,
        "retained_max": max(retained) if retained else None,
        "retained_thresholds": retained,
        "retained_clean_fprs": [
            fpr_by_threshold[threshold]
            for threshold in retained
        ],
    }
    return retained, metadata


def build_phenotype_detector_configs(
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build stagnation grids plus empirical short-window regression grids.

    Regression thresholds are actual observed failure-side rolling-window
    minimum values that still keep the regression branch by itself within the
    loosest clean-rollout FPR budget. The final OR ensemble is still evaluated
    against the exact 5/10/20% total clean-FPR constraints.
    """
    configs: list[dict[str, Any]] = []
    index = 0

    def add(family: str, **parameters: Any) -> None:
        nonlocal index
        index += 1
        configs.append(
            make_config(
                f"phen{index:05d}",
                family,
                **parameters,
            )
        )

    for delta in STAGNATION_DELTAS:
        for n in CONSECUTIVE_NS:
            add(
                "stagnation_consecutive",
                delta=delta,
                n=n,
            )

    for delta in STAGNATION_DELTAS:
        for m in KOFM_MS:
            for k in range(1, m + 1):
                add(
                    "stagnation_k_of_m",
                    delta=delta,
                    m=m,
                    k=k,
                )

    max_budget = max(CLEAN_FPR_CONSTRAINTS)
    threshold_metadata: dict[str, Any] = {}
    for m in REGRESSION_MIN_MS:
        thresholds, metadata = _candidate_thresholds_for_m(
            signals,
            events,
            no_event_failures,
            clean_rollouts,
            m=m,
            max_clean_fpr=max_budget,
        )
        threshold_metadata[str(m)] = metadata
        for theta in thresholds:
            add(
                "regression_window_min",
                m=m,
                theta=theta,
            )

    family_counts: dict[str, int] = defaultdict(int)
    for config in configs:
        family_counts[str(config["detector_family"])] += 1

    return configs, {
        "phenotypes": ["stagnation", "regression"],
        "stagnation_families": [
            "stagnation_consecutive",
            "stagnation_k_of_m",
        ],
        "regression_family": "regression_window_min",
        "regression_semantics": "min(h[t-m+1:t]) <= theta_r",
        "regression_window_ms": list(REGRESSION_MIN_MS),
        "regression_threshold_source": (
            "actual observed failure-side rolling-window minima at "
            "Recall@1/@3/@5/@10/@20/eventual horizons; negative values only"
        ),
        "regression_prefilter": (
            "retain only thresholds whose regression branch alone has "
            f"clean-rollout FPR <= {max_budget:.2f}"
        ),
        "family_config_counts": dict(sorted(family_counts.items())),
        "regression_thresholds_by_m": threshold_metadata,
    }
