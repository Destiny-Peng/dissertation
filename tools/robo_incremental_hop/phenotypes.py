"""Empirical detector grids for fused-hop stagnation/regression phenotypes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .core import (
    CLEAN_FPR_CONSTRAINTS,
    CONSECUTIVE_NS,
    KOFM_MS,
    RECALL_SAMPLE_WINDOWS,
    REGRESSION_MIN_MS,
    make_config,
)


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


def _stagnation_consecutive_scores(
    hops: Sequence[float],
    n: int,
) -> list[float | None]:
    """Minimum delta needed for the consecutive rule to fire at each sample."""
    values = [abs(float(value)) for value in hops]
    scores: list[float | None] = [None] * len(values)
    for index in range(n - 1, len(values)):
        scores[index] = max(values[index - n + 1 : index + 1])
    return scores


def _stagnation_k_of_m_scores(
    hops: Sequence[float],
    m: int,
    k: int,
) -> list[float | None]:
    """Minimum delta needed for k-of-m stagnation to fire at each sample."""
    values = [abs(float(value)) for value in hops]
    scores: list[float | None] = [None] * len(values)
    for index in range(m - 1, len(values)):
        window = sorted(values[index - m + 1 : index + 1])
        scores[index] = window[k - 1]
    return scores


def _regression_window_min_scores(
    hops: Sequence[float],
    m: int,
) -> list[float | None]:
    """Minimum theta_r needed for rolling-window-min regression to fire."""
    values = [float(value) for value in hops]
    scores: list[float | None] = [None] * len(values)
    for index in range(m - 1, len(values)):
        scores[index] = min(values[index - m + 1 : index + 1])
    return scores


def _critical_value(
    scores: Sequence[float | None],
    indices: Sequence[int],
) -> float | None:
    values = [
        float(scores[index])
        for index in indices
        if 0 <= index < len(scores) and scores[index] is not None
    ]
    return min(values) if values else None


def _empirical_thresholds(
    score_by_rollout: Mapping[str, Sequence[float | None]],
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
    *,
    max_clean_fpr: float,
    require_negative: bool,
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Build operational and oracle threshold views from empirical critical values.

    Oracle keeps every empirical critical threshold because localization is not
    monotone after the positive-episode-start guard: widening a threshold can
    make an alarm start too early. Operational search may safely compress states
    by clean false-positive set because its standard post-onset recall metric is
    monotone, and it only needs branch states feasible under the loosest FPR cap.
    """
    operational_candidates: set[float] = set()
    oracle_candidates: set[float] = set()

    for event in events:
        rollout_id = str(event["rollout_id"])
        signal = signals.get(rollout_id)
        scores = score_by_rollout.get(rollout_id)
        if signal is None or scores is None:
            continue
        post = _event_post_indices(signal["frames"], event)
        for index in post:
            value = scores[index]
            if value is not None:
                oracle_candidates.add(float(value))
        for window in (*RECALL_SAMPLE_WINDOWS, None):
            indices = post if window is None else post[:window]
            value = _critical_value(scores, indices)
            if value is not None:
                operational_candidates.add(value)

    for failure in no_event_failures:
        rollout_id = str(failure["rollout_id"])
        signal = signals.get(rollout_id)
        scores = score_by_rollout.get(rollout_id)
        if signal is None or scores is None:
            continue
        indices = list(range(len(signal["hops"])))
        for index in indices:
            value = scores[index]
            if value is not None:
                oracle_candidates.add(float(value))
        for window in (*RECALL_SAMPLE_WINDOWS, None):
            selected = indices if window is None else indices[:window]
            value = _critical_value(scores, selected)
            if value is not None:
                operational_candidates.add(value)

    clean_ids = [
        str(clean["rollout_id"])
        for clean in clean_rollouts
        if str(clean["rollout_id"]) in score_by_rollout
    ]
    clean_critical: dict[str, float] = {}
    for rollout_id in clean_ids:
        scores = score_by_rollout[rollout_id]
        value = _critical_value(scores, list(range(len(scores))))
        if value is not None:
            clean_critical[rollout_id] = value
            operational_candidates.add(value)

    oracle_thresholds = sorted(
        value
        for value in oracle_candidates
        if not require_negative or value < 0.0
    )
    operational_raw = sorted(
        value
        for value in operational_candidates
        if not require_negative or value < 0.0
    )
    signature_best: dict[tuple[str, ...], float] = {}
    fpr_by_signature: dict[tuple[str, ...], float] = {}
    clean_n = len(clean_ids)

    for threshold in operational_raw:
        signature = tuple(
            sorted(
                rollout_id
                for rollout_id, critical in clean_critical.items()
                if critical <= threshold
            )
        )
        fpr = len(signature) / clean_n if clean_n else 0.0
        if fpr > max_clean_fpr + 1e-12:
            continue
        previous = signature_best.get(signature)
        if previous is None or threshold > previous:
            signature_best[signature] = threshold
            fpr_by_signature[signature] = fpr

    operational_thresholds = sorted(signature_best.values())
    operational_fprs = []
    for threshold in operational_thresholds:
        signature = next(
            signature
            for signature, candidate in signature_best.items()
            if candidate == threshold
        )
        operational_fprs.append(fpr_by_signature[signature])

    return operational_thresholds, oracle_thresholds, {
        "raw_empirical_candidate_n": len(operational_raw),
        "oracle_candidate_n": len(oracle_thresholds),
        "operational_candidate_n": len(operational_thresholds),
        "max_clean_fpr_filter": max_clean_fpr,
        "oracle_min": min(oracle_thresholds) if oracle_thresholds else None,
        "oracle_max": max(oracle_thresholds) if oracle_thresholds else None,
        "operational_min": (
            min(operational_thresholds) if operational_thresholds else None
        ),
        "operational_max": (
            max(operational_thresholds) if operational_thresholds else None
        ),
        "oracle_thresholds": oracle_thresholds,
        "operational_thresholds": operational_thresholds,
        "operational_clean_fprs": operational_fprs,
        "clean_rollout_n": clean_n,
    }


def build_phenotype_detector_configs(
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build operational and oracle empirical phenotype grids separately."""
    configs: list[dict[str, Any]] = []
    oracle_configs: list[dict[str, Any]] = []
    index = 0
    oracle_index = 0
    max_budget = max(CLEAN_FPR_CONSTRAINTS)
    calibration: dict[str, Any] = {}

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

    def add_oracle(family: str, **parameters: Any) -> None:
        nonlocal oracle_index
        oracle_index += 1
        oracle_configs.append(
            make_config(
                f"oracle{oracle_index:05d}",
                family,
                **parameters,
            )
        )

    for n in CONSECUTIVE_NS:
        scores = {
            rollout_id: _stagnation_consecutive_scores(signal["hops"], n)
            for rollout_id, signal in signals.items()
        }
        deltas, oracle_deltas, metadata = _empirical_thresholds(
            scores,
            signals,
            events,
            no_event_failures,
            clean_rollouts,
            max_clean_fpr=max_budget,
            require_negative=False,
        )
        key = f"stagnation_consecutive:n={n}"
        calibration[key] = metadata
        for delta in deltas:
            add(
                "stagnation_consecutive",
                delta=delta,
                n=n,
            )
        for delta in oracle_deltas:
            add_oracle(
                "stagnation_consecutive",
                delta=delta,
                n=n,
            )

    for m in KOFM_MS:
        for k in range(1, m + 1):
            scores = {
                rollout_id: _stagnation_k_of_m_scores(
                    signal["hops"], m, k
                )
                for rollout_id, signal in signals.items()
            }
            deltas, oracle_deltas, metadata = _empirical_thresholds(
                scores,
                signals,
                events,
                no_event_failures,
                clean_rollouts,
                max_clean_fpr=max_budget,
                require_negative=False,
            )
            key = f"stagnation_k_of_m:m={m},k={k}"
            calibration[key] = metadata
            for delta in deltas:
                add(
                    "stagnation_k_of_m",
                    delta=delta,
                    m=m,
                    k=k,
                )
            for delta in oracle_deltas:
                add_oracle(
                    "stagnation_k_of_m",
                    delta=delta,
                    m=m,
                    k=k,
                )

    for m in REGRESSION_MIN_MS:
        scores = {
            rollout_id: _regression_window_min_scores(signal["hops"], m)
            for rollout_id, signal in signals.items()
        }
        thresholds, oracle_thresholds, metadata = _empirical_thresholds(
            scores,
            signals,
            events,
            no_event_failures,
            clean_rollouts,
            max_clean_fpr=max_budget,
            require_negative=True,
        )
        key = f"regression_window_min:m={m}"
        calibration[key] = metadata
        for theta in thresholds:
            add(
                "regression_window_min",
                m=m,
                theta=theta,
            )
        for theta in oracle_thresholds:
            add_oracle(
                "regression_window_min",
                m=m,
                theta=theta,
            )

    family_counts: dict[str, int] = {}
    for config in configs:
        family = str(config["detector_family"])
        family_counts[family] = family_counts.get(family, 0) + 1
    oracle_family_counts: dict[str, int] = {}
    for config in oracle_configs:
        family = str(config["detector_family"])
        oracle_family_counts[family] = oracle_family_counts.get(family, 0) + 1

    return configs, oracle_configs, {
        "phenotypes": ["stagnation", "regression"],
        "stagnation_families": [
            "stagnation_consecutive",
            "stagnation_k_of_m",
        ],
        "regression_family": "regression_window_min",
        "stagnation_semantics": "abs(h_t) <= empirical delta",
        "regression_semantics": "min(h[t-m+1:t]) <= empirical theta_r",
        "regression_window_ms": list(REGRESSION_MIN_MS),
        "threshold_source": (
            "operational thresholds use rule-specific failure critical values at "
            "Recall@1/@3/@5/@10/@20/eventual plus clean critical values; oracle "
            "thresholds use every distinct rule-critical value inside annotated "
            "failure episodes and no-event failure rollouts"
        ),
        "threshold_compression": (
            "operational grid only: keep the largest threshold for each distinct "
            "clean false-positive rollout set after the <=20% branch filter; "
            "oracle grid is never compressed this way because threshold widening "
            "can move a positive-episode start far before onset"
        ),
        "oracle_grid": (
            "retain every empirical critical threshold, including states above "
            "operational clean-FPR budgets, for localization-capacity analysis"
        ),
        "operational_prefilter": (
            "the constrained OR sweep later discards branch states whose "
            f"branch-alone clean-rollout FPR exceeds {max_budget:.2f}; this "
            "does not affect the oracle grid"
        ),
        "family_config_counts": dict(sorted(family_counts.items())),
        "oracle_family_config_counts": dict(sorted(oracle_family_counts.items())),
        "operational_config_n": len(configs),
        "oracle_config_n": len(oracle_configs),
        "calibration": calibration,
    }
