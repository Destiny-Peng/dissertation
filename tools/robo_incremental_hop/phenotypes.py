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
) -> tuple[list[float], dict[str, Any]]:
    """Return only threshold values that change failure evidence or clean FP sets.

    For a fixed temporal rule, detector positivity is monotone in the threshold:
    a sample fires iff its rule-specific critical score <= threshold. Candidate
    thresholds therefore only need to occur at empirical critical values. We
    collect failure critical values at @1/@3/@5/@10/@20/eventual, add clean
    rollout critical values, reject configurations above the loosest clean-FPR
    budget, then keep only the largest threshold for each distinct clean-FP set.
    """
    raw_candidates: set[float] = set()

    for event in events:
        rollout_id = str(event["rollout_id"])
        signal = signals.get(rollout_id)
        scores = score_by_rollout.get(rollout_id)
        if signal is None or scores is None:
            continue
        post = _event_post_indices(signal["frames"], event)
        for window in (*RECALL_SAMPLE_WINDOWS, None):
            indices = post if window is None else post[:window]
            value = _critical_value(scores, indices)
            if value is not None:
                raw_candidates.add(value)

    for failure in no_event_failures:
        rollout_id = str(failure["rollout_id"])
        signal = signals.get(rollout_id)
        scores = score_by_rollout.get(rollout_id)
        if signal is None or scores is None:
            continue
        indices = list(range(len(signal["hops"])))
        for window in (*RECALL_SAMPLE_WINDOWS, None):
            selected = indices if window is None else indices[:window]
            value = _critical_value(scores, selected)
            if value is not None:
                raw_candidates.add(value)

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
            raw_candidates.add(value)

    candidates = sorted(
        value
        for value in raw_candidates
        if not require_negative or value < 0.0
    )
    signature_best: dict[tuple[str, ...], float] = {}
    fpr_by_signature: dict[tuple[str, ...], float] = {}
    clean_n = len(clean_ids)

    for threshold in candidates:
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

    retained = sorted(signature_best.values())
    retained_fprs = []
    for threshold in retained:
        signature = next(
            signature
            for signature, candidate in signature_best.items()
            if candidate == threshold
        )
        retained_fprs.append(fpr_by_signature[signature])

    return retained, {
        "raw_empirical_candidate_n": len(candidates),
        "retained_candidate_n": len(retained),
        "max_clean_fpr_filter": max_clean_fpr,
        "retained_min": min(retained) if retained else None,
        "retained_max": max(retained) if retained else None,
        "retained_thresholds": retained,
        "retained_clean_fprs": retained_fprs,
        "clean_rollout_n": clean_n,
    }


def build_phenotype_detector_configs(
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build fully empirical stagnation and regression threshold grids."""
    configs: list[dict[str, Any]] = []
    index = 0
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

    for n in CONSECUTIVE_NS:
        scores = {
            rollout_id: _stagnation_consecutive_scores(signal["hops"], n)
            for rollout_id, signal in signals.items()
        }
        deltas, metadata = _empirical_thresholds(
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

    for m in KOFM_MS:
        for k in range(1, m + 1):
            scores = {
                rollout_id: _stagnation_k_of_m_scores(
                    signal["hops"], m, k
                )
                for rollout_id, signal in signals.items()
            }
            deltas, metadata = _empirical_thresholds(
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

    for m in REGRESSION_MIN_MS:
        scores = {
            rollout_id: _regression_window_min_scores(signal["hops"], m)
            for rollout_id, signal in signals.items()
        }
        thresholds, metadata = _empirical_thresholds(
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

    family_counts: dict[str, int] = {}
    for config in configs:
        family = str(config["detector_family"])
        family_counts[family] = family_counts.get(family, 0) + 1

    return configs, {
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
            "empirical rule-specific critical values from saved fused-hop failure "
            "and clean rollouts; failure values are sampled at "
            "Recall@1/@3/@5/@10/@20/eventual horizons"
        ),
        "threshold_compression": (
            "for each temporal rule, keep only the largest empirical threshold "
            "for each distinct clean false-positive rollout set"
        ),
        "prefilter": (
            "discard threshold states whose branch-alone clean-rollout FPR "
            f"exceeds {max_budget:.2f}; final OR still uses exact 5/10/20% caps"
        ),
        "family_config_counts": dict(sorted(family_counts.items())),
        "calibration": calibration,
    }
