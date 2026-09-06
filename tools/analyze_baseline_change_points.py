#!/usr/bin/env python3
"""Analyze LF3R baseline signals as local, interpretable change points.

This analysis deliberately keeps the existing baseline raw outputs untouched.
Each native signal is evaluated at native sample centers using three local
features (level, variance, and slope change) over several frame scales.  A
robust score is calibrated from non-onset regions of failure rollouts and from
clean-success rollouts.  Event localization and direction are evaluated only
after the local change has been identified as unusual.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from analyze_baseline_temporal_signals import (  # noqa: E402
    METHODS,
    METHOD_LABELS,
    METHOD_SIGNAL_DIRECTIONS,
    DEFAULT_ANNOTATIONS,
    DEFAULT_MANIFEST,
    PROJECT_ROOT,
    _event_window_bounds,
    _validate_run_sources,
    json_safe,
    load_annotations,
    load_manifest,
    load_method_signal,
    load_selection,
    resolve_path,
    sha256,
    normalize_outcome,
)


FEATURES = ("level", "variance", "slope")
FEATURE_LABELS = {
    "level": "local level change",
    "variance": "local variance change",
    "slope": "local slope change",
}
DEFAULT_SCALES = (8, 16, 32, 64)
THRESHOLDS = ("q90", "q95", "q99")
TOLERANCES = (4, 8, 16, 30)
FAILURE_DIRECTION_FEATURE = "level"


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def robust_location_scale(values: list[float] | np.ndarray) -> tuple[float, float, str]:
    """Return robust center/scale for candidate feature values."""
    clean = _finite(values)
    if len(clean) == 0:
        return math.nan, math.nan, "missing"
    center = float(np.median(clean))
    mad = float(np.median(np.abs(clean - center)))
    if mad > 0.0 and math.isfinite(mad):
        return center, 1.4826 * mad, "mad"
    if len(clean) >= 4:
        q25, q75 = np.percentile(clean, [25, 75])
        iqr_scale = float((q75 - q25) / 1.349)
        if iqr_scale > 0.0 and math.isfinite(iqr_scale):
            return center, iqr_scale, "iqr"
    floor = max(1e-8, 1e-6 * max(1.0, abs(center)))
    return center, floor, "floor"


def _sample_variance(values: np.ndarray) -> float:
    if len(values) < 2:
        return math.nan
    return float(np.var(values, ddof=1))


def _linear_slope(frames: np.ndarray, values: np.ndarray) -> float:
    if len(frames) < 2 or len(np.unique(frames)) < 2:
        return math.nan
    return float(np.polyfit(frames.astype(float), values.astype(float), 1)[0])


def local_feature(
    frames: np.ndarray,
    values: np.ndarray,
    center: int,
    scale: int,
    feature: str,
    min_samples: int,
) -> dict[str, Any] | None:
    """Compute one local left/right feature using native samples only."""
    center = int(center)
    scale = int(scale)
    left_mask = (frames >= center - scale) & (frames < center)
    right_mask = (frames >= center) & (frames < center + scale)
    left_frames = frames[left_mask]
    left_values = values[left_mask]
    right_frames = frames[right_mask]
    right_values = values[right_mask]
    if len(left_values) < min_samples or len(right_values) < min_samples:
        return None
    left_center = float(np.median(left_values))
    right_center = float(np.median(right_values))
    left_variance = _sample_variance(left_values)
    right_variance = _sample_variance(right_values)
    left_slope = _linear_slope(left_frames, left_values)
    right_slope = _linear_slope(right_frames, right_values)
    if feature == "level":
        feature_value = right_center - left_center
    elif feature == "variance":
        epsilon = 1e-12 * max(
            1.0,
            float(np.max(np.abs(values))) if len(values) else 1.0,
        )
        feature_value = math.log((right_variance + epsilon) / (left_variance + epsilon))
    elif feature == "slope":
        feature_value = right_slope - left_slope
    else:
        raise ValueError(f"Unknown local feature: {feature}")
    if not math.isfinite(feature_value):
        return None
    return {
        "center_frame": center,
        "feature_value": float(feature_value),
        "left_count": int(len(left_values)),
        "right_count": int(len(right_values)),
        "left_center": left_center,
        "right_center": right_center,
        "left_variance": left_variance,
        "right_variance": right_variance,
        "left_slope": left_slope,
        "right_slope": right_slope,
    }


def build_candidates(
    series: dict[str, Any],
    scale: int,
    feature: str,
    min_samples: int,
) -> list[dict[str, Any]]:
    frames = np.asarray(series["frames"], dtype=int)
    values = np.asarray(series["values"], dtype=float)
    candidates = []
    for center in frames.tolist():
        candidate = local_feature(frames, values, center, scale, feature, min_samples)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def candidate_score(candidate: dict[str, Any], center: float, scale: float) -> float:
    return abs(float(candidate["feature_value"]) - center) / scale


def score_candidates(
    candidates: list[dict[str, Any]],
    center: float,
    scale: float,
) -> list[dict[str, Any]]:
    return [
        {
            **candidate,
            "unusual_score": float(candidate_score(candidate, center, scale)),
        }
        for candidate in candidates
    ]


def nearest_candidate(
    candidates: list[dict[str, Any]],
    target: int,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(int(row["center_frame"]) - int(target)))


def bounded_candidates(
    candidates: list[dict[str, Any]],
    low: int,
    high: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in candidates
        if int(low) <= int(row["center_frame"]) < int(high)
    ]


def quantile_dict(values: list[float]) -> dict[str, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {name: math.nan for name in THRESHOLDS}
    return {
        "q90": float(np.quantile(clean, 0.90)),
        "q95": float(np.quantile(clean, 0.95)),
        "q99": float(np.quantile(clean, 0.99)),
    }


def sign_label(value: Any) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number == 0:
        return "flat"
    return "increase" if number > 0 else "decrease"


def _median(values: list[Any]) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(clean)) if clean else math.nan


def _mean(values: list[Any]) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else math.nan


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _rank_auc(scores: list[float], labels: list[int]) -> float:
    """Compute AUROC without adding scipy/sklearn dependencies."""
    pairs = [
        (float(score), int(label))
        for score, label in zip(scores, labels)
        if math.isfinite(float(score))
    ]
    positives = sum(label == 1 for _, label in pairs)
    negatives = sum(label == 0 for _, label in pairs)
    if not positives or not negatives:
        return math.nan
    ordered = sorted(pairs, key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label == 1 for _, label in ordered[index:end])
        index = end
    return float(
        (rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _average_precision(scores: list[float], labels: list[int]) -> float:
    """Compute average precision for event peaks versus clean pseudo-events."""
    pairs = [
        (float(score), int(label))
        for score, label in zip(scores, labels)
        if math.isfinite(float(score))
    ]
    positives = sum(label == 1 for _, label in pairs)
    if not positives:
        return math.nan
    ordered = sorted(pairs, key=lambda pair: pair[0], reverse=True)
    seen = 0
    precision_sum = 0.0
    for position, (_, label) in enumerate(ordered, 1):
        if label:
            seen += 1
            precision_sum += seen / position
    return float(precision_sum / positives)


def event_metric_summary(
    rows: list[dict[str, Any]],
    reference: dict[str, Any],
    feature: str,
    method: str,
    threshold_name: str,
) -> dict[str, Any]:
    """Summarize thresholded localization and clean-success discrimination."""
    valid = [
        row
        for row in rows
        if row.get("feature_valid")
        and row.get("peak_score") is not None
        and math.isfinite(float(row["peak_score"]))
    ]
    hits = [row for row in valid if row.get("hit")]
    peak_distances = [
        float(row["peak_distance_frames"])
        for row in valid
        if row.get("peak_distance_frames") is not None
    ]
    first_errors = [
        float(row["first_exceedance_absolute_error_frames"])
        for row in hits
        if row.get("first_exceedance_absolute_error_frames") is not None
    ]
    onset_unusual = [
        row
        for row in valid
        if row.get("onset_score") is not None
        and math.isfinite(float(row["onset_score"]))
    ]
    onset_unusual_hits = [row for row in onset_unusual if row.get("onset_near_unusual")]
    unusual_peak_rows = [
        row for row in hits if row.get("peak_feature_value") is not None
    ]
    tolerance_rates = {}
    for tolerance in TOLERANCES:
        tolerance_rates[f"peak_hit_rate_within_{tolerance}_frames"] = _rate(
            sum(
                bool(row.get("peak_distance_frames") is not None)
                and float(row["peak_distance_frames"]) <= tolerance
                and bool(row.get("hit"))
                for row in valid
            ),
            len(valid),
        )
        tolerance_rates[f"first_hit_rate_within_{tolerance}_frames"] = _rate(
            sum(
                bool(row.get("first_exceedance_absolute_error_frames") is not None)
                and float(row["first_exceedance_absolute_error_frames"]) <= tolerance
                for row in hits
            ),
            len(valid),
        )

    direction_consistent = []
    positive_fraction = []
    for row in unusual_peak_rows:
        value = float(row["peak_feature_value"])
        if feature == FAILURE_DIRECTION_FEATURE:
            direction_consistent.append(
                METHOD_SIGNAL_DIRECTIONS[method] * value > 0.0
            )
        else:
            positive_fraction.append(value > 0.0)

    clean_scores = [
        float(value)
        for value in reference.get("clean_peak_scores", [])
        if math.isfinite(float(value))
    ]
    threshold_value = reference.get("thresholds", {}).get(threshold_name, math.nan)
    threshold_is_finite = math.isfinite(float(threshold_value))
    false_alarm_count = (
        int(sum(score >= float(threshold_value) for score in clean_scores))
        if threshold_is_finite
        else 0
    )
    true_positives = int(len(hits))
    false_positives = false_alarm_count
    false_negatives = int(len(valid) - len(hits))
    precision = (
        _rate(true_positives, true_positives + false_positives)
        if true_positives + false_positives
        else math.nan
    )
    recall = _rate(true_positives, true_positives + false_negatives)
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if math.isfinite(precision)
        and math.isfinite(recall)
        and precision + recall > 0.0
        else math.nan
    )
    positive_scores = [float(row["peak_score"]) for row in valid]
    discrimination_scores = positive_scores + clean_scores
    discrimination_labels = [1] * len(positive_scores) + [0] * len(clean_scores)

    summary = {
        "n_events": int(len(valid)),
        "n_hits": int(len(hits)),
        "n_misses": int(len(valid) - len(hits)),
        "recall": recall,
        "event_hit_rate": recall,
        "n_events_with_onset_candidate": int(len(onset_unusual)),
        "onset_near_unusual_count": int(len(onset_unusual_hits)),
        "onset_near_unusual_rate": _rate(
            len(onset_unusual_hits),
            len(onset_unusual),
        ),
        "median_peak_score": _median([row.get("peak_score") for row in valid]),
        "median_peak_distance_frames": _median(peak_distances),
        "mean_peak_distance_frames": _mean(peak_distances),
        "median_first_exceedance_absolute_error_frames": _median(first_errors),
        "mean_first_exceedance_absolute_error_frames": _mean(first_errors),
        "n_early_peak_events": int(
            sum(
                row.get("peak_signed_offset_frames") is not None
                and float(row["peak_signed_offset_frames"]) < 0
                for row in valid
            )
        ),
        "early_peak_rate": _rate(
            sum(
                row.get("peak_signed_offset_frames") is not None
                and float(row["peak_signed_offset_frames"]) < 0
                for row in valid
            ),
            len(valid),
        ),
        "n_early_first_exceedances": int(
            sum(
                row.get("first_exceedance_signed_offset_frames") is not None
                and float(row["first_exceedance_signed_offset_frames"]) < 0
                for row in hits
            )
        ),
        "early_first_exceedance_rate": _rate(
            sum(
                row.get("first_exceedance_signed_offset_frames") is not None
                and float(row["first_exceedance_signed_offset_frames"]) < 0
                for row in hits
            ),
            len(hits),
        ),
        "reference_rollouts": reference["reference_rollouts"],
        "reference_candidate_count": reference["reference_candidate_count"],
        "reference_feature_median": reference["feature_median"],
        "reference_feature_q95": reference["feature_q95"],
        "reference_peak_median": reference["peak_median"],
        "reference_peak_q95": reference["peak_q95"],
        "clean_success_rollouts": reference["clean_success_rollouts"],
        "clean_success_false_alarm_rate": reference[
            "clean_success_false_alarm_rate"
        ].get(threshold_name, math.nan),
        "same_rollout_non_onset_false_alarm_rate": reference[
            "same_rollout_non_onset_false_alarm_rate"
        ].get(threshold_name, math.nan),
        "clean_success_false_alarm_count": reference[
            "clean_success_false_alarm_count"
        ].get(threshold_name, 0),
        "same_rollout_non_onset_false_alarm_count": reference[
            "same_rollout_non_onset_false_alarm_count"
        ].get(threshold_name, 0),
        "non_onset_rollouts": reference["non_onset_rollouts"],
        "clean_pseudo_events": int(len(clean_scores)),
        "false_alarm_count": false_alarm_count,
        "false_alarm_rate": _rate(false_alarm_count, len(clean_scores)),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "f1": f1,
        "auroc": _rank_auc(discrimination_scores, discrimination_labels),
        "average_precision": _average_precision(
            discrimination_scores,
            discrimination_labels,
        ),
        "threshold_source": reference.get("threshold_source"),
        "failure_direction_consistency_after_unusual": (
            _rate(sum(direction_consistent), len(direction_consistent))
            if feature == FAILURE_DIRECTION_FEATURE
            else math.nan
        ),
        "positive_change_fraction_after_unusual": (
            _rate(sum(positive_fraction), len(positive_fraction))
            if feature != FAILURE_DIRECTION_FEATURE
            else math.nan
        ),
        "negative_change_fraction_after_unusual": (
            _rate(
                len(positive_fraction) - sum(positive_fraction),
                len(positive_fraction),
            )
            if feature != FAILURE_DIRECTION_FEATURE
            else math.nan
        ),
    }
    summary.update(tolerance_rates)
    return summary


def event_rows_for_key(
    *,
    key: tuple[str, str, int, str],
    candidates_by_rollout: dict[str, list[dict[str, Any]]],
    scored_by_rollout: dict[str, list[dict[str, Any]]],
    rollouts: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
    reference: dict[str, Any],
    pre_window_frames: int,
    post_window_frames: int,
) -> list[dict[str, Any]]:
    method, signal, scale, feature = key
    rows: list[dict[str, Any]] = []
    for rollout_id, rollout in rollouts.items():
        candidates = candidates_by_rollout.get(rollout_id)
        scored = scored_by_rollout.get(rollout_id)
        if not candidates or not scored:
            continue
        events = rollout["events"]
        for event_position, event in enumerate(events):
            event_frame = int(event["observable_onset_frame"])
            next_event_frame = (
                int(events[event_position + 1]["observable_onset_frame"])
                if event_position + 1 < len(events)
                else None
            )
            low, high = _event_window_bounds(
                event_frame=event_frame,
                total_frames=int(rollout["record"]["total_frames"]),
                pre_window_frames=pre_window_frames,
                post_window_frames=post_window_frames,
                recovery_frame=event.get("recovery_frame"),
                terminal_failure_frame=event.get("terminal_failure_frame"),
                next_event_frame=next_event_frame,
            )
            window = bounded_candidates(scored, low, high)
            onset_candidate = nearest_candidate(
                [row for row in scored if abs(int(row["center_frame"]) - event_frame) <= scale],
                event_frame,
            )
            onset_score = (
                float(onset_candidate["unusual_score"])
                if onset_candidate is not None
                else None
            )
            onset_feature_value = (
                float(onset_candidate["feature_value"])
                if onset_candidate is not None
                else None
            )
            peak = max(window, key=lambda row: row["unusual_score"]) if window else None
            first_by_threshold = {}
            for threshold_name, threshold_value in thresholds.items():
                if not math.isfinite(float(threshold_value)):
                    first_by_threshold[threshold_name] = None
                    continue
                crossings = [
                    row
                    for row in window
                    if float(row["unusual_score"]) >= float(threshold_value)
                ]
                first_by_threshold[threshold_name] = (
                    min(crossings, key=lambda row: int(row["center_frame"]))
                    if crossings
                    else None
                )
            for threshold_name, threshold_value in thresholds.items():
                first = first_by_threshold[threshold_name]
                peak_frame = int(peak["center_frame"]) if peak else None
                peak_offset = peak_frame - event_frame if peak_frame is not None else None
                first_frame = int(first["center_frame"]) if first else None
                first_offset = first_frame - event_frame if first_frame is not None else None
                peak_value = float(peak["feature_value"]) if peak else None
                peak_score = float(peak["unusual_score"]) if peak else None
                hit = first is not None
                direction_consistent = None
                if (
                    feature == FAILURE_DIRECTION_FEATURE
                    and hit
                    and peak_value is not None
                ):
                    direction_consistent = bool(
                        METHOD_SIGNAL_DIRECTIONS[method] * peak_value > 0.0
                    )
                rows.append({
                    "method": method,
                    "signal": signal,
                    "scale_frames": int(scale),
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "threshold": threshold_name,
                    "threshold_value": float(threshold_value),
                    "rollout_id": rollout_id,
                    "event_index": int(event["event_index"]),
                    "event_type": "annotated_event",
                    "outcome_group": normalize_outcome(rollout["annotation"]),
                    "failure_type": event.get("failure_type", "other"),
                    "event_frame": event_frame,
                    "observable_onset_frame": event_frame,
                    "recovery_frame": event.get("recovery_frame"),
                    "terminal_failure_frame": event.get("terminal_failure_frame"),
                    "next_event_frame": next_event_frame,
                    "event_window_low": int(low),
                    "event_window_high_exclusive": int(high),
                    "onset_candidate_frame": (
                        int(onset_candidate["center_frame"])
                        if onset_candidate is not None
                        else None
                    ),
                    "onset_candidate_distance_frames": (
                        abs(int(onset_candidate["center_frame"]) - event_frame)
                        if onset_candidate is not None
                        else None
                    ),
                    "onset_feature_value": onset_feature_value,
                    "onset_score": onset_score,
                    "peak_frame": peak_frame,
                    "peak_feature_value": peak_value,
                    "peak_score": peak_score,
                    "peak_signed_offset_frames": peak_offset,
                    "peak_distance_frames": abs(peak_offset) if peak_offset is not None else None,
                    "first_exceedance_frame": first_frame,
                    "first_exceedance_signed_offset_frames": first_offset,
                    "first_exceedance_absolute_error_frames": (
                        abs(first_offset) if first_offset is not None else None
                    ),
                    "feature_valid": bool(window),
                    "hit": bool(hit),
                    "miss": not bool(hit),
                    "early_peak": bool(peak_offset is not None and peak_offset < 0),
                    "early_first_exceedance": bool(first_offset is not None and first_offset < 0),
                    "onset_near_unusual": bool(
                        onset_score is not None
                        and math.isfinite(onset_score)
                        and onset_score >= float(threshold_value)
                    ),
                    "direction_after_unusual": sign_label(peak_value) if hit else None,
                    "failure_direction_consistent": direction_consistent,
                    "native_samples_only": True,
                    "clean_reference_peak_q95": reference["peak_q95"],
                    "same_rollout_reference_peak_q95": reference["non_onset_peak_q95"],
                    "clean_success_false_alarm_rate": reference["clean_success_false_alarm_rate"].get(threshold_name, math.nan),
                    "same_rollout_non_onset_false_alarm_rate": reference["same_rollout_non_onset_false_alarm_rate"].get(threshold_name, math.nan),
                    "task_suite": rollout["record"].get("task_suite"),
                    "task_id": rollout["record"].get("task_id"),
                    "task_description": rollout["record"].get("task_description"),
                    "source_kind": rollout["record"].get("source_kind"),
                    "analysis_partition": rollout["record"].get("analysis_partition"),
                    "dataset_role": rollout["record"].get("dataset_role"),
                })
    return rows


def _reference_candidates(
    candidates: list[dict[str, Any]],
    onset_frames: list[int],
    exclusion_radius: int,
) -> list[dict[str, Any]]:
    if not onset_frames:
        return candidates
    return [
        row
        for row in candidates
        if all(
            abs(int(row["center_frame"]) - int(onset)) > int(exclusion_radius)
            for onset in onset_frames
        )
    ]


def _reference_stats(
    *,
    candidates_by_rollout: dict[str, list[dict[str, Any]]],
    rollouts: dict[str, dict[str, Any]],
    location: float,
    scale: float,
    exclusion_radius: int,
) -> dict[str, Any]:
    reference_values: list[float] = []
    clean_peak_scores: list[float] = []
    non_onset_peak_scores: list[float] = []
    clean_rollouts = 0
    non_onset_rollouts = 0
    reference_candidate_count = 0
    for rollout_id, candidates in candidates_by_rollout.items():
        rollout = rollouts[rollout_id]
        onset_frames = [
            int(event["observable_onset_frame"])
            for event in rollout["events"]
        ]
        outcome = normalize_outcome(rollout["annotation"])
        if not onset_frames and outcome != "clean_success":
            continue
        reference_candidates = _reference_candidates(
            candidates, onset_frames, exclusion_radius
        )
        if not reference_candidates:
            continue
        reference_values.extend(
            float(row["feature_value"]) for row in reference_candidates
        )
        reference_candidate_count += len(reference_candidates)
        scored = score_candidates(reference_candidates, location, scale)
        peak = max(float(row["unusual_score"]) for row in scored)
        if outcome == "clean_success":
            clean_rollouts += 1
            clean_peak_scores.append(peak)
        elif onset_frames:
            non_onset_rollouts += 1
            non_onset_peak_scores.append(peak)
    combined_peaks = clean_peak_scores + non_onset_peak_scores
    if clean_peak_scores:
        thresholds = quantile_dict(clean_peak_scores)
        threshold_source = "clean_success_pseudo_events"
    else:
        thresholds = quantile_dict(combined_peaks)
        threshold_source = "combined_reference_fallback"
    clean_false_alarm = {
        name: _rate(
            sum(score >= thresholds[name] for score in clean_peak_scores),
            len(clean_peak_scores),
        )
        for name in THRESHOLDS
    }
    non_onset_false_alarm = {
        name: _rate(
            sum(score >= thresholds[name] for score in non_onset_peak_scores),
            len(non_onset_peak_scores),
        )
        for name in THRESHOLDS
    }
    feature_q = quantile_dict(reference_values)
    peak_q = quantile_dict(combined_peaks)
    non_onset_peak_q = quantile_dict(non_onset_peak_scores)
    return {
        "reference_rollouts": int(clean_rollouts + non_onset_rollouts),
        "reference_candidate_count": int(reference_candidate_count),
        "feature_median": _median(reference_values),
        "feature_q25": float(np.quantile(reference_values, 0.25)) if reference_values else math.nan,
        "feature_q75": float(np.quantile(reference_values, 0.75)) if reference_values else math.nan,
        "feature_q95": feature_q["q95"],
        "feature_location": float(location),
        "feature_scale": float(scale),
        "feature_scale_source": None,
        "peak_median": _median(combined_peaks),
        "peak_q90": peak_q["q90"],
        "peak_q95": peak_q["q95"],
        "peak_q99": peak_q["q99"],
        "clean_success_rollouts": int(clean_rollouts),
        "clean_success_pseudo_events": int(len(clean_peak_scores)),
        "clean_success_peak_median": _median(clean_peak_scores),
        "clean_success_peak_q95": quantile_dict(clean_peak_scores)["q95"],
        "non_onset_rollouts": int(non_onset_rollouts),
        "non_onset_peak_median": _median(non_onset_peak_scores),
        "non_onset_peak_q95": non_onset_peak_q["q95"],
        "clean_success_false_alarm_rate": clean_false_alarm,
        "same_rollout_non_onset_false_alarm_rate": non_onset_false_alarm,
        "clean_success_false_alarm_count": {
            name: int(sum(score >= thresholds[name] for score in clean_peak_scores))
            for name in THRESHOLDS
        },
        "same_rollout_non_onset_false_alarm_count": {
            name: int(sum(score >= thresholds[name] for score in non_onset_peak_scores))
            for name in THRESHOLDS
        },
        "thresholds": thresholds,
        "threshold_source": threshold_source,
        # Kept in memory for event-level precision/F1/AUROC/AP; excluded from tables.
        "clean_peak_scores": clean_peak_scores,
    }


def _format_source_roots(source_roots: dict[str, list[Path]]) -> dict[str, Any]:
    return {
        method: (
            [str(path) for path in roots]
            if len(roots) != 1
            else str(roots[0])
        )
        for method, roots in source_roots.items()
    }


def load_rollout_data(
    *,
    selection_path: Path,
    manifest_path: Path,
    annotation_dir: Path,
    source_roots: dict[str, list[Path]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    selection_doc, selections = load_selection(selection_path)
    manifest = load_manifest(manifest_path)
    selected_ids = [row["id"] for row in selections]
    annotations: dict[str, dict[str, Any]] = {}
    rollouts: dict[str, dict[str, Any]] = {}
    for selection in selections:
        rollout_id = selection["id"]
        if rollout_id not in manifest:
            raise KeyError(f"Selection ID is not in manifest: {rollout_id}")
        annotation = load_annotations(annotation_dir, rollout_id)
        annotations[rollout_id] = annotation
        observable_events = []
        for event_index, event in enumerate(annotation["failure_events"]):
            if event.get("observable_onset_frame") is None:
                continue
            observable_events.append({
                "event_index": int(event_index),
                "observable_onset_frame": int(event["observable_onset_frame"]),
                "failure_type": event.get(
                    "failure_type",
                    annotation.get("failure_type", "other"),
                ),
                "recovery_frame": event.get("recovery_frame"),
                "terminal_failure_frame": event.get("terminal_failure_frame"),
            })
        rollouts[rollout_id] = {
            "record": manifest[rollout_id],
            "selection": selection,
            "annotation": annotation,
            "events": observable_events,
            "methods": {},
            "method_errors": {},
        }
    source_by_id, run_metadata = _validate_run_sources(
        source_roots,
        set(selected_ids),
    )
    for rollout_id, rollout in rollouts.items():
        for method in METHODS:
            root = source_by_id[method].get(rollout_id)
            if root is None:
                rollout["method_errors"][method] = (
                    f"rollout {rollout_id} is absent from the selected {method} run(s)"
                )
                continue
            try:
                rollout["methods"][method] = load_method_signal(
                    method,
                    root,
                    rollout_id,
                    rollout["record"],
                )
            except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                rollout["method_errors"][method] = str(error)
    return rollouts, run_metadata, selections


def project_path(value: str | Path) -> Path:
    path = resolve_path(value)
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"Path must remain inside project root: {path}") from error
    return path


def parse_scales(value: str) -> tuple[int, ...]:
    result = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            scale = int(token)
        except ValueError as error:
            raise ValueError(f"Invalid local scale: {token}") from error
        if scale < 2:
            raise ValueError("Local scales must be integers >= 2")
        result.append(scale)
    if not result:
        raise ValueError("At least one local scale is required")
    return tuple(sorted(set(result)))


def _safe_table_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else math.nan
    return value


def write_tables(
    *,
    output_dir: Path,
    event_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    failure_type_rows: list[dict[str, Any]],
    scale_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> None:
    event_payload = "\n".join(
        json.dumps(json_safe(row), ensure_ascii=False)
        for row in event_rows
    )
    if event_payload:
        event_payload += "\n"
    for filename in (
        "changepoint_event_metrics.jsonl",
        "localization_event_metrics.jsonl",
    ):
        (output_dir / filename).write_text(event_payload, encoding="utf-8")

    summary_frame = pd.DataFrame(summary_rows)
    for filename in (
        "changepoint_summary.csv",
        "localization_summary.csv",
    ):
        summary_frame.to_csv(output_dir / filename, index=False)

    reference_frame = pd.DataFrame(reference_rows)
    reference_frame.to_csv(
        output_dir / "changepoint_reference_summary.csv",
        index=False,
    )

    failure_type_frame = pd.DataFrame(failure_type_rows)
    for filename in (
        "changepoint_by_failure_type.csv",
        "localization_by_failure_type.csv",
    ):
        failure_type_frame.to_csv(output_dir / filename, index=False)

    scale_frame = pd.DataFrame(scale_rows)
    scale_frame.to_csv(output_dir / "changepoint_scales.csv", index=False)
    scale_frame.to_csv(output_dir / "localization_thresholds.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(
        output_dir / "method_coverage.csv",
        index=False,
    )


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _comparison_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def write_comparison_table(
    *,
    output_dir: Path,
    change_point_summary: pd.DataFrame,
    previous_snapshot: Path | None,
    current_global_snapshot: Path | None,
) -> Path:
    """Write a transparent comparison between legacy global and local metrics."""
    rows: list[dict[str, Any]] = []
    previous_snapshot = previous_snapshot or Path()
    current_global_snapshot = current_global_snapshot or Path()
    previous_summary = _read_optional_csv(
        previous_snapshot / "summary_by_method_signal_outcome.csv"
    )
    current_summary = _read_optional_csv(
        current_global_snapshot / "summary_by_method_signal_outcome.csv"
    )
    global_metrics = (
        ("response", "normalized_response_magnitude_median"),
        ("persistence", "post_event_persistence_fraction_median"),
        ("recovery", "recovery_fraction_toward_baseline_median"),
    )
    global_keys = ["method", "signal", "outcome_group"]
    for family, metric in global_metrics:
        if metric not in previous_summary.columns and metric not in current_summary.columns:
            continue
        current_fields = [field for field in [*global_keys, metric] if field in current_summary.columns]
        previous_fields = [field for field in [*global_keys, metric] if field in previous_summary.columns]
        if len(current_fields) < len(global_keys) + 1 and len(previous_fields) < len(global_keys) + 1:
            continue
        current = current_summary[current_fields].copy() if current_fields else pd.DataFrame()
        previous = previous_summary[previous_fields].copy() if previous_fields else pd.DataFrame()
        if metric in current.columns:
            current = current.rename(columns={metric: "current_value"})
        else:
            current["current_value"] = math.nan
        if metric in previous.columns:
            previous = previous.rename(columns={metric: "previous_value"})
        else:
            previous["previous_value"] = math.nan
        merged = current.merge(previous, on=global_keys, how="outer")
        for _, item in merged.iterrows():
            current_value = _comparison_number(item.get("current_value"))
            previous_value = _comparison_number(item.get("previous_value"))
            if not (math.isfinite(current_value) or math.isfinite(previous_value)):
                continue
            rows.append({
                "comparison_family": family,
                "method": item.get("method"),
                "signal": item.get("signal"),
                "feature": None,
                "scale_frames": None,
                "outcome_group": item.get("outcome_group"),
                "threshold": None,
                "metric": metric,
                "current_value": current_value,
                "previous_full_136_value": previous_value,
                "delta_from_previous": (
                    current_value - previous_value
                    if math.isfinite(current_value) and math.isfinite(previous_value)
                    else math.nan
                ),
                "current_source": str(current_global_snapshot) if current_global_snapshot.is_dir() else None,
                "previous_source": str(previous_snapshot) if previous_snapshot.is_dir() else None,
                "comparison_note": "High-resolution global temporal summary versus the legacy full_136 summary.",
            })

    local_metrics = (
        "recall",
        "false_alarm_rate",
        "precision",
        "f1",
        "auroc",
        "average_precision",
        "clean_success_false_alarm_rate",
        "same_rollout_non_onset_false_alarm_rate",
        "median_peak_distance_frames",
        "median_first_exceedance_absolute_error_frames",
        "peak_hit_rate_within_8_frames",
        "first_hit_rate_within_8_frames",
        "onset_near_unusual_rate",
    )
    if not change_point_summary.empty:
        local = change_point_summary[
            (change_point_summary["threshold"] == "q95")
            & (change_point_summary["outcome_group"] == "all_events")
        ]
        for _, item in local.iterrows():
            for metric in local_metrics:
                current_value = _comparison_number(item.get(metric))
                if not math.isfinite(current_value):
                    continue
                rows.append({
                    "comparison_family": "local_change_point",
                    "method": item.get("method"),
                    "signal": item.get("signal"),
                    "feature": item.get("feature"),
                    "scale_frames": item.get("scale_frames"),
                    "outcome_group": item.get("outcome_group"),
                    "threshold": item.get("threshold"),
                    "metric": metric,
                    "current_value": current_value,
                    "previous_full_136_value": math.nan,
                    "delta_from_previous": math.nan,
                    "current_source": str(output_dir),
                    "previous_source": str(previous_snapshot) if previous_snapshot.is_dir() else None,
                    "comparison_note": "New native-sample local change-point metric; full_136 has no equivalent metric definition.",
                })

    columns = [
        "comparison_family",
        "method",
        "signal",
        "feature",
        "scale_frames",
        "outcome_group",
        "threshold",
        "metric",
        "current_value",
        "previous_full_136_value",
        "delta_from_previous",
        "current_source",
        "previous_source",
        "comparison_note",
    ]
    output_path = output_dir / "comparison_with_full_136_20260827.csv"
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
    return output_path


def plot_change_point_summaries(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
) -> list[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return []
    q95 = summary[
        (summary["threshold"] == "q95")
        & (summary["outcome_group"] == "all_events")
    ].copy()
    if q95.empty:
        return []
    q95["series"] = q95.apply(
        lambda row: f"{METHOD_LABELS.get(row['method'], row['method'])} / {row['signal']} / {row['feature']}",
        axis=1,
    )
    outputs: list[str] = []
    for metric, ylabel, filename, title in (
        (
            "recall",
            "event recall",
            "changepoint_recall_q95_by_scale.png",
            "Q95 local-change event recall by temporal scale",
        ),
        (
            "median_peak_distance_frames",
            "median peak distance (video frames)",
            "changepoint_peak_distance_q95_by_scale.png",
            "Q95 local-change peak distance to observable onset",
        ),
        (
            "onset_near_unusual_rate",
            "onset-near unusual rate",
            "changepoint_onset_near_q95_by_scale.png",
            "Q95 unusual local changes near observable onset",
        ),
    ):
        fig, ax = plt.subplots(figsize=(15, 8))
        plotted = 0
        for series, group in q95.groupby("series", sort=False):
            group = group.sort_values("scale_frames")
            y = pd.to_numeric(group[metric], errors="coerce")
            if y.notna().sum() == 0:
                continue
            ax.plot(
                group["scale_frames"],
                y,
                marker="o",
                linewidth=1.2,
                markersize=3.5,
                label=series,
            )
            plotted += 1
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("local half-window scale (video frames)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if "recall" in metric or "rate" in metric:
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        path = plot_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(str(path.relative_to(output_dir)))
    return outputs


def write_report(
    *,
    output_dir: Path,
    selection_path: Path,
    manifest_path: Path,
    annotation_dir: Path,
    selections: list[dict[str, Any]],
    rollouts: dict[str, dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    source_roots: dict[str, list[Path]],
    source_metadata: dict[str, dict[str, Any]],
    scales: tuple[int, ...],
    pre_window_frames: int,
    post_window_frames: int,
    exclusion_radius: int | None,
    min_samples: int,
    summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    plot_paths: list[str],
    counts: dict[str, int],
    comparison_path: Path | None,
) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "n/a"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"

    method_coverage_text = "; ".join(
        f"{METHOD_LABELS[row['method']]} {row['available_rollouts']}/{row['selected_rollouts']}"
        for row in coverage_rows
    )
    exclusion_text = (
        "the current local scale (per-scale exclusion)"
        if exclusion_radius is None
        else f"{exclusion_radius} frames"
    )
    lines = [
        "# LF3R Local Change-Point Analysis",
        "",
        f"Based on {len(selections)} selected rollouts and existing baseline raw outputs; no GPU inference was rerun and no raw file was modified.",
        "",
        "## Scope and coverage",
        "",
        f"- Selection: {selection_path}; manifest: {manifest_path}; annotations: {annotation_dir}.",
        f"- Rollouts: {len(selections)}; observable-onset events: {counts['observable_events']}; clean-success rollouts: {counts['clean_success_rollouts']}.",
        f"- Method raw coverage: {method_coverage_text}. Missing raw outputs are excluded from feature and event denominators and remain in method_coverage.csv.",
        f"- Input runs: {json.dumps(_format_source_roots(source_roots), ensure_ascii=False)}.",
        "",
        "## Change-point definition",
        "",
        f"- Local half-window scales: {', '.join(str(scale) for scale in scales)} video frames; each feature compares native samples in [center-scale, center) and [center, center+scale).",
        f"- Minimum native samples per side: {min_samples}. No interpolation, resampling, or frame synthesis is used.",
        "- level: right-window median minus left-window median.",
        "- variance: log(right-window sample variance / left-window sample variance), with a numerical epsilon only for zero-variance windows.",
        "- slope: right-window least-squares slope minus left-window least-squares slope.",
        f"- Reference regions exclude candidate centers within {exclusion_text} of any annotated observable onset in failure rollouts. Clean-success rollouts contribute all valid candidate centers.",
        "- A robust center and MAD/IQR scale are fitted to reference candidate feature values. The unusual score is abs(feature - reference_center) / reference_scale; direction is not used to select an unusual change.",
        "- Q90/Q95/Q99 thresholds are calibrated from one maximum unusual-score pseudo-event per clean-success trajectory. Clean-success and same-rollout false-alarm rates are therefore trajectory-level rather than proportional to the number of sampled points; the combined reference remains available for descriptive peak summaries.",
        "",
        "## Localization definition",
        "",
        f"- For each observable event, the search window is {pre_window_frames} frames before and {post_window_frames} frames after onset, clipped by recovery, terminal failure, the next observable event, and trajectory bounds.",
        "- The event peak is the highest unusual local change at native centers in that bounded window. Peak distance is measured against the annotated observable onset.",
        "- The first threshold exceedance is recorded separately, including signed lead/lag and absolute error. An early peak/exceedance is retained as an early alarm.",
        "- Direction is inspected only for threshold-exceeding changes: level changes are compared with the method's established failure direction; variance and slope report positive/negative change fractions without a preselected failure direction.",
        "- Misses have no first-exceedance error denominator. Failure-type and outcome summaries keep every signal, feature, and scale separate.",
        "",
        "## Q95 summary",
        "",
        "| Method / signal / feature | Scale | N events | Recall | Clean FA | Same-rollout FA | Precision | F1 | AUROC | AP | Median peak distance | within-8 peak hit | Direction after unusual |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if summary.empty:
        lines.append("| No analyzable local-change rows | n/a | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        q95 = summary[
            (summary["threshold"] == "q95")
            & (summary["outcome_group"] == "all_events")
        ].sort_values(["method", "signal", "feature", "scale_frames"])
        for _, row in q95.iterrows():
            direction = (
                row["failure_direction_consistency_after_unusual"]
                if row["feature"] == FAILURE_DIRECTION_FEATURE
                else row["positive_change_fraction_after_unusual"]
            )
            lines.append(
                f"| {METHOD_LABELS.get(row['method'], row['method'])} / {row['signal']} / {row['feature']} "
                f"| {int(row['scale_frames'])} | {int(row['n_events'])} | {fmt(row['recall'])} "
                f"| {fmt(row['clean_success_false_alarm_rate'])} | {fmt(row['same_rollout_non_onset_false_alarm_rate'])} "
                f"| {fmt(row['precision'])} | {fmt(row['f1'])} | {fmt(row['auroc'])} | {fmt(row['average_precision'])} "
                f"| {fmt(row['median_peak_distance_frames'], 1)} | {fmt(row['peak_hit_rate_within_8_frames'])} "
                f"| {fmt(direction)} |"
            )
    lines += [
        "",
        "The Q95 table is a descriptive change-point view, not an independently validated detector benchmark. A high recall with a large peak distance indicates an unusual change that is not localized to onset; false alarms are trajectory-level maxima from the same annotated collection.",
        "",
        "## Comparison with the legacy full_136 snapshot",
        "",
        f"- Comparison table: {comparison_path if comparison_path is not None else 'not generated'}. Response, persistence, and recovery use the high-resolution global snapshot as the current value and full_136_20260827 as the previous value.",
        "- Local change-point rows are included separately. The legacy full_136 snapshot has no equivalent local level/variance/slope metric, so those rows intentionally leave the previous value and delta empty rather than comparing unlike definitions.",
        "",
        "## Native sampling and limitations",
        "",
        "- SAFE, ProcVLM, RynnValue, and Robo-Dopamine remain on their native sample grids. Temporal density is therefore not equal across methods, and a scale is a frame-radius parameter rather than a fixed number of samples.",
        "- The reference set combines clean-success trajectories with non-onset regions from event-bearing trajectories. Thresholds use clean-success pseudo-events; same-rollout non-onset regions are a separate contrast. Neither is an independent validation split.",
        "- Local variance and slope can be underdetermined at sparse RynnValue scales; those rows are retained as unavailable/insufficient rather than imputed.",
        "",
        "## Artifacts",
        "",
        *[f"- {path}" for path in plot_paths],
        "- changepoint_event_metrics.jsonl (primary) and localization_event_metrics.jsonl (compatibility alias): event x method x signal x feature x scale x threshold rows with peak/first-crossing localization, direction, classification metrics, and rollout metadata.",
        "- changepoint_summary.csv and localization_summary.csv: outcome/all-event summaries with recall, false alarm, precision/F1, AUROC/AP, peak distance, tolerance hits, and post-identification direction.",
        "- changepoint_reference_summary.csv: reference candidate and per-trajectory peak distributions for clean-success and same-rollout non-onset regions.",
        "- changepoint_by_failure_type.csv and localization_by_failure_type.csv: the same event summaries grouped by failure type.",
        "- changepoint_scales.csv and localization_thresholds.csv: one row per method/signal/feature/scale with valid counts and clean-success Q90/Q95/Q99 thresholds.",
        "- comparison_with_full_136_20260827.csv: legacy global response/persistence/recovery comparison plus Q95 primary local recall, false alarm, precision/F1, AUROC/AP, localization, and tolerance metrics; legacy local cells are explicitly non-comparable.",
        "- metadata.json and method_coverage.csv: source provenance, manifest hashes, native alignment, and explicit missing-output coverage.",
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "outputs/baseline_signal_analysis/highres_primary_selection.json",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/baseline_signal_analysis/changepoint_primary_20260830",
    )
    for method in METHODS:
        flag = f"--{method.replace('_', '-')}-run"
        if method == "rynnvalue":
            parser.add_argument(flag, dest=f"{method}_run", type=Path, action="append", required=True)
        else:
            parser.add_argument(flag, dest=f"{method}_run", type=Path, required=True)
    parser.add_argument(
        "--scales",
        default=",".join(str(scale) for scale in DEFAULT_SCALES),
        help="Comma-separated local half-window scales in video frames",
    )
    parser.add_argument("--pre-window-frames", type=int, default=60)
    parser.add_argument("--post-window-frames", type=int, default=60)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument(
        "--onset-exclusion-radius",
        type=int,
        default=None,
        help="Reference exclusion radius; defaults to the current local scale",
    )
    parser.add_argument(
        "--compare-snapshot",
        type=Path,
        default=PROJECT_ROOT / "outputs/baseline_signal_analysis/full_136_20260827",
        help="Previous global temporal snapshot used for the comparison table",
    )
    parser.add_argument(
        "--current-global-snapshot",
        type=Path,
        default=PROJECT_ROOT / "outputs/baseline_signal_analysis/highres_primary_20260830",
        help="Current global temporal snapshot used for response/persistence/recovery comparison",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scales = parse_scales(args.scales)
    if args.pre_window_frames < 1 or args.post_window_frames < 1:
        raise ValueError("Event windows must be positive")
    if args.min_samples < 2:
        raise ValueError("min-samples must be at least 2")
    if args.onset_exclusion_radius is not None and args.onset_exclusion_radius < 0:
        raise ValueError("onset-exclusion-radius must be non-negative")
    selection_path = project_path(args.selection)
    manifest_path = project_path(args.manifest)
    annotation_dir = project_path(args.annotations_dir)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_roots = {}
    for method in METHODS:
        values = getattr(args, f"{method}_run")
        source_roots[method] = [
            project_path(value)
            for value in (values if isinstance(values, list) else [values])
        ]

    rollouts, source_metadata, selections = load_rollout_data(
        selection_path=selection_path,
        manifest_path=manifest_path,
        annotation_dir=annotation_dir,
        source_roots=source_roots,
    )
    coverage_rows = []
    for method in METHODS:
        available_rollouts = [
            rollout_id
            for rollout_id, rollout in rollouts.items()
            if method in rollout["methods"]
        ]
        missing_rollouts = [
            rollout_id
            for rollout_id, rollout in rollouts.items()
            if method not in rollout["methods"]
        ]
        coverage_rows.append({
            "method": method,
            "selected_rollouts": len(rollouts),
            "available_rollouts": len(available_rollouts),
            "missing_rollouts": len(missing_rollouts),
            "missing_rollout_ids": json.dumps(missing_rollouts, ensure_ascii=False),
            "raw_load_errors": json.dumps({
                rollout_id: rollouts[rollout_id]["method_errors"].get(method)
                for rollout_id in missing_rollouts
            }, ensure_ascii=False),
            "source_run_count": len(source_metadata[method]["source_runs"]),
        })

    event_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    key_event_rows: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    references_by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    total_keys = 0
    for method in METHODS:
        signal_names = sorted({
            signal
            for rollout in rollouts.values()
            if method in rollout["methods"]
            for signal in rollout["methods"][method]["signals"]
        })
        for signal in signal_names:
            for scale in scales:
                exclusion_radius = (
                    int(args.onset_exclusion_radius)
                    if args.onset_exclusion_radius is not None
                    else int(scale)
                )
                for feature in FEATURES:
                    total_keys += 1
                    key = (method, signal, scale, feature)
                    candidates_by_rollout = {}
                    all_reference_values = []
                    for rollout_id, rollout in rollouts.items():
                        method_data = rollout["methods"].get(method)
                        if not method_data or signal not in method_data["signals"]:
                            continue
                        candidates = build_candidates(
                            method_data["signals"][signal],
                            scale,
                            feature,
                            args.min_samples,
                        )
                        if not candidates:
                            continue
                        candidates_by_rollout[rollout_id] = candidates
                        onset_frames = [
                            int(event["observable_onset_frame"])
                            for event in rollout["events"]
                        ]
                        outcome = normalize_outcome(rollout["annotation"])
                        if not onset_frames and outcome != "clean_success":
                            continue
                        reference_candidates = _reference_candidates(
                            candidates,
                            onset_frames,
                            exclusion_radius,
                        )
                        all_reference_values.extend(
                            float(row["feature_value"])
                            for row in reference_candidates
                        )
                    if len(all_reference_values) == 0:
                        continue
                    location, feature_scale, scale_source = robust_location_scale(
                        all_reference_values
                    )
                    reference = _reference_stats(
                        candidates_by_rollout=candidates_by_rollout,
                        rollouts=rollouts,
                        location=location,
                        scale=feature_scale,
                        exclusion_radius=exclusion_radius,
                    )
                    reference["feature_scale_source"] = scale_source
                    references_by_key[key] = reference
                    if not reference["thresholds"] or not any(
                        math.isfinite(value)
                        for value in reference["thresholds"].values()
                    ):
                        continue
                    scored_by_rollout = {
                        rollout_id: score_candidates(
                            candidates,
                            location,
                            feature_scale,
                        )
                        for rollout_id, candidates in candidates_by_rollout.items()
                    }
                    rows = event_rows_for_key(
                        key=key,
                        candidates_by_rollout=candidates_by_rollout,
                        scored_by_rollout=scored_by_rollout,
                        rollouts=rollouts,
                        thresholds=reference["thresholds"],
                        reference=reference,
                        pre_window_frames=args.pre_window_frames,
                        post_window_frames=args.post_window_frames,
                    )
                    metric_fields = (
                        "clean_pseudo_events",
                        "false_alarm_count",
                        "false_alarm_rate",
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                        "precision",
                        "f1",
                        "auroc",
                        "average_precision",
                    )
                    for threshold_name in THRESHOLDS:
                        threshold_rows = [
                            row for row in rows
                            if row["threshold"] == threshold_name
                        ]
                        threshold_metrics = event_metric_summary(
                            threshold_rows,
                            reference,
                            feature,
                            method,
                            threshold_name,
                        )
                        for row in threshold_rows:
                            row.update({
                                field: threshold_metrics.get(field)
                                for field in metric_fields
                            })
                    event_rows.extend(rows)
                    for row in rows:
                        key_event_rows[
                            (
                                method,
                                signal,
                                scale,
                                feature,
                                str(row["threshold"]),
                            )
                        ].append(row)
                    common_reference = {
                        "method": method,
                        "signal": signal,
                        "scale_frames": int(scale),
                        "feature": feature,
                        "feature_label": FEATURE_LABELS[feature],
                        "reference_exclusion_radius_frames": int(exclusion_radius),
                        "reference_scale_source": scale_source,
                        **{
                            key_name: value
                            for key_name, value in reference.items()
                            if key_name not in {
                                "thresholds",
                                "clean_success_false_alarm_rate",
                                "same_rollout_non_onset_false_alarm_rate",
                                "clean_success_false_alarm_count",
                                "same_rollout_non_onset_false_alarm_count",
                                "clean_peak_scores",
                            }
                        },
                    }
                    reference_rows.append({
                        **common_reference,
                        **{
                            f"threshold_{name}": value
                            for name, value in reference["thresholds"].items()
                        },
                        **{
                            f"clean_success_false_alarm_rate_{name}": value
                            for name, value in reference["clean_success_false_alarm_rate"].items()
                        },
                        **{
                            f"same_rollout_non_onset_false_alarm_rate_{name}": value
                            for name, value in reference["same_rollout_non_onset_false_alarm_rate"].items()
                        },
                    })
                    for threshold_name in THRESHOLDS:
                        rows_for_threshold = key_event_rows.get(
                            (method, signal, scale, feature, threshold_name),
                            [],
                        )
                        for outcome_group in ["all_events", *sorted({
                            str(row["outcome_group"]) for row in rows_for_threshold
                        })]:
                            grouped = (
                                rows_for_threshold
                                if outcome_group == "all_events"
                                else [
                                    row
                                    for row in rows_for_threshold
                                    if row["outcome_group"] == outcome_group
                                ]
                            )
                            metrics = event_metric_summary(
                                grouped,
                                reference,
                                feature,
                                method,
                                threshold_name,
                            )
                            summary_rows.append({
                                "method": method,
                                "signal": signal,
                                "scale_frames": int(scale),
                                "feature": feature,
                                "feature_label": FEATURE_LABELS[feature],
                                "threshold": threshold_name,
                                "outcome_group": outcome_group,
                                **metrics,
                            })
                    scale_rows.append({
                        **common_reference,
                        **{
                            f"threshold_{name}": value
                            for name, value in reference["thresholds"].items()
                        },
                        "valid_rollout_count": len(candidates_by_rollout),
                        "event_row_count": len(rows),
                    })

    summary_frame = pd.DataFrame(summary_rows)
    reference_frame = pd.DataFrame(reference_rows)
    by_failure_type_rows = []
    if event_rows:
        event_frame = pd.DataFrame(event_rows)
        for keys, group in event_frame.groupby(
            ["method", "signal", "scale_frames", "feature", "threshold", "failure_type"],
            dropna=False,
        ):
            method, signal, scale, feature, threshold, failure_type = keys
            reference = references_by_key.get(
                (str(method), str(signal), int(scale), str(feature))
            )
            if reference is None:
                continue
            metrics = event_metric_summary(
                group.to_dict("records"),
                reference,
                str(feature),
                str(method),
                str(threshold),
            )
            by_failure_type_rows.append({
                "method": method,
                "signal": signal,
                "scale_frames": int(scale),
                "feature": feature,
                "feature_label": FEATURE_LABELS[str(feature)],
                "threshold": threshold,
                "failure_type": failure_type,
                **metrics,
            })
    plot_paths = plot_change_point_summaries(
        output_dir=output_dir,
        summary=summary_frame,
    )
    comparison_snapshot = project_path(args.compare_snapshot)
    current_global_snapshot = project_path(args.current_global_snapshot)
    comparison_path = write_comparison_table(
        output_dir=output_dir,
        change_point_summary=summary_frame,
        previous_snapshot=comparison_snapshot,
        current_global_snapshot=current_global_snapshot,
    )
    counts = {
        "rollouts": len(rollouts),
        "observable_events": sum(len(rollout["events"]) for rollout in rollouts.values()),
        "clean_success_rollouts": sum(
            normalize_outcome(rollout["annotation"]) == "clean_success"
            for rollout in rollouts.values()
        ),
        "feature_keys_with_reference": len(reference_rows),
        "event_metric_rows": len(event_rows),
        "summary_rows": len(summary_rows),
        "reference_rows": len(reference_rows),
        "failure_type_rows": len(by_failure_type_rows),
        "scale_rows": len(scale_rows),
        "localization_event_metrics": len(event_rows),
        "localization_summary_rows": len(summary_rows),
        "localization_failure_type_rows": len(by_failure_type_rows),
        "localization_threshold_rows": len(scale_rows),
    }
    metadata = {
        "schema_version": 1,
        "analysis": "lf3r_baseline_change_points",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "annotations_dir": str(annotation_dir),
        "run_roots": _format_source_roots(source_roots),
        "source_runs": {
            method: source_metadata[method]["source_runs"]
            for method in METHODS
        },
        "run_metadata": source_metadata,
        "method_coverage": {
            row["method"]: {
                **row,
                "missing_rollout_ids": json.loads(row["missing_rollout_ids"]),
                "raw_load_errors": json.loads(row["raw_load_errors"]),
            }
            for row in coverage_rows
        },
        "rollouts": [row["id"] for row in selections],
        "methods": list(METHODS),
        "features": list(FEATURES),
        "feature_labels": FEATURE_LABELS,
        "local_scales_frames": list(scales),
        "pre_window_frames": int(args.pre_window_frames),
        "post_window_frames": int(args.post_window_frames),
        "min_samples_per_side": int(args.min_samples),
        "reference_exclusion_radius_frames": (
            "scale" if args.onset_exclusion_radius is None
            else int(args.onset_exclusion_radius)
        ),
        "thresholds": list(THRESHOLDS),
        "threshold_calibration": "clean_success_pseudo_events",
        "tolerances_frames": list(TOLERANCES),
        "frame_coordinate": "video_frame_index",
        "native_sampling_preserved": True,
        "direction_used_after_detection": {
            "level": METHOD_SIGNAL_DIRECTIONS,
            "variance": "reported positive/negative only; no expected failure direction",
            "slope": "reported positive/negative only; no expected failure direction",
        },
        "counts": counts,
        "native_alignment": {
            rollout_id: {
                method: {
                    signal: {
                        "sample_count": int(len(series["frames"])),
                        "first_analysis_frame": int(series["frames"][0]),
                        "last_analysis_frame": int(series["frames"][-1]),
                    }
                    for signal, series in rollout["methods"][method]["signals"].items()
                }
                for method in rollout["methods"]
            }
            for rollout_id, rollout in rollouts.items()
        },
        "plots": plot_paths,
        "localization": {
            "available": bool(event_rows or summary_rows),
            "event_metrics_file": "localization_event_metrics.jsonl",
            "summary_file": "localization_summary.csv",
            "failure_type_file": "localization_by_failure_type.csv",
            "thresholds_file": "localization_thresholds.csv",
            "event_metrics": len(event_rows),
            "summary_rows": len(summary_rows),
            "failure_type_rows": len(by_failure_type_rows),
            "threshold_rows": len(scale_rows),
            "threshold_calibration": "clean_success_pseudo_events",
        },
        "comparison_snapshot": str(comparison_snapshot) if comparison_snapshot.is_dir() else None,
        "current_global_snapshot": str(current_global_snapshot) if current_global_snapshot.is_dir() else None,
        "comparison_path": str(comparison_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(json_safe(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_tables(
        output_dir=output_dir,
        event_rows=event_rows,
        summary_rows=summary_rows,
        reference_rows=reference_rows,
        failure_type_rows=by_failure_type_rows,
        scale_rows=scale_rows,
        coverage_rows=coverage_rows,
    )
    write_report(
        output_dir=output_dir,
        selection_path=selection_path,
        manifest_path=manifest_path,
        annotation_dir=annotation_dir,
        selections=selections,
        rollouts=rollouts,
        coverage_rows=coverage_rows,
        source_roots=source_roots,
        source_metadata=source_metadata,
        scales=scales,
        pre_window_frames=args.pre_window_frames,
        post_window_frames=args.post_window_frames,
        exclusion_radius=(
            None
            if args.onset_exclusion_radius is None
            else int(args.onset_exclusion_radius)
        ),
        min_samples=args.min_samples,
        summary=summary_frame,
        reference_summary=reference_frame,
        plot_paths=plot_paths,
        counts=counts,
        comparison_path=comparison_path,
    )
    print("BASELINE_CHANGE_POINT_ANALYSIS_OK")
    print(f"output_dir={output_dir}")
    print(f"rollouts={counts['rollouts']} observable_events={counts['observable_events']}")
    print(f"feature_keys_with_reference={counts['feature_keys_with_reference']}")
    print(f"event_metric_rows={counts['event_metric_rows']}")
    print(f"summary_rows={counts['summary_rows']}")
    print(f"plots={len(plot_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
