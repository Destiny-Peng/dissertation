"""Descriptive grasp-failure diagnosis on saved fused Robo-Dopamine hop.

This module does not define or tune a new detector. It describes event-centered
fused-hop signatures, compares detected and missed grasp failures, pairs each
event with a same-task clean-success phase control, and estimates a deliberately
broad oracle-style visible-signal coverage.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import detector_mask, evaluate_event

DIAGNOSIS_PRE_SAMPLES = 5
DIAGNOSIS_POST_SAMPLES = 20
FEATURE_WINDOWS = (3, 5, 10, 20)
NEAR_ZERO_THRESHOLD = 0.01
STAGNATION_RUN_THRESHOLD = 3
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 20260920

FEATURE_DIRECTIONS = {
    **{f"min_hop_{window}": "lower" for window in FEATURE_WINDOWS},
    **{f"mean_hop_{window}": "lower" for window in FEATURE_WINDOWS},
    **{f"negative_fraction_{window}": "higher" for window in FEATURE_WINDOWS},
    **{f"near_zero_fraction_{window}": "higher" for window in FEATURE_WINDOWS},
    "longest_negative_run": "higher",
    "longest_stagnation_run": "higher",
    "min_hop_delay_samples": "lower",
    "min_hop_delay_frames": "lower",
}


def event_identity(event: Mapping[str, Any]) -> str:
    event_id = event.get("event_id")
    if event_id not in (None, ""):
        return str(event_id)
    return (
        f"{event.get('rollout_id')}::event"
        f"{int(event.get('event_index') or 0)}"
    )


def percentile(values: Sequence[float], q: float) -> float | None:
    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return data[low]
    weight = position - low
    return data[low] * (1.0 - weight) + data[high] * weight


def longest_run(values: Sequence[float], predicate: Any) -> int:
    best = 0
    current = 0
    for value in values:
        if predicate(float(value)):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _event_anchor_index(
    frames: Sequence[int],
    event: Mapping[str, Any],
) -> int | None:
    onset = int(event["observable_onset_frame"])
    end = event.get("episode_end_frame")
    end_frame = int(end) if end is not None else None
    return next(
        (
            index
            for index, frame in enumerate(frames)
            if frame >= onset and (end_frame is None or frame < end_frame)
        ),
        None,
    )


def _eligible_post_indices(
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


def _window_features(
    frames: Sequence[int],
    hops: Sequence[float],
    *,
    anchor_index: int,
    max_samples: int,
    episode_end_frame: int | None = None,
    onset_frame: int | None = None,
) -> dict[str, Any]:
    eligible = []
    for index in range(anchor_index, len(hops)):
        if episode_end_frame is not None and frames[index] >= episode_end_frame:
            break
        eligible.append(index)
        if len(eligible) >= max_samples:
            break

    result: dict[str, Any] = {}
    for window in FEATURE_WINDOWS:
        indices = eligible[:window]
        values = [float(hops[index]) for index in indices]
        result[f"available_samples_{window}"] = len(values)
        result[f"min_hop_{window}"] = min(values) if values else None
        result[f"mean_hop_{window}"] = (
            statistics.fmean(values) if values else None
        )
        result[f"negative_fraction_{window}"] = (
            sum(value < 0.0 for value in values) / len(values)
            if values
            else None
        )
        result[f"near_zero_fraction_{window}"] = (
            sum(abs(value) < NEAR_ZERO_THRESHOLD for value in values)
            / len(values)
            if values
            else None
        )

    full_values = [float(hops[index]) for index in eligible]
    result["longest_negative_run"] = longest_run(
        full_values, lambda value: value < 0.0
    )
    result["longest_stagnation_run"] = longest_run(
        full_values,
        lambda value: abs(value) < NEAR_ZERO_THRESHOLD,
    )

    if full_values:
        minimum = min(full_values)
        minimum_local = full_values.index(minimum)
        minimum_index = eligible[minimum_local]
        result["min_hop"] = minimum
        result["min_hop_frame"] = int(frames[minimum_index])
        result["min_hop_delay_samples"] = minimum_local + 1
        result["min_hop_delay_frames"] = (
            int(frames[minimum_index]) - int(onset_frame)
            if onset_frame is not None
            else None
        )
        first_negative = next(
            (offset for offset, value in enumerate(full_values) if value < 0.0),
            None,
        )
        result["first_negative_delay_samples"] = (
            first_negative + 1 if first_negative is not None else None
        )
    else:
        result.update(
            {
                "min_hop": None,
                "min_hop_frame": None,
                "min_hop_delay_samples": None,
                "min_hop_delay_frames": None,
                "first_negative_delay_samples": None,
            }
        )
    return result


def _category(features: Mapping[str, Any]) -> str:
    negative_3 = float(features.get("negative_fraction_3") or 0.0) > 0.0
    negative_20 = float(features.get("negative_fraction_20") or 0.0) > 0.0
    if negative_3:
        return "immediate_regression"
    if negative_20:
        return "delayed_regression"
    if (
        (
            int(features.get("available_samples_20") or 0)
            >= STAGNATION_RUN_THRESHOLD
            and float(features.get("near_zero_fraction_20") or 0.0) >= 0.5
        )
        or int(features.get("longest_stagnation_run") or 0)
        >= STAGNATION_RUN_THRESHOLD
    ):
        return "stagnation"
    return "no_clear_hop_response"


def choose_reference_ensemble(
    ensemble_selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    rows = [
        dict(row)
        for row in ensemble_selected
        if row.get("selection_status") == "selected"
        and row.get("selection_target") == "grasp_recall_eventual"
    ]
    if not rows:
        rows = [
            dict(row)
            for row in ensemble_selected
            if row.get("selection_status") == "selected"
            and row.get("selection_target") == "grasp_recall_at_10"
        ]
    if not rows:
        return None
    max_cap = max(float(row["clean_fpr_constraint"]) for row in rows)
    rows = [
        row
        for row in rows
        if abs(float(row["clean_fpr_constraint"]) - max_cap) < 1e-12
    ]
    rows.sort(
        key=lambda row: (
            -float(row.get("grasp_recall_eventual") or 0.0),
            -float(row.get("grasp_recall_at_10") or 0.0),
            -float(row.get("overall_failed_rollout_coverage") or 0.0),
            float(row.get("clean_rollout_fpr") or 0.0),
            int(row.get("pair_priority") or 999),
            str(row.get("a_config_id") or ""),
            str(row.get("b_config_id") or ""),
        )
    )
    return rows[0]


def build_grasp_event_features(
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]],
    reference_ensemble: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    config_by_id = {
        str(config["config_id"]): config
        for config in configs
    }
    reference_masks: dict[str, list[bool]] = {}
    if reference_ensemble is not None:
        config_a = config_by_id.get(str(reference_ensemble.get("a_config_id")))
        config_b = config_by_id.get(str(reference_ensemble.get("b_config_id")))
        if config_a is not None and config_b is not None:
            for rollout_id, signal in signals.items():
                mask_a = detector_mask(signal["hops"], config_a)
                mask_b = detector_mask(signal["hops"], config_b)
                reference_masks[rollout_id] = [
                    bool(left or right)
                    for left, right in zip(mask_a, mask_b)
                ]

    rows: list[dict[str, Any]] = []
    for event in events:
        if str(event.get("failure_type") or "") != "grasp_failure":
            continue
        rollout_id = str(event["rollout_id"])
        signal = signals.get(rollout_id)
        if signal is None:
            continue
        anchor = _event_anchor_index(signal["frames"], event)
        if anchor is None:
            continue
        end = event.get("episode_end_frame")
        end_frame = int(end) if end is not None else None
        onset = int(event["observable_onset_frame"])
        features = _window_features(
            signal["frames"],
            signal["hops"],
            anchor_index=anchor,
            max_samples=DIAGNOSIS_POST_SAMPLES,
            episode_end_frame=end_frame,
            onset_frame=onset,
        )
        detected = None
        detection_delay_samples = None
        detection_delay_frames = None
        if rollout_id in reference_masks:
            metrics = evaluate_event(
                signal["frames"],
                reference_masks[rollout_id],
                onset,
                episode_end_frame=end_frame,
                episode_end_source=str(
                    event.get("episode_end_source") or "rollout_end"
                ),
            )
            detected = bool(metrics["eventual_recall"])
            detection_delay_samples = metrics.get("delay_samples")
            detection_delay_frames = metrics.get("delay_frames")

        oracle_visible = bool(
            (features.get("min_hop_20") is not None
             and float(features["min_hop_20"]) < 0.0)
            or int(features.get("longest_stagnation_run") or 0)
            >= STAGNATION_RUN_THRESHOLD
        )
        aligned_hops: dict[str, Any] = {}
        for offset in range(
            -DIAGNOSIS_PRE_SAMPLES,
            DIAGNOSIS_POST_SAMPLES + 1,
        ):
            index = anchor + offset
            key = (
                f"hop_offset_m{abs(offset)}"
                if offset < 0
                else (
                    "hop_offset_0"
                    if offset == 0
                    else f"hop_offset_p{offset}"
                )
            )
            value = None
            if 0 <= index < len(signal["hops"]):
                if (
                    offset < 0
                    or end_frame is None
                    or int(signal["frames"][index]) < end_frame
                ):
                    value = float(signal["hops"][index])
            aligned_hops[key] = value
        denominator = max(1, len(signal["frames"]) - 1)
        rows.append(
            {
                "event_id": event_identity(event),
                "rollout_id": rollout_id,
                "event_index": event.get("event_index"),
                "task_key": event.get("task_key"),
                "task_suite": event.get("task_suite"),
                "task_id": event.get("task_id"),
                "outcome": event.get("outcome"),
                "observable_onset_frame": onset,
                "episode_end_frame": end_frame,
                "episode_end_source": event.get("episode_end_source"),
                "onset_anchor_index": anchor,
                "onset_anchor_frame": int(signal["frames"][anchor]),
                "normalized_phase": anchor / denominator,
                **aligned_hops,
                "reference_detector_a_family": (
                    reference_ensemble.get("detector_a_family")
                    if reference_ensemble
                    else None
                ),
                "reference_detector_b_family": (
                    reference_ensemble.get("detector_b_family")
                    if reference_ensemble
                    else None
                ),
                "reference_a_config_id": (
                    reference_ensemble.get("a_config_id")
                    if reference_ensemble
                    else None
                ),
                "reference_b_config_id": (
                    reference_ensemble.get("b_config_id")
                    if reference_ensemble
                    else None
                ),
                "reference_clean_fpr_constraint": (
                    reference_ensemble.get("clean_fpr_constraint")
                    if reference_ensemble
                    else None
                ),
                "reference_clean_rollout_fpr": (
                    reference_ensemble.get("clean_rollout_fpr")
                    if reference_ensemble
                    else None
                ),
                "reference_overall_failed_rollout_coverage": (
                    reference_ensemble.get("overall_failed_rollout_coverage")
                    if reference_ensemble
                    else None
                ),
                "reference_detected": detected,
                "reference_detection_delay_samples": detection_delay_samples,
                "reference_detection_delay_frames": detection_delay_frames,
                **features,
                "failure_mode_category": _category(features),
                "oracle_visible_abnormality": oracle_visible,
            }
        )
    return rows


def summarize_detected_vs_missed(
    feature_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    features = list(FEATURE_DIRECTIONS)
    result: list[dict[str, Any]] = []
    for feature in features:
        detected = [
            float(row[feature])
            for row in feature_rows
            if row.get("reference_detected") is True
            and row.get(feature) is not None
        ]
        missed = [
            float(row[feature])
            for row in feature_rows
            if row.get("reference_detected") is False
            and row.get(feature) is not None
        ]
        result.append(
            {
                "feature": feature,
                "failure_evidence_direction": FEATURE_DIRECTIONS[feature],
                "detected_n": len(detected),
                "detected_median": (
                    statistics.median(detected) if detected else None
                ),
                "detected_q1": percentile(detected, 0.25),
                "detected_q3": percentile(detected, 0.75),
                "missed_n": len(missed),
                "missed_median": (
                    statistics.median(missed) if missed else None
                ),
                "missed_q1": percentile(missed, 0.25),
                "missed_q3": percentile(missed, 0.75),
                "detected_minus_missed_median": (
                    statistics.median(detected) - statistics.median(missed)
                    if detected and missed
                    else None
                ),
            }
        )
    return result


def _control_features(
    signal: Mapping[str, Any],
    phase: float,
) -> tuple[int, dict[str, Any]]:
    frames = signal["frames"]
    hops = signal["hops"]
    if not frames:
        raise ValueError("clean control has no frames")
    anchor = min(
        range(len(frames)),
        key=lambda index: abs(
            (index / max(1, len(frames) - 1)) - phase
        ),
    )
    features = _window_features(
        frames,
        hops,
        anchor_index=anchor,
        max_samples=DIAGNOSIS_POST_SAMPLES,
        onset_frame=int(frames[anchor]),
    )
    return anchor, features


def build_matched_clean_pairs(
    feature_rows: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    clean_by_task: dict[str, list[str]] = defaultdict(list)
    for row in clean_rollouts:
        rollout_id = str(row["rollout_id"])
        if rollout_id in signals:
            clean_by_task[str(row["task_key"])].append(rollout_id)

    rows: list[dict[str, Any]] = []
    for failure in feature_rows:
        task_key = str(failure.get("task_key") or "")
        candidates = clean_by_task.get(task_key, [])
        if not candidates:
            continue
        phase = float(failure["normalized_phase"])

        best: tuple[Any, ...] | None = None
        for rollout_id in candidates:
            signal = signals[rollout_id]
            anchor, control = _control_features(signal, phase)
            control_phase = anchor / max(1, len(signal["frames"]) - 1)
            phase_error = abs(control_phase - phase)
            available = int(control.get("available_samples_20") or 0)
            rank = (phase_error, -available, rollout_id)
            if best is None or rank < best[0]:
                best = (rank, rollout_id, anchor, control, control_phase)

        assert best is not None
        _rank, control_id, control_anchor, control, control_phase = best
        control_signal = signals[control_id]
        row: dict[str, Any] = {
            "event_id": failure["event_id"],
            "rollout_id": failure["rollout_id"],
            "task_key": task_key,
            "reference_detected": failure.get("reference_detected"),
            "failure_mode_category": failure.get("failure_mode_category"),
            "failure_phase": phase,
            "control_rollout_id": control_id,
            "control_anchor_index": control_anchor,
            "control_anchor_frame": int(control_signal["frames"][control_anchor]),
            "control_phase": control_phase,
            "phase_error": abs(control_phase - phase),
        }
        for feature in FEATURE_DIRECTIONS:
            failure_value = failure.get(feature)
            control_value = control.get(feature)
            row[f"failure_{feature}"] = failure_value
            row[f"control_{feature}"] = control_value
            row[f"difference_{feature}"] = (
                float(failure_value) - float(control_value)
                if failure_value is not None and control_value is not None
                else None
            )
        failure_oracle = bool(failure.get("oracle_visible_abnormality"))
        control_oracle = bool(
            (control.get("min_hop_20") is not None
             and float(control["min_hop_20"]) < 0.0)
            or int(control.get("longest_stagnation_run") or 0)
            >= STAGNATION_RUN_THRESHOLD
        )
        row["failure_oracle_visible_abnormality"] = failure_oracle
        row["control_oracle_visible_abnormality"] = control_oracle
        rows.append(row)
    return rows


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> tuple[float | None, float | None]:
    data = [float(value) for value in values if math.isfinite(float(value))]
    if not data:
        return None, None
    rng = random.Random(seed)
    medians = []
    n = len(data)
    for _ in range(iterations):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(sample))
    return percentile(medians, 0.025), percentile(medians, 0.975)


def summarize_matched_controls(
    matched_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(FEATURE_DIRECTIONS):
        differences = [
            float(row[f"difference_{feature}"])
            for row in matched_rows
            if row.get(f"difference_{feature}") is not None
        ]
        lower, upper = bootstrap_median_ci(
            differences,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED + feature_index,
        )
        result.append(
            {
                "feature": feature,
                "failure_evidence_direction": FEATURE_DIRECTIONS[feature],
                "matched_n": len(differences),
                "median_paired_difference": (
                    statistics.median(differences) if differences else None
                ),
                "paired_difference_q1": percentile(differences, 0.25),
                "paired_difference_q3": percentile(differences, 0.75),
                "bootstrap_median_ci_low": lower,
                "bootstrap_median_ci_high": upper,
                "proportion_difference_gt_0": (
                    sum(value > 0.0 for value in differences)
                    / len(differences)
                    if differences
                    else None
                ),
                "proportion_difference_lt_0": (
                    sum(value < 0.0 for value in differences)
                    / len(differences)
                    if differences
                    else None
                ),
            }
        )
    return result


def category_rows(
    feature_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "event_id": row["event_id"],
            "rollout_id": row["rollout_id"],
            "event_index": row.get("event_index"),
            "task_key": row.get("task_key"),
            "reference_detected": row.get("reference_detected"),
            "failure_mode_category": row.get("failure_mode_category"),
            "oracle_visible_abnormality": row.get(
                "oracle_visible_abnormality"
            ),
            "first_negative_delay_samples": row.get(
                "first_negative_delay_samples"
            ),
            "min_hop_20": row.get("min_hop_20"),
            "negative_fraction_20": row.get("negative_fraction_20"),
            "near_zero_fraction_20": row.get("near_zero_fraction_20"),
            "longest_negative_run": row.get("longest_negative_run"),
            "longest_stagnation_run": row.get("longest_stagnation_run"),
        }
        for row in feature_rows
    ]


def category_summary(
    feature_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    counts = Counter(
        str(row.get("failure_mode_category") or "unknown")
        for row in feature_rows
    )
    total = len(feature_rows)
    rows = [
        {
            "failure_mode_category": category,
            "event_n": count,
            "fraction": count / total if total else None,
            "detected_n": sum(
                row.get("failure_mode_category") == category
                and row.get("reference_detected") is True
                for row in feature_rows
            ),
            "missed_n": sum(
                row.get("failure_mode_category") == category
                and row.get("reference_detected") is False
                for row in feature_rows
            ),
        }
        for category, count in sorted(counts.items())
    ]
    oracle_n = sum(
        bool(row.get("oracle_visible_abnormality"))
        for row in feature_rows
    )
    rows.append(
        {
            "failure_mode_category": "__oracle_visible_abnormality__",
            "event_n": oracle_n,
            "fraction": oracle_n / total if total else None,
            "detected_n": None,
            "missed_n": None,
        }
    )
    return rows


def plot_grasp_event_heatmap(
    path: Path,
    feature_rows: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    event_by_id = {
        event_identity(event): event
        for event in events
        if str(event.get("failure_type") or "") == "grasp_failure"
    }
    ordered = sorted(
        feature_rows,
        key=lambda row: (
            row.get("reference_detected") is True,
            str(row.get("failure_mode_category") or ""),
            float(row.get("min_hop_20") or 0.0),
            str(row["event_id"]),
        ),
    )
    offsets = list(
        range(-DIAGNOSIS_PRE_SAMPLES, DIAGNOSIS_POST_SAMPLES + 1)
    )
    matrix = np.full((len(ordered), len(offsets)), np.nan, dtype=float)

    for row_index, row in enumerate(ordered):
        event = event_by_id.get(str(row["event_id"]))
        signal = signals.get(str(row["rollout_id"]))
        if event is None or signal is None:
            continue
        anchor = _event_anchor_index(signal["frames"], event)
        if anchor is None:
            continue
        end = event.get("episode_end_frame")
        end_frame = int(end) if end is not None else None
        for column, offset in enumerate(offsets):
            index = anchor + offset
            if index < 0 or index >= len(signal["hops"]):
                continue
            if offset >= 0 and end_frame is not None:
                if int(signal["frames"][index]) >= end_frame:
                    continue
            matrix[row_index, column] = float(signal["hops"][index])

    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = float(np.quantile(finite, 0.98)) if finite.size else 1.0
    limit = max(limit, 1e-6)

    height = max(5.0, min(14.0, 0.24 * max(1, len(ordered)) + 2.0))
    fig, axis = plt.subplots(figsize=(13, height))
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    onset_column = offsets.index(0)
    axis.axvline(onset_column - 0.5, linewidth=1.2, linestyle="--")
    axis.set_xticks(range(len(offsets)))
    axis.set_xticklabels([str(offset) for offset in offsets])
    axis.set_xlabel("native-sample offset from first sample at/after observable onset")
    axis.set_ylabel("grasp failure event")
    labels = [
        (
            ("D" if row.get("reference_detected") is True else "M")
            + " · "
            + str(row["event_id"])
            + " · "
            + str(row.get("failure_mode_category") or "")
        )
        for row in ordered
    ]
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_title("Fused-hop grasp failures · missed first, then failure-mode signature")
    fig.colorbar(image, ax=axis, label="fused hop")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def diagnosis_metadata(
    feature_rows: Sequence[Mapping[str, Any]],
    matched_rows: Sequence[Mapping[str, Any]],
    reference_ensemble: Mapping[str, Any] | None,
) -> dict[str, Any]:
    total = len(feature_rows)
    detected = sum(row.get("reference_detected") is True for row in feature_rows)
    oracle = sum(bool(row.get("oracle_visible_abnormality")) for row in feature_rows)
    matched_oracle = [
        row
        for row in matched_rows
        if row.get("control_oracle_visible_abnormality") is not None
    ]
    reference_recall = detected / total if total else None
    oracle_coverage = oracle / total if total else None
    matched_clean_positive = sum(
        bool(row.get("control_oracle_visible_abnormality"))
        for row in matched_oracle
    )
    matched_clean_coverage = (
        matched_clean_positive / len(matched_oracle)
        if matched_oracle
        else None
    )
    return {
        "enabled": True,
        "failure_type": "grasp_failure",
        "event_n": total,
        "reference_detected_n": detected,
        "reference_missed_n": (
            sum(row.get("reference_detected") is False for row in feature_rows)
        ),
        "reference_recall": reference_recall,
        "event_window": {
            "pre_samples": DIAGNOSIS_PRE_SAMPLES,
            "post_samples": DIAGNOSIS_POST_SAMPLES,
            "feature_windows": list(FEATURE_WINDOWS),
            "near_zero_abs_hop_threshold": NEAR_ZERO_THRESHOLD,
        },
        "taxonomy": {
            "immediate_regression": "any negative fused hop in first 3 post-onset native samples",
            "delayed_regression": "no negative hop in first 3, but at least one negative hop within first 20",
            "stagnation": (
                "no negative hop within first 20 and either near-zero fraction >=0.5 "
                "with at least 3 available samples, or longest |hop|<0.01 run >=3"
            ),
            "no_clear_hop_response": "none of the above",
            "descriptive_only": True,
        },
        "oracle_visible_abnormality": {
            "definition": "min hop within first 20 < 0 OR longest |hop|<0.01 run >=3",
            "event_n": oracle,
            "coverage": oracle_coverage,
            "reference_recall_gap": (
                oracle_coverage - reference_recall
                if oracle_coverage is not None and reference_recall is not None
                else None
            ),
            "matched_clean_n": len(matched_oracle),
            "matched_clean_positive_n": matched_clean_positive,
            "matched_clean_positive_fraction": matched_clean_coverage,
            "not_a_detector": True,
        },
        "matched_control": {
            "rule": "same task; clean-success anchor with nearest normalized native-sample phase",
            "matched_event_n": len(matched_rows),
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "reference_ensemble": dict(reference_ensemble) if reference_ensemble else None,
    }
