"""Earliest-global-progress-maximum localization against first failure onset."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

LOCALIZATION_WINDOWS = (1, 3, 5, 10, 20)


def _percentile(values: Sequence[float | int], q: float) -> float | None:
    data = sorted(float(value) for value in values)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return data[low]
    weight = position - low
    return data[low] * (1.0 - weight) + data[high] * weight


def evaluate_progress_peak_localization(
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate t*=min argmax P_t once per terminal-failure annotated rollout.

    Multiple annotated events in one rollout are intentionally collapsed to the
    earliest observable failure onset.  Progress and t* use the saved native
    fused-progress sample grid without interpolation.
    """
    events_by_rollout: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        if str(event.get("outcome") or "") != "terminal_failure":
            continue
        rollout_id = str(event["rollout_id"])
        if rollout_id not in signals:
            continue
        events_by_rollout[rollout_id].append(event)

    rows: list[dict[str, Any]] = []
    for rollout_id in sorted(events_by_rollout):
        rollout_events = sorted(
            events_by_rollout[rollout_id],
            key=lambda row: (
                int(row["observable_onset_frame"]),
                int(row.get("event_index") or 0),
            ),
        )
        first_event = rollout_events[0]
        signal = signals[rollout_id]
        frames = [int(value) for value in signal["frames"]]
        progress = [float(value) for value in signal.get("progress") or []]
        if not frames or len(frames) != len(progress):
            continue

        max_progress = max(progress)
        peak_index = next(
            index
            for index, value in enumerate(progress)
            if value == max_progress
        )
        peak_frame = frames[peak_index]
        onset_frame = int(first_event["observable_onset_frame"])
        onset_anchor_index = next(
            (
                index
                for index, frame in enumerate(frames)
                if frame >= onset_frame
            ),
            None,
        )
        if onset_anchor_index is None:
            continue
        onset_anchor_frame = frames[onset_anchor_index]

        offset_samples = peak_index - onset_anchor_index
        offset_frames = peak_frame - onset_frame
        abs_offset_samples = abs(offset_samples)
        abs_offset_frames = abs(offset_frames)
        max_count = sum(value == max_progress for value in progress)

        row: dict[str, Any] = {
            "rollout_id": rollout_id,
            "task_key": first_event.get("task_key"),
            "task_suite": first_event.get("task_suite"),
            "task_id": first_event.get("task_id"),
            "outcome": first_event.get("outcome"),
            "first_failure_type": first_event.get("failure_type"),
            "annotated_event_n": len(rollout_events),
            "first_event_id": first_event.get("event_id"),
            "first_event_index": first_event.get("event_index"),
            "first_observable_onset_frame": onset_frame,
            "onset_anchor_sample_index": onset_anchor_index,
            "onset_anchor_frame": onset_anchor_frame,
            "t_star_sample_index": peak_index,
            "t_star_frame": peak_frame,
            "max_progress": max_progress,
            "max_progress_sample_n": max_count,
            "t_star_minus_onset_samples": offset_samples,
            "t_star_minus_onset_frames": offset_frames,
            "abs_error_samples": abs_offset_samples,
            "abs_error_frames": abs_offset_frames,
            "t_star_relation": (
                "before_onset"
                if offset_samples < 0
                else ("at_onset_anchor" if offset_samples == 0 else "after_onset")
            ),
            "native_sample_n": len(frames),
            "signal_source": str(signal.get("prediction_path") or ""),
        }
        for window in LOCALIZATION_WINDOWS:
            row[f"within_{window}_samples"] = abs_offset_samples <= window
        rows.append(row)
    return rows


def summarize_progress_peak_localization(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[Mapping[str, Any]]]] = [
        ("overall", "all", list(rows))
    ]
    for failure_type in sorted(
        {
            str(row.get("first_failure_type") or "other")
            for row in rows
        }
    ):
        groups.append(
            (
                "first_failure_type",
                failure_type,
                [
                    row
                    for row in rows
                    if str(row.get("first_failure_type") or "other")
                    == failure_type
                ],
            )
        )

    result: list[dict[str, Any]] = []
    for group, value, subset in groups:
        if not subset:
            continue
        signed_samples = [
            float(row["t_star_minus_onset_samples"])
            for row in subset
        ]
        signed_frames = [
            float(row["t_star_minus_onset_frames"])
            for row in subset
        ]
        abs_samples = [abs(value) for value in signed_samples]
        abs_frames = [abs(value) for value in signed_frames]
        n = len(subset)
        summary: dict[str, Any] = {
            "group": group,
            "value": value,
            "rollout_n": n,
            "median_signed_offset_samples": float(statistics.median(signed_samples)),
            "p25_signed_offset_samples": _percentile(signed_samples, 0.25),
            "p75_signed_offset_samples": _percentile(signed_samples, 0.75),
            "mean_absolute_error_samples": sum(abs_samples) / n,
            "median_absolute_error_samples": float(statistics.median(abs_samples)),
            "mean_absolute_error_frames": sum(abs_frames) / n,
            "median_absolute_error_frames": float(statistics.median(abs_frames)),
            "before_onset_fraction": (
                sum(value < 0 for value in signed_samples) / n
            ),
            "at_onset_anchor_fraction": (
                sum(value == 0 for value in signed_samples) / n
            ),
            "after_onset_fraction": (
                sum(value > 0 for value in signed_samples) / n
            ),
            "multi_event_rollout_fraction": (
                sum(int(row.get("annotated_event_n") or 0) > 1 for row in subset)
                / n
            ),
        }
        for window in LOCALIZATION_WINDOWS:
            summary[f"within_{window}_samples_fraction"] = (
                sum(
                    bool(row.get(f"within_{window}_samples"))
                    for row in subset
                )
                / n
            )
        result.append(summary)
    return result
