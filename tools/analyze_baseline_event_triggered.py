#!/usr/bin/env python3
"""Analyze baseline signals with event-triggered, onset-aligned curves.

This complementary diagnostic reads existing baseline raw outputs only.  It
aligns observable-onset events to relative frame zero, keeps native samples
without interpolation, and compares terminal/recovered event curves with
task-matched clean-success pseudo-onset controls.
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

from analyze_baseline_change_points import (  # noqa: E402
    DEFAULT_ANNOTATIONS,
    DEFAULT_MANIFEST,
    METHODS,
    METHOD_LABELS,
    METHOD_SIGNAL_DIRECTIONS,
    PROJECT_ROOT,
    json_safe,
    load_rollout_data,
    normalize_outcome,
    resolve_path,
    sha256,
)


DEFAULT_SELECTION = (
    PROJECT_ROOT / "outputs/baseline_signal_analysis/highres_primary_selection.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/baseline_signal_analysis/event_triggered_primary_20260831"
)
DEFAULT_SCALES = (8, 16, 32, 64)
EVENT_GROUPS = ("terminal_failure", "recovered_success", "uncertain")
CONTROL_GROUP = "matched_clean_success"
REPRESENTATIONS = ("normalized_signal", "local_change_score")
PHASES = ("before", "at_onset", "after")
OUTCOME_LABELS = {
    "terminal_failure": "Terminal failure",
    "recovered_success": "Recovered success",
    "uncertain": "Uncertain",
    CONTROL_GROUP: "Matched clean success",
}
PLOT_COLORS = {
    "terminal_failure": "#ef7869",
    "recovered_success": "#67d9b5",
    "uncertain": "#e7c15c",
}


def parse_scales(value: str) -> tuple[int, ...]:
    scales = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            scale = int(token)
        except ValueError as error:
            raise ValueError(f"Invalid scale: {token}") from error
        if scale < 2:
            raise ValueError("Scales must be integers >= 2")
        scales.append(scale)
    if not scales:
        raise ValueError("At least one local scale is required")
    return tuple(sorted(set(scales)))


def finite_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def robust_center_scale(values: np.ndarray) -> tuple[float, float, str]:
    values = finite_array(values)
    if len(values) == 0:
        return math.nan, math.nan, "missing"
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    if mad > 0 and math.isfinite(mad):
        return center, 1.4826 * mad, "mad"
    if len(values) >= 4:
        q25, q75 = np.percentile(values, [25, 75])
        scale = float((q75 - q25) / 1.349)
        if scale > 0 and math.isfinite(scale):
            return center, scale, "iqr"
    floor = max(1e-8, 1e-6 * max(1.0, abs(center)))
    return center, floor, "floor"


def quantiles(values: list[float]) -> tuple[float, float, float, int]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return math.nan, math.nan, math.nan, 0
    q25, median, q75 = np.quantile(clean, [0.25, 0.5, 0.75])
    return float(q25), float(median), float(q75), len(clean)


def phase_for(relative_frame: float, grid_step: int) -> str:
    del grid_step
    if relative_frame < 0:
        return "before"
    if relative_frame > 0:
        return "after"
    return "at_onset"


def task_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("task_suite") or ""),
        str(record.get("task_id") or ""),
    )


def native_sampling_summary(
    rollouts: dict[str, dict[str, Any]],
    signals_by_method: dict[str, list[str]],
) -> dict[str, dict[str, dict[str, Any]]]:
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for method, signals in signals_by_method.items():
        summary[method] = {}
        for signal_name in signals:
            intervals: list[float] = []
            sample_counts: list[int] = []
            available = 0
            for rollout in rollouts.values():
                method_data = rollout["methods"].get(method)
                if not method_data or signal_name not in method_data["signals"]:
                    continue
                frame_values = method_data["signals"][signal_name].get("frames")
                frames = np.asarray(
                    frame_values if frame_values is not None else [],
                    dtype=float,
                )
                frames = frames[np.isfinite(frames)].astype(int)
                if not len(frames):
                    continue
                available += 1
                unique_frames = np.unique(frames)
                sample_counts.append(int(len(unique_frames)))
                if len(unique_frames) > 1:
                    intervals.extend(
                        np.diff(unique_frames).astype(float).tolist()
                    )
            iq25, imedian, iq75, _ = quantiles(intervals)
            sq25, smedian, sq75, _ = quantiles(
                [float(value) for value in sample_counts]
            )
            summary[method][signal_name] = {
                "available_rollouts": available,
                "sample_count_q25": sq25,
                "sample_count_median": smedian,
                "sample_count_q75": sq75,
                "interval_q25_frames": iq25,
                "interval_median_frames": imedian,
                "interval_q75_frames": iq75,
                "interval_min_frames": (
                    min(intervals) if intervals else math.nan
                ),
                "interval_max_frames": (
                    max(intervals) if intervals else math.nan
                ),
            }
    return summary


def event_records(rollouts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for rollout_id, rollout in rollouts.items():
        record = rollout["record"]
        total_frames = max(1, int(record["total_frames"]))
        for event in rollout["events"]:
            onset = int(event["observable_onset_frame"])
            event_index = int(event["event_index"])
            outcome = normalize_outcome(rollout["annotation"])
            records.append({
                "event_id": f"{rollout_id}::event-{event_index}",
                "rollout_id": rollout_id,
                "event_index": event_index,
                "outcome_group": outcome if outcome in EVENT_GROUPS else "uncertain",
                "observable_onset_frame": onset,
                "event_phase": float(onset / max(1, total_frames - 1)),
                "failure_type": event.get("failure_type", "other"),
                "recovery_frame": event.get("recovery_frame"),
                "terminal_failure_frame": event.get("terminal_failure_frame"),
                "task_suite": record.get("task_suite"),
                "task_id": record.get("task_id"),
                "task_description": record.get("task_description"),
                "source_kind": record.get("source_kind"),
                "analysis_partition": record.get("analysis_partition"),
                "dataset_role": record.get("dataset_role"),
                "total_frames": total_frames,
            })
    return records


def _clean_candidates(
    event: dict[str, Any],
    rollouts: dict[str, dict[str, Any]],
    method: str,
    signal: str,
) -> tuple[str, list[dict[str, Any]]]:
    clean = []
    event_task = (str(event["task_suite"] or ""), str(event["task_id"] or ""))
    event_suite = str(event["task_suite"] or "")
    for rollout_id, rollout in rollouts.items():
        if normalize_outcome(rollout["annotation"]) != "clean_success":
            continue
        method_data = rollout["methods"].get(method)
        if not method_data or signal not in method_data["signals"]:
            continue
        record = rollout["record"]
        total_frames = max(1, int(record["total_frames"]))
        control_phase = float(event["event_phase"])
        anchor = int(round(control_phase * max(1, total_frames - 1)))
        candidate = {
            "rollout_id": rollout_id,
            "anchor_frame": max(0, min(anchor, total_frames - 1)),
            "phase_distance": abs(
                (anchor / max(1, total_frames - 1)) - control_phase
            ),
            "task_suite": record.get("task_suite"),
            "task_id": record.get("task_id"),
            "total_frames": total_frames,
        }
        candidate["exact_task"] = task_key(record) == event_task
        candidate["same_suite"] = str(record.get("task_suite") or "") == event_suite
        clean.append(candidate)
    exact = [row for row in clean if row["exact_task"]]
    if exact:
        return "exact_task", sorted(exact, key=lambda row: (row["phase_distance"], row["rollout_id"]))
    same_suite = [row for row in clean if row["same_suite"]]
    if same_suite:
        return "same_suite_fallback", sorted(
            same_suite, key=lambda row: (row["phase_distance"], row["rollout_id"])
        )
    return "global_fallback", sorted(
        clean, key=lambda row: (row["phase_distance"], row["rollout_id"])
    )


def make_instances(
    *,
    event_list: list[dict[str, Any]],
    rollouts: dict[str, dict[str, Any]],
    method: str,
    signal: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    instances: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for event in event_list:
        rollout = rollouts[event["rollout_id"]]
        method_data = rollout["methods"].get(method)
        if not method_data or signal not in method_data["signals"]:
            controls.append({
                "method": method,
                "signal": signal,
                "event_id": event["event_id"],
                "event_rollout_id": event["rollout_id"],
                "event_group": event["outcome_group"],
                "event_index": event["event_index"],
                "status": "event_signal_unavailable",
                "matched_rollout_id": None,
                "match_level": None,
            })
            continue
        event_instance_id = f"event::{event['event_id']}::{method}::{signal}"
        instances.append({
            "instance_id": event_instance_id,
            "rollout_id": event["rollout_id"],
            "event_id": event["event_id"],
            "event_index": event["event_index"],
            "group": event["outcome_group"],
            "comparison_group": event["outcome_group"],
            "is_control": False,
            "anchor_frame": event["observable_onset_frame"],
            "failure_type": event["failure_type"],
            "observable_onset_frame": event["observable_onset_frame"],
            "task_suite": event["task_suite"],
            "task_id": event["task_id"],
            "task_description": event["task_description"],
            "total_frames": event["total_frames"],
            "match_level": None,
            "control_rollout_id": None,
        })
        match_level, candidates = _clean_candidates(
            event, rollouts, method, signal
        )
        if not candidates:
            controls.append({
                "method": method,
                "signal": signal,
                "event_id": event["event_id"],
                "event_rollout_id": event["rollout_id"],
                "event_group": event["outcome_group"],
                "event_index": event["event_index"],
                "status": "no_clean_control",
                "matched_rollout_id": None,
                "match_level": match_level,
            })
            continue
        selected = candidates[0]
        controls.append({
            "method": method,
            "signal": signal,
            "event_id": event["event_id"],
            "event_rollout_id": event["rollout_id"],
            "event_group": event["outcome_group"],
            "event_index": event["event_index"],
            "status": "matched",
            "matched_rollout_id": selected["rollout_id"],
            "match_level": match_level,
            "phase_distance": selected["phase_distance"],
            "control_anchor_frame": selected["anchor_frame"],
            "control_task_suite": selected["task_suite"],
            "control_task_id": selected["task_id"],
        })
        instances.append({
            "instance_id": (
                f"control::{event['event_id']}::{method}::{signal}"
            ),
            "rollout_id": selected["rollout_id"],
            "event_id": event["event_id"],
            "event_index": event["event_index"],
            "group": CONTROL_GROUP,
            "comparison_group": event["outcome_group"],
            "is_control": True,
            "anchor_frame": selected["anchor_frame"],
            "failure_type": None,
            "observable_onset_frame": None,
            "task_suite": selected["task_suite"],
            "task_id": selected["task_id"],
            "task_description": rollouts[selected["rollout_id"]]["record"].get(
                "task_description"
            ),
            "total_frames": selected["total_frames"],
            "match_level": match_level,
            "control_rollout_id": selected["rollout_id"],
        })
    return instances, controls


def normalize_instance(
    signal: dict[str, Any],
    anchor_frame: int,
    pre_window: int,
    post_window: int,
) -> dict[str, Any]:
    frames = np.asarray(signal["frames"], dtype=int)
    values = np.asarray(signal["values"], dtype=float)
    finite = np.isfinite(values)
    frames = frames[finite]
    values = values[finite]
    relative = frames - int(anchor_frame)
    window_mask = (relative >= -pre_window) & (relative <= post_window)
    pre_mask = (relative >= -pre_window) & (relative < 0)
    center, scale, source = robust_center_scale(values[pre_mask])
    if not math.isfinite(center) or not math.isfinite(scale):
        center, scale, source = robust_center_scale(values[window_mask])
        source = "window_fallback_" + source
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
        center = float(np.median(values)) if len(values) else 0.0
        scale = 1.0
        source = "unit_fallback"
    return {
        "frames": frames[window_mask],
        "relative": relative[window_mask],
        "raw": values[window_mask],
        "normalized": (values[window_mask] - center) / scale,
        "normalization_center": float(center),
        "normalization_scale": float(scale),
        "normalization_source": source,
    }


def local_change_scores(
    series: dict[str, Any],
    *,
    anchor_frame: int,
    normalization_scale: float,
    direction: float,
    scales: tuple[int, ...],
    min_samples: int,
    pre_window: int,
    post_window: int,
) -> dict[int, list[dict[str, Any]]]:
    frames = np.asarray(series["frames"], dtype=int)
    values = np.asarray(series["values"], dtype=float)
    finite = np.isfinite(values)
    frames = frames[finite]
    values = values[finite]
    result: dict[int, list[dict[str, Any]]] = {}
    for scale in scales:
        points = []
        for center in frames.tolist():
            left = (frames >= center - scale) & (frames < center)
            right = (frames >= center) & (frames < center + scale)
            if int(left.sum()) < min_samples or int(right.sum()) < min_samples:
                continue
            left_median = float(np.median(values[left]))
            right_median = float(np.median(values[right]))
            change = right_median - left_median
            score = direction * change / max(normalization_scale, 1e-8)
            relative = int(center - anchor_frame)
            if relative < -pre_window or relative > post_window:
                continue
            points.append({
                "relative_frame": relative,
                "center_frame": int(center),
                "change": float(change),
                "score": float(score),
                "left_count": int(left.sum()),
                "right_count": int(right.sum()),
            })
        result[scale] = points
    return result


def aggregate_rows(
    accumulator: dict[tuple[Any, ...], dict[str, Any]],
    *,
    method: str,
    signal: str,
    kind: str,
    scales: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for key, values in sorted(accumulator.items(), key=lambda item: item[0]):
        if kind == "signal":
            comparison_group, group, relative_frame = key
            q25, median, q75, count = quantiles(values["raw"])
            nq25, nmedian, nq75, _ = quantiles(values["normalized"])
            rows.append({
                "method": method,
                "signal": signal,
                "comparison_group": comparison_group,
                "group": group,
                "relative_frame": int(relative_frame),
                "raw_q25": q25,
                "raw_median": median,
                "raw_q75": q75,
                "normalized_q25": nq25,
                "normalized_median": nmedian,
                "normalized_q75": nq75,
                "n_samples": count,
                "n_trajectories": len(values["instance_ids"]),
            })
        else:
            comparison_group, group, scale, relative_frame = key
            q25, median, q75, count = quantiles(values["change"])
            sq25, smedian, sq75, _ = quantiles(values["score"])
            rows.append({
                "method": method,
                "signal": signal,
                "comparison_group": comparison_group,
                "group": group,
                "scale_frames": int(scale),
                "relative_frame": int(relative_frame),
                "change_q25": q25,
                "change_median": median,
                "change_q75": q75,
                "score_q25": sq25,
                "score_median": smedian,
                "score_q75": sq75,
                "n_samples": count,
                "n_trajectories": len(values["instance_ids"]),
            })
    return rows


def _pooled_scale(case_q25: float, case_q75: float, clean_q25: float, clean_q75: float) -> float:
    case_scale = (case_q75 - case_q25) / 1.349
    clean_scale = (clean_q75 - clean_q25) / 1.349
    finite = [float(value) for value in (case_scale, clean_scale) if math.isfinite(float(value))]
    if not finite:
        return 1.0
    # The curves are already normalized per trajectory.  A small floor keeps a
    # nearly constant control curve from producing infinite-looking effects.
    return max(0.25, float(np.sqrt(np.mean(np.square(finite)))))


def _curve_index(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row.get(field) for field in fields): row for row in rows}


def summarize_representation(
    *,
    method: str,
    signal: str,
    event_group: str,
    representation: str,
    scale: int | None,
    signal_rows: list[dict[str, Any]],
    change_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    controls: list[dict[str, Any]],
    grid_step: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    if representation == "normalized_signal":
        case_rows = [
            row for row in signal_rows
            if row["comparison_group"] == event_group
            and row["group"] == event_group
        ]
        clean_rows = [
            row for row in signal_rows
            if row["comparison_group"] == event_group
            and row["group"] == CONTROL_GROUP
        ]
        value_fields = ("normalized_median", "normalized_q25", "normalized_q75")
        index_fields = ("relative_frame",)
    else:
        case_rows = [
            row for row in change_rows
            if row["comparison_group"] == event_group
            and row["group"] == event_group
            and int(row["scale_frames"]) == int(scale)
        ]
        clean_rows = [
            row for row in change_rows
            if row["comparison_group"] == event_group
            and row["group"] == CONTROL_GROUP
            and int(row["scale_frames"]) == int(scale)
        ]
        value_fields = ("score_median", "score_q25", "score_q75")
        index_fields = ("relative_frame",)
    case_index = _curve_index(case_rows, index_fields)
    clean_index = _curve_index(clean_rows, index_fields)
    separation_rows = []
    common = sorted(set(case_index) & set(clean_index))
    for key in common:
        case = case_index[key]
        clean = clean_index[key]
        case_median = float(case[value_fields[0]])
        clean_median = float(clean[value_fields[0]])
        signed = case_median - clean_median
        scale_value = _pooled_scale(
            float(case[value_fields[1]]),
            float(case[value_fields[2]]),
            float(clean[value_fields[1]]),
            float(clean[value_fields[2]]),
        )
        relative = int(key[0])
        separation_rows.append({
            "method": method,
            "signal": signal,
            "event_group": event_group,
            "representation": representation,
            "scale_frames": scale,
            "relative_frame": relative,
            "case_median": case_median,
            "clean_median": clean_median,
            "signed_separation": float(signed),
            "absolute_separation": float(abs(signed)),
            "separation_scale": float(scale_value),
            "separation_z": float(abs(signed) / scale_value),
            "case_q25": float(case[value_fields[1]]),
            "case_q75": float(case[value_fields[2]]),
            "clean_q25": float(clean[value_fields[1]]),
            "clean_q75": float(clean[value_fields[2]]),
            "case_n_trajectories": int(case["n_trajectories"]),
            "clean_n_trajectories": int(clean["n_trajectories"]),
            "case_n_samples": int(case["n_samples"]),
            "clean_n_samples": int(clean["n_samples"]),
            "phase": phase_for(relative, grid_step),
        })
    if not separation_rows:
        return [], None, []

    group_peak = max(
        separation_rows,
        key=lambda row: (
            float(row["separation_z"]),
            -abs(int(row["relative_frame"])),
            -int(row["relative_frame"]),
        ),
    )
    event_peaks = []
    event_instances = [
        payload for payload in payloads.values()
        if not payload["instance"]["is_control"]
        and payload["instance"]["group"] == event_group
    ]
    clean_curve_by_relative = {
        int(row["relative_frame"]): row for row in clean_rows
    }
    case_curve_by_relative = {
        int(row["relative_frame"]): row for row in case_rows
    }
    for payload in event_instances:
        if representation == "normalized_signal":
            values = payload["normalized_by_relative"]
        else:
            values = payload["scores_by_scale"].get(int(scale), {})
        candidates = []
        for relative, value in values.items():
            clean = clean_curve_by_relative.get(int(relative))
            case = case_curve_by_relative.get(int(relative))
            if clean is None or case is None:
                continue
            clean_median = float(clean[value_fields[0]])
            clean_q25 = float(clean[value_fields[1]])
            clean_q75 = float(clean[value_fields[2]])
            case_q25 = float(case[value_fields[1]])
            case_q75 = float(case[value_fields[2]])
            scale_value = _pooled_scale(
                case_q25, case_q75, clean_q25, clean_q75
            )
            signed = float(value) - clean_median
            candidates.append({
                "relative_frame": int(relative),
                "signed": signed,
                "absolute": abs(signed),
                "z": abs(signed) / scale_value,
            })
        if not candidates:
            continue
        peak = max(
            candidates,
            key=lambda row: (
                row["z"],
                -abs(row["relative_frame"]),
                -row["relative_frame"],
            ),
        )
        event = payload["instance"]
        event_peaks.append({
            "method": method,
            "signal": signal,
            "event_group": event_group,
            "representation": representation,
            "scale_frames": scale,
            "event_id": event["event_id"],
            "rollout_id": event["rollout_id"],
            "event_index": event["event_index"],
            "observable_onset_frame": event["observable_onset_frame"],
            "failure_type": event["failure_type"],
            "task_suite": event["task_suite"],
            "task_id": event["task_id"],
            "peak_relative_frame": peak["relative_frame"],
            "peak_phase": phase_for(peak["relative_frame"], grid_step),
            "peak_signed_separation": peak["signed"],
            "peak_absolute_separation": peak["absolute"],
            "peak_separation_z": peak["z"],
            "control_rollout_id": event["control_rollout_id"],
            "control_match_level": event["match_level"],
        })
    lags = [float(row["peak_relative_frame"]) for row in event_peaks]
    q25, typical, q75, n_peaks = quantiles(lags)
    phase_counts = {phase: sum(row["peak_phase"] == phase for row in event_peaks) for phase in PHASES}
    control_matches = [
        row for row in controls
        if row.get("event_group") == event_group and row.get("status") == "matched"
    ]
    summary = {
        "method": method,
        "signal": signal,
        "event_group": event_group,
        "representation": representation,
        "scale_frames": scale,
        "n_events": len(event_instances),
        "n_controls": len(control_matches),
        "n_curve_points": len(separation_rows),
        "strongest_relative_frame": int(group_peak["relative_frame"]),
        "strongest_phase": group_peak["phase"],
        "strongest_signed_separation": group_peak["signed_separation"],
        "strongest_absolute_separation": group_peak["absolute_separation"],
        "strongest_separation_z": group_peak["separation_z"],
        "typical_lag_frames": typical,
        "typical_lag_q25_frames": q25,
        "typical_lag_q75_frames": q75,
        "typical_peak_absolute_separation_z": quantiles(
            [float(row["peak_separation_z"]) for row in event_peaks]
        )[1],
        "n_event_peaks": n_peaks,
        "before_fraction": (
            phase_counts["before"] / n_peaks if n_peaks else math.nan
        ),
        "at_onset_fraction": (
            phase_counts["at_onset"] / n_peaks if n_peaks else math.nan
        ),
        "after_fraction": (
            phase_counts["after"] / n_peaks if n_peaks else math.nan
        ),
        "matched_exact_task": sum(
            row.get("match_level") == "exact_task" for row in control_matches
        ),
        "matched_fallback": sum(
            row.get("match_level") != "exact_task" for row in control_matches
        ),
    }
    return separation_rows, summary, event_peaks


def build_plots(
    *,
    output_dir: Path,
    signal_rows: list[dict[str, Any]],
    change_rows: list[dict[str, Any]],
    methods: list[str],
    signals_by_method: dict[str, list[str]],
    scales: tuple[int, ...],
    pre_window: int,
    post_window: int,
) -> list[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for method in methods:
        for signal in signals_by_method.get(method, []):
            curves = [
                row for row in signal_rows
                if row["method"] == method and row["signal"] == signal
            ]
            if not curves:
                continue
            fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
            for axis, median_field, q25_field, q75_field, title in (
                (axes[0], "raw_median", "raw_q25", "raw_q75", "Raw signal"),
                (
                    axes[1],
                    "normalized_median",
                    "normalized_q25",
                    "normalized_q75",
                    "Onset-normalized signal",
                ),
            ):
                for event_group in EVENT_GROUPS:
                    event_rows = sorted(
                        [
                            row for row in curves
                            if row["comparison_group"] == event_group
                            and row["group"] == event_group
                        ],
                        key=lambda row: row["relative_frame"],
                    )
                    clean = sorted(
                        [
                            row for row in curves
                            if row["comparison_group"] == event_group
                            and row["group"] == CONTROL_GROUP
                        ],
                        key=lambda row: row["relative_frame"],
                    )
                    color = PLOT_COLORS[event_group]
                    if event_rows:
                        x = [row["relative_frame"] for row in event_rows]
                        y = [row[median_field] for row in event_rows]
                        lo = [row[q25_field] for row in event_rows]
                        hi = [row[q75_field] for row in event_rows]
                        axis.plot(x, y, color=color, label=OUTCOME_LABELS[event_group])
                        axis.fill_between(x, lo, hi, color=color, alpha=0.13)
                    if clean:
                        x = [row["relative_frame"] for row in clean]
                        y = [row[median_field] for row in clean]
                        lo = [row[q25_field] for row in clean]
                        hi = [row[q75_field] for row in clean]
                        axis.plot(
                            x, y, color=color, linestyle="--", alpha=0.8,
                            label=OUTCOME_LABELS[event_group] + " control",
                        )
                        axis.fill_between(x, lo, hi, color=color, alpha=0.06)
                axis.axvline(0, color="#f4f6f7", linewidth=0.8, alpha=0.8)
                axis.axhline(0, color="#98a3ad", linewidth=0.5, alpha=0.5)
                axis.set_title(title)
                axis.grid(alpha=0.2)
            axes[1].set_xlabel("relative video frame (observable onset = 0)")
            axes[0].set_ylabel("native units")
            axes[1].set_ylabel("robust z-like units")
            axes[0].legend(frameon=False, fontsize=7, ncol=2)
            fig.suptitle(f"{METHOD_LABELS.get(method, method)} / {signal}: event-triggered response")
            fig.tight_layout()
            path = plot_dir / f"{method}__{signal}__signal.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(str(path.relative_to(output_dir)))

            score_curves = [
                row for row in change_rows
                if row["method"] == method and row["signal"] == signal
            ]
            fig, axes = plt.subplots(
                2, 2, figsize=(14, 9), sharex=True, sharey=False
            )
            axes = axes.flatten()
            for index, scale in enumerate(scales):
                axis = axes[index]
                scale_rows = [
                    row for row in score_curves
                    if int(row["scale_frames"]) == int(scale)
                ]
                for event_group in EVENT_GROUPS:
                    color = PLOT_COLORS[event_group]
                    for group, linestyle, label_suffix in (
                        (event_group, "-", ""),
                        (CONTROL_GROUP, "--", " control"),
                    ):
                        rows = sorted(
                            [
                                row for row in scale_rows
                                if row["comparison_group"] == event_group
                                and row["group"] == group
                            ],
                            key=lambda row: row["relative_frame"],
                        )
                        if not rows:
                            continue
                        x = [row["relative_frame"] for row in rows]
                        y = [row["score_median"] for row in rows]
                        lo = [row["score_q25"] for row in rows]
                        hi = [row["score_q75"] for row in rows]
                        axis.plot(
                            x, y, color=color, linestyle=linestyle,
                            label=OUTCOME_LABELS[event_group] + label_suffix,
                        )
                        axis.fill_between(x, lo, hi, color=color, alpha=0.1)
                axis.axvline(0, color="#f4f6f7", linewidth=0.8, alpha=0.8)
                axis.axhline(0, color="#98a3ad", linewidth=0.5, alpha=0.5)
                axis.set_title(f"Local level change score, scale={scale} frames")
                axis.grid(alpha=0.2)
                if index >= 2:
                    axis.set_xlabel("relative frame")
                if index % 2 == 0:
                    axis.set_ylabel("failure-oriented score")
            axes[0].legend(frameon=False, fontsize=6, ncol=2)
            fig.suptitle(f"{METHOD_LABELS.get(method, method)} / {signal}: local change-score curves")
            fig.tight_layout()
            path = plot_dir / f"{method}__{signal}__change_scores.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(str(path.relative_to(output_dir)))
    return paths


def write_report(
    *,
    output_dir: Path,
    selection_path: Path,
    manifest_path: Path,
    annotation_dir: Path,
    metadata: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
    summary_frame: pd.DataFrame,
    plot_paths: list[str],
) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"

    event_counts = metadata["event_group_counts"]
    coverage = "; ".join(
        f"{METHOD_LABELS.get(row['method'], row['method'])} / {row['signal']}: "
        f"{row.get('available_signal_rollouts', row['available_method_rollouts'])}/{row['selected_rollouts']} "
        f"signal rollouts, {row['event_bearing_rollouts']} event-bearing "
        f"rollouts, {row['event_count']} events"
        for row in coverage_rows
    )
    sampling = []
    for method, signal_map in (metadata.get("native_sampling") or {}).items():
        for signal, values in signal_map.items():
            interval = fmt(values.get("interval_median_frames"), 2)
            sample_count = fmt(values.get("sample_count_median"), 1)
            sampling.append(
                f"{METHOD_LABELS.get(method, method)} / {signal}: "
                f"median native sample count {sample_count}, "
                f"median interval {interval} frames"
            )
    sampling_text = "; ".join(sampling)
    lines = [
        "# LF3R Event-Triggered Signal Analysis",
        "",
        "This complementary diagnostic reuses existing baseline raw outputs and observable-onset annotations only. It does not rerun GPU inference, modify raw outputs, or replace the local change-point analysis.",
        "",
        "## Scope and controls",
        "",
        f"- Selection: {selection_path}; manifest: {manifest_path}; annotations: {annotation_dir}.",
        f"- Selected rollouts: {metadata['counts']['rollouts']}; observable events: {metadata['counts']['observable_events']}.",
        f"- Event groups: {json.dumps(event_counts, ensure_ascii=False)}.",
        f"- Method/signal coverage: {coverage}.",
        f"- Native sampling summary: {sampling_text}.",
        "- Each event is aligned at observable_onset_frame = 0. Clean-success controls are selected by exact task_suite + task_id first; their pseudo-onset is placed at the same normalized trajectory phase as the event. Fallback levels are recorded explicitly.",
        "- Raw and normalized curves report median and IQR across native samples. A curve point contains only samples whose native frame offset equals that relative frame; no interpolation or resampling is performed.",
        "",
        "## Representations",
        "",
        f"- Event-triggered window: {metadata['parameters']['pre_window_frames']} frames before to {metadata['parameters']['post_window_frames']} frames after onset; grid step {metadata['parameters']['grid_step']} frame.",
        f"- Normalized signal: (native value - onset-pre-window median) / onset-pre-window robust MAD/IQR scale, with explicit fallback metadata when a pre-window is unavailable.",
        f"- Local change score: failure-direction * (right-window native median - left-window native median) / the instance normalization scale. Local half-window scales are {', '.join(str(scale) for scale in metadata['parameters']['local_scales_frames'])} frames; this is an interpretable level-change curve, not the change-point detector score.",
        "- Separation is the case median minus its matched-clean median at the same relative native offset. The reported separation_z divides its absolute value by the pooled IQR-derived curve scale with a small floor.",
        "- strongest_phase is before/at_onset/after using relative frame <0, =0, >0. typical_lag_frames is the median event-level peak offset; misses with no common native control point are retained as unavailable.",
        "",
        "## Interpretation",
        "",
        "A reproducible response should appear as a stable case-versus-control separation over nearby relative frames, with a concentrated event-level lag rather than a single isolated point. Different methods have different native temporal densities, so sparse RynnValue curves and denser SAFE/ProcVLM curves must not be treated as equally sampled continuous signals.",
        "All representations and signals are reported independently. These are descriptive diagnostics on the annotated collection, not independently validated detector or causal-performance estimates.",
        "",
        "## Strongest separation summary",
        "",
        "| Method / signal | Event group | Representation | Scale | Strongest phase | Strongest relative frame | Separation z | Typical lag | Before | At onset | After | N peaks |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if summary_frame.empty:
        lines.append("| No matched event/control curves | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 |")
    else:
        rows = summary_frame.sort_values(
            ["method", "signal", "event_group", "representation", "scale_frames"],
            na_position="first",
        )
        for _, row in rows.iterrows():
            scale = "signal" if pd.isna(row["scale_frames"]) else str(int(row["scale_frames"]))
            lines.append(
                f"| {METHOD_LABELS.get(row['method'], row['method'])} / {row['signal']} "
                f"| {OUTCOME_LABELS.get(row['event_group'], row['event_group'])} "
                f"| {row['representation']} | {scale} | {row['strongest_phase']} "
                f"| {int(row['strongest_relative_frame'])} | {fmt(row['strongest_separation_z'])} "
                f"| {fmt(row['typical_lag_frames'], 1)} | {fmt(row['before_fraction'])} "
                f"| {fmt(row['at_onset_fraction'])} | {fmt(row['after_fraction'])} "
                f"| {int(row['n_event_peaks'])} |"
            )
    lines += [
        "",
        "## Artifacts",
        "",
        *[f"- {path}" for path in plot_paths],
        "- event_triggered_curves.csv: raw and onset-normalized median/IQR curves for event groups and matched clean controls.",
        "- event_triggered_change_scores.csv: native local level-change median/IQR curves at each requested scale.",
        "- event_triggered_separation.csv: relative-frame case/control separation and robust separation score.",
        "- event_triggered_summary.csv: aggregate strongest separation phase and event-level typical lag.",
        "- event_triggered_peak_events.csv: compact event-level peak offsets with rollout, task, and failure-type metadata.",
        "- event_triggered_controls.csv and method_coverage.csv: clean-control provenance, matching level, missing method/signal output, and coverage.",
        "- metadata.json: source runs, manifest hash, annotation counts, parameters, native sampling policy, and plot list.",
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_tables(
    output_dir: Path,
    signal_rows: list[dict[str, Any]],
    change_rows: list[dict[str, Any]],
    separation_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    peak_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> None:
    pd.DataFrame(signal_rows).to_csv(
        output_dir / "event_triggered_curves.csv", index=False
    )
    pd.DataFrame(change_rows).to_csv(
        output_dir / "event_triggered_change_scores.csv", index=False
    )
    pd.DataFrame(separation_rows).to_csv(
        output_dir / "event_triggered_separation.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "event_triggered_summary.csv", index=False
    )
    pd.DataFrame(peak_rows).to_csv(
        output_dir / "event_triggered_peak_events.csv", index=False
    )
    pd.DataFrame(control_rows).to_csv(
        output_dir / "event_triggered_controls.csv", index=False
    )
    pd.DataFrame(coverage_rows).to_csv(
        output_dir / "method_coverage.csv", index=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    for method in METHODS:
        flag = f"--{method.replace('_', '-')}-run"
        if method == "rynnvalue":
            parser.add_argument(
                flag, dest=f"{method}_run", type=Path, action="append", required=True
            )
        else:
            parser.add_argument(
                flag, dest=f"{method}_run", type=Path, required=True
            )
    parser.add_argument(
        "--scales",
        default=",".join(str(scale) for scale in DEFAULT_SCALES),
        help="Comma-separated local level-change half-window scales in video frames",
    )
    parser.add_argument("--pre-window-frames", type=int, default=60)
    parser.add_argument("--post-window-frames", type=int, default=60)
    parser.add_argument("--grid-step", type=int, default=1)
    parser.add_argument("--min-samples", type=int, default=2)
    args = parser.parse_args()

    scales = parse_scales(args.scales)
    if args.pre_window_frames < 1 or args.post_window_frames < 1:
        raise ValueError("Event windows must be positive")
    if args.grid_step < 1:
        raise ValueError("grid-step must be positive")
    if args.min_samples < 2:
        raise ValueError("min-samples must be at least 2")

    selection_path = resolve_path(args.selection)
    manifest_path = resolve_path(args.manifest)
    annotation_dir = resolve_path(args.annotations_dir)
    output_dir = resolve_path(args.output_dir)
    try:
        selection_path.relative_to(PROJECT_ROOT)
        manifest_path.relative_to(PROJECT_ROOT)
        annotation_dir.relative_to(PROJECT_ROOT)
        output_dir.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("All paths must remain inside the project root") from error
    output_dir.mkdir(parents=True, exist_ok=True)

    source_roots: dict[str, list[Path]] = {}
    for method in METHODS:
        values = getattr(args, f"{method}_run")
        roots = values if isinstance(values, list) else [values]
        source_roots[method] = [resolve_path(value) for value in roots]
        for root in source_roots[method]:
            try:
                root.relative_to(PROJECT_ROOT)
            except ValueError as error:
                raise ValueError("Run roots must remain inside the project root") from error

    rollouts, source_metadata, selections = load_rollout_data(
        selection_path=selection_path,
        manifest_path=manifest_path,
        annotation_dir=annotation_dir,
        source_roots=source_roots,
    )
    events = event_records(rollouts)
    signals_by_method: dict[str, list[str]] = {}
    for method in METHODS:
        signals_by_method[method] = sorted({
            signal
            for rollout in rollouts.values()
            if method in rollout["methods"]
            for signal in rollout["methods"][method]["signals"]
        })

    all_signal_rows: list[dict[str, Any]] = []
    all_change_rows: list[dict[str, Any]] = []
    all_separation_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    all_peak_rows: list[dict[str, Any]] = []
    all_control_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    normalization_sources: defaultdict[str, int] = defaultdict(int)

    for method in METHODS:
        method_available = [
            rollout_id for rollout_id, rollout in rollouts.items()
            if method in rollout["methods"]
        ]
        missing_method = [
            rollout_id for rollout_id, rollout in rollouts.items()
            if method not in rollout["methods"]
        ]
        for signal_name in signals_by_method[method]:
            signal_available = [
                rollout_id for rollout_id, rollout in rollouts.items()
                if method in rollout["methods"]
                and signal_name in rollout["methods"][method]["signals"]
            ]
            signal_missing = [
                rollout_id for rollout_id, rollout in rollouts.items()
                if method not in rollout["methods"]
                or signal_name not in rollout["methods"][method]["signals"]
            ]
            instances, control_rows = make_instances(
                event_list=events,
                rollouts=rollouts,
                method=method,
                signal=signal_name,
            )
            all_control_rows.extend(control_rows)
            signal_acc: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
                lambda: {"raw": [], "normalized": [], "instance_ids": set()}
            )
            change_acc: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
                lambda: {"change": [], "score": [], "instance_ids": set()}
            )
            payloads: dict[str, dict[str, Any]] = {}
            for instance in instances:
                rollout = rollouts[instance["rollout_id"]]
                series = rollout["methods"][method]["signals"][signal_name]
                normalized = normalize_instance(
                    series,
                    instance["anchor_frame"],
                    args.pre_window_frames,
                    args.post_window_frames,
                )
                normalization_sources[normalized["normalization_source"]] += 1
                payload = {
                    "instance": instance,
                    "normalized_by_relative": {},
                    "scores_by_scale": defaultdict(dict),
                }
                for relative, raw, norm in zip(
                    normalized["relative"],
                    normalized["raw"],
                    normalized["normalized"],
                ):
                    relative = int(relative)
                    if relative % args.grid_step:
                        continue
                    payload["normalized_by_relative"][relative] = float(norm)
                    key = (
                        instance["comparison_group"],
                        instance["group"],
                        relative,
                    )
                    signal_acc[key]["raw"].append(float(raw))
                    signal_acc[key]["normalized"].append(float(norm))
                    signal_acc[key]["instance_ids"].add(instance["instance_id"])
                scores = local_change_scores(
                    series,
                    anchor_frame=instance["anchor_frame"],
                    normalization_scale=float(normalized["normalization_scale"]),
                    direction=float(METHOD_SIGNAL_DIRECTIONS[method]),
                    scales=scales,
                    min_samples=args.min_samples,
                    pre_window=args.pre_window_frames,
                    post_window=args.post_window_frames,
                )
                for scale, points in scores.items():
                    for point in points:
                        relative = int(point["relative_frame"])
                        if relative % args.grid_step:
                            continue
                        payload["scores_by_scale"][int(scale)][relative] = float(
                            point["score"]
                        )
                        key = (
                            instance["comparison_group"],
                            instance["group"],
                            int(scale),
                            relative,
                        )
                        change_acc[key]["change"].append(float(point["change"]))
                        change_acc[key]["score"].append(float(point["score"]))
                        change_acc[key]["instance_ids"].add(instance["instance_id"])
                payloads[instance["instance_id"]] = payload
            signal_rows = aggregate_rows(
                signal_acc,
                method=method,
                signal=signal_name,
                kind="signal",
            )
            change_rows = aggregate_rows(
                change_acc,
                method=method,
                signal=signal_name,
                kind="change",
                scales=scales,
            )
            all_signal_rows.extend(signal_rows)
            all_change_rows.extend(change_rows)
            event_signal_count = sum(
                1 for instance in instances if not instance["is_control"]
            )
            matched_controls = sum(
                row.get("status") == "matched" for row in control_rows
            )
            coverage_rows.append({
                "method": method,
                "signal": signal_name,
                "selected_rollouts": len(rollouts),
                "available_method_rollouts": len(method_available),
                "missing_method_rollouts": len(missing_method),
                "missing_method_rollout_ids": json.dumps(
                    missing_method, ensure_ascii=False
                ),
                "available_signal_rollouts": len(signal_available),
                "missing_signal_rollouts": len(signal_missing),
                "missing_signal_rollout_ids": json.dumps(
                    signal_missing, ensure_ascii=False
                ),
                "event_signal_rollouts": len({
                    instance["rollout_id"]
                    for instance in instances
                    if not instance["is_control"]
                }),
                "event_bearing_rollouts": len({
                    instance["rollout_id"]
                    for instance in instances
                    if not instance["is_control"]
                }),
                "event_count": event_signal_count,
                "matched_clean_controls": matched_controls,
                "control_rows": len(control_rows),
                "source_run_count": len(source_metadata[method]["source_runs"]),
            })
            for event_group in EVENT_GROUPS:
                signal_sep, signal_summary, signal_peaks = summarize_representation(
                    method=method,
                    signal=signal_name,
                    event_group=event_group,
                    representation="normalized_signal",
                    scale=None,
                    signal_rows=signal_rows,
                    change_rows=change_rows,
                    payloads=payloads,
                    controls=control_rows,
                    grid_step=args.grid_step,
                )
                all_separation_rows.extend(signal_sep)
                if signal_summary:
                    all_summary_rows.append(signal_summary)
                all_peak_rows.extend(signal_peaks)
                for scale in scales:
                    score_sep, score_summary, score_peaks = summarize_representation(
                        method=method,
                        signal=signal_name,
                        event_group=event_group,
                        representation="local_change_score",
                        scale=scale,
                        signal_rows=signal_rows,
                        change_rows=change_rows,
                        payloads=payloads,
                        controls=control_rows,
                        grid_step=args.grid_step,
                    )
                    all_separation_rows.extend(score_sep)
                    if score_summary:
                        all_summary_rows.append(score_summary)
                    all_peak_rows.extend(score_peaks)

    sampling_summary = native_sampling_summary(rollouts, signals_by_method)
    plot_paths = build_plots(
        output_dir=output_dir,
        signal_rows=all_signal_rows,
        change_rows=all_change_rows,
        methods=list(METHODS),
        signals_by_method=signals_by_method,
        scales=scales,
        pre_window=args.pre_window_frames,
        post_window=args.post_window_frames,
    )
    summary_frame = pd.DataFrame(all_summary_rows)
    counts = {
        "rollouts": len(rollouts),
        "observable_events": len(events),
        "event_group_counts": {
            group: sum(event["outcome_group"] == group for event in events)
            for group in EVENT_GROUPS
        },
        "clean_success_rollouts": sum(
            normalize_outcome(rollout["annotation"]) == "clean_success"
            for rollout in rollouts.values()
        ),
        "signal_curve_rows": len(all_signal_rows),
        "change_score_rows": len(all_change_rows),
        "separation_rows": len(all_separation_rows),
        "summary_rows": len(all_summary_rows),
        "peak_event_rows": len(all_peak_rows),
        "control_rows": len(all_control_rows),
        "plot_count": len(plot_paths),
    }
    metadata = {
        "schema_version": 1,
        "analysis": "lf3r_event_triggered_signal_analysis",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "annotations_dir": str(annotation_dir),
        "source_runs": {
            method: source_metadata[method]["source_runs"]
            for method in METHODS
        },
        "run_metadata": source_metadata,
        "rollouts": [row["id"] for row in selections],
        "methods": list(METHODS),
        "signals_by_method": signals_by_method,
        "parameters": {
            "pre_window_frames": int(args.pre_window_frames),
            "post_window_frames": int(args.post_window_frames),
            "grid_step": int(args.grid_step),
            "min_samples_per_side": int(args.min_samples),
            "local_scales_frames": list(scales),
            "alignment": "observable_onset_frame=0",
            "native_sampling_preserved": True,
            "interpolation": False,
            "normalization": "per-instance pre-onset robust median and MAD/IQR scale",
            "local_change_score": "method failure direction times native right-minus-left median divided by instance normalization scale",
            "control_matching": "exact task_suite/task_id, same_suite fallback, then global fallback",
            "control_anchor": "clean pseudo-onset at event normalized trajectory phase",
            "phase_definition": "relative frame < 0 = before; 0 = at onset; > 0 = after",
            "direction_by_method_signal": {
                method: {
                    signal: float(METHOD_SIGNAL_DIRECTIONS[method])
                    for signal in signals
                }
                for method, signals in signals_by_method.items()
            },
        },
        "event_group_counts": counts["event_group_counts"],
        "native_sampling": sampling_summary,
        "normalization_source_counts": dict(normalization_sources),
        "method_coverage": coverage_rows,
        "counts": counts,
        "plots": plot_paths,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(json_safe(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_tables(
        output_dir,
        all_signal_rows,
        all_change_rows,
        all_separation_rows,
        all_summary_rows,
        all_peak_rows,
        all_control_rows,
        coverage_rows,
    )
    write_report(
        output_dir=output_dir,
        selection_path=selection_path,
        manifest_path=manifest_path,
        annotation_dir=annotation_dir,
        metadata=metadata,
        coverage_rows=coverage_rows,
        summary_frame=summary_frame,
        plot_paths=plot_paths,
    )
    print("BASELINE_EVENT_TRIGGERED_ANALYSIS_OK")
    print(f"output_dir={output_dir}")
    print(f"rollouts={counts['rollouts']} observable_events={counts['observable_events']}")
    print(f"signal_curve_rows={counts['signal_curve_rows']}")
    print(f"change_score_rows={counts['change_score_rows']}")
    print(f"summary_rows={counts['summary_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
