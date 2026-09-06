#!/usr/bin/env python3
"""Analyze LF3R baseline signals around human-annotated observable failures.

The adapters keep each method's native temporal samples.  Event windows are
defined in video-frame coordinates because that is the coordinate system used
by the annotator.  SAFE's environment timestep is converted back to video
frame coordinates using the manifest's first_environment_timestep.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = PROJECT_ROOT / "outputs/baselines/representative_rollouts_20260826.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "annotations/failure_annotations/v1/records"

METHODS = ("safe", "procvlm", "rynnvalue", "robo_dopamine")
LOCALIZATION_THRESHOLDS = ("q90", "q95", "q99")
METHOD_LABELS = {
    "safe": "SAFE",
    "procvlm": "ProcVLM",
    "rynnvalue": "RynnValue",
    "robo_dopamine": "Robo-Dopamine",
}
METHOD_SIGNAL_DIRECTIONS = {
    "safe": 1.0,
    "procvlm": -1.0,
    "rynnvalue": 1.0,
    "robo_dopamine": -1.0,
}
METHOD_SIGNAL_UNITS = {
    "safe": "official handcrafted feature units",
    "procvlm": "progress in [0, 1]",
    "rynnvalue": "official value-head units (remaining-time plot semantics)",
    "robo_dopamine": "official forward progress / hop in [approximately -1, 1]",
}
SAFE_PRIMARY_SIGNALS = (
    "max_token_prob",
    "avg_token_prob",
    "max_token_entropy",
    "avg_token_entropy",
)
ROBO_SIGNALS = ("progress", "hop")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_arrays(frames: list[int] | np.ndarray, values: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_array = np.asarray(frames, dtype=int)
    value_array = np.asarray(values, dtype=float)
    mask = np.isfinite(value_array)
    frame_array = frame_array[mask]
    value_array = value_array[mask]
    order = np.argsort(frame_array, kind="stable")
    frame_array = frame_array[order]
    value_array = value_array[order]
    unique, positions = np.unique(frame_array, return_index=True)
    return unique.astype(int), value_array[positions].astype(float)


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Manifest contains duplicate IDs: {path}")
    return result


def load_selection(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        # A baseline jobs.jsonl is an authoritative full selection when the
        # launcher did not persist a separate selection JSON file.
        rows = [{"id": row["rollout_id"]} for row in load_jsonl(path)]
        if not rows:
            raise ValueError(f"Selection is empty: {path}")
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Selection contains duplicate rollout IDs")
        return {"selection": rows, "source_format": "baseline_jobs_jsonl"}, rows
    rows = document["selection"] if isinstance(document, dict) else document
    if not rows:
        raise ValueError(f"Selection is empty: {path}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Selection contains duplicate rollout IDs")
    return document if isinstance(document, dict) else {"selection": rows}, rows


def run_rollout_id_list(run_root: Path) -> list[str]:
    """Return rollout IDs recorded by a run, preserving duplicate detection."""
    jobs_path = run_root / "jobs.jsonl"
    if not jobs_path.is_file():
        return []
    result: list[str] = []
    for row in load_jsonl(jobs_path):
        rollout_id = row.get("rollout_id", row.get("id"))
        if isinstance(rollout_id, str) and rollout_id:
            result.append(rollout_id)
    return result


def run_rollout_ids(run_root: Path) -> set[str]:
    """Return rollout IDs recorded by a baseline run's authoritative jobs file."""
    return set(run_rollout_id_list(run_root))


def load_annotations(annotation_dir: Path, rollout_id: str) -> dict[str, Any]:
    path = annotation_dir / f"{rollout_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Annotation record is missing: {path}")
    annotation = json.loads(path.read_text(encoding="utf-8"))
    events = list(annotation.get("failure_events") or [])
    # Schema v1 stores only the first event at the top level.
    if not events and annotation.get("observable_onset_frame") is not None:
        events = [{
            "failure_type": annotation.get("failure_type", "other"),
            "causal_onset_frame": annotation.get("causal_onset_frame"),
            "observable_onset_frame": annotation.get("observable_onset_frame"),
            "terminal_failure_frame": annotation.get("terminal_failure_frame"),
            "recovery_frame": annotation.get("recovery_frame"),
            "notes": annotation.get("notes", ""),
        }]
    for event_index, event in enumerate(events):
        event["_analysis_event_index"] = event_index
    annotation["failure_events"] = events
    return annotation


def normalize_outcome(annotation: dict[str, Any]) -> str:
    outcome = annotation.get("outcome_label")
    if outcome == "failure":
        return "terminal_failure"
    if outcome == "recovered_success":
        return "recovered_success"
    if outcome == "success":
        return "clean_success"
    return str(outcome or "uncertain")


def extract_frame_from_path(value: str) -> int | None:
    match = re.search(r"frame_(\d+)\.png", value)
    return int(match.group(1)) if match else None


def load_safe_signal(run_root: Path, rollout_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    path = run_root / "raw" / rollout_id / "safe_features.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    if "action_timestep" in table:
        first_timestep = int(manifest.get("first_environment_timestep") or 0)
        frames = (table["action_timestep"].to_numpy(dtype=int) - first_timestep).tolist()
    else:
        frames = list(range(len(table)))
    signals = {}
    for column in table.columns:
        if column == "action_timestep" or column.endswith("_rmean"):
            continue
        numeric = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        signal_frames, signal_values = finite_arrays(frames, numeric)
        if len(signal_frames):
            signals[column] = {
                "frames": signal_frames,
                "values": signal_values,
                "direction": METHOD_SIGNAL_DIRECTIONS["safe"],
                "unit": METHOD_SIGNAL_UNITS["safe"],
            }
    if not signals:
        raise ValueError(f"SAFE has no numeric signal columns: {path}")
    return {
        "signals": signals,
        "source_files": [str(path.resolve())],
        "alignment": {
            "raw_frame_indices": [int(value) for value in table["action_timestep"].tolist()] if "action_timestep" in table else frames,
            "analysis_frame_indices": frames,
            "correction": "subtract first_environment_timestep for annotation alignment",
        },
        "native_format": "CSV rows",
    }


def load_procvlm_signal(run_root: Path, rollout_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    path = run_root / "raw" / rollout_id / "procvlm_raw.jsonl"
    rows = load_jsonl(path)
    raw_frames = [int(row["frame_index"]) for row in rows]
    total_frames = int(manifest["total_frames"])
    frames = [min(max(frame, 0), total_frames - 1) for frame in raw_frames]
    values = [float(row["progress"]) for row in rows]
    signal_frames, signal_values = finite_arrays(frames, values)
    if len(signal_frames) == 0:
        raise ValueError(f"ProcVLM has no finite progress rows: {path}")
    return {
        "signals": {
            "progress": {
                "frames": signal_frames,
                "values": signal_values,
                "direction": METHOD_SIGNAL_DIRECTIONS["procvlm"],
                "unit": METHOD_SIGNAL_UNITS["procvlm"],
            }
        },
        "source_files": [str(path.resolve())],
        "alignment": {
            "raw_frame_indices": raw_frames,
            "analysis_frame_indices": frames,
            "out_of_range_raw_indices": [frame for frame in raw_frames if not 0 <= frame < total_frames],
            "correction": "clip to final valid video frame for annotation alignment",
        },
        "native_format": "JSONL frame_index/progress",
    }


def load_rynnvalue_signal(run_root: Path, rollout_id: str) -> dict[str, Any]:
    paths = sorted((run_root / "raw" / rollout_id).glob("*/raw_model_outputs.json"))
    if not paths:
        raise FileNotFoundError(f"RynnValue raw_model_outputs.json missing for {rollout_id}")
    if len(paths) != 1:
        raise ValueError(f"Expected one RynnValue raw_model_outputs.json for {rollout_id}, got {paths}")
    path = paths[0]
    raw = json.loads(path.read_text(encoding="utf-8"))
    frames, values = finite_arrays(raw["sampled_indices"], raw["values"])
    if len(frames) == 0:
        raise ValueError(f"RynnValue has no finite values: {path}")
    return {
        "signals": {
            "value": {
                "frames": frames,
                "values": values,
                "direction": METHOD_SIGNAL_DIRECTIONS["rynnvalue"],
                "unit": METHOD_SIGNAL_UNITS["rynnvalue"],
            }
        },
        "source_files": [str(path.resolve())],
        "alignment": {
            "raw_frame_indices": [int(value) for value in raw["sampled_indices"]],
            "analysis_frame_indices": [int(value) for value in frames],
            "correction": "RynnValue sampled_indices are already video-frame indices",
        },
        "native_format": "JSON sampled_indices/values",
    }


def load_robo_signal(run_root: Path, rollout_id: str) -> dict[str, Any]:
    result_path = run_root / "raw" / rollout_id / "worker_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    prediction_path = Path(result["raw_model_output"])
    rows = json.loads(prediction_path.read_text(encoding="utf-8"))
    frames = []
    progress = []
    hop = []
    for row in rows:
        frame = extract_frame_from_path(row["image"][5]) if row.get("image") else None
        if frame is None:
            match = re.search(r"af_(\d+)$", str(row.get("id", "")))
            frame = int(match.group(1)) if match else None
        if frame is None:
            continue
        frames.append(frame)
        progress.append(float(row["progress"]))
        hop.append(float(row["hop"]))
    progress_frames, progress_values = finite_arrays(frames, progress)
    hop_frames, hop_values = finite_arrays(frames, hop)
    if len(progress_frames) == 0:
        raise ValueError(f"Robo-Dopamine has no frame-bound records: {prediction_path}")
    signals = {}
    for name, signal_frames, signal_values in (
        ("progress", progress_frames, progress_values),
        ("hop", hop_frames, hop_values),
    ):
        signals[name] = {
            "frames": signal_frames,
            "values": signal_values,
            "direction": METHOD_SIGNAL_DIRECTIONS["robo_dopamine"],
            "unit": METHOD_SIGNAL_UNITS["robo_dopamine"],
        }
    return {
        "signals": signals,
        "source_files": [str(result_path.resolve()), str(prediction_path.resolve())],
        "alignment": {
            "raw_frame_indices": [int(value) for value in frames],
            "analysis_frame_indices": [int(value) for value in progress_frames],
            "correction": "official AF image paths already use video-frame indices",
        },
        "native_format": "official pred_vllm.json AF image frame indices",
    }


def load_method_signal(method: str, run_root: Path, rollout_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    if method == "safe":
        return load_safe_signal(run_root, rollout_id, manifest)
    if method == "procvlm":
        return load_procvlm_signal(run_root, rollout_id, manifest)
    if method == "rynnvalue":
        return load_rynnvalue_signal(run_root, rollout_id)
    if method == "robo_dopamine":
        return load_robo_signal(run_root, rollout_id)
    raise AssertionError(method)


def values_in_window(frames: np.ndarray, values: np.ndarray, low: int, high: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (frames >= low) & (frames < high)
    return frames[mask], values[mask]


def robust_center_scale(values: np.ndarray) -> tuple[float, float, str, float]:
    if len(values) == 0:
        return math.nan, math.nan, "missing", math.nan
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    if mad > 0.0 and math.isfinite(mad):
        return center, 1.4826 * mad, "mad", mad
    if len(values) >= 4:
        q25, q75 = np.percentile(values, [25, 75])
        iqr_scale = float((q75 - q25) / 1.349)
        if iqr_scale > 0.0 and math.isfinite(iqr_scale):
            return center, iqr_scale, "iqr", mad
    floor = max(1e-8, 1e-6 * max(1.0, abs(center)))
    return center, floor, "floor", mad


def nearest_value(frames: np.ndarray, values: np.ndarray, target: int) -> tuple[int, float] | tuple[None, None]:
    if len(frames) == 0:
        return None, None
    position = int(np.argmin(np.abs(frames - int(target))))
    return int(frames[position]), float(values[position])


def event_metric(
    *,
    series: dict[str, Any],
    event_frame: int,
    total_frames: int,
    event_index: int,
    event_type: str,
    outcome_group: str,
    failure_type: str,
    recovery_frame: int | None,
    terminal_failure_frame: int | None,
    next_event_frame: int | None,
    pre_window_frames: int,
    post_window_frames: int,
    is_background: bool = False,
    scale_override: float | None = None,
) -> dict[str, Any]:
    frames = series["frames"]
    values = series["values"]
    direction = float(series["direction"])
    pre_frames, pre_values = values_in_window(
        frames,
        values,
        max(0, int(event_frame) - pre_window_frames),
        int(event_frame),
    )
    pre_center, raw_pre_scale, scale_source, pre_mad = robust_center_scale(pre_values)
    robust_scale = raw_pre_scale
    if scale_override is not None and scale_source == "floor" and scale_override > 0.0:
        robust_scale = float(scale_override)
        scale_source = "clean_background_fallback"

    nominal_post_high = min(total_frames, int(event_frame) + post_window_frames)
    if next_event_frame is not None and next_event_frame > event_frame:
        nominal_post_high = min(nominal_post_high, int(next_event_frame))
    response_high = nominal_post_high
    if recovery_frame is not None and recovery_frame > event_frame:
        response_high = min(response_high, int(recovery_frame))
    post_frames, post_values = values_in_window(frames, values, int(event_frame), response_high)
    post_fallback = False
    if len(post_values) == 0:
        post_frame, post_value = nearest_value(frames[frames >= event_frame], values[frames >= event_frame], event_frame)
        if post_frame is not None:
            post_frames = np.asarray([post_frame], dtype=int)
            post_values = np.asarray([post_value], dtype=float)
            post_fallback = True
    persistence_high = nominal_post_high
    if recovery_frame is not None and recovery_frame > event_frame:
        persistence_high = min(persistence_high, int(recovery_frame))
    persistence_frames, persistence_values = values_in_window(
        frames, values, int(event_frame), persistence_high
    )
    if len(persistence_values) == 0:
        persistence_frames, persistence_values = post_frames, post_values
    post_center = float(np.median(post_values)) if len(post_values) else math.nan
    signed_change = post_center - pre_center if len(pre_values) and len(post_values) else math.nan
    oriented_change = direction * signed_change if math.isfinite(signed_change) else math.nan
    normalized_magnitude = abs(signed_change) / robust_scale if math.isfinite(signed_change) else math.nan
    normalized_oriented = oriented_change / robust_scale if math.isfinite(oriented_change) else math.nan
    if len(persistence_values) and math.isfinite(pre_center) and math.isfinite(robust_scale):
        oriented_persistence = direction * (persistence_values - pre_center)
        persistence_fraction = float(np.mean(oriented_persistence >= 0.5 * robust_scale))
        persistence_median = float(np.median(oriented_persistence) / robust_scale)
    else:
        persistence_fraction = math.nan
        persistence_median = math.nan

    recovery_sample_frame = None
    recovery_value = None
    recovery_signed_change = math.nan
    recovery_toward_baseline = math.nan
    recovery_normalized = math.nan
    recovery_fraction = math.nan
    recovery_frame_error = math.nan
    if recovery_frame is not None and math.isfinite(post_center):
        recovery_sample_frame, recovery_value = nearest_value(frames, values, int(recovery_frame))
        if recovery_value is not None:
            recovery_frame_error = abs(int(recovery_sample_frame) - int(recovery_frame))
            recovery_signed_change = float(recovery_value - post_center)
            recovery_toward_baseline = float(direction * (post_center - recovery_value))
            recovery_normalized = recovery_toward_baseline / robust_scale
            event_deviation = direction * (post_center - pre_center)
            if event_deviation > 0.0:
                recovery_fraction = float(np.clip(recovery_toward_baseline / event_deviation, 0.0, 1.0))

    return {
        "event_index": int(event_index),
        "event_type": event_type,
        "outcome_group": outcome_group,
        "failure_type": failure_type,
        "is_background": bool(is_background),
        "event_frame": int(event_frame),
        "recovery_frame": recovery_frame,
        "terminal_failure_frame": terminal_failure_frame,
        "next_event_frame": next_event_frame,
        "recovery_offset_frames": (int(recovery_frame) - int(event_frame)) if recovery_frame is not None else None,
        "native_direction": direction,
        "pre_window_requested_frames": pre_window_frames,
        "post_window_requested_frames": post_window_frames,
        "pre_sample_count": int(len(pre_values)),
        "post_sample_count": int(len(post_values)),
        "persistence_sample_count": int(len(persistence_values)),
        "pre_sample_frames": [int(x) for x in pre_frames],
        "post_sample_frames": [int(x) for x in post_frames],
        "persistence_sample_frames": [int(x) for x in persistence_frames],
        "post_window_fallback": post_fallback,
        "pre_center": pre_center,
        "pre_mad": pre_mad,
        "raw_pre_scale": raw_pre_scale,
        "robust_pre_scale": robust_scale,
        "robust_scale_source": scale_source,
        "post_center": post_center,
        "signed_pre_post_change": signed_change,
        "failure_oriented_change": oriented_change,
        "normalized_response_magnitude": normalized_magnitude,
        "normalized_failure_oriented_response": normalized_oriented,
        "post_event_persistence_fraction": persistence_fraction,
        "post_event_persistence_median_oriented": persistence_median,
        "recovery_sample_frame": recovery_sample_frame,
        "recovery_frame_nearest_sample_error": recovery_frame_error,
        "recovery_value": recovery_value,
        "recovery_rebound_signed_change": recovery_signed_change,
        "recovery_rebound_toward_baseline": recovery_toward_baseline,
        "recovery_rebound_normalized": recovery_normalized,
        "recovery_fraction_toward_baseline": recovery_fraction,
    }


def background_centers(total_frames: int, stride_frames: int) -> list[int]:
    start = max(1, stride_frames // 2)
    stop = max(start + 1, total_frames - stride_frames // 2)
    return list(range(start, stop, max(1, stride_frames)))


def finite_or_none(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return finite_or_none(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value



def _score_window(
    series: dict[str, Any],
    pre_center: Any,
    robust_scale: Any,
    low: int,
    high: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return native sample frames and failure-oriented robust scores."""
    try:
        center = float(pre_center)
        scale = float(robust_scale)
    except (TypeError, ValueError):
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    frames = np.asarray(series["frames"], dtype=int)
    values = np.asarray(series["values"], dtype=float)
    mask = (frames >= int(low)) & (frames < int(high)) & np.isfinite(values)
    selected_frames = frames[mask]
    scores = float(series["direction"]) * (values[mask] - center) / scale
    finite = np.isfinite(scores)
    return selected_frames[finite], scores[finite]


def _event_window_bounds(
    *,
    event_frame: int,
    total_frames: int,
    pre_window_frames: int,
    post_window_frames: int,
    recovery_frame: int | None = None,
    terminal_failure_frame: int | None = None,
    next_event_frame: int | None = None,
) -> tuple[int, int]:
    low = max(0, int(event_frame) - max(1, int(pre_window_frames)))
    high = min(int(total_frames), int(event_frame) + max(1, int(post_window_frames)))
    for boundary in (recovery_frame, terminal_failure_frame, next_event_frame):
        if boundary is not None and int(boundary) > int(event_frame):
            high = min(high, int(boundary))
    return low, max(low + 1, high)


def _rank_auc(scores: list[float], labels: list[int]) -> float:
    """Compute event-level AUROC without adding scipy/sklearn dependencies."""
    pairs = [(float(score), int(label)) for score, label in zip(scores, labels) if math.isfinite(float(score))]
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
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _average_precision(scores: list[float], labels: list[int]) -> float:
    pairs = [(float(score), int(label)) for score, label in zip(scores, labels) if math.isfinite(float(score))]
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


def _metric_summary(
    rows: list[dict[str, Any]],
    *,
    threshold_value: float,
    clean_rows: list[dict[str, Any]],
    all_event_scores: list[tuple[float, str]],
) -> dict[str, Any]:
    """Summarize thresholded event localization and clean-success false alarms."""
    valid_rows = [row for row in rows if row.get("event_score") is not None and math.isfinite(float(row["event_score"]))]
    hits = [row for row in valid_rows if row.get("hit")]
    early = [row for row in valid_rows if row.get("early_alarm")]
    errors = [
        abs(float(row["signed_lead_lag_frames"]))
        for row in hits
        if row.get("signed_lead_lag_frames") is not None
    ]
    signed_delays = [
        float(row["signed_lead_lag_frames"])
        for row in valid_rows
        if row.get("signed_lead_lag_frames") is not None
    ]
    tolerance_rates = {}
    for tolerance in (4, 8, 16, 30):
        tolerance_rates[f"hit_rate_within_{tolerance}_frames"] = (
            float(sum(error <= tolerance for error in errors) / len(valid_rows))
            if valid_rows else math.nan
        )
    clean_scores = [
        float(row["event_score"])
        for row in clean_rows
        if row.get("event_score") is not None and math.isfinite(float(row["event_score"]))
    ]
    false_alarm_count = sum(score >= threshold_value for score in clean_scores)
    n_events = len(valid_rows)
    n_hits = len(hits)
    n_misses = n_events - n_hits
    clean_count = len(clean_scores)
    tp = n_hits
    fp = false_alarm_count
    fn = n_misses
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2.0 * precision * recall / (precision + recall) if math.isfinite(precision) and math.isfinite(recall) and precision + recall else math.nan
    scores = [score for score, _ in all_event_scores] + clean_scores
    labels = [1] * len(all_event_scores) + [0] * len(clean_scores)
    result = {
        "n_events": n_events,
        "n_hits": n_hits,
        "n_misses": n_misses,
        "event_hit_rate": recall,
        "recall": recall,
        "n_early_alarms": len(early),
        "early_alarm_rate": len(early) / n_events if n_events else math.nan,
        "median_signed_lead_lag_frames": float(np.median(signed_delays)) if signed_delays else math.nan,
        "median_post_onset_delay_frames": float(np.median([float(row["post_onset_delay_frames"]) for row in hits if row.get("post_onset_delay_frames") is not None])) if hits else math.nan,
        "median_absolute_localization_error_frames": float(np.median(errors)) if errors else math.nan,
        "mean_absolute_localization_error_frames": float(np.mean(errors)) if errors else math.nan,
        "clean_pseudo_events": clean_count,
        "false_alarm_count": false_alarm_count,
        "false_alarm_rate": false_alarm_count / clean_count if clean_count else math.nan,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "f1": f1,
        "auroc": _rank_auc(scores, labels),
        "average_precision": _average_precision(scores, labels),
    }
    result.update(tolerance_rates)
    return result


def compute_failure_localization(
    *,
    event_rows: list[dict[str, Any]],
    rollouts: dict[str, dict[str, Any]],
    post_window_frames: int,
) -> dict[str, Any]:
    """Calibrate clean-success thresholds and report native-sample localization.

    Every available method/signal is kept separate. Event windows are clipped
    by recovery, terminal failure, the next annotated event, and the requested
    post window. Threshold crossings only use native sample points.
    """
    clean_scores: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    event_scores: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        method = str(row.get("method"))
        signal = str(row.get("signal"))
        rollout = rollouts.get(str(row.get("rollout_id")))
        if not rollout or method not in rollout.get("methods", {}):
            continue
        series = rollout["methods"][method]["signals"].get(signal)
        if not series:
            continue
        if row.get("is_background"):
            low = int(row["event_frame"])
            high = int(row["event_frame"]) + max(1, int(row.get("post_window_requested_frames") or post_window_frames))
        else:
            low, high = _event_window_bounds(
                event_frame=int(row["event_frame"]),
                total_frames=int(rollout["record"]["total_frames"]),
                pre_window_frames=int(row.get("pre_window_requested_frames") or post_window_frames),
                post_window_frames=post_window_frames,
                recovery_frame=row.get("recovery_frame"),
                terminal_failure_frame=row.get("terminal_failure_frame"),
                next_event_frame=row.get("next_event_frame"),
            )
        frames, scores = _score_window(
            series,
            row.get("pre_center"),
            row.get("robust_pre_scale"),
            low,
            high,
        )
        if len(scores) == 0:
            continue
        candidate = {
            "row": row,
            "frames": frames,
            "scores": scores,
            "event_score": float(np.max(scores)),
            "score_peak_frame": int(frames[int(np.argmax(scores))]),
            "window_low": int(low),
            "window_high_exclusive": int(high),
        }
        if row.get("is_background"):
            clean_scores[(method, signal)].append(candidate)
        else:
            event_scores[(method, signal)].append(candidate)

    threshold_rows: list[dict[str, Any]] = []
    threshold_map: dict[tuple[str, str], dict[str, float]] = {}
    for key, candidates in clean_scores.items():
        values = [candidate["event_score"] for candidate in candidates if math.isfinite(candidate["event_score"])]
        if not values:
            continue
        quantiles = {
            "q90": float(np.quantile(values, 0.90)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
        }
        threshold_map[key] = quantiles
        threshold_rows.append({
            "method": key[0],
            "signal": key[1],
            "clean_pseudo_events": len(values),
            "clean_score_median": float(np.median(values)),
            "clean_score_q90": quantiles["q90"],
            "clean_score_q95": quantiles["q95"],
            "clean_score_q99": quantiles["q99"],
            **quantiles,
        })

    localization_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    failure_type_rows: list[dict[str, Any]] = []
    for key, candidates in event_scores.items():
        thresholds = threshold_map.get(key)
        if not thresholds:
            continue
        method, signal = key
        clean_candidates = clean_scores.get(key, [])
        for threshold_name in LOCALIZATION_THRESHOLDS:
            threshold_value = thresholds[threshold_name]
            threshold_event_rows: list[dict[str, Any]] = []
            for candidate in candidates:
                row = candidate["row"]
                onset = int(row["event_frame"])
                frames = candidate["frames"]
                scores = candidate["scores"]
                full_alarm_mask = scores >= threshold_value
                post_alarm_mask = (frames >= onset) & (scores >= threshold_value)
                full_position = int(np.flatnonzero(full_alarm_mask)[0]) if np.any(full_alarm_mask) else None
                post_position = int(np.flatnonzero(post_alarm_mask)[0]) if np.any(post_alarm_mask) else None
                first_alarm = int(frames[full_position]) if full_position is not None else None
                first_post = int(frames[post_position]) if post_position is not None else None
                hit = first_alarm is not None
                signed_lead_lag = (first_alarm - onset) if first_alarm is not None else None
                post_delay = (first_post - onset) if first_post is not None else None
                event_result = {
                    "method": method,
                    "signal": signal,
                    "threshold": threshold_name,
                    "threshold_value": threshold_value,
                    "rollout_id": row.get("rollout_id"),
                    "event_index": row.get("event_index"),
                    "event_type": row.get("event_type"),
                    "outcome_group": row.get("outcome_group"),
                    "failure_type": row.get("failure_type"),
                    "event_frame": onset,
                    "observable_onset_frame": onset,
                    "recovery_frame": row.get("recovery_frame"),
                    "terminal_failure_frame": row.get("terminal_failure_frame"),
                    "next_event_frame": row.get("next_event_frame"),
                    "window_low": candidate["window_low"],
                    "window_high_exclusive": candidate["window_high_exclusive"],
                    "pre_center": row.get("pre_center"),
                    "robust_scale": row.get("robust_pre_scale"),
                    "event_score": candidate["event_score"],
                    "score_peak_frame": candidate["score_peak_frame"],
                    "score_peak_offset_frames": candidate["score_peak_frame"] - onset,
                    "first_alarm_frame": first_alarm,
                    "first_post_onset_frame": first_post,
                    "signed_lead_lag_frames": signed_lead_lag,
                    "post_onset_delay_frames": post_delay,
                    "localization_error_frames": abs(signed_lead_lag) if signed_lead_lag is not None else None,
                    "absolute_localization_error_frames": abs(signed_lead_lag) if signed_lead_lag is not None else None,
                    "hit": hit,
                    "miss": not hit,
                    "early_alarm": first_alarm is not None and first_alarm < onset,
                    "native_samples_only": True,
                    "clean_pseudo_event_count": len(clean_candidates),
                }
                for tolerance in (4, 8, 16, 30):
                    event_result[f"hit_within_{tolerance}_frames"] = bool(
                        hit and signed_lead_lag is not None and abs(signed_lead_lag) <= tolerance
                    )
                record = rollouts[str(row["rollout_id"])]["record"]
                event_result.update({
                    "task_suite": record.get("task_suite"),
                    "task_id": record.get("task_id"),
                    "task_description": record.get("task_description"),
                    "source_kind": record.get("source_kind"),
                    "analysis_partition": record.get("analysis_partition"),
                    "dataset_role": record.get("dataset_role"),
                })
                localization_rows.append(event_result)
                threshold_event_rows.append(event_result)
            clean_metric_rows = [{"event_score": candidate["event_score"]} for candidate in clean_candidates]
            group_specs: list[tuple[str, list[dict[str, Any]]]] = [("all_events", threshold_event_rows)]
            outcomes = sorted({str(row.get("outcome_group", "uncertain")) for row in threshold_event_rows})
            group_specs.extend(
                (outcome, [row for row in threshold_event_rows if row.get("outcome_group") == outcome])
                for outcome in outcomes
            )
            for group_name, group_rows in group_specs:
                group_scores = [(float(row["event_score"]), str(row.get("outcome_group", "uncertain"))) for row in group_rows]
                metrics = _metric_summary(
                    group_rows,
                    threshold_value=threshold_value,
                    clean_rows=clean_metric_rows,
                    all_event_scores=group_scores,
                )
                summary_rows.append({
                    "method": method,
                    "signal": signal,
                    "threshold": threshold_name,
                    "threshold_value": threshold_value,
                    "outcome_group": group_name,
                    **metrics,
                })
            failure_types = sorted({str(row.get("failure_type") or "other") for row in threshold_event_rows})
            for failure_type in failure_types:
                group_rows = [row for row in threshold_event_rows if str(row.get("failure_type") or "other") == failure_type]
                group_scores = [(float(row["event_score"]), str(row.get("outcome_group", "uncertain"))) for row in group_rows]
                metrics = _metric_summary(
                    group_rows,
                    threshold_value=threshold_value,
                    clean_rows=clean_metric_rows,
                    all_event_scores=group_scores,
                )
                failure_type_rows.append({
                    "method": method,
                    "signal": signal,
                    "threshold": threshold_name,
                    "threshold_value": threshold_value,
                    "failure_type": failure_type,
                    **metrics,
                })
    return {
        "event_rows": localization_rows,
        "summary": pd.DataFrame(summary_rows),
        "by_failure_type": pd.DataFrame(failure_type_rows),
        "thresholds": pd.DataFrame(threshold_rows),
    }


def write_localization_outputs(localization: dict[str, Any], output_dir: Path) -> None:
    event_path = output_dir / "localization_event_metrics.jsonl"
    with event_path.open("w", encoding="utf-8") as handle:
        for row in localization["event_rows"]:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
    for key, filename in (
        ("summary", "localization_summary.csv"),
        ("by_failure_type", "localization_by_failure_type.csv"),
        ("thresholds", "localization_thresholds.csv"),
    ):
        frame = localization[key]
        frame.to_csv(output_dir / filename, index=False)


def write_comparison_table(
    *,
    output_dir: Path,
    current_summary: pd.DataFrame,
    localization_summary: pd.DataFrame,
    old_snapshot: Path | None,
) -> Path | None:
    if old_snapshot is None or not old_snapshot.is_dir():
        return None
    old_path = old_snapshot / "summary_by_method_signal_outcome.csv"
    if not old_path.is_file():
        return None
    old = pd.read_csv(old_path)
    current = current_summary.copy()
    keys = ["method", "signal", "outcome_group"]
    wanted = [
        "normalized_response_magnitude_median",
        "post_event_persistence_fraction_median",
        "recovery_fraction_toward_baseline_median",
    ]
    old = old[[key for key in keys + wanted if key in old.columns]].rename(
        columns={field: "old_" + field for field in wanted if field in old.columns}
    )
    current = current[[key for key in keys + wanted if key in current.columns]].rename(
        columns={field: "highres_" + field for field in wanted if field in current.columns}
    )
    comparison = current.merge(old, on=keys, how="outer")
    for field in wanted:
        high = "highres_" + field
        old_field = "old_" + field
        if high in comparison.columns and old_field in comparison.columns:
            comparison[field + "_delta"] = pd.to_numeric(comparison[high], errors="coerce") - pd.to_numeric(comparison[old_field], errors="coerce")
    if len(localization_summary):
        q95 = localization_summary[localization_summary["threshold"] == "q95"]
        q95 = q95[q95["outcome_group"] == "all_events"][["method", "signal", "recall", "false_alarm_rate", "f1", "auroc", "average_precision"]].rename(
            columns={
                "recall": "highres_localization_recall_q95",
                "false_alarm_rate": "highres_false_alarm_rate_q95",
                "f1": "highres_localization_f1_q95",
                "auroc": "highres_localization_auroc",
                "average_precision": "highres_localization_average_precision",
            }
        )
        comparison = comparison.merge(q95, on=["method", "signal"], how="outer")
    output_path = output_dir / "comparison_with_full_136_20260827.csv"
    comparison.to_csv(output_path, index=False)
    return output_path


def summarize_metrics(event_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, pd.DataFrame]:
    table = pd.DataFrame(event_rows)
    event_table = table[~table["is_background"].astype(bool)].copy()
    background_table = table[table["is_background"].astype(bool)].copy()
    metric_names = [
        "signed_pre_post_change",
        "normalized_response_magnitude",
        "normalized_failure_oriented_response",
        "post_event_persistence_fraction",
        "recovery_rebound_normalized",
        "recovery_fraction_toward_baseline",
    ]
    group_columns = ["method", "signal", "outcome_group"]
    summary_rows = []
    for keys, group in event_table.groupby(group_columns, dropna=False):
        method, signal, outcome = keys
        unique_events = group[["rollout_id", "event_index"]].drop_duplicates()
        row = {"method": method, "signal": signal, "outcome_group": outcome, "n_events": len(unique_events)}
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_median"] = float(values.median()) if len(values) else math.nan
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else math.nan
            row[f"{metric}_q25"] = float(values.quantile(0.25)) if len(values) else math.nan
            row[f"{metric}_q75"] = float(values.quantile(0.75)) if len(values) else math.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    background_summary_rows = []
    if len(background_table):
        for keys, group in background_table.groupby(["method", "signal"], dropna=False):
            method, signal = keys
            row = {"method": method, "signal": signal, "n_pseudo_events": len(group)}
            for metric in metric_names:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                row[f"{metric}_median"] = float(values.median()) if len(values) else math.nan
                row[f"{metric}_q95"] = float(values.quantile(0.95)) if len(values) else math.nan
            background_summary_rows.append(row)
    background_summary = pd.DataFrame(background_summary_rows)

    summary.to_csv(output_dir / "summary_by_method_signal_outcome.csv", index=False)
    background_summary.to_csv(output_dir / "clean_background_summary.csv", index=False)

    method_summary_rows = []
    for keys, group in event_table.groupby(["method", "outcome_group"], dropna=False):
        method, outcome = keys
        unique_events = group[["rollout_id", "event_index"]].drop_duplicates()
        row = {"method": method, "outcome_group": outcome, "n_event_signal_rows": len(group), "n_events": len(unique_events)}
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_median"] = float(values.median()) if len(values) else math.nan
        method_summary_rows.append(row)
    method_summary = pd.DataFrame(method_summary_rows)
    method_summary.to_csv(output_dir / "summary_by_method_outcome.csv", index=False)

    onset_rows = []
    if len(event_table):
        for (method, signal, outcome), group in event_table.groupby(
            ["method", "signal", "outcome_group"], dropna=False
        ):
            unique_events = group[["rollout_id", "event_index"]].drop_duplicates()
            response = pd.to_numeric(
                group["normalized_response_magnitude"], errors="coerce"
            )
            q95 = pd.to_numeric(
                group["clean_background_response_q95"], errors="coerce"
            )
            oriented = pd.to_numeric(
                group["normalized_failure_oriented_response"], errors="coerce"
            )
            valid_response = response.notna() & q95.notna()
            valid_oriented = oriented.notna()
            exceed = (response > q95) & valid_response
            consistent = (oriented > 0) & valid_oriented
            onset_rows.append({
                "method": method,
                "signal": signal,
                "outcome_group": outcome,
                "n_onset_events": len(unique_events),
                "n_exceeding_q95": int(exceed.sum()),
                "exceeding_q95_fraction": float(exceed.sum() / valid_response.sum())
                if int(valid_response.sum()) else math.nan,
                "direction_consistency_count_all": int(consistent.sum()),
                "direction_consistency_fraction_all": float(consistent.sum() / valid_oriented.sum())
                if int(valid_oriented.sum()) else math.nan,
            })
    onset_statistics = pd.DataFrame(onset_rows)
    onset_statistics.to_csv(output_dir / "onset_signal_statistics.csv", index=False)
    return {
        "events": event_table,
        "background": background_table,
        "summary": summary,
        "background_summary": background_summary,
        "method_summary": method_summary,
        "onset_statistics": onset_statistics,
    }


def plot_rollout_method(
    *,
    method: str,
    rollout_id: str,
    rollout: dict[str, Any],
    annotation: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    output_path: Path,
    pre_window_frames: int,
    post_window_frames: int,
) -> None:
    events = annotation["failure_events"]
    signals = rollout["methods"].get(method, {}).get("signals", {})
    if not signals:
        fig, ax = plt.subplots(figsize=(12, 3.6))
        reason = rollout.get("method_errors", {}).get(method, "raw signal unavailable")
        ax.text(0.5, 0.5, f"Raw signal unavailable\n{reason}", ha="center", va="center", wrap=True)
        ax.set_axis_off()
        ax.set_title(f"{METHOD_LABELS[method]} — unavailable — {rollout_id}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    if events:
        fig, axes = plt.subplots(len(events), 1, figsize=(12, max(3.2, 2.8 * len(events))), squeeze=False)
        axes_flat = axes[:, 0]
        for event_position, event in enumerate(events):
            ax = axes_flat[event_position]
            event_index = int(event.get("_analysis_event_index", event_position))
            if event.get("observable_onset_frame") is None:
                ax.text(0.5, 0.5, "No observable_onset_frame in annotation; event excluded from metrics", ha="center", va="center", wrap=True)
                ax.set_axis_off()
                ax.set_title(f"event {event_position + 1}: {event.get('failure_type', 'unknown')} — no observable onset")
                continue
            event_frame = int(event["observable_onset_frame"])
            rows = [row for row in metric_rows if row["event_index"] == event_index]
            for signal_name, series in signals.items():
                matching = next(row for row in rows if row["signal"] == signal_name)
                frames = series["frames"]
                values = series["values"]
                mask = (frames >= event_frame - pre_window_frames) & (frames <= event_frame + post_window_frames)
                scale = matching["robust_pre_scale"] or 1.0
                center = matching["pre_center"] or 0.0
                oriented = series["direction"] * (values[mask] - center) / scale
                ax.plot(frames[mask] - event_frame, oriented, marker="o", markersize=2.7, linewidth=1.1, label=signal_name)
            ax.axvline(0, color="#b2182b", linewidth=1.4, label="observable onset" if event_position == 0 else None)
            if event.get("recovery_frame") is not None:
                ax.axvline(int(event["recovery_frame"]) - event_frame, color="#2166ac", linestyle="--", linewidth=1.2, label="recovery" if event_position == 0 else None)
            if event.get("terminal_failure_frame") is not None:
                ax.axvline(int(event["terminal_failure_frame"]) - event_frame, color="#762a83", linestyle=":", linewidth=1.2, label="terminal failure" if event_position == 0 else None)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xlim(-pre_window_frames, post_window_frames)
            ax.set_ylabel("failure-oriented\nrobust units")
            ax.set_title(f"event {event_position + 1}: {event.get('failure_type', 'unknown')} @ video frame {event_frame}")
            ax.grid(alpha=0.22)
        axes_flat[-1].set_xlabel("video-frame offset from observable onset (native samples only)")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=False)
        fig.suptitle(f"{METHOD_LABELS[method]} — {rollout_id}\n{normalize_outcome(annotation)}", y=1.01)
    else:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for signal_name, series in signals.items():
            center, scale, _, _ = robust_center_scale(series["values"])
            oriented = series["direction"] * (series["values"] - center) / scale
            ax.plot(series["frames"], oriented, marker="o", markersize=2.7, linewidth=1.1, label=signal_name)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"{METHOD_LABELS[method]} — clean-success background — {rollout_id}")
        ax.set_xlabel("video frame (native samples only)")
        ax.set_ylabel("failure-oriented robust units")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_summaries(summaries: dict[str, pd.DataFrame], output_dir: Path) -> None:
    event_table = summaries["events"]
    background_summary = summaries["background_summary"]
    methods = list(METHODS)
    outcomes = ["recovered_success", "terminal_failure"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    for axis, method in zip(axes.flat, methods):
        group = event_table[event_table["method"] == method]
        signals = list(group["signal"].drop_duplicates())
        x = np.arange(len(outcomes))
        width = 0.8 / max(1, len(signals))
        for signal_index, signal in enumerate(signals):
            medians = []
            lows = []
            highs = []
            for outcome in outcomes:
                values = pd.to_numeric(group[(group["signal"] == signal) & (group["outcome_group"] == outcome)]["normalized_response_magnitude"], errors="coerce").dropna()
                medians.append(float(values.median()) if len(values) else math.nan)
                lows.append(float(values.quantile(0.25)) if len(values) else math.nan)
                highs.append(float(values.quantile(0.75)) if len(values) else math.nan)
            offset = (signal_index - (len(signals) - 1) / 2) * width
            axis.errorbar(x + offset, medians, yerr=[np.asarray(medians) - np.asarray(lows), np.asarray(highs) - np.asarray(medians)], fmt="o-", capsize=3, linewidth=1.1, label=signal)
        axis.set_xticks(x, ["recovered\nsuccess", "terminal\nfailure"])
        axis.set_title(METHOD_LABELS[method])
        axis.set_ylabel("normalized response magnitude\nmedian ± IQR")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle("Event response magnitude by outcome (higher = larger raw change vs robust pre-event scale)")
    fig.tight_layout()
    fig.savefig(output_dir / "grouped_response_magnitude.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    for axis, method in zip(axes.flat, methods):
        group = event_table[event_table["method"] == method]
        signals = list(group["signal"].drop_duplicates())
        x = np.arange(len(outcomes))
        width = 0.8 / max(1, len(signals))
        for signal_index, signal in enumerate(signals):
            medians = []
            lows = []
            highs = []
            for outcome in outcomes:
                values = pd.to_numeric(group[(group["signal"] == signal) & (group["outcome_group"] == outcome)]["post_event_persistence_fraction"], errors="coerce").dropna()
                medians.append(float(values.median()) if len(values) else math.nan)
                lows.append(float(values.quantile(0.25)) if len(values) else math.nan)
                highs.append(float(values.quantile(0.75)) if len(values) else math.nan)
            offset = (signal_index - (len(signals) - 1) / 2) * width
            axis.errorbar(x + offset, medians, yerr=[np.asarray(medians) - np.asarray(lows), np.asarray(highs) - np.asarray(medians)], fmt="o-", capsize=3, linewidth=1.1, label=signal)
        axis.set_xticks(x, ["recovered\nsuccess", "terminal\nfailure"])
        axis.set_ylim(-0.05, 1.05)
        axis.set_title(METHOD_LABELS[method])
        axis.set_ylabel("failure-oriented persistence fraction\nmedian ± IQR")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle("Post-event persistence by outcome")
    fig.tight_layout()
    fig.savefig(output_dir / "grouped_persistence.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    for axis, method in zip(axes.flat, methods):
        group = event_table[(event_table["method"] == method) & (event_table["outcome_group"] == "recovered_success")]
        signals = list(group["signal"].drop_duplicates())
        x = np.arange(len(signals))
        medians = []
        lows = []
        highs = []
        for signal in signals:
            values = pd.to_numeric(group[group["signal"] == signal]["recovery_rebound_normalized"], errors="coerce").dropna()
            medians.append(float(values.median()) if len(values) else math.nan)
            lows.append(float(values.quantile(0.25)) if len(values) else math.nan)
            highs.append(float(values.quantile(0.75)) if len(values) else math.nan)
        axis.errorbar(x, medians, yerr=[np.asarray(medians) - np.asarray(lows), np.asarray(highs) - np.asarray(medians)], fmt="o", capsize=4, linewidth=1.2)
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_xticks(x, signals, rotation=30, ha="right")
        axis.set_title(METHOD_LABELS[method])
        axis.set_ylabel("recovery rebound toward baseline\nnormalized median ± IQR")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Recovery rebound on recovered-success events")
    fig.tight_layout()
    fig.savefig(output_dir / "grouped_recovery_rebound.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Compare event response with the clean-success pseudo-event 95th percentile.
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = []
    event_medians = []
    background_p95 = []
    for method in methods:
        for signal in event_table[event_table["method"] == method]["signal"].drop_duplicates():
            event_values = pd.to_numeric(event_table[(event_table["method"] == method) & (event_table["signal"] == signal)]["normalized_response_magnitude"], errors="coerce").dropna()
            background_values = background_summary[(background_summary["method"] == method) & (background_summary["signal"] == signal)]
            if not len(event_values) or not len(background_values):
                continue
            labels.append(f"{METHOD_LABELS[method]}\n{signal}")
            event_medians.append(float(event_values.median()))
            background_p95.append(float(background_values.iloc[0]["normalized_response_magnitude_q95"]))
    positions = np.arange(len(labels))
    ax.scatter(positions, event_medians, label="annotated event median", zorder=3)
    ax.scatter(positions, background_p95, marker="x", label="clean background 95th percentile", zorder=3)
    for position, label in zip(positions, labels):
        ax.plot([position, position], [background_p95[position], event_medians[position]], color="#888888", linewidth=0.8)
    ax.set_xticks(positions, labels, rotation=45, ha="right")
    ax.set_ylabel("normalized response magnitude")
    ax.set_title("Annotated-event response versus clean-success background")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "clean_background_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_report(
    *,
    output_dir: Path,
    selection_path: Path,
    manifest_path: Path,
    annotation_dir: Path,
    selection_doc: dict[str, Any],
    selections: list[dict[str, Any]],
    rollouts: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    summaries: dict[str, pd.DataFrame],
    pre_window_frames: int,
    post_window_frames: int,
    background_stride_frames: int,
    localization: dict[str, Any] | None = None,
    comparison_path: Path | None = None,
) -> None:
    event_table = summaries["events"]
    background_summary = summaries["background_summary"]
    event_identity = event_table[["rollout_id", "event_index"]].drop_duplicates() if len(event_table) else pd.DataFrame()
    recovered_event_count = len(event_table[event_table["outcome_group"] == "recovered_success"][["rollout_id", "event_index"]].drop_duplicates()) if len(event_table) else 0
    terminal_event_count = len(event_table[event_table["outcome_group"] == "terminal_failure"][["rollout_id", "event_index"]].drop_duplicates()) if len(event_table) else 0
    uncertain_event_count = len(event_table[event_table["outcome_group"] == "uncertain"][["rollout_id", "event_index"]].drop_duplicates()) if len(event_table) else 0
    annotated_event_total = sum(len(annotation["failure_events"]) for annotation in annotations.values())
    observable_event_total = sum(
        sum(event.get("observable_onset_frame") is not None for event in annotation["failure_events"])
        for annotation in annotations.values()
    )
    clean_success_count = sum(normalize_outcome(annotation) == "clean_success" for annotation in annotations.values())
    method_coverage_text = "; ".join(
        f"{METHOD_LABELS[method]} {sum(method in rollout['methods'] for rollout in rollouts.values())}/{len(rollouts)}"
        for method in METHODS
    )
    lines = [
        "# LF3R Baseline Temporal Signal Analysis",
        "",
        f"范围：{len(selections)} 条 baseline selection rollout；使用当前 run 的实际输出覆盖进行分析。",
        "",
        "## Data and alignment",
        "",
        f"- Selection: `{selection_path}` (SHA-256 `{sha256(selection_path)}`).",
        f"- Manifest: `{manifest_path}`; annotation records: `{annotation_dir}`.",
        f"- Rollouts: {len(rollouts)}; annotated events total: {annotated_event_total}; observable-onset events: {observable_event_total}; skipped without observable onset: {annotated_event_total - observable_event_total}; recovered-success events: {recovered_event_count}; terminal-failure events: {terminal_event_count}; uncertain events: {uncertain_event_count}.",
        f"- Method raw coverage (available/selected): {method_coverage_text}. Missing outputs are retained as explicit unavailable plots and excluded from metric denominators.",
        f"- Degenerate pre-event windows: {int((event_table['robust_scale_source'] == 'clean_background_fallback').sum())} signal rows use the median non-flat clean-success scale for the same method/signal; raw pre-scale is retained.",
        f"- Event window: [{-pre_window_frames}, +{post_window_frames}] video frames around `observable_onset_frame`; no interpolation or resampling is performed.",
        f"- Clean background: pseudo-event centers every {background_stride_frames} video frames on {clean_success_count} clean-success trajectories, using the same native samples and robust statistics.",
        "- Annotation frame semantics are video indices. SAFE converts action_timestep minus first_environment_timestep; ProcVLM uses raw frame_index and clips any decoder endpoint outside range only in analysis coordinates; RynnValue uses sampled_indices; Robo-Dopamine parses the AF frame from official image paths. Raw indices remain in metadata.",
        "",
        "## Metric definitions",
        "",
        "- Pre/post centers are medians of native samples in the requested windows; signed_pre_post_change = post_center - pre_center preserves the raw sign.",
        "- robust_pre_scale is 1.4826 times MAD(pre-window); IQR is used only when MAD is zero, and flat windows use the same method/signal clean-success median scale with robust_scale_source=clean_background_fallback.",
        "- normalized_response_magnitude = abs(signed_pre_post_change) / robust_pre_scale; normalized_failure_oriented_response multiplies the signed change by the method direction.",
        "- Persistence is the fraction of native post-event samples at least 0.5 times robust_pre_scale in the failure direction, capped before the next annotated event or recovery when present.",
        "- For annotated recovery, rebound is the recovery sample movement from the post-event center toward the pre-event center, normalized by the same scale; the nearest native sample and its frame error are recorded.",
        "",
        "Failure localization uses failure-oriented robust scores at native sample points only. Clean-success pseudo-events calibrate independent Q90/Q95/Q99 thresholds for every method/signal. The search window is clipped by recovery, terminal failure, the next annotated event, and the requested post window.",
        "A detected event uses the first threshold crossing anywhere in the bounded window; its signed lead/lag and absolute localization error are measured against observable onset. The first post-onset crossing is also retained, while an earlier crossing is marked as an early alarm. Misses have no localization error denominator. Precision/F1 and AUROC/AUPRC are descriptive event-level diagnostics using clean-success pseudo-events as negatives.",
        "",
        "## Native temporal coverage",
        "",
        "| Method | Native signal(s) | Samples per rollout (range) | Failure direction used for oriented metrics |",
        "| --- | --- | ---: | --- |",
    ]
    for method in METHODS:
        counts = []
        signal_names = []
        for rollout in rollouts.values():
            if method not in rollout["methods"]:
                continue
            signal_names = list(rollout["methods"][method]["signals"])
            counts.append(max(len(series["frames"]) for series in rollout["methods"][method]["signals"].values()))
        lines.append(f"| {METHOD_LABELS[method]} | {', '.join(signal_names)} | {min(counts) if counts else 'n/a'}–{max(counts) if counts else 'n/a'} | {('increase' if METHOD_SIGNAL_DIRECTIONS[method] > 0 else 'decrease')} |")
    lines += [
        "",
        "Failure-oriented metrics multiply the raw signed change by the method direction: SAFE and RynnValue use increases; ProcVLM progress and Robo-Dopamine progress/hop use decreases. Raw signed changes remain in the machine-readable files.",
        "",
        "## Event-level findings",
        "",
    ]
    if len(event_table):
        for method in METHODS:
            group = event_table[event_table["method"] == method]
            if not len(group):
                continue
            lines.append(f"### {METHOD_LABELS[method]}")
            for signal in group["signal"].drop_duplicates():
                signal_group = group[group["signal"] == signal]
                summaries_by_outcome = []
                for outcome in ("recovered_success", "terminal_failure"):
                    subset = signal_group[signal_group["outcome_group"] == outcome]
                    if len(subset):
                        summaries_by_outcome.append(
                            f"{outcome}: response {fmt(subset['normalized_response_magnitude'].median())}, persistence {fmt(subset['post_event_persistence_fraction'].median())}"
                        )
                lines.append(f"- `{signal}` — " + "; ".join(summaries_by_outcome) + ".")
    pattern_lines = ["", "## Recurring patterns", ""]
    for method in METHODS:
        method_events = event_table[event_table["method"] == method]
        recovered = method_events[method_events["outcome_group"] == "recovered_success"]
        terminal = method_events[method_events["outcome_group"] == "terminal_failure"]
        if len(recovered) and len(terminal):
            recovered_response = pd.to_numeric(recovered["normalized_response_magnitude"], errors="coerce").dropna()
            terminal_response = pd.to_numeric(terminal["normalized_response_magnitude"], errors="coerce").dropna()
            recovered_persistence = pd.to_numeric(recovered["post_event_persistence_fraction"], errors="coerce").dropna()
            terminal_persistence = pd.to_numeric(terminal["post_event_persistence_fraction"], errors="coerce").dropna()
            pattern_lines.append(f"- {METHOD_LABELS[method]}: normalized response median recovered={fmt(recovered_response.median() if len(recovered_response) else math.nan)}, terminal={fmt(terminal_response.median() if len(terminal_response) else math.nan)}; persistence median recovered={fmt(recovered_persistence.median() if len(recovered_persistence) else math.nan)}, terminal={fmt(terminal_persistence.median() if len(terminal_persistence) else math.nan)}.")
    for method in METHODS:
        method_events = event_table[event_table["method"] == method]
        recovered = method_events[method_events["outcome_group"] == "recovered_success"]
        terminal = method_events[method_events["outcome_group"] == "terminal_failure"]
        available = sum(method in rollout["methods"] for rollout in rollouts.values())
        if len(recovered) and len(terminal):
            recovered_magnitude = pd.to_numeric(recovered["normalized_response_magnitude"], errors="coerce").dropna()
            terminal_magnitude = pd.to_numeric(terminal["normalized_response_magnitude"], errors="coerce").dropna()
            recovered_oriented = pd.to_numeric(recovered["normalized_failure_oriented_response"], errors="coerce").dropna()
            terminal_oriented = pd.to_numeric(terminal["normalized_failure_oriented_response"], errors="coerce").dropna()
            recovered_positive = float((recovered_oriented > 0).mean()) if len(recovered_oriented) else math.nan
            terminal_positive = float((terminal_oriented > 0).mean()) if len(terminal_oriented) else math.nan
            pattern_lines.append(
                f"- {METHOD_LABELS[method]} (raw coverage {available}/{len(rollouts)}): terminal/recovered response medians {fmt(terminal_magnitude.median() if len(terminal_magnitude) else math.nan)}/{fmt(recovered_magnitude.median() if len(recovered_magnitude) else math.nan)}; failure-oriented medians {fmt(terminal_oriented.median() if len(terminal_oriented) else math.nan)}/{fmt(recovered_oriented.median() if len(recovered_oriented) else math.nan)}; positive-oriented fractions {fmt(terminal_positive)}/{fmt(recovered_positive)}."
            )
            recovery = pd.to_numeric(recovered["recovery_rebound_normalized"], errors="coerce").dropna()
            if len(recovery):
                pattern_lines.append(f"- {METHOD_LABELS[method]} recovery rebound median on recovered-success events: {fmt(recovery.median())} robust units.")
        elif len(method_events):
            pattern_lines.append(f"- {METHOD_LABELS[method]}: {len(method_events)} event-signal rows available; recovered/terminal comparison is incomplete in this coverage.")
    pattern_lines += [
        "- Normalized response magnitude is unsigned; the failure-oriented signed response and persistence must be read together, especially when a raw change is large but moves opposite the expected failure direction.",
        "- Missing method outputs are explicit in the coverage table and unavailable plots; they are not imputed or included in event/background metric denominators.",
        "",
        "## Clean-success reference and exceptions",
        "",
    ]
    lines += pattern_lines
    above = 0
    total = 0
    for _, row in event_table.iterrows():
        bg = background_summary[(background_summary["method"] == row["method"]) & (background_summary["signal"] == row["signal"])]
        if len(bg) and pd.notna(row["normalized_response_magnitude"]):
            total += 1
            if row["normalized_response_magnitude"] > bg.iloc[0]["normalized_response_magnitude_q95"]:
                above += 1
    lines.append(f"{above}/{total} event-signal rows exceed the corresponding clean-success background 95th percentile of normalized response magnitude.")
    if localization is not None and len(localization.get("summary", [])):
        q95 = localization["summary"][
            (localization["summary"]["threshold"] == "q95")
            & (localization["summary"]["outcome_group"] == "all_events")
        ]
        lines += [
            "",
            "## Failure localization summary",
            "",
            f"Generated {len(localization['event_rows'])} event-level threshold rows across {len(localization['thresholds'])} method/signal calibrations.",
        ]
        for _, row in q95.iterrows():
            lines.append(
                f"{METHOD_LABELS.get(str(row['method']), row['method'])}/{row['signal']}: "
                f"Q95 recall={fmt(row.get('recall'))}, false-alarm rate={fmt(row.get('false_alarm_rate'))}, "
                f"F1={fmt(row.get('f1'))}, AUROC={fmt(row.get('auroc'))}, AP={fmt(row.get('average_precision'))}."
            )
    if comparison_path is not None:
        lines.append(f"Comparison with previous full_136 snapshot: {comparison_path}.")
    if len(event_table):
        largest = event_table.sort_values("normalized_response_magnitude", ascending=False).head(5)
        lines.append("")
        lines.append("Largest normalized responses:")
        for _, row in largest.iterrows():
            lines.append(f"- `{row['method']}/{row['signal']}` rollout `{row['rollout_id']}`, event {int(row['event_index']) + 1} ({row['outcome_group']}, {row['failure_type']}): {fmt(row['normalized_response_magnitude'])} robust units.")
    lines += [
        "",
        f"Interpretation is deliberately descriptive: {len(rollouts)} trajectories and sparse native samples are not sufficient to estimate detector operating characteristics or causality. In particular, missing post samples are recorded with `post_window_fallback=true` rather than filled by interpolation, and recovery frames near an episode boundary may use the nearest native sample.",
        "",
        "## Artifacts",
        "",
        "localization_event_metrics.jsonl: compact event x method x signal x Q90/Q95/Q99 rows with native crossing, lead/lag, error, and tolerance flags.",
        "localization_summary.csv, localization_by_failure_type.csv, localization_thresholds.csv: thresholded recall, false-alarm, precision/F1, AUROC/AUPRC, error/tolerance and failure-type summaries.",
        "- `event_metrics.jsonl`: one row per analyzable observable-onset event × available method × native signal, including raw signed change, robust pre-scale, persistence, recovery rebound, and exact window sample frames.",
        "- `clean_background_metrics.jsonl` and `clean_background_summary.csv`: pseudo-event reference distribution from clean success.",
        "- `summary_by_method_signal_outcome.csv`, `summary_by_method_outcome.csv`, `method_coverage.csv`: grouped metrics and method-coverage tables.",
        "- `plots/per_rollout/<method>/<rollout_id>.png`: event-aligned plots; clean successes use full-trajectory background plots.",
        "- `plots/grouped_*.png`: grouped response, persistence, recovery, and clean-background comparisons.",
        "",
        "The analysis is rerunnable with a separate selection JSON or an authoritative baseline jobs.jsonl plus the corresponding four run roots; incomplete method coverage remains explicit in metadata and method_coverage.csv.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_run_sources(
    source_roots: dict[str, list[Path]],
    selected_ids: set[str],
) -> tuple[
    dict[str, dict[str, Path]],
    dict[str, dict[str, Any]],
]:
    """Validate run metadata and map each recorded rollout to one source root."""
    source_by_id: dict[str, dict[str, Path]] = {}
    metadata_by_method: dict[str, dict[str, Any]] = {}
    accepted_statuses = {"complete", "complete_with_errors"}
    for method in METHODS:
        roots = source_roots.get(method) or []
        if not roots:
            raise ValueError(f"Missing {method} run")
        method_source_by_id: dict[str, Path] = {}
        source_runs = []
        for root in roots:
            root = root.resolve()
            metadata_path = root / "run.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("baseline") not in (None, method):
                raise ValueError(f"Run is not a {method} run: {metadata_path}")
            if metadata.get("status") not in accepted_statuses:
                raise ValueError(f"Run is not complete: {metadata_path}")
            ids = run_rollout_id_list(root)
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate {method} rollout IDs inside source run: {metadata_path}")
            duplicates = sorted(set(ids) & set(method_source_by_id))
            if duplicates:
                raise ValueError(
                    f"Duplicate {method} rollout IDs across source runs: {duplicates[:5]}"
                )
            for rollout_id in ids:
                method_source_by_id[rollout_id] = root
            try:
                selected_count = int(metadata.get("selected_rollouts", 0))
                completed_count = int(metadata.get("completed_jobs", 0))
                failed_count = int(metadata.get("failed_jobs", 0))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Run has invalid job counts: {metadata_path}") from error
            source_runs.append({
                "run_root": str(root),
                "run_json": str(metadata_path.resolve()),
                "status": metadata.get("status"),
                "manifest": metadata.get("manifest"),
                "manifest_sha256": metadata.get("manifest_sha256"),
                "parameters": metadata.get("arguments") or {},
                "environment_python": metadata.get("environment_python"),
                "command": metadata.get("command"),
                "selected_rollouts": selected_count,
                "completed_jobs": completed_count,
                "failed_jobs": failed_count,
                "available_job_ids": sorted(set(ids)),
            })
        missing_ids = sorted(set(selected_ids) - set(method_source_by_id))
        status_values = {str(source["status"]) for source in source_runs}
        aggregate_status = (
            next(iter(status_values))
            if len(status_values) == 1 and len(source_runs) == 1
            else "merged"
        )
        if missing_ids:
            aggregate_status = "partial"
        source_by_id[method] = method_source_by_id
        metadata_by_method[method] = {
            "status": aggregate_status,
            "selected_rollouts": len(selected_ids),
            "available_job_ids": sorted(method_source_by_id),
            "missing_rollout_ids": missing_ids,
            "source_runs": source_runs,
            "source_run_count": len(source_runs),
            "completed_jobs": sum(source["completed_jobs"] for source in source_runs),
            "failed_jobs": sum(source["failed_jobs"] for source in source_runs),
        }
    return source_by_id, metadata_by_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/baseline_signal_analysis/representative_20260826")
    for method in METHODS:
        primary_flag = f"--{method}-run"
        aliases = [primary_flag]
        if "_" in method:
            aliases.append(f"--{method.replace('_', '-')}-run")
        if method == "rynnvalue":
            parser.add_argument(*aliases, dest=f"{method}_run", type=Path, action="append", required=True)
        else:
            parser.add_argument(*aliases, dest=f"{method}_run", type=Path, required=True)
    parser.add_argument("--pre-window-frames", type=int, default=60)
    parser.add_argument("--post-window-frames", type=int, default=60)
    parser.add_argument("--background-stride-frames", type=int, default=30)
    parser.add_argument(
        "--compare-snapshot",
        type=Path,
        default=PROJECT_ROOT / "outputs/baseline_signal_analysis/full_136_20260827",
        help="Optional previous snapshot used for a descriptive comparison table",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pre_window_frames < 1 or args.post_window_frames < 1 or args.background_stride_frames < 1:
        raise ValueError("window and stride values must be positive")
    selection_path = resolve_path(args.selection)
    manifest_path = resolve_path(args.manifest)
    annotation_dir = resolve_path(args.annotations_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    selection_doc, selections = load_selection(selection_path)
    manifest = load_manifest(manifest_path)
    selected_ids = [row["id"] for row in selections]
    annotations = {}
    for row in selections:
        rollout_id = row["id"]
        if rollout_id not in manifest:
            raise KeyError(f"Selection ID is not in manifest: {rollout_id}")
        annotations[rollout_id] = load_annotations(annotation_dir, rollout_id)

    source_roots = {
        method: [
            resolve_path(value)
            for value in (
                getattr(args, f"{method}_run")
                if isinstance(getattr(args, f"{method}_run"), list)
                else [getattr(args, f"{method}_run")]
            )
        ]
        for method in METHODS
    }
    source_by_id, run_metadata_by_method = _validate_run_sources(
        source_roots,
        set(selected_ids),
    )

    rollouts: dict[str, dict[str, Any]] = {}
    event_rows: list[dict[str, Any]] = []
    for selection in selections:
        rollout_id = selection["id"]
        record = manifest[rollout_id]
        total_frames = int(record["total_frames"])
        annotation = annotations[rollout_id]
        methods = {}
        method_errors = {}
        for method in METHODS:
            run_root = source_by_id[method].get(rollout_id)
            if run_root is None:
                method_errors[method] = (
                    f"rollout {rollout_id} is not recorded in the selected {method} source run(s)"
                )
                continue
            try:
                methods[method] = load_method_signal(method, run_root, rollout_id, record)
            except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                method_errors[method] = str(error)
        rollouts[rollout_id] = {
            "record": record,
            "selection": selection,
            "annotation": annotation,
            "methods": methods,
            "method_errors": method_errors,
        }
        annotated_events = annotation["failure_events"]
        observable_events = [
            (event_index, event)
            for event_index, event in enumerate(annotated_events)
            if event.get("observable_onset_frame") is not None
        ]
        annotated_onsets = [int(event["observable_onset_frame"]) for _, event in observable_events]
        outcome_group = normalize_outcome(annotation)
        for observable_position, (event_index, event) in enumerate(observable_events):
            event_frame = int(event["observable_onset_frame"])
            next_event = annotated_onsets[observable_position + 1] if observable_position + 1 < len(annotated_onsets) else None
            for method, method_data in methods.items():
                for signal_name, series in method_data["signals"].items():
                    row = event_metric(
                        series=series,
                        event_frame=event_frame,
                        total_frames=total_frames,
                        event_index=event_index,
                        event_type="annotated_event",
                        outcome_group=outcome_group,
                        failure_type=str(event.get("failure_type", annotation.get("failure_type", "other"))),
                        recovery_frame=event.get("recovery_frame"),
                        terminal_failure_frame=event.get("terminal_failure_frame"),
                        next_event_frame=next_event,
                        pre_window_frames=args.pre_window_frames,
                        post_window_frames=args.post_window_frames,
                    )
                    row.update({"rollout_id": rollout_id, "method": method, "signal": signal_name})
                    event_rows.append(row)
        # Clean successes have no annotated event; generate pseudo-events to anchor background variability.
        if outcome_group == "clean_success":
            for method, method_data in methods.items():
                for center in background_centers(total_frames, args.background_stride_frames):
                    for signal_name, series in method_data["signals"].items():
                        row = event_metric(
                            series=series,
                            event_frame=center,
                            total_frames=total_frames,
                            event_index=-1,
                            event_type="clean_background_pseudo_event",
                            outcome_group="clean_success_background",
                            failure_type="none_success",
                            recovery_frame=None,
                            terminal_failure_frame=None,
                            next_event_frame=None,
                            pre_window_frames=args.pre_window_frames,
                            post_window_frames=args.post_window_frames,
                            is_background=True,
                        )
                        row.update({"rollout_id": rollout_id, "method": method, "signal": signal_name})
                        if row["pre_sample_count"] >= 2 and row["post_sample_count"] >= 1:
                            event_rows.append(row)

    # A flat pre-event window has no empirical noise estimate. Recompute those rows
    # with the median non-flat clean-success scale for the same method/signal;
    # retain raw_pre_scale and mark the source explicitly.
    background_scale_refs = {}
    for background_row in event_rows:
        if not background_row["is_background"] or background_row["robust_scale_source"] == "floor":
            continue
        scale = background_row.get("robust_pre_scale")
        if scale is None or not math.isfinite(scale) or scale <= 0.0:
            continue
        background_scale_refs.setdefault((background_row["method"], background_row["signal"]), []).append(float(scale))
    background_scale_refs = {key: float(np.median(values)) for key, values in background_scale_refs.items()}
    normalization_fallback_rows = 0
    for row_position, row in enumerate(event_rows):
        if row["robust_scale_source"] != "floor":
            continue
        scale_override = background_scale_refs.get((row["method"], row["signal"]))
        if scale_override is None:
            continue
        rollout = rollouts[row["rollout_id"]]
        series = rollout["methods"][row["method"]]["signals"][row["signal"]]
        updated = event_metric(
            series=series,
            event_frame=int(row["event_frame"]),
            total_frames=int(rollout["record"]["total_frames"]),
            event_index=int(row["event_index"]),
            event_type=row["event_type"],
            outcome_group=row["outcome_group"],
            failure_type=row["failure_type"],
            recovery_frame=row["recovery_frame"],
            terminal_failure_frame=row["terminal_failure_frame"],
            next_event_frame=row["next_event_frame"],
            pre_window_frames=args.pre_window_frames,
            post_window_frames=args.post_window_frames,
            is_background=bool(row["is_background"]),
            scale_override=scale_override,
        )
        updated.update({"rollout_id": row["rollout_id"], "method": row["method"], "signal": row["signal"]})
        event_rows[row_position] = updated
        normalization_fallback_rows += 1

    # Attach clean-background reference values to every annotated event row.
    background_rows = [row for row in event_rows if row["is_background"]]
    for row in event_rows:
        if row["is_background"]:
            continue
        matching = [x for x in background_rows if x["method"] == row["method"] and x["signal"] == row["signal"]]
        response_values = [x["normalized_response_magnitude"] for x in matching if x["normalized_response_magnitude"] is not None and math.isfinite(x["normalized_response_magnitude"])]
        row["clean_background_pseudo_event_count"] = len(response_values)
        if response_values and row["normalized_response_magnitude"] is not None and math.isfinite(row["normalized_response_magnitude"]):
            row["clean_background_response_median"] = float(np.median(response_values))
            row["clean_background_response_q95"] = float(np.quantile(response_values, 0.95))
            row["clean_background_response_percentile"] = float((np.sum(np.asarray(response_values) <= row["normalized_response_magnitude"]) + 0.5) / len(response_values))
        else:
            row["clean_background_response_median"] = math.nan
            row["clean_background_response_q95"] = math.nan
            row["clean_background_response_percentile"] = math.nan

    localization = compute_failure_localization(
        event_rows=event_rows,
        rollouts=rollouts,
        post_window_frames=args.post_window_frames,
    )
    write_localization_outputs(localization, output_dir)

    method_coverage = {}
    coverage_rows = []
    for method in METHODS:
        available_rollouts = [rollout_id for rollout_id, rollout in rollouts.items() if method in rollout["methods"]]
        missing_rollouts = [rollout_id for rollout_id, rollout in rollouts.items() if method not in rollout["methods"]]
        load_errors = {rollout_id: rollouts[rollout_id]["method_errors"].get(method, "raw signal unavailable") for rollout_id in missing_rollouts}
        method_coverage[method] = {
            "selected_rollouts": len(selections),
            "available_rollouts": len(available_rollouts),
            "missing_rollouts": len(missing_rollouts),
            "available_rollout_ids": available_rollouts,
            "missing_rollout_ids": missing_rollouts,
            "load_errors": load_errors,
            "run": run_metadata_by_method[method],
        }
        coverage_rows.append({
            "method": method,
            "selected_rollouts": len(selections),
            "available_rollouts": len(available_rollouts),
            "missing_rollouts": len(missing_rollouts),
            "run_status": run_metadata_by_method[method]["status"],
            "run_completed_jobs": run_metadata_by_method[method]["completed_jobs"],
            "run_failed_jobs": run_metadata_by_method[method]["failed_jobs"],
        })
    pd.DataFrame(coverage_rows).to_csv(output_dir / "method_coverage.csv", index=False)

    event_jsonl = output_dir / "event_metrics.jsonl"
    background_jsonl = output_dir / "clean_background_metrics.jsonl"
    with event_jsonl.open("w", encoding="utf-8") as handle:
        for row in event_rows:
            if not row["is_background"]:
                handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
    with background_jsonl.open("w", encoding="utf-8") as handle:
        for row in event_rows:
            if row["is_background"]:
                handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")

    summaries = summarize_metrics(event_rows, output_dir)
    comparison_path = write_comparison_table(
        output_dir=output_dir,
        current_summary=summaries["summary"],
        localization_summary=localization["summary"],
        old_snapshot=resolve_path(args.compare_snapshot) if args.compare_snapshot else None,
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    for rollout_id, rollout in rollouts.items():
        annotation = annotations[rollout_id]
        for method in METHODS:
            plot_rollout_method(
                method=method,
                rollout_id=rollout_id,
                rollout=rollout,
                annotation=annotation,
                metric_rows=[row for row in event_rows if row["rollout_id"] == rollout_id and row["method"] == method and not row["is_background"]],
                output_path=plot_dir / "per_rollout" / method / f"{rollout_id}.png",
                pre_window_frames=args.pre_window_frames,
                post_window_frames=args.post_window_frames,
            )
    plot_grouped_summaries(summaries, plot_dir)

    native_alignment = {}
    for rollout_id, rollout in rollouts.items():
        native_alignment[rollout_id] = {}
        for method, method_data in rollout["methods"].items():
            native_alignment[rollout_id][method] = {
                "adapter_alignment": method_data.get("alignment", {}),
                "signals": {
                    signal_name: {
                        "sample_count": int(len(series["frames"])),
                        "first_analysis_frame": int(series["frames"][0]),
                        "last_analysis_frame": int(series["frames"][-1]),
                    }
                    for signal_name, series in method_data["signals"].items()
                },
            }

    annotated_event_total = sum(len(annotation["failure_events"]) for annotation in annotations.values())
    observable_annotated_event_total = sum(
        sum(event.get("observable_onset_frame") is not None for event in annotation["failure_events"])
        for annotation in annotations.values()
    )
    annotation_coverage = []
    for rollout_id in selected_ids:
        annotation = annotations[rollout_id]
        for event_index, event in enumerate(annotation["failure_events"]):
            annotation_coverage.append({
                "rollout_id": rollout_id,
                "event_index": event_index,
                "failure_type": event.get("failure_type", annotation.get("failure_type", "other")),
                "outcome_group": normalize_outcome(annotation),
                "observable_onset_frame": event.get("observable_onset_frame"),
                "has_observable_onset": event.get("observable_onset_frame") is not None,
                "recovery_frame": event.get("recovery_frame"),
                "terminal_failure_frame": event.get("terminal_failure_frame"),
            })
    metadata = {
        "schema_version": 2,
        "analysis": "lf3r_baseline_temporal_signals",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "annotations_dir": str(annotation_dir),
        "run_roots": {
            method: [str(root) for root in source_roots[method]]
            if len(source_roots[method]) != 1
            else str(source_roots[method][0])
            for method in METHODS
        },
        "source_runs": {
            method: run_metadata_by_method[method]["source_runs"]
            for method in METHODS
        },
        "run_metadata": run_metadata_by_method,
        "method_coverage": method_coverage,
        "native_alignment": native_alignment,
        "annotation_coverage": annotation_coverage,
        "rollouts": selected_ids,
        "methods": list(METHODS),
        "pre_window_frames": args.pre_window_frames,
        "post_window_frames": args.post_window_frames,
        "background_stride_frames": args.background_stride_frames,
        "frame_coordinate": "video_frame_index",
        "native_sampling_preserved": True,
        "orientation": METHOD_SIGNAL_DIRECTIONS,
        "signal_units": METHOD_SIGNAL_UNITS,
        "comparison_snapshot": str(resolve_path(args.compare_snapshot)) if args.compare_snapshot and resolve_path(args.compare_snapshot).is_dir() else None,
        "comparison_path": str(comparison_path) if comparison_path else None,
        "localization": {
            "available": bool(len(localization["event_rows"])),
            "thresholds": list(LOCALIZATION_THRESHOLDS),
            "event_metrics": len(localization["event_rows"]),
            "summary_rows": len(localization["summary"]),
            "failure_type_rows": len(localization["by_failure_type"]),
        },
        "counts": {
            "rollouts": len(rollouts),
            "annotated_events_total": annotated_event_total,
            "annotated_events_with_observable_onset": observable_annotated_event_total,
            "annotated_events_skipped_no_observable_onset": annotated_event_total - observable_annotated_event_total,
            "annotated_events": len({(row["rollout_id"], row["event_index"]) for row in event_rows if not row["is_background"]}),
            "event_signal_rows": sum(not row["is_background"] for row in event_rows),
            "background_pseudo_event_signal_rows": sum(row["is_background"] for row in event_rows),
            "normalization_fallback_rows": normalization_fallback_rows,
            "localization_event_metrics": len(localization["event_rows"]),
            "localization_summary_rows": len(localization["summary"]),
            "localization_failure_type_rows": len(localization["by_failure_type"]),
            "localization_threshold_rows": len(localization["thresholds"]),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(json_safe(metadata), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(
        output_dir=output_dir,
        selection_path=selection_path,
        manifest_path=manifest_path,
        annotation_dir=annotation_dir,
        selection_doc=selection_doc,
        selections=selections,
        rollouts=rollouts,
        annotations=annotations,
        summaries=summaries,
        pre_window_frames=args.pre_window_frames,
        post_window_frames=args.post_window_frames,
        background_stride_frames=args.background_stride_frames,
        localization=localization,
        comparison_path=comparison_path,
    )

    print("BASELINE_TEMPORAL_SIGNAL_ANALYSIS_OK")
    print(f"output_dir={output_dir}")
    print(f"rollouts={len(rollouts)} annotated_events={metadata['counts']['annotated_events']} event_signal_rows={metadata['counts']['event_signal_rows']}")
    print(f"background_pseudo_event_signal_rows={metadata['counts']['background_pseudo_event_signal_rows']}")
    print(f"localization_event_metrics={metadata['counts']['localization_event_metrics']}")
    print(f"localization_summary_rows={metadata['counts']['localization_summary_rows']}")
    print(f"plots={len(list(plot_dir.rglob('*.png')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
