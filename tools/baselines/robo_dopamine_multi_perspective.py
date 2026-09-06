#!/usr/bin/env python3
"""Helpers for official Robo-Dopamine multi-perspective fusion.

The official repository exposes the three prediction modes independently and
recommends averaging their inference reward results.  This module keeps that
operation deliberately small and explicit: it never resamples a curve and it
only fuses rows whose native sampled frame indices match exactly.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


PERSPECTIVE_MODES = ("incremental", "forward", "backward")
FUSED_EVAL_MODE = "fused"
_FRAME_RE = re.compile(r"frame_(\d+)[.]png|af_(\d+)(?:$|[^0-9])")


def normalize_eval_modes(value: Any) -> list[str]:
    """Return unique modes in official order, accepting CLI/list forms."""
    if value is None:
        return []
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in value if str(item).strip()]
    unknown = sorted(set(values) - set(PERSPECTIVE_MODES))
    if unknown:
        raise ValueError(f"Unknown Robo-Dopamine evaluation mode(s): {unknown}")
    if len(set(values)) != len(values):
        raise ValueError("Robo-Dopamine evaluation modes must not contain duplicates")
    return [mode for mode in PERSPECTIVE_MODES if mode in values]


def resolve_eval_modes(eval_mode: Any = None, explicit_modes: Any = None) -> list[str]:
    """Resolve the wrapper mode into the official mode list.

    ``fused`` is a wrapper-level mode: the official model still runs its
    three native perspectives and LF3R fuses their progress outputs. An
    explicit ``eval_modes`` list remains authoritative for compatibility with
    existing CLI callers.
    """
    if explicit_modes:
        return normalize_eval_modes(explicit_modes)
    mode = str(eval_mode or "").strip().lower()
    if mode == FUSED_EVAL_MODE:
        return list(PERSPECTIVE_MODES)
    if not mode:
        return []
    return normalize_eval_modes([mode])


def frame_index(row: Mapping[str, Any]) -> int:
    """Extract the official AF frame index without changing its native grid."""
    explicit = row.get("frame_index")
    if explicit is not None:
        return int(explicit)
    images = row.get("image") or []
    candidates = [images[5]] if len(images) > 5 else []
    candidates.append(row.get("id", ""))
    for candidate in candidates:
        match = _FRAME_RE.search(str(candidate))
        if match:
            return int(match.group(1) or match.group(2))
    raise ValueError(f"Cannot determine AF frame index from Robo-Dopamine row: {row.get('id')!r}")


def _finite_number(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def _rows_by_frame(path: Path, mode: str) -> tuple[list[int], dict[int, dict[str, Any]]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Robo-Dopamine prediction file is not a list: {path}")
    ordered: list[int] = []
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Robo-Dopamine prediction row is not an object: {path}")
        index = frame_index(row)
        if index in indexed:
            raise ValueError(f"Duplicate frame index {index} in {mode}: {path}")
        # Official GRMInference writes this progress field after parsing the
        # mode-specific score.  Fusion must use that field, not re-interpret
        # the raw <score> text with a new formula.
        _finite_number(row.get("progress"), label=f"{mode} progress at frame {index}")
        _finite_number(row.get("hop"), label=f"{mode} hop at frame {index}")
        ordered.append(index)
        indexed[index] = row
    return ordered, indexed


def fuse_prediction_files(
    prediction_paths: Mapping[str, Path],
    output_path: Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse official mode outputs using the documented arithmetic mean.

    The official ``run_pipeline`` emits one row per sampled AFTER frame and
    uses the same ``make_sample_indices_by_interval`` grid for all modes.  We
    verify that invariant by frame index instead of relying on row position.
    No interpolation, padding, or endpoint insertion is performed.
    """
    missing = [mode for mode in PERSPECTIVE_MODES if mode not in prediction_paths]
    if missing:
        raise ValueError(f"Official fusion requires all modes; missing {missing}")

    ordered_by_mode: dict[str, list[int]] = {}
    rows_by_mode: dict[str, dict[int, dict[str, Any]]] = {}
    for mode in PERSPECTIVE_MODES:
        ordered, indexed = _rows_by_frame(Path(prediction_paths[mode]), mode)
        ordered_by_mode[mode] = ordered
        rows_by_mode[mode] = indexed

    reference_indices = ordered_by_mode["incremental"]
    reference_set = set(reference_indices)
    for mode in PERSPECTIVE_MODES[1:]:
        current = ordered_by_mode[mode]
        if set(current) != reference_set:
            raise ValueError(
                "Robo-Dopamine mode grids do not match exactly: "
                f"incremental={reference_indices}, {mode}={current}"
            )

    fused_rows: list[dict[str, Any]] = []
    previous = 0.0
    for index in reference_indices:
        progress = {
            mode: _finite_number(
                rows_by_mode[mode][index]["progress"],
                label=f"{mode} progress at frame {index}",
            )
            for mode in PERSPECTIVE_MODES
        }
        hops = {
            mode: _finite_number(
                rows_by_mode[mode][index]["hop"],
                label=f"{mode} hop at frame {index}",
            )
            for mode in PERSPECTIVE_MODES
        }
        fused_progress = sum(progress.values()) / len(PERSPECTIVE_MODES)
        fused_hop = fused_progress - previous
        source = rows_by_mode["incremental"][index]
        fused_rows.append(
            {
                "id": f"fused-{source.get('id', f'frame-{index:06d}')}",
                "frame_index": index,
                "image": source.get("image", []),
                "progress": fused_progress,
                "hop": fused_hop,
                "component_progress": progress,
                "component_hop": hops,
                "fusion_rule": "arithmetic_mean_of_official_progress",
            }
        )
        previous = fused_progress

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(fused_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "output_path": str(output_path),
        "row_count": len(fused_rows),
        "frame_indices": reference_indices,
        "finite": True,
        "fusion_rule": "arithmetic_mean_of_official_progress",
        "source_prediction_paths": {mode: str(prediction_paths[mode]) for mode in PERSPECTIVE_MODES},
        "metadata": dict(metadata or {}),
    }


def write_progress_csv(
    output_path: Path,
    prediction_paths: Mapping[str, Path],
    fused_path: Path,
) -> None:
    """Write a compact native-grid table for reports and plotting."""
    _, fused_rows = _rows_by_frame(fused_path, "fused")
    mode_rows = {
        mode: _rows_by_frame(Path(prediction_paths[mode]), mode)[1]
        for mode in PERSPECTIVE_MODES
    }
    indices = sorted(fused_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame_index"] + [f"{mode}_progress" for mode in PERSPECTIVE_MODES] + ["fused_progress"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in indices:
            row = {"frame_index": index}
            for mode in PERSPECTIVE_MODES:
                row[f"{mode}_progress"] = mode_rows[mode][index]["progress"]
            row["fused_progress"] = fused_rows[index]["progress"]
            writer.writerow(row)


def plot_progress_curves(
    output_path: Path,
    prediction_paths: Mapping[str, Path],
    fused_path: Path,
    *,
    title: str,
) -> bool:
    """Save the requested four-curve sanity plot when matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    mode_rows = {
        mode: _rows_by_frame(Path(prediction_paths[mode]), mode)[1]
        for mode in PERSPECTIVE_MODES
    }
    _, fused_rows = _rows_by_frame(fused_path, "fused")
    indices = sorted(fused_rows)
    fig, axis = plt.subplots(figsize=(10, 4.8), dpi=140)
    colors = {
        "incremental": "#e69f00",
        "forward": "#0072b2",
        "backward": "#009e73",
        "fused": "#cc79a7",
    }
    for mode in PERSPECTIVE_MODES:
        axis.plot(indices, [mode_rows[mode][index]["progress"] for index in indices], label=mode, color=colors[mode])
    axis.plot(indices, [fused_rows[index]["progress"] for index in indices], label="fused", color=colors["fused"], linewidth=2.2)
    axis.set_xlabel("video frame index (native sampled grid)")
    axis.set_ylabel("progress")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return True


def summarize_curve(prediction_path: Path, label: str) -> dict[str, float | int | str]:
    """Return comparable descriptive smoothness statistics for one curve."""
    _, rows = _rows_by_frame(prediction_path, label)
    values = [_finite_number(row["progress"], label=f"{label} progress") for row in rows.values()]
    diffs = [right - left for left, right in zip(values, values[1:])]
    if not diffs:
        return {
            "label": label,
            "samples": len(values),
            "diff_std": 0.0,
            "mean_abs_diff": 0.0,
            "sign_change_rate": 0.0,
            "range": max(values) - min(values) if values else 0.0,
        }
    mean = sum(diffs) / len(diffs)
    variance = sum((value - mean) ** 2 for value in diffs) / len(diffs)
    sign_changes = sum(
        1
        for left, right in zip(diffs, diffs[1:])
        if left != 0.0 and right != 0.0 and (left > 0) != (right > 0)
    )
    return {
        "label": label,
        "samples": len(values),
        "diff_std": math.sqrt(variance),
        "mean_abs_diff": sum(abs(value) for value in diffs) / len(diffs),
        "sign_change_rate": sign_changes / max(1, len(diffs) - 1),
        "range": max(values) - min(values),
    }


def summarize_incremental_noise(prediction_path: Path) -> dict[str, float | int]:
    """Return simple descriptive noise statistics for an incremental curve."""
    _, rows = _rows_by_frame(prediction_path, "incremental")
    hops = [_finite_number(row["hop"], label="incremental hop") for row in rows.values()]
    if not hops:
        return {"samples": 0, "mean_abs_hop": 0.0, "hop_std": 0.0, "sign_change_rate": 0.0}
    mean = sum(hops) / len(hops)
    variance = sum((value - mean) ** 2 for value in hops) / len(hops)
    sign_changes = sum(
        1
        for left, right in zip(hops, hops[1:])
        if left != 0.0 and right != 0.0 and (left > 0) != (right > 0)
    )
    transitions = max(0, len(hops) - 1)
    return {
        "samples": len(hops),
        "mean_abs_hop": sum(abs(value) for value in hops) / len(hops),
        "hop_std": math.sqrt(variance),
        "sign_change_rate": sign_changes / transitions if transitions else 0.0,
    }

