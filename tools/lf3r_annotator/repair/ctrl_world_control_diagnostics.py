#!/usr/bin/env python3
"""Compare prepared LF3R Ctrl-World controls with DROID training bounds.

This is a CPU-only diagnostic. It reports, per Ctrl-World state dimension,
LIBERO raw ranges, DROID 1st/99th percentile bounds, normalized ranges before
clipping, and the percentage of values that would be clipped to -1 or +1 by
Ctrl-World's released normalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DIM_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def project_path(value: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controls",
        type=Path,
        default=None,
        help=(
            "Prepared *.ctrl_controls.npz. If omitted, use the first one found "
            "under datasets/libero_official_success/v1."
        ),
    )
    parser.add_argument(
        "--droid-stat",
        type=Path,
        default=Path("repos/Ctrl-World/dataset_meta_info/droid/stat.json"),
    )
    parser.add_argument(
        "--droid-annotation",
        type=Path,
        default=Path(
            "repos/Ctrl-World/dataset_example/droid_subset/annotation/val/199.json"
        ),
        help="Optional DROID example annotation used for a reference raw range.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the full diagnostic JSON payload.",
    )
    return parser.parse_args()


def default_controls_path() -> Path:
    root = PROJECT_ROOT / "datasets" / "libero_official_success" / "v1"
    matches = sorted(root.rglob("*.ctrl_controls.npz")) if root.is_dir() else []
    if not matches:
        raise FileNotFoundError(
            "No prepared *.ctrl_controls.npz found under "
            f"{root}; pass --controls explicitly."
        )
    return matches[0].resolve()


def load_controls(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        controls = np.asarray(payload["controls"], dtype=np.float64)
    if controls.ndim != 2 or controls.shape[1] != 7:
        raise ValueError(f"Expected controls [T,7], got {controls.shape}: {path}")
    if not np.isfinite(controls).all():
        raise ValueError(f"Controls contain non-finite values: {path}")
    return controls


def load_droid_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    p01 = np.asarray(payload["state_01"], dtype=np.float64)
    p99 = np.asarray(payload["state_99"], dtype=np.float64)
    if p01.shape != (7,) or p99.shape != (7,):
        raise ValueError(f"DROID stat bounds must be 7D, got {p01.shape}/{p99.shape}")
    return p01, p99


def load_droid_example(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    states = np.asarray(payload.get("states"), dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 7:
        return None
    return states


def summarize(
    controls: np.ndarray,
    p01: np.ndarray,
    p99: np.ndarray,
    droid_example: np.ndarray | None,
) -> dict[str, Any]:
    denom = p99 - p01 + 1e-8
    normalized = 2.0 * (controls - p01[None, :]) / denom[None, :] - 1.0
    below = normalized < -1.0
    above = normalized > 1.0

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(DIM_NAMES):
        row: dict[str, Any] = {
            "dimension": name,
            "droid_p01": float(p01[index]),
            "droid_p99": float(p99[index]),
            "libero_min": float(np.min(controls[:, index])),
            "libero_max": float(np.max(controls[:, index])),
            "libero_mean": float(np.mean(controls[:, index])),
            "normalized_preclip_min": float(np.min(normalized[:, index])),
            "normalized_preclip_max": float(np.max(normalized[:, index])),
            "clip_minus1_percent": float(np.mean(below[:, index]) * 100.0),
            "clip_plus1_percent": float(np.mean(above[:, index]) * 100.0),
            "outside_droid_1_99_percent": float(
                np.mean(below[:, index] | above[:, index]) * 100.0
            ),
        }
        if droid_example is not None:
            row["droid_example_min"] = float(np.min(droid_example[:, index]))
            row["droid_example_max"] = float(np.max(droid_example[:, index]))
        rows.append(row)

    any_outside = np.any(below | above, axis=1)
    return {
        "control_points": int(len(controls)),
        "rows": rows,
        "samples_with_any_clipped_dimension_percent": float(
            np.mean(any_outside) * 100.0
        ),
        "mean_outside_droid_1_99_percent_across_dimensions": float(
            np.mean(below | above) * 100.0
        ),
    }


def print_report(payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    print(
        "dim      DROID p01     DROID p99     LIBERO min    LIBERO max    "
        "norm min   norm max   clip -1%  clip +1%  outside%"
    )
    print("-" * 124)
    for row in rows:
        print(
            f"{row['dimension']:<8}"
            f"{row['droid_p01']:>12.5f}  "
            f"{row['droid_p99']:>12.5f}  "
            f"{row['libero_min']:>12.5f}  "
            f"{row['libero_max']:>12.5f}  "
            f"{row['normalized_preclip_min']:>9.3f}  "
            f"{row['normalized_preclip_max']:>9.3f}  "
            f"{row['clip_minus1_percent']:>8.1f}  "
            f"{row['clip_plus1_percent']:>8.1f}  "
            f"{row['outside_droid_1_99_percent']:>8.1f}"
        )
    print()
    print(
        "Samples with >=1 clipped dimension: "
        f"{payload['samples_with_any_clipped_dimension_percent']:.1f}%"
    )
    print(
        "Mean outside-DROID rate across all values: "
        f"{payload['mean_outside_droid_1_99_percent_across_dimensions']:.1f}%"
    )


def main() -> None:
    args = parse_args()
    controls_path = (
        project_path(args.controls) if args.controls is not None else default_controls_path()
    )
    stat_path = project_path(args.droid_stat)
    annotation_path = project_path(args.droid_annotation)

    controls = load_controls(controls_path)
    p01, p99 = load_droid_bounds(stat_path)
    droid_example = load_droid_example(annotation_path)
    payload = summarize(controls, p01, p99, droid_example)
    payload.update(
        {
            "controls_path": str(controls_path),
            "droid_stat_path": str(stat_path),
            "droid_annotation_path": (
                str(annotation_path) if droid_example is not None else None
            ),
        }
    )

    print("Controls:", controls_path)
    print("DROID stats:", stat_path)
    if droid_example is not None:
        print("DROID example:", annotation_path)
    print()
    print_report(payload)

    if args.json_output is not None:
        output = project_path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("Saved:", output)


if __name__ == "__main__":
    main()
