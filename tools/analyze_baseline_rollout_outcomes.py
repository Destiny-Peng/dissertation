#!/usr/bin/env python3
"""Evaluate final rollout outcome from the newest saved baseline output per rollout."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from analyze_baseline_change_points import project_path  # noqa: E402
from analyze_baseline_temporal_signals import (  # noqa: E402
    METHODS,
    DEFAULT_ANNOTATIONS,
    DEFAULT_MANIFEST,
    ROLLOUT_OUTCOME_PRIMARY_SIGNALS,
    ROLLOUT_OUTCOME_SIGNAL_NOTES,
    ROLLOUT_OUTCOME_THRESHOLDS,
    compute_rollout_outcome_classification,
    json_safe,
    load_annotations,
    load_manifest,
    load_method_signal,
    load_selection,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_source_map(path: Path) -> dict[str, dict[str, Path]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    raw_maps = document.get("source_maps") if isinstance(document, dict) else None
    if not isinstance(raw_maps, dict):
        raise ValueError("source-map JSON must contain source_maps")
    result: dict[str, dict[str, Path]] = {}
    for method in METHODS:
        raw = raw_maps.get(method) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"source map for {method} must be an object")
        result[method] = {
            str(rollout_id): project_path(run_root)
            for rollout_id, run_root in raw.items()
            if isinstance(rollout_id, str) and run_root
        }
    return result


def _load_rollouts(
    selection_path: Path,
    manifest_path: Path,
    annotation_dir: Path,
    source_maps: dict[str, dict[str, Path]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    _selection_doc, selections = load_selection(selection_path)
    manifest = load_manifest(manifest_path)
    rollouts: dict[str, dict[str, Any]] = {}
    source_runs: dict[str, set[str]] = {method: set() for method in METHODS}

    for selection in selections:
        rollout_id = str(selection["id"])
        if rollout_id not in manifest:
            raise KeyError(f"Selection ID is not in manifest: {rollout_id}")
        annotation = load_annotations(annotation_dir, rollout_id)
        rollout = {
            "record": manifest[rollout_id],
            "selection": selection,
            "annotation": annotation,
            "events": [],
            "methods": {},
            "method_errors": {},
        }
        for method in METHODS:
            run_root = source_maps[method].get(rollout_id)
            if run_root is None:
                rollout["method_errors"][method] = "no saved output for this rollout"
                continue
            try:
                rollout["methods"][method] = load_method_signal(
                    method,
                    run_root,
                    rollout_id,
                    rollout["record"],
                )
                source_runs[method].add(str(run_root))
            except (
                FileNotFoundError,
                OSError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as error:
                rollout["method_errors"][method] = str(error)
        rollouts[rollout_id] = rollout

    return (
        rollouts,
        selections,
        {method: sorted(paths) for method, paths in source_runs.items()},
    )


def main() -> int:
    args = parse_args()
    selection_path = project_path(args.selection)
    source_map_path = project_path(args.source_map)
    manifest_path = project_path(args.manifest)
    annotation_dir = project_path(args.annotations_dir)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_maps = _load_source_map(source_map_path)
    rollouts, selections, source_runs = _load_rollouts(
        selection_path=selection_path,
        manifest_path=manifest_path,
        annotation_dir=annotation_dir,
        source_maps=source_maps,
    )

    summary, predictions = compute_rollout_outcome_classification(rollouts)
    summary.to_csv(output_dir / "rollout_outcome_summary.csv", index=False)
    predictions.to_csv(output_dir / "rollout_outcome_predictions.csv", index=False)

    coverage_rows = []
    for method in METHODS:
        available = [
            rollout_id
            for rollout_id, rollout in rollouts.items()
            if method in rollout["methods"]
        ]
        missing = [
            rollout_id
            for rollout_id, rollout in rollouts.items()
            if method not in rollout["methods"]
        ]
        coverage_rows.append({
            "method": method,
            "evaluation_population": len(rollouts),
            "available_rollouts": len(available),
            "missing_rollouts": len(missing),
            "coverage_fraction": (
                len(available) / len(rollouts) if rollouts else 0.0
            ),
            "source_run_count": len(source_runs[method]),
            "missing_rollout_ids": json.dumps(missing, ensure_ascii=False),
        })
    pd.DataFrame(coverage_rows).to_csv(
        output_dir / "method_coverage.csv",
        index=False,
    )

    metadata = {
        "schema_version": 2,
        "analysis": "lf3r_rollout_outcome_evaluation",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "source_map": str(source_map_path),
        "source_map_sha256": sha256(source_map_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "annotations_dir": str(annotation_dir),
        "rollouts": [row["id"] for row in selections],
        "methods": list(METHODS),
        "source_runs": source_runs,
        "source_resolution": (
            "newest parseable completed output per rollout; "
            "method coverage is the union across completed runs"
        ),
        "rollout_outcome_classification": {
            "positive_class": "clean_success+recovered_success",
            "negative_class": "terminal_failure",
            "uncertain_policy": "excluded_from_metrics",
            "primary_signals": ROLLOUT_OUTCOME_PRIMARY_SIGNALS,
            "signal_notes": ROLLOUT_OUTCOME_SIGNAL_NOTES,
            "thresholds": list(ROLLOUT_OUTCOME_THRESHOLDS),
            "default_threshold": "q95",
            "threshold_calibration": "final_success_terminal_score_quantile",
            "summary_rows": int(len(summary)),
            "prediction_rows": int(len(predictions)),
        },
        "counts": {
            "evaluation_population": len(rollouts),
            "summary_rows": int(len(summary)),
            "prediction_rows": int(len(predictions)),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(json_safe(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("ROLLOUT_OUTCOME_EVALUATION_OK")
    print(f"output_dir={output_dir}")
    print(f"evaluation_population={len(rollouts)}")
    for row in coverage_rows:
        print(
            f"{row['method']}: {row['available_rollouts']}/"
            f"{row['evaluation_population']} available from "
            f"{row['source_run_count']} run(s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
