#!/usr/bin/env python3
"""Evaluate final rollout outcome from the newest saved baseline output per rollout."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from analyze_baseline_change_points import project_path  # noqa: E402
from analyze_baseline_temporal_signals import (  # noqa: E402
    ROLLOUT_OUTCOME_METHODS,
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
    for method in ROLLOUT_OUTCOME_METHODS:
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
    source_runs: dict[str, set[str]] = {method: set() for method in ROLLOUT_OUTCOME_METHODS}

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
        for method in ROLLOUT_OUTCOME_METHODS:
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


PROGRESS_SWEEP_METHODS = ("procvlm", "robo_dopamine")
PROGRESS_SWEEP_AGGREGATIONS = ("final", "maximum")
PROGRESS_SWEEP_THRESHOLDS = tuple(round(value / 100.0, 2) for value in range(50, 100, 5))


def _progress_score(
    rollouts: dict[str, dict[str, Any]],
    row: pd.Series,
    aggregation: str,
) -> float | None:
    if aggregation == "final":
        try:
            value = float(row["terminal_value"])
        except (KeyError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
    if aggregation != "maximum":
        raise ValueError(f"Unknown progress aggregation: {aggregation}")

    rollout = rollouts.get(str(row.get("rollout_id") or ""))
    if not rollout:
        return None
    method = str(row.get("method") or "")
    signal = str(row.get("signal") or "progress")
    method_data = (rollout.get("methods") or {}).get(method) or {}
    series = (method_data.get("signals") or {}).get(signal)
    if not isinstance(series, dict):
        return None
    raw_values = series.get("values")
    if raw_values is None:
        return None
    finite: list[float] = []
    for raw in raw_values:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite.append(value)
    return max(finite) if finite else None


def _progress_threshold_sweep(
    predictions: pd.DataFrame,
    rollouts: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(rows)
    base = predictions.drop_duplicates(subset=["rollout_id", "method"])
    for method in PROGRESS_SWEEP_METHODS:
        method_rows = base[
            (base["method"] == method)
            & (base["included_in_metrics"] == True)
        ]
        if method_rows.empty:
            continue
        signal = str(method_rows.iloc[0]["signal"])
        for aggregation in PROGRESS_SWEEP_AGGREGATIONS:
            scored_rows: list[tuple[pd.Series, float]] = []
            for _, row in method_rows.iterrows():
                value = _progress_score(rollouts, row, aggregation)
                if value is not None:
                    scored_rows.append((row, value))
            if not scored_rows:
                continue

            for threshold in PROGRESS_SWEEP_THRESHOLDS:
                tp = fn = fp = tn = 0
                for row, value in scored_rows:
                    predicted_success = value > threshold
                    truth = int(row["true_success"])
                    if truth == 1 and predicted_success:
                        tp += 1
                    elif truth == 1:
                        fn += 1
                    elif predicted_success:
                        fp += 1
                    else:
                        tn += 1
                success_recall = tp / (tp + fn) if tp + fn else float("nan")
                failure_recall = tn / (tn + fp) if tn + fp else float("nan")
                precision = tp / (tp + fp) if tp + fp else float("nan")
                accuracy = (
                    (tp + tn) / (tp + tn + fp + fn)
                    if tp + tn + fp + fn
                    else float("nan")
                )
                f1 = (
                    2.0 * precision * success_recall / (precision + success_recall)
                    if pd.notna(precision)
                    and pd.notna(success_recall)
                    and precision + success_recall > 0
                    else float("nan")
                )
                rows.append({
                    "method": method,
                    "signal": signal,
                    "score_aggregation": aggregation,
                    "score_aggregation_label": (
                        "Final progress"
                        if aggregation == "final"
                        else "Maximum progress"
                    ),
                    "raw_progress_threshold": threshold,
                    # Keep the old column so older consumers remain compatible.
                    "raw_terminal_threshold": threshold,
                    "n_resolved": int(len(scored_rows)),
                    "tp": tp,
                    "fn": fn,
                    "fp": fp,
                    "tn": tn,
                    "accuracy": accuracy,
                    "success_recall": success_recall,
                    "failure_recall": failure_recall,
                    "precision": precision,
                    "f1": f1,
                })
    return pd.DataFrame(rows)


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
    threshold_sweep = _progress_threshold_sweep(predictions, rollouts)
    threshold_sweep.to_csv(
        output_dir / "rollout_outcome_threshold_sweep.csv",
        index=False,
    )

    coverage_rows = []
    for method in ROLLOUT_OUTCOME_METHODS:
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
        "methods": list(ROLLOUT_OUTCOME_METHODS),
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
            "progress_threshold_sweep": {
                "methods": list(PROGRESS_SWEEP_METHODS),
                "aggregations": list(PROGRESS_SWEEP_AGGREGATIONS),
                "thresholds": list(PROGRESS_SWEEP_THRESHOLDS),
                "rows": int(len(threshold_sweep)),
            },
        },
        "counts": {
            "evaluation_population": len(rollouts),
            "summary_rows": int(len(summary)),
            "prediction_rows": int(len(predictions)),
            "threshold_sweep_rows": int(len(threshold_sweep)),
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
