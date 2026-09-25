#!/usr/bin/env python3
"""Evaluate final rollout outcome from saved LF3R baseline terminal signals."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from analyze_baseline_change_points import load_rollout_data, project_path  # noqa: E402
from analyze_baseline_temporal_signals import (  # noqa: E402
    METHODS,
    DEFAULT_ANNOTATIONS,
    DEFAULT_MANIFEST,
    PROJECT_ROOT,
    ROLLOUT_OUTCOME_PRIMARY_SIGNALS,
    ROLLOUT_OUTCOME_SIGNAL_NOTES,
    ROLLOUT_OUTCOME_THRESHOLDS,
    compute_rollout_outcome_classification,
    json_safe,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    for method in METHODS:
        flag = f"--{method.replace('_', '-')}-run"
        if method == "rynnvalue":
            parser.add_argument(flag, dest=f"{method}_run", type=Path, action="append", required=True)
        else:
            parser.add_argument(flag, dest=f"{method}_run", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection_path = project_path(args.selection)
    manifest_path = project_path(args.manifest)
    annotation_dir = project_path(args.annotations_dir)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_roots = {}
    for method in METHODS:
        value = getattr(args, f"{method}_run")
        source_roots[method] = [
            project_path(path)
            for path in (value if isinstance(value, list) else [value])
        ]

    rollouts, source_metadata, selections = load_rollout_data(
        selection_path=selection_path,
        manifest_path=manifest_path,
        annotation_dir=annotation_dir,
        source_roots=source_roots,
    )
    summary, predictions = compute_rollout_outcome_classification(rollouts)
    summary.to_csv(output_dir / "rollout_outcome_summary.csv", index=False)
    predictions.to_csv(output_dir / "rollout_outcome_predictions.csv", index=False)

    coverage_rows = []
    for method in METHODS:
        available = [rid for rid, rollout in rollouts.items() if method in rollout["methods"]]
        missing = [rid for rid, rollout in rollouts.items() if method not in rollout["methods"]]
        coverage_rows.append({
            "method": method,
            "selected_rollouts": len(rollouts),
            "available_rollouts": len(available),
            "missing_rollouts": len(missing),
            "missing_rollout_ids": json.dumps(missing, ensure_ascii=False),
        })
    pd.DataFrame(coverage_rows).to_csv(output_dir / "method_coverage.csv", index=False)

    metadata = {
        "schema_version": 1,
        "analysis": "lf3r_rollout_outcome_evaluation",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "annotations_dir": str(annotation_dir),
        "rollouts": [row["id"] for row in selections],
        "methods": list(METHODS),
        "run_roots": {
            method: [str(path) for path in roots] if len(roots) > 1 else str(roots[0])
            for method, roots in source_roots.items()
        },
        "source_runs": {
            method: source_metadata[method]["source_runs"]
            for method in METHODS
        },
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
            "rollouts": len(rollouts),
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
    print(f"rollouts={len(rollouts)} summary_rows={len(summary)} predictions={len(predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
