#!/usr/bin/env python3
"""CPU-only failure analysis from Robo-Dopamine incremental hop.

Use only already saved Robo-Dopamine incremental pred_vllm.json outputs and
existing LF3R human annotations. No inference is launched. No progress,
fusion, generic change-point, or whole-rollout Q95/std detector is used.

Example:
    python3 tools/analyze_robo_dopamine_incremental_hop.py \
        --run-root outputs/baselines/<completed-robo-run>

Add --task-cv for optional leave-one-task-out parameter tuning/evaluation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from robo_incremental_hop.core import (
    CLEAN_FPR_CONSTRAINTS,
    CONSECUTIVE_NS,
    EPSILONS,
    KOFM_MS,
    MEAN_MS,
    MEAN_THRESHOLDS,
    REGRESSION_MS,
    REGRESSION_THRESHOLDS,
    STAGNATION_DELTAS,
    build_detector_configs,
)
from robo_incremental_hop.io import (
    PROJECT_ROOT,
    build_base_records,
    ensure_within_project,
    load_manifest,
    project_relative,
    resolve_project_path,
)
from robo_incremental_hop.report import (
    build_recovery_rows,
    choose_representative_rollouts,
    evaluate_all_configs,
    plot_delay_distributions,
    plot_detector_heatmaps,
    plot_representative_rollouts,
    plot_tradeoff,
    select_best_configs,
    selected_plot_configs,
    selected_unique_configs,
    summarize_breakdowns,
    task_cross_validation,
    write_csv,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
)
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT
    / "annotations/failure_annotations/v1/records"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/robo_dopamine_incremental_hop"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None
    return result.stdout.strip() or None


def load_selection_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("selection") if isinstance(document, dict) else document
    if not isinstance(rows, list):
        raise ValueError("Selection must be an array or object with selection[]")
    result: set[str] = set()
    for row in rows:
        rollout_id = row.get("id") if isinstance(row, dict) else row
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError("Selection contains an invalid rollout id")
        if rollout_id in result:
            raise ValueError(f"Selection contains duplicate rollout id: {rollout_id}")
        result.add(rollout_id)
    if not result:
        raise ValueError("Selection is empty")
    return result


def selected_count(
    rows: Sequence[Mapping[str, Any]],
) -> int:
    return sum(
        row.get("selection_status") == "selected"
        for row in rows
    )


def write_metadata(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    run_root: Path,
    manifest_path: Path,
    annotation_dir: Path,
    configs: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    best_rows: Sequence[Mapping[str, Any]],
) -> None:
    run_json = run_root / "run.json"
    jobs_jsonl = run_root / "jobs.jsonl"
    scale_examples = [
        signal["scale_detection"]
        for signal in signals.values()
    ][:10]

    metadata = {
        "schema_version": 1,
        "generated_at": (
            dt.datetime.now(
                dt.timezone.utc
            ).isoformat()
        ),
        "script": project_relative(
            Path(__file__)
        ),
        "git_revision": git_revision(),
        "analysis_mode": (
            "CPU-only saved-output post-processing; "
            "no inference"
        ),
        "input": {
            "run_root": project_relative(
                run_root
            ),
            "selection": (
                project_relative(resolve_project_path(args.selection))
                if args.selection
                else None
            ),
            "run_json": (
                project_relative(run_json)
                if run_json.is_file()
                else None
            ),
            "run_json_sha256": (
                sha256(run_json)
                if run_json.is_file()
                else None
            ),
            "jobs_jsonl": (
                project_relative(jobs_jsonl)
                if jobs_jsonl.is_file()
                else None
            ),
            "jobs_jsonl_sha256": (
                sha256(jobs_jsonl)
                if jobs_jsonl.is_file()
                else None
            ),
            "manifest": project_relative(
                manifest_path
            ),
            "manifest_sha256": sha256(
                manifest_path
            ),
            "annotation_dir": (
                project_relative(
                    annotation_dir
                )
            ),
        },
        "signal": {
            "name": (
                "Robo-Dopamine incremental hop"
            ),
            "native_grid_only": True,
            "interpolation": False,
            "progress_used": False,
            "fusion_used": False,
            "hop_scale_observed": (
                provenance.get(
                    "hop_source_scales"
                )
            ),
            "normalization": (
                "Use saved hop directly when "
                "confirmed in [-1,1]; divide by "
                "100 only when saved values are "
                "confirmed percentage points."
            ),
            "scale_detection_examples": (
                scale_examples
            ),
            "frame_intervals": (
                provenance.get(
                    "frame_intervals"
                )
            ),
            "official_semantics": (
                "Current official GRMInference "
                "parses <score> percentage / 100 "
                "and sets incremental hop = "
                "raw_score."
            ),
        },
        "parameter_grids": {
            "epsilon": list(EPSILONS),
            "consecutive_n": list(
                CONSECUTIVE_NS
            ),
            "k_of_m_m": list(KOFM_MS),
            "k_rule": (
                "ceil(0.6*m) ... m"
            ),
            "window_mean_m": list(
                MEAN_MS
            ),
            "window_mean_theta": list(
                MEAN_THRESHOLDS
            ),
            "cumulative_regression_m": (
                list(REGRESSION_MS)
            ),
            "cumulative_regression_A": (
                list(
                    REGRESSION_THRESHOLDS
                )
            ),
            "stagnation_delta": list(
                STAGNATION_DELTAS
            ),
            "clean_fpr_constraints": list(
                CLEAN_FPR_CONSTRAINTS
            ),
        },
        "detector_semantics": {
            "consecutive": (
                "h_t <= epsilon for n "
                "consecutive native samples"
            ),
            "k_of_m": (
                "at least k of latest m "
                "native hops satisfy "
                "h_i <= epsilon"
            ),
            "window_mean": (
                "mean of latest m native "
                "hops <= theta"
            ),
            "cumulative_regression": (
                "sum(max(0,-h_i)) over "
                "latest m native hops >= A"
            ),
            "stagnation": (
                "abs(h_t) <= delta, "
                "aggregated with consecutive-n "
                "or k-of-m"
            ),
            "window_warmup": (
                "k-of-m, mean and cumulative "
                "regression require a full "
                "m-sample native window"
            ),
        },
        "evaluation_semantics": {
            "primary_reference": (
                "human observable_onset_frame"
            ),
            "sample_delay": (
                "1-based count from the first "
                "native sample at/after onset; "
                "that first sample is +1"
            ),
            "frame_delay": (
                "detector_frame - "
                "observable_onset_frame using "
                "actual saved frame indices"
            ),
            "early_positives": (
                "reported separately and never "
                "converted to zero-delay "
                "post-onset detections"
            ),
            "pre_onset_lookback_samples": 10,
            "clean_false_positive": (
                "any detector-positive native "
                "sample on a clean-success "
                "rollout"
            ),
            "positive_episode": (
                "maximal consecutive run of "
                "detector-positive native "
                "samples"
            ),
            "parameter_selection": (
                "within each detector family "
                "and clean-rollout FPR "
                "constraint: maximize event "
                "recall@3, then smaller median "
                "sample delay, then lower "
                "clean-rollout FPR"
            ),
            "recovery_state": (
                "latest native sample at/before "
                "recovery_frame; clearance is "
                "first detector-negative sample "
                "at/after recovery_frame"
            ),
            "recovery_hop_window_samples": (
                args.recovery_window_samples
            ),
        },
        "generalization": {
            "task_cv_enabled": bool(
                args.task_cv
            ),
            "method": (
                "optional leave-one-task-out "
                "tuning/evaluation"
            ),
            "full_dataset_sweep_separate": True,
            "established_project_split_used": (
                False
            ),
            "note": (
                "The discovered train/validation "
                "split is SAFE-specific rather "
                "than an established "
                "Robo-Dopamine analysis split."
            ),
        },
        "counts": dict(provenance),
        "detector_config_n": len(configs),
        "selected_config_n": selected_count(
            best_rows
        ),
        "outputs": [
            "sweep_summary.csv",
            "event_results.csv",
            "clean_rollout_results.csv",
            "best_configs.csv",
            "recovery_results.csv",
            "breakdown_summary.csv",
            "metadata.json",
            (
                "task_cv_results.csv "
                "(only with --task-cv)"
            ),
            (
                "plots/ "
                "(unless --no-plots)"
            ),
        ],
    }
    (
        output_dir / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def analyse(
    args: argparse.Namespace,
) -> Path:
    run_root = ensure_within_project(
        resolve_project_path(
            args.run_root
        ),
        "run root",
    )
    manifest_path = ensure_within_project(
        resolve_project_path(
            args.manifest
        ),
        "manifest",
    )
    annotation_dir = ensure_within_project(
        resolve_project_path(
            args.annotations
        ),
        "annotation directory",
    )

    if not run_root.is_dir():
        raise FileNotFoundError(
            f"Run root does not exist: {run_root}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            manifest_path
        )
    if not annotation_dir.is_dir():
        raise FileNotFoundError(
            annotation_dir
        )

    if args.output_dir:
        output_dir = (
            ensure_within_project(
                resolve_project_path(
                    args.output_dir
                ),
                "output directory",
            )
        )
    else:
        timestamp = (
            dt.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )
        output_dir = (
            DEFAULT_OUTPUT_ROOT
            / timestamp
        )

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise FileExistsError(
            "Output directory is not empty: "
            f"{output_dir}"
        )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest(
        manifest_path
    )
    selection_path = (
        ensure_within_project(resolve_project_path(args.selection), "selection")
        if args.selection
        else None
    )
    if selection_path is not None and not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    allowed_rollout_ids = load_selection_ids(selection_path)

    (
        signals,
        events,
        clean_rollouts,
        provenance,
    ) = build_base_records(
        run_root,
        manifest,
        annotation_dir,
        allowed_rollout_ids=allowed_rollout_ids,
    )

    configs = build_detector_configs()
    (
        summary_rows,
        event_rows,
        clean_rows,
    ) = evaluate_all_configs(
        configs,
        signals,
        events,
        clean_rollouts,
    )
    best_rows = select_best_configs(
        summary_rows
    )
    breakdown_rows = (
        summarize_breakdowns(
            configs,
            event_rows,
            clean_rows,
        )
    )

    selected_configs = (
        selected_unique_configs(
            best_rows, configs
        )
    )
    recovery_rows = (
        build_recovery_rows(
            selected_configs,
            signals,
            events,
            window_samples=(
                args.recovery_window_samples
            ),
        )
    )

    write_csv(
        output_dir
        / "sweep_summary.csv",
        summary_rows,
    )
    write_csv(
        output_dir
        / "event_results.csv",
        event_rows,
    )
    write_csv(
        output_dir
        / "clean_rollout_results.csv",
        clean_rows,
    )
    write_csv(
        output_dir
        / "best_configs.csv",
        best_rows,
    )
    write_csv(
        output_dir
        / "recovery_results.csv",
        recovery_rows,
    )
    write_csv(
        output_dir
        / "breakdown_summary.csv",
        breakdown_rows,
    )

    if args.task_cv:
        write_csv(
            output_dir
            / "task_cv_results.csv",
            task_cross_validation(
                configs,
                event_rows,
                clean_rows,
            ),
        )

    if not args.no_plots:
        plot_dir = output_dir / "plots"
        plot_tradeoff(
            summary_rows,
            plot_dir
            / "recall_at_3_vs_clean_fpr.png",
        )
        plot_detector_heatmaps(
            summary_rows,
            plot_dir,
        )
        plot_delay_distributions(
            best_rows,
            event_rows,
            plot_dir
            / (
                "selected_detection_"
                "delay_boxplot.png"
            ),
        )
        representative_ids = (
            choose_representative_rollouts(
                (
                    args.representative_rollout
                    or []
                ),
                events,
                signals,
                args.max_representative_rollouts,
            )
        )
        plot_representative_rollouts(
            selected_plot_configs(
                best_rows,
                configs,
            ),
            representative_ids,
            signals,
            events,
            plot_dir,
        )

    write_metadata(
        output_dir,
        args=args,
        run_root=run_root,
        manifest_path=manifest_path,
        annotation_dir=annotation_dir,
        configs=configs,
        signals=signals,
        provenance=provenance,
        best_rows=best_rows,
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--run-root",
        required=True,
        help=(
            "Completed Robo-Dopamine run "
            "containing saved incremental "
            "pred_vllm.json outputs."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )
    parser.add_argument(
        "--selection",
        default=None,
        help=(
            "Optional project-local JSON/JSONL-style selection document "
            "containing rollout ids. Only completed run outputs whose ids "
            "appear in selection are analyzed."
        ),
    )
    parser.add_argument(
        "--annotations",
        default=str(
            DEFAULT_ANNOTATIONS
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "New project-local output "
            "directory. Default: "
            "outputs/robo_dopamine_"
            "incremental_hop/<timestamp>."
        ),
    )
    parser.add_argument(
        "--task-cv",
        action="store_true",
        help=(
            "Also run optional "
            "leave-one-task-out parameter "
            "tuning/evaluation."
        ),
    )
    parser.add_argument(
        "--recovery-window-samples",
        type=int,
        default=3,
        help=(
            "Native hop samples summarized "
            "immediately before/after recovery."
        ),
    )
    parser.add_argument(
        "--representative-rollout",
        action="append",
        default=None,
        help=(
            "Rollout ID to plot; repeatable. "
            "If omitted, deterministic event "
            "examples are selected."
        ),
    )
    parser.add_argument(
        "--max-representative-rollouts",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help=(
            "Write CSV/JSON only; skip "
            "matplotlib outputs."
        ),
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.recovery_window_samples < 1:
        parser.error(
            "--recovery-window-samples "
            "must be positive"
        )
    if (
        args.max_representative_rollouts
        < 1
    ):
        parser.error(
            "--max-representative-rollouts "
            "must be positive"
        )

    try:
        output = analyse(args)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
