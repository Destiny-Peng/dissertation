#!/usr/bin/env python3
"""CPU-only failure analysis from saved Robo-Dopamine fused hop.

Evaluate event-localized failures, terminal failures without event annotations,
and clean-success controls using existing LF3R annotations. Jointly sweep simple
pairwise OR detector ensembles on the fused signal only. No inference is
launched and no new annotation type is introduced.

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
    build_pairwise_ensemble_rows,
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
ANALYSIS_SIGNAL_MODE = "fused"


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
    signals_by_mode: Mapping[str, Mapping[str, Mapping[str, Any]]],
    provenance_by_mode: Mapping[str, Mapping[str, Any]],
    common_rollout_ids: set[str],
    best_rows: Sequence[Mapping[str, Any]],
) -> None:
    run_json = run_root / "run.json"
    jobs_jsonl = run_root / "jobs.jsonl"
    scale_examples = {
        mode: [
            signal["scale_detection"]
            for signal in signals_by_mode.get(mode, {}).values()
        ][:10]
        for mode in (ANALYSIS_SIGNAL_MODE,)
    }

    metadata = {
        "schema_version": 2,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script": project_relative(Path(__file__)),
        "git_revision": git_revision(),
        "analysis_mode": "CPU-only saved-output post-processing; no inference",
        "input": {
            "run_root": project_relative(run_root),
            "selection": (
                project_relative(resolve_project_path(args.selection))
                if args.selection
                else None
            ),
            "run_json": project_relative(run_json) if run_json.is_file() else None,
            "run_json_sha256": sha256(run_json) if run_json.is_file() else None,
            "jobs_jsonl": (
                project_relative(jobs_jsonl) if jobs_jsonl.is_file() else None
            ),
            "jobs_jsonl_sha256": (
                sha256(jobs_jsonl) if jobs_jsonl.is_file() else None
            ),
            "manifest": project_relative(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "annotation_dir": project_relative(annotation_dir),
        },
        "signal": {
            "name": "Robo-Dopamine fused hop failure detection",
            "modes": [ANALYSIS_SIGNAL_MODE],
            "native_grid_only": True,
            "interpolation": False,
            "same_rollout_intersection": False,
            "common_rollout_n": len(common_rollout_ids),
            "common_rollout_ids": sorted(common_rollout_ids),
            "normalization": {
                "fused": "Use saved fused-progress difference hop without rescaling.",
            },
            "semantics": {
                "fused": "difference of consecutive arithmetic-mean fused progress values",
            },
            "scale_detection_examples": scale_examples,
        },
        "parameter_grids": {
            "epsilon": list(EPSILONS),
            "consecutive_n": list(CONSECUTIVE_NS),
            "k_of_m_m": list(KOFM_MS),
            "k_rule": "ceil(0.6*m) ... m",
            "window_mean_m": list(MEAN_MS),
            "window_mean_theta": list(MEAN_THRESHOLDS),
            "cumulative_regression_m": list(REGRESSION_MS),
            "cumulative_regression_A": list(REGRESSION_THRESHOLDS),
            "stagnation_delta": list(STAGNATION_DELTAS),
            "clean_fpr_constraints": list(CLEAN_FPR_CONSTRAINTS),
        },
        "detector_semantics": {
            "consecutive": "h_t <= epsilon for n consecutive native samples",
            "k_of_m": "at least k of latest m native hops satisfy h_i <= epsilon",
            "window_mean": "mean of latest m native hops <= theta",
            "cumulative_regression": "sum(max(0,-h_i)) over latest m native hops >= A",
            "stagnation": (
                "abs(h_t) <= delta, aggregated with consecutive-n or k-of-m"
            ),
        },
        "evaluation_semantics": {
            "primary_reference": "human observable_onset_frame",
            "recall_profile_native_samples": [1, 3, 5, 10, 20],
            "eventual_recall": (
                "positive at least once from observable onset until the failure "
                "episode end; recovery_frame, terminal_failure_frame, or the next "
                "observable event onset is an exclusive boundary, otherwise the "
                "last available rollout sample is included"
            ),
            "sample_delay": (
                "1-based count from the first native sample at/after onset"
            ),
            "frame_delay": (
                "detector_frame - observable_onset_frame using actual saved frame indices"
            ),
            "early_positives": (
                "reported separately and never converted to zero-delay detections"
            ),
            "pre_onset_lookback_samples": 10,
            "parameter_selection": (
                "Single-detector representatives keep the existing event Recall@3 "
                "selection rule. Pairwise OR ensembles jointly sweep both detector "
                "parameter grids under total clean-rollout FPR caps."
            ),
            "no_event_failure": (
                "For terminal-failure rollouts with no failure-event annotation, "
                "do not synthesize an onset. Treat failure as present from rollout "
                "start and report first-alarm Recall@1/@3/@5/@10/@20/eventual plus "
                "start-to-alarm delay on the native fused-signal grid."
            ),
            "overall_failed_rollout_coverage": (
                "A terminal-failure rollout is covered when any annotated event is "
                "eventually detected, or when a no-event terminal-failure rollout "
                "has any alarm before rollout end."
            ),
            "recovery_hop_window_samples": args.recovery_window_samples,
            "pairwise_ensemble": (
                "Fused-only OR ensemble search for three prioritized family pairs. "
                "Each selection target is optimized separately under total clean "
                "FPR caps 5%, 10%, and 20%; inference outputs are reused."
            ),
        },
        "generalization": {
            "task_cv_enabled": bool(args.task_cv),
            "method": "optional leave-one-task-out tuning/evaluation on fused hop",
            "full_dataset_sweep_separate": True,
        },
        "counts_by_signal_mode": {
            mode: dict(provenance_by_mode.get(mode, {}))
            for mode in (ANALYSIS_SIGNAL_MODE,)
        },
        "detector_config_n_per_signal": len(configs),
        "detector_config_n_total": len(configs),
        "selected_config_n": selected_count(best_rows),
        "outputs": [
            "sweep_summary.csv",
            "event_results.csv",
            "no_event_failure_results.csv",
            "clean_rollout_results.csv",
            "best_configs.csv",
            "recovery_results.csv",
            "breakdown_summary.csv",
            "ensemble_sweep.csv",
            "ensemble_selected.csv",
            "ensemble_by_failure_type.csv",
            "metadata.json",
            "task_cv_results.csv (only with --task-cv)",
            "plots/<signal_mode>/ (unless --no-plots)",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

def analyse(
    args: argparse.Namespace,
) -> Path:
    run_root = ensure_within_project(
        resolve_project_path(args.run_root), "run root"
    )
    manifest_path = ensure_within_project(
        resolve_project_path(args.manifest), "manifest"
    )
    annotation_dir = ensure_within_project(
        resolve_project_path(args.annotations), "annotation directory"
    )

    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not annotation_dir.is_dir():
        raise FileNotFoundError(annotation_dir)

    if args.output_dir:
        output_dir = ensure_within_project(
            resolve_project_path(args.output_dir), "output directory"
        )
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / timestamp

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    selection_path = (
        ensure_within_project(
            resolve_project_path(args.selection), "selection"
        )
        if args.selection
        else None
    )
    if selection_path is not None and not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    allowed_rollout_ids = load_selection_ids(selection_path)

    signals, events, no_event_failures, clean_rollouts, provenance = (
        build_base_records(
            run_root,
            manifest,
            annotation_dir,
            allowed_rollout_ids=allowed_rollout_ids,
            signal_mode=ANALYSIS_SIGNAL_MODE,
        )
    )
    analysis_rollout_ids = set(signals)
    if not analysis_rollout_ids:
        raise ValueError(
            "No rollout has a usable saved Robo-Dopamine fused hop signal"
        )

    configs = build_detector_configs()
    summary_rows, event_rows, no_event_rows, clean_rows = evaluate_all_configs(
        configs,
        signals,
        events,
        no_event_failures,
        clean_rollouts,
    )
    best_rows = select_best_configs(summary_rows)
    breakdown_rows = summarize_breakdowns(
        configs,
        event_rows,
        clean_rows,
    )
    selected_configs = selected_unique_configs(best_rows, configs)
    recovery_rows = build_recovery_rows(
        selected_configs,
        signals,
        events,
        window_samples=args.recovery_window_samples,
    )

    def tag(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"signal_mode": ANALYSIS_SIGNAL_MODE, **dict(row)}
            for row in rows
        ]

    tagged_summary = tag(summary_rows)
    tagged_events = tag(event_rows)
    tagged_no_event = tag(no_event_rows)
    tagged_clean = tag(clean_rows)
    tagged_best = tag(best_rows)
    tagged_breakdown = tag(breakdown_rows)
    tagged_recovery = tag(recovery_rows)

    ensemble_sweep, ensemble_selected, ensemble_failure_types = (
        build_pairwise_ensemble_rows(
            configs,
            event_rows,
            no_event_rows,
            clean_rows,
        )
    )

    cv_rows: list[dict[str, Any]] = []
    if args.task_cv:
        cv_rows = tag(
            task_cross_validation(
                configs,
                event_rows,
                clean_rows,
            )
        )

    if not args.no_plots:
        plot_dir = output_dir / "plots" / ANALYSIS_SIGNAL_MODE
        plot_tradeoff(
            summary_rows,
            plot_dir / "recall_at_3_vs_clean_fpr.png",
        )
        plot_detector_heatmaps(summary_rows, plot_dir)
        plot_delay_distributions(
            best_rows,
            event_rows,
            plot_dir / "selected_detection_delay_boxplot.png",
        )
        representative_ids = choose_representative_rollouts(
            args.representative_rollout or [],
            events,
            signals,
            args.max_representative_rollouts,
        )
        plot_representative_rollouts(
            selected_plot_configs(best_rows, configs),
            representative_ids,
            signals,
            events,
            plot_dir,
            signal_mode=ANALYSIS_SIGNAL_MODE,
        )

    write_csv(output_dir / "sweep_summary.csv", tagged_summary)
    write_csv(output_dir / "event_results.csv", tagged_events)
    write_csv(
        output_dir / "no_event_failure_results.csv",
        tagged_no_event,
    )
    write_csv(output_dir / "clean_rollout_results.csv", tagged_clean)
    write_csv(output_dir / "best_configs.csv", tagged_best)
    write_csv(output_dir / "recovery_results.csv", tagged_recovery)
    write_csv(output_dir / "breakdown_summary.csv", tagged_breakdown)
    write_csv(output_dir / "ensemble_sweep.csv", ensemble_sweep)
    write_csv(output_dir / "ensemble_selected.csv", ensemble_selected)
    write_csv(
        output_dir / "ensemble_by_failure_type.csv",
        ensemble_failure_types,
    )
    if args.task_cv:
        write_csv(output_dir / "task_cv_results.csv", cv_rows)

    write_metadata(
        output_dir,
        args=args,
        run_root=run_root,
        manifest_path=manifest_path,
        annotation_dir=annotation_dir,
        configs=configs,
        signals_by_mode={ANALYSIS_SIGNAL_MODE: signals},
        provenance_by_mode={ANALYSIS_SIGNAL_MODE: provenance},
        common_rollout_ids=analysis_rollout_ids,
        best_rows=tagged_best,
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
            "Completed Robo-Dopamine run containing a saved fused hop output."
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
