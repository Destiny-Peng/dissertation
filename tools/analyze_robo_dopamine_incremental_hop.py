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
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_CPU_LIMIT = 4
DEFAULT_NICE_TARGET = 10
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _bootstrap_thread_limit(argv: Sequence[str]) -> None:
    """Apply thread-pool caps before project imports can load native libraries."""
    cpu_limit = DEFAULT_CPU_LIMIT
    for index, token in enumerate(argv):
        if token == "--cpu-limit" and index + 1 < len(argv):
            try:
                cpu_limit = max(1, int(argv[index + 1]))
            except ValueError:
                break
        elif token.startswith("--cpu-limit="):
            try:
                cpu_limit = max(1, int(token.split("=", 1)[1]))
            except ValueError:
                break
    for name in THREAD_ENV_VARS:
        current = os.environ.get(name)
        try:
            current_value = int(current) if current is not None else None
        except ValueError:
            current_value = None
        os.environ[name] = str(
            min(cpu_limit, current_value)
            if current_value is not None and current_value > 0
            else cpu_limit
        )


_bootstrap_thread_limit(sys.argv[1:])

from robo_incremental_hop.core import CLEAN_FPR_CONSTRAINTS
from robo_incremental_hop.diagnosis import (
    build_grasp_event_features,
    build_matched_clean_pairs,
    category_rows,
    category_summary,
    choose_reference_ensemble,
    diagnosis_metadata,
    plot_grasp_event_heatmap,
    summarize_detected_vs_missed,
    summarize_matched_controls,
)
from robo_incremental_hop.phenotypes import build_phenotype_detector_configs
from robo_incremental_hop.oracle import build_oracle_analysis
from robo_incremental_hop.search_cache import (
    SEARCH_SEMANTICS_VERSION,
    load_legacy_search_seed,
    load_search_cache,
    search_fingerprint,
    write_search_cache,
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
    select_pairwise_ensemble_rows,
    build_recovery_rows,
    choose_representative_rollouts,
    evaluate_all_configs,
    plot_delay_distributions,
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


def apply_cpu_limit(cpu_limit: int) -> dict[str, Any]:
    """Bound analysis CPU concurrency without requiring elevated privileges."""
    if cpu_limit < 1:
        raise ValueError("cpu_limit must be positive")

    for name in THREAD_ENV_VARS:
        current = os.environ.get(name)
        try:
            current_value = int(current) if current is not None else None
        except ValueError:
            current_value = None
        os.environ[name] = str(
            min(cpu_limit, current_value)
            if current_value is not None and current_value > 0
            else cpu_limit
        )

    available_cpus: list[int] = []
    selected_cpus: list[int] = []
    affinity_applied = False
    affinity_error = None
    if hasattr(os, "sched_getaffinity"):
        try:
            available_cpus = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        except OSError as exc:
            affinity_error = str(exc)
    if not available_cpus:
        available_cpus = list(range(max(1, int(os.cpu_count() or 1))))

    selected_cpus = available_cpus[: min(cpu_limit, len(available_cpus))]
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(selected_cpus))
            affinity_applied = True
        except (OSError, PermissionError) as exc:
            affinity_error = str(exc)

    nice_before = None
    nice_after = None
    nice_error = None
    if hasattr(os, "getpriority") and hasattr(os, "setpriority"):
        try:
            nice_before = int(os.getpriority(os.PRIO_PROCESS, 0))
            target = max(nice_before, DEFAULT_NICE_TARGET)
            os.setpriority(os.PRIO_PROCESS, 0, target)
            nice_after = int(os.getpriority(os.PRIO_PROCESS, 0))
        except (OSError, PermissionError) as exc:
            nice_error = str(exc)

    return {
        "requested_logical_cpus": cpu_limit,
        "available_logical_cpus": len(available_cpus),
        "selected_logical_cpus": selected_cpus,
        "applied_logical_cpu_count": len(selected_cpus),
        "affinity_applied": affinity_applied,
        "affinity_error": affinity_error,
        "thread_env": {
            name: os.environ.get(name)
            for name in THREAD_ENV_VARS
        },
        "nice_target": DEFAULT_NICE_TARGET,
        "nice_before": nice_before,
        "nice_after": nice_after,
        "nice_error": nice_error,
    }


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
    phenotype_grid: Mapping[str, Any],
    oracle_summary: Sequence[Mapping[str, Any]],
    search_cache_info: Mapping[str, Any],
    grasp_diagnosis: Mapping[str, Any] | None = None,
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
        "resource_limits": dict(
            getattr(args, "resource_limits", {})
        ),
        "search_cache": dict(search_cache_info),
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
        "phenotype_detector": {
            **dict(phenotype_grid),
            "clean_fpr_constraints": list(CLEAN_FPR_CONSTRAINTS),
        },
        "oracle_analysis": {
            "enabled": True,
            "clean_fpr_constraint": None,
            "early_tolerance_native_samples": 1,
            "localization_rule": (
                "a positive episode counts only if its start is at/after the "
                "observable-onset anchor, or at most one native sample early"
            ),
            "delay_semantics": (
                "best_start_offset_samples is 0-based relative to the onset "
                "anchor; best_delay_samples preserves existing Recall@d semantics "
                "with onset-anchor sample=1 and one-sample-early alarm=0"
            ),
            "summary_rows": [dict(row) for row in oracle_summary],
        },
        "detector_semantics": {
            "stagnation_consecutive": (
                "abs(h_t) <= delta for n consecutive native samples"
            ),
            "stagnation_k_of_m": (
                "at least k of latest m native hops satisfy abs(h_i) <= delta"
            ),
            "regression_window_min": (
                "min(h[t-m+1:t]) <= empirical theta_r; theta_r values come "
                "from observed fused-hop failure evidence, not a hand-written grid"
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
                "The detector has two phenotypes: stagnation and short/strong "
                "regression. OR ensembles jointly sweep stagnation parameters and "
                "empirical regression-window-min parameters under total clean FPR "
                "caps. Primary selection targets are grasp-failure eventual recall "
                "and grasp-failure Recall@10; overall failure coverage is reported "
                "as a secondary outcome."
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
                "Fused-only stagnation OR regression search. Stagnation is swept "
                "as consecutive or k-of-m near-zero evidence; regression uses "
                "rolling-window minimum <= empirical theta_r. Grasp eventual and "
                "grasp Recall@10 are optimized separately under total clean FPR "
                "caps 5%, 10%, and 20%; inference outputs are reused."
            ),
        },
        "generalization": {
            "task_cv_enabled": bool(args.task_cv),
            "method": "optional leave-one-task-out tuning/evaluation on fused hop",
            "full_dataset_sweep_separate": True,
        },
        "grasp_failure_diagnosis": dict(grasp_diagnosis or {}),
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
            "oracle_global_best.csv",
            "oracle_event_detectability.csv",
            "oracle_summary.csv",
            "grasp_event_features.csv",
            "grasp_detected_vs_missed.csv",
            "grasp_matched_control.csv",
            "grasp_matched_control_summary.csv",
            "grasp_failure_categories.csv",
            "grasp_failure_category_summary.csv",
            "grasp_event_heatmap.png (unless --no-plots)",
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

    fingerprint = search_fingerprint(
        signals,
        events,
        no_event_failures,
        clean_rollouts,
    )
    cached_search = (
        None
        if args.refresh_search_cache
        else load_search_cache(fingerprint)
    )
    search_cache_hit = cached_search is not None
    legacy_seed_dir: Path | None = None
    search_reuse_source = "cache" if search_cache_hit else "computed"

    if cached_search is not None:
        configs = list(cached_search["configs"])
        oracle_configs = list(cached_search["oracle_configs"])
        phenotype_grid = dict(cached_search["phenotype_grid"])
        summary_rows = list(cached_search["summary_rows"])
        event_rows = list(cached_search["event_rows"])
        no_event_rows = list(cached_search["no_event_rows"])
        clean_rows = list(cached_search["clean_rows"])
        ensemble_sweep = list(cached_search["ensemble_sweep"])
        oracle_global_best = list(cached_search["oracle_global_best"])
        oracle_event_detectability = list(
            cached_search["oracle_event_detectability"]
        )
        oracle_summary = list(cached_search["oracle_summary"])
        print(
            "Search cache hit: "
            f"{fingerprint[:12]} · reusing detector/oracle/pair-sweep results"
        )
    else:
        configs, oracle_configs, phenotype_grid = build_phenotype_detector_configs(
            signals,
            events,
            no_event_failures,
            clean_rollouts,
        )

        legacy_seed = None
        if not args.refresh_search_cache:
            legacy_seed, legacy_seed_dir = load_legacy_search_seed(
                DEFAULT_OUTPUT_ROOT,
                current_output_dir=output_dir,
                run_root_relative=project_relative(run_root),
                manifest_sha256=sha256(manifest_path),
                annotation_dir=annotation_dir,
                signals=signals,
                events=events,
                no_event_failures=no_event_failures,
                clean_rollouts=clean_rollouts,
                configs=configs,
                oracle_configs=oracle_configs,
                phenotype_grid=phenotype_grid,
            )

        if legacy_seed is not None:
            search_reuse_source = "legacy_analysis"
            summary_rows = list(legacy_seed["summary_rows"])
            event_rows = list(legacy_seed["event_rows"])
            no_event_rows = list(legacy_seed["no_event_rows"])
            clean_rows = list(legacy_seed["clean_rows"])
            ensemble_sweep = list(legacy_seed["ensemble_sweep"])
            oracle_global_best = list(legacy_seed["oracle_global_best"])
            oracle_event_detectability = list(
                legacy_seed["oracle_event_detectability"]
            )
            oracle_summary = list(legacy_seed["oracle_summary"])
            print(
                "Compatible prior analysis found: "
                f"{project_relative(legacy_seed_dir)} · importing search results"
            )
        else:
            print(
                "Search cache miss: "
                f"{fingerprint[:12]} · no compatible prior analysis; "
                "running detector/oracle/pair search once"
            )
            summary_rows, event_rows, no_event_rows, clean_rows = evaluate_all_configs(
                configs,
                signals,
                events,
                no_event_failures,
                clean_rollouts,
            )
            oracle_global_best, oracle_event_detectability, oracle_summary = (
                build_oracle_analysis(
                    oracle_configs,
                    signals,
                    events,
                    no_event_failures,
                    clean_rollouts,
                    early_tolerance_samples=1,
                )
            )
            ensemble_sweep, _unused_selected, _unused_failure_types = (
                build_pairwise_ensemble_rows(
                    configs,
                    event_rows,
                    no_event_rows,
                    clean_rows,
                )
            )

        cache_file = write_search_cache(
            fingerprint,
            configs=configs,
            oracle_configs=oracle_configs,
            phenotype_grid=phenotype_grid,
            summary_rows=summary_rows,
            event_rows=event_rows,
            no_event_rows=no_event_rows,
            clean_rows=clean_rows,
            ensemble_sweep=ensemble_sweep,
            oracle_global_best=oracle_global_best,
            oracle_event_detectability=oracle_event_detectability,
            oracle_summary=oracle_summary,
        )
        print(f"Search cache written: {project_relative(cache_file)}")

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

    ensemble_selected, ensemble_failure_types = (
        select_pairwise_ensemble_rows(
            ensemble_sweep,
            event_rows,
        )
    )
    reference_ensemble = choose_reference_ensemble(
        ensemble_selected
    )
    grasp_features = build_grasp_event_features(
        signals,
        events,
        configs,
        reference_ensemble,
    )
    grasp_detected_vs_missed = summarize_detected_vs_missed(
        grasp_features
    )
    grasp_matched_pairs = build_matched_clean_pairs(
        grasp_features,
        signals,
        clean_rollouts,
    )
    grasp_matched_summary = summarize_matched_controls(
        grasp_matched_pairs
    )
    grasp_categories = category_rows(grasp_features)
    grasp_category_summary = category_summary(grasp_features)
    grasp_diagnosis = diagnosis_metadata(
        grasp_features,
        grasp_matched_pairs,
        reference_ensemble,
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
            plot_dir / "grasp_eventual_vs_clean_fpr.png",
        )
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
        plot_grasp_event_heatmap(
            output_dir / "grasp_event_heatmap.png",
            grasp_features,
            signals,
            events,
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
    write_csv(
        output_dir / "oracle_global_best.csv",
        oracle_global_best,
    )
    write_csv(
        output_dir / "oracle_event_detectability.csv",
        oracle_event_detectability,
    )
    write_csv(
        output_dir / "oracle_summary.csv",
        oracle_summary,
    )
    write_csv(
        output_dir / "grasp_event_features.csv",
        grasp_features,
    )
    write_csv(
        output_dir / "grasp_detected_vs_missed.csv",
        grasp_detected_vs_missed,
    )
    write_csv(
        output_dir / "grasp_matched_control.csv",
        grasp_matched_pairs,
    )
    write_csv(
        output_dir / "grasp_matched_control_summary.csv",
        grasp_matched_summary,
    )
    write_csv(
        output_dir / "grasp_failure_categories.csv",
        grasp_categories,
    )
    write_csv(
        output_dir / "grasp_failure_category_summary.csv",
        grasp_category_summary,
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
        phenotype_grid=phenotype_grid,
        oracle_summary=oracle_summary,
        search_cache_info={
            "enabled": True,
            "hit": search_cache_hit,
            "reuse_source": search_reuse_source,
            "legacy_seed_dir": (
                project_relative(legacy_seed_dir)
                if legacy_seed_dir is not None
                else None
            ),
            "fingerprint": fingerprint,
            "search_semantics_version": SEARCH_SEMANTICS_VERSION,
            "refresh_requested": bool(args.refresh_search_cache),
        },
        grasp_diagnosis=grasp_diagnosis,
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
        "--cpu-limit",
        type=int,
        default=DEFAULT_CPU_LIMIT,
        help=(
            "Maximum logical CPUs available to this analysis process. "
            f"Default: {DEFAULT_CPU_LIMIT}. Also caps common BLAS/OpenMP thread pools."
        ),
    )
    parser.add_argument(
        "--refresh-search-cache",
        action="store_true",
        help=(
            "Force detector/oracle search to run again even if a compatible "
            "content-addressed search cache already exists."
        ),
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
    if args.cpu_limit < 1:
        parser.error("--cpu-limit must be positive")
    args.resource_limits = apply_cpu_limit(args.cpu_limit)
    print(
        "CPU resource limit: "
        f"requested={args.cpu_limit}, "
        f"applied={args.resource_limits['applied_logical_cpu_count']}, "
        f"affinity={args.resource_limits['affinity_applied']}, "
        f"nice={args.resource_limits['nice_after']}"
    )

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
