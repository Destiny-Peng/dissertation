"""FPR-unconstrained global-config localization on saved fused-hop signals.

This module is intentionally distinct from event-wise oracle analysis:
every candidate configuration is fixed globally across the evaluated population.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import build_detector_configs, detector_mask, make_config
from .search_cache import SEARCH_CACHE_ROOT

GLOBAL_LOCALIZATION_SCHEMA = 1
GLOBAL_LOCALIZATION_SEMANTICS_VERSION = (
    "fixed-config-capacity-v1-earliest-positive-episode-localization-v1"
)
LOCALIZATION_WINDOWS = (1, 3, 5, 10)
CACHE_ROOT = SEARCH_CACHE_ROOT / "global_localization"


def _config_key(config: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(config["detector_family"]),
        str(config.get("parameters_json") or ""),
    )


def _copy_config(
    config: Mapping[str, Any],
    *,
    config_id: str,
    config_source: str,
) -> dict[str, Any]:
    result = dict(config)
    result["config_id"] = config_id
    result["config_source"] = config_source
    return result


def build_global_config_space(
    empirical_phenotype_configs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Union the historical full grid with current empirical phenotype states.

    Returns single-detector configs, OR-pair configs, and compact metadata.
    Duplicate single configs are removed by family+parameters, preferring the
    historical fixed-grid ID only for provenance stability.
    """
    singles: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for index, config in enumerate(build_detector_configs(), 1):
        key = _config_key(config)
        if key in seen:
            continue
        seen.add(key)
        singles.append(
            _copy_config(
                config,
                config_id=f"global_fixed_{index:05d}",
                config_source="historical_full_grid",
            )
        )

    empirical_added = 0
    for config in empirical_phenotype_configs:
        key = _config_key(config)
        if key in seen:
            continue
        seen.add(key)
        empirical_added += 1
        singles.append(
            _copy_config(
                config,
                config_id=f"global_emp_{empirical_added:05d}",
                config_source="empirical_phenotype_grid",
            )
        )

    stagnation = [
        config
        for config in singles
        if str(config["detector_family"]).startswith("stagnation_")
    ]
    regression = [
        config
        for config in singles
        if str(config["detector_family"]) == "regression_window_min"
    ]

    pairs: list[dict[str, Any]] = []
    pair_index = 0
    for stagnation_config in stagnation:
        for regression_config in regression:
            pair_index += 1
            pair_id = f"global_or_{pair_index:07d}"
            parameters = {
                "logic": "OR",
                "stagnation_config_id": stagnation_config["config_id"],
                "stagnation_family": stagnation_config["detector_family"],
                "stagnation_parameters": json.loads(
                    str(stagnation_config["parameters_json"])
                ),
                "regression_config_id": regression_config["config_id"],
                "regression_family": regression_config["detector_family"],
                "regression_parameters": json.loads(
                    str(regression_config["parameters_json"])
                ),
            }
            pairs.append(
                {
                    "config_id": pair_id,
                    "detector_family": "phenotype_or",
                    "config_source": "complete_stagnation_regression_or",
                    "parameters_json": json.dumps(
                        parameters,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "stagnation_config_id": stagnation_config["config_id"],
                    "regression_config_id": regression_config["config_id"],
                }
            )

    family_counts: dict[str, int] = defaultdict(int)
    for config in singles:
        family_counts[str(config["detector_family"])] += 1

    return singles, pairs, {
        "single_config_n": len(singles),
        "or_config_n": len(pairs),
        "total_config_n": len(singles) + len(pairs),
        "historical_fixed_grid_n": len(build_detector_configs()),
        "empirical_unique_added_n": empirical_added,
        "single_family_counts": dict(sorted(family_counts.items())),
        "or_definition": (
            "every stagnation_consecutive/stagnation_k_of_m single config OR "
            "every regression_window_min config from the existing two-phenotype "
            "branch; legacy detector families remain standalone candidates"
        ),
    }


def _episode_starts(mask: Sequence[bool]) -> list[int]:
    return [
        index
        for index, value in enumerate(mask)
        if bool(value) and (index == 0 or not bool(mask[index - 1]))
    ]


def _event_identity(event: Mapping[str, Any]) -> str:
    value = event.get("event_id")
    if value not in (None, ""):
        return str(value)
    return (
        f"{event.get('rollout_id')}::event"
        f"{int(event.get('event_index') or 0)}"
    )


def _population_specs(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    all_events = list(events)
    terminal_by_rollout: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        if str(event.get("outcome") or "") == "terminal_failure":
            terminal_by_rollout[str(event["rollout_id"])].append(event)
    first_terminal = [
        min(
            rollout_events,
            key=lambda row: (
                int(row["observable_onset_frame"]),
                int(row.get("event_index") or 0),
            ),
        )
        for rollout_events in terminal_by_rollout.values()
        if rollout_events
    ]
    grasp = [
        event
        for event in events
        if str(event.get("failure_type") or "") == "grasp_failure"
    ]
    return [
        {
            "population": "all_annotated_failure_events",
            "events": all_events,
        },
        {
            "population": "first_event_per_failed_rollout",
            "events": first_terminal,
        },
        {
            "population": "grasp_failure",
            "events": grasp,
        },
    ]


def _event_anchor(
    frames: Sequence[int],
    onset_frame: int,
) -> int | None:
    return next(
        (
            index
            for index, frame in enumerate(frames)
            if int(frame) >= int(onset_frame)
        ),
        None,
    )


def _capacity_metrics(
    starts: Sequence[int],
    anchor: int,
    end_index_exclusive: int,
) -> dict[str, Any]:
    eligible = [
        index
        for index in starts
        if index < end_index_exclusive
    ]
    result: dict[str, Any] = {}
    for window in LOCALIZATION_WINDOWS:
        result[f"capacity_within_{window}"] = any(
            abs(index - anchor) <= window
            for index in eligible
        )
    # Keep eventual compatible with prior detector semantics: evidence must
    # start at/after onset and before the failure episode end.
    result["capacity_eventual"] = any(
        anchor <= index < end_index_exclusive
        for index in eligible
    )
    return result


def _event_end_index(
    frames: Sequence[int],
    event: Mapping[str, Any],
) -> int:
    end = event.get("episode_end_frame")
    if end is None:
        return len(frames)
    end_frame = int(end)
    return next(
        (
            index
            for index, frame in enumerate(frames)
            if int(frame) >= end_frame
        ),
        len(frames),
    )


def _evaluate_config_masks(
    config_id: str,
    config_family: str,
    config_source: str,
    parameters_json: str,
    masks: Mapping[str, Sequence[bool]],
    signals: Mapping[str, Mapping[str, Any]],
    populations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in populations:
        population = str(spec["population"])
        events = list(spec["events"])
        per_event: list[dict[str, Any]] = []
        for event in events:
            rollout_id = str(event["rollout_id"])
            signal = signals.get(rollout_id)
            mask = masks.get(rollout_id)
            if signal is None or mask is None:
                continue
            frames = [int(value) for value in signal["frames"]]
            if len(frames) != len(mask):
                continue
            anchor = _event_anchor(
                frames,
                int(event["observable_onset_frame"]),
            )
            if anchor is None:
                continue
            starts = _episode_starts(mask)
            end_index = _event_end_index(frames, event)
            capacity = _capacity_metrics(
                starts,
                anchor,
                end_index,
            )
            unique_index = starts[0] if starts else None
            signed_offset = (
                unique_index - anchor
                if unique_index is not None
                else None
            )
            row: dict[str, Any] = {
                "config_id": config_id,
                "detector_family": config_family,
                "config_source": config_source,
                "parameters_json": parameters_json,
                "population": population,
                "event_id": _event_identity(event),
                "rollout_id": rollout_id,
                "event_index": event.get("event_index"),
                "failure_type": event.get("failure_type"),
                "outcome": event.get("outcome"),
                "observable_onset_frame": event.get("observable_onset_frame"),
                "onset_anchor_sample_index": anchor,
                "onset_anchor_frame": frames[anchor],
                "positive_episode_n": len(starts),
                "unique_localization_sample_index": unique_index,
                "unique_localization_frame": (
                    frames[unique_index]
                    if unique_index is not None
                    else None
                ),
                "localization_output": unique_index is not None,
                "localization_signed_offset_samples": signed_offset,
                "localization_abs_error_samples": (
                    abs(signed_offset)
                    if signed_offset is not None
                    else None
                ),
                "localization_relation": (
                    None
                    if signed_offset is None
                    else (
                        "before_onset"
                        if signed_offset < 0
                        else (
                            "at_onset"
                            if signed_offset == 0
                            else "after_onset"
                        )
                    )
                ),
                **capacity,
            }
            for window in LOCALIZATION_WINDOWS:
                row[f"localization_within_{window}"] = bool(
                    signed_offset is not None
                    and abs(signed_offset) <= window
                )
            per_event.append(row)
            detail_rows.append(row)

        if not per_event:
            continue
        n = len(per_event)
        localized = [
            row
            for row in per_event
            if bool(row["localization_output"])
        ]
        signed = [
            float(row["localization_signed_offset_samples"])
            for row in localized
        ]
        absolute = [abs(value) for value in signed]
        summary: dict[str, Any] = {
            "config_id": config_id,
            "detector_family": config_family,
            "config_source": config_source,
            "parameters_json": parameters_json,
            "population": population,
            "event_n": n,
            "localization_output_n": len(localized),
            "localization_output_coverage": len(localized) / n,
            "median_signed_offset_samples": (
                float(statistics.median(signed))
                if signed
                else None
            ),
            "median_absolute_error_samples": (
                float(statistics.median(absolute))
                if absolute
                else None
            ),
            "mae_samples": (
                sum(absolute) / len(absolute)
                if absolute
                else None
            ),
            "before_onset_n": sum(
                row.get("localization_relation") == "before_onset"
                for row in per_event
            ),
            "at_onset_n": sum(
                row.get("localization_relation") == "at_onset"
                for row in per_event
            ),
            "after_onset_n": sum(
                row.get("localization_relation") == "after_onset"
                for row in per_event
            ),
            "no_localization_output_n": n - len(localized),
        }
        for window in LOCALIZATION_WINDOWS:
            summary[f"capacity_within_{window}_recall"] = (
                sum(bool(row[f"capacity_within_{window}"]) for row in per_event)
                / n
            )
            summary[f"localization_within_{window}_recall"] = (
                sum(bool(row[f"localization_within_{window}"]) for row in per_event)
                / n
            )
        summary["capacity_eventual_recall"] = (
            sum(bool(row["capacity_eventual"]) for row in per_event) / n
        )
        summary_rows.append(summary)

    return detail_rows, summary_rows


def evaluate_global_config_space(
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    single_configs: Sequence[Mapping[str, Any]],
    or_configs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate every fixed config without any clean-FPR filtering."""
    populations = _population_specs(events)
    masks_by_config: dict[str, dict[str, list[bool]]] = {}

    for config in single_configs:
        config_id = str(config["config_id"])
        masks_by_config[config_id] = {
            rollout_id: detector_mask(signal["hops"], config)
            for rollout_id, signal in signals.items()
        }

    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for config in single_configs:
        config_id = str(config["config_id"])
        config_details, config_summaries = _evaluate_config_masks(
            config_id,
            str(config["detector_family"]),
            str(config.get("config_source") or "single"),
            str(config["parameters_json"]),
            masks_by_config[config_id],
            signals,
            populations,
        )
        details.extend(config_details)
        summaries.extend(config_summaries)

    for pair in or_configs:
        a_id = str(pair["stagnation_config_id"])
        b_id = str(pair["regression_config_id"])
        if a_id not in masks_by_config or b_id not in masks_by_config:
            continue
        pair_masks = {
            rollout_id: [
                bool(a) or bool(b)
                for a, b in zip(
                    masks_by_config[a_id][rollout_id],
                    masks_by_config[b_id][rollout_id],
                )
            ]
            for rollout_id in signals
        }
        pair_details, pair_summaries = _evaluate_config_masks(
            str(pair["config_id"]),
            "phenotype_or",
            str(pair["config_source"]),
            str(pair["parameters_json"]),
            pair_masks,
            signals,
            populations,
        )
        details.extend(pair_details)
        summaries.extend(pair_summaries)

    return details, summaries


def select_global_config_results(
    summaries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select metric-specific capacity envelope and GT-independent localizers."""
    envelope: list[dict[str, Any]] = []
    single_best: list[dict[str, Any]] = []
    populations = sorted(
        {
            str(row["population"])
            for row in summaries
        }
    )

    for population in populations:
        rows = [
            row
            for row in summaries
            if str(row["population"]) == population
        ]
        for target in (
            "capacity_within_1_recall",
            "capacity_within_3_recall",
            "capacity_within_5_recall",
            "capacity_within_10_recall",
            "capacity_eventual_recall",
        ):
            eligible = [row for row in rows if row.get(target) is not None]
            if not eligible:
                continue
            best = min(
                eligible,
                key=lambda row: (
                    -float(row[target]),
                    float(row.get("median_absolute_error_samples") or math.inf),
                    float(row.get("mae_samples") or math.inf),
                    str(row["config_id"]),
                ),
            )
            envelope.append(
                {
                    "population": population,
                    "selection_kind": "metric_specific_capacity_upper_envelope",
                    "selection_target": target,
                    "selection_value": best[target],
                    **dict(best),
                }
            )

        full_coverage = [
            row
            for row in rows
            if float(row.get("localization_output_coverage") or 0.0) >= 1.0 - 1e-12
            and row.get("median_absolute_error_samples") is not None
            and row.get("mae_samples") is not None
        ]
        if full_coverage:
            selection_pool = full_coverage
            coverage_status = "full_coverage"
        else:
            max_coverage = max(
                (
                    float(row.get("localization_output_coverage") or 0.0)
                    for row in rows
                ),
                default=0.0,
            )
            selection_pool = [
                row
                for row in rows
                if abs(
                    float(row.get("localization_output_coverage") or 0.0)
                    - max_coverage
                ) <= 1e-12
                and row.get("median_absolute_error_samples") is not None
                and row.get("mae_samples") is not None
            ]
            coverage_status = "max_coverage_fallback"

        if selection_pool:
            median_best = min(
                selection_pool,
                key=lambda row: (
                    float(row["median_absolute_error_samples"]),
                    float(row["mae_samples"]),
                    str(row["config_id"]),
                ),
            )
            single_best.append(
                {
                    "population": population,
                    "selection_kind": "single_global_config",
                    "selection_target": "minimum_median_absolute_error",
                    "coverage_requirement_status": coverage_status,
                    **dict(median_best),
                }
            )

            mae_best = min(
                selection_pool,
                key=lambda row: (
                    float(row["mae_samples"]),
                    float(row["median_absolute_error_samples"]),
                    str(row["config_id"]),
                ),
            )
            single_best.append(
                {
                    "population": population,
                    "selection_kind": "single_global_config",
                    "selection_target": "minimum_mae",
                    "coverage_requirement_status": coverage_status,
                    **dict(mae_best),
                }
            )

    return envelope, single_best


def ranking_rows(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in summaries),
        key=lambda row: (
            str(row["population"]),
            -float(row.get("capacity_within_3_recall") or 0.0),
            -float(row.get("capacity_within_10_recall") or 0.0),
            -float(row.get("capacity_eventual_recall") or 0.0),
            float(row.get("median_absolute_error_samples") or math.inf),
            float(row.get("mae_samples") or math.inf),
            str(row["config_id"]),
        ),
    )


def _cache_fingerprint(
    base_search_fingerprint: str,
    single_configs: Sequence[Mapping[str, Any]],
    or_configs: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "schema": GLOBAL_LOCALIZATION_SCHEMA,
        "semantics": GLOBAL_LOCALIZATION_SEMANTICS_VERSION,
        "base_search_fingerprint": base_search_fingerprint,
        "single_configs": [
            (
                config["config_id"],
                config["detector_family"],
                config["parameters_json"],
            )
            for config in single_configs
        ],
        "or_configs": [
            (
                config["config_id"],
                config["parameters_json"],
            )
            for config in or_configs
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def load_or_compute_global_localization(
    *,
    base_search_fingerprint: str,
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    empirical_phenotype_configs: Sequence[Mapping[str, Any]],
    refresh: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    singles, pairs, space_metadata = build_global_config_space(
        empirical_phenotype_configs
    )
    fingerprint = _cache_fingerprint(
        base_search_fingerprint,
        singles,
        pairs,
    )
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = CACHE_ROOT / f"{fingerprint}.json.gz"

    if not refresh and path.is_file():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                cached = json.load(handle)
            if (
                cached.get("schema") == GLOBAL_LOCALIZATION_SCHEMA
                and cached.get("semantics")
                == GLOBAL_LOCALIZATION_SEMANTICS_VERSION
                and cached.get("fingerprint") == fingerprint
            ):
                return (
                    list(cached["ranking"]),
                    list(cached["envelope"]),
                    list(cached["single_best"]),
                    list(cached["details"]),
                    {
                        **space_metadata,
                        "cache_hit": True,
                        "cache_fingerprint": fingerprint,
                    },
                )
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    details, summaries = evaluate_global_config_space(
        signals,
        events,
        singles,
        pairs,
    )
    envelope, single_best = select_global_config_results(summaries)
    ranking = ranking_rows(summaries)
    document = {
        "schema": GLOBAL_LOCALIZATION_SCHEMA,
        "semantics": GLOBAL_LOCALIZATION_SEMANTICS_VERSION,
        "fingerprint": fingerprint,
        "ranking": ranking,
        "envelope": envelope,
        "single_best": single_best,
        "details": details,
    }

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{fingerprint}.",
        suffix=".json.gz.tmp",
        dir=CACHE_ROOT,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with gzip.open(
            temporary,
            "wt",
            encoding="utf-8",
            compresslevel=1,
        ) as handle:
            json.dump(
                document,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return (
        ranking,
        envelope,
        single_best,
        details,
        {
            **space_metadata,
            "cache_hit": False,
            "cache_fingerprint": fingerprint,
        },
    )
