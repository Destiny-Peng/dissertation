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


def _number_or_inf(value: Any) -> float:
    return math.inf if value is None else float(value)


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
    existing_phenotype_configs: Sequence[Mapping[str, Any]],
    existing_ensemble_sweep: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """Reuse the historical full grid plus already-searched phenotype configs.

    No new phenotype pair search is created here. OR candidates are exactly the
    unique parameter pairs already present in ensemble_sweep; this analysis
    merely removes clean-FPR filtering from their localization evaluation.
    """
    ranked_singles: list[dict[str, Any]] = []
    support_configs: dict[str, dict[str, Any]] = {}
    seen_ranked: set[tuple[str, str]] = set()

    for index, config in enumerate(build_detector_configs(), 1):
        key = _config_key(config)
        if key in seen_ranked:
            continue
        seen_ranked.add(key)
        copied = _copy_config(
            config,
            config_id=f"global_fixed_{index:05d}",
            config_source="historical_full_grid",
        )
        ranked_singles.append(copied)
        support_configs[str(copied["config_id"])] = copied

    phenotype_by_original_id: dict[str, dict[str, Any]] = {}
    phenotype_added = 0
    for config in existing_phenotype_configs:
        original_id = str(config["config_id"])
        copied = dict(config)
        copied["config_source"] = "existing_phenotype_component_grid"
        phenotype_by_original_id[original_id] = copied
        support_configs[original_id] = copied

        key = _config_key(config)
        if key in seen_ranked:
            continue
        seen_ranked.add(key)
        phenotype_added += 1
        ranked_copy = _copy_config(
            config,
            config_id=f"global_phen_{phenotype_added:05d}",
            config_source="existing_phenotype_component_grid",
        )
        ranked_singles.append(ranked_copy)
        support_configs[str(ranked_copy["config_id"])] = ranked_copy

    pair_keys: set[tuple[str, str]] = set()
    pairs: list[dict[str, Any]] = []
    skipped_missing_components = 0
    for row in existing_ensemble_sweep:
        a_id = str(row.get("a_config_id") or "")
        b_id = str(row.get("b_config_id") or "")
        if not a_id or not b_id:
            continue
        key = (a_id, b_id)
        if key in pair_keys:
            continue
        pair_keys.add(key)
        config_a = phenotype_by_original_id.get(a_id)
        config_b = phenotype_by_original_id.get(b_id)
        if config_a is None or config_b is None:
            skipped_missing_components += 1
            continue
        pair_id = f"global_or_{len(pairs) + 1:07d}"
        parameters = {
            "logic": "OR",
            "stagnation_config_id": a_id,
            "stagnation_family": config_a["detector_family"],
            "stagnation_parameters": json.loads(
                str(config_a["parameters_json"])
            ),
            "regression_config_id": b_id,
            "regression_family": config_b["detector_family"],
            "regression_parameters": json.loads(
                str(config_b["parameters_json"])
            ),
        }
        pairs.append(
            {
                "config_id": pair_id,
                "detector_family": "phenotype_or",
                "config_source": "reused_existing_ensemble_sweep",
                "parameters_json": json.dumps(
                    parameters,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "stagnation_config_id": a_id,
                "regression_config_id": b_id,
            }
        )

    family_counts: dict[str, int] = defaultdict(int)
    for config in ranked_singles:
        family_counts[str(config["detector_family"])] += 1

    return ranked_singles, pairs, support_configs, {
        "single_config_n": len(ranked_singles),
        "or_config_n": len(pairs),
        "total_config_n": len(ranked_singles) + len(pairs),
        "historical_fixed_grid_n": len(build_detector_configs()),
        "existing_phenotype_unique_added_n": phenotype_added,
        "existing_ensemble_sweep_row_n": len(existing_ensemble_sweep),
        "or_missing_component_pair_n": skipped_missing_components,
        "single_family_counts": dict(sorted(family_counts.items())),
        "or_definition": (
            "unique stagnation/regression parameter pairs already present in "
            "the cached existing ensemble_sweep; no new pair search and no "
            "clean-FPR filtering are applied in this localization analysis"
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
) -> list[dict[str, Any]]:
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
            capacity_n = sum(
                bool(row[f"capacity_within_{window}"])
                for row in per_event
            )
            localization_n = sum(
                bool(row[f"localization_within_{window}"])
                for row in per_event
            )
            summary[f"capacity_within_{window}_n"] = capacity_n
            summary[f"capacity_within_{window}_recall"] = capacity_n / n
            summary[f"localization_within_{window}_n"] = localization_n
            summary[f"localization_within_{window}_recall"] = (
                localization_n / n
            )
        capacity_eventual_n = sum(
            bool(row["capacity_eventual"]) for row in per_event
        )
        summary["capacity_eventual_detected_n"] = capacity_eventual_n
        summary["capacity_eventual_recall"] = capacity_eventual_n / n
        summary_rows.append(summary)

    return summary_rows


def evaluate_global_config_space(
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    single_configs: Sequence[Mapping[str, Any]],
    or_configs: Sequence[Mapping[str, Any]],
    support_configs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate every fixed config without any clean-FPR filtering."""
    populations = _population_specs(events)
    target_rollout_ids = sorted(
        {
            str(event["rollout_id"])
            for spec in populations
            for event in spec["events"]
            if str(event["rollout_id"]) in signals
        }
    )
    masks_by_config: dict[str, dict[str, list[bool]]] = {}

    needed_ids = {
        str(config["config_id"])
        for config in single_configs
    }
    for pair in or_configs:
        needed_ids.add(str(pair["stagnation_config_id"]))
        needed_ids.add(str(pair["regression_config_id"]))

    for config_id in sorted(needed_ids):
        config = support_configs.get(config_id)
        if config is None:
            continue
        masks_by_config[config_id] = {
            rollout_id: detector_mask(signals[rollout_id]["hops"], config)
            for rollout_id in target_rollout_ids
        }

    summaries: list[dict[str, Any]] = []
    for config in single_configs:
        config_id = str(config["config_id"])
        config_summaries = _evaluate_config_masks(
            config_id,
            str(config["detector_family"]),
            str(config.get("config_source") or "single"),
            str(config["parameters_json"]),
            masks_by_config[config_id],
            signals,
            populations,
        )
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
            for rollout_id in target_rollout_ids
        }
        pair_summaries = _evaluate_config_masks(
            str(pair["config_id"]),
            "phenotype_or",
            str(pair["config_source"]),
            str(pair["parameters_json"]),
            pair_masks,
            signals,
            populations,
        )
        summaries.extend(pair_summaries)

    return summaries


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
                    _number_or_inf(row.get("median_absolute_error_samples")),
                    _number_or_inf(row.get("mae_samples")),
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
            _number_or_inf(row.get("median_absolute_error_samples")),
            _number_or_inf(row.get("mae_samples")),
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
    existing_phenotype_configs: Sequence[Mapping[str, Any]],
    existing_ensemble_sweep: Sequence[Mapping[str, Any]],
    refresh: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    singles, pairs, support_configs, space_metadata = build_global_config_space(
        existing_phenotype_configs,
        existing_ensemble_sweep,
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
                    {
                        **space_metadata,
                        "cache_hit": True,
                        "cache_fingerprint": fingerprint,
                    },
                )
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    summaries = evaluate_global_config_space(
        signals,
        events,
        singles,
        pairs,
        support_configs,
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
        {
            **space_metadata,
            "cache_hit": False,
            "cache_fingerprint": fingerprint,
        },
    )
