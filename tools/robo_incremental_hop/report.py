from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import (
    CLEAN_FPR_CONSTRAINTS,
    CONSECUTIVE_NS,
    EPSILONS,
    KOFM_MS,
    aggregate_clean_metrics,
    aggregate_event_metrics,
    config_row,
    detector_mask,
    evaluate_event,
    evaluate_failure_rollout_from_start,
    positive_episode_count,
    recovery_metrics,
)
from .io import project_relative


def _aggregate_no_event_failure_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    n = len(rows)
    detected = [row for row in rows if row.get("detected")]
    sample_delays = [
        float(row["delay_samples"])
        for row in detected
        if row.get("delay_samples") is not None
    ]
    frame_delays = [
        float(row["delay_frames"])
        for row in detected
        if row.get("delay_frames") is not None
    ]
    result: dict[str, Any] = {
        "no_event_failure_n": n,
        "no_event_detected_n": len(detected),
        "no_event_recall_eventual": (
            len(detected) / n if n else None
        ),
        "no_event_median_delay_samples": (
            statistics.median(sample_delays)
            if sample_delays
            else None
        ),
        "no_event_median_delay_frames": (
            statistics.median(frame_delays)
            if frame_delays
            else None
        ),
    }
    for window in (1, 3, 5, 10, 20):
        result[f"no_event_recall_at_{window}"] = (
            sum(bool(row.get(f"recall_at_{window}")) for row in rows) / n
            if n
            else None
        )
    return result


def _aggregate_grasp_event_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grasp = [
        row
        for row in rows
        if str(row.get("failure_type") or "") == "grasp_failure"
    ]
    n = len(grasp)
    detected = [row for row in grasp if row.get("eventual_recall")]
    result: dict[str, Any] = {
        "grasp_event_n": n,
        "grasp_detected_eventual_n": len(detected),
        "grasp_recall_eventual": len(detected) / n if n else None,
        "grasp_median_delay_samples": _median_or_none(
            [
                row["delay_samples"]
                for row in detected
                if row.get("delay_samples") is not None
            ]
        ),
        "grasp_median_delay_frames": _median_or_none(
            [
                row["delay_frames"]
                for row in detected
                if row.get("delay_frames") is not None
            ]
        ),
    }
    for window in (1, 3, 5, 10, 20):
        result[f"grasp_recall_at_{window}"] = (
            sum(bool(row.get(f"recall_at_{window}")) for row in grasp) / n
            if n
            else None
        )
    return result


def _aggregate_failed_rollout_coverage(
    event_rows: Sequence[Mapping[str, Any]],
    no_event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    annotated_terminal: dict[str, bool] = {}
    for row in event_rows:
        if row.get("outcome") != "terminal_failure":
            continue
        rollout_id = str(row["rollout_id"])
        annotated_terminal[rollout_id] = (
            annotated_terminal.get(rollout_id, False)
            or bool(row.get("eventual_recall"))
        )
    no_event_terminal = {
        str(row["rollout_id"]): bool(row.get("eventual_recall"))
        for row in no_event_rows
    }
    combined = {**annotated_terminal, **no_event_terminal}
    return {
        "failed_rollout_n": len(combined),
        "failed_rollout_detected_n": sum(combined.values()),
        "overall_failed_rollout_coverage": (
            sum(combined.values()) / len(combined)
            if combined
            else None
        ),
        "annotated_failed_rollout_n": len(annotated_terminal),
        "no_event_failed_rollout_n": len(no_event_terminal),
    }


def evaluate_all_configs(
    configs: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    events_by_rollout: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    for event in events:
        events_by_rollout[str(event["rollout_id"])].append(event)

    no_event_by_id = {
        str(row["rollout_id"]): row
        for row in no_event_failures
    }
    clean_by_id = {
        str(row["rollout_id"]): row
        for row in clean_rollouts
    }
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    no_event_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []

    for config in configs:
        masks = {
            rollout_id: detector_mask(signal["hops"], config)
            for rollout_id, signal in signals.items()
        }
        config_events: list[dict[str, Any]] = []
        config_no_event: list[dict[str, Any]] = []
        config_clean: list[dict[str, Any]] = []

        for rollout_id, rollout_events in events_by_rollout.items():
            signal = signals.get(rollout_id)
            if signal is None:
                continue
            mask = masks[rollout_id]
            for event in rollout_events:
                metrics = evaluate_event(
                    signal["frames"],
                    mask,
                    int(event["observable_onset_frame"]),
                    episode_end_frame=event.get("episode_end_frame"),
                    episode_end_source=str(
                        event.get("episode_end_source") or "rollout_end"
                    ),
                )
                row = {
                    **config_row(config),
                    **event,
                    **metrics,
                    "native_sample_n": len(signal["frames"]),
                    "signal_source": project_relative(
                        signal["prediction_path"]
                    ),
                }
                config_events.append(row)
                event_rows.append(row)

        for rollout_id, failure in no_event_by_id.items():
            signal = signals.get(rollout_id)
            if signal is None:
                continue
            metrics = evaluate_failure_rollout_from_start(
                signal["frames"],
                masks[rollout_id],
            )
            row = {
                **config_row(config),
                **failure,
                **metrics,
                "native_sample_n": len(signal["frames"]),
                "signal_source": project_relative(
                    signal["prediction_path"]
                ),
            }
            config_no_event.append(row)
            no_event_rows.append(row)

        for rollout_id, clean in clean_by_id.items():
            signal = signals.get(rollout_id)
            if signal is None:
                continue
            mask = masks[rollout_id]
            positive = [
                index
                for index, value in enumerate(mask)
                if value
            ]
            row = {
                **config_row(config),
                **clean,
                "native_sample_n": len(mask),
                "positive_sample_n": len(positive),
                "positive_sample_fraction": (
                    len(positive) / len(mask)
                    if mask
                    else None
                ),
                "any_positive": bool(positive),
                "first_positive_frame": (
                    signal["frames"][positive[0]]
                    if positive
                    else None
                ),
                "positive_episode_n": positive_episode_count(mask),
                "signal_source": project_relative(
                    signal["prediction_path"]
                ),
            }
            config_clean.append(row)
            clean_rows.append(row)

        summary_rows.append(
            {
                **config_row(config),
                **aggregate_event_metrics(config_events),
                **_aggregate_grasp_event_metrics(config_events),
                **_aggregate_no_event_failure_metrics(config_no_event),
                **_aggregate_failed_rollout_coverage(
                    config_events,
                    config_no_event,
                ),
                **aggregate_clean_metrics(config_clean),
            }
        )

    return summary_rows, event_rows, no_event_rows, clean_rows

def select_best_configs(
    summary_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_family: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in summary_rows:
        by_family[str(row["detector_family"])].append(row)

    result: list[dict[str, Any]] = []
    for family in sorted(by_family):
        family_rows = by_family[family]
        grasp_mode = any(
            row.get("grasp_recall_eventual") is not None
            for row in family_rows
        )
        for constraint in CLEAN_FPR_CONSTRAINTS:
            eligible = [
                row
                for row in family_rows
                if (
                    row.get("clean_rollout_fpr") is not None
                    and float(row["clean_rollout_fpr"])
                    <= constraint + 1e-12
                    and (
                        row.get("grasp_recall_eventual") is not None
                        if grasp_mode
                        else row.get("event_recall_at_3") is not None
                    )
                )
            ]
            if not eligible:
                result.append(
                    {
                        "detector_family": family,
                        "clean_fpr_constraint": constraint,
                        "selection_status": "no_eligible_config",
                        "selection_target": (
                            "grasp_recall_eventual"
                            if grasp_mode
                            else "event_recall_at_3"
                        ),
                    }
                )
                continue

            def rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
                fpr = float(row.get("clean_rollout_fpr") or 0.0)
                if grasp_mode:
                    delay = row.get("grasp_median_delay_samples")
                    return (
                        -float(row.get("grasp_recall_eventual") or 0.0),
                        -float(row.get("grasp_recall_at_10") or 0.0),
                        -float(
                            row.get("overall_failed_rollout_coverage")
                            or 0.0
                        ),
                        (
                            float(delay)
                            if delay is not None
                            else math.inf
                        ),
                        fpr,
                        str(row["config_id"]),
                    )

                delay = row.get("median_delay_samples")
                return (
                    -float(row.get("event_recall_at_3") or 0.0),
                    (
                        float(delay)
                        if delay is not None
                        else math.inf
                    ),
                    fpr,
                    str(row["config_id"]),
                )

            best = min(eligible, key=rank)
            result.append(
                {
                    "clean_fpr_constraint": constraint,
                    "selection_status": "selected",
                    "selection_target": (
                        "grasp_recall_eventual"
                        if grasp_mode
                        else "event_recall_at_3"
                    ),
                    **dict(best),
                }
            )
    return result


def summarize_breakdowns(
    configs: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    clean_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    events_by_config: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    clean_by_config: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)

    for row in event_rows:
        events_by_config[
            str(row["config_id"])
        ].append(row)
    for row in clean_rows:
        clean_by_config[
            str(row["config_id"])
        ].append(row)

    result: list[dict[str, Any]] = []
    for config in configs:
        config_id = str(config["config_id"])
        events = events_by_config.get(
            config_id, []
        )
        clean = clean_by_config.get(
            config_id, []
        )
        groups: list[
            tuple[
                str,
                str,
                list[Mapping[str, Any]],
                list[Mapping[str, Any]],
            ]
        ] = [
            (
                "overall",
                "all",
                list(events),
                list(clean),
            )
        ]

        for failure_type in sorted(
            {
                str(row["failure_type"])
                for row in events
            }
        ):
            groups.append(
                (
                    "failure_type",
                    failure_type,
                    [
                        row
                        for row in events
                        if str(
                            row["failure_type"]
                        )
                        == failure_type
                    ],
                    [],
                )
            )

        for outcome in (
            "terminal_failure",
            "recovered_success",
        ):
            subset = [
                row
                for row in events
                if row.get("outcome")
                == outcome
            ]
            if subset:
                groups.append(
                    (
                        "outcome",
                        outcome,
                        subset,
                        [],
                    )
                )

        task_keys = sorted(
            {
                str(row["task_key"])
                for row in events
            }
            | {
                str(row["task_key"])
                for row in clean
            }
        )
        for task_key in task_keys:
            groups.append(
                (
                    "task",
                    task_key,
                    [
                        row
                        for row in events
                        if str(
                            row["task_key"]
                        )
                        == task_key
                    ],
                    [
                        row
                        for row in clean
                        if str(
                            row["task_key"]
                        )
                        == task_key
                    ],
                )
            )

        for (
            dimension,
            value,
            event_subset,
            clean_subset,
        ) in groups:
            result.append(
                {
                    **config_row(config),
                    "group_dimension": dimension,
                    "group_value": value,
                    **aggregate_event_metrics(
                        event_subset
                    ),
                    **aggregate_clean_metrics(
                        clean_subset
                    ),
                }
            )
    return result



PAIRWISE_HORIZONS: tuple[tuple[str, str], ...] = (
    ("1", "recall_at_1"),
    ("3", "recall_at_3"),
    ("5", "recall_at_5"),
    ("10", "recall_at_10"),
    ("20", "recall_at_20"),
    ("eventual", "eventual_recall"),
)

ENSEMBLE_FAMILY_PAIRS: tuple[tuple[str, str], ...] = (
    ("stagnation_consecutive", "regression_window_min"),
    ("stagnation_k_of_m", "regression_window_min"),
)

ENSEMBLE_SELECTION_TARGETS: tuple[str, ...] = (
    "grasp_recall_eventual",
    "grasp_recall_at_10",
)


def _event_identity(row: Mapping[str, Any]) -> str:
    event_id = row.get("event_id")
    if event_id not in (None, ""):
        return str(event_id)
    return (
        f"{row.get('rollout_id')}::event"
        f"{int(row.get('event_index') or 0)}"
    )


def _median_or_none(values: Sequence[float | int]) -> float | None:
    return float(statistics.median(values)) if values else None


def _detected_ids(
    rows: Mapping[str, Mapping[str, Any]],
    field: str,
) -> set[str]:
    return {
        row_id
        for row_id, row in rows.items()
        if bool(row.get(field))
    }


def _or_delay_median(
    ids: set[str],
    rows_a: Mapping[str, Mapping[str, Any]],
    rows_b: Mapping[str, Mapping[str, Any]],
    field: str,
) -> tuple[float | None, float | None]:
    sample_delays: list[float | int] = []
    frame_delays: list[float | int] = []
    for row_id in ids:
        candidates = []
        if bool(rows_a[row_id].get(field)):
            candidates.append(rows_a[row_id])
        if bool(rows_b[row_id].get(field)):
            candidates.append(rows_b[row_id])
        sample_values = [
            row["delay_samples"]
            for row in candidates
            if row.get("delay_samples") is not None
        ]
        frame_values = [
            row["delay_frames"]
            for row in candidates
            if row.get("delay_frames") is not None
        ]
        if sample_values:
            sample_delays.append(min(sample_values))
        if frame_values:
            frame_delays.append(min(frame_values))
    return (
        _median_or_none(sample_delays),
        _median_or_none(frame_delays),
    )


def _prefixed_config(
    prefix: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in config_row(config).items()
    }


def build_pairwise_ensemble_rows(
    configs: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    no_event_rows: Sequence[Mapping[str, Any]],
    clean_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Jointly sweep prioritized detector-family pairs using OR logic.

    This uses only already evaluated detector outputs. No detector threshold is
    retuned outside the existing parameter grid, and no Robo-Dopamine inference
    is launched.
    """
    configs_by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for config in configs:
        configs_by_family[str(config["detector_family"])].append(config)

    event_by_config: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in event_rows:
        event_by_config[str(row["config_id"])][_event_identity(row)] = row

    no_event_by_config: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in no_event_rows:
        no_event_by_config[str(row["config_id"])][str(row["rollout_id"])] = row

    clean_by_config: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in clean_rows:
        clean_by_config[str(row["config_id"])][str(row["rollout_id"])] = row

    max_budget = max(CLEAN_FPR_CONSTRAINTS)
    for family, family_configs in list(configs_by_family.items()):
        feasible = []
        for config in family_configs:
            rows = clean_by_config.get(str(config["config_id"]), {})
            if not rows:
                feasible.append(config)
                continue
            fpr = (
                sum(bool(row.get("any_positive")) for row in rows.values())
                / len(rows)
            )
            if fpr <= max_budget + 1e-12:
                feasible.append(config)
        configs_by_family[family] = feasible

    sweep_rows: list[dict[str, Any]] = []

    for pair_index, (family_a, family_b) in enumerate(
        ENSEMBLE_FAMILY_PAIRS, 1
    ):
        for config_a in configs_by_family.get(family_a, []):
            for config_b in configs_by_family.get(family_b, []):
                config_a_id = str(config_a["config_id"])
                config_b_id = str(config_b["config_id"])
                event_a = event_by_config.get(config_a_id, {})
                event_b = event_by_config.get(config_b_id, {})
                no_event_a = no_event_by_config.get(config_a_id, {})
                no_event_b = no_event_by_config.get(config_b_id, {})
                clean_a = clean_by_config.get(config_a_id, {})
                clean_b = clean_by_config.get(config_b_id, {})

                if set(event_a) != set(event_b):
                    raise ValueError(
                        "Ensemble sweep requires identical event denominators: "
                        f"{config_a_id} vs {config_b_id}"
                    )
                if set(no_event_a) != set(no_event_b):
                    raise ValueError(
                        "Ensemble sweep requires identical no-event failure denominators: "
                        f"{config_a_id} vs {config_b_id}"
                    )
                if set(clean_a) != set(clean_b):
                    raise ValueError(
                        "Ensemble sweep requires identical clean denominators: "
                        f"{config_a_id} vs {config_b_id}"
                    )

                event_ids = set(event_a)
                grasp_event_ids = {
                    event_id
                    for event_id, event_row in event_a.items()
                    if str(event_row.get("failure_type") or "")
                    == "grasp_failure"
                }
                no_event_ids = set(no_event_a)
                clean_ids = set(clean_a)

                fp_a = {
                    rollout_id
                    for rollout_id in clean_ids
                    if bool(clean_a[rollout_id].get("any_positive"))
                }
                fp_b = {
                    rollout_id
                    for rollout_id in clean_ids
                    if bool(clean_b[rollout_id].get("any_positive"))
                }
                fp_overlap = fp_a & fp_b
                fp_union = fp_a | fp_b
                clean_n = len(clean_ids)

                row: dict[str, Any] = {
                    "signal_mode": "fused",
                    "ensemble_logic": "OR",
                    "pair_priority": pair_index,
                    "detector_a_family": family_a,
                    "detector_b_family": family_b,
                    **_prefixed_config("a", config_a),
                    **_prefixed_config("b", config_b),
                    "clean_rollout_n": clean_n,
                    "a_fp_n": len(fp_a),
                    "b_fp_n": len(fp_b),
                    "fp_overlap_n": len(fp_overlap),
                    "a_only_fp_n": len(fp_a - fp_b),
                    "b_only_fp_n": len(fp_b - fp_a),
                    "or_fp_n": len(fp_union),
                    "clean_rollout_fpr": (
                        len(fp_union) / clean_n if clean_n else None
                    ),
                    "fp_jaccard": (
                        len(fp_overlap) / len(fp_union)
                        if fp_union
                        else None
                    ),
                    "event_n": len(event_ids),
                    "grasp_event_n": len(grasp_event_ids),
                    "no_event_failure_n": len(no_event_ids),
                }

                for horizon, field in PAIRWISE_HORIZONS:
                    detected_event_a = _detected_ids(event_a, field)
                    detected_event_b = _detected_ids(event_b, field)
                    event_overlap = detected_event_a & detected_event_b
                    event_union = detected_event_a | detected_event_b
                    row[f"event_detected_at_{horizon}_n"] = len(event_union)
                    event_recall_key = (
                        "event_recall_eventual"
                        if horizon == "eventual"
                        else f"event_recall_at_{horizon}"
                    )
                    row[event_recall_key] = (
                        len(event_union) / len(event_ids)
                        if event_ids
                        else None
                    )
                    row[f"event_overlap_at_{horizon}_n"] = len(event_overlap)
                    row[f"event_a_only_at_{horizon}_n"] = len(
                        detected_event_a - detected_event_b
                    )
                    row[f"event_b_only_at_{horizon}_n"] = len(
                        detected_event_b - detected_event_a
                    )
                    row[f"event_tp_jaccard_at_{horizon}"] = (
                        len(event_overlap) / len(event_union)
                        if event_union
                        else None
                    )

                    grasp_a = detected_event_a & grasp_event_ids
                    grasp_b = detected_event_b & grasp_event_ids
                    grasp_overlap = grasp_a & grasp_b
                    grasp_union = grasp_a | grasp_b
                    grasp_recall_key = (
                        "grasp_recall_eventual"
                        if horizon == "eventual"
                        else f"grasp_recall_at_{horizon}"
                    )
                    row[grasp_recall_key] = (
                        len(grasp_union) / len(grasp_event_ids)
                        if grasp_event_ids
                        else None
                    )
                    grasp_a_recall = (
                        len(grasp_a) / len(grasp_event_ids)
                        if grasp_event_ids
                        else None
                    )
                    grasp_b_recall = (
                        len(grasp_b) / len(grasp_event_ids)
                        if grasp_event_ids
                        else None
                    )
                    grasp_suffix = (
                        "eventual" if horizon == "eventual" else f"at_{horizon}"
                    )
                    row[f"grasp_a_recall_{grasp_suffix}"] = grasp_a_recall
                    row[f"grasp_b_recall_{grasp_suffix}"] = grasp_b_recall
                    row[f"grasp_gain_vs_best_{grasp_suffix}"] = (
                        row[grasp_recall_key] - max(grasp_a_recall, grasp_b_recall)
                        if row[grasp_recall_key] is not None
                        and grasp_a_recall is not None
                        and grasp_b_recall is not None
                        else None
                    )
                    row[f"grasp_detected_at_{horizon}_n"] = len(grasp_union)
                    row[f"grasp_overlap_at_{horizon}_n"] = len(grasp_overlap)
                    row[f"grasp_a_only_at_{horizon}_n"] = len(
                        grasp_a - grasp_b
                    )
                    row[f"grasp_b_only_at_{horizon}_n"] = len(
                        grasp_b - grasp_a
                    )
                    row[f"grasp_tp_jaccard_at_{horizon}"] = (
                        len(grasp_overlap) / len(grasp_union)
                        if grasp_union
                        else None
                    )

                    detected_no_event_a = _detected_ids(no_event_a, field)
                    detected_no_event_b = _detected_ids(no_event_b, field)
                    no_event_overlap = detected_no_event_a & detected_no_event_b
                    no_event_union = detected_no_event_a | detected_no_event_b
                    row[f"no_event_detected_at_{horizon}_n"] = len(
                        no_event_union
                    )
                    no_event_recall_key = (
                        "no_event_recall_eventual"
                        if horizon == "eventual"
                        else f"no_event_recall_at_{horizon}"
                    )
                    row[no_event_recall_key] = (
                        len(no_event_union) / len(no_event_ids)
                        if no_event_ids
                        else None
                    )
                    row[f"no_event_overlap_at_{horizon}_n"] = len(
                        no_event_overlap
                    )
                    row[f"no_event_a_only_at_{horizon}_n"] = len(
                        detected_no_event_a - detected_no_event_b
                    )
                    row[f"no_event_b_only_at_{horizon}_n"] = len(
                        detected_no_event_b - detected_no_event_a
                    )
                    row[f"no_event_tp_jaccard_at_{horizon}"] = (
                        len(no_event_overlap) / len(no_event_union)
                        if no_event_union
                        else None
                    )

                event_eventual = (
                    _detected_ids(event_a, "eventual_recall")
                    | _detected_ids(event_b, "eventual_recall")
                )
                (
                    row["event_median_delay_samples"],
                    row["event_median_delay_frames"],
                ) = _or_delay_median(
                    event_eventual,
                    event_a,
                    event_b,
                    "eventual_recall",
                )
                grasp_eventual = event_eventual & grasp_event_ids
                (
                    row["grasp_median_delay_samples"],
                    row["grasp_median_delay_frames"],
                ) = _or_delay_median(
                    grasp_eventual,
                    event_a,
                    event_b,
                    "eventual_recall",
                )
                no_event_eventual = (
                    _detected_ids(no_event_a, "eventual_recall")
                    | _detected_ids(no_event_b, "eventual_recall")
                )
                (
                    row["no_event_median_delay_samples"],
                    row["no_event_median_delay_frames"],
                ) = _or_delay_median(
                    no_event_eventual,
                    no_event_a,
                    no_event_b,
                    "eventual_recall",
                )

                annotated_terminal: dict[str, bool] = {}
                for event_id in event_ids:
                    event_row = event_a[event_id]
                    if event_row.get("outcome") != "terminal_failure":
                        continue
                    rollout_id = str(event_row["rollout_id"])
                    detected = event_id in event_eventual
                    annotated_terminal[rollout_id] = (
                        annotated_terminal.get(rollout_id, False)
                        or detected
                    )
                no_event_terminal = {
                    rollout_id: rollout_id in no_event_eventual
                    for rollout_id in no_event_ids
                }
                failed_rollouts = {
                    **annotated_terminal,
                    **no_event_terminal,
                }
                row["failed_rollout_n"] = len(failed_rollouts)
                row["failed_rollout_detected_n"] = sum(
                    failed_rollouts.values()
                )
                row["overall_failed_rollout_coverage"] = (
                    sum(failed_rollouts.values()) / len(failed_rollouts)
                    if failed_rollouts
                    else None
                )
                sweep_rows.append(row)

    selected_rows: list[dict[str, Any]] = []
    for constraint in CLEAN_FPR_CONSTRAINTS:
        eligible = [
            row
            for row in sweep_rows
            if row.get("clean_rollout_fpr") is not None
            and float(row["clean_rollout_fpr"]) <= constraint + 1e-12
        ]
        for target in ENSEMBLE_SELECTION_TARGETS:
            target_eligible = [
                row
                for row in eligible
                if row.get(target) is not None
            ]
            if not target_eligible:
                selected_rows.append(
                    {
                        "signal_mode": "fused",
                        "ensemble_logic": "OR",
                        "clean_fpr_constraint": constraint,
                        "selection_target": target,
                        "selection_status": "no_eligible_config",
                    }
                )
                continue

            def rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
                secondary_target = (
                    "grasp_recall_at_10"
                    if target == "grasp_recall_eventual"
                    else "grasp_recall_eventual"
                )
                delay = row.get("grasp_median_delay_samples")
                return (
                    -float(row[target]),
                    -float(row.get(secondary_target) or 0.0),
                    -float(
                        row.get("overall_failed_rollout_coverage")
                        or 0.0
                    ),
                    (
                        float(delay)
                        if delay is not None
                        else math.inf
                    ),
                    float(row.get("clean_rollout_fpr") or 0.0),
                    int(row.get("pair_priority") or 999),
                    str(row["a_config_id"]),
                    str(row["b_config_id"]),
                )

            best = min(target_eligible, key=rank)
            selected_rows.append(
                {
                    "clean_fpr_constraint": constraint,
                    "selection_target": target,
                    "selection_value": best.get(target),
                    "selection_status": "selected",
                    **dict(best),
                }
            )

    selected_failure_rows: list[dict[str, Any]] = []
    unique_selected: dict[
        tuple[float, str, str, str, str],
        dict[str, Any],
    ] = {}
    for row in selected_rows:
        if row.get("selection_status") != "selected":
            continue
        key = (
            float(row["clean_fpr_constraint"]),
            str(row["detector_a_family"]),
            str(row["detector_b_family"]),
            str(row["a_config_id"]),
            str(row["b_config_id"]),
        )
        entry = unique_selected.setdefault(
            key,
            {
                "row": row,
                "targets": [],
            },
        )
        entry["targets"].append(str(row["selection_target"]))

    for (
        constraint,
        family_a,
        family_b,
        config_a_id,
        config_b_id,
    ), entry in unique_selected.items():
        event_a = event_by_config[config_a_id]
        event_b = event_by_config[config_b_id]
        failure_types = sorted(
            {
                str(row.get("failure_type") or "other")
                for row in event_a.values()
            }
        )
        for failure_type in failure_types:
            typed_ids = {
                event_id
                for event_id, row in event_a.items()
                if str(row.get("failure_type") or "other") == failure_type
            }
            for horizon, field in PAIRWISE_HORIZONS:
                detected_a = _detected_ids(event_a, field) & typed_ids
                detected_b = _detected_ids(event_b, field) & typed_ids
                overlap = detected_a & detected_b
                union = detected_a | detected_b
                selected_failure_rows.append(
                    {
                        "signal_mode": "fused",
                        "clean_fpr_constraint": constraint,
                        "selected_for": "|".join(sorted(entry["targets"])),
                        "detector_a_family": family_a,
                        "detector_b_family": family_b,
                        "a_config_id": config_a_id,
                        "b_config_id": config_b_id,
                        "failure_type": failure_type,
                        "horizon": horizon,
                        "event_n": len(typed_ids),
                        "a_detected_n": len(detected_a),
                        "b_detected_n": len(detected_b),
                        "overlap_n": len(overlap),
                        "a_only_n": len(detected_a - detected_b),
                        "b_only_n": len(detected_b - detected_a),
                        "or_detected_n": len(union),
                        "or_recall": (
                            len(union) / len(typed_ids)
                            if typed_ids
                            else None
                        ),
                        "tp_jaccard": (
                            len(overlap) / len(union)
                            if union
                            else None
                        ),
                    }
                )

    return sweep_rows, selected_rows, selected_failure_rows


def selected_unique_configs(
    best_rows: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    config_by_id = {
        str(config["config_id"]): dict(config)
        for config in configs
    }
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    for row in best_rows:
        if (
            row.get("selection_status")
            != "selected"
        ):
            continue
        config_id = str(row["config_id"])
        if (
            config_id not in seen
            and config_id in config_by_id
        ):
            seen.add(config_id)
            selected.append(
                config_by_id[config_id]
            )
    return selected


def build_recovery_rows(
    selected_configs: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    *,
    window_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recovery_events = [
        event
        for event in events
        if event.get("recovery_frame")
        is not None
    ]
    for config in selected_configs:
        for event in recovery_events:
            signal = signals.get(
                str(event["rollout_id"])
            )
            if signal is None:
                continue
            mask = detector_mask(
                signal["hops"], config
            )
            rows.append(
                {
                    **config_row(config),
                    **event,
                    **recovery_metrics(
                        signal["frames"],
                        signal["raw_hops"],
                        signal["hops"],
                        mask,
                        int(
                            event[
                                "recovery_frame"
                            ]
                        ),
                        window_samples=(
                            window_samples
                        ),
                    ),
                }
            )
    return rows


def task_cross_validation(
    configs: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    clean_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Tune on all other tasks and evaluate the selected config on one held-out task."""
    task_keys = sorted(
        {str(row["task_key"]) for row in event_rows}
        | {str(row["task_key"]) for row in clean_rows}
    )
    config_by_id = {
        str(config["config_id"]): config
        for config in configs
    }
    events_by_config: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    clean_by_config: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in event_rows:
        events_by_config[
            str(row["config_id"])
        ].append(row)
    for row in clean_rows:
        clean_by_config[
            str(row["config_id"])
        ].append(row)

    result: list[dict[str, Any]] = []
    for held_out in task_keys:
        train_summary: list[dict[str, Any]] = []
        for config in configs:
            config_id = str(config["config_id"])
            train_events = [
                row
                for row in events_by_config.get(config_id, [])
                if str(row["task_key"]) != held_out
            ]
            train_clean = [
                row
                for row in clean_by_config.get(config_id, [])
                if str(row["task_key"]) != held_out
            ]
            train_summary.append(
                {
                    **config_row(config),
                    **aggregate_event_metrics(train_events),
                    **aggregate_clean_metrics(train_clean),
                }
            )

        selected = select_best_configs(train_summary)
        for row in selected:
            base = {
                "held_out_task": held_out,
                "train_task_n": max(0, len(task_keys) - 1),
                "detector_family": row["detector_family"],
                "clean_fpr_constraint": row[
                    "clean_fpr_constraint"
                ],
                "selection_status": row["selection_status"],
            }
            if row["selection_status"] != "selected":
                result.append(base)
                continue

            config_id = str(row["config_id"])
            heldout_events = [
                item
                for item in events_by_config.get(config_id, [])
                if str(item["task_key"]) == held_out
            ]
            heldout_clean = [
                item
                for item in clean_by_config.get(config_id, [])
                if str(item["task_key"]) == held_out
            ]
            heldout = {
                **aggregate_event_metrics(heldout_events),
                **aggregate_clean_metrics(heldout_clean),
            }
            config = config_by_id[config_id]
            result.append(
                {
                    **base,
                    **config_row(config),
                    "train_event_n": row.get("event_n"),
                    "train_event_recall_at_3": row.get(
                        "event_recall_at_3"
                    ),
                    "train_median_delay_samples": row.get(
                        "median_delay_samples"
                    ),
                    "train_clean_rollout_n": row.get(
                        "clean_rollout_n"
                    ),
                    "train_clean_rollout_fpr": row.get(
                        "clean_rollout_fpr"
                    ),
                    **{
                        f"heldout_{key}": value
                        for key, value in heldout.items()
                    },
                }
            )
    return result

def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True, exist_ok=True
    )
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_tradeoff(
    summary_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    plt = _plt()
    fig, axis = plt.subplots(
        figsize=(8, 5), dpi=150
    )
    grouped: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    metric = (
        "grasp_recall_eventual"
        if any(
            row.get("grasp_recall_eventual") is not None
            for row in summary_rows
        )
        else "event_recall_at_3"
    )
    for row in summary_rows:
        if (
            row.get("clean_rollout_fpr") is not None
            and row.get(metric) is not None
        ):
            grouped[str(row["detector_family"])].append(row)

    for family, rows in sorted(grouped.items()):
        axis.scatter(
            [float(row["clean_rollout_fpr"]) for row in rows],
            [float(row[metric]) for row in rows],
            label=family,
            s=20,
            alpha=0.7,
        )
    for threshold in (
        CLEAN_FPR_CONSTRAINTS
    ):
        axis.axvline(
            threshold,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )
    axis.set_xlabel(
        "clean-rollout false-positive rate"
    )
    axis.set_ylabel(
        (
            "grasp-failure eventual recall"
            if metric == "grasp_recall_eventual"
            else "event recall @ 3 native samples"
        )
    )
    axis.set_title(
        "Fused-hop phenotype detector trade-off"
    )
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(
        parents=True, exist_ok=True
    )
    fig.savefig(output)
    plt.close(fig)


def heatmap(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_values: Sequence[Any],
    y_values: Sequence[Any],
    x_key: str,
    y_key: str,
    metric: str,
    title: str,
    output: Path,
) -> None:
    plt = _plt()
    index = {
        (row.get(y_key), row.get(x_key)): (
            row.get(metric)
        )
        for row in rows
    }
    matrix = [
        [
            (
                float(index[(y, x)])
                if index.get((y, x))
                is not None
                else math.nan
            )
            for x in x_values
        ]
        for y in y_values
    ]
    fig, axis = plt.subplots(
        figsize=(
            max(6, len(x_values) * 0.8),
            max(4, len(y_values) * 0.45),
        ),
        dpi=150,
    )
    image = axis.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_xticks(
        range(len(x_values))
    )
    axis.set_xticklabels(
        [str(value) for value in x_values]
    )
    axis.set_yticks(
        range(len(y_values))
    )
    axis.set_yticklabels(
        [str(value) for value in y_values]
    )
    axis.set_xlabel(x_key)
    axis.set_ylabel(y_key)
    axis.set_title(title)
    for row_index, values in enumerate(
        matrix
    ):
        for column_index, value in enumerate(
            values
        ):
            if math.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    fig.colorbar(
        image, ax=axis, label=metric
    )
    fig.tight_layout()
    output.parent.mkdir(
        parents=True, exist_ok=True
    )
    fig.savefig(output)
    plt.close(fig)


def plot_detector_heatmaps(
    summary_rows: Sequence[Mapping[str, Any]],
    plot_dir: Path,
) -> None:
    consecutive = [
        row
        for row in summary_rows
        if row["detector_family"]
        == "consecutive"
    ]
    for metric, suffix in (
        (
            "event_recall_at_3",
            "recall_at_3",
        ),
        (
            "clean_rollout_fpr",
            "clean_fpr",
        ),
    ):
        heatmap(
            consecutive,
            x_values=CONSECUTIVE_NS,
            y_values=EPSILONS,
            x_key="n",
            y_key="epsilon",
            metric=metric,
            title=(
                "Consecutive non-progress/regression: "
                + suffix
            ),
            output=(
                plot_dir
                / f"consecutive_{suffix}_heatmap.png"
            ),
        )

    for m in KOFM_MS:
        rows = [
            row
            for row in summary_rows
            if (
                row["detector_family"]
                == "k_of_m"
                and int(row["m"]) == m
            )
        ]
        ks = list(
            range(
                math.ceil(0.6 * m),
                m + 1,
            )
        )
        for metric, suffix in (
            (
                "event_recall_at_3",
                "recall_at_3",
            ),
            (
                "clean_rollout_fpr",
                "clean_fpr",
            ),
        ):
            heatmap(
                rows,
                x_values=EPSILONS,
                y_values=ks,
                x_key="epsilon",
                y_key="k",
                metric=metric,
                title=(
                    f"k-of-m, m={m}: "
                    + suffix
                ),
                output=(
                    plot_dir
                    / (
                        f"k_of_m_m{m}_"
                        f"{suffix}_heatmap.png"
                    )
                ),
            )


def plot_delay_distributions(
    best_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    selected: list[
        tuple[str, str]
    ] = []
    seen: set[str] = set()
    for row in best_rows:
        if (
            row.get("selection_status")
            != "selected"
        ):
            continue
        config_id = str(row["config_id"])
        if config_id in seen:
            continue
        seen.add(config_id)
        label = (
            f"{row['detector_family']} "
            f"≤{int(round(float(row['clean_fpr_constraint']) * 100))}%"
        )
        selected.append(
            (config_id, label)
        )

    pairs = []
    for config_id, label in selected:
        values = [
            float(row["delay_samples"])
            for row in event_rows
            if (
                str(row["config_id"])
                == config_id
                and row.get(
                    "delay_samples"
                )
                is not None
            )
        ]
        if values:
            pairs.append(
                (label, values)
            )
    if not pairs:
        return

    plt = _plt()
    fig, axis = plt.subplots(
        figsize=(
            max(8, len(pairs) * 0.65),
            5,
        ),
        dpi=150,
    )
    axis.boxplot(
        [values for _, values in pairs],
        labels=[
            label for label, _ in pairs
        ],
        showfliers=False,
    )
    axis.set_ylabel(
        "post-onset delay (native samples, 1-based)"
    )
    axis.set_title(
        "Detection-delay distributions for selected configs"
    )
    axis.tick_params(
        axis="x",
        rotation=60,
        labelsize=7,
    )
    axis.grid(
        axis="y", alpha=0.2
    )
    fig.tight_layout()
    output.parent.mkdir(
        parents=True, exist_ok=True
    )
    fig.savefig(output)
    plt.close(fig)


def _positive_spans(
    frames: Sequence[int],
    mask: Sequence[bool],
) -> list[tuple[float, float]]:
    if not frames:
        return []
    if len(frames) == 1:
        boundaries = [
            frames[0] - 0.5,
            frames[0] + 0.5,
        ]
    else:
        boundaries = [
            frames[0]
            - (
                frames[1] - frames[0]
            )
            / 2.0
        ]
        boundaries.extend(
            (left + right) / 2.0
            for left, right in zip(
                frames, frames[1:]
            )
        )
        boundaries.append(
            frames[-1]
            + (
                frames[-1]
                - frames[-2]
            )
            / 2.0
        )

    spans: list[
        tuple[float, float]
    ] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if (
            start is not None
            and (
                not value
                or index
                == len(mask) - 1
            )
        ):
            end = (
                index
                if (
                    value
                    and index
                    == len(mask) - 1
                )
                else index - 1
            )
            spans.append(
                (
                    boundaries[start],
                    boundaries[end + 1],
                )
            )
            start = None
    return spans


def selected_plot_configs(
    best_rows: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_family: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in best_rows:
        if (
            row.get("selection_status")
            == "selected"
        ):
            by_family[
                str(
                    row[
                        "detector_family"
                    ]
                )
            ].append(row)

    config_by_id = {
        str(config["config_id"]): dict(config)
        for config in configs
    }
    preference = {
        0.10: 0,
        0.05: 1,
        0.20: 2,
    }
    selected: list[dict[str, Any]] = []
    for family in sorted(by_family):
        row = min(
            by_family[family],
            key=lambda item: preference.get(
                float(
                    item[
                        "clean_fpr_constraint"
                    ]
                ),
                99,
            ),
        )
        config = config_by_id.get(
            str(row["config_id"])
        )
        if config:
            selected.append(config)
    return selected


def choose_representative_rollouts(
    explicit: Sequence[str],
    events: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    limit: int,
) -> list[str]:
    if explicit:
        unknown = [
            rollout_id
            for rollout_id in explicit
            if rollout_id not in signals
        ]
        if unknown:
            raise ValueError(
                "Representative rollout(s) "
                "have no usable hop signal: "
                f"{unknown}"
            )
        return list(
            dict.fromkeys(explicit)
        )[:limit]

    ordered: list[str] = []
    groups = [
        [
            event
            for event in events
            if event.get(
                "recovery_frame"
            )
            is not None
        ],
        [
            event
            for event in events
            if event.get("outcome")
            == "terminal_failure"
        ],
        list(events),
    ]
    for group in groups:
        for event in sorted(
            group,
            key=lambda row: (
                str(
                    row[
                        "failure_type"
                    ]
                ),
                str(
                    row["rollout_id"]
                ),
                int(
                    row["event_index"]
                ),
            ),
        ):
            rollout_id = str(
                event["rollout_id"]
            )
            if (
                rollout_id in signals
                and rollout_id
                not in ordered
            ):
                ordered.append(
                    rollout_id
                )
            if len(ordered) >= limit:
                return ordered
    return ordered


def plot_representative_rollouts(
    configs: Sequence[Mapping[str, Any]],
    rollout_ids: Sequence[str],
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    plot_dir: Path,
    signal_mode: str = "incremental",
) -> None:
    events_by_rollout: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    for event in events:
        events_by_rollout[
            str(event["rollout_id"])
        ].append(event)

    plt = _plt()
    for rollout_id in rollout_ids:
        signal = signals[rollout_id]
        frames = signal["frames"]
        hops = signal["hops"]
        rollout_events = sorted(
            events_by_rollout.get(
                rollout_id, []
            ),
            key=lambda row: int(
                row["event_index"]
            ),
        )

        for config in configs:
            mask = detector_mask(
                hops, config
            )
            fig, axis = plt.subplots(
                figsize=(10, 4.5),
                dpi=150,
            )
            axis.plot(
                frames,
                hops,
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=f"{signal_mode} hop",
            )
            axis.axhline(
                0.0,
                linewidth=1.0,
                linestyle="--",
                label="zero hop",
            )
            for left, right in _positive_spans(
                frames, mask
            ):
                axis.axvspan(
                    left,
                    right,
                    alpha=0.15,
                )

            for event in rollout_events:
                axis.axvline(
                    int(
                        event[
                            "observable_onset_frame"
                        ]
                    ),
                    linestyle="-",
                    linewidth=1.0,
                    label="observable onset",
                )
                if (
                    event.get(
                        "recovery_frame"
                    )
                    is not None
                ):
                    axis.axvline(
                        int(
                            event[
                                "recovery_frame"
                            ]
                        ),
                        linestyle=":",
                        linewidth=1.2,
                        label="recovery",
                    )

            axis.set_xlabel(
                "video frame index "
                "(native Robo-Dopamine samples)"
            )
            axis.set_ylabel(
                (
                    "incremental hop (normalized [-1,1])"
                    if signal_mode == "incremental"
                    else f"{signal_mode} hop (saved native scale)"
                )
            )
            axis.set_title(
                f"{rollout_id} · {signal_mode} · "
                f"{config['detector_family']} · "
                f"{config['parameters_json']}"
            )
            handles, labels = (
                axis.get_legend_handles_labels()
            )
            unique: dict[str, Any] = {}
            for handle, label in zip(
                handles, labels
            ):
                unique.setdefault(
                    label, handle
                )
            axis.legend(
                unique.values(),
                unique.keys(),
                fontsize=8,
                loc="best",
            )
            axis.grid(alpha=0.2)
            fig.tight_layout()

            output = (
                plot_dir
                / "representative"
                / (
                    f"{rollout_id}__"
                    f"{config['config_id']}.png"
                )
            )
            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            fig.savefig(output)
            plt.close(fig)
