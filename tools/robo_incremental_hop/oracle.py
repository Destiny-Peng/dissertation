"""FPR-unconstrained oracle detectability for fused-hop phenotype detectors."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .core import RECALL_SAMPLE_WINDOWS, config_row, detector_mask


ORACLE_EARLY_TOLERANCE_SAMPLES = 1
ORACLE_HORIZONS: tuple[str, ...] = ("1", "3", "5", "10", "20", "eventual")


def _event_identity(event: Mapping[str, Any]) -> str:
    event_id = event.get("event_id")
    if event_id not in (None, ""):
        return str(event_id)
    return f"{event.get('rollout_id')}::event{int(event.get('event_index') or 0)}"


def _positive_episode_starts(mask: Sequence[bool]) -> list[int]:
    return [
        index
        for index, value in enumerate(mask)
        if bool(value) and (index == 0 or not bool(mask[index - 1]))
    ]


def _percentile(values: Sequence[float | int], q: float) -> float | None:
    data = sorted(float(value) for value in values)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return data[low]
    weight = position - low
    return data[low] * (1.0 - weight) + data[high] * weight


def _event_localization(
    frames: Sequence[int],
    mask: Sequence[bool],
    event: Mapping[str, Any],
    *,
    early_tolerance_samples: int,
) -> dict[str, Any]:
    onset = int(event["observable_onset_frame"])
    end_value = event.get("episode_end_frame")
    end = int(end_value) if end_value is not None else None
    eligible = [
        index
        for index, frame in enumerate(frames)
        if frame >= onset and (end is None or frame < end)
    ]
    anchor = eligible[0] if eligible else None
    result: dict[str, Any] = {
        "oracle_detectable": False,
        "oracle_detectable_strict": False,
        "best_episode_start_index": None,
        "best_alarm_frame": None,
        "best_start_offset_samples": None,
        "best_delay_samples": None,
        "best_delay_frames": None,
    }
    for window in RECALL_SAMPLE_WINDOWS:
        result[f"recall_at_{window}"] = False
    result["eventual_recall"] = False
    if anchor is None:
        return result

    starts = [
        index
        for index in _positive_episode_starts(mask)
        if end is None or int(frames[index]) < end
    ]
    strict = [index for index in starts if index >= anchor]
    tolerant = [
        index
        for index in starts
        if index >= max(0, anchor - early_tolerance_samples)
    ]
    result["oracle_detectable_strict"] = bool(strict)
    if not tolerant:
        return result

    start = min(tolerant)
    offset = start - anchor
    # Preserve the existing Recall@d convention: the onset-anchor sample is
    # delay 1. A one-native-sample early-tolerance alarm therefore has delay 0.
    delay_samples = offset + 1
    result.update(
        {
            "oracle_detectable": True,
            "best_episode_start_index": start,
            "best_alarm_frame": int(frames[start]),
            "best_start_offset_samples": offset,
            "best_delay_samples": delay_samples,
            "best_delay_frames": int(frames[start]) - onset,
            "eventual_recall": True,
        }
    )
    for window in RECALL_SAMPLE_WINDOWS:
        result[f"recall_at_{window}"] = delay_samples <= window
    return result


def _rollout_from_start(
    frames: Sequence[int],
    mask: Sequence[bool],
) -> dict[str, Any]:
    start = next(
        (index for index in _positive_episode_starts(mask)),
        None,
    )
    result: dict[str, Any] = {
        "oracle_detectable": start is not None,
        "oracle_detectable_strict": start is not None,
        "best_episode_start_index": start,
        "best_alarm_frame": int(frames[start]) if start is not None else None,
        "best_start_offset_samples": start if start is not None else None,
        "best_delay_samples": start + 1 if start is not None else None,
        "best_delay_frames": int(frames[start]) if start is not None else None,
        "eventual_recall": start is not None,
    }
    for window in RECALL_SAMPLE_WINDOWS:
        result[f"recall_at_{window}"] = bool(
            start is not None and start + 1 <= window
        )
    return result


def _family_groups(
    configs: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    stagnation = [
        config
        for config in configs
        if str(config.get("detector_family") or "").startswith("stagnation_")
    ]
    regression = [
        config
        for config in configs
        if str(config.get("detector_family") or "") == "regression_window_min"
    ]
    return {
        "stagnation": stagnation,
        "regression": regression,
        "combined": [*stagnation, *regression],
    }


def _target_key(horizon: str) -> str:
    return "eventual_recall" if horizon == "eventual" else f"recall_at_{horizon}"


def _bitset(
    ids: Sequence[str],
    details: Mapping[str, Mapping[str, Any]],
    field: str,
) -> int:
    result = 0
    for index, item_id in enumerate(ids):
        if bool(details[item_id].get(field)):
            result |= 1 << index
    return result


def _median_delay_for_mask(
    ids: Sequence[str],
    mask_bits: int,
    details_a: Mapping[str, Mapping[str, Any]],
    details_b: Mapping[str, Mapping[str, Any]] | None,
    *,
    event_population: bool,
) -> float:
    values: list[float] = []
    key = "best_start_offset_samples" if event_population else "best_delay_samples"
    for index, item_id in enumerate(ids):
        if not (mask_bits & (1 << index)):
            continue
        candidates = []
        for details in (details_a, details_b):
            if details is None:
                continue
            value = details[item_id].get(key)
            if value is not None and bool(details[item_id].get("oracle_detectable")):
                candidates.append(float(value))
        if candidates:
            values.append(min(candidates))
    return float(statistics.median(values)) if values else math.inf


def _profile_for_choice(
    ids: Sequence[str],
    details_a: Mapping[str, Mapping[str, Any]],
    details_b: Mapping[str, Mapping[str, Any]] | None,
    *,
    event_population: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    n = len(ids)
    for horizon in ORACLE_HORIZONS:
        field = _target_key(horizon)
        detected = 0
        for item_id in ids:
            hit = bool(details_a[item_id].get(field))
            if details_b is not None:
                hit = hit or bool(details_b[item_id].get(field))
            detected += int(hit)
        key = "eventual" if horizon == "eventual" else f"at_{horizon}"
        result[f"recall_{key}"] = detected / n if n else None
        result[f"detected_{key}_n"] = detected

    delay_key = "best_start_offset_samples" if event_population else "best_delay_samples"
    delays = []
    for item_id in ids:
        candidates = []
        for details in (details_a, details_b):
            if details is None:
                continue
            if not bool(details[item_id].get("oracle_detectable")):
                continue
            value = details[item_id].get(delay_key)
            if value is not None:
                candidates.append(float(value))
        if candidates:
            delays.append(min(candidates))
    result["median_detection_delay_samples"] = (
        float(statistics.median(delays)) if delays else None
    )
    return result


def _population_specs(
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    event_ids = [_event_identity(event) for event in events]
    by_id = {_event_identity(event): event for event in events}
    grasp = [
        event_id
        for event_id in event_ids
        if str(by_id[event_id].get("failure_type") or "") == "grasp_failure"
    ]
    non_grasp = [event_id for event_id in event_ids if event_id not in set(grasp)]
    specs = [
        {"population": "all_annotated_events", "kind": "event", "ids": event_ids},
        {"population": "grasp_failure", "kind": "event", "ids": grasp},
        {"population": "non_grasp_events", "kind": "event", "ids": non_grasp},
    ]
    failure_types = sorted(
        {
            str(event.get("failure_type") or "other")
            for event in events
            if str(event.get("failure_type") or "other") != "grasp_failure"
        }
    )
    for failure_type in failure_types:
        specs.append(
            {
                "population": f"failure_type:{failure_type}",
                "kind": "event",
                "ids": [
                    event_id
                    for event_id in event_ids
                    if str(by_id[event_id].get("failure_type") or "other")
                    == failure_type
                ],
            }
        )
    specs.append(
        {
            "population": "no_event_failure_rollouts",
            "kind": "no_event",
            "ids": [
                str(row["rollout_id"])
                for row in no_event_failures
            ],
        }
    )
    return [spec for spec in specs if spec["ids"]]


def build_oracle_analysis(
    configs: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
    *,
    early_tolerance_samples: int = ORACLE_EARLY_TOLERANCE_SAMPLES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if early_tolerance_samples < 0 or early_tolerance_samples > 1:
        raise ValueError("oracle early tolerance must be 0 or 1 native sample")

    groups = _family_groups(configs)
    config_by_id = {str(config["config_id"]): config for config in configs}
    event_ids = [_event_identity(event) for event in events]
    event_by_id = {_event_identity(event): event for event in events}
    no_event_ids = [str(row["rollout_id"]) for row in no_event_failures]

    event_details: dict[str, dict[str, dict[str, Any]]] = {}
    no_event_details: dict[str, dict[str, dict[str, Any]]] = {}
    clean_positive: dict[str, set[str]] = {}

    for config in configs:
        config_id = str(config["config_id"])
        masks = {
            rollout_id: detector_mask(signal["hops"], config)
            for rollout_id, signal in signals.items()
        }
        event_details[config_id] = {}
        for event_id in event_ids:
            event = event_by_id[event_id]
            rollout_id = str(event["rollout_id"])
            signal = signals.get(rollout_id)
            if signal is None:
                continue
            event_details[config_id][event_id] = _event_localization(
                signal["frames"],
                masks[rollout_id],
                event,
                early_tolerance_samples=early_tolerance_samples,
            )
        no_event_details[config_id] = {}
        for rollout_id in no_event_ids:
            signal = signals.get(rollout_id)
            if signal is None:
                continue
            no_event_details[config_id][rollout_id] = _rollout_from_start(
                signal["frames"],
                masks[rollout_id],
            )
        clean_positive[config_id] = {
            str(row["rollout_id"])
            for row in clean_rollouts
            if str(row["rollout_id"]) in masks
            and any(masks[str(row["rollout_id"])])
        }

    populations = _population_specs(events, no_event_failures)

    def detail_map(config_id: str, kind: str) -> Mapping[str, Mapping[str, Any]]:
        return event_details[config_id] if kind == "event" else no_event_details[config_id]

    global_rows: list[dict[str, Any]] = []
    for signal_family in ("stagnation", "regression", "combined"):
        configs_group = groups[signal_family]
        if not configs_group:
            continue
        first_config_id = str(configs_group[0]["config_id"])
        first_details = detail_map(first_config_id, "event")
        first_no_event = detail_map(first_config_id, "no_event")
        for spec in populations:
            available = first_details if spec["kind"] == "event" else first_no_event
            ids = [item_id for item_id in spec["ids"] if item_id in available]
            if not ids:
                continue
            event_population = spec["kind"] == "event"

            for horizon in ORACLE_HORIZONS:
                field = _target_key(horizon)
                if signal_family != "combined":
                    best: tuple[Any, ...] | None = None
                    best_config: Mapping[str, Any] | None = None
                    best_bits = 0
                    for config in configs_group:
                        config_id = str(config["config_id"])
                        details = detail_map(config_id, spec["kind"])
                        bits = _bitset(ids, details, field)
                        rank = (
                            -bits.bit_count(),
                            _median_delay_for_mask(
                                ids,
                                bits,
                                details,
                                None,
                                event_population=event_population,
                            ),
                            config_id,
                        )
                        if best is None or rank < best:
                            best = rank
                            best_config = config
                            best_bits = bits
                    if best_config is None:
                        continue
                    config_id = str(best_config["config_id"])
                    details = detail_map(config_id, spec["kind"])
                    profile = _profile_for_choice(
                        ids,
                        details,
                        None,
                        event_population=event_population,
                    )
                    clean_n = len(clean_rollouts)
                    global_rows.append(
                        {
                            "signal_family": signal_family,
                            "population": spec["population"],
                            "population_n": len(ids),
                            "selection_target": (
                                "eventual" if horizon == "eventual" else f"recall_at_{horizon}"
                            ),
                            "selected_detected_n": best_bits.bit_count(),
                            "selected_recall": (
                                best_bits.bit_count() / len(ids)
                                if ids
                                else None
                            ),
                            "rmax_recall": (
                                best_bits.bit_count() / len(ids)
                                if ids
                                else None
                            ),
                            "clean_fpr_constraint_applied": False,
                            **config_row(best_config),
                            **profile,
                            "clean_rollout_fpr_ignored": (
                                len(clean_positive[config_id]) / clean_n
                                if clean_n
                                else None
                            ),
                        }
                    )
                    continue

                stagnation = groups["stagnation"]
                regression = groups["regression"]
                stagnation_states: dict[int, tuple[Mapping[str, Any], float]] = {}
                regression_states: dict[int, tuple[Mapping[str, Any], float]] = {}
                for family_configs, states in (
                    (stagnation, stagnation_states),
                    (regression, regression_states),
                ):
                    for config in family_configs:
                        config_id = str(config["config_id"])
                        details = detail_map(config_id, spec["kind"])
                        bits = _bitset(ids, details, field)
                        delay = _median_delay_for_mask(
                            ids,
                            bits,
                            details,
                            None,
                            event_population=event_population,
                        )
                        current = states.get(bits)
                        if current is None or (delay, config_id) < (
                            current[1],
                            str(current[0]["config_id"]),
                        ):
                            states[bits] = (config, delay)

                best_pair: tuple[Any, ...] | None = None
                chosen_a: Mapping[str, Any] | None = None
                chosen_b: Mapping[str, Any] | None = None
                chosen_bits = 0
                for bits_a, (config_a, _) in stagnation_states.items():
                    details_a = detail_map(str(config_a["config_id"]), spec["kind"])
                    for bits_b, (config_b, _) in regression_states.items():
                        union = bits_a | bits_b
                        details_b = detail_map(str(config_b["config_id"]), spec["kind"])
                        delay = _median_delay_for_mask(
                            ids,
                            union,
                            details_a,
                            details_b,
                            event_population=event_population,
                        )
                        rank = (
                            -union.bit_count(),
                            delay,
                            str(config_a["config_id"]),
                            str(config_b["config_id"]),
                        )
                        if best_pair is None or rank < best_pair:
                            best_pair = rank
                            chosen_a = config_a
                            chosen_b = config_b
                            chosen_bits = union
                if chosen_a is None or chosen_b is None:
                    continue
                a_id = str(chosen_a["config_id"])
                b_id = str(chosen_b["config_id"])
                details_a = detail_map(a_id, spec["kind"])
                details_b = detail_map(b_id, spec["kind"])
                profile = _profile_for_choice(
                    ids,
                    details_a,
                    details_b,
                    event_population=event_population,
                )
                clean_union = clean_positive[a_id] | clean_positive[b_id]
                clean_n = len(clean_rollouts)
                global_rows.append(
                    {
                        "signal_family": "combined",
                        "population": spec["population"],
                        "population_n": len(ids),
                        "selection_target": (
                            "eventual" if horizon == "eventual" else f"recall_at_{horizon}"
                        ),
                        "selected_detected_n": chosen_bits.bit_count(),
                        "selected_recall": (
                            chosen_bits.bit_count() / len(ids)
                            if ids
                            else None
                        ),
                        "rmax_recall": (
                            chosen_bits.bit_count() / len(ids)
                            if ids
                            else None
                        ),
                        "clean_fpr_constraint_applied": False,
                        "stagnation_config_id": a_id,
                        "stagnation_family": chosen_a["detector_family"],
                        "stagnation_parameters_json": chosen_a["parameters_json"],
                        "regression_config_id": b_id,
                        "regression_family": chosen_b["detector_family"],
                        "regression_parameters_json": chosen_b["parameters_json"],
                        **profile,
                        "clean_rollout_fpr_ignored": (
                            len(clean_union) / clean_n if clean_n else None
                        ),
                    }
                )

    detectability_rows: list[dict[str, Any]] = []
    for signal_family in ("stagnation", "regression", "combined"):
        configs_group = groups[signal_family]
        for event_id in event_ids:
            event = event_by_id[event_id]
            candidates = []
            strict = False
            for config in configs_group:
                config_id = str(config["config_id"])
                detail = event_details.get(config_id, {}).get(event_id)
                if detail is None:
                    continue
                strict = strict or bool(detail.get("oracle_detectable_strict"))
                if bool(detail.get("oracle_detectable")):
                    candidates.append((config, detail))
            best = min(
                candidates,
                key=lambda pair: (
                    float(pair[1]["best_start_offset_samples"]),
                    str(pair[0]["config_id"]),
                ),
                default=None,
            )
            row = {
                "signal_family": signal_family,
                "record_kind": "annotated_event",
                "event_id": event_id,
                "rollout_id": event["rollout_id"],
                "event_index": event.get("event_index"),
                "failure_type": event.get("failure_type"),
                "outcome": event.get("outcome"),
                "observable_onset_frame": event.get("observable_onset_frame"),
                "oracle_detectable": best is not None,
                "oracle_detectable_strict": strict,
                "early_tolerance_samples": early_tolerance_samples,
            }
            if best is not None:
                config, detail = best
                row.update(
                    {
                        "best_config_id": config["config_id"],
                        "best_detector_family": config["detector_family"],
                        "best_parameters_json": config["parameters_json"],
                        **detail,
                    }
                )
            else:
                for window in RECALL_SAMPLE_WINDOWS:
                    row[f"recall_at_{window}"] = False
                row["eventual_recall"] = False
            detectability_rows.append(row)

        for rollout_id in no_event_ids:
            candidates = []
            for config in configs_group:
                config_id = str(config["config_id"])
                detail = no_event_details.get(config_id, {}).get(rollout_id)
                if detail is not None and bool(detail.get("oracle_detectable")):
                    candidates.append((config, detail))
            best = min(
                candidates,
                key=lambda pair: (
                    float(pair[1]["best_delay_samples"]),
                    str(pair[0]["config_id"]),
                ),
                default=None,
            )
            row = {
                "signal_family": signal_family,
                "record_kind": "no_event_failure_rollout",
                "event_id": None,
                "rollout_id": rollout_id,
                "event_index": None,
                "failure_type": "no_event_failure",
                "outcome": "terminal_failure",
                "observable_onset_frame": None,
                "oracle_detectable": best is not None,
                "oracle_detectable_strict": best is not None,
                "early_tolerance_samples": 0,
            }
            if best is not None:
                config, detail = best
                row.update(
                    {
                        "best_config_id": config["config_id"],
                        "best_detector_family": config["detector_family"],
                        "best_parameters_json": config["parameters_json"],
                        **detail,
                    }
                )
            else:
                for window in RECALL_SAMPLE_WINDOWS:
                    row[f"recall_at_{window}"] = False
                row["eventual_recall"] = False
            detectability_rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    for signal_family in ("stagnation", "regression", "combined"):
        family_rows = [
            row for row in detectability_rows
            if row["signal_family"] == signal_family
        ]
        population_rows: list[tuple[str, list[Mapping[str, Any]]]] = [
            (
                "all_annotated_events",
                [row for row in family_rows if row["record_kind"] == "annotated_event"],
            ),
            (
                "grasp_failure",
                [
                    row for row in family_rows
                    if row["record_kind"] == "annotated_event"
                    and str(row.get("failure_type") or "") == "grasp_failure"
                ],
            ),
            (
                "non_grasp_events",
                [
                    row for row in family_rows
                    if row["record_kind"] == "annotated_event"
                    and str(row.get("failure_type") or "") != "grasp_failure"
                ],
            ),
            (
                "no_event_failure_rollouts",
                [
                    row for row in family_rows
                    if row["record_kind"] == "no_event_failure_rollout"
                ],
            ),
        ]
        other_types = sorted(
            {
                str(row.get("failure_type") or "other")
                for row in family_rows
                if row["record_kind"] == "annotated_event"
                and str(row.get("failure_type") or "") != "grasp_failure"
            }
        )
        for failure_type in other_types:
            population_rows.append(
                (
                    f"failure_type:{failure_type}",
                    [
                        row for row in family_rows
                        if row["record_kind"] == "annotated_event"
                        and str(row.get("failure_type") or "other") == failure_type
                    ],
                )
            )

        for population, rows in population_rows:
            if not rows:
                continue
            detected = [row for row in rows if bool(row.get("oracle_detectable"))]
            offset_values = [
                float(row["best_start_offset_samples"])
                for row in detected
                if row.get("best_start_offset_samples") is not None
            ]
            delay_values = [
                float(row["best_delay_samples"])
                for row in detected
                if row.get("best_delay_samples") is not None
            ]
            summary = {
                "signal_family": signal_family,
                "population": population,
                "population_n": len(rows),
                "oracle_detected_n": len(detected),
                "oracle_recall_eventual": len(detected) / len(rows),
                "strict_detected_n": sum(
                    bool(row.get("oracle_detectable_strict")) for row in rows
                ),
                "strict_recall_eventual": sum(
                    bool(row.get("oracle_detectable_strict")) for row in rows
                ) / len(rows),
                "median_best_delay_samples": (
                    float(statistics.median(delay_values))
                    if delay_values else None
                ),
                "p25_best_delay_samples": _percentile(delay_values, 0.25),
                "p75_best_delay_samples": _percentile(delay_values, 0.75),
                "median_best_start_offset_samples": (
                    float(statistics.median(offset_values))
                    if offset_values else None
                ),
                "p25_best_start_offset_samples": _percentile(offset_values, 0.25),
                "p75_best_start_offset_samples": _percentile(offset_values, 0.75),
                "early_tolerance_samples": (
                    early_tolerance_samples
                    if population != "no_event_failure_rollouts"
                    else 0
                ),
            }
            for window in RECALL_SAMPLE_WINDOWS:
                summary[f"oracle_recall_at_{window}"] = (
                    sum(bool(row.get(f"recall_at_{window}")) for row in rows)
                    / len(rows)
                )
                summary[f"oracle_offset_le_{window}_fraction"] = (
                    sum(
                        row.get("best_start_offset_samples") is not None
                        and float(row["best_start_offset_samples"]) <= window
                        for row in rows
                    )
                    / len(rows)
                )
            summary_rows.append(summary)

    return global_rows, detectability_rows, summary_rows
