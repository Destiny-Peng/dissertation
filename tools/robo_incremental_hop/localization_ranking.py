"""Offline change-point selection over the existing fused-hop detector sweep."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .core import CONFIG_FIELDS, detector_mask

INTERVAL_LOCALIZATION_WINDOWS = (1, 3, 5)
CHANGE_POINT_WINDOWS = (2, 3, 5)


def _interval_population_specs(
    event_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    events: dict[str, dict[str, Any]] = {}
    for row in event_rows:
        if str(row.get("outcome") or "") != "terminal_failure":
            continue
        causal = row.get("causal_onset_frame")
        observable = row.get("observable_onset_frame")
        if causal is None or observable is None:
            continue
        causal_frame = int(causal)
        observable_frame = int(observable)
        if causal_frame > observable_frame:
            continue
        event_id = str(
            row.get("event_id")
            or f"{row.get('rollout_id')}::event{int(row.get('event_index') or 0)}"
        )
        events.setdefault(
            event_id,
            {
                "event_id": event_id,
                "rollout_id": str(row["rollout_id"]),
                "event_index": int(row.get("event_index") or 0),
                "failure_type": str(row.get("failure_type") or ""),
                "causal_onset_frame": causal_frame,
                "observable_onset_frame": observable_frame,
            },
        )

    all_events = sorted(
        events.values(),
        key=lambda row: (
            row["rollout_id"],
            row["causal_onset_frame"],
            row["observable_onset_frame"],
            row["event_index"],
        ),
    )
    by_rollout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in all_events:
        by_rollout[event["rollout_id"]].append(event)
    first_events = [
        min(
            rollout_events,
            key=lambda row: (
                row["causal_onset_frame"],
                row["observable_onset_frame"],
                row["event_index"],
            ),
        )
        for rollout_events in by_rollout.values()
    ]
    first_ids = {row["event_id"] for row in first_events}
    for event in all_events:
        event["is_first_eligible_event_per_failed_rollout"] = (
            event["event_id"] in first_ids
        )
    grasp_events = [
        event
        for event in all_events
        if event["failure_type"] == "grasp_failure"
    ]
    return [
        ("first_eligible_event_per_failed_rollout", first_events),
        ("all_eligible_failure_events", all_events),
        ("grasp_failure", grasp_events),
    ]


def _onset_anchor(frames: Sequence[int], onset_frame: int) -> int | None:
    return next(
        (
            index
            for index, frame in enumerate(frames)
            if int(frame) >= int(onset_frame)
        ),
        None,
    )


def _interval_error_samples(
    *,
    trigger_index: int,
    causal_index: int,
    observable_index: int,
) -> int:
    if trigger_index < causal_index:
        return trigger_index - causal_index
    if trigger_index > observable_index:
        return trigger_index - observable_index
    return 0


def _positive_episode_starts(mask: Sequence[bool]) -> list[int]:
    return [
        index
        for index, positive in enumerate(mask)
        if bool(positive) and (index == 0 or not bool(mask[index - 1]))
    ]


def _change_score(
    hops: Sequence[float],
    start: int,
    left: int,
    right: int,
) -> float | None:
    """Score one positive-episode start on the native sample grid.

    The requested definition uses hop[s-L:s] and hop[s:s+R]. Candidates without
    a complete left or right window are kept in diagnostics but are not eligible
    for max-score selection.
    """
    if start - left < 0 or start + right > len(hops):
        return None
    before = [float(value) for value in hops[start - left : start]]
    after = [float(value) for value in hops[start : start + right]]
    if not before or not after:
        return None
    return float(statistics.median(before) - statistics.median(after))


def _earliest_global_progress_argmax(progress: Sequence[float]) -> int | None:
    if not progress:
        return None
    maximum = max(float(value) for value in progress)
    return next(
        index
        for index, value in enumerate(progress)
        if float(value) == maximum
    )


def _config_metadata(
    event_rows: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for row in configs or ():
        config_id = str(row.get("config_id") or "")
        if not config_id:
            continue
        catalog[config_id] = {
            field: row.get(field)
            for field in CONFIG_FIELDS
        }
    for row in event_rows:
        config_id = str(row.get("config_id") or "")
        if not config_id or config_id in catalog:
            continue
        catalog[config_id] = {
            field: row.get(field)
            for field in CONFIG_FIELDS
        }
    return catalog


def _pair_candidates(
    ensemble_sweep: Sequence[Mapping[str, Any]],
    config_meta: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand existing OR family rules across all existing detector configs.

    The historical ensemble_sweep was produced after a clean-FPR feasibility
    filter. Reusing its rows verbatim would silently retain that constraint.
    Instead, use it only to recover the already-existing family-combination
    rules, then form the Cartesian product of the already-existing configs in
    those families. No threshold/config/family is created here.
    """
    family_pairs: set[tuple[str, str]] = set()
    for row in ensemble_sweep:
        family_a = str(row.get("detector_a_family") or "")
        family_b = str(row.get("detector_b_family") or "")
        if not family_a or not family_b:
            a = config_meta.get(str(row.get("a_config_id") or ""), {})
            b = config_meta.get(str(row.get("b_config_id") or ""), {})
            family_a = str(a.get("detector_family") or "")
            family_b = str(b.get("detector_family") or "")
        if family_a and family_b:
            family_pairs.add((family_a, family_b))

    by_family: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for config_id, meta in config_meta.items():
        family = str(meta.get("detector_family") or "")
        if family:
            by_family[family].append((config_id, meta))
    for rows in by_family.values():
        rows.sort(key=lambda item: item[0])

    result: list[dict[str, Any]] = []
    for family_a, family_b in sorted(family_pairs):
        for a_id, a in by_family.get(family_a, []):
            for b_id, b in by_family.get(family_b, []):
                result.append(
                    {
                        "config_id": f"OR:{a_id}|{b_id}",
                        "detector_family": "phenotype_or",
                        "parameters_json": None,
                        "a_config_id": a_id,
                        "b_config_id": b_id,
                        "a_detector_family": family_a,
                        "b_detector_family": family_b,
                        "a_parameters_json": a.get("parameters_json"),
                        "b_parameters_json": b.get("parameters_json"),
                    }
                )
    return result


def _evaluate_predictions(
    *,
    candidate: Mapping[str, Any],
    selector: str,
    left_window: int | None,
    right_window: int | None,
    predictions: Mapping[str, int | None],
    signals: Mapping[str, Mapping[str, Any]],
    population: str,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    event_errors: list[int | None] = []
    errors: list[int] = []
    for event in events:
        rollout_id = str(event["rollout_id"])
        signal = signals.get(rollout_id)
        if signal is None:
            continue
        frames = [int(value) for value in signal["frames"]]
        causal_index = _onset_anchor(frames, int(event["causal_onset_frame"]))
        observable_index = _onset_anchor(
            frames, int(event["observable_onset_frame"])
        )
        if causal_index is None or observable_index is None:
            continue
        prediction = predictions.get(rollout_id)
        error = (
            _interval_error_samples(
                trigger_index=int(prediction),
                causal_index=causal_index,
                observable_index=observable_index,
            )
            if prediction is not None
            else None
        )
        event_errors.append(error)
        if error is not None:
            errors.append(error)

    n = len(event_errors)
    triggered_n = len(errors)
    absolute = [abs(value) for value in errors]
    squared = [value * value for value in errors]
    result: dict[str, Any] = {
        **dict(candidate),
        "selector": selector,
        "L": left_window,
        "R": right_window,
        "population": population,
        "eligible_event_n": n,
        "triggered_n": triggered_n,
        "no_trigger_n": n - triggered_n,
        "trigger_coverage": triggered_n / n if n else None,
        "in_interval_n": sum(value == 0 for value in errors),
        "in_interval_rate": (
            sum(value == 0 for value in errors) / n if n else None
        ),
        "before_interval_n": sum(value < 0 for value in errors),
        "before_interval_rate": (
            sum(value < 0 for value in errors) / n if n else None
        ),
        "after_interval_n": sum(value > 0 for value in errors),
        "after_interval_rate": (
            sum(value > 0 for value in errors) / n if n else None
        ),
        "median_signed_interval_error_samples": (
            float(statistics.median(errors)) if errors else None
        ),
        "median_absolute_interval_error_samples": (
            float(statistics.median(absolute)) if absolute else None
        ),
        "mae_samples": sum(absolute) / len(absolute) if absolute else None,
        "mse_samples": sum(squared) / len(squared) if squared else None,
    }
    for window in INTERVAL_LOCALIZATION_WINDOWS:
        hit_n = sum(
            error is not None and abs(error) <= window
            for error in event_errors
        )
        result[f"within_{window}_n"] = hit_n
        result[f"within_{window}"] = hit_n / n if n else None
    return result


def _add_interval_ranks(rows: list[dict[str, Any]]) -> None:
    groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        groups[(str(row["population"]), str(row["selector"]))].append(row)

    for group_rows in groups.values():
        for row in group_rows:
            row["rank_mse"] = None
            row["rank_mae"] = None
            row["rank_median_abs_error"] = None
        for metric, rank_field in (
            ("mse_samples", "rank_mse"),
            ("mae_samples", "rank_mae"),
            (
                "median_absolute_interval_error_samples",
                "rank_median_abs_error",
            ),
        ):
            ordered = sorted(
                [row for row in group_rows if row.get(metric) is not None],
                key=lambda row: (
                    float(row[metric]),
                    -float(row.get("trigger_coverage") or 0.0),
                    int(row.get("L") or 0),
                    int(row.get("R") or 0),
                    str(row.get("config_id") or ""),
                ),
            )
            for rank, row in enumerate(ordered, 1):
                row[rank_field] = rank


def offline_change_point_localization(
    *,
    event_rows: Sequence[Mapping[str, Any]],
    ensemble_sweep: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select max-score positive episodes and compare against two baselines.

    Detector masks are recomputed from the already-loaded fused-hop signal using
    the existing detector configs. This does not run detector search and does not
    invoke Robo-Dopamine.
    """
    populations = _interval_population_specs(event_rows)
    all_events = next(
        events
        for population, events in populations
        if population == "all_eligible_failure_events"
    )
    config_meta = _config_metadata(event_rows, configs)
    detector_candidates = [
        dict(meta)
        for _, meta in sorted(config_meta.items())
    ]
    detector_candidates.extend(_pair_candidates(ensemble_sweep, config_meta))

    global_predictions = {
        rollout_id: _earliest_global_progress_argmax(signal.get("progress") or [])
        for rollout_id, signal in signals.items()
    }

    ranking: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    global_candidate = {
        "config_id": "__global_fused_progress_argmax__",
        "detector_family": "global_fused_progress_argmax",
        "parameters_json": None,
    }
    for population, events in populations:
        ranking.append(
            _evaluate_predictions(
                candidate=global_candidate,
                selector="global_fused_progress_argmax",
                left_window=None,
                right_window=None,
                predictions=global_predictions,
                signals=signals,
                population=population,
                events=events,
            )
        )

    for candidate in detector_candidates:
        config_id = str(candidate["config_id"])
        pair = str(candidate.get("detector_family") or "") == "phenotype_or"
        a_id = str(candidate.get("a_config_id") or "")
        b_id = str(candidate.get("b_config_id") or "")

        starts_by_rollout: dict[str, list[int]] = {}
        first_predictions: dict[str, int | None] = {}
        for rollout_id, signal in signals.items():
            hops = signal.get("hops") or []
            if pair:
                a_mask = detector_mask(hops, config_meta[a_id])
                b_mask = detector_mask(hops, config_meta[b_id])
                mask = [
                    bool(left) or bool(right)
                    for left, right in zip(a_mask, b_mask)
                ]
            else:
                base = config_meta.get(config_id)
                if base is None:
                    continue
                mask = detector_mask(hops, base)
            starts = _positive_episode_starts(mask)
            starts_by_rollout[rollout_id] = starts
            first_predictions[rollout_id] = starts[0] if starts else None

        for population, events in populations:
            ranking.append(
                _evaluate_predictions(
                    candidate=candidate,
                    selector="first_trigger",
                    left_window=None,
                    right_window=None,
                    predictions=first_predictions,
                    signals=signals,
                    population=population,
                    events=events,
                )
            )

        for left in CHANGE_POINT_WINDOWS:
            for right in CHANGE_POINT_WINDOWS:
                selected_predictions: dict[str, int | None] = {}
                selected_scores: dict[str, float | None] = {}
                scored_counts: dict[str, int] = {}
                for rollout_id, signal in signals.items():
                    hops = signal.get("hops") or []
                    scored = [
                        (start, _change_score(hops, start, left, right))
                        for start in starts_by_rollout.get(rollout_id, [])
                    ]
                    valid = [
                        (start, score)
                        for start, score in scored
                        if score is not None
                    ]
                    scored_counts[rollout_id] = len(valid)
                    if valid:
                        selected_start, selected_score = max(
                            valid,
                            key=lambda item: (float(item[1]), -int(item[0])),
                        )
                        selected_predictions[rollout_id] = int(selected_start)
                        selected_scores[rollout_id] = float(selected_score)
                    else:
                        selected_predictions[rollout_id] = None
                        selected_scores[rollout_id] = None

                for population, events in populations:
                    ranking.append(
                        _evaluate_predictions(
                            candidate=candidate,
                            selector="offline_max_change_score",
                            left_window=left,
                            right_window=right,
                            predictions=selected_predictions,
                            signals=signals,
                            population=population,
                            events=events,
                        )
                    )

                for event in all_events:
                    rollout_id = str(event["rollout_id"])
                    signal = signals.get(rollout_id)
                    if signal is None:
                        continue
                    frames = [int(value) for value in signal["frames"]]
                    causal_index = _onset_anchor(
                        frames, int(event["causal_onset_frame"])
                    )
                    observable_index = _onset_anchor(
                        frames, int(event["observable_onset_frame"])
                    )
                    if causal_index is None or observable_index is None:
                        continue
                    starts = starts_by_rollout.get(rollout_id, [])
                    selected = selected_predictions.get(rollout_id)
                    first = first_predictions.get(rollout_id)
                    global_argmax = global_predictions.get(rollout_id)
                    diagnostics.append(
                        {
                            **dict(candidate),
                            "L": left,
                            "R": right,
                            **dict(event),
                            "candidate_count": len(starts),
                            "scored_candidate_count": scored_counts.get(
                                rollout_id, 0
                            ),
                            "candidates_before_interval": sum(
                                start < causal_index for start in starts
                            ),
                            "candidates_in_interval": sum(
                                causal_index <= start <= observable_index
                                for start in starts
                            ),
                            "candidates_after_interval": sum(
                                start > observable_index for start in starts
                            ),
                            "first_trigger_sample_index": first,
                            "first_trigger_frame": (
                                frames[first] if first is not None else None
                            ),
                            "selected_trigger_sample_index": selected,
                            "selected_trigger_frame": (
                                frames[selected]
                                if selected is not None
                                else None
                            ),
                            "selected_score": selected_scores.get(rollout_id),
                            "global_progress_argmax_sample_index": global_argmax,
                            "global_progress_argmax_frame": (
                                frames[global_argmax]
                                if global_argmax is not None
                                else None
                            ),
                        }
                    )

    _add_interval_ranks(ranking)
    selector_order = {
        "offline_max_change_score": 0,
        "first_trigger": 1,
        "global_fused_progress_argmax": 2,
    }
    ranking.sort(
        key=lambda row: (
            0
            if row["population"] == "first_eligible_event_per_failed_rollout"
            else 1,
            str(row["population"]),
            selector_order.get(str(row["selector"]), 99),
            int(row["rank_mse"]) if row.get("rank_mse") is not None else math.inf,
            int(row.get("L") or 0),
            int(row.get("R") or 0),
            str(row.get("config_id") or ""),
        )
    )
    diagnostics.sort(
        key=lambda row: (
            str(row["rollout_id"]),
            int(row["event_index"]),
            str(row["config_id"]),
            int(row["L"]),
            int(row["R"]),
        )
    )
    return ranking, diagnostics


def rank_existing_sweep_interval_localization(
    *,
    event_rows: Sequence[Mapping[str, Any]],
    ensemble_sweep: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible ranking-only wrapper."""
    ranking, _diagnostics = offline_change_point_localization(
        event_rows=event_rows,
        ensemble_sweep=ensemble_sweep,
        signals=signals,
        configs=configs,
    )
    return ranking
