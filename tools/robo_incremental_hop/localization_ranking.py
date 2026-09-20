"""Rank existing fused-hop sweep configs by first-trigger localization only."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

LOCALIZATION_WINDOWS = (1, 3, 5, 10)


def _first_trigger_frames(
    event_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str], int],
    dict[str, dict[str, Any]],
]:
    """Recover each existing config's first trigger per rollout from sweep rows."""
    triggers: dict[tuple[str, str], int] = {}
    configs: dict[str, dict[str, Any]] = {}
    for row in event_rows:
        config_id = str(row.get("config_id") or "")
        rollout_id = str(row.get("rollout_id") or "")
        if not config_id or not rollout_id:
            continue
        configs.setdefault(
            config_id,
            {
                key: row.get(key)
                for key in (
                    "config_id",
                    "detector_family",
                    "epsilon",
                    "n",
                    "m",
                    "k",
                    "theta",
                    "A",
                    "delta",
                    "parameters_json",
                )
            },
        )
        candidates = [
            row.get("earliest_early_positive_frame"),
            row.get("detection_frame"),
        ]
        frames = [
            int(value)
            for value in candidates
            if value is not None and value != ""
        ]
        if not frames:
            continue
        key = (config_id, rollout_id)
        frame = min(frames)
        if key not in triggers or frame < triggers[key]:
            triggers[key] = frame
    return triggers, configs


def _population_specs(
    event_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Build GT populations once, independent of detector config."""
    events: dict[str, dict[str, Any]] = {}
    for row in event_rows:
        if str(row.get("outcome") or "") != "terminal_failure":
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
                "observable_onset_frame": int(row["observable_onset_frame"]),
            },
        )

    all_events = sorted(
        events.values(),
        key=lambda row: (
            row["rollout_id"],
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
                row["observable_onset_frame"],
                row["event_index"],
            ),
        )
        for rollout_events in by_rollout.values()
    ]
    grasp_events = [
        event
        for event in all_events
        if event["failure_type"] == "grasp_failure"
    ]
    return [
        ("all_failure_events", all_events),
        ("first_event_per_failed_rollout", first_events),
        ("grasp_failure", grasp_events),
    ]


def _sample_index(frames: Sequence[int], frame: int) -> int | None:
    for index, value in enumerate(frames):
        if int(value) == int(frame):
            return index
    return None


def _onset_anchor(frames: Sequence[int], onset_frame: int) -> int | None:
    return next(
        (
            index
            for index, frame in enumerate(frames)
            if int(frame) >= int(onset_frame)
        ),
        None,
    )


def _evaluate_candidate(
    *,
    candidate: Mapping[str, Any],
    trigger_by_rollout: Mapping[str, int],
    signals: Mapping[str, Mapping[str, Any]],
    population: str,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    signed: list[int] = []
    rows: list[tuple[int | None, int | None]] = []

    for event in events:
        rollout_id = str(event["rollout_id"])
        signal = signals.get(rollout_id)
        if signal is None:
            continue
        frames = [int(value) for value in signal["frames"]]
        anchor = _onset_anchor(
            frames,
            int(event["observable_onset_frame"]),
        )
        if anchor is None:
            continue
        trigger_frame = trigger_by_rollout.get(rollout_id)
        trigger_index = (
            _sample_index(frames, trigger_frame)
            if trigger_frame is not None
            else None
        )
        offset = (
            trigger_index - anchor
            if trigger_index is not None
            else None
        )
        rows.append((offset, trigger_frame))
        if offset is not None:
            signed.append(offset)

    n = len(rows)
    triggered_n = len(signed)
    absolute = [abs(value) for value in signed]
    result = {
        **dict(candidate),
        "population": population,
        "event_n": n,
        "triggered_n": triggered_n,
        "no_trigger_n": n - triggered_n,
        "trigger_coverage": triggered_n / n if n else None,
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
        "mse_samples": (
            sum(value * value for value in signed) / len(signed)
            if signed
            else None
        ),
        "rmse_samples": (
            math.sqrt(
                sum(value * value for value in signed) / len(signed)
            )
            if signed
            else None
        ),
        "before_onset_n": sum(value < 0 for value in signed),
        "at_onset_n": sum(value == 0 for value in signed),
        "after_onset_n": sum(value > 0 for value in signed),
    }
    for window in LOCALIZATION_WINDOWS:
        hit_n = sum(
            offset is not None and abs(offset) <= window
            for offset, _frame in rows
        )
        result[f"within_{window}_n"] = hit_n
        result[f"within_{window}"] = hit_n / n if n else None
    return result


def _pair_candidates(
    ensemble_sweep: Sequence[Mapping[str, Any]],
    config_meta: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for row in ensemble_sweep:
        a_id = str(row.get("a_config_id") or "")
        b_id = str(row.get("b_config_id") or "")
        if not a_id or not b_id:
            continue
        key = (a_id, b_id)
        if key in seen:
            continue
        seen.add(key)
        a = config_meta.get(a_id, {})
        b = config_meta.get(b_id, {})
        rows.append(
            {
                "config_id": f"OR:{a_id}|{b_id}",
                "detector_family": "phenotype_or",
                "parameters_json": (
                    f"{a.get('detector_family')} {a.get('parameters_json')} OR "
                    f"{b.get('detector_family')} {b.get('parameters_json')}"
                ),
                "a_config_id": a_id,
                "b_config_id": b_id,
            }
        )
    return rows


def _add_ranks(rows: list[dict[str, Any]]) -> None:
    by_population: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_population[str(row["population"])].append(row)

    for population_rows in by_population.values():
        for window in LOCALIZATION_WINDOWS:
            ordered = sorted(
                population_rows,
                key=lambda row: (
                    -float(row.get(f"within_{window}") or 0.0),
                    -float(row.get("trigger_coverage") or 0.0),
                    math.inf
                    if row.get("median_absolute_error_samples") is None
                    else float(row["median_absolute_error_samples"]),
                    str(row["config_id"]),
                ),
            )
            for rank, row in enumerate(ordered, 1):
                row[f"rank_within_{window}"] = rank

        max_coverage = max(
            (
                float(row.get("trigger_coverage") or 0.0)
                for row in population_rows
            ),
            default=0.0,
        )
        coverage_floor = (
            1.0
            if any(
                float(row.get("trigger_coverage") or 0.0) >= 1.0 - 1e-12
                for row in population_rows
            )
            else max_coverage
        )
        eligible = [
            row
            for row in population_rows
            if float(row.get("trigger_coverage") or 0.0)
            >= coverage_floor - 1e-12
        ]
        for row in population_rows:
            row["error_rank_coverage_floor"] = coverage_floor
            row["error_rank_eligible"] = row in eligible
            row["rank_rmse"] = None
            row["rank_mae"] = None
            row["rank_median_abs_error"] = None

        ordered_rmse = sorted(
            eligible,
            key=lambda row: (
                math.inf
                if row.get("rmse_samples") is None
                else float(row["rmse_samples"]),
                math.inf
                if row.get("mae_samples") is None
                else float(row["mae_samples"]),
                math.inf
                if row.get("median_absolute_error_samples") is None
                else float(row["median_absolute_error_samples"]),
                str(row["config_id"]),
            ),
        )
        for rank, row in enumerate(ordered_rmse, 1):
            row["rank_rmse"] = rank

        for metric, rank_field in (
            ("mae_samples", "rank_mae"),
            ("median_absolute_error_samples", "rank_median_abs_error"),
        ):
            ordered = sorted(
                eligible,
                key=lambda row: (
                    math.inf if row.get(metric) is None else float(row[metric]),
                    str(row["config_id"]),
                ),
            )
            for rank, row in enumerate(ordered, 1):
                row[rank_field] = rank


def rank_existing_sweep_localization(
    *,
    event_rows: Sequence[Mapping[str, Any]],
    ensemble_sweep: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank only configurations/combinations already present in sweep artifacts."""
    first_triggers, config_meta = _first_trigger_frames(event_rows)
    populations = _population_specs(event_rows)
    ranking: list[dict[str, Any]] = []

    for config_id, meta in sorted(config_meta.items()):
        trigger_by_rollout = {
            rollout_id: frame
            for (candidate_id, rollout_id), frame in first_triggers.items()
            if candidate_id == config_id
        }
        for population, events in populations:
            ranking.append(
                _evaluate_candidate(
                    candidate=meta,
                    trigger_by_rollout=trigger_by_rollout,
                    signals=signals,
                    population=population,
                    events=events,
                )
            )

    for pair in _pair_candidates(ensemble_sweep, config_meta):
        a_id = str(pair["a_config_id"])
        b_id = str(pair["b_config_id"])
        rollout_ids = {
            rollout_id
            for candidate_id, rollout_id in first_triggers
            if candidate_id in {a_id, b_id}
        }
        trigger_by_rollout: dict[str, int] = {}
        for rollout_id in rollout_ids:
            candidates = [
                first_triggers.get((a_id, rollout_id)),
                first_triggers.get((b_id, rollout_id)),
            ]
            frames = [value for value in candidates if value is not None]
            if frames:
                trigger_by_rollout[rollout_id] = min(frames)
        for population, events in populations:
            ranking.append(
                _evaluate_candidate(
                    candidate=pair,
                    trigger_by_rollout=trigger_by_rollout,
                    signals=signals,
                    population=population,
                    events=events,
                )
            )

    _add_ranks(ranking)
    return sorted(
        ranking,
        key=lambda row: (
            str(row["population"]),
            (
                int(row["rank_rmse"])
                if row.get("rank_rmse") is not None
                else math.inf
            ),
            (
                int(row["rank_mae"])
                if row.get("rank_mae") is not None
                else math.inf
            ),
            str(row["config_id"]),
        ),
    )
