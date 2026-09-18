#!/usr/bin/env python3
"""Summarize external ProcVLM procedure tracking from one raw JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def add_posthoc_procedure_states(
    rows: list[dict[str, Any]],
    procedure_path: Path,
) -> list[dict[str, Any]]:
    """Parse baseline/free-form ProcVLM reasoning against an external ontology.

    Existing tracker fields are left untouched. Baseline inference itself is never
    changed; this is CPU-only post-processing.
    """
    from procvlm_procedure_state import load_procedure, parse_remaining_actions

    procedure = load_procedure(procedure_path)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if "parse_valid" not in item:
            answer = str(item.get("model_output") or item.get("reasoning") or "")
            parsed = parse_remaining_actions(answer, procedure)
            item["parsed_actions"] = list(parsed.parsed_actions)
            item["canonical_remaining_ids"] = list(parsed.remaining_ids)
            item["parsed_remaining_ids"] = list(parsed.remaining_ids)
            item["parse_valid"] = bool(parsed.parse_valid)
            item["parse_source"] = "posthoc_" + parsed.source
            item["parse_errors"] = list(parsed.errors)
            item["observed_stage"] = parsed.observed_stage
            item["observed_state"] = parsed.observed_stage
        enriched.append(item)
    return enriched


def normalized_state(row: dict[str, Any]) -> tuple[tuple[str, int], ...] | None:
    if not row.get("parse_valid"):
        return None
    state = row.get("observed_stage")
    if not isinstance(state, dict):
        return None
    try:
        return tuple(sorted((str(chain), int(stage)) for chain, stage in state.items()))
    except (TypeError, ValueError):
        return None


def transition_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in rows:
        values = row.get("transition_events")
        if not isinstance(values, list):
            single = row.get("transition_event")
            values = [single] if isinstance(single, dict) else []
        for event in values:
            if not isinstance(event, dict):
                continue
            item = dict(event)
            item.setdefault("frame_index", row.get("frame_index"))
            events.append(item)
    return events


def summarize(
    rows: list[dict[str, Any]],
    *,
    progress_regression_threshold: float = -20.0,
) -> dict[str, Any]:
    valid_states: list[tuple[int, tuple[tuple[str, int], ...]]] = []
    previous_state: tuple[tuple[str, int], ...] | None = None
    state_switch_count = 0
    reopen_count = 0
    previous_by_chain: dict[str, int] = {}

    persistent_regression_count = 0
    previous_persistent: dict[str, int] = {}

    finite_progress: list[tuple[int, float]] = []
    progress_regression_count = 0
    maximum_progress_regression = 0.0
    previous_progress: float | None = None

    for row in rows:
        frame = int(row.get("frame_index", len(valid_states)))
        state = normalized_state(row)
        if state is not None:
            valid_states.append((frame, state))
            if previous_state is not None and state != previous_state:
                state_switch_count += 1
            current_by_chain = dict(state)
            for chain, stage in current_by_chain.items():
                if chain in previous_by_chain and stage < previous_by_chain[chain]:
                    reopen_count += 1
                previous_by_chain[chain] = stage
            previous_state = state

        persistent = row.get("persistent_state") or row.get("persistent_stage")
        if isinstance(persistent, dict):
            try:
                current_persistent = {str(k): int(v) for k, v in persistent.items()}
            except (TypeError, ValueError):
                current_persistent = {}
            for chain, stage in current_persistent.items():
                if chain in previous_persistent and stage < previous_persistent[chain]:
                    persistent_regression_count += 1
            previous_persistent.update(current_persistent)

        value = row.get("progress")
        try:
            progress = float(value)
        except (TypeError, ValueError):
            continue
        if previous_progress is not None:
            delta = progress - previous_progress
            if delta < progress_regression_threshold:
                progress_regression_count += 1
            if delta < maximum_progress_regression:
                maximum_progress_regression = delta
        previous_progress = progress
        finite_progress.append((frame, progress))

    events = transition_events(rows)
    return {
        "sample_count": len(rows),
        "parse_valid_count": sum(bool(row.get("parse_valid")) for row in rows),
        "parse_invalid_count": sum(row.get("parse_valid") is False for row in rows),
        "reasoning_state_switch_count": state_switch_count,
        "completed_subtask_reopen_count": reopen_count,
        "raw_subtask_reopen_count": reopen_count,
        "raw_subtask_reopen_rate": (
            reopen_count / max(len(valid_states) - 1, 1) if valid_states else 0.0
        ),
        "persistent_state_regression_count": persistent_regression_count,
        "state_update_count": len(events),
        "state_updates": events,
        "confirmed_transition_count": len(events),
        "confirmed_transitions": events,
        "progress_regression_threshold": progress_regression_threshold,
        "progress_regression_count": progress_regression_count,
        "maximum_progress_regression": maximum_progress_regression,
        "first_frame": int(rows[0].get("frame_index", 0)) if rows else None,
        "last_frame": int(rows[-1].get("frame_index", 0)) if rows else None,
    }


def load_ground_truth(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("transitions", [])
    if not isinstance(value, list):
        raise ValueError("Ground truth must be a list or {'transitions': [...]} object")
    transitions: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Ground-truth transition {index} is not an object")
        chain = str(item.get("chain", "")).strip()
        try:
            to_stage = int(item["to_stage"])
            frame = int(item["frame"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Ground-truth transition {index} requires chain, to_stage, frame"
            ) from error
        if not chain or to_stage < 1 or frame < 0:
            raise ValueError(f"Invalid ground-truth transition {index}")
        transitions.append({"chain": chain, "to_stage": to_stage, "frame": frame})
    return transitions


def compare_transitions(
    confirmed: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    early_tolerance_frames: int = 0,
) -> dict[str, Any]:
    truth = {(item["chain"], item["to_stage"]): item for item in ground_truth}
    seen: set[tuple[str, int]] = set()
    lags: list[dict[str, Any]] = []
    false_commits: list[dict[str, Any]] = []

    for event in confirmed:
        try:
            to_stage = event.get("to", event.get("to_stage"))
            key = (str(event["chain"]), int(to_stage))
            frame = int(event["frame_index"])
        except (KeyError, TypeError, ValueError):
            continue
        expected = truth.get(key)
        if expected is None:
            false_commits.append({**event, "false_commit_reason": "no_matching_ground_truth"})
            continue
        seen.add(key)
        lag = frame - int(expected["frame"])
        lags.append({
            "chain": key[0],
            "to_stage": key[1],
            "human_frame": int(expected["frame"]),
            "confirmed_frame": frame,
            "lag_frames": lag,
        })
        if lag < -int(early_tolerance_frames):
            false_commits.append({
                **event,
                "human_frame": int(expected["frame"]),
                "false_commit_reason": "confirmed_before_human_transition",
            })

    missed = [
        item for key, item in truth.items()
        if key not in seen
    ]
    return {
        "false_state_update_count": len(false_commits),
        "false_state_updates": false_commits,
        "state_update_lag": lags,
        "false_commit_count": len(false_commits),
        "false_commits": false_commits,
        "transition_detection_lag": lags,
        "missed_human_transition_count": len(missed),
        "missed_human_transitions": missed,
        "early_tolerance_frames": int(early_tolerance_frames),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_jsonl", type=Path)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument(
        "--procedure-config",
        type=Path,
        default=None,
        help="External ontology for post-hoc baseline canonicalization",
    )
    parser.add_argument("--progress-regression-threshold", type=float, default=-20.0)
    parser.add_argument("--early-tolerance-frames", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.early_tolerance_frames < 0:
        parser.error("--early-tolerance-frames must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    raw_path = args.raw_jsonl.expanduser().resolve()
    rows = load_jsonl(raw_path)
    if args.procedure_config is not None:
        procedure_path = args.procedure_config.expanduser().resolve()
        rows = add_posthoc_procedure_states(rows, procedure_path)
    else:
        procedure_path = None
    report = summarize(
        rows,
        progress_regression_threshold=args.progress_regression_threshold,
    )
    report["raw_jsonl"] = str(raw_path)
    report["posthoc_procedure_config"] = str(procedure_path) if procedure_path else None

    if args.ground_truth is not None:
        ground_truth_path = args.ground_truth.expanduser().resolve()
        comparison = compare_transitions(
            report["confirmed_transitions"],
            load_ground_truth(ground_truth_path),
            early_tolerance_frames=args.early_tolerance_frames,
        )
        report.update(comparison)
        report["ground_truth"] = str(ground_truth_path)
    else:
        report["false_commit_count"] = None
        report["transition_detection_lag"] = None
        report["ground_truth_note"] = (
            "false_commit_count and transition_detection_lag require --ground-truth"
        )

    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
