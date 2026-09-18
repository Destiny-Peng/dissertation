from __future__ import annotations

import sys
from pathlib import Path


BASELINES_DIR = Path(__file__).resolve().parents[2] / "baselines"
sys.path.insert(0, str(BASELINES_DIR))

from analyze_procvlm_procedure_tracker import (  # noqa: E402
    add_posthoc_procedure_states,
    compare_transitions,
    summarize,
)


def test_tracker_summary_counts_reopen_progress_and_persistent_regression() -> None:
    rows = [
        {
            "frame_index": 0,
            "parse_valid": True,
            "observed_state": {"a": 0, "b": 0},
            "observed_stage": {"a": 0, "b": 0},
            "persistent_state": {"a": 0, "b": 0},
            "progress": 10,
            "transition_events": [],
        },
        {
            "frame_index": 3,
            "parse_valid": True,
            "observed_state": {"a": 1, "b": 0},
            "observed_stage": {"a": 1, "b": 0},
            "persistent_state": {"a": 1, "b": 0},
            "progress": 40,
            "transition_events": [],
        },
        {
            "frame_index": 6,
            "parse_valid": True,
            "observed_state": {"a": 0, "b": 0},
            "observed_stage": {"a": 0, "b": 0},
            "persistent_state": {"a": 1, "b": 0},
            "progress": 15,
            "transition_events": [
                {"chain": "a", "from": 0, "to": 1, "frame_index": 6}
            ],
        },
    ]
    report = summarize(rows)
    assert report["reasoning_state_switch_count"] == 2
    assert report["raw_subtask_reopen_count"] == 1
    assert report["persistent_state_regression_count"] == 0
    assert report["progress_regression_count"] == 1
    assert report["maximum_progress_regression"] == -25
    assert report["state_update_count"] == 1


def test_transition_ground_truth_reports_lag_and_false_early_update() -> None:
    updates = [
        {"chain": "a", "to": 1, "frame_index": 8},
        {"chain": "a", "to": 2, "frame_index": 20},
    ]
    ground_truth = [
        {"chain": "a", "to_stage": 1, "frame": 10},
        {"chain": "a", "to_stage": 2, "frame": 18},
    ]
    report = compare_transitions(updates, ground_truth)
    assert report["false_state_update_count"] == 1
    assert [item["lag_frames"] for item in report["state_update_lag"]] == [-2, 2]


def test_posthoc_baseline_parser_adds_semantic_state(tmp_path) -> None:
    procedure = tmp_path / "procedure.json"
    procedure.write_text(
        """{
          "task_id": "x",
          "task": "put both cans away",
          "verb_aliases": {
            "grasp": ["grasp", "pick up"],
            "place": ["place", "put"]
          },
          "objects": {
            "alphabet": ["alphabet"],
            "tomato": ["tomato"]
          },
          "targets": {
            "basket": ["basket"]
          },
          "actions": {
            "A1": {"predicate": "grasp", "object": "alphabet", "text": "grasp alphabet"},
            "A2": {"predicate": "place", "object": "alphabet", "target": "basket", "text": "place alphabet in basket"},
            "B1": {"predicate": "grasp", "object": "tomato", "text": "grasp tomato"},
            "B2": {"predicate": "place", "object": "tomato", "target": "basket", "text": "place tomato in basket"}
          },
          "chains": {
            "a": ["A1", "A2"],
            "b": ["B1", "B2"]
          }
        }""",
        encoding="utf-8",
    )
    rows = [{
        "frame_index": 0,
        "model_output": (
            "The following actions are required:\n"
            "1. put alphabet in basket\n"
            "2. pick up tomato\n"
            "3. place tomato in basket\n"
            "<progress>25%</progress>"
        ),
        "progress": 25,
    }]
    enriched = add_posthoc_procedure_states(rows, procedure)
    assert enriched[0]["parse_valid"] is True
    assert enriched[0]["parse_source"] == "posthoc_remaining_section"
    assert enriched[0]["canonical_remaining_ids"] == ["A2", "B1", "B2"]
    assert enriched[0]["observed_state"] == {"a": 1, "b": 0}
