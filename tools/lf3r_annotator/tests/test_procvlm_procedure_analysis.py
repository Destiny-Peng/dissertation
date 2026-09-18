from __future__ import annotations

import sys
from pathlib import Path


BASELINES_DIR = Path(__file__).resolve().parents[2] / "baselines"
sys.path.insert(0, str(BASELINES_DIR))

from analyze_procvlm_procedure_tracker import compare_transitions, summarize  # noqa: E402


def test_tracker_summary_counts_state_reopen_and_progress_regression() -> None:
    rows = [
        {
            "frame_index": 0,
            "parse_valid": True,
            "observed_stage": {"a": 0, "b": 0},
            "progress": 10,
            "transition_events": [],
        },
        {
            "frame_index": 3,
            "parse_valid": True,
            "observed_stage": {"a": 1, "b": 0},
            "progress": 40,
            "transition_events": [],
        },
        {
            "frame_index": 6,
            "parse_valid": True,
            "observed_stage": {"a": 0, "b": 0},
            "progress": 15,
            "transition_events": [
                {"chain": "a", "from_stage": 0, "to_stage": 1, "frame_index": 6}
            ],
        },
    ]
    report = summarize(rows)
    assert report["reasoning_state_switch_count"] == 2
    assert report["completed_subtask_reopen_count"] == 1
    assert report["progress_regression_count"] == 1
    assert report["maximum_progress_regression"] == -25
    assert report["confirmed_transition_count"] == 1


def test_transition_ground_truth_reports_lag_and_false_early_commit() -> None:
    confirmed = [
        {"chain": "a", "to_stage": 1, "frame_index": 8},
        {"chain": "a", "to_stage": 2, "frame_index": 20},
    ]
    ground_truth = [
        {"chain": "a", "to_stage": 1, "frame": 10},
        {"chain": "a", "to_stage": 2, "frame": 18},
    ]
    report = compare_transitions(confirmed, ground_truth)
    assert report["false_commit_count"] == 1
    assert [item["lag_frames"] for item in report["transition_detection_lag"]] == [-2, 2]
