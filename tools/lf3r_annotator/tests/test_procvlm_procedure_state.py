from __future__ import annotations

import sys
from pathlib import Path


BASELINES_DIR = Path(__file__).resolve().parents[2] / "baselines"
sys.path.insert(0, str(BASELINES_DIR))

from procvlm_procedure_state import (  # noqa: E402
    Action,
    Chain,
    Procedure,
    StatefulProcedureTracker,
    parse_remaining_actions,
)


def procedure() -> Procedure:
    return Procedure(
        "task0",
        "put both cans in the basket",
        (
            Chain("alphabet", (Action("A1", "grasp alphabet"), Action("A2", "place alphabet"))),
            Chain("tomato", (Action("B1", "grasp tomato"), Action("B2", "place tomato"))),
        ),
    )


def parsed(text: str):
    return parse_remaining_actions(text, procedure(), allow_text_fallback=False)


def test_suffix_property_rejects_non_suffix_remaining_set() -> None:
    result = parsed(
        "The following actions are required:\n"
        "1. [A1] grasp alphabet\n"
        "2. [B1] grasp tomato\n"
        "3. [B2] place tomato\n"
        "<progress>20%</progress>"
    )
    assert result.parse_valid is False
    assert any("suffix property" in error for error in result.errors)


def test_forward_transition_requires_three_of_four_and_minimum_span() -> None:
    tracker = StatefulProcedureTracker(procedure(), fps=30.0)
    obs = parsed(
        "The following actions are required:\n"
        "1. [A2] place alphabet\n"
        "2. [B1] grasp tomato\n"
        "3. [B2] place tomato\n"
        "<progress>25%</progress>"
    )
    assert obs.parse_valid
    assert tracker.update(0, obs)["confirmed_stage"]["alphabet"] == 0
    assert tracker.update(3, obs)["confirmed_stage"]["alphabet"] == 0
    out = tracker.update(6, obs)
    assert out["confirmed_stage"]["alphabet"] == 1
    assert out["transition_event"]["from_stage"] == 0
    assert out["transition_event"]["to_stage"] == 1


def test_chain_completion_requires_four_of_five_and_point_three_seconds() -> None:
    tracker = StatefulProcedureTracker(procedure(), fps=30.0)
    stage1 = parsed(
        "The following actions are required:\n"
        "1. [A2] place alphabet\n"
        "2. [B1] grasp tomato\n"
        "3. [B2] place tomato\n"
        "<progress>25%</progress>"
    )
    for frame in (0, 3, 6):
        tracker.update(frame, stage1)
    assert tracker.confirmed_stage["alphabet"] == 1

    done_alphabet = parsed(
        "The following actions are required:\n"
        "1. [B1] grasp tomato\n"
        "2. [B2] place tomato\n"
        "<progress>50%</progress>"
    )
    for frame in (9, 12, 15):
        assert tracker.update(frame, done_alphabet)["confirmed_stage"]["alphabet"] == 1
    out = tracker.update(18, done_alphabet)
    assert out["confirmed_stage"]["alphabet"] == 2
    assert out["transition_event"]["reason"].startswith("4-of-5")


def test_maximum_forward_jump_is_one_stage() -> None:
    tracker = StatefulProcedureTracker(
        procedure(),
        fps=30.0,
        forward_votes=1,
        forward_window=1,
        forward_min_span_sec=0.0,
        completion_votes=1,
        completion_window=1,
        completion_min_span_sec=0.0,
    )
    all_done = parsed("The following actions are required: none. <progress>100%</progress>")
    out = tracker.update(0, all_done)
    assert out["confirmed_stage"]["alphabet"] == 1
    assert out["confirmed_stage"]["tomato"] == 1


def test_candidate_timeout_discards_old_support() -> None:
    tracker = StatefulProcedureTracker(
        procedure(),
        fps=30.0,
        decision_interval_frames=3,
        forward_votes=2,
        forward_window=4,
        forward_min_span_sec=0.0,
        candidate_timeout_sec=0.5,
    )
    obs = parsed(
        "The following actions are required:\n"
        "1. [A2] place alphabet\n"
        "2. [B1] grasp tomato\n"
        "3. [B2] place tomato\n"
        "<progress>25%</progress>"
    )
    tracker.update(0, obs)
    neutral = parsed(
        "The following actions are required:\n"
        "1. [A1] grasp alphabet\n"
        "2. [A2] place alphabet\n"
        "3. [B1] grasp tomato\n"
        "4. [B2] place tomato\n"
        "<progress>0%</progress>"
    )
    tracker.update(18, neutral)
    out = tracker.update(21, obs)
    assert out["confirmed_stage"]["alphabet"] == 0
    assert out["transition_support"]["alphabet"]["votes"] == 1


def test_baseline_parser_is_not_required_for_baseline_mode_contract() -> None:
    # Baseline mode is intentionally handled outside the tracker and may emit
    # unrestricted ProcVLM text without canonical action IDs.
    unrestricted = parsed("Pick up the can and put it away. <progress>10%</progress>")
    assert unrestricted.parse_valid is False
