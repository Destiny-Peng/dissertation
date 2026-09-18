from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


BASELINES_DIR = Path(__file__).resolve().parents[2] / "baselines"
sys.path.insert(0, str(BASELINES_DIR))

from procvlm_worker import infer_rollout  # noqa: E402
from procvlm_procedure_state import (  # noqa: E402
    Action,
    Chain,
    Procedure,
    StatefulProcedureTracker,
    build_stateful_history_prompt,
    parse_remaining_actions,
    task_history_text,
)


def procedure() -> Procedure:
    alphabet = (
        Action(
            "A1", "grasp", "alphabet_soup_can",
            text="grasp the alphabet soup can",
            history="The alphabet soup can has already been grasped.",
        ),
        Action(
            "A2", "place", "alphabet_soup_can", "basket",
            text="place the alphabet soup can into the basket",
            history="The alphabet soup can has already been placed into the basket.",
        ),
    )
    tomato = (
        Action(
            "B1", "grasp", "tomato_sauce_can",
            text="grasp the tomato sauce can",
            history="The tomato sauce can has already been grasped.",
        ),
        Action(
            "B2", "place", "tomato_sauce_can", "basket",
            text="place the tomato sauce can into the basket",
            history="The tomato sauce can has already been placed into the basket.",
        ),
    )
    return Procedure(
        "task0",
        "put both cans in the basket",
        (Chain("alphabet", alphabet), Chain("tomato", tomato)),
        verb_aliases={
            "grasp": ("grasp", "pick up", "grab"),
            "place": ("place", "put"),
        },
        object_aliases={
            "alphabet_soup_can": ("alphabet soup", "alphabet soup can"),
            "tomato_sauce_can": ("tomato sauce", "tomato sauce can"),
        },
        target_aliases={"basket": ("basket",)},
    )


def parsed(text: str):
    return parse_remaining_actions(text, procedure())


def test_semantic_parser_uses_remaining_actions_section_and_aliases() -> None:
    result = parsed(
        "Earlier explanation mentions grasping the alphabet soup can.\n"
        "The following actions are required:\n"
        "1. Put the alphabet soup can in the basket.\n"
        "2. Pick up the tomato sauce can.\n"
        "3. Place the tomato sauce can into the basket.\n"
        "Therefore, the estimated progress is <progress>25%</progress>."
    )
    assert result.parse_valid is True
    assert result.remaining_ids == ("A2", "B1", "B2")
    assert result.observed_stage == {"alphabet": 1, "tomato": 0}
    assert len(result.parsed_actions) == 3


def test_suffix_property_rejects_non_suffix_remaining_set() -> None:
    result = parsed(
        "The following actions are required:\n"
        "1. grasp the alphabet soup can\n"
        "2. grasp the tomato sauce can\n"
        "3. place the tomato sauce can into the basket\n"
        "<progress>20%</progress>"
    )
    assert result.parse_valid is False
    assert any("suffix property" in error for error in result.errors)


def test_forward_transition_requires_seven_of_nine_valid_observations() -> None:
    tracker = StatefulProcedureTracker(procedure())
    stage1 = parsed(
        "The following actions are required:\n"
        "1. place the alphabet soup can into the basket\n"
        "2. grasp the tomato sauce can\n"
        "3. place the tomato sauce can into the basket\n"
        "<progress>25%</progress>"
    )
    assert stage1.parse_valid

    for frame in range(6):
        out = tracker.update(frame, stage1)
        assert out["persistent_stage"]["alphabet"] == 0

    out = tracker.update(6, stage1)
    assert out["persistent_stage"]["alphabet"] == 1
    assert out["state_update"]["from"] == 0
    assert out["state_update"]["to"] == 1
    assert out["transition_support"]["alphabet"]["positive"] == 7
    assert out["transition_support"]["alphabet"]["required"] == 7


def test_invalid_observations_do_not_enter_tracker_window() -> None:
    tracker = StatefulProcedureTracker(procedure())
    invalid = parsed("No useful structured reasoning here.")
    assert invalid.parse_valid is False
    out = tracker.update(0, invalid)
    assert out["valid_observation_count"] == 0


def test_maximum_forward_jump_is_one_stage() -> None:
    tracker = StatefulProcedureTracker(
        procedure(),
        support_threshold=1,
        window_size=1,
    )
    all_done = parsed("The following actions are required: none. <progress>100%</progress>")
    out = tracker.update(0, all_done)
    assert out["persistent_stage"]["alphabet"] == 1
    assert out["persistent_stage"]["tomato"] == 1


def test_history_uses_only_highest_postcondition_per_chain() -> None:
    text = task_history_text(procedure(), {"alphabet": 2, "tomato": 0})
    assert text == "The alphabet soup can has already been placed into the basket."
    assert "grasped" not in text


def test_stateful_prompt_contains_history_but_no_canonical_ids() -> None:
    history = "The alphabet soup can has already been placed into the basket."
    prompt = build_stateful_history_prompt("put both cans in the basket", history)
    assert "Task history:" in prompt
    assert history in prompt
    assert "A1" not in prompt
    assert "A2" not in prompt
    assert "Confirmed" not in prompt


def test_baseline_mode_keeps_upstream_inference_path_without_procedure_config(monkeypatch) -> None:
    calls = {}
    inference_module = types.ModuleType("evqa.inference")

    def fake_infer_progress_from_video(**kwargs):
        calls.update(kwargs)
        return [{"progress": 10.0}]

    inference_module.infer_progress_from_video = fake_infer_progress_from_video
    evqa_module = types.ModuleType("evqa")
    evqa_module.__path__ = []
    evqa_module.inference = inference_module
    monkeypatch.setitem(sys.modules, "evqa", evqa_module)
    monkeypatch.setitem(sys.modules, "evqa.inference", inference_module)

    args = argparse.Namespace(
        procedure_mode="baseline",
        max_sampled_frames=None,
        model_path=Path("/tmp/model"),
        window_size=4,
        frame_stride=3,
        torch_dtype="bf16",
        max_new_tokens=128,
        tp=1,
        enable_value_head=False,
        use_lora=False,
    )
    job = {
        "video_path": "/tmp/video.mp4",
        "task": "original task",
        "output_path": "/tmp/procvlm_raw.jsonl",
    }
    result = infer_rollout(job, args, object(), {"gpu_memory_utilization": 0.5})
    assert result == [{"progress": 10.0}]
    assert calls["task"] == "original task"
    assert calls["frame_stride"] == 3
    assert calls["window_size"] == 4
    assert "procedure_config" not in calls
