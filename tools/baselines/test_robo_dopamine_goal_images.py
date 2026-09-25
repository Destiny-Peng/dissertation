#!/usr/bin/env python3
"""Tests for task-specific Robo-Dopamine goal-image selection."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from unittest import mock

import robo_dopamine_runner


def make_args(
    root: Path,
    goal_image: Path | None = None,
    camera_mode: str = "auto",
) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=root,
        goal_image=goal_image,
        robo_camera_mode=camera_mode,
    )


def make_config(root: Path) -> dict[str, Path]:
    repo = root / "robo"
    blank = repo / "examples" / "blank_goal.png"
    blank.parent.mkdir(parents=True)
    blank.touch()
    return {"repo": repo}


def test_libero10_goal_image_is_selected_by_task_id() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-goals-") as temporary:
        root = Path(temporary)
        goal_root = root / "outputs" / "robodopamine_goal"
        goal_root.mkdir(parents=True)
        task0 = goal_root / "libero-10-task0.jpg"
        task9 = goal_root / "libero-10-task9.jpg"
        task0.touch()
        task9.touch()
        config = make_config(root)

        with mock.patch.object(
            robo_dopamine_runner,
            "ROBO_LIBERO10_GOAL_ROOT",
            goal_root,
        ):
            resolved0 = robo_dopamine_runner.resolve_goal_image(
                {"id": "task0-rollout", "task_suite": "libero_10", "task_id": 0},
                make_args(root),
                config,
            )
            resolved9 = robo_dopamine_runner.resolve_goal_image(
                {"id": "task9-rollout", "dataset_role": "libero_10", "task_id": "9"},
                make_args(root),
                config,
            )

        assert resolved0 == task0.resolve()
        assert resolved9 == task9.resolve()


def test_explicit_goal_image_overrides_task_default() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-goals-") as temporary:
        root = Path(temporary)
        explicit = root / "custom.jpg"
        explicit.touch()
        config = make_config(root)

        with mock.patch.object(
            robo_dopamine_runner,
            "ROBO_LIBERO10_GOAL_ROOT",
            root / "missing-defaults",
        ):
            resolved = robo_dopamine_runner.resolve_goal_image(
                {"id": "task3-rollout", "task_suite": "libero_10", "task_id": 3},
                make_args(root, explicit),
                config,
            )

        assert resolved == explicit.resolve()


def test_missing_libero10_task_goal_fails_loudly() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-goals-") as temporary:
        root = Path(temporary)
        config = make_config(root)
        goal_root = root / "outputs" / "robodopamine_goal"
        goal_root.mkdir(parents=True)

        with mock.patch.object(
            robo_dopamine_runner,
            "ROBO_LIBERO10_GOAL_ROOT",
            goal_root,
        ):
            try:
                robo_dopamine_runner.resolve_goal_image(
                    {"id": "task5-rollout", "task_suite": "libero_10", "task_id": 5},
                    make_args(root),
                    config,
                )
            except FileNotFoundError as error:
                assert "libero-10-task5.jpg" in str(error)
            else:
                raise AssertionError("Missing task-specific goal image was not rejected")


def test_non_libero10_keeps_blank_goal_fallback() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-goals-") as temporary:
        root = Path(temporary)
        config = make_config(root)
        resolved = robo_dopamine_runner.resolve_goal_image(
            {"id": "spatial-rollout", "task_suite": "libero_spatial", "task_id": 0},
            make_args(root),
            config,
        )
        assert resolved == (config["repo"] / "examples" / "blank_goal.png").resolve()



def test_robo_camera_inputs_adapt_physical_wrist_to_three_slots() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-cameras-") as temporary:
        root = Path(temporary)
        canonical = root / "canonical.mp4"
        high = root / "cam_high.mp4"
        wrist = root / "cam_wrist.mp4"
        canonical.touch()
        high.touch()
        wrist.touch()

        resolved, mode = robo_dopamine_runner.resolve_robo_camera_inputs(
            {
                "id": "multi-rollout",
                "camera_video_paths": {
                    "cam_high": high.name,
                    "cam_wrist": wrist.name,
                },
            },
            make_args(root),
            canonical,
        )

        assert mode == "multi_view"
        assert resolved == {
            "cam_high": str(high.resolve()),
            "cam_left_wrist": str(wrist.resolve()),
            "cam_right_wrist": str(wrist.resolve()),
        }
        assert resolved["cam_left_wrist"] == resolved["cam_right_wrist"]


def test_robo_camera_inputs_repeat_primary_for_single_view() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-cameras-") as temporary:
        root = Path(temporary)
        canonical = root / "canonical.mp4"
        canonical.touch()

        resolved, mode = robo_dopamine_runner.resolve_robo_camera_inputs(
            {"id": "single-rollout"},
            make_args(root),
            canonical,
        )

        assert mode == "single_view"
        assert set(resolved) == set(robo_dopamine_runner.ROBO_CAMERA_SLOTS)
        assert set(resolved.values()) == {str(canonical.resolve())}



def test_robo_camera_inputs_force_single_view_even_when_multiview_exists() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-cameras-") as temporary:
        root = Path(temporary)
        canonical = root / "canonical.mp4"
        high = root / "cam_high.mp4"
        wrist = root / "cam_wrist.mp4"
        canonical.touch()
        high.touch()
        wrist.touch()

        resolved, mode = robo_dopamine_runner.resolve_robo_camera_inputs(
            {
                "id": "force-single",
                "camera_video_paths": {
                    "cam_high": high.name,
                    "cam_wrist": wrist.name,
                },
            },
            make_args(root, camera_mode="single_view"),
            canonical,
        )

        assert mode == "single_view"
        assert set(resolved.values()) == {str(canonical.resolve())}


def test_robo_camera_inputs_require_multiview_when_requested() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-cameras-") as temporary:
        root = Path(temporary)
        canonical = root / "canonical.mp4"
        canonical.touch()

        try:
            robo_dopamine_runner.resolve_robo_camera_inputs(
                {"id": "missing-multiview"},
                make_args(root, camera_mode="multi_view"),
                canonical,
            )
        except ValueError as error:
            assert "has no camera_video_paths" in str(error)
        else:
            raise AssertionError("Explicit multi_view did not reject missing camera videos")

def test_robo_camera_inputs_auto_falls_back_for_single_camera() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-cameras-") as temporary:
        root = Path(temporary)
        canonical = root / "canonical.mp4"
        canonical.touch()
        high = root / "cam_high.mp4"
        high.touch()

        resolved, mode = robo_dopamine_runner.resolve_robo_camera_inputs(
            {
                "id": "single-camera-rollout",
                "camera_video_paths": {"cam_high": high.name},
            },
            make_args(root),
            high,
        )
        assert mode == "single_view"
        assert set(resolved.values()) == {str(high.resolve())}


def test_robo_camera_inputs_use_distinct_left_and_right_wrist_views() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-cameras-") as temporary:
        root = Path(temporary)
        high = root / "high.mp4"
        left = root / "left.mp4"
        right = root / "right.mp4"
        for path in (high, left, right):
            path.touch()

        resolved, mode = robo_dopamine_runner.resolve_robo_camera_inputs(
            {
                "id": "three-camera-rollout",
                "camera_video_paths": {
                    "cam_high": high.name,
                    "cam_left_wrist": left.name,
                    "cam_right_wrist": right.name,
                },
            },
            make_args(root),
            high,
        )

        assert mode == "multi_view"
        assert resolved == {
            "cam_high": str(high.resolve()),
            "cam_left_wrist": str(left.resolve()),
            "cam_right_wrist": str(right.resolve()),
        }


if __name__ == "__main__":
    test_libero10_goal_image_is_selected_by_task_id()
    test_explicit_goal_image_overrides_task_default()
    test_missing_libero10_task_goal_fails_loudly()
    test_non_libero10_keeps_blank_goal_fallback()
    test_robo_camera_inputs_adapt_physical_wrist_to_three_slots()
    test_robo_camera_inputs_repeat_primary_for_single_view()
    test_robo_camera_inputs_force_single_view_even_when_multiview_exists()
    test_robo_camera_inputs_require_multiview_when_requested()
    test_robo_camera_inputs_auto_falls_back_for_single_camera()
    test_robo_camera_inputs_use_distinct_left_and_right_wrist_views()
    print("ROBODOPAMINE_GOAL_IMAGE_TESTS_OK")
