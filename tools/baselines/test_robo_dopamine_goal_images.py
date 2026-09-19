#!/usr/bin/env python3
"""Tests for task-specific Robo-Dopamine goal-image selection."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from unittest import mock

import robo_dopamine_runner


def make_args(root: Path, goal_image: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(data_root=root, goal_image=goal_image)


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


if __name__ == "__main__":
    test_libero10_goal_image_is_selected_by_task_id()
    test_explicit_goal_image_overrides_task_default()
    test_missing_libero10_task_goal_fails_loudly()
    test_non_libero10_keeps_blank_goal_fallback()
    print("ROBODOPAMINE_GOAL_IMAGE_TESTS_OK")
