#!/usr/bin/env python3
"""Tests for manifest-provided Robo-Dopamine goal images."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import robo_dopamine_runner


def make_args(root: Path, goal_image: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(data_root=root, goal_image=goal_image)


def make_config(root: Path) -> dict[str, Path]:
    repo = root / "robo"
    blank = repo / "examples" / "blank_goal.png"
    blank.parent.mkdir(parents=True)
    blank.touch()
    return {"repo": repo}


def test_manifest_goal_image_path_overrides_run_goal_image() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-manifest-goal-") as temporary:
        root = Path(temporary)
        manifest_goal = root / "goals" / "episode_goal.jpg"
        manifest_goal.parent.mkdir(parents=True)
        manifest_goal.touch()
        run_goal = root / "run_override.jpg"
        run_goal.touch()

        resolved = robo_dopamine_runner.resolve_goal_image(
            {
                "id": "rollout-a",
                "task_suite": "libero_10",
                "task_id": 0,
                "goal_image_path": "goals/episode_goal.jpg",
            },
            make_args(root, run_goal),
            make_config(root),
        )

        assert resolved == manifest_goal.resolve()


def test_absolute_manifest_goal_image_path_is_used_directly() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-manifest-goal-") as temporary:
        root = Path(temporary)
        manifest_goal = root / "goal.jpg"
        manifest_goal.touch()

        resolved = robo_dopamine_runner.resolve_goal_image(
            {
                "id": "rollout-b",
                "task_suite": "libero_spatial",
                "goal_image_path": str(manifest_goal),
            },
            make_args(root),
            make_config(root),
        )

        assert resolved == manifest_goal.resolve()


def test_missing_manifest_goal_image_path_fails_instead_of_falling_back() -> None:
    with tempfile.TemporaryDirectory(prefix="robo-manifest-goal-") as temporary:
        root = Path(temporary)
        try:
            robo_dopamine_runner.resolve_goal_image(
                {
                    "id": "rollout-c",
                    "task_suite": "libero_spatial",
                    "goal_image_path": "goals/missing.jpg",
                },
                make_args(root),
                make_config(root),
            )
        except FileNotFoundError as error:
            assert "missing.jpg" in str(error)
        else:
            raise AssertionError("Missing manifest goal_image_path was not rejected")


if __name__ == "__main__":
    test_manifest_goal_image_path_overrides_run_goal_image()
    test_absolute_manifest_goal_image_path_is_used_directly()
    test_missing_manifest_goal_image_path_fails_instead_of_falling_back()
    print("ROBODOPAMINE_MANIFEST_GOAL_IMAGE_TESTS_OK")
