#!/usr/bin/env python3
"""Create separate per-camera LIBERO videos by replaying recorded rollout actions.

This is intentionally a post-processing step. It does not query OpenVLA and it
never changes the canonical single-view MP4 used by LF3R baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


HIGH_CAMERA = "agentview"
WRIST_CAMERA = "robot0_eye_in_hand"
CAMERA_SLOTS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ACTION_FIELDS = (
    "action/dx",
    "action/dy",
    "action/dz",
    "action/droll",
    "action/dpitch",
    "action/dyaw",
    "action/dgripper",
)
ROLLOUT_RE = re.compile(
    r"^task(?P<task>[0-9]+)--ep(?P<episode>[0-9]+)--succ(?P<success>[01])[.]mp4$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--record-resolution", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def read_actions(path: Path) -> list[np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing rollout action CSV: {path}")
    actions: list[np.ndarray] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in ACTION_FIELDS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"Rollout CSV {path} is missing executed-action fields: {', '.join(missing)}"
            )
        for row in reader:
            actions.append(np.asarray([float(row[name]) for name in ACTION_FIELDS], dtype=np.float64))
    if not actions:
        raise ValueError(f"Rollout CSV contains no executed actions: {path}")
    return actions


def make_env(task, resolution: int) -> OffScreenRenderEnv:
    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=[HIGH_CAMERA, WRIST_CAMERA],
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    return env


def oriented_rgb(obs: dict, camera: str) -> np.ndarray:
    key = camera + "_image"
    if key not in obs:
        available = ", ".join(sorted(name for name in obs if name.endswith("_image")))
        raise KeyError(f"Missing {key}; available image observations: {available}")
    image = np.asarray(obs[key])
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unexpected {key} shape: {image.shape}")
    # LIBERO/robosuite off-screen camera observations use the same orientation
    # convention as agentview_image. Rotate 180 degrees for human-readable video.
    return np.ascontiguousarray(image[::-1, ::-1])


def sidecar_paths(video: Path) -> tuple[dict[str, Path], Path]:
    high_path = video.with_name(video.stem + ".cam_high.mp4")
    wrist_path = video.with_name(video.stem + ".cam_left_wrist.mp4")
    camera_paths = {
        "cam_high": high_path,
        "cam_left_wrist": wrist_path,
        "cam_right_wrist": wrist_path,
    }
    return camera_paths, video.with_name(video.stem + ".camera_videos.json")


def rollout_videos(run_dir: Path) -> Iterable[Path]:
    for path in sorted(run_dir.glob("*.mp4")):
        if ROLLOUT_RE.fullmatch(path.name):
            yield path


def record_rollout(
    task_suite,
    task_suite_name: str,
    video: Path,
    record_resolution: int,
    fps: float,
) -> dict[str, Path]:
    match = ROLLOUT_RE.fullmatch(video.name)
    if match is None:
        raise ValueError(f"Unexpected rollout filename: {video.name}")
    task_id = int(match.group("task"))
    episode_idx = int(match.group("episode"))
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    if episode_idx >= len(initial_states):
        raise ValueError(
            f"Episode index {episode_idx} exceeds available initial states for task {task_id}"
        )
    actions = read_actions(video.with_suffix(".csv"))
    camera_paths, metadata_path = sidecar_paths(video)
    physical_paths = list(dict.fromkeys(camera_paths.values()))
    existing = [path for path in [*physical_paths, metadata_path] if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing multiview artifact(s) for {video.name}: {names}"
        )

    env = make_env(task, record_resolution)
    writers: dict[str, object] = {}
    frame_count = 0
    try:
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

        writers = {
            "cam_high": imageio.get_writer(str(camera_paths["cam_high"]), fps=fps),
            "cam_left_wrist": imageio.get_writer(
                str(camera_paths["cam_left_wrist"]),
                fps=fps,
            ),
        }
        for action in actions:
            writers["cam_high"].append_data(oriented_rgb(obs, HIGH_CAMERA))
            writers["cam_left_wrist"].append_data(oriented_rgb(obs, WRIST_CAMERA))
            frame_count += 1
            obs, _, _, _ = env.step(action.tolist())
    except Exception:
        for path in physical_paths:
            if path.exists():
                path.unlink()
        raise
    finally:
        for writer in writers.values():
            writer.close()
        env.close()

    metadata = {
        "schema_version": 1,
        "camera_video_paths": {
            slot: path.name
            for slot, path in camera_paths.items()
        },
        "record_resolution": int(record_resolution),
        "camera_width": int(record_resolution),
        "camera_height": int(record_resolution),
        "fps": float(fps),
        "frames": frame_count,
        "source_video": video.name,
        "source_actions": video.with_suffix(".csv").name,
        "task_suite": task_suite_name,
        "task_id": task_id,
        "episode_index": episode_idx,
        "replay_only": True,
        "policy_inference_reused": True,
        "consumer_interface": "robo_dopamine_three_view",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    outputs = ",".join(f"{slot}={path.name}" for slot, path in camera_paths.items())
    print(
        "LF3R_MULTIVIEW_RECORDED "
        f"source={video.name} outputs={outputs} frames={frame_count}"
    )
    return camera_paths


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    if args.record_resolution < 64 or args.record_resolution > 2048 or args.record_resolution % 2:
        raise SystemExit("record-resolution must be an even integer between 64 and 2048")
    if args.fps <= 0:
        raise SystemExit("fps must be positive")

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite not in benchmark_dict:
        raise SystemExit(f"Unknown LIBERO task suite: {args.task_suite}")
    task_suite = benchmark_dict[args.task_suite]()

    videos = list(rollout_videos(run_dir))
    if not videos:
        raise SystemExit(f"No canonical rollout MP4s found under {run_dir}")

    outputs = [
        record_rollout(
            task_suite=task_suite,
            task_suite_name=args.task_suite,
            video=video,
            record_resolution=args.record_resolution,
            fps=args.fps,
        )
        for video in videos
    ]
    print(f"LF3R_MULTIVIEW_COMPLETE rollouts={len(outputs)}")


if __name__ == "__main__":
    main()
