#!/usr/bin/env python3
"""Generate successful ManiSkill3 motion-planning rollouts for LF3R.

This wrapper deliberately leaves the pinned ManiSkill checkout untouched.  It
reuses the official Panda motion-planning runner and only injects camera width
and height into the environment construction so WebUI render resolution is
explicit and reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mani_skill.examples.motionplanning.panda import run as maniskill_run

from build_maniskill3_manifest import DEFAULT_OUTPUT, build_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_ENVS = tuple(maniskill_run.MP_SOLUTIONS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", choices=SUPPORTED_ENVS, default="PickCube-v1")
    parser.add_argument("--num-traj", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument(
        "--shader",
        choices=("default", "rt-fast", "rt"),
        default="default",
    )
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> None:
    if args.num_traj < 1 or args.num_traj > 50:
        raise SystemExit("--num-traj must be between 1 and 50")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    for name, value in (
        ("render width", args.render_width),
        ("render height", args.render_height),
    ):
        if value < 64 or value > 2048 or value % 2:
            raise SystemExit(f"{name} must be an even integer from 64 to 2048")


def main() -> None:
    args = parse_args()
    _validate(args)
    record_dir = args.record_dir.expanduser().resolve()
    record_dir.mkdir(parents=True, exist_ok=True)

    original_make = maniskill_run.gym.make

    def make_with_resolution(env_id: str, **kwargs):
        for key in (
            "sensor_configs",
            "human_render_camera_configs",
            "viewer_camera_configs",
        ):
            config = dict(kwargs.get(key) or {})
            config["width"] = args.render_width
            config["height"] = args.render_height
            kwargs[key] = config
        return original_make(env_id, **kwargs)

    # The pinned upstream runner calls this module-level gym.make from _main.
    # We restore it in finally so importing this wrapper never leaks state.
    maniskill_run.gym.make = make_with_resolution
    try:
        upstream_args = maniskill_run.parse_args(
            [
                "--env-id",
                args.env_id,
                "--num-traj",
                str(args.num_traj),
                "--only-count-success",
                "--save-video",
                "--obs-mode",
                "none",
                "--sim-backend",
                "cpu",
                "--render-mode",
                "rgb_array",
                "--shader",
                args.shader,
                "--traj-name",
                "trajectory",
                "--record-dir",
                str(record_dir),
                "--num-procs",
                "1",
            ]
        )
        trajectory_path = Path(
            maniskill_run._main(upstream_args, start_seed=args.seed)
        ).resolve()
    finally:
        maniskill_run.gym.make = original_make

    video_dir = record_dir / args.env_id / "motionplanning"
    videos = sorted(video_dir.glob("*.mp4"))
    if not trajectory_path.is_file():
        raise SystemExit(f"ManiSkill3 trajectory file was not created: {trajectory_path}")
    if len(videos) < args.num_traj:
        raise SystemExit(
            f"Expected {args.num_traj} successful rollout videos, found {len(videos)}"
        )

    manifest_summary = build_manifest(
        project_root=PROJECT_ROOT,
        scan_root=PROJECT_ROOT / "outputs" / "maniskill3",
        output=DEFAULT_OUTPUT,
    )

    print(f"LF3R_MANISKILL3_TRAJECTORY={trajectory_path}")
    print(f"LF3R_MANISKILL3_VIDEO_DIR={video_dir}")
    print(f"LF3R_MANISKILL3_SUCCESS_ROLLOUTS={len(videos)}")
    print(f"LF3R_MANISKILL3_RENDER={args.render_width}x{args.render_height}")
    print(f"LF3R_MANISKILL3_MANIFEST={DEFAULT_OUTPUT}")
    print(
        "LF3R_MANISKILL3_MANIFEST_ROLLOUTS="
        + str(manifest_summary["total_rollouts"])
    )


if __name__ == "__main__":
    main()
