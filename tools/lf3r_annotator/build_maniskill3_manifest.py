#!/usr/bin/env python3
"""Build the standalone LF3R ManiSkill3 rollout manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAN_ROOT = PROJECT_ROOT / "outputs" / "maniskill3"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "datasets"
    / "lf3r_failure_rollouts"
    / "v1"
    / "maniskill3_manifest.jsonl"
)

MANISKILL3_ENVS = (
    "PickCube-v1",
    "StackCube-v1",
    "PegInsertionSide-v1",
    "PlugCharger-v1",
    "PlaceSphere-v1",
    "PushCube-v1",
    "PullCubeTool-v1",
    "LiftPegUpright-v1",
    "PullCube-v1",
    "DrawTriangle-v1",
    "DrawSVG-v1",
    "StackPyramid-v1",
)
TASK_IDS = {env_id: index for index, env_id in enumerate(MANISKILL3_ENVS)}
TASK_DESCRIPTIONS = {
    "PickCube-v1": "Pick up the cube",
    "StackCube-v1": "Stack the cubes",
    "PegInsertionSide-v1": "Insert the peg from the side",
    "PlugCharger-v1": "Plug in the charger",
    "PlaceSphere-v1": "Place the sphere",
    "PushCube-v1": "Push the cube",
    "PullCubeTool-v1": "Pull the cube with the tool",
    "LiftPegUpright-v1": "Lift the peg upright",
    "PullCube-v1": "Pull the cube",
    "DrawTriangle-v1": "Draw a triangle",
    "DrawSVG-v1": "Draw the SVG path",
    "StackPyramid-v1": "Stack a pyramid",
}


def _relative(path: Path, project_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError as error:
        raise RuntimeError(f"Path escapes project root: {resolved}") from error


def _probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,r_frame_rate,duration,width,height",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    frame_text = stream.get("nb_read_frames") or stream.get("nb_frames")
    if not frame_text or frame_text == "N/A":
        raise RuntimeError(f"Could not determine frame count for {path}")
    numerator, denominator = stream["r_frame_rate"].split("/", 1)
    fps = float(numerator) / float(denominator)
    frames = int(frame_text)
    duration = float(stream.get("duration") or frames / fps)
    return {
        "total_frames": frames,
        "fps": round(fps, 6),
        "duration_seconds": round(duration, 6),
        "render_width": int(stream["width"]),
        "render_height": int(stream["height"]),
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _episode_index(video: Path, fallback: int) -> int:
    match = re.fullmatch(r"(?:episode[_-]?)?(\d+)", video.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else fallback


def _stable_id(relative_video: str, env_id: str, episode_index: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", env_id).strip("-").lower()
    digest = hashlib.sha1(relative_video.encode("utf-8")).hexdigest()[:10]
    return f"maniskill3-{slug}-ep{episode_index:03d}-{digest}"


def _trajectory_files(video_dir: Path) -> tuple[Path | None, Path | None]:
    h5_candidates = sorted(video_dir.glob("*.h5"))
    json_candidates = sorted(video_dir.glob("*.json"))
    trajectory = next(
        (path for path in h5_candidates if path.stem == "trajectory"),
        h5_candidates[0] if h5_candidates else None,
    )
    metadata = next(
        (path for path in json_candidates if path.stem == "trajectory"),
        json_candidates[0] if json_candidates else None,
    )
    return trajectory, metadata


def build_manifest(
    *,
    project_root: Path = PROJECT_ROOT,
    scan_root: Path = DEFAULT_SCAN_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    scan_root = scan_root.expanduser().resolve()
    output = output.expanduser().resolve()

    grouped: dict[tuple[str, str], list[Path]] = {}
    if scan_root.is_dir():
        for video in sorted(scan_root.rglob("*.mp4")):
            if video.parent.name != "motionplanning":
                continue
            try:
                env_id = video.parents[1].name
                run_name = video.parents[2].name
            except IndexError:
                continue
            if env_id not in TASK_IDS:
                continue
            grouped.setdefault((run_name, env_id), []).append(video)

    records: list[dict[str, Any]] = []
    for (run_name, env_id), videos in sorted(grouped.items()):
        trajectory, trajectory_metadata = _trajectory_files(videos[0].parent)
        for fallback, video in enumerate(sorted(videos)):
            episode_index = _episode_index(video, fallback)
            relative_video = _relative(video, project_root)
            probe = _probe_video(video)
            record: dict[str, Any] = {
                "schema_version": 1,
                "id": _stable_id(relative_video, env_id, episode_index),
                "task_suite": "maniskill3",
                "task_id": TASK_IDS[env_id],
                "task_key": env_id,
                "episode_index": episode_index,
                "task_description": TASK_DESCRIPTIONS[env_id],
                "ground_truth_outcome": "success",
                "source_kind": "maniskill3_motionplanning_success",
                "analysis_partition": "synthetic_success",
                "dataset_role": "maniskill3",
                "camera_video_paths": {"cam_high": relative_video},
                "camera_view_semantics": {
                    "cam_high": "ManiSkill3 motion-planning render camera"
                },
                "run_name": run_name,
                "policy_family": "ManiSkill3 Panda motion planning",
                "policy_checkpoint": None,
                "generation_entrypoint": (
                    "tools/lf3r_annotator/generate_maniskill3_success.py"
                ),
                "generated_at": dt.datetime.fromtimestamp(
                    video.stat().st_mtime, tz=dt.timezone.utc
                ).isoformat(),
                "trajectory_format": "maniskill3_recordepisode_hdf5",
                "trajectory_group": f"traj_{episode_index}",
                **probe,
            }
            if trajectory is not None:
                record["trajectory_path"] = _relative(trajectory, project_root)
            if trajectory_metadata is not None:
                record["trajectory_metadata_path"] = _relative(
                    trajectory_metadata, project_root
                )
            records.append(record)

    records.sort(
        key=lambda row: (
            row["task_id"],
            row["run_name"],
            row["episode_index"],
            row["id"],
        )
    )
    manifest_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records
    )
    _atomic_write(output, manifest_text)

    counts = Counter(row["task_key"] for row in records)
    summary = {
        "schema_version": 1,
        "manifest": _relative(output, project_root),
        "scan_root": _relative(scan_root, project_root),
        "total_rollouts": len(records),
        "counts": dict(sorted(counts.items())),
    }
    summary_path = output.with_name("maniskill3_summary.json")
    _atomic_write(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_manifest(
        project_root=args.project_root,
        scan_root=args.scan_root,
        output=args.output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
