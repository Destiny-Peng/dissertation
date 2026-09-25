#!/usr/bin/env python3
"""Build the LF3R baseline manifest for the current RealRobot tube batch.

This intentionally targets only:

    outputs/realrobot/tubes/*/manifest.json

The synchronized realsense_color video is the canonical baseline input.
wrist_right is retained as auxiliary metadata, but it is not silently merged
into a multiview video. Existing per-episode manifests and videos are read
only. The aggregate JSONL manifest and its summary are regenerated atomically.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs/realrobot/tubes"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/tube_manifest.jsonl"
PRIMARY_CAMERA = "realsense_color"
AUXILIARY_CAMERAS = ("wrist_right",)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def ensure_inside_project(path: Path, label: str) -> Path:
    path = path.resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must stay inside PROJECT_ROOT: {path}") from error
    return path


def relative_project_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,r_frame_rate,avg_frame_rate,duration,codec_name,codec_long_name,width,height",
        "-of",
        "json",
        str(path),
    ]
    try:
        payload = json.loads(subprocess.check_output(command, text=True))
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise RuntimeError(f"ffprobe failed for {path}: {error}") from error
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream: {path}")
    stream = streams[0]
    frame_text = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_text in (None, "", "N/A"):
        raise RuntimeError(f"ffprobe could not determine frame count: {path}")
    try:
        frame_count = int(frame_text)
        width = int(stream["width"])
        height = int(stream["height"])
        numerator, denominator = str(stream["r_frame_rate"]).split("/", 1)
        fps = float(numerator) / float(denominator)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"Invalid ffprobe metadata for {path}: {error}") from error
    if frame_count < 2 or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions/count/fps for {path}: {stream}")
    duration = stream.get("duration")
    try:
        duration_seconds = float(duration) if duration not in (None, "", "N/A") else frame_count / fps
    except (TypeError, ValueError):
        duration_seconds = frame_count / fps
    return {
        "total_frames": frame_count,
        "fps": round(fps, 6),
        "duration_seconds": round(duration_seconds, 6),
        "video_width": width,
        "video_height": height,
        "video_codec": str(stream.get("codec_name") or ""),
        "video_codec_long_name": str(stream.get("codec_long_name") or ""),
    }


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return slug or "episode"


def episode_number(label: str, fallback: int) -> int:
    match = re.search(r"_(\d+)$", label)
    return int(match.group(1)) if match else fallback


def stable_rollout_id(episode_label: str, episode_dir: Path) -> str:
    relative_dir = relative_project_path(episode_dir)
    digest = hashlib.sha1(relative_dir.encode("utf-8")).hexdigest()[:10]
    return f"realrobot-tube-{safe_slug(episode_label)}-{digest}"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def build_record(episode_dir: Path, index: int) -> dict[str, Any]:
    source_manifest_path = episode_dir / "manifest.json"
    source = read_json(source_manifest_path)
    if source.get("schema_version") != 2:
        raise ValueError(f"Expected tube recorder schema_version=2: {source_manifest_path}")
    if source.get("status") != "complete":
        raise ValueError(f"Tube recording is not complete: {source_manifest_path}")
    worker_errors = source.get("worker_errors") or []
    if worker_errors:
        raise ValueError(f"Tube recording has worker errors: {source_manifest_path}: {worker_errors}")

    episode_label = str(source.get("episode_label") or episode_dir.name)
    task = str(source.get("task") or "").strip()
    if not task:
        raise ValueError(f"Tube manifest has no task text: {source_manifest_path}")

    camera_paths: dict[str, str] = {}
    for camera in (PRIMARY_CAMERA,) + AUXILIARY_CAMERAS:
        video = episode_dir / "videos" / f"{camera}.mp4"
        if camera == PRIMARY_CAMERA and not video.is_file():
            raise FileNotFoundError(f"Missing canonical {camera} video: {video}")
        if video.is_file():
            camera_paths[camera] = relative_project_path(video)
    primary_video = episode_dir / "videos" / f"{PRIMARY_CAMERA}.mp4"
    video_metadata = probe_video(primary_video)

    config = source.get("config") if isinstance(source.get("config"), dict) else {}
    sample_hz = config.get("sample_hz")
    synchronized_frames = source.get("synchronized_frame_count")
    try:
        synchronized_frames = int(synchronized_frames)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid synchronized_frame_count: {source_manifest_path}") from error

    recording_success = "success" in episode_label.lower()
    rollout_id = stable_rollout_id(episode_label, episode_dir)
    return {
        "schema_version": 1,
        "id": rollout_id,
        "task_suite": "realrobot_tube",
        "task_id": 0,
        "episode_index": episode_number(episode_label, index),
        "episode_label": episode_label,
        "task": task,
        "task_description": task,
        "ground_truth_outcome": "success" if recording_success else "unknown",
        "source_kind": "realrobot_tube",
        "analysis_partition": "natural_observation",
        "dataset_role": "realrobot_tube",
        "camera_video_paths": camera_paths,
        "primary_camera": PRIMARY_CAMERA,
        **video_metadata,
        "synchronized_frame_count": synchronized_frames,
        "sample_hz": sample_hz,
        "recorded_duration_seconds": source.get("duration_s"),
        "recording_status": source.get("status"),
        "source_manifest_path": relative_project_path(source_manifest_path),
        "created_at_utc": source.get("created_at_utc"),
        "recording_started_at_utc": source.get("recording_started_at_utc"),
        "recorder_git_commit": source.get("recorder_git_commit"),
        "recorder_git_dirty": source.get("recorder_git_dirty"),
        "source_counts": source.get("source_counts") or {},
        "policy_family": "realrobot_recording",
        "policy_checkpoint": None,
        "csv_path": None,
        "first_environment_timestep": None,
        "last_environment_timestep": None,
        "generated_at": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        input_root = ensure_inside_project(resolve_project_path(args.input_root), "input root")
        output_path = ensure_inside_project(resolve_project_path(args.output), "output manifest")
        if not input_root.is_dir():
            raise FileNotFoundError(f"Tube input directory does not exist: {input_root}")
        episode_dirs = sorted(
            path.parent for path in input_root.glob("*/manifest.json") if path.parent.is_dir()
        )
        if not episode_dirs:
            raise FileNotFoundError(f"No per-episode manifest.json files under {input_root}")

        records = [build_record(episode_dir, index) for index, episode_dir in enumerate(episode_dirs)]
        ids = [record["id"] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Generated tube rollout IDs are not unique")
        records.sort(key=lambda record: (int(record["episode_index"]), record["id"]))
        content = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        )
        atomic_write(output_path, content)

        frame_counts = [int(record["total_frames"]) for record in records]
        codecs = Counter(record["video_codec"] for record in records)
        tasks = sorted({record["task_description"] for record in records})
        summary = {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "input_root": relative_project_path(input_root),
            "manifest": relative_project_path(output_path),
            "canonical_camera": PRIMARY_CAMERA,
            "auxiliary_cameras": list(AUXILIARY_CAMERAS),
            "total_rollouts": len(records),
            "task_count": len(tasks),
            "tasks": tasks,
            "frame_count_min": min(frame_counts),
            "frame_count_max": max(frame_counts),
            "codec_counts": dict(sorted(codecs.items())),
            "analysis_partition": "natural_observation",
            "dataset_role": "realrobot_tube",
            "baseline_usage": {
                "data_root": "PROJECT_ROOT",
                "manifest_argument": relative_project_path(output_path),
                "video_field": "camera_video_paths",
                "task_field": "task_description",
            },
        }
        atomic_write(
            output_path.with_name(f"{output_path.stem}.summary.json"),
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
