#!/usr/bin/env python3
"""Export the selected RealRobot fail-recovery media for LF3R.

The source episode SQLite files and recordings are read only. Each exported
episode contains two RGB videos, five tactile byte streams, a tactile event
index, and a compact synchronized-frame index. Paths in the aggregate manifest
are relative to PROJECT_ROOT, as expected by the LF3R annotator.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "outputs/realrobot/failrecovery"
DATASET_ROOT = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1"
EPISODE_ROOT = DATASET_ROOT / "failrecovery"
MANIFEST_PATH = DATASET_ROOT / "failrecovery_manifest.jsonl"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CAMERAS = {"cam_high": "realsense_color", "cam_wrist": "wrist_right"}


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate,width,height,codec_name,duration",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.check_output(command, text=True))
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream: {path}")
    stream = streams[0]
    frames = stream.get("nb_frames")
    if frames in (None, "N/A"):
        command[5:5] = ["-count_frames"]
        command[command.index("stream=nb_frames,r_frame_rate,width,height,codec_name,duration")] = (
            "stream=nb_read_frames,nb_frames,r_frame_rate,width,height,codec_name,duration"
        )
        stream = json.loads(subprocess.check_output(command, text=True))["streams"][0]
        frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    return {
        "frames": int(frames),
        "fps": numerator / denominator,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream.get("codec_name"),
        "duration_seconds": float(stream.get("duration") or 0),
    }


def ensure_h264(path: Path, expected_frames: int | None = None) -> dict[str, Any]:
    before = probe_video(path)
    if expected_frames is not None and before["frames"] != expected_frames:
        raise ValueError(f"Video frame count changed: {path}")
    if before["codec"] == "h264":
        return before
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".mp4", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(path), "-map", "0:v:0", "-an",
                "-fps_mode", "passthrough", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(temporary),
            ],
            check=True,
        )
        after = probe_video(temporary)
        if after["codec"] != "h264" or after["frames"] != before["frames"]:
            raise ValueError(f"H.264 frame count mismatch: {path}")
        os.replace(temporary, path)
        return after
    finally:
        temporary.unlink(missing_ok=True)


def source_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def export_tactile(connection: sqlite3.Connection, destination: Path) -> dict[str, Any]:
    tactile_dir = destination / "tactile"
    tactile_dir.mkdir()
    counts = {finger: 0 for finger in FINGERS}
    event_ids: set[int] = set()
    with ExitStack() as stack:
        index = stack.enter_context((tactile_dir / "events.jsonl").open("w", encoding="utf-8"))
        outputs = {
            finger: (
                stack.enter_context((tactile_dir / f"{finger}.raw.u8").open("wb")),
                stack.enter_context((tactile_dir / f"{finger}.deform.u8").open("wb")),
            )
            for finger in FINGERS
        }
        query = """
            SELECT e.id, e.source, e.sensor_ts_ns, e.receive_wall_ns,
                   e.receive_mono_ns, e.valid, e.source_time_status,
                   t.side, t.finger, t.channel, t.local_channel, t.f6_json,
                   t.raw_shape_json, t.deform_shape_json, t.raw_blob, t.deform_blob
            FROM events AS e JOIN tactile_frames AS t ON t.event_id = e.id
            WHERE e.source LIKE 'tactile:right:%'
            ORDER BY e.id
        """
        for row in connection.execute(query):
            finger = row["finger"]
            if finger not in outputs or row["side"] != "right":
                raise ValueError(f"Unexpected tactile source: {row['source']}")
            raw = row["raw_blob"]
            deform = row["deform_blob"]
            raw_shape = json.loads(row["raw_shape_json"])
            deform_shape = json.loads(row["deform_shape_json"])
            if raw is None or deform is None:
                raise ValueError(f"Missing tactile image at event {row['id']}")
            if len(raw) != raw_shape[0] * raw_shape[1]:
                raise ValueError(f"Unexpected tactile raw size at event {row['id']}")
            if len(deform) != deform_shape[0] * deform_shape[1]:
                raise ValueError(f"Unexpected tactile deform size at event {row['id']}")
            raw_file, deform_file = outputs[finger]
            record = {
                "event_id": row["id"], "finger": finger,
                "sample_index": counts[finger], "sensor_ts_ns": row["sensor_ts_ns"],
                "receive_wall_ns": row["receive_wall_ns"],
                "receive_mono_ns": row["receive_mono_ns"],
                "valid": bool(row["valid"]),
                "source_time_status": row["source_time_status"],
                "channel": row["channel"], "local_channel": row["local_channel"],
                "f6": json.loads(row["f6_json"]),
                "raw_shape": raw_shape, "deform_shape": deform_shape,
                "raw_offset_bytes": raw_file.tell(), "raw_length_bytes": len(raw),
                "deform_offset_bytes": deform_file.tell(),
                "deform_length_bytes": len(deform),
            }
            raw_file.write(raw)
            deform_file.write(deform)
            index.write(json_line(record))
            event_ids.add(row["id"])
            counts[finger] += 1
    return {"counts": counts, "event_ids": event_ids}


def export_sync_frames(
    connection: sqlite3.Connection, destination: Path, event_ids: set[int]
) -> dict[str, int]:
    frames = 0
    complete = 0
    missing_tactile = 0
    path = destination / "frames.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in connection.execute(
            "SELECT frame_index, elapsed_s, tick_wall_ns, tick_mono_ns, "
            "complete, stale_sources_json, snapshot_json "
            "FROM synchronized_frames ORDER BY frame_index"
        ):
            snapshot = json.loads(row["snapshot_json"])
            cameras = {}
            for key, source in CAMERAS.items():
                camera = snapshot.get(f"camera:{source}")
                cameras[key] = camera.get("frame_index") if camera else None
            tactile = {}
            for finger in FINGERS:
                item = snapshot.get(f"tactile:right:{finger}")
                event_id = item.get("event_id") if item else None
                if event_id is not None and event_id not in event_ids:
                    missing_tactile += 1
                tactile[finger] = {
                    "event_id": event_id,
                    "age_ms": item.get("age_ms") if item else None,
                    "stale": item.get("stale") if item else None,
                    "valid": item.get("valid") if item else None,
                }
            handle.write(json_line({
                "frame_index": row["frame_index"], "elapsed_s": row["elapsed_s"],
                "tick_wall_ns": row["tick_wall_ns"],
                "tick_mono_ns": row["tick_mono_ns"],
                "complete": bool(row["complete"]),
                "stale_sources": json.loads(row["stale_sources_json"]),
                "camera_frame_indices": cameras,
                "tactile": tactile,
            }))
            frames += 1
            complete += bool(row["complete"])
    if missing_tactile:
        raise ValueError(f"{missing_tactile} synchronized tactile references are missing")
    return {"frames": frames, "complete": complete}


def export_episode(source: Path, destination: Path) -> dict[str, Any]:
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != 3:
        raise ValueError(f"Unsupported recorder schema: {source}")
    if source_manifest.get("status") != "complete":
        raise ValueError(f"Incomplete source recording: {source}")
    if destination.exists():
        metadata_path = destination / "export.json"
        if not metadata_path.is_file():
            raise ValueError(f"Existing episode has no export metadata: {destination}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, camera in CAMERAS.items():
            metadata["videos"][key] = ensure_h264(
                destination / "videos" / f"{camera}.mp4",
                metadata["videos"][key]["frames"],
            )
        atomic_text(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        return metadata

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{source.name}.", dir=destination.parent))
    try:
        videos = temp_dir / "videos"
        videos.mkdir()
        video_info = {}
        for key, camera in CAMERAS.items():
            src = source / "videos" / f"{camera}.mp4"
            target = videos / src.name
            shutil.copy2(src, target)
            video_info[key] = ensure_h264(target)
        with source_db(source / "episode.sqlite3") as connection:
            tactile = export_tactile(connection, temp_dir)
            sync = export_sync_frames(connection, temp_dir, tactile["event_ids"])
            for key, camera in CAMERAS.items():
                count, maximum = connection.execute(
                    "SELECT COUNT(*), MAX(frame_index) FROM camera_frames WHERE source=?",
                    (f"camera:{camera}",),
                ).fetchone()
                if count != video_info[key]["frames"] or maximum != count - 1:
                    raise ValueError(f"Camera index/video count mismatch: {source.name} {camera}")
        if sync["frames"] != source_manifest["synchronized_frame_count"]:
            raise ValueError(f"Synchronized frame count mismatch: {source}")
        metadata = {
            "schema_version": 1, "episode": source.name,
            "source_manifest_path": rel(source / "manifest.json"),
            "source_database_path": rel(source / "episode.sqlite3"),
            "videos": video_info, "synchronized": sync,
            "tactile_counts": tactile["counts"],
            "tactile_encoding": "uint8 row-major; byte offsets and shapes in tactile/events.jsonl",
            "tactile_fingers": list(FINGERS),
        }
        (temp_dir / "export.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temp_dir, destination)
        return metadata
    except BaseException:
        shutil.rmtree(temp_dir)
        raise


def manifest_record(source: Path, destination: Path, index: int, meta: dict[str, Any]) -> dict[str, Any]:
    recorded = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    primary = meta["videos"]["cam_high"]
    digest = hashlib.sha1(rel(source).encode("utf-8")).hexdigest()[:10]
    return {
        "schema_version": 1,
        "id": f"realrobot-failrecovery-{source.name}-{digest}",
        "task_suite": "realrobot_failrecovery", "task_id": 0,
        "episode_index": index, "episode_label": recorded["episode_label"],
        "task": recorded["task"], "task_description": recorded["task"],
        "ground_truth_outcome": "unknown",
        "source_kind": "realrobot_failrecovery",
        "analysis_partition": "natural_observation",
        "dataset_role": "realrobot_failrecovery",
        "camera_video_paths": {
            key: rel(destination / "videos" / f"{camera}.mp4")
            for key, camera in CAMERAS.items()
        },
        "total_frames": primary["frames"], "fps": primary["fps"],
        "duration_seconds": primary["duration_seconds"],
        "video_width": primary["width"], "video_height": primary["height"],
        "video_codec": primary["codec"],
        "synchronized_frame_count": meta["synchronized"]["frames"],
        "complete_synchronized_frame_count": meta["synchronized"]["complete"],
        "synchronized_frames_path": rel(destination / "frames.jsonl"),
        "tactile_events_path": rel(destination / "tactile/events.jsonl"),
        "tactile_stream_paths": {
            finger: {
                "raw": rel(destination / "tactile" / f"{finger}.raw.u8"),
                "deform": rel(destination / "tactile" / f"{finger}.deform.u8"),
            }
            for finger in FINGERS
        },
        "tactile_counts": meta["tactile_counts"],
        "sample_hz": recorded["config"]["sample_hz"],
        "recorded_duration_seconds": recorded["duration_s"],
        "recording_status": recorded["status"],
        "source_manifest_path": rel(source / "manifest.json"),
        "policy_family": "realrobot_recording", "policy_checkpoint": None,
        "csv_path": None, "first_environment_timestep": None,
        "last_environment_timestep": None,
    }


def save_goal_image(record: dict[str, Any], path: Path) -> dict[str, Any]:
    video = PROJECT_ROOT / record["camera_video_paths"]["cam_high"]
    final_frame = int(record["total_frames"]) - 1
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(video), "-vf", f"select=eq(n\\,{final_frame})",
                "-vsync", "0", "-frames:v", "1", str(temporary),
            ],
            check=True,
        )
        if temporary.stat().st_size == 0:
            raise ValueError(f"No final frame extracted from {video}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": rel(path), "source_rollout_id": record["id"],
        "camera": "cam_high", "video_frame_index": final_frame,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=EPISODE_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = args.manifest.resolve()
    for path in (source_root, output_root, manifest_path):
        path.relative_to(PROJECT_ROOT)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    sources = sorted(path.parent for path in source_root.glob("*/manifest.json"))
    if not sources:
        raise ValueError(f"No episodes under {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, source in enumerate(sources):
        destination = output_root / source.name
        print(f"[{index + 1}/{len(sources)}] {source.name}", flush=True)
        metadata = export_episode(source, destination)
        records.append(manifest_record(source, destination, index, metadata))
    goal = save_goal_image(records[0], output_root / "goal_image.png")
    atomic_text(manifest_path, "".join(json_line(record) for record in records))
    summary = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_root": rel(source_root), "episode_root": rel(output_root),
        "manifest": rel(manifest_path), "total_rollouts": len(records),
        "camera_keys": list(CAMERAS), "tactile_fingers": list(FINGERS),
        "synchronized_frames": sum(r["synchronized_frame_count"] for r in records),
        "complete_synchronized_frames": sum(
            r["complete_synchronized_frame_count"] for r in records
        ),
        "tactile_samples": sum(
            sum(r["tactile_counts"].values()) for r in records
        ),
        "ground_truth_outcome": "unknown",
        "goal_image": goal,
    }
    atomic_text(
        manifest_path.with_name(manifest_path.stem + ".summary.json"),
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
