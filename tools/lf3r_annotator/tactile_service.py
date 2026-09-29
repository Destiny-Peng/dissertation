"""FailRecovery tactile reader used only by the LF3R Results WebUI."""

from __future__ import annotations

import bisect
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
KINDS = ("raw", "deform")


@dataclass
class EpisodeTactile:
    rollout_id: str
    frames: list[dict[str, Any]]
    camera_frames: dict[str, list[int]]
    camera_rows: dict[str, list[int]]
    events: dict[str, dict[str, Any]]
    finger_events: dict[tuple[str, str], dict[str, Any]]
    streams: dict[str, dict[str, Path]]


class FailRecoveryTactileService:
    """Small, dataset-specific index for failrecovery_manifest.jsonl."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = (
            self.project_root
            / "datasets"
            / "lf3r_failure_rollouts"
            / "v1"
            / "failrecovery_manifest.jsonl"
        )
        self.episodes: dict[str, EpisodeTactile] = {}
        self._load()

    def _project_file(self, value: Any) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("missing project-relative tactile path")
        path = (self.project_root / value).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("tactile path escapes project root") from exc
        return path

    @staticmethod
    def _jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
        return rows

    def _stream_paths(self, row: dict[str, Any]) -> dict[str, dict[str, Path]]:
        raw = row.get("tactile_stream_paths")
        if not isinstance(raw, dict):
            return {}
        result: dict[str, dict[str, Path]] = {}
        for finger in FINGERS:
            entry = raw.get(finger)
            finger_paths: dict[str, Path] = {}
            if isinstance(entry, dict):
                for kind in KINDS:
                    value = (
                        entry.get(kind)
                        or entry.get(kind + "_path")
                        or entry.get(kind + "_u8")
                    )
                    if value:
                        finger_paths[kind] = self._project_file(value)
            for kind in KINDS:
                if kind in finger_paths:
                    continue
                value = (
                    raw.get(finger + "_" + kind)
                    or raw.get(finger + "." + kind)
                    or raw.get(finger + "_" + kind + "_path")
                )
                if value:
                    finger_paths[kind] = self._project_file(value)
            if finger_paths:
                result[finger] = finger_paths
        return result

    def _load(self) -> None:
        if not self.manifest_path.is_file():
            return
        try:
            manifest_rows = self._jsonl(self.manifest_path)
        except (OSError, json.JSONDecodeError):
            return

        for row in manifest_rows:
            rollout_id = str(row.get("id") or "").strip()
            if not rollout_id:
                continue
            try:
                frames_path = self._project_file(row.get("synchronized_frames_path"))
                events_path = self._project_file(row.get("tactile_events_path"))
                streams = self._stream_paths(row)
                frames = self._jsonl(frames_path)
                event_rows = self._jsonl(events_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue

            events: dict[str, dict[str, Any]] = {}
            finger_events: dict[tuple[str, str], dict[str, Any]] = {}
            for event in event_rows:
                event_id = event.get("event_id")
                if event_id is None:
                    continue
                key = str(event_id)
                events[key] = event
                finger = str(event.get("finger") or event.get("finger_name") or "").strip().lower()
                if finger:
                    finger_events[(finger, key)] = event

            camera_pairs: dict[str, list[tuple[int, int]]] = {}
            for row_index, frame_row in enumerate(frames):
                indices = frame_row.get("camera_frame_indices")
                if not isinstance(indices, dict):
                    continue
                for camera, value in indices.items():
                    try:
                        frame_number = int(value)
                    except (TypeError, ValueError):
                        continue
                    camera_pairs.setdefault(str(camera), []).append((frame_number, row_index))

            camera_frames: dict[str, list[int]] = {}
            camera_rows: dict[str, list[int]] = {}
            for camera, pairs in camera_pairs.items():
                pairs.sort(key=lambda item: item[0])
                camera_frames[camera] = [item[0] for item in pairs]
                camera_rows[camera] = [item[1] for item in pairs]

            self.episodes[rollout_id] = EpisodeTactile(
                rollout_id=rollout_id,
                frames=frames,
                camera_frames=camera_frames,
                camera_rows=camera_rows,
                events=events,
                finger_events=finger_events,
                streams=streams,
            )

    def has_rollout(self, rollout_id: str) -> bool:
        return rollout_id in self.episodes

    @staticmethod
    def _nearest_index(values: list[int], requested: int) -> int:
        if not values:
            raise KeyError("camera has no synchronized frames")
        pos = bisect.bisect_left(values, requested)
        if pos <= 0:
            return 0
        if pos >= len(values):
            return len(values) - 1
        before = values[pos - 1]
        after = values[pos]
        return pos - 1 if abs(requested - before) <= abs(after - requested) else pos

    def _event(self, episode: EpisodeTactile, finger: str, event_id: Any) -> dict[str, Any] | None:
        if event_id is None:
            return None
        key = str(event_id)
        return episode.finger_events.get((finger, key)) or episode.events.get(key)

    def frame(self, rollout_id: str, camera: str, video_frame: int) -> dict[str, Any]:
        episode = self.episodes.get(rollout_id)
        if episode is None:
            raise KeyError("tactile rollout not found")
        frames = episode.camera_frames.get(camera)
        rows = episode.camera_rows.get(camera)
        if not frames or not rows:
            raise KeyError("camera has no tactile synchronization")

        index = self._nearest_index(frames, int(video_frame))
        row_index = rows[index]
        row = episode.frames[row_index]
        tactile = row.get("tactile") if isinstance(row.get("tactile"), dict) else {}

        fingers: dict[str, Any] = {}
        for finger in FINGERS:
            sync = tactile.get(finger) if isinstance(tactile.get(finger), dict) else {}
            event_id = sync.get("event_id")
            event = self._event(episode, finger, event_id)
            if event is None and event_id is None:
                continue
            event = event or {}
            fingers[finger] = {
                "event_id": event_id,
                "timestamp": event.get("timestamp"),
                "f6": event.get("f6"),
                "valid": event.get("valid"),
                "stale": sync.get("stale"),
                "age_ms": sync.get("age_ms"),
                "image_kinds": [
                    kind for kind in KINDS
                    if kind in episode.streams.get(finger, {})
                    and event.get(kind + "_offset_bytes") is not None
                    and event.get(kind + "_length_bytes") is not None
                    and event.get(kind + "_shape") is not None
                ],
            }

        return {
            "available": True,
            "rollout_id": rollout_id,
            "camera": camera,
            "requested_video_frame": int(video_frame),
            "matched_video_frame": frames[index],
            "sync_row": row_index,
            "complete": row.get("complete"),
            "timestamp": row.get("timestamp"),
            "fingers": fingers,
        }

    @staticmethod
    def _png_chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    @classmethod
    def _png(cls, data: bytes, shape: Any) -> bytes:
        if not isinstance(shape, (list, tuple)):
            raise ValueError("invalid tactile image shape")
        dims = [int(value) for value in shape]
        if len(dims) == 2:
            height, width = dims
            channels = 1
        elif len(dims) == 3 and dims[2] in (1, 3, 4):
            height, width, channels = dims
        else:
            raise ValueError("unsupported tactile image shape")
        if height <= 0 or width <= 0:
            raise ValueError("invalid tactile image dimensions")
        expected = height * width * channels
        if len(data) != expected:
            raise ValueError("tactile image byte length does not match shape")

        color_type = {1: 0, 3: 2, 4: 6}[channels]
        stride = width * channels
        scanlines = b"".join(
            b"\x00" + data[offset:offset + stride]
            for offset in range(0, len(data), stride)
        )
        signature = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
        return (
            signature
            + cls._png_chunk(b"IHDR", ihdr)
            + cls._png_chunk(b"IDAT", zlib.compress(scanlines, level=3))
            + cls._png_chunk(b"IEND", b"")
        )

    def image(self, rollout_id: str, finger: str, event_id: str, kind: str) -> bytes:
        if finger not in FINGERS or kind not in KINDS:
            raise KeyError("unknown tactile image")
        episode = self.episodes.get(rollout_id)
        if episode is None:
            raise KeyError("tactile rollout not found")
        event = self._event(episode, finger, event_id)
        stream = episode.streams.get(finger, {}).get(kind)
        if event is None or stream is None:
            raise KeyError("tactile image not found")

        offset = int(event[kind + "_offset_bytes"])
        length = int(event[kind + "_length_bytes"])
        shape = event[kind + "_shape"]
        if offset < 0 or length <= 0:
            raise ValueError("invalid tactile image byte range")
        with stream.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise ValueError("short tactile image read")
        return self._png(data, shape)
