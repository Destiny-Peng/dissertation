#!/usr/bin/env python3
"""Convert real-robot transition pickles into multi-view LF3R rollouts.

By default, each input pickle is exported as separate ``side_policy_256`` and
``wrist_1`` videos, with a baseline manifest for each view. Source pickles are
never changed.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _pick_frame(
    observations: Any,
    preferred: str,
    history_index: int,
    *,
    allow_fallback: bool = True,
) -> np.ndarray:
    if not isinstance(observations, dict):
        raise ValueError("transition must contain an observations mapping")
    candidates = (
        [preferred, "side_policy_256", "image", "rgb", "agentview_rgb"]
        if allow_fallback
        else [preferred]
    )
    for name in candidates:
        value = observations.get(name)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim == 4:
            # Real-robot files use (history, H, W, C). Also accept CHW history.
            if array.shape[-1] in (1, 3, 4):
                index = history_index if history_index >= 0 else array.shape[0] + history_index
                if not 0 <= index < array.shape[0]:
                    raise ValueError(f"history index {history_index} is out of range for {name}: {array.shape}")
                return array[index]
            if array.shape[1] in (1, 3, 4):
                index = history_index if history_index >= 0 else array.shape[0] + history_index
                if not 0 <= index < array.shape[0]:
                    raise ValueError(f"history index {history_index} is out of range for {name}: {array.shape}")
                return np.transpose(array[index], (1, 2, 0))
        if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            return array
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            return np.transpose(array, (1, 2, 0))
    available = ", ".join(sorted(str(key) for key in observations))
    raise ValueError(f"no image observation found; available keys: {available}")


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Load the NumPy transition format without importing arbitrary globals."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED:
            raise ValueError(f"blocked pickle global: {module}.{name}")
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"}:
            import numpy.core.multiarray as multiarray
            return getattr(multiarray, name)
        import numpy as np
        return getattr(np, name)


def _load_source(path: Path) -> list[dict[str, Any]]:
    """Read one transition dict while allowing only NumPy reconstruction."""
    with path.open("rb") as handle:
        value = _NumpyOnlyUnpickler(handle).load()
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(item, dict) for item in value):
        if not value:
            raise ValueError("pickle contains an empty transition list")
        return list(value)
    raise ValueError("top-level pickle object must be a transition dictionary or list of dictionaries")


def _frames_to_bgr(frames: np.ndarray) -> list[np.ndarray]:
    frames = np.asarray(frames)
    if frames.ndim == 4 and frames.shape[1] in (1, 3, 4) and frames.shape[-1] not in (1, 3, 4):
        frames = np.transpose(frames, (0, 2, 3, 1))
    if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected frames shaped (T,H,W,C) or (T,C,H,W), got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]


def _write_video(frames: np.ndarray, output: Path, fps: float, codec: str) -> tuple[int, int, int]:
    bgr = _frames_to_bgr(frames)
    height, width = bgr[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    for frame in bgr:
        if frame.shape[:2] != (height, width):
            raise ValueError("frame sequence contains inconsistent dimensions")

    if codec == "h264":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("H.264 output requires ffmpeg with the libx264 encoder")
        temporary = output.with_name(f".{output.stem}.tmp-{os.getpid()}{output.suffix}")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        try:
            result = subprocess.run(
                command,
                input=b"".join(frame.tobytes() for frame in bgr),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0:
                detail = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg H.264 encoding failed: {detail}")
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
    elif codec == "mp4v":
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {output}")
        try:
            for frame in bgr:
                writer.write(frame)
        finally:
            writer.release()
    else:
        raise ValueError(f"unsupported video codec: {codec}")
    return len(bgr), width, height


def _safe_id(stem: str, index: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-._") or f"episode-{index:05d}"
    return f"realrobot-{index:05d}-{clean}"


def _safe_camera_name(camera: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", camera).strip("-._")
    if not clean:
        raise ValueError(f"camera key cannot be converted to a safe output name: {camera!r}")
    return clean


def convert_one(
    source: Path,
    project_root: Path,
    output_root: Path,
    index: int,
    camera_keys: list[str],
    history_index: int,
    fps: float,
    task: str,
    append_final_next: bool,
    codec: str,
    write_cameras: set[str],
) -> dict[str, dict[str, Any]]:
    transitions = _load_source(source)
    rollout_id = _safe_id(source.stem, index)
    metadata = transitions[0]
    task_text = metadata.get("task_description") or metadata.get("language_instruction") or metadata.get("instruction") or task
    success = metadata.get("episode_success")
    outcome = "success" if success is True or success == 1 else "unknown"

    camera_data: dict[str, dict[str, Any]] = {}
    for camera in camera_keys:
        frames = [
            _pick_frame(item.get("observations"), camera, history_index, allow_fallback=False)
            for item in transitions
        ]
        if append_final_next and transitions[-1].get("next_observations") is not None:
            frames.append(
                _pick_frame(
                    transitions[-1]["next_observations"],
                    camera,
                    history_index,
                    allow_fallback=False,
                )
            )
        frames = np.stack(frames, axis=0)
        if frames.shape[0] < 2:
            raise ValueError(f"{source} has fewer than two video frames for camera {camera!r}")

        video_path = output_root / "videos" / _safe_camera_name(camera) / f"{rollout_id}.mp4"
        if camera in write_cameras:
            count, width, height = _write_video(frames, video_path, fps, codec)
        else:
            count, height, width = len(frames), int(frames.shape[1]), int(frames.shape[2])
        camera_data[camera] = {
            "path": str(video_path.relative_to(project_root)),
            "count": count,
            "width": width,
            "height": height,
        }

    paths = {camera: info["path"] for camera, info in camera_data.items()}
    rows: dict[str, dict[str, Any]] = {}
    for camera, info in camera_data.items():
        rows[camera] = {
            "id": rollout_id,
            "video_path": info["path"],
            "camera_video_paths": paths,
            "camera_views": list(camera_keys),
            "task_suite": "realrobot",
            "task_description": str(task_text),
            "analysis_partition": "natural_observation",
            "dataset_role": "realrobot",
            "total_frames": info["count"],
            "fps": float(fps),
            "ground_truth_outcome": outcome,
            "source_kind": "realrobot_pkl",
            "source_pkl": str(source.relative_to(project_root)),
            "observation_key": camera,
            "history_index": history_index,
            "append_final_next": append_final_next,
            "video_width": info["width"],
            "video_height": info["height"],
            "video_codec": codec,
            "video_encoder": "libx264" if codec == "h264" else "mp4v",
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/realrobot/demo_buffer"))
    parser.add_argument("--output", type=Path, default=Path("outputs/realrobot/baseline_rollouts"))
    parser.add_argument("--manifest", type=Path, default=None, help="Canonical manifest for --primary-camera")
    parser.add_argument(
        "--observation-keys",
        nargs="+",
        default=["side_policy_256", "wrist_1"],
        help="At least two camera keys to extract (default: side_policy_256 wrist_1)",
    )
    parser.add_argument(
        "--primary-camera",
        default=None,
        help="Camera used by the canonical manifest.jsonl (default: side_policy_256 when selected, otherwise first camera)",
    )
    parser.add_argument("--history-index", type=int, default=-1, help="Frame index from each stacked observation; default -1 uses the latest history frame")
    parser.add_argument("--task", default="real-robot demonstration", help="Task text passed to the baseline model when the PKL has no instruction field")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--video-codec", choices=("h264", "mp4v"), default="h264", help="Output codec; H.264/libx264 is the default")
    parser.add_argument("--append-final-next", action=argparse.BooleanOptionalAction, default=True, help="Append the final transition's next_observations frame (default: enabled)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing converted videos and replace manifests")
    parser.add_argument("--resume", action="store_true", help="Keep existing converted videos and rebuild all manifests")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_root = (project_root / args.input).resolve() if not args.input.is_absolute() else args.input.resolve()
    output_root = (project_root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"input directory does not exist: {input_root}")

    camera_keys = args.observation_keys
    if len(camera_keys) < 2:
        parser.error("at least two camera keys are required; single-view extraction is not supported")
    if len(camera_keys) != len(set(camera_keys)):
        parser.error("camera observation keys must be unique")
    camera_names = [_safe_camera_name(camera) for camera in camera_keys]
    if len(camera_names) != len(set(camera_names)):
        parser.error("camera keys collide after conversion to safe output names")
    primary_camera = args.primary_camera or (
        "side_policy_256" if "side_policy_256" in camera_keys else camera_keys[0]
    )
    if primary_camera not in camera_keys:
        parser.error("--primary-camera must be one of --observation-keys")

    manifest = (
        output_root / "manifest.jsonl"
        if args.manifest is None
        else ((project_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve())
    )
    camera_manifests = {
        camera: output_root / f"manifest_{_safe_camera_name(camera)}.jsonl"
        for camera in camera_keys
    }
    if manifest in camera_manifests.values() and manifest != camera_manifests[primary_camera]:
        parser.error("--manifest conflicts with a non-primary camera manifest path")
    manifest_paths = set(camera_manifests.values()) | {manifest}

    sources = sorted(input_root.glob("*.pkl"))
    if not sources:
        raise SystemExit(f"no .pkl files found under {input_root}")
    rollout_paths = [
        output_root / "videos" / camera_name / f"{_safe_id(source.stem, index)}.mp4"
        for index, source in enumerate(sources)
        for camera_name in camera_names
    ]
    if not args.overwrite and not args.resume:
        existing = [path for path in [*manifest_paths, *rollout_paths] if path.exists()]
        if existing:
            parser.error("output exists; pass --resume or --overwrite: " + ", ".join(str(path) for path in existing[:8]))

    rows_by_camera = {camera: [] for camera in camera_keys}
    for index, source in enumerate(sources):
        rollout_id = _safe_id(source.stem, index)
        expected = {
            camera: output_root / "videos" / _safe_camera_name(camera) / f"{rollout_id}.mp4"
            for camera in camera_keys
        }
        write_cameras = {
            camera
            for camera, path in expected.items()
            if args.overwrite or not path.exists()
        }
        rows = convert_one(
            source,
            project_root,
            output_root,
            index,
            camera_keys,
            args.history_index,
            args.fps,
            args.task,
            args.append_final_next,
            args.video_codec,
            write_cameras,
        )
        for camera, row in rows.items():
            rows_by_camera[camera].append(row)

    for camera, camera_manifest in camera_manifests.items():
        camera_manifest.parent.mkdir(parents=True, exist_ok=True)
        camera_manifest.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_by_camera[camera]),
            encoding="utf-8",
        )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest not in camera_manifests.values():
        manifest.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_by_camera[primary_camera]),
            encoding="utf-8",
        )
    print(json.dumps({
        "converted_rollouts": len(sources),
        "camera_views": camera_keys,
        "primary_camera": primary_camera,
        "primary_manifest": str(manifest),
        "camera_manifests": {camera: str(path) for camera, path in camera_manifests.items()},
        "output": str(output_root),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
