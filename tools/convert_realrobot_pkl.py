#!/usr/bin/env python3
"""Convert real-robot transition pickles into LF3R baseline rollouts.

The source files are transition dictionaries whose image observations are
stored as NumPy arrays.  This converter writes an MP4 for each input pickle
and a manifest JSONL that can be passed to ``run_lf3r_baseline.py``.  Source
pickles are never changed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _pick_frame(observations: Any, preferred: str, history_index: int) -> np.ndarray:
    if not isinstance(observations, dict):
        raise ValueError("transition must contain an observations mapping")
    candidates = [preferred, "side_policy_256", "image", "rgb", "agentview_rgb"]
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


def _write_video(frames: np.ndarray, output: Path, fps: float) -> tuple[int, int, int]:
    bgr = _frames_to_bgr(frames)
    height, width = bgr[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output}")
    try:
        for frame in bgr:
            if frame.shape[:2] != (height, width):
                raise ValueError("frame sequence contains inconsistent dimensions")
            writer.write(frame)
    finally:
        writer.release()
    return len(bgr), width, height


def _safe_id(stem: str, index: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-._") or f"episode-{index:05d}"
    return f"realrobot-{index:05d}-{clean}"


def convert_one(source: Path, project_root: Path, output_root: Path, index: int, preferred: str, history_index: int, fps: float, task: str, append_final_next: bool, write_video: bool = True) -> dict[str, Any]:
    transitions = _load_source(source)
    frames = [_pick_frame(item.get("observations"), preferred, history_index) for item in transitions]
    if append_final_next and transitions[-1].get("next_observations") is not None:
        frames.append(_pick_frame(transitions[-1]["next_observations"], preferred, history_index))
    frames = np.stack(frames, axis=0)
    if frames.shape[0] < 2:
        raise ValueError(f"{source} has fewer than two video frames")
    rollout_id = _safe_id(source.stem, index)
    video_path = output_root / "videos" / f"{rollout_id}.mp4"
    if write_video:
        count, width, height = _write_video(frames, video_path, fps)
    else:
        count, height, width = len(frames), int(frames.shape[1]), int(frames.shape[2])
    metadata = transitions[0]
    task_text = metadata.get("task_description") or metadata.get("language_instruction") or metadata.get("instruction") or task
    success = metadata.get("episode_success")
    outcome = "success" if success is True or success == 1 else "unknown"
    return {
        "id": rollout_id,
        "video_path": str(video_path.relative_to(project_root)),
        "task_suite": "realrobot",
        "task_description": str(task_text),
        "analysis_partition": "natural_observation",
        "dataset_role": "realrobot",
        "total_frames": count,
        "fps": float(fps),
        "ground_truth_outcome": outcome,
        "source_kind": "realrobot_pkl",
        "source_pkl": str(source.relative_to(project_root)),
        "observation_key": preferred,
        "history_index": history_index,
        "append_final_next": append_final_next,
        "video_width": width,
        "video_height": height,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/realrobot/demo_buffer"))
    parser.add_argument("--output", type=Path, default=Path("outputs/realrobot/baseline_rollouts"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--observation-key", default="side_policy_256", help="Image key under each transition observation")
    parser.add_argument("--history-index", type=int, default=-1, help="Frame index from each stacked observation; default -1 uses the latest history frame")
    parser.add_argument("--task", default="real-robot demonstration", help="Task text passed to the baseline model when the PKL has no instruction field")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--append-final-next", action=argparse.BooleanOptionalAction, default=True, help="Append the final transition's next_observations frame (default: enabled)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing converted videos and replace the manifest")
    parser.add_argument("--resume", action="store_true", help="Keep existing converted videos and rebuild the complete manifest")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    input_root = (project_root / args.input).resolve() if not args.input.is_absolute() else args.input.resolve()
    output_root = (project_root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"input directory does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = (output_root / "manifest.jsonl") if args.manifest is None else ((project_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve())
    sources = sorted(input_root.glob("*.pkl"))
    if not sources:
        raise SystemExit(f"no .pkl files found under {input_root}")
    rows = []
    for index, source in enumerate(sources):
        expected_video = output_root / "videos" / f"{_safe_id(source.stem, index)}.mp4"
        exists = expected_video.exists()
        if exists and not args.overwrite and not args.resume:
            raise SystemExit(f"output exists; pass --resume to keep it or --overwrite to replace: {expected_video}")
        row = convert_one(
            source, project_root, output_root, index, args.observation_key,
            args.history_index, args.fps, args.task, args.append_final_next,
            write_video=not exists or args.overwrite,
        )
        rows.append(row)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and not args.overwrite and not args.resume:
        raise SystemExit(f"manifest exists; pass --resume to rebuild or --overwrite to replace: {manifest}")
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"converted": len(rows), "manifest": str(manifest), "output": str(output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
