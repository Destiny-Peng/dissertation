#!/usr/bin/env python3
"""Run the LF3R Ctrl-World runner on an official DROID subset trajectory.

This is a diagnostic adapter only. It converts one Ctrl-World
``dataset_example/droid_subset`` trajectory into the exact prepared-input
contract already consumed by ``ctrl_world_runner.py`` and then launches that
runner unchanged. This keeps the inference path identical to LF3R LIBERO runs
while swapping only the input domain.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_START_INDEX = {
    "899": 8,
    "18599": 14,
    "199": 8,
    "1799": 23,
}


def project_path(value: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("repos/Ctrl-World"),
    )
    parser.add_argument(
        "--droid-root",
        type=Path,
        default=Path("repos/Ctrl-World/dataset_example/droid_subset"),
    )
    parser.add_argument("--droid-id", default="199")
    parser.add_argument("--droid-start-idx", type=int, default=None)
    parser.add_argument("--interactions", type=int, default=12)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/ctrl-world/Ctrl-World/checkpoint-10000.pt"),
    )
    parser.add_argument(
        "--svd-model-path",
        type=Path,
        default=Path("checkpoints/stable-video-diffusion-img2vid"),
    )
    parser.add_argument(
        "--clip-model-path",
        type=Path,
        default=Path("checkpoints/clip-vit-base-patch32"),
    )
    parser.add_argument(
        "--data-stat-path",
        type=Path,
        default=Path("repos/Ctrl-World/dataset_meta_info/droid/stat.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-fps", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-text-conditioning", action="store_true")
    return parser.parse_args()


def read_annotation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected DROID annotation payload: {path}")
    return payload


def prepare_droid_input(
    *,
    droid_root: Path,
    trajectory_id: str,
    start_idx: int,
    interactions: int,
    prepared_dir: Path,
) -> dict[str, Any]:
    import numpy as np
    from decord import VideoReader, cpu
    from PIL import Image

    if interactions < 1:
        raise ValueError("--interactions must be at least 1")

    annotation_path = droid_root / "annotation" / "val" / f"{trajectory_id}.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"DROID annotation not found: {annotation_path}")
    annotation = read_annotation(annotation_path)

    states = np.asarray(annotation.get("states"), dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 7:
        raise ValueError(
            f"DROID annotation states must be [T,7], got {states.shape}"
        )

    action_rows = annotation.get("action")
    if isinstance(action_rows, list) and action_rows:
        official_length = len(action_rows)
    else:
        official_length = int(annotation.get("video_length") or len(states))
    usable_length = min(int(official_length), int(len(states)))
    if usable_length < 2:
        raise ValueError(f"DROID trajectory is too short: {usable_length}")
    if start_idx < 0 or start_idx >= usable_length:
        raise ValueError(
            f"DROID start index {start_idx} is outside trajectory length {usable_length}"
        )

    # Official replay uses pred_step=5 and advances by pred_step-1=4. For N
    # interactions this consumes 1 + 4*N unique trajectory samples.
    sample_count = 1 + 4 * int(interactions)
    frame_ids = np.arange(start_idx, start_idx + sample_count, dtype=np.int64)
    frame_ids = np.minimum(frame_ids, usable_length - 1)
    controls = np.asarray(states[frame_ids], dtype=np.float32)

    videos = annotation.get("videos")
    if not isinstance(videos, list) or len(videos) != 3:
        raise ValueError(
            "DROID sanity mode expects exactly three videos, matching the released replay path"
        )

    prepared_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    video_paths: list[Path] = []
    for view_index, video_info in enumerate(videos):
        if not isinstance(video_info, dict):
            raise ValueError(f"Invalid DROID video entry at index {view_index}")
        raw_video_path = str(video_info.get("video_path") or "").strip()
        if not raw_video_path:
            raise ValueError(f"DROID video entry {view_index} has no video_path")
        video_path = (droid_root / raw_video_path).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"DROID video not found: {video_path}")
        video_paths.append(video_path)

        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
        if int(frame_ids[0]) >= len(reader):
            raise ValueError(
                f"DROID condition frame {int(frame_ids[0])} exceeds video {view_index} length {len(reader)}"
            )
        frame = reader[int(frame_ids[0])].asnumpy()
        image_path = prepared_dir / f"droid_view_{view_index}_condition.png"
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(image_path)
        image_paths.append(image_path)

    controls_path = prepared_dir / "droid_controls.npz"
    np.savez_compressed(
        controls_path,
        controls=controls,
        source_frame_indices=frame_ids,
    )

    texts = annotation.get("texts")
    instruction = (
        str(texts[0])
        if isinstance(texts, list) and texts
        else ""
    )

    metadata = {
        "mode": "droid_sanity",
        "trajectory_id": str(trajectory_id),
        "annotation_path": str(annotation_path),
        "video_paths": [str(path) for path in video_paths],
        "condition_frame": int(frame_ids[0]),
        "source_frame_indices": [int(value) for value in frame_ids.tolist()],
        "control_points": int(len(controls)),
        "instruction": instruction,
        "official_replay_contract": {
            "pred_step": 5,
            "chunk_overlap": 1,
            "advance_per_interaction": 4,
            "num_history": 6,
        },
    }
    (prepared_dir / "droid_sanity_input.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        **metadata,
        "controls_path": controls_path,
        "condition_images": image_paths,
    }


def main() -> None:
    args = parse_args()
    trajectory_id = str(args.droid_id)
    start_idx = (
        int(args.droid_start_idx)
        if args.droid_start_idx is not None
        else int(DEFAULT_START_INDEX.get(trajectory_id, 0))
    )

    source_root = project_path(args.source_root)
    droid_root = project_path(args.droid_root)
    checkpoint = project_path(args.checkpoint)
    svd_model_path = project_path(args.svd_model_path)
    clip_model_path = project_path(args.clip_model_path)
    data_stat_path = project_path(args.data_stat_path)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir is not None
        else (
            PROJECT_ROOT
            / "artifacts"
            / "repair"
            / "ctrl_world_droid_sanity"
            / trajectory_id
        ).resolve()
    )
    prepared_dir = output_dir / "prepared"
    generated_dir = output_dir / "generated"

    prepared = prepare_droid_input(
        droid_root=droid_root,
        trajectory_id=trajectory_id,
        start_idx=start_idx,
        interactions=int(args.interactions),
        prepared_dir=prepared_dir,
    )
    condition_images = prepared["condition_images"]

    runner = Path(__file__).with_name("ctrl_world_runner.py").resolve()
    command = [
        sys.executable,
        str(runner),
        "--source-root",
        str(source_root),
        "--checkpoint",
        str(checkpoint),
        "--svd-model-path",
        str(svd_model_path),
        "--clip-model-path",
        str(clip_model_path),
        "--data-stat-path",
        str(data_stat_path),
        "--controls",
        str(prepared["controls_path"]),
        "--exterior-1",
        str(condition_images[0]),
        "--exterior-2",
        str(condition_images[1]),
        "--wrist",
        str(condition_images[2]),
        "--instruction",
        str(prepared["instruction"]),
        "--output-dir",
        str(generated_dir),
        "--output-fps",
        str(float(args.output_fps)),
        "--num-inference-steps",
        str(int(args.num_inference_steps)),
        "--guidance-scale",
        str(float(args.guidance_scale)),
        "--seed",
        str(int(args.seed)),
    ]
    if args.no_text_conditioning:
        command.append("--no-text-conditioning")

    print(
        "Running LF3R Ctrl-World DROID sanity check: "
        f"id={trajectory_id} start={start_idx} controls={prepared['control_points']}"
    )
    print("Output:", generated_dir)
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))

    generation_path = generated_dir / "ctrl_world_generation.json"
    generation = (
        json.loads(generation_path.read_text(encoding="utf-8"))
        if generation_path.is_file()
        else {}
    )
    if isinstance(generation, dict):
        generation["input_mode"] = "droid_sanity"
        generation["droid_trajectory_id"] = trajectory_id
        generation["droid_start_idx"] = start_idx
        generation["droid_annotation_instruction"] = prepared["instruction"]
        generation_path.write_text(
            json.dumps(generation, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
