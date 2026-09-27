#!/usr/bin/env python3
"""Run one Ctrl-World autoregressive suffix from LF3R-prepared inputs.

This script executes inside the user's Ctrl-World environment. It imports the
released Ctrl-World source without modifying it and standardizes the generated
three-view rollout back to LF3R's two physical LIBERO views.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--svd-model-path", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--data-stat-path", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--exterior-1", type=Path, required=True)
    parser.add_argument("--exterior-2", type=Path, required=True)
    parser.add_argument("--wrist", type=Path, required=True)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-fps", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-text-conditioning",
        action="store_true",
        help="Disable the released Ctrl-World CLIP text condition.",
    )
    return parser.parse_args()


def normalize_bound(
    data: Any,
    data_min: Any,
    data_max: Any,
    *,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
) -> Any:
    import numpy as np

    normalized = 2.0 * (data - data_min) / (data_max - data_min + 1e-8) - 1.0
    return np.clip(normalized, clip_min, clip_max)


def load_rgb(path: Path) -> Any:
    import numpy as np
    from PIL import Image

    frame = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Unexpected Ctrl-World condition image shape: {frame.shape}")
    return frame


def encode_condition_views(model: Any, paths: list[Path], device: Any, dtype: Any) -> Any:
    import numpy as np
    import torch

    latents = []
    with torch.no_grad():
        for path in paths:
            frame = load_rgb(path)
            tensor = (
                torch.from_numpy(np.asarray(frame))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=dtype)
                / 255.0
                * 2.0
                - 1.0
            )
            latent = (
                model.pipeline.vae.encode(tensor)
                .latent_dist.sample()
                .mul_(model.pipeline.vae.config.scaling_factor)
            )
            latents.append(latent)
    first_latent = torch.cat(latents, dim=2)
    if tuple(first_latent.shape[1:]) != (4, 72, 40):
        raise ValueError(
            "Ctrl-World expects three 192x320 views producing a stacked "
            f"(4,72,40) latent, got {tuple(first_latent.shape)}"
        )
    return first_latent


def split_view_latents(latents: Any) -> Any:
    import torch

    if latents.ndim != 5 or latents.shape[0] != 1:
        raise ValueError(f"Unexpected Ctrl-World latent shape: {tuple(latents.shape)}")
    batch, frames, channels, height, width = latents.shape
    if height % 3:
        raise ValueError(
            f"Ctrl-World stacked latent height must be divisible by 3, got {height}"
        )
    view_height = height // 3
    return (
        latents.reshape(batch, frames, channels, 3, view_height, width)
        .permute(0, 3, 1, 2, 4, 5)[0]
        .contiguous()
    )


def decode_view_latents(model: Any, view_latents: Any) -> Any:
    import numpy as np
    import torch

    views, frames = view_latents.shape[:2]
    flat = view_latents.reshape(-1, *view_latents.shape[2:])
    decoded = []
    chunk_size = 7
    with torch.no_grad():
        for start in range(0, len(flat), chunk_size):
            chunk = flat[start : start + chunk_size]
            chunk = chunk / model.pipeline.vae.config.scaling_factor
            kwargs = {"num_frames": int(len(chunk))}
            decoded.append(model.pipeline.vae.decode(chunk, **kwargs).sample)
    video = torch.cat(decoded, dim=0)
    video = video.reshape(views, frames, *video.shape[1:])
    video = ((video / 2.0 + 0.5).clamp(0, 1) * 255.0)
    return (
        video.detach()
        .to(torch.float32)
        .cpu()
        .numpy()
        .transpose(0, 1, 3, 4, 2)
        .astype(np.uint8)
    )


def load_checkpoint_state(path: Path) -> Any:
    import torch

    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    return payload


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"Ctrl-World source root does not exist: {source_root}")
    sys.path.insert(0, str(source_root))

    import mediapy
    import numpy as np
    import torch

    from config import wm_args
    from models.ctrl_world import CrtlWorld
    from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

    if args.output_fps <= 0:
        raise SystemExit("--output-fps must be positive")
    if args.num_inference_steps < 1:
        raise SystemExit("--num-inference-steps must be at least 1")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("Ctrl-World generation requires CUDA")

    config = wm_args(task_type="replay")
    config.svd_model_path = str(args.svd_model_path.resolve())
    config.clip_model_path = str(args.clip_model_path.resolve())
    config.ckpt_path = str(args.checkpoint.resolve())
    config.val_model_path = config.ckpt_path
    config.data_stat_path = str(args.data_stat_path.resolve())
    config.num_frames = 5
    config.num_history = 6
    config.action_dim = 7
    config.width = 320
    config.height = 192
    config.num_inference_steps = int(args.num_inference_steps)
    config.guidance_scale = float(args.guidance_scale)
    config.decode_chunk_size = 7
    config.text_cond = not args.no_text_conditioning
    config.frame_level_cond = True
    config.his_cond_zero = False

    model = CrtlWorld(config)
    state_dict = load_checkpoint_state(args.checkpoint.resolve())
    model.load_state_dict(state_dict)
    dtype = getattr(config, "dtype", torch.bfloat16)
    model.to(device).to(dtype)
    model.eval()

    stats = json.loads(args.data_stat_path.read_text(encoding="utf-8"))
    state_p01 = np.asarray(stats["state_01"], dtype=np.float32)[None, :]
    state_p99 = np.asarray(stats["state_99"], dtype=np.float32)[None, :]
    if state_p01.shape != (1, 7) or state_p99.shape != (1, 7):
        raise ValueError(
            "Ctrl-World DROID stat.json must contain state_01/state_99 with seven values"
        )

    payload = np.load(args.controls.resolve(), allow_pickle=False)
    controls = np.asarray(payload["controls"], dtype=np.float32)
    source_indices = np.asarray(payload["source_frame_indices"], dtype=np.int64)
    if controls.ndim != 2 or controls.shape[1] != 7 or len(controls) < 2:
        raise ValueError(f"Ctrl-World controls must be [T,7] with T>=2, got {controls.shape}")
    if len(source_indices) != len(controls):
        raise ValueError("Ctrl-World control/source index lengths do not match")

    first_latent = encode_condition_views(
        model,
        [
            args.exterior_1.resolve(),
            args.exterior_2.resolve(),
            args.wrist.resolve(),
        ],
        device,
        dtype,
    )

    # Match the released autoregressive replay initialization: the first
    # observation fills the sparse history buffer, then each 5-frame prediction
    # advances by four new temporal samples.
    history_latents = [first_latent] * (config.num_history * 4)
    history_controls = [controls[0:1]] * (config.num_history * 4)
    history_idx = [0, 0, -8, -6, -4, -2]

    per_view_frames: list[list[Any]] = [[], [], []]
    start = 0
    chunk_index = 0
    while start < len(controls):
        valid_count = min(config.num_frames, len(controls) - start)
        chunk_controls = np.asarray(
            controls[start : start + valid_count],
            dtype=np.float32,
        )
        if valid_count < config.num_frames:
            padding = np.repeat(
                chunk_controls[-1:],
                config.num_frames - valid_count,
                axis=0,
            )
            chunk_controls = np.concatenate([chunk_controls, padding], axis=0)

        history_pose = np.concatenate(
            [history_controls[index] for index in history_idx],
            axis=0,
        )
        action_cond = np.concatenate([history_pose, chunk_controls], axis=0)
        action_cond = normalize_bound(
            action_cond,
            state_p01,
            state_p99,
        )
        action_tensor = (
            torch.from_numpy(action_cond)
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )
        history_input = (
            torch.cat(
                [history_latents[index] for index in history_idx],
                dim=0,
            )
            .unsqueeze(0)
        )
        current_latent = history_latents[-1]

        with torch.no_grad():
            if config.text_cond:
                text_token = model.action_encoder(
                    action_tensor,
                    args.instruction,
                    model.tokenizer,
                    model.text_encoder,
                )
            else:
                text_token = model.action_encoder(action_tensor)
            _, predicted_latents = CtrlWorldDiffusionPipeline.__call__(
                model.pipeline,
                image=current_latent,
                text=text_token,
                width=config.width,
                height=int(config.height * 3),
                num_frames=config.num_frames,
                history=history_input,
                num_inference_steps=config.num_inference_steps,
                decode_chunk_size=config.decode_chunk_size,
                max_guidance_scale=config.guidance_scale,
                fps=config.fps,
                motion_bucket_id=config.motion_bucket_id,
                output_type="latent",
                return_dict=False,
                frame_level_cond=True,
            )

        split_latents = split_view_latents(predicted_latents)
        decoded = decode_view_latents(model, split_latents)

        take_start = 0 if chunk_index == 0 else 1
        take_stop = valid_count
        for view_index in range(3):
            for frame in decoded[view_index, take_start:take_stop]:
                per_view_frames[view_index].append(frame)

        if start + config.num_frames - 1 >= len(controls) - 1:
            break
        last_views = split_latents[:, config.num_frames - 1]
        history_latents.append(
            last_views.permute(1, 0, 2, 3)
            .reshape(
                1,
                last_views.shape[1],
                last_views.shape[0] * last_views.shape[2],
                last_views.shape[3],
            )
            .contiguous()
        )
        history_controls.append(
            chunk_controls[config.num_frames - 1 : config.num_frames]
        )
        start += config.num_frames - 1
        chunk_index += 1

    if not per_view_frames[0] or len(per_view_frames[0]) != len(controls):
        raise RuntimeError(
            "Ctrl-World autoregressive assembly produced an unexpected frame count: "
            f"generated={len(per_view_frames[0])}, controls={len(controls)}"
        )

    # Standard LF3R generated artifacts are suffix-only. Ctrl-World predicts a
    # sample aligned with the condition pose first, so remove that first sample.
    suffix_views = [frames[1:] for frames in per_view_frames]
    if not suffix_views[0]:
        raise RuntimeError("Ctrl-World produced no suffix after removing condition-aligned frame")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cam_high": (args.output_dir / "cam_high.mp4", suffix_views[0]),
        "ctrl_world_exterior_2_duplicate": (
            args.output_dir / "ctrl_world_exterior_2_duplicate.mp4",
            suffix_views[1],
        ),
        "cam_wrist": (args.output_dir / "cam_wrist.mp4", suffix_views[2]),
    }
    for _, (path, frames) in outputs.items():
        mediapy.write_video(
            str(path),
            np.stack(
                [np.asarray(frame, dtype=np.uint8) for frame in frames],
                axis=0,
            ),
            fps=float(args.output_fps),
        )

    metadata = {
        "model": "ctrl_world",
        "checkpoint": str(args.checkpoint.resolve()),
        "source_root": str(source_root),
        "view_order": ["exterior_1", "exterior_2", "wrist"],
        "published_view_mapping": {
            "cam_high": "exterior_1",
            "cam_wrist": "wrist",
        },
        "debug_duplicate_output": "ctrl_world_exterior_2_duplicate.mp4",
        "condition_aligned_frame_removed": True,
        "control_points": int(len(controls)),
        "generated_suffix_frames": int(len(suffix_views[0])),
        "source_frame_indices": [int(x) for x in source_indices.tolist()],
        "generated_real_frame_indices": [
            int(x) for x in source_indices[1:].tolist()
        ],
        "generated_real_start_frame": int(source_indices[1]),
        "output_fps": float(args.output_fps),
        "svd_microcondition_fps": int(config.fps),
        "num_frames_per_chunk": int(config.num_frames),
        "num_history": int(config.num_history),
        "num_inference_steps": int(config.num_inference_steps),
        "guidance_scale": float(config.guidance_scale),
        "seed": int(args.seed),
        "text_conditioning": bool(config.text_cond),
        "instruction": str(args.instruction),
    }
    (args.output_dir / "ctrl_world_generation.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
