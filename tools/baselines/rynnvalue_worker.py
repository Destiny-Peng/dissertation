#!/usr/bin/env python3
"""Run the existing RynnValue CLI while preserving raw value-head outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of uniformly sampled frames in each prefix (must be positive)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=16,
        help="Number of prefix endpoints; the LF3R runner computes this from the target interval",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=8,
        help="Target source-frame interval used by the runner when calculating num-steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prefix sub-samples processed per forward pass; choose based on available GPU memory",
    )
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--robot-description", required=True)
    parser.add_argument("--camera-description", required=True)
    parser.add_argument("--render-video", action="store_true")
    args = parser.parse_args()
    if args.num_frames < 1:
        parser.error("--num-frames must be positive; all-frame-per-prefix mode is not supported")
    if args.num_steps < 1:
        parser.error("--num-steps must be positive; all-frame endpoint evaluation is not supported")
    if args.evaluation_interval < 1:
        parser.error("--evaluation-interval must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    inference_dir = (args.repo / "rynn_infer").resolve()
    for required in (inference_dir / "inference.py", args.model_path, args.video_path):
        if not required.exists():
            raise FileNotFoundError(required)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(inference_dir))
    import torch

    torch.backends.cudnn.enabled = False
    import inference as official

    captured: dict = {"values": None, "sampled_indices": None, "analysis_text": None, "parsed_analysis": None}
    output_path: dict[str, Path] = {}
    original_build_output_path = official.build_output_path
    original_parse_analysis = official.parse_analysis
    original_save_video = official.save_video_with_trend

    def capture_output_path(cli_args):
        resolved = Path(original_build_output_path(cli_args)).resolve()
        output_path["value"] = resolved
        return str(resolved)

    def capture_analysis(text: str):
        parsed = original_parse_analysis(text)
        captured["analysis_text"] = text
        captured["parsed_analysis"] = parsed
        return parsed

    def capture_values(*, images, value, output_path: str, sampled_indices=None, **kwargs):
        captured["values"] = [float(item) for item in value]
        captured["sampled_indices"] = [int(item) for item in sampled_indices] if sampled_indices is not None else None
        if args.render_video:
            return original_save_video(
                images=images,
                value=value,
                output_path=output_path,
                sampled_indices=sampled_indices,
                **kwargs,
            )
        print("Visualization skipped; raw value-head outputs are still saved.")
        return None

    official.build_output_path = capture_output_path
    official.parse_analysis = capture_analysis
    official.save_video_with_trend = capture_values
    effective_batch_size = args.batch_size
    print(
        "RynnValue approximate fixed-interval mode: target interval="
        f"{args.evaluation_interval} source frames; the runner supplied "
        f"{args.num_steps} uniformly spaced prefix endpoints; batch_size={args.batch_size}.",
        flush=True,
    )
    sys.argv = [
        str(inference_dir / "inference.py"),
        "--model_path", str(args.model_path.resolve()),
        "--video_path", str(args.video_path.resolve()),
        "--instruction", args.instruction,
        "--output_path", str(args.output_dir.resolve()),
        "--num_frames", str(args.num_frames),
        "--num_steps", str(args.num_steps),
        "--batch_size", str(effective_batch_size),
        "--max_image_side", str(args.max_image_side),
        "--max_new_tokens", str(args.max_new_tokens),
        "--robot_description", args.robot_description,
        "--camera_description", args.camera_description,
    ]
    official.main()

    if captured["values"] is None or "value" not in output_path:
        raise RuntimeError("RynnValue completed without exposing value-head outputs")
    raw_path = output_path["value"] / "raw_model_outputs.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({
        "schema_version": 1,
        "baseline": "rynnvalue",
        "model_path": str(args.model_path.resolve()),
        "video_path": str(args.video_path.resolve()),
        "instruction": args.instruction,
        "num_frames": args.num_frames,
        "num_frames_mode": "uniform_subsample",
        "sampling_mode": "approximate_fixed_interval_prefix_uniform",
        "evaluation_interval": args.evaluation_interval,
        "num_steps": args.num_steps,
        "requested_batch_size": args.batch_size,
        "effective_batch_size": effective_batch_size,
        **captured,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"Raw RynnValue outputs: {raw_path}")


if __name__ == "__main__":
    main()
