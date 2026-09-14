#!/usr/bin/env python3
"""Run the existing RynnValue CLI while preserving both value heads.

The official inference script currently serializes only the absolute value
head through ``save_video_with_trend``.  The model forward already returns the
relative ``<relative_value>`` head in the same call, so this wrapper captures
that output with a thin model proxy instead of running a second inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


def _nested_output_field(output, name):
    """Read a field from a ModelOutput/dataclass or a mapping."""
    if output is None:
        return None
    if isinstance(output, dict):
        return output.get(name)
    return getattr(output, name, None)


def _relative_prediction_rows(prediction, batch_size: int) -> list[list[float]]:
    """Normalize official relative-head predictions to one row per sample.

    RynnValue's interleaved history emits one relative slot between each pair
    of images.  The official head therefore commonly returns a flat tensor of
    ``batch_size * (num_images - 1)`` values.  The small amount of shape
    handling below also covers a leading head dimension and keeps this wrapper
    tolerant of checkpoint/config variants without changing the official
    forward path.
    """
    if prediction is None or batch_size < 1:
        return []
    tensor = prediction.detach().float().cpu()
    if tensor.numel() == 0:
        return []

    # The relative head is normally single-headed and flat.  If a checkpoint
    # exposes a leading head dimension, average only that dimension when the
    # remaining tensor already has one row per batch item.
    if tensor.dim() == 3:
        if tensor.shape[0] == batch_size:
            tensor = tensor.mean(dim=1)
        elif tensor.shape[1] == batch_size:
            tensor = tensor.mean(dim=0)
        else:
            tensor = tensor.reshape(batch_size, -1)
    elif tensor.dim() == 2:
        if tensor.shape[0] == batch_size:
            pass
        elif tensor.shape[1] == batch_size:
            tensor = tensor.transpose(0, 1)
        elif tensor.numel() % batch_size == 0:
            tensor = tensor.reshape(batch_size, -1)
        else:
            tensor = tensor.reshape(batch_size, -1)
    else:
        tensor = tensor.reshape(batch_size, -1)

    rows: list[list[float]] = []
    for row in tensor.tolist():
        if not isinstance(row, list):
            row = [row]
        normalized = []
        for value in row:
            number = float(value)
            normalized.append(number if math.isfinite(number) else None)
        rows.append(normalized)
    return rows


def _last_finite(values) -> float | None:
    """Return the last finite scalar from one relative-slot row."""
    for value in reversed(values or []):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and abs(number) != float("inf"):
            return number
    return None


class _ModelOutputCaptureProxy:
    """Keep the official model instance while observing ordinary forward calls."""

    def __init__(self, model, callback):
        self._model = model
        self._callback = callback

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        outputs = self._model(*args, **kwargs)
        input_ids = kwargs.get("input_ids")
        batch_size = int(input_ids.shape[0]) if input_ids is not None else 1
        self._callback(outputs, batch_size)
        return outputs

    def to(self, *args, **kwargs):
        self._model = self._model.to(*args, **kwargs)
        return self

    def eval(self):
        self._model.eval()
        return self


def _capture_relative_model(official, callback):
    """Replace only the official AutoModel factory with a capture proxy.

    Returns the original factory object so callers can restore the module
    after ``official.main()``.  Fake contract runners without AutoModel remain
    untouched.
    """
    original_auto_model = getattr(official, "AutoModel", None)
    if original_auto_model is None or not hasattr(original_auto_model, "from_pretrained"):
        return None

    class AutoModelCaptureFactory:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            model = original_auto_model.from_pretrained(*args, **kwargs)
            return _ModelOutputCaptureProxy(model, callback)

    official.AutoModel = AutoModelCaptureFactory
    return original_auto_model


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

    captured: dict = {
        "values": None,
        "sampled_indices": None,
        "relative_values": None,
        "relative_values_by_prefix": None,
        "relative_sampled_indices": None,
        "relative_slot_counts": None,
        "relative_output_available": False,
        "analysis_text": None,
        "parsed_analysis": None,
    }
    relative_rows: list[list[float]] = []
    output_path: dict[str, Path] = {}
    original_build_output_path = official.build_output_path
    original_parse_analysis = official.parse_analysis
    original_save_video = official.save_video_with_trend

    def capture_model_outputs(outputs, batch_size: int):
        relative_output = _nested_output_field(outputs, "relative")
        prediction = _nested_output_field(relative_output, "pred_value")
        rows = _relative_prediction_rows(prediction, batch_size)
        if rows:
            relative_rows.extend(rows)

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
        if relative_rows:
            expected = len(captured["values"])
            if len(relative_rows) != expected:
                print(
                    "WARNING: relative-head prefix count "
                    f"{len(relative_rows)} does not match value count {expected}; "
                    "retaining the common prefix only.",
                    flush=True,
                )
            aligned_rows = list(relative_rows[:expected])
            if len(aligned_rows) < expected:
                aligned_rows.extend([[] for _ in range(expected - len(aligned_rows))])
            captured["relative_values_by_prefix"] = aligned_rows
            captured["relative_values"] = [_last_finite(row) for row in aligned_rows]
            captured["relative_sampled_indices"] = list(captured["sampled_indices"] or [])
            captured["relative_slot_counts"] = [len(row) for row in aligned_rows]
            captured["relative_output_available"] = any(
                value is not None for value in captured["relative_values"]
            )
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
    original_auto_model = _capture_relative_model(official, capture_model_outputs)
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
    try:
        official.main()
    finally:
        official.build_output_path = original_build_output_path
        official.parse_analysis = original_parse_analysis
        official.save_video_with_trend = original_save_video
        if original_auto_model is not None:
            official.AutoModel = original_auto_model

    if captured["values"] is None or "value" not in output_path:
        raise RuntimeError("RynnValue completed without exposing value-head outputs")
    raw_path = output_path["value"] / "raw_model_outputs.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({
        "schema_version": 2,
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
        "relative_output_semantics": (
            "official <relative_value> head; relative_values contains the last "
            "relative slot for each prefix, while relative_values_by_prefix "
            "preserves every native relative slot"
        ),
        **captured,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"Raw RynnValue outputs: {raw_path}")


if __name__ == "__main__":
    main()
