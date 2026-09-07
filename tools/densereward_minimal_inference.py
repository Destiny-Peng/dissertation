#!/usr/bin/env python3
"""Run the official DenseReward 3-frame model-card inference path once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


REASON_RE = re.compile(r"<think>\s*([A-Za-z]+)\s*</think>", re.IGNORECASE)
REWARD_RE = re.compile(r"</think>\s*([01](?:\.\d+)?)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--frame", dest="frames", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def parse_output(raw: str) -> tuple[str, float, bool]:
    reason_match = REASON_RE.search(raw)
    reward_match = REWARD_RE.search(raw)
    if reason_match is None:
        raise ValueError(f"DenseReward output has no <think> reason: {raw!r}")
    if reward_match is None:
        raise ValueError(f"DenseReward output has no scalar reward: {raw!r}")
    reason = reason_match.group(1).lower()
    reward = float(reward_match.group(1))
    if not math.isfinite(reward):
        raise ValueError(f"DenseReward reward is not finite: {raw!r}")
    clipped = not 0.0 <= reward <= 1.0
    reward = min(1.0, max(0.0, reward))
    return reason, reward, clipped


def main() -> None:
    args = parse_args()
    if len(args.frames) != 3:
        raise SystemExit("exactly three --frame arguments are required, oldest first")
    model_path = args.model_path.expanduser().resolve()
    frames = [frame.expanduser().resolve() for frame in args.frames]
    if not model_path.is_dir():
        raise SystemExit(f"missing DenseReward model directory: {model_path}")
    system_prompt_path = model_path / "system_prompt.txt"
    if not system_prompt_path.is_file():
        raise SystemExit(f"missing official system prompt: {system_prompt_path}")
    missing_frames = [str(frame) for frame in frames if not frame.is_file()]
    if missing_frames:
        raise SystemExit("missing input frame(s): " + ", ".join(missing_frames))

    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    frame_uris = [frame.as_uri() for frame in frames]
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame_uris[0]},
                {"type": "image", "image": frame_uris[1]},
                {"type": "image", "image": frame_uris[2]},
                {"type": "text", "text": args.task},
            ],
        },
    ]

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    reason, reward, clipped = parse_output(raw)
    result = {
        "model_path": str(model_path),
        "system_prompt_path": str(system_prompt_path),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "frames_chronological_oldest_to_current": [str(frame) for frame in frames],
        "task": args.task,
        "dtype": "bfloat16",
        "decoding": {"do_sample": False, "max_new_tokens": 32},
        "input_tokens": int(inputs.input_ids.shape[-1]),
        "generated_tokens": int(generated_ids.shape[-1]),
        "raw_text": raw,
        "reason": reason,
        "reward": reward,
        "reward_was_clipped": clipped,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
