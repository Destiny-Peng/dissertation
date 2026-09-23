#!/usr/bin/env python3
"""Post-hoc LF3R localization inference over saved Robo-Dopamine fused outputs.

This tool never initializes Robo-Dopamine/vLLM. It loads one saved tiny BiLSTM
localization checkpoint on CPU, applies it to one or more existing
worker_result.json files, and writes checkpoint-keyed sidecars next to those
baseline artifacts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from robo_localization_head.core import TinyBiLSTM  # noqa: E402

_FRAME_RE = re.compile(r"(?:frame_([0-9]+)[.]png|af_([0-9]+)$)")


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    resolved.relative_to(project_root)
    return resolved


def frame_index(row: dict[str, Any]) -> int:
    explicit = row.get("frame_index")
    if explicit is not None:
        return int(explicit)
    images = row.get("image") or []
    candidates = [images[5]] if len(images) > 5 else []
    candidates.append(row.get("id", ""))
    for candidate in candidates:
        match = _FRAME_RE.search(str(candidate))
        if match:
            return int(match.group(1) or match.group(2))
    raise ValueError(
        "Cannot determine Robo-Dopamine frame index from row "
        + repr(row.get("id"))
    )


def checkpoint_bundle(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Localization checkpoint is not a mapping")
    state = payload.get("model_state_dict")
    config = payload.get("config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError(
            "Localization checkpoint must contain model_state_dict and config"
        )
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("Localization checkpoint config is missing model settings")
    hidden = int(model_config.get("hidden", 16))
    model = TinyBiLSTM(hidden=hidden)
    model.load_state_dict(state)
    model.eval()

    mean = np.asarray(
        torch.as_tensor(payload.get("normalization_mean")).cpu().numpy(),
        dtype=np.float32,
    ).reshape(1, 2)
    std = np.asarray(
        torch.as_tensor(payload.get("normalization_std")).cpu().numpy(),
        dtype=np.float32,
    ).reshape(1, 2)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("Localization checkpoint normalization is not finite")
    if np.any(std <= 0):
        raise ValueError("Localization checkpoint normalization std must be positive")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": path,
        "sha256": digest,
        "model": model,
        "mean": mean,
        "std": std,
        "hidden": hidden,
        "stage": payload.get("stage"),
        "config_id": payload.get("config_id"),
        "repeat": payload.get("repeat"),
        "seed": payload.get("seed"),
    }


def fused_prediction_path(project_root: Path, worker_result: Path) -> Path:
    payload = json.loads(worker_result.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid worker result: {worker_result}")

    value = payload.get("fused_model_output")
    if not value and payload.get("multi_perspective"):
        value = payload.get("raw_model_output")
    if not value:
        eval_modes = payload.get("eval_modes")
        if isinstance(eval_modes, list) and set(eval_modes) == {
            "incremental", "forward", "backward"
        }:
            value = payload.get("raw_model_output")
    if not value:
        raise ValueError(
            "Saved Robo-Dopamine result has no fused output; post-hoc localization "
            "requires an existing fused progress/hop artifact"
        )
    path = project_path(project_root, str(value))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def infer_one(
    project_root: Path,
    worker_result: Path,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    prediction_path = fused_prediction_path(project_root, worker_result)
    rows = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Fused prediction is empty: {prediction_path}")

    frames: list[int] = []
    features: list[list[float]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Fused prediction contains a non-object row")
        progress = float(row["progress"])
        hop = float(row["hop"])
        if not math.isfinite(progress) or not math.isfinite(hop):
            raise ValueError("Fused prediction contains non-finite progress/hop")
        frames.append(frame_index(row))
        features.append([progress, hop])

    sequence = np.asarray(features, dtype=np.float32)
    normalized = (sequence - bundle["mean"]) / bundle["std"]
    tensor = torch.from_numpy(normalized).unsqueeze(0)
    with torch.no_grad():
        logits = bundle["model"](tensor)[0].cpu()
    predicted_index = int(torch.argmax(logits).item())
    predicted_frame = int(frames[predicted_index])
    scores = torch.sigmoid(logits)

    output_root = worker_result.parent / "posthoc_localization"
    output_root.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", bundle["path"].stem)
    output_path = output_root / (
        f"{safe_stem}-{bundle['sha256'][:12]}.json"
    )
    result = {
        "schema_version": 1,
        "kind": "posthoc_robo_localization",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(bundle["path"]),
        "checkpoint_sha256": bundle["sha256"],
        "checkpoint_stage": bundle.get("stage"),
        "checkpoint_config_id": bundle.get("config_id"),
        "checkpoint_repeat": bundle.get("repeat"),
        "checkpoint_seed": bundle.get("seed"),
        "hidden": bundle["hidden"],
        "source_worker_result": str(worker_result),
        "source_prediction": str(prediction_path),
        "input": "saved_fused_robo_dopamine_progress_hop",
        "frame_count": len(frames),
        "frames": frames,
        "logits": [float(value) for value in logits.tolist()],
        "sigmoid_scores": [float(value) for value in scores.tolist()],
        "predicted_index": predicted_index,
        "predicted_frame": predicted_frame,
        "predicted_logit": float(logits[predicted_index].item()),
        "predicted_sigmoid": float(scores[predicted_index].item()),
    }
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        **result,
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--worker-result",
        action="append",
        type=Path,
        required=True,
        help="Existing Robo-Dopamine worker_result.json; may be repeated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    checkpoint = project_path(project_root, args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    # Explicitly keep this lightweight inference off the GPU.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    bundle = checkpoint_bundle(checkpoint)
    results = []
    errors = []
    for raw_path in args.worker_result:
        try:
            worker_result = project_path(project_root, raw_path)
            if not worker_result.is_file():
                raise FileNotFoundError(worker_result)
            full = infer_one(project_root, worker_result, bundle)
            results.append({
                key: full.get(key)
                for key in (
                    "source_worker_result",
                    "source_prediction",
                    "output_path",
                    "checkpoint",
                    "checkpoint_sha256",
                    "checkpoint_config_id",
                    "checkpoint_repeat",
                    "predicted_index",
                    "predicted_frame",
                    "predicted_logit",
                    "predicted_sigmoid",
                    "frame_count",
                )
            })
        except Exception as error:  # per-rollout failure should not drop the batch
            errors.append({
                "worker_result": str(raw_path),
                "error": str(error),
            })

    print(json.dumps({
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": bundle["sha256"],
        "results": results,
        "errors": errors,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
