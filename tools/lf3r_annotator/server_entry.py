#!/usr/bin/env python3
"""LF3R annotator entrypoint with ProcVLM checkpoint compatibility fixes.

The core server historically validated every ``model_path`` with ``Path.is_file()``.
That rejects normal Hugging Face checkpoint directories and, in particular,
ProcVLM one-shot LoRA adapter directories.  Keep the server API unchanged while
using the path semantics expected by the baseline runners.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import server


def _validate_baseline_options(
    self: server.BaselineService,
    baseline: str,
    raw: Any,
) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise server.ValidationError("options must be a JSON object")

    unknown = set(raw) - server.BASELINE_ADVANCED_FIELDS
    if unknown:
        raise server.ValidationError(
            "Unknown baseline option(s): " + ", ".join(sorted(unknown))
        )
    unsupported = set(raw) - server.BASELINE_METHOD_OPTION_FIELDS[baseline]
    if unsupported:
        raise server.ValidationError(
            f"Options not supported by {baseline}: " + ", ".join(sorted(unsupported))
        )

    options = dict(raw)
    integer_fields = {
        "tensor_parallel_size": (1, 32),
        "procvlm_window_size": (1, 4096),
        "procvlm_max_sampled_frames": (1, 1_000_000),
        "procvlm_max_new_tokens": (1, 1_000_000),
        "rynn_num_frames": (1, 1_000_000),
        "rynn_num_steps": (1, 1_000_000),
        "rynn_evaluation_interval": (1, 1_000_000),
        "rynn_max_image_side": (1, 8192),
        "rynn_max_new_tokens": (1, 1_000_000),
        "robo_frame_interval": (1, 1_000_000),
        "robo_batch_size": (1, 4096),
        "densereward_frame_interval": (1, 1_000_000),
        "densereward_max_new_tokens": (1, 1_000_000),
    }
    for name, (minimum, maximum) in integer_fields.items():
        if name in options and options[name] is not None:
            options[name] = self._integer(options[name], name, minimum, maximum)

    if "rynn_batch_size" in options and options["rynn_batch_size"] is not None:
        options["rynn_batch_size"] = self._positive_integer(
            options["rynn_batch_size"], "rynn_batch_size"
        )

    for name in ("dtype", "robo_eval_mode"):
        if name in options and options[name] is not None:
            value = str(options[name]).strip()
            if not value or len(value) > 80:
                raise server.ValidationError(f"{name} must be a non-empty short string")
            options[name] = value

    if (
        "robo_eval_mode" in options
        and options["robo_eval_mode"]
        not in {"fused", "forward", "incremental", "backward"}
    ):
        raise server.ValidationError(
            "robo_eval_mode must be fused, forward, incremental, or backward"
        )

    for name in ("robot_description", "camera_description"):
        if name in options and options[name] is not None:
            value = str(options[name])
            if len(value) > 4000:
                raise server.ValidationError(f"{name} is too long")
            options[name] = value

    # Model checkpoints are normally directories. ProcVLM one-shot LoRA
    # checkpoints are adapter directories containing adapter_config.json.
    if "model_path" in options and options["model_path"] not in (None, ""):
        resolved = self._project_path(str(options["model_path"]))
        if not resolved.exists():
            raise server.ValidationError(
                f"model_path does not exist inside the project: {options['model_path']}"
            )
        options["model_path"] = str(resolved)

        # ProcVLM's official inference.py routes both --use_lora and
        # --enable_value_head through batch_chat_with_value_head().  Its
        # loader recognizes adapter_config.json and applies the PEFT adapter.
        # Auto-select that same loader for an adapter directory so WebUI LoRA
        # runs cannot accidentally be sent to the vLLM full-checkpoint path.
        if (
            baseline == "procvlm"
            and resolved.is_dir()
            and (resolved / "adapter_config.json").is_file()
        ):
            options["procvlm_enable_value_head"] = True

    if "goal_image" in options and options["goal_image"] not in (None, ""):
        resolved = self._project_path(str(options["goal_image"]))
        if not resolved.is_file():
            raise server.ValidationError(
                f"goal_image does not exist inside the project: {options['goal_image']}"
            )
        options["goal_image"] = str(resolved)

    for name in (
        "render_video",
        "validate_environment",
        "dry_run",
        "procvlm_enable_value_head",
    ):
        if name in options and not isinstance(options[name], bool):
            raise server.ValidationError(f"{name} must be boolean")

    return options


# Patch only the validation seam; request routing and job supervision remain the
# existing server implementation.
server.BaselineService._validate_options = _validate_baseline_options


if __name__ == "__main__":
    server.main()
