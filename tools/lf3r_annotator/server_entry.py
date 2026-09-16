#!/usr/bin/env python3
"""LF3R annotator entrypoint with ProcVLM checkpoint compatibility fixes.

The core server historically validated every ``model_path`` with ``Path.is_file()``.
That rejects normal Hugging Face checkpoint directories and ProcVLM one-shot LoRA
adapter directories.  This entrypoint keeps the server API stable while exposing
an explicit ``procvlm_use_lora`` option and translating it to the existing
persistent ProcVLM worker's PEFT/value-head loading path.
"""

from __future__ import annotations

from typing import Any

import server


# ``server.py`` predates the explicit LoRA mode. Extend its allowlists before
# request validation so batch and single-rollout calls can use the semantic
# option without changing the legacy server module in-place.
server.BASELINE_METHOD_OPTION_FIELDS["procvlm"].add("procvlm_use_lora")
server.BASELINE_ADVANCED_FIELDS.add("procvlm_use_lora")


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

    for name in (
        "render_video",
        "validate_environment",
        "dry_run",
        "procvlm_enable_value_head",
        "procvlm_use_lora",
    ):
        if name in options and not isinstance(options[name], bool):
            raise server.ValidationError(f"{name} must be boolean")

    model_path = None
    is_procvlm_adapter = False
    if "model_path" in options and options["model_path"] not in (None, ""):
        model_path = self._project_path(str(options["model_path"]))
        if not model_path.exists():
            raise server.ValidationError(
                f"model_path does not exist inside the project: {options['model_path']}"
            )
        options["model_path"] = str(model_path)
        is_procvlm_adapter = bool(
            baseline == "procvlm"
            and model_path.is_dir()
            and (model_path / "adapter_config.json").is_file()
        )

    if baseline == "procvlm":
        use_lora = bool(options.get("procvlm_use_lora", False))
        if use_lora:
            if model_path is None:
                raise server.ValidationError(
                    "ProcVLM One-shot LoRA mode requires model_path to point to the saved LoRA checkpoint directory"
                )
            if not is_procvlm_adapter:
                raise server.ValidationError(
                    "ProcVLM One-shot LoRA mode requires a checkpoint directory containing adapter_config.json"
                )
            # The current persistent worker reaches ProcVLM's official PEFT
            # loader through the same PyTorch/value-head branch used by
            # upstream ``--use_lora``. Keep that implementation detail hidden
            # from the UI while preserving the explicit LoRA request semantic.
            options["procvlm_enable_value_head"] = True
        elif is_procvlm_adapter:
            raise server.ValidationError(
                "model_path is a ProcVLM LoRA adapter checkpoint; select Inference mode = One-shot LoRA"
            )

    if "goal_image" in options and options["goal_image"] not in (None, ""):
        resolved = self._project_path(str(options["goal_image"]))
        if not resolved.is_file():
            raise server.ValidationError(
                f"goal_image does not exist inside the project: {options['goal_image']}"
            )
        options["goal_image"] = str(resolved)

    return options


_original_baseline_command = server.BaselineService._baseline_command


def _baseline_command_with_procvlm_lora(
    self: server.BaselineService,
    baseline: str,
    scope: str,
    gpu: str,
    utilization: float,
    run_parent,
    options: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> list[str]:
    # ``run_lf3r_baseline.py`` currently exposes the PyTorch/value-head switch,
    # not a separate LoRA flag. Validation above already verified the adapter
    # directory and enabled that loader. Strip only the WebUI semantic flag
    # before forwarding the remaining runner options.
    forwarded = dict(options)
    forwarded.pop("procvlm_use_lora", None)
    return _original_baseline_command(
        self,
        baseline,
        scope,
        gpu,
        utilization,
        run_parent,
        forwarded,
        *args,
        **kwargs,
    )


server.BaselineService._validate_options = _validate_baseline_options
server.BaselineService._baseline_command = _baseline_command_with_procvlm_lora


if __name__ == "__main__":
    server.main()
