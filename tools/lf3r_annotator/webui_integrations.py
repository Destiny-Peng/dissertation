#!/usr/bin/env python3
"""WebUI integrations for baseline options and non-Analysis project tools.

The core server remains the stable implementation. This entrypoint keeps the
ProcVLM LoRA compatibility layer and adds bounded non-Analysis project-tool
endpoints used by the Runs console.
"""

from __future__ import annotations

import json
import math
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import server
from non_analysis_tools import NonAnalysisToolService, gpu_status


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
        "procvlm_frame_stride": (1, 1_000_000),
        "procvlm_max_sampled_frames": (1, 1_000_000),
        "procvlm_max_new_tokens": (1, 1_000_000),
        "procvlm_tracker_support_threshold": (1, 1_000_000),
        "procvlm_tracker_window_size": (1, 1_000_000),
        "procvlm_tracker_max_forward_jump": (1, 1),
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

    for name in ("dtype", "robo_eval_mode", "procvlm_procedure_mode"):
        if name in options and options[name] is not None:
            value = str(options[name]).strip()
            if not value or len(value) > 80:
                raise server.ValidationError(f"{name} must be a non-empty short string")
            options[name] = value

    if (
        "procvlm_procedure_mode" in options
        and options["procvlm_procedure_mode"] not in {"baseline", "tracker_only", "stateful_history"}
    ):
        raise server.ValidationError(
            "procvlm_procedure_mode must be baseline, tracker_only, or stateful_history"
        )

    if baseline == "procvlm" and options.get("procvlm_procedure_mode", "baseline") == "baseline":
        for name in (
            "procvlm_procedure_config",
            "procvlm_tracker_support_threshold",
            "procvlm_tracker_window_size",
            "procvlm_tracker_max_forward_jump",
        ):
            options.pop(name, None)

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
        elif is_procvlm_adapter:
            raise server.ValidationError(
                "model_path is a ProcVLM LoRA adapter checkpoint; select Inference mode = One-shot LoRA"
            )

    if "procvlm_procedure_config" in options and options["procvlm_procedure_config"] not in (None, ""):
        resolved = self._project_path(str(options["procvlm_procedure_config"]))
        if not resolved.is_file():
            raise server.ValidationError(
                f"procvlm_procedure_config does not exist inside the project: {options['procvlm_procedure_config']}"
            )
        options["procvlm_procedure_config"] = str(resolved)

    if options.get("procvlm_tracker_support_threshold", 7) > options.get("procvlm_tracker_window_size", 9):
        raise server.ValidationError(
            "procvlm_tracker_support_threshold cannot exceed procvlm_tracker_window_size"
        )
    if (
        baseline == "procvlm"
        and options.get("procvlm_procedure_mode", "baseline") != "baseline"
        and not options.get("procvlm_procedure_config")
    ):
        raise server.ValidationError(
            "tracker_only/stateful_history ProcVLM requires procvlm_procedure_config"
        )

    if "goal_image" in options and options["goal_image"] not in (None, ""):
        resolved = self._project_path(str(options["goal_image"]))
        if not resolved.is_file():
            raise server.ValidationError(
                f"goal_image does not exist inside the project: {options['goal_image']}"
            )
        options["goal_image"] = str(resolved)

    if "robo_localization_ckpt" in options and options["robo_localization_ckpt"] not in (None, ""):
        if baseline != "robo_dopamine":
            raise server.ValidationError(
                "robo_localization_ckpt is only valid for Robo-Dopamine"
            )
        resolved = self._project_path(str(options["robo_localization_ckpt"]))
        if not resolved.is_file():
            raise server.ValidationError(
                "robo_localization_ckpt does not exist inside the project: "
                + str(options["robo_localization_ckpt"])
            )
        if resolved.suffix.lower() not in {".pt", ".pth"}:
            raise server.ValidationError(
                "robo_localization_ckpt must be a .pt or .pth checkpoint"
            )
        if options.get("robo_eval_mode", "fused") != "fused":
            raise server.ValidationError(
                "robo_localization_ckpt requires Robo-Dopamine eval mode = fused"
            )
        options["robo_localization_ckpt"] = str(resolved)

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


# Attach the non-Analysis project-tool service after the legacy application has
# registered its own job handlers. A second recover pass is intentional: the
# first pass cannot see project_tool records before this handler exists.
_original_application_init = server.LF3RApplication.__init__


def _application_init_with_tools(self, *args: Any, **kwargs: Any) -> None:
    _original_application_init(self, *args, **kwargs)
    self.project_tools = NonAnalysisToolService(self.project_root, self.tmux)
    self.tmux.recover()


server.LF3RApplication.__init__ = _application_init_with_tools


def _read_json_body(handler: server.LF3RHandler, *, maximum: int = 100_000) -> dict[str, Any]:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise ValueError("Invalid Content-Length") from exc
    if length <= 0 or length > maximum:
        raise ValueError("Invalid request size")
    payload = json.loads(handler.rfile.read(length))
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


_original_do_get = server.LF3RHandler.do_GET


def _do_get_with_tools(self: server.LF3RHandler) -> None:
    parsed = urlparse(self.path)
    path = unquote(parsed.path)
    query = parse_qs(parsed.query, keep_blank_values=True)
    try:
        if path == "/api/gpu-status":
            self.json_response(HTTPStatus.OK, {"gpu_status": gpu_status()})
            return
        if path == "/api/tool-jobs":
            status = query.get("status", [None])[0] or None
            self.json_response(
                HTTPStatus.OK,
                {
                    "jobs": self.app.project_tools.list(status=status),
                    "project_tools_protocol": "immediate-registry-v1",
                },
            )
            return
        if path.startswith("/api/tool-jobs/"):
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[3] == "log":
                job_id = parts[2]
                tail = query.get("tail", ["240"])[0]
                self.json_response(
                    HTTPStatus.OK,
                    {"log": self.app.project_tools.log(job_id, int(tail))},
                )
                return
            if len(parts) == 3:
                self.json_response(
                    HTTPStatus.OK,
                    {"job": self.app.project_tools.get(parts[2])},
                )
                return
            self.json_error(HTTPStatus.NOT_FOUND, "Not found")
            return
    except KeyError:
        self.json_error(HTTPStatus.NOT_FOUND, "Unknown project-tool job")
        return
    except (TypeError, ValueError) as exc:
        self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
        return
    return _original_do_get(self)


server.LF3RHandler.do_GET = _do_get_with_tools


_original_do_post = server.LF3RHandler.do_POST


def _do_post_with_tools(self: server.LF3RHandler) -> None:
    path = unquote(urlparse(self.path).path)
    if path != "/api/tools/run":
        return _original_do_post(self)
    try:
        payload = _read_json_body(self)
        action = str(payload.get("action") or "").strip()
        options = payload.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError("options must be a JSON object")
        client_request_id = str(payload.get("client_request_id") or "").strip()
        if client_request_id:
            if len(client_request_id) > 120 or not all(
                character.isalnum() or character in "._:-"
                for character in client_request_id
            ):
                raise ValueError("invalid client_request_id")
        self.log_message(
            "project-tool submit received action=%s client_request_id=%s",
            action,
            client_request_id or "-",
        )
        job = self.app.project_tools.submit(
            action,
            options,
            client_request_id=client_request_id or None,
        )
        self.log_message(
            "project-tool submit registered action=%s job_id=%s status=%s",
            action,
            job.get("job_id"),
            job.get("status"),
        )
        self.json_response(
            HTTPStatus.ACCEPTED,
            {
                "job": job,
                "project_tools_protocol": "immediate-registry-v1",
            },
        )
    except server.TmuxSupervisorError as exc:
        self.json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except OSError as exc:
        self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


server.LF3RHandler.do_POST = _do_post_with_tools
