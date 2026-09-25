#!/usr/bin/env python3
"""Run Robo-Dopamine rollouts with one persistent GRMInference/vLLM engine."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from procvlm_worker import (
    FATAL_EXIT_CODE,
    TERMINAL_JOB_STATUSES,
    append_jsonl,
    atomic_json,
    cleanup_after_recoverable_failure,
    is_fatal_engine_failure,
    job_counts,
    load_jsonl,
    upsert_job,
)
from robo_dopamine_multi_perspective import (
    FUSED_EVAL_MODE,
    PERSPECTIVE_MODES,
    frame_index,
    fuse_prediction_files,
    plot_progress_curves,
    resolve_eval_modes,
    summarize_incremental_noise,
    write_progress_csv,
)


DEFAULT_VLLM_MEMORY_SAFETY_BUFFER_MIB = 2048

_LOCALIZATION_CHECKPOINT_CACHE: dict[str, dict[str, Any]] = {}


def _probe_video_frame_count(path: Path) -> int:
    """Return the decoded video-frame count reported by ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream found while counting frames: {path}")
    stream = streams[0]
    raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if raw_count in (None, "", "N/A"):
        raise ValueError(f"Could not determine video frame count: {path}")
    count = int(raw_count)
    if count <= 0:
        raise ValueError(f"Video has no decodable frames: {path}")
    return count


def _truncate_video_to_frame_count(source: Path, destination: Path, frame_count: int) -> str:
    """Create a project-local aligned copy without modifying the source video."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    temporary.unlink(missing_ok=True)

    copy_command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frame_count),
        "-c:v",
        "copy",
        "-an",
        str(temporary),
    ]
    subprocess.run(copy_command, check=True)
    method = "stream_copy"

    if _probe_video_frame_count(temporary) != frame_count:
        temporary.unlink(missing_ok=True)
        reencode_command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-frames:v",
            str(frame_count),
            "-vsync",
            "0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(temporary),
        ]
        subprocess.run(reencode_command, check=True)
        method = "lossless_reencode_fallback"

    actual = _probe_video_frame_count(temporary)
    if actual != frame_count:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Aligned camera copy has {actual} frames; expected {frame_count}: {source}"
        )
    temporary.replace(destination)
    return method


def _align_multiview_camera_inputs(
    *,
    cam_high: str,
    cam_left: str,
    cam_right: str,
    output_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Tolerate only a one-frame terminal mismatch among multiview inputs."""
    source_paths = {
        "cam_high": Path(cam_high).resolve(),
        "cam_left_wrist": Path(cam_left).resolve(),
        "cam_right_wrist": Path(cam_right).resolve(),
    }
    counts = {
        slot: _probe_video_frame_count(path)
        for slot, path in source_paths.items()
    }
    ordered_counts = [
        counts["cam_high"],
        counts["cam_left_wrist"],
        counts["cam_right_wrist"],
    ]
    target = min(ordered_counts)
    spread = max(ordered_counts) - target
    if spread == 0:
        return (
            {slot: str(path) for slot, path in source_paths.items()},
            {
                "applied": False,
                "policy": "terminal_off_by_one_only",
                "original_frame_counts": counts,
                "effective_frame_count": target,
                "dropped_frames": {slot: 0 for slot in source_paths},
            },
        )
    if spread > 1:
        raise ValueError(f"Frame count mismatch among cameras: {ordered_counts}")

    aligned_root = output_dir / "aligned_camera_inputs"
    aligned_by_source: dict[str, tuple[Path, str]] = {}
    effective: dict[str, str] = {}
    methods: dict[str, str] = {}
    for slot, source in source_paths.items():
        if counts[slot] == target:
            effective[slot] = str(source)
            continue
        source_key = str(source)
        cached = aligned_by_source.get(source_key)
        if cached is None:
            destination = aligned_root / f"{slot}.frames{target}.mp4"
            method = _truncate_video_to_frame_count(source, destination, target)
            cached = (destination, method)
            aligned_by_source[source_key] = cached
        effective[slot] = str(cached[0])
        methods[slot] = cached[1]

    dropped = {slot: counts[slot] - target for slot in source_paths}
    metadata = {
        "applied": True,
        "policy": "terminal_off_by_one_only",
        "original_frame_counts": counts,
        "effective_frame_count": target,
        "dropped_frames": dropped,
        "effective_camera_video_paths": effective,
        "trim_methods": methods,
    }
    print(
        "CAMERA_FRAME_ALIGNMENT "
        f"counts={ordered_counts} using={target} "
        f"dropped={[dropped['cam_high'], dropped['cam_left_wrist'], dropped['cam_right_wrist']]}",
        flush=True,
    )
    return effective, metadata


def _load_localization_checkpoint(path: Path) -> dict[str, Any]:
    """Load one saved LF3R localization head for lightweight CPU inference."""
    resolved = path.expanduser().resolve()
    key = str(resolved)
    cached = _LOCALIZATION_CHECKPOINT_CACHE.get(key)
    if cached is not None:
        return cached

    # Keep this import lazy so ordinary Robo-Dopamine runs do not import the
    # localization training package unless the optional head is requested.
    tools_root = Path(__file__).resolve().parents[1]
    if str(tools_root) not in sys.path:
        sys.path.insert(0, str(tools_root))
    import numpy as np
    import torch
    from robo_localization_head.core import TinyBiLSTM

    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Localization checkpoint is not a mapping: {resolved}")
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

    bundle = {
        "path": resolved,
        "model": model,
        "mean": mean,
        "std": std,
        "config": config,
        "stage": payload.get("stage"),
        "config_id": payload.get("config_id"),
        "repeat": payload.get("repeat"),
        "seed": payload.get("seed"),
    }
    _LOCALIZATION_CHECKPOINT_CACHE[key] = bundle
    return bundle


def run_localization_checkpoint(
    prediction_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Infer one localization point from a fused Robo-Dopamine progress/hop curve."""
    import numpy as np
    import torch

    rows = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Localization input is empty: {prediction_path}")

    frames: list[int] = []
    features: list[list[float]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Localization input contains a non-object row")
        progress = float(row["progress"])
        hop = float(row["hop"])
        if not math.isfinite(progress) or not math.isfinite(hop):
            raise ValueError("Localization input contains non-finite progress/hop")
        frames.append(frame_index(row))
        features.append([progress, hop])

    bundle = _load_localization_checkpoint(checkpoint_path)
    sequence = np.asarray(features, dtype=np.float32)
    normalized = (sequence - bundle["mean"]) / bundle["std"]
    tensor = torch.from_numpy(normalized).unsqueeze(0)
    with torch.no_grad():
        logits = bundle["model"](tensor)[0].cpu()
    predicted_index = int(torch.argmax(logits).item())
    predicted_frame = int(frames[predicted_index])
    predicted_logit = float(logits[predicted_index].item())
    probabilities = torch.sigmoid(logits)

    result = {
        "schema_version": 1,
        "checkpoint": str(bundle["path"]),
        "checkpoint_stage": bundle.get("stage"),
        "checkpoint_config_id": bundle.get("config_id"),
        "checkpoint_repeat": bundle.get("repeat"),
        "checkpoint_seed": bundle.get("seed"),
        "input": "fused_robo_dopamine_progress_hop",
        "source_prediction": str(prediction_path),
        "frame_count": len(frames),
        "frames": frames,
        "logits": [float(value) for value in logits.tolist()],
        "sigmoid_scores": [float(value) for value in probabilities.tolist()],
        "predicted_index": predicted_index,
        "predicted_frame": predicted_frame,
        "predicted_logit": predicted_logit,
        "predicted_sigmoid": float(probabilities[predicted_index].item()),
        "hidden": int(bundle["model"].hidden),
    }
    output_path = output_dir / "localization_prediction.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {**result, "output_path": str(output_path)}



def _visible_physical_gpu_ids() -> list[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(
            "Robo-Dopamine worker requires numeric CUDA_VISIBLE_DEVICES so "
            f"GPU memory can be measured immediately before vLLM init; got {raw!r}"
        )
    return [int(part) for part in parts]


def query_worker_gpu_memory() -> list[dict[str, int]]:
    requested = _visible_physical_gpu_ids()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"Unable to query GPU memory immediately before vLLM init: {error}"
        ) from error

    rows: dict[int, dict[str, int]] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not all(field.isdigit() for field in fields):
            continue
        index, total, free = (int(field) for field in fields)
        rows[index] = {"gpu": index, "total_mib": total, "free_mib": free}
    missing = [str(index) for index in requested if index not in rows]
    if missing:
        raise ValueError(
            f"nvidia-smi did not report worker GPU indices: {missing}"
        )
    return [rows[index] for index in requested]


def resolve_worker_vllm_memory_budget(
    requested_free_fraction: float,
    safety_buffer_mib: int,
) -> tuple[float, dict[str, Any]]:
    if not 0.0 < requested_free_fraction <= 1.0:
        raise ValueError("requested_free_fraction must be in (0, 1]")
    if safety_buffer_mib < 0:
        raise ValueError("safety_buffer_mib must be non-negative")

    snapshots = query_worker_gpu_memory()
    per_gpu: list[dict[str, Any]] = []
    effective_limits: list[float] = []
    for row in snapshots:
        total = int(row["total_mib"])
        free = int(row["free_mib"])
        if total <= 0 or free < 0 or free > total:
            raise ValueError(f"Invalid GPU memory snapshot: {row}")
        requested_target_mib = requested_free_fraction * free
        buffered_target_mib = max(0.0, free - safety_buffer_mib)
        target_mib = min(requested_target_mib, buffered_target_mib)
        total_fraction = target_mib / total
        effective_limits.append(total_fraction)
        per_gpu.append(
            {
                **row,
                "requested_target_mib": requested_target_mib,
                "buffered_target_mib": buffered_target_mib,
                "effective_target_mib": target_mib,
                "effective_total_fraction": total_fraction,
            }
        )

    effective = math.floor(min(effective_limits) * 1_000_000) / 1_000_000
    if effective <= 0.0:
        raise ValueError(
            "Resolved vLLM memory fraction is non-positive after applying "
            f"{safety_buffer_mib} MiB safety buffer"
        )
    budget = {
        "scope": "free_gpu_memory",
        "requested_free_fraction": requested_free_fraction,
        "resolved_total_fraction": effective,
        "resolution": "worker_immediately_before_vllm_init",
        "resolution_stage": "bounded_llm_pre_init",
        "safety_buffer_mib_per_gpu": safety_buffer_mib,
        "selected_gpus": snapshots,
        "per_gpu_limits": per_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    return effective, budget


def initialize_robo_model(args: argparse.Namespace) -> Any:
    """Import the official module and construct GRMInference exactly once."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    repo = str(args.repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    # The official module sets CUDA_VISIBLE_DEVICES=0 at import time.  Preserve
    # the worker assignment so vLLM sees the GPU selected by the parent runner.
    requested_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    requested_local_rank = os.environ.get("LOCAL_RANK")
    try:
        import examples.inference as official
    finally:
        if requested_cuda_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = requested_cuda_visible
        if requested_local_rank is None:
            os.environ.pop("LOCAL_RANK", None)
        else:
            os.environ["LOCAL_RANK"] = requested_local_rank

    # The official constructor hard-codes 0.9. Patch only its module-local LLM
    # symbol; the normal LF3R path resolves the free-memory budget here, inside
    # the persistent worker, immediately before the official vLLM constructor.
    official_llm = getattr(official, "LLM", None)
    if official_llm is None:
        raise RuntimeError("Robo-Dopamine examples.inference has no LLM symbol")

    def bounded_llm(*model_args: Any, **model_kwargs: Any) -> Any:
        if args.vllm_total_memory_fraction is not None:
            resolved_fraction = float(args.vllm_total_memory_fraction)
            resolved_budget = {
                **dict(getattr(args, "memory_budget", {}) or {}),
                "resolved_total_fraction": resolved_fraction,
                "resolution": "fixed_total_fraction_override",
                "resolution_stage": "bounded_llm_pre_init",
                "safety_buffer_mib_per_gpu": args.vllm_memory_safety_buffer_mib,
            }
        else:
            resolved_fraction, resolved_budget = resolve_worker_vllm_memory_budget(
                float(args.vllm_free_memory_fraction),
                int(args.vllm_memory_safety_buffer_mib),
            )
        args.resolved_memory_budget = resolved_budget
        model_kwargs["gpu_memory_utilization"] = resolved_fraction
        model_kwargs["tensor_parallel_size"] = args.tp
        return official_llm(*model_args, **model_kwargs)

    official.LLM = bounded_llm
    return official.GRMInference(str(args.model_path.resolve()))


def official_source_revision(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def cleanup_official_frame_cache(output_dir: Path) -> tuple[int, int]:
    """Remove Robo-Dopamine's extracted image cache after predictions are durable."""
    cache_dir = output_dir / ".cache"
    if not cache_dir.exists():
        return 0, 0
    if not cache_dir.is_dir():
        raise RuntimeError(
            f"Expected Robo-Dopamine frame cache to be a directory: {cache_dir}"
        )
    files = sum(1 for path in cache_dir.rglob("*") if path.is_file())
    directories = sum(1 for path in cache_dir.rglob("*") if path.is_dir())
    shutil.rmtree(cache_dir)
    print(
        f"ROBODOPAMINE_CACHE_CLEANUP dir={cache_dir} "
        f"files={files} directories={directories}",
        flush=True,
    )
    return files, directories


def _run_official_mode(
    *,
    model: Any,
    cam_high: str,
    cam_left: str,
    cam_right: str,
    goal_image: str,
    output_dir: Path,
    task: str,
    frame_interval: int,
    batch_size: int,
    eval_mode: str,
    render_video: bool,
) -> tuple[Path, float]:
    started = time.perf_counter()
    output = model.run_pipeline(
        cam_high_path=cam_high,
        cam_left_path=cam_left,
        cam_right_path=cam_right,
        out_root=str(output_dir),
        task=task,
        frame_interval=frame_interval,
        batch_size=batch_size,
        goal_image=goal_image,
        eval_mode=eval_mode,
        visualize=render_video,
    )
    output_path = Path(output).resolve()
    prediction = output_path / "pred_vllm.json"
    if not prediction.is_file():
        raise RuntimeError(
            f"Robo-Dopamine {eval_mode} mode did not write raw predictions: {prediction}"
        )
    cleanup_official_frame_cache(output_path)
    return prediction, time.perf_counter() - started


def infer_rollout(
    job: dict[str, Any], args: argparse.Namespace, model: Any
) -> dict[str, Any]:
    """Run one rollout through the already-initialized GRMInference object."""
    output_dir = Path(job["raw_output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video = str(Path(job["video_path"]).resolve())
    cam_high = str(Path(job.get("cam_high_path", video)).resolve())
    cam_left = str(Path(job.get("cam_left_path", video)).resolve())
    cam_right = str(Path(job.get("cam_right_path", video)).resolve())
    source_camera_video_paths = {
        "cam_high": cam_high,
        "cam_left_wrist": cam_left,
        "cam_right_wrist": cam_right,
    }
    camera_alignment: dict[str, Any] | None = None
    camera_input_mode = str(job.get("camera_input_mode", "single_view"))
    goal_image = str(Path(job.get("goal_image", args.goal_image)).resolve())
    frame_interval = int(job.get("frame_interval", args.frame_interval))
    batch_size = int(job.get("batch_size", args.batch_size))
    eval_mode = str(job.get("eval_mode", args.eval_mode))
    requested_modes = job.get("eval_modes") or getattr(args, "eval_modes", None)
    eval_modes = resolve_eval_modes(eval_mode, requested_modes)
    if not eval_modes:
        eval_modes = [eval_mode]
    render_video = bool(job.get("render_video", args.render_video))
    task = str(job["task"])

    mode_predictions: dict[str, Path] = {}
    mode_seconds: dict[str, float] = {}
    for mode in eval_modes:
        try:
            prediction, elapsed = _run_official_mode(
                model=model,
                cam_high=cam_high,
                cam_left=cam_left,
                cam_right=cam_right,
                goal_image=goal_image,
                output_dir=output_dir,
                task=task,
                frame_interval=frame_interval,
                batch_size=batch_size,
                eval_mode=mode,
                render_video=render_video,
            )
        except ValueError as error:
            mismatch = "Frame count mismatch among cameras:" in str(error)
            if camera_input_mode != "multi_view" or not mismatch or camera_alignment is not None:
                raise
            effective_paths, camera_alignment = _align_multiview_camera_inputs(
                cam_high=cam_high,
                cam_left=cam_left,
                cam_right=cam_right,
                output_dir=output_dir,
            )
            cam_high = effective_paths["cam_high"]
            cam_left = effective_paths["cam_left_wrist"]
            cam_right = effective_paths["cam_right_wrist"]
            prediction, elapsed = _run_official_mode(
                model=model,
                cam_high=cam_high,
                cam_left=cam_left,
                cam_right=cam_right,
                goal_image=goal_image,
                output_dir=output_dir,
                task=task,
                frame_interval=frame_interval,
                batch_size=batch_size,
                eval_mode=mode,
                render_video=render_video,
            )
        mode_predictions[mode] = prediction
        mode_seconds[mode] = elapsed

    multi_perspective = len(eval_modes) > 1
    fused_path: Path | None = None
    fusion_metadata: dict[str, Any] = {}
    if multi_perspective:
        if set(eval_modes) != set(PERSPECTIVE_MODES):
            raise ValueError(
                "Multi-perspective Robo-Dopamine output requires exactly "
                f"{', '.join(PERSPECTIVE_MODES)}"
            )
        fused_root = output_dir / "multi_perspective"
        fused_path = fused_root / "fused_progress.json"
        fusion_metadata = fuse_prediction_files(
            mode_predictions,
            fused_path,
            metadata={
                "checkpoint": str(args.model_path.resolve()),
                "eval_modes": list(PERSPECTIVE_MODES),
                "frame_interval": frame_interval,
                "goal_image": goal_image,
                "fusion_rule": "arithmetic_mean_of_official_progress",
                "source_commit": official_source_revision(args.repo),
            },
        )
        write_progress_csv(fused_root / "progress_curves.csv", mode_predictions, fused_path)
        fusion_metadata["incremental_noise"] = summarize_incremental_noise(
            mode_predictions["incremental"]
        )
        plot_path = fused_root / "progress_curves.png"
        if plot_progress_curves(
            plot_path,
            mode_predictions,
            fused_path,
            title=f"Robo-Dopamine multi-perspective: {job['rollout_id']}",
        ):
            fusion_metadata["plot_path"] = str(plot_path)
        else:
            fusion_metadata["plot_path"] = None
        (fused_root / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "checkpoint": str(args.model_path.resolve()),
                    "eval_modes": list(PERSPECTIVE_MODES),
                    "frame_interval": frame_interval,
                    "goal_image": goal_image,
                    "fusion_rule": "arithmetic_mean_of_official_progress",
                    "source_commit": official_source_revision(args.repo),
                    "prediction_paths": {mode: str(path) for mode, path in mode_predictions.items()},
                    "fused_path": str(fused_path),
                    "mode_seconds": mode_seconds,
                    "fusion": fusion_metadata,
                },
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

    # Keep the legacy single-mode contract unchanged. For a multi-mode job,
    # raw_model_output points to the fused curve so existing readers show the
    # fused result, while every official per-mode file remains available.
    raw_model_output = fused_path or mode_predictions[eval_modes[0]]
    official_output_dir = mode_predictions[eval_modes[0]].parent

    localization_prediction: dict[str, Any] | None = None
    if args.localization_checkpoint is not None:
        if fused_path is None:
            raise ValueError(
                "Localization checkpoint inference requires fused Robo-Dopamine "
                "(incremental + forward + backward)"
            )
        localization_prediction = run_localization_checkpoint(
            fused_path,
            args.localization_checkpoint,
            output_dir,
        )

    result_path = output_dir / "worker_result.json"
    result = {
        "schema_version": 2 if multi_perspective else 1,
        "baseline": "robo_dopamine",
        "official_output_dir": str(official_output_dir),
        "raw_model_output": str(raw_model_output),
        "eval_modes": eval_modes,
        "frame_interval": frame_interval,
        "batch_size": batch_size,
        "goal_image": goal_image,
        "checkpoint": str(args.model_path.resolve()),
        "source_commit": official_source_revision(args.repo),
        "video_path": video,
        "camera_input_mode": camera_input_mode,
        "camera_video_paths": source_camera_video_paths,
        "task": task,
        "eval_mode": eval_mode,
    }
    if camera_alignment is not None:
        result["camera_alignment"] = camera_alignment
    if localization_prediction is not None:
        result["localization_prediction"] = {
            key: localization_prediction[key]
            for key in (
                "output_path",
                "checkpoint",
                "checkpoint_stage",
                "checkpoint_config_id",
                "checkpoint_repeat",
                "checkpoint_seed",
                "frames",
                "logits",
                "sigmoid_scores",
                "predicted_index",
                "predicted_frame",
                "predicted_logit",
                "predicted_sigmoid",
                "frame_count",
            )
        }
    if multi_perspective:
        result.update(
            {
                "multi_perspective": True,
                "fusion_rule": "arithmetic_mean_of_official_progress",
                "fused_model_output": str(fused_path),
                "perspective_outputs": {
                    mode: {
                        "official_output_dir": str(path.parent),
                        "raw_model_output": str(path),
                        "seconds": mode_seconds[mode],
                    }
                    for mode, path in mode_predictions.items()
                },
                "mode_seconds": mode_seconds,
                "fusion": fusion_metadata,
            }
        )
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for mode, prediction in mode_predictions.items():
        print(f"Raw Robo-Dopamine {mode} output: {prediction}", flush=True)
    if fused_path is not None:
        print(f"Fused Robo-Dopamine output: {fused_path}", flush=True)
    if localization_prediction is not None:
        print(
            "Localization checkpoint prediction: "
            f"frame={localization_prediction['predicted_frame']} "
            f"index={localization_prediction['predicted_index']} "
            f"checkpoint={localization_prediction['checkpoint']}",
            flush=True,
        )
    return {
        "official_output_dir": str(official_output_dir),
        "raw_model_output": str(raw_model_output),
        "worker_result_path": str(result_path),
        "eval_modes": eval_modes,
        "perspective_outputs": result.get("perspective_outputs"),
        "fused_model_output": str(fused_path) if fused_path else None,
        "mode_seconds": mode_seconds,
        "fusion": fusion_metadata,
        "camera_alignment": result.get("camera_alignment"),
        "localization_prediction": result.get("localization_prediction"),
    }


def refresh_state(
    state_path: Path,
    *,
    status: str,
    total_jobs: int,
    records: list[dict[str, Any]],
    engine_initialization_seconds: float | None,
    engine_memory_budget: dict[str, Any],
    last_rollout_id: str | None = None,
    fatal_error: dict[str, Any] | None = None,
) -> None:
    counts = job_counts(records)
    state = {
        "schema_version": 1,
        "status": status,
        "updated_at": datetime_now(),
        "total_jobs": total_jobs,
        **counts,
        "pending_jobs": total_jobs - counts["completed_jobs"] - counts["failed_jobs"],
        "engine_initialization_seconds": engine_initialization_seconds,
        "engine_memory_budget": engine_memory_budget,
        "last_rollout_id": last_rollout_id,
    }
    if fatal_error is not None:
        state["fatal_error"] = fatal_error
    atomic_json(state_path, state)


def datetime_now() -> str:
    # Kept local to make this worker easy to exercise without importing the
    # baseline runner (which would otherwise initialize project-level paths).
    from datetime import datetime

    return datetime.now().astimezone().isoformat()


def command_metadata(job: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "job_index",
        "rollout_id",
        "cwd",
        "argv",
        "shell_preview",
        "vllm_memory_budget",
        "execution_scope",
    )
    return {key: job[key] for key in keys if key in job}


def find_prediction(output_dir: Path) -> Path | None:
    predictions = sorted(path for path in output_dir.rglob("pred_vllm.json") if path.is_file())
    return predictions[-1] if predictions else None


def run_persistent_jobs(
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    jobs_output_path: Path,
    progress_path: Path,
    state_path: Path,
    engine_memory_budget: dict[str, Any],
    initialize_model: Callable[[argparse.Namespace], Any] = initialize_robo_model,
    infer_one: Callable[[dict[str, Any], argparse.Namespace, Any], Any] = infer_rollout,
) -> int:
    """Initialize one GRMInference, then process pending jobs in plan order."""
    existing = load_jsonl(jobs_output_path)
    existing_by_id = {record.get("rollout_id"): record for record in existing}
    pending = [
        job
        for job in jobs
        if existing_by_id.get(job.get("rollout_id"), {}).get("status")
        not in TERMINAL_JOB_STATUSES
    ]

    if not pending:
        refresh_state(
            state_path,
            status="no_pending_jobs",
            total_jobs=len(jobs),
            records=existing,
            engine_initialization_seconds=None,
            engine_memory_budget=engine_memory_budget,
        )
        append_jsonl(
            progress_path,
            {
                "event": "worker_complete",
                "status": "no_pending_jobs",
                "at": datetime_now(),
                **job_counts(existing),
            },
        )
        return 0

    append_jsonl(
        progress_path,
        {
            "event": "engine_initialization_started",
            "at": datetime_now(),
            "pending_jobs": len(pending),
            "total_jobs": len(jobs),
            "memory_budget": engine_memory_budget,
        },
    )
    init_started = time.perf_counter()
    try:
        if args.localization_checkpoint is not None:
            # Validate and cache the tiny CPU head before paying the vLLM
            # initialization cost.
            _load_localization_checkpoint(args.localization_checkpoint)
        model = initialize_model(args)
    except BaseException as error:
        init_seconds = time.perf_counter() - init_started
        engine_memory_budget = dict(
            getattr(args, "resolved_memory_budget", engine_memory_budget)
        )
        fatal = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        refresh_state(
            state_path,
            status="fatal_engine_failure",
            total_jobs=len(jobs),
            records=existing,
            engine_initialization_seconds=init_seconds,
            engine_memory_budget=engine_memory_budget,
            fatal_error=fatal,
        )
        append_jsonl(
            progress_path,
            {
                "event": "fatal_engine_failure",
                "phase": "initialization",
                "at": datetime_now(),
                "initialization_seconds": init_seconds,
                "memory_budget": engine_memory_budget,
                **fatal,
            },
        )
        return FATAL_EXIT_CODE

    init_seconds = time.perf_counter() - init_started
    engine_memory_budget = dict(
        getattr(args, "resolved_memory_budget", engine_memory_budget)
    )
    append_jsonl(
        progress_path,
        {
            "event": "engine_initialized",
            "at": datetime_now(),
            "initialization_seconds": init_seconds,
            "memory_budget": engine_memory_budget,
            "engine_reused_for_jobs": len(pending),
        },
    )
    refresh_state(
        state_path,
        status="running",
        total_jobs=len(jobs),
        records=existing,
        engine_initialization_seconds=init_seconds,
        engine_memory_budget=engine_memory_budget,
    )

    for job in pending:
        rollout_id = str(job["rollout_id"])
        output_dir = Path(job["raw_output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime_now()
        started = time.perf_counter()
        append_jsonl(
            progress_path,
            {
                "event": "rollout_started",
                "at": started_at,
                "job_index": job.get("job_index"),
                "rollout_id": rollout_id,
            },
        )
        result_metadata: dict[str, Any] = {}
        try:
            returned = infer_one(job, args, model)
            if isinstance(returned, dict):
                result_metadata = {
                    key: value
                    for key, value in returned.items()
                    if key in {
                        "official_output_dir",
                        "raw_model_output",
                        "worker_result_path",
                        "eval_modes",
                        "perspective_outputs",
                        "fused_model_output",
                        "mode_seconds",
                        "fusion",
                        "camera_alignment",
                        "localization_prediction",
                    }
                }
            prediction_value = result_metadata.get("raw_model_output")
            prediction = (
                Path(str(prediction_value)).resolve()
                if prediction_value
                else find_prediction(output_dir)
            )
            if prediction is None or not prediction.is_file():
                raise RuntimeError(
                    f"Robo-Dopamine produced no pred_vllm.json under {output_dir}"
                )
            result_metadata.setdefault("raw_model_output", str(prediction))
            return_code = 0
            status = "complete"
            error_details: dict[str, Any] = {}
        except BaseException as error:
            return_code = FATAL_EXIT_CODE if is_fatal_engine_failure(error) else 1
            status = "interrupted" if return_code == FATAL_EXIT_CODE else "failed"
            error_details = {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }

        elapsed = time.perf_counter() - started
        raw_files = sorted(str(path) for path in output_dir.rglob("*") if path.is_file())
        result = {
            **command_metadata(job),
            **result_metadata,
            "status": status,
            "return_code": return_code,
            "started_at": started_at,
            "completed_at": datetime_now(),
            "inference_seconds": elapsed,
            "raw_output_dir": str(output_dir),
            "raw_output_files": raw_files,
        }
        result.update(error_details)
        if status == "interrupted":
            result["resume_required"] = True
        records = upsert_job(jobs_output_path, result)
        append_jsonl(
            progress_path,
            {
                "event": "rollout_finished",
                "at": result["completed_at"],
                "job_index": job.get("job_index"),
                "rollout_id": rollout_id,
                "status": status,
                "inference_seconds": elapsed,
                **error_details,
            },
        )
        refresh_state(
            state_path,
            status="fatal_engine_failure" if status == "interrupted" else "running",
            total_jobs=len(jobs),
            records=records,
            engine_initialization_seconds=init_seconds,
            engine_memory_budget=engine_memory_budget,
            last_rollout_id=rollout_id,
            fatal_error=error_details if status == "interrupted" else None,
        )
        if status == "interrupted":
            append_jsonl(
                progress_path,
                {
                    "event": "fatal_engine_failure",
                    "phase": "rollout",
                    "at": datetime_now(),
                    "rollout_id": rollout_id,
                    "memory_budget": engine_memory_budget,
                    **error_details,
                },
            )
            return FATAL_EXIT_CODE
        if status == "failed":
            cleanup_after_recoverable_failure()

    records = load_jsonl(jobs_output_path)
    counts = job_counts(records)
    refresh_state(
        state_path,
        status="complete",
        total_jobs=len(jobs),
        records=records,
        engine_initialization_seconds=init_seconds,
        engine_memory_budget=engine_memory_budget,
        last_rollout_id=records[-1].get("rollout_id") if records else None,
    )
    append_jsonl(
        progress_path,
        {
            "event": "worker_complete",
            "status": "complete",
            "at": datetime_now(),
            "engine_initialization_seconds": init_seconds,
            **counts,
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--jobs-output-file", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--goal-image", type=Path, required=True)
    parser.add_argument("--frame-interval", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--eval-mode",
        choices=(FUSED_EVAL_MODE, "forward", "incremental", "backward"),
        default=FUSED_EVAL_MODE,
        help="Wrapper mode; fused runs the official incremental, forward, and backward perspectives",
    )
    parser.add_argument(
        "--eval-modes",
        nargs="+",
        choices=PERSPECTIVE_MODES,
        default=None,
        help="Official modes to run with one persistent GRM; all three are required for fusion",
    )
    parser.add_argument("--vllm-total-memory-fraction", type=float, default=None)
    parser.add_argument("--vllm-free-memory-fraction", type=float, default=None)
    parser.add_argument(
        "--vllm-memory-safety-buffer-mib",
        type=int,
        default=DEFAULT_VLLM_MEMORY_SAFETY_BUFFER_MIB,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--memory-budget-json", required=True)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument(
        "--localization-checkpoint",
        type=Path,
        default=None,
        help="Optional LF3R BiLSTM localization checkpoint; requires fused output",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.frame_interval < 1:
        parser.error("--frame-interval must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.tp < 1:
        parser.error("--tp must be positive")
    if args.localization_checkpoint is not None:
        args.localization_checkpoint = args.localization_checkpoint.expanduser().resolve()
        if not args.localization_checkpoint.is_file():
            parser.error(
                f"--localization-checkpoint does not exist: {args.localization_checkpoint}"
            )
    if args.eval_modes:
        if len(set(args.eval_modes)) != len(args.eval_modes):
            parser.error("--eval-modes must not contain duplicates")
        if len(args.eval_modes) > 1 and set(args.eval_modes) != set(PERSPECTIVE_MODES):
            parser.error("multi-perspective mode requires incremental, forward, and backward")
    if args.vllm_memory_safety_buffer_mib < 0:
        parser.error("--vllm-memory-safety-buffer-mib must be non-negative")
    if (
        args.vllm_free_memory_fraction is not None
        and not 0.0 < args.vllm_free_memory_fraction <= 1.0
    ):
        parser.error("--vllm-free-memory-fraction must be in (0, 1]")
    if (
        args.vllm_total_memory_fraction is not None
        and not 0.0 < args.vllm_total_memory_fraction <= 1.0
    ):
        parser.error("--vllm-total-memory-fraction must be in (0, 1]")
    if (
        not args.dry_run
        and args.vllm_total_memory_fraction is None
        and args.vllm_free_memory_fraction is None
    ):
        parser.error(
            "--vllm-free-memory-fraction is required unless a legacy "
            "--vllm-total-memory-fraction override is supplied"
        )
    try:
        args.memory_budget = json.loads(args.memory_budget_json)
    except json.JSONDecodeError as error:
        parser.error(f"--memory-budget-json is invalid JSON: {error}")
    if not isinstance(args.memory_budget, dict):
        parser.error("--memory-budget-json must contain an object")
    return args


def main() -> int:
    args = parse_args()
    args.repo = args.repo.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.jobs_file = args.jobs_file.expanduser().resolve()
    args.jobs_output_file = args.jobs_output_file.expanduser().resolve()
    args.progress_file = args.progress_file.expanduser().resolve()
    args.state_file = args.state_file.expanduser().resolve()
    args.goal_image = args.goal_image.expanduser().resolve()
    jobs = load_jsonl(args.jobs_file)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "jobs_file": str(args.jobs_file),
                    "jobs": len(jobs),
                    "requested_free_memory_fraction": args.vllm_free_memory_fraction,
                    "memory_safety_buffer_mib": args.vllm_memory_safety_buffer_mib,
                    "memory_resolution": "worker_immediately_before_vllm_init",
                    "tensor_parallel_size": args.tp,
                },
                ensure_ascii=False,
            )
        )
        return 0
    for required in (args.repo / "examples/inference.py", args.model_path, args.goal_image):
        if not required.exists():
            raise FileNotFoundError(required)
    if not jobs:
        raise ValueError(f"No Robo-Dopamine jobs found in {args.jobs_file}")
    return run_persistent_jobs(
        jobs,
        args,
        jobs_output_path=args.jobs_output_file,
        progress_path=args.progress_file,
        state_path=args.state_file,
        engine_memory_budget=args.memory_budget,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
