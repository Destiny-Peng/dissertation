#!/usr/bin/env python3
"""Execute one LF3R Synthetic Suffix run.

Every world-model run shares the same official LIBERO cut/alignment smoke test.
Model-specific adapters then prepare their own conditioning without changing the
manifest or the underlying demonstration.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

from backend_core import ValidationError
from non_analysis_tools import gpu_status
from repair.adapters import A2WorldAdapter
from repair.alignment_runner import run_alignment_subprocess
from repair.ctrl_world import CtrlWorldAdapter
from repair.trajectory import load_actions


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def update_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "status.json"
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        payload = {}
    payload.update(updates)
    payload["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(path, payload)
    return payload


def read_frames(path: Path) -> list[Any]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    try:
        return [frame for frame in reader]
    finally:
        reader.close()


def metric_aligned_frames(real: Any, generated: Any) -> tuple[Any, Any, bool]:
    """Align only image resolution for diagnostics, never time or camera semantics."""

    import numpy as np

    a = np.asarray(real, dtype=np.uint8)
    b = np.asarray(generated, dtype=np.uint8)
    if a.shape == b.shape:
        return a, b, False
    if (
        a.ndim != 3
        or b.ndim != 3
        or a.shape[-1] != 3
        or b.shape[-1] != 3
    ):
        return a, b, False
    try:
        from PIL import Image
    except ImportError:
        return a, b, False
    resized = np.asarray(
        Image.fromarray(b).resize(
            (int(a.shape[1]), int(a.shape[0])),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )
    return a, resized, True


def image_metrics(real: Any, generated: Any) -> dict[str, float | None]:
    import numpy as np

    a, b, _ = metric_aligned_frames(real, generated)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return {"psnr": None, "ssim": None}
    mse = float(np.mean((a - b) ** 2))
    psnr = (
        float("inf")
        if mse <= 1e-12
        else 20.0 * math.log10(255.0 / math.sqrt(mse))
    )
    try:
        from skimage.metrics import structural_similarity

        ssim = float(
            structural_similarity(
                a.astype(np.uint8),
                b.astype(np.uint8),
                channel_axis=-1,
                data_range=255,
            )
        )
    except (ImportError, ValueError):
        ssim = None
    return {"psnr": psnr, "ssim": ssim}


def _optional_lpips() -> tuple[Any, Any, dict[str, Any]]:
    """Return LPIPS only when explicitly enabled.

    LPIPS / torchvision may download trunk weights when first constructed.
    Repair jobs never trigger an implicit checkpoint download.
    """

    if os.environ.get("LF3R_ENABLE_LPIPS", "").strip() != "1":
        return (
            None,
            None,
            {
                "available": False,
                "reason": (
                    "disabled to prevent implicit model downloads; set "
                    "LF3R_ENABLE_LPIPS=1 only after local LPIPS weights exist"
                ),
            },
        )
    try:
        import lpips as _lpips
        import torch as _torch

        model = _lpips.LPIPS(net="alex")
        model.eval()
        return model, _torch, {
            "available": True,
            "backend": "lpips/alex",
            "download_policy": "explicit_opt_in",
        }
    except Exception as error:
        return None, None, {
            "available": False,
            "reason": f"{type(error).__name__}: {error}",
        }


def compute_metrics(
    *,
    project_root: Path,
    rollout: dict[str, Any],
    generated: dict[str, Path],
    cut_frame: int,
    real_frame_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Compare generated frames with their actual source-time counterparts.

    A2World predicts every post-cut step, so its default mapping is c+1+i.
    Ctrl-World runs at its own sampled temporal rate and supplies explicit source
    frame indices (for example c+4, c+8, ... for a 20 Hz LIBERO source -> 5 Hz).
    """

    camera_paths = rollout.get("camera_video_paths") or {}
    lpips_model, torch, lpips_status = _optional_lpips()
    explicit_indices = (
        [int(index) for index in real_frame_indices]
        if real_frame_indices is not None
        else None
    )
    result: dict[str, Any] = {
        "alignment": {
            "real_suffix_start_frame": (
                explicit_indices[0]
                if explicit_indices
                else cut_frame + 1
            ),
            "generated_includes_condition": False,
            "comparison_basis": (
                "explicit source frame indices"
                if explicit_indices is not None
                else "frame/action index"
            ),
            "real_frame_indices": explicit_indices,
        },
        "views": {},
        "lpips": lpips_status,
        "human_evaluation": {
            "usable_for_policy_training": None,
            "failure_reasons": [],
        },
    }
    try:
        import numpy as np
    except ImportError:
        return {
            **result,
            "error": "numpy unavailable; visual metrics were not computed",
        }

    for view, generated_path in generated.items():
        source_value = camera_paths.get(view)
        if not isinstance(source_value, str):
            continue
        real_path = (project_root / source_value).resolve()
        real_frames = read_frames(real_path)
        generated_frames = read_frames(generated_path)

        if explicit_indices is None:
            indices = list(
                range(
                    cut_frame + 1,
                    min(
                        len(real_frames),
                        cut_frame + 1 + len(generated_frames),
                    ),
                )
            )
        else:
            indices = [
                index
                for index in explicit_indices[: len(generated_frames)]
                if 0 <= index < len(real_frames)
            ]
        count = min(len(generated_frames), len(indices))

        psnr_values: list[float] = []
        ssim_values: list[float] = []
        lpips_values: list[float] = []
        resolution_resized_frames = 0
        generated_shape = None
        real_shape = None
        for index in range(count):
            real_frame = real_frames[indices[index]]
            generated_frame = generated_frames[index]
            aligned_real, aligned_generated, resized = metric_aligned_frames(
                real_frame,
                generated_frame,
            )
            if resized:
                resolution_resized_frames += 1
            if real_shape is None:
                real_shape = list(getattr(real_frame, "shape", ()))
                generated_shape = list(getattr(generated_frame, "shape", ()))
            per_frame = image_metrics(aligned_real, aligned_generated)
            if (
                per_frame["psnr"] is not None
                and math.isfinite(float(per_frame["psnr"]))
            ):
                psnr_values.append(float(per_frame["psnr"]))
            if per_frame["ssim"] is not None:
                ssim_values.append(float(per_frame["ssim"]))
            if lpips_model is not None and torch is not None:
                a = np.asarray(aligned_real, dtype=np.float32)
                b = np.asarray(aligned_generated, dtype=np.float32)
                if a.shape == b.shape:
                    aa = (
                        torch.from_numpy(a)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        / 127.5
                        - 1.0
                    )
                    bb = (
                        torch.from_numpy(b)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        / 127.5
                        - 1.0
                    )
                    with torch.no_grad():
                        lpips_values.append(
                            float(lpips_model(aa, bb).item())
                        )

        result["views"][view] = {
            "compared_frames": count,
            "real_available_frames": len(indices),
            "generated_frames": len(generated_frames),
            "real_frame_indices": indices[:count],
            "real_frame_shape": real_shape,
            "generated_frame_shape": generated_shape,
            "resolution_resized_frames": resolution_resized_frames,
            "resolution_alignment": (
                "generated resized to real frame size with bilinear interpolation for diagnostics"
                if resolution_resized_frames
                else "native shapes matched"
            ),
            "psnr_mean": (
                sum(psnr_values) / len(psnr_values)
                if psnr_values
                else None
            ),
            "ssim_mean": (
                sum(ssim_values) / len(ssim_values)
                if ssim_values
                else None
            ),
            "lpips_mean": (
                sum(lpips_values) / len(lpips_values)
                if lpips_values
                else None
            ),
        }
    return result


def ensure_gpu_below_threshold(
    gpu_index: int,
    threshold: float = 50.0,
) -> dict[str, Any]:
    status = gpu_status()
    if not status.get("available"):
        raise ValidationError(
            "GPU status is unavailable before world-model generation: "
            + str(status.get("error") or "unknown error")
        )
    selected = next(
        (
            gpu
            for gpu in status.get("gpus", [])
            if int(gpu.get("index", -1)) == int(gpu_index)
        ),
        None,
    )
    if selected is None:
        raise ValidationError(f"Selected GPU {gpu_index} is no longer available")
    utilization = selected.get("gpu_utilization_percent")
    if utilization is None or float(utilization) >= threshold:
        raise ValidationError(
            f"Selected GPU {gpu_index} utilization is {utilization}%; "
            f"Repair requires < {threshold:.0f}% before world-model generation"
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    try:
        run_dir.relative_to(project_root)
    except ValueError as error:
        raise SystemExit(
            "run-dir must be inside PROJECT_ROOT"
        ) from error

    input_payload = json.loads(
        (run_dir / "input.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    rollout = input_payload["rollout"]
    alignment = input_payload["alignment"]
    cut_frame = int(alignment["cut_rgb_frame"])
    wm_config = dict(config.get("world_model") or {})
    model_name = str(wm_config.get("name") or "a2world").strip().lower().replace("-", "_")

    try:
        update_status(
            run_dir,
            status="running",
            phase="load_trajectory",
            progress=0.08,
        )
        actions = load_actions(project_root, rollout)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValidationError(
                f"LIBERO actions must be [T,7], got {actions.shape}"
            )
        action_start = int(alignment["gt_action_start"])
        if action_start >= len(actions):
            raise ValidationError(
                "No future action remains after LIBERO alignment"
            )
        future_actions = actions[action_start:]

        update_status(
            run_dir,
            phase="alignment_validation",
            progress=0.18,
        )
        gpu_value = wm_config.get("gpu_index")
        smoke = run_alignment_subprocess(
            project_root=project_root,
            rollout=rollout,
            cut_frame=cut_frame,
            min_psnr=float(config.get("alignment_min_psnr", 20.0)),
            gpu_index=int(gpu_value) if gpu_value is not None else None,
        )
        atomic_json(run_dir / "alignment.json", smoke)
        if not smoke["passed"]:
            raise ValidationError(
                "LIBERO alignment smoke test failed; generation was not started"
            )

        prepared_dir = run_dir / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        generated_dir = run_dir / "generated"

        model_provenance: dict[str, Any]
        real_frame_indices: list[int] | None

        if model_name == "a2world":
            update_status(
                run_dir,
                phase="prepare_a2world",
                progress=0.30,
            )
            adapter = A2WorldAdapter(project_root, wm_config)
            adapter_status = adapter.validate_rollout(rollout)
            if not adapter_status["available"]:
                raise ValidationError(
                    "; ".join(adapter_status["unavailable_reasons"])
                )
            rgb_adapter = adapter.configure_alignment_rgb(smoke)
            condition = adapter.prepare_condition(
                rollout,
                cut_frame=cut_frame,
                output_dir=prepared_dir,
            )
            condition["future_action_count"] = int(len(future_actions))
            actions_path = adapter.prepare_actions(
                future_actions,
                output_dir=prepared_dir,
            )
            action_metadata = read_json(
                prepared_dir / "future_actions_a2world.json",
                {},
            )

            update_status(
                run_dir,
                phase="gpu_recheck",
                progress=0.40,
            )
            gpu_index = int(wm_config["gpu_index"])
            gpu_before_generation = ensure_gpu_below_threshold(gpu_index)

            update_status(
                run_dir,
                phase="generate_suffix",
                progress=0.45,
            )
            generated = adapter.generate(
                condition=condition,
                actions_path=actions_path,
                output_dir=generated_dir,
            )
            generated_paths = adapter.save_result(generated, generated_dir)
            real_frame_indices = None
            model_provenance = {
                "camera_mapping": adapter_status["camera_mapping"],
                "duplicated_camera": adapter_status["duplicated_camera"],
                "a2world_view_ids": adapter_status["view_ids"],
                "a2world_source_root": adapter_status["source_root"],
                "a2world_python": adapter_status["python"],
                "a2world_rgb_adapter": rgb_adapter,
                "condition_preparation": {
                    "condition_images": condition["condition_images"],
                    "height": condition["height"],
                    "width": condition["width"],
                    "artifact_playback_fps": condition["output_fps"],
                },
                "action_adapter": adapter_status["action_adapter"],
                "action_preparation": action_metadata,
                "prepared_actions_path": str(
                    actions_path.relative_to(project_root)
                ),
                "a2world_combined_output": str(
                    (generated_dir / "a2world_combined.mp4").relative_to(project_root)
                ),
            }

        elif model_name in {"ctrl", "ctrl_world"}:
            model_name = "ctrl_world"
            update_status(
                run_dir,
                phase="prepare_ctrl_world",
                progress=0.30,
            )
            adapter = CtrlWorldAdapter(project_root, wm_config)
            adapter_status = adapter.validate_rollout(rollout)
            if not adapter_status["available"]:
                raise ValidationError(
                    "; ".join(adapter_status["unavailable_reasons"])
                )
            condition = adapter.prepare_condition(
                rollout,
                cut_frame=cut_frame,
                output_dir=prepared_dir,
            )
            controls = adapter.prepare_controls(
                rollout,
                cut_frame=cut_frame,
                output_dir=prepared_dir,
            )

            update_status(
                run_dir,
                phase="gpu_recheck",
                progress=0.40,
            )
            gpu_index = int(wm_config["gpu_index"])
            gpu_before_generation = ensure_gpu_below_threshold(gpu_index)

            update_status(
                run_dir,
                phase="generate_suffix",
                progress=0.45,
            )
            generated = adapter.generate(
                rollout=rollout,
                condition=condition,
                controls=controls,
                output_dir=generated_dir,
            )
            generated_paths = adapter.save_result(generated, generated_dir)
            ctrl_generation = read_json(
                generated_dir / "ctrl_world_generation.json",
                {},
            )
            raw_indices = (
                ctrl_generation.get("generated_real_frame_indices")
                if isinstance(ctrl_generation, dict)
                else None
            )
            if not isinstance(raw_indices, list):
                raw_indices = controls.get("generated_real_frame_indices")
            real_frame_indices = (
                [int(index) for index in raw_indices]
                if isinstance(raw_indices, list)
                else None
            )
            if not real_frame_indices:
                raise ValidationError(
                    "Ctrl-World generation did not provide source-frame alignment"
                )

            model_provenance = {
                "camera_mapping": adapter_status["camera_mapping"],
                "duplicated_camera": adapter_status["duplicated_camera"],
                "ctrl_world_source_root": adapter_status["source_root"],
                "ctrl_world_python": adapter_status["python"],
                "ctrl_world_svd_model_path": adapter_status["svd_model_path"],
                "ctrl_world_clip_model_path": adapter_status["clip_model_path"],
                "ctrl_world_data_stat_path": adapter_status["data_stat_path"],
                "condition_preparation": {
                    "condition_images": condition["condition_images"],
                    "height": condition["height"],
                    "width": condition["width"],
                },
                "control_adapter": adapter_status["control_adapter"],
                "control_semantics": adapter_status["control_semantics"],
                "control_preparation": {
                    key: value
                    for key, value in controls.items()
                    if key != "path"
                },
                "prepared_controls_path": str(
                    Path(controls["path"]).relative_to(project_root)
                ),
                "ctrl_world_generation": ctrl_generation,
                "generated_real_frame_indices": real_frame_indices,
                "generated_real_start_frame": int(real_frame_indices[0]),
                "artifact_playback_fps": controls["effective_fps"],
                "ctrl_world_exterior_2_debug_output": (
                    str(
                        (
                            generated_dir
                            / "ctrl_world_exterior_2_duplicate.mp4"
                        ).relative_to(project_root)
                    )
                    if (
                        generated_dir
                        / "ctrl_world_exterior_2_duplicate.mp4"
                    ).is_file()
                    else None
                ),
            }
        else:
            raise ValidationError(
                f"Unsupported Repair world model in worker: {model_name}"
            )

        update_status(
            run_dir,
            phase="metrics",
            progress=0.88,
        )
        metrics = compute_metrics(
            project_root=project_root,
            rollout=rollout,
            generated=generated,
            cut_frame=cut_frame,
            real_frame_indices=real_frame_indices,
        )
        atomic_json(run_dir / "metrics.json", metrics)

        provenance = read_json(run_dir / "provenance.json", {})
        if not isinstance(provenance, dict):
            provenance = {}
        provenance.update(
            {
                "world_model": model_name,
                "gt_action_end": int(len(actions)),
                "gt_future_action_count": int(len(future_actions)),
                "alignment_validation": smoke,
                "gpu_recheck_before_generation": {
                    "index": gpu_index,
                    "gpu_utilization_percent": gpu_before_generation.get(
                        "gpu_utilization_percent"
                    ),
                    "memory_free_mib": gpu_before_generation.get(
                        "memory_free_mib"
                    ),
                },
                **model_provenance,
                "standardized_generated_includes_condition": False,
                "output_paths": generated_paths,
                "completed_at": dt.datetime.now(
                    dt.timezone.utc
                ).isoformat(),
            }
        )
        atomic_json(run_dir / "provenance.json", provenance)
        update_status(
            run_dir,
            status="complete",
            phase="complete",
            progress=1.0,
            generated=generated_paths,
            error=None,
        )
    except Exception as error:
        update_status(
            run_dir,
            status="failed",
            phase="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
