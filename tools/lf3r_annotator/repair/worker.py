#!/usr/bin/env python3
"""Execute one LF3R Synthetic Suffix run.

The worker validates LIBERO indexing before invoking the world model. A failed
alignment smoke test is terminal and no generated suffix is accepted.
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
from repair.adapters import A2WorldAdapter
from repair.trajectory import load_actions, load_states, run_libero_alignment_smoke


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


def image_metrics(real: Any, generated: Any) -> dict[str, float | None]:
    import numpy as np

    a = np.asarray(real, dtype=np.float32)
    b = np.asarray(generated, dtype=np.float32)
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
    Repair jobs must never trigger an implicit checkpoint download, so the
    metric is opt-in through LF3R_ENABLE_LPIPS=1 after weights are installed.
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
) -> dict[str, Any]:
    camera_paths = rollout.get("camera_video_paths") or {}
    lpips_model, torch, lpips_status = _optional_lpips()
    result: dict[str, Any] = {
        "alignment": {
            "real_suffix_start_frame": cut_frame + 1,
            "generated_includes_condition": False,
            "comparison_basis": "frame/action index",
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
        real_start = cut_frame + 1
        count = min(
            len(generated_frames),
            max(0, len(real_frames) - real_start),
        )
        psnr_values: list[float] = []
        ssim_values: list[float] = []
        lpips_values: list[float] = []
        for index in range(count):
            real_frame = real_frames[real_start + index]
            generated_frame = generated_frames[index]
            per_frame = image_metrics(real_frame, generated_frame)
            if (
                per_frame["psnr"] is not None
                and math.isfinite(float(per_frame["psnr"]))
            ):
                psnr_values.append(float(per_frame["psnr"]))
            if per_frame["ssim"] is not None:
                ssim_values.append(float(per_frame["ssim"]))
            if lpips_model is not None and torch is not None:
                a = np.asarray(real_frame, dtype=np.float32)
                b = np.asarray(generated_frame, dtype=np.float32)
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
            "real_available_frames": max(
                0,
                len(real_frames) - real_start,
            ),
            "generated_frames": len(generated_frames),
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

    try:
        update_status(
            run_dir,
            status="running",
            phase="load_trajectory",
            progress=0.08,
        )
        actions = load_actions(project_root, rollout)
        states = load_states(project_root, rollout)
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
        smoke = run_libero_alignment_smoke(
            project_root=project_root,
            rollout=rollout,
            states=states,
            actions=actions,
            cut_frame=cut_frame,
            min_psnr=float(config.get("alignment_min_psnr", 20.0)),
        )
        atomic_json(run_dir / "alignment.json", smoke)
        if not smoke["passed"]:
            raise ValidationError(
                "LIBERO alignment smoke test failed; generation was not started"
            )

        update_status(
            run_dir,
            phase="prepare_a2world",
            progress=0.30,
        )
        wm_config = dict(config["world_model"])
        adapter = A2WorldAdapter(project_root, wm_config)
        adapter_status = adapter.validate_rollout(rollout)
        if not adapter_status["available"]:
            raise ValidationError(
                "; ".join(adapter_status["unavailable_reasons"])
            )

        prepared_dir = run_dir / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
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
            phase="generate_suffix",
            progress=0.45,
        )
        generated = adapter.generate(
            condition=condition,
            actions_path=actions_path,
            output_dir=run_dir / "generated",
        )
        generated_paths = adapter.save_result(
            generated,
            run_dir / "generated",
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
        )
        atomic_json(run_dir / "metrics.json", metrics)

        provenance = read_json(
            run_dir / "provenance.json",
            {},
        )
        if not isinstance(provenance, dict):
            provenance = {}
        provenance.update(
            {
                "gt_action_end": int(len(actions)),
                "gt_future_action_count": int(len(future_actions)),
                "alignment_validation": smoke,
                "camera_mapping": adapter_status["camera_mapping"],
                "duplicated_camera": adapter_status["duplicated_camera"],
                "a2world_view_ids": adapter_status["view_ids"],
                "action_adapter": adapter_status["action_adapter"],
                "action_preparation": action_metadata,
                "prepared_actions_path": str(
                    actions_path.relative_to(project_root)
                ),
                "a2world_combined_output": str(
                    (
                        run_dir
                        / "generated"
                        / "a2world_combined.mp4"
                    ).relative_to(project_root)
                ),
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
