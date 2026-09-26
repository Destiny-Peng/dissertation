"""Launch LIBERO alignment validation in the project-local OpenVLA environment."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from backend_core import ValidationError


def _project_python(project_root: Path) -> Path:
    configured = str(os.environ.get("LF3R_ENV_OPENVLA") or "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured) / "bin" / "python")
    candidates.append(project_root / "conda_envs" / "LF3R-openvla" / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValidationError(
        "LF3R OpenVLA Python is unavailable; expected conda_envs/LF3R-openvla/bin/python"
    )


def run_alignment_subprocess(
    *,
    project_root: Path,
    rollout: dict[str, Any],
    cut_frame: int,
    min_psnr: float,
    gpu_index: int | None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    python = _project_python(project_root)
    scratch = project_root / "cache" / "lf3r_annotator" / "repair_alignment"
    scratch.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).resolve().parent / "alignment_cli.py"

    with tempfile.TemporaryDirectory(prefix="alignment-", dir=scratch) as temp_name:
        temp_dir = Path(temp_name)
        rollout_path = temp_dir / "rollout.json"
        output_path = temp_dir / "result.json"
        rollout_path.write_text(
            json.dumps(rollout, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        python_paths = [
            str(Path(__file__).resolve().parents[1]),
            str(project_root / "repos" / "LIBERO"),
        ]
        existing = environment.get("PYTHONPATH", "")
        if existing:
            python_paths.append(existing)
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        environment["MUJOCO_GL"] = "egl"
        environment["PYOPENGL_PLATFORM"] = "egl"
        environment["LIBERO_CONFIG_PATH"] = str(project_root / "cache" / "libero")
        if gpu_index is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))

        command = [
            str(python),
            str(helper),
            "--project-root",
            str(project_root),
            "--rollout-json",
            str(rollout_path),
            "--cut-frame",
            str(int(cut_frame)),
            "--min-psnr",
            str(float(min_psnr)),
            "--output-json",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(project_root),
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise ValidationError(
                f"LIBERO alignment smoke test timed out after {timeout_seconds:.0f}s"
            ) from error
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            if len(message) > 3000:
                message = message[-3000:]
            raise ValidationError(
                "LIBERO alignment smoke subprocess failed"
                + (": " + message if message else f" (exit {completed.returncode})")
            )
        if not output_path.is_file():
            raise ValidationError("LIBERO alignment smoke test produced no result JSON")
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValidationError("LIBERO alignment smoke result is not an object")
        result["runtime"] = {
            "python": str(python.relative_to(project_root)),
            "gpu_index": gpu_index,
            "mujoco_gl": "egl",
        }
        return result
