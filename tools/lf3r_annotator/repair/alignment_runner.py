"""Launch LIBERO alignment validation in the project-local OpenVLA environment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from backend_core import ValidationError


REPAIR_RUNTIME_MODULES = (
    "numpy",
    "h5py",
    "imageio",
    "scipy",
    "robosuite",
    "libero.libero",
)


def _runtime_environment(project_root: Path) -> dict[str, str]:
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
    return environment


def _configured_python(value: str) -> Path:
    path = Path(value).expanduser()
    if path.name.startswith("python") and path.is_file():
        return path.resolve()
    return (path / "bin" / "python").resolve()


def repair_python_candidates(project_root: Path) -> list[Path]:
    raw_candidates: list[Path] = []
    for key in ("LF3R_ENV_REPAIR", "LF3R_ENV_OPENVLA"):
        configured = str(os.environ.get(key) or "").strip()
        if configured:
            raw_candidates.append(_configured_python(configured))
    raw_candidates.extend(
        [
            project_root / "conda_envs" / "LF3R-openvla" / "bin" / "python",
            project_root / "conda_envs" / "LF3R-ctrl-world" / "bin" / "python",
            project_root / "conda_envs" / "LF3R-Ctrl-World" / "bin" / "python",
        ]
    )
    current = Path(sys.executable).resolve()
    try:
        current.relative_to(project_root)
        raw_candidates.append(current)
    except ValueError:
        pass

    candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        resolved = candidate.expanduser().resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            candidates.append(resolved)
    return candidates


def _probe_repair_python(project_root: Path, python: Path) -> tuple[bool, str]:
    if not python.is_file():
        return False, "python executable not found"
    code = (
        "import importlib\n"
        f"mods={REPAIR_RUNTIME_MODULES!r}\n"
        "missing=[]\n"
        "for name in mods:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        missing.append(name + ': ' + type(exc).__name__ + ': ' + str(exc))\n"
        "print('\\n'.join(missing))\n"
        "raise SystemExit(1 if missing else 0)\n"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", code],
            cwd=str(project_root),
            env=_runtime_environment(project_root),
            text=True,
            capture_output=True,
            check=False,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"{type(error).__name__}: {error}"
    if completed.returncode == 0:
        return True, "ok"
    detail = (completed.stdout or completed.stderr or "import preflight failed").strip()
    return False, detail[-1600:]


def resolve_repair_python(
    project_root: Path,
    *,
    preferred_python: Path | None = None,
    runtime_label: str = "Repair/LIBERO",
) -> Path:
    candidates = (
        [preferred_python.expanduser().resolve()]
        if preferred_python is not None
        else repair_python_candidates(project_root)
    )
    diagnostics: list[str] = []
    for candidate in candidates:
        passed, detail = _probe_repair_python(project_root, candidate)
        if passed:
            return candidate.resolve()
        try:
            label = str(candidate.relative_to(project_root))
        except ValueError:
            label = str(candidate)
        diagnostics.append(f"{label}: {detail}")
    if preferred_python is not None:
        raise ValidationError(
            f"{runtime_label} runtime is missing required modules "
            + ", ".join(REPAIR_RUNTIME_MODULES)
            + ". "
            + " | ".join(diagnostics)
        )
    raise ValidationError(
        "No project-local Repair/LIBERO Python has the required runtime modules "
        + ", ".join(REPAIR_RUNTIME_MODULES)
        + ". Checked: "
        + " | ".join(diagnostics)
    )


def _project_python(
    project_root: Path,
    *,
    preferred_python: Path | None = None,
    runtime_label: str = "Repair/LIBERO",
) -> Path:
    return resolve_repair_python(
        project_root,
        preferred_python=preferred_python,
        runtime_label=runtime_label,
    )


def run_alignment_subprocess(
    *,
    project_root: Path,
    rollout: dict[str, Any],
    cut_frame: int,
    min_psnr: float,
    gpu_index: int | None,
    python_override: Path | None = None,
    runtime_label: str = "Repair/LIBERO",
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    python = _project_python(
        project_root,
        preferred_python=python_override,
        runtime_label=runtime_label,
    )
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
        environment = _runtime_environment(project_root)
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
