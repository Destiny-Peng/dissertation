"""Wan2.2 I2V integration; LF3R exports inputs, external Wan runs inference."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from backend_core import ValidationError


def instruction_for(rollout: dict[str, Any]) -> str:
    for key in ("task_description", "instruction"):
        value = rollout.get(key)
        if isinstance(value, str) and value.strip():
            return value  # Preserve the original instruction, without prompt expansion.
    return ""


def wan_eligibility(project_root: Path, rollout: dict[str, Any]) -> list[str]:
    reasons = []
    if str(rollout.get("ground_truth_outcome") or "").lower() != "success":
        reasons.append("Wan2.2 requires a success rollout")
    value = (rollout.get("camera_video_paths") or {}).get("cam_high")
    if not value or not (project_root / str(value)).is_file():
        reasons.append("Wan2.2 cam_high RGB video is unavailable")
    if not instruction_for(rollout):
        reasons.append("Wan2.2 task instruction is unavailable")
    return reasons


class WanAdapter:
    name = "wan2_2"

    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        self.project_root = project_root.resolve()
        self.config = dict(config)

    def _path(self, value: Any) -> Path | None:
        if not value or not str(value).strip():
            return None
        path = Path(str(value)).expanduser()
        # External environments/checkpoints/source are intentionally supported.
        return Path(os.path.abspath(path if path.is_absolute() else self.project_root / path))

    def validate_rollout(self, rollout: dict[str, Any]) -> dict[str, Any]:
        python = self._path(self.config.get("python"))
        checkpoint = self._path(self.config.get("checkpoint"))
        source = self._path(self.config.get("source_root") or os.environ.get("WAN_SOURCE_ROOT") or "repos/Wan2.2")
        reasons = wan_eligibility(self.project_root, rollout)
        if python is None or not python.is_file() or not os.access(python, os.X_OK):
            reasons.append(f"Wan Python does not exist or is not executable: {python or '(unset)'}")
        if checkpoint is None or not checkpoint.is_dir():
            reasons.append(f"Wan checkpoint directory does not exist: {checkpoint or '(unset)'}")
        if source is None or not source.is_dir() or not (source / "generate.py").is_file():
            reasons.append(f"Wan source / generate.py is unavailable: {source or '(set Wan source directory or WAN_SOURCE_ROOT)'}")
        self.status = {
            "adapter": self.name, "available": not reasons,
            "unavailable_reasons": reasons, "python": str(python) if python else None,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "source_root": str(source) if source else None,
            "checkpoint_type": "i2v_A14B", "task": "i2v-A14B",
            "instruction": instruction_for(rollout), "camera_mapping": {"image": "cam_high"},
            "duplicated_camera": False, "runtime_libero_required": False,
            "requires_actions": False, "requires_sim_state": False,
        }
        return self.status

    def prepare_condition(self, rollout: dict[str, Any], *, cut_frame: int, output_dir: Path) -> dict[str, Any]:
        import imageio.v2 as imageio
        from PIL import Image

        reasons = wan_eligibility(self.project_root, rollout)
        if reasons:
            raise ValidationError("; ".join(reasons))
        if cut_frame < 0 or cut_frame >= int(rollout.get("total_frames") or 0) - 1:
            raise ValidationError("Wan2.2 cut point is outside the valid RGB span")
        source = (self.project_root / rollout["camera_video_paths"]["cam_high"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        reader = imageio.get_reader(str(source))
        try:
            frame = reader.get_data(cut_frame)
        finally:
            reader.close()
        image = (output_dir / "condition_cam_high.png").resolve()
        Image.fromarray(frame).convert("RGB").save(image)
        instruction = instruction_for(rollout)
        instruction_path = (output_dir / "instruction.txt").resolve()
        instruction_path.write_text(instruction, encoding="utf-8")
        condition = {
            "image": str(image), "instruction": instruction,
            "instruction_path": str(instruction_path), "source_video": str(source),
            "condition_frame": cut_frame, "rollout_id": rollout.get("id"),
            "input_contract": "single cam_high RGB frame + original task instruction",
            "future_actions_used": False, "sim_state_used": False,
            "prefix_video_used": False, "runtime_libero_used": False,
        }
        (output_dir / "wan_input.json").write_text(json.dumps(condition, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return condition

    def generate(self, *, condition: dict[str, Any], output_dir: Path) -> dict[str, Path]:
        status = self.status
        if not status["available"]:
            raise ValidationError("; ".join(status["unavailable_reasons"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        output = (output_dir / "cam_high.mp4").resolve()
        command = [status["python"], "-u", str(Path(status["source_root"]) / "generate.py"),
                   "--task", "i2v-A14B", "--ckpt_dir", status["checkpoint"],
                   "--image", condition["image"], "--prompt", condition["instruction"],
                   "--save_file", str(output)]
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)  # Do not inject LF3R dependencies into Wan.
        env["CUDA_VISIBLE_DEVICES"] = str(self.config.get("gpu_index", 0))
        print("Wan2.2 inference: " + json.dumps(command, ensure_ascii=False), flush=True)
        try:
            completed = subprocess.run(command, cwd=status["source_root"], env=env, check=False)
        except OSError as error:
            raise ValidationError(f"Wan2.2 inference failed: {error}") from error
        if completed.returncode:
            raise ValidationError(f"Wan2.2 inference failed with exit code {completed.returncode}")
        if not output.is_file() or output.stat().st_size == 0:
            raise ValidationError("Wan2.2 output video is missing or empty")
        return {"cam_high": output}
