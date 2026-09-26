"""World-model adapter contracts for LF3R Repair."""

from __future__ import annotations

import abc
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from backend_core import ValidationError
from .alignment import project_path


class WorldModelAdapter(abc.ABC):
    """Consumer-specific mapping from LF3R data into a world model."""

    name = "world_model"

    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        self.project_root = project_root.resolve()
        self.config = dict(config)

    @abc.abstractmethod
    def validate_rollout(self, rollout: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_condition(
        self,
        rollout: dict[str, Any],
        *,
        cut_frame: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_actions(self, actions: Any, *, output_dir: Path) -> Path:
        raise NotImplementedError

    @abc.abstractmethod
    def generate(
        self,
        *,
        condition: dict[str, Any],
        actions_path: Path,
        output_dir: Path,
    ) -> dict[str, Path]:
        raise NotImplementedError

    def save_result(self, generated: dict[str, Path], output_dir: Path) -> dict[str, str]:
        return {
            camera: str(path.resolve().relative_to(self.project_root))
            for camera, path in generated.items()
        }


class A2WorldAdapter(WorldModelAdapter):
    """A2World LIBERO adapter.

    Manifest camera names stay physical.  The consumer-specific A2World names
    are introduced here and are recorded in provenance.
    """

    name = "a2world"
    DEFAULT_CAMERA_MAPPING = {
        "agentview": "cam_high",
        "eye_in_hand": "cam_wrist",
    }
    ACTION_ADAPTER = "libero_7d_to_a2world_libero"

    def _camera_mapping(self, rollout: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, str]]]:
        cameras = rollout.get("camera_video_paths")
        if not isinstance(cameras, dict) or not cameras:
            raise ValidationError("Selected rollout has no camera_video_paths")
        requested = self.config.get("camera_mapping")
        mapping = (
            {str(k): str(v) for k, v in requested.items()}
            if isinstance(requested, dict) and requested
            else dict(self.DEFAULT_CAMERA_MAPPING)
        )
        duplicated: list[dict[str, str]] = []
        allow_duplicate = bool(self.config.get("duplicate_missing_views", False))
        for consumer_view, manifest_view in list(mapping.items()):
            if manifest_view in cameras:
                continue
            if allow_duplicate and "cam_high" in cameras:
                mapping[consumer_view] = "cam_high"
                duplicated.append({
                    "consumer_view": consumer_view,
                    "requested_manifest_view": manifest_view,
                    "used_manifest_view": "cam_high",
                })
                continue
            raise ValidationError(
                f"A2World view {consumer_view} requires manifest camera {manifest_view}; "
                "enable duplicate_missing_views explicitly to substitute cam_high"
            )
        return mapping, duplicated

    def validate_rollout(self, rollout: dict[str, Any]) -> dict[str, Any]:
        mapping, duplicated = self._camera_mapping(rollout)
        checkpoint_type = str(self.config.get("checkpoint_type") or "libero_adapted")
        if checkpoint_type not in {"generic_pretrained", "libero_adapted", "custom"}:
            raise ValidationError(
                "A2World checkpoint_type must be generic_pretrained, libero_adapted, or custom"
            )
        checkpoint = str(self.config.get("checkpoint") or "").strip()
        if not checkpoint:
            checkpoint = (
                "checkpoints/a2world-pretrained.pt"
                if checkpoint_type == "generic_pretrained"
                else "checkpoints/a2world-libero.pt"
            )
        checkpoint_path = project_path(self.project_root, checkpoint)
        base_value = str(self.config.get("base_checkpoints") or "checkpoints").strip()
        base_path = project_path(self.project_root, base_value)
        command = str(self.config.get("command") or os.environ.get("LF3R_A2WORLD_COMMAND") or "a2world-demo")
        executable = shutil.which(command) if os.sep not in command else command
        available = bool(executable) and checkpoint_path.is_file() and base_path.exists()
        reasons: list[str] = []
        if not executable:
            reasons.append(f"A2World command not found: {command}")
        if not checkpoint_path.is_file():
            reasons.append(
                "checkpoint is not installed inside PROJECT_ROOT: "
                + str(checkpoint_path.relative_to(self.project_root))
            )
        if not base_path.exists():
            reasons.append(
                "base checkpoints are not installed inside PROJECT_ROOT: "
                + str(base_path.relative_to(self.project_root))
            )
        return {
            "adapter": self.name,
            "available": available,
            "unavailable_reasons": reasons,
            "checkpoint": str(checkpoint_path.relative_to(self.project_root)),
            "checkpoint_type": checkpoint_type,
            "base_checkpoints": str(base_path.relative_to(self.project_root)),
            "camera_mapping": mapping,
            "duplicated_camera": duplicated,
            "action_adapter": self.ACTION_ADAPTER,
            "command": command,
            "variant": "libero",
            "rollout_mode": "autoregressive",
        }

    @staticmethod
    def _read_video_frame(path: Path, frame_index: int) -> Any:
        try:
            import imageio.v2 as imageio
        except ImportError as error:
            raise ValidationError("imageio is required to prepare A2World condition frames") from error
        reader = imageio.get_reader(str(path))
        try:
            return reader.get_data(frame_index)
        finally:
            reader.close()

    def prepare_condition(
        self,
        rollout: dict[str, Any],
        *,
        cut_frame: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        try:
            import imageio.v2 as imageio
        except ImportError as error:
            raise ValidationError("imageio is required to prepare A2World inputs") from error

        mapping, duplicated = self._camera_mapping(rollout)
        camera_paths = rollout["camera_video_paths"]
        fps = float(rollout.get("fps") or 30.0)
        condition_dir = output_dir / "condition"
        condition_dir.mkdir(parents=True, exist_ok=True)
        videos: dict[str, Path] = {}
        for consumer_view, manifest_view in mapping.items():
            source = project_path(self.project_root, str(camera_paths[manifest_view]))
            frame = self._read_video_frame(source, cut_frame)
            target = condition_dir / f"{consumer_view}.mp4"
            writer = imageio.get_writer(str(target), fps=fps)
            try:
                writer.append_data(frame)
            finally:
                writer.close()
            videos[consumer_view] = target
        metadata = {
            "cut_frame": int(cut_frame),
            "camera_mapping": mapping,
            "duplicated_camera": duplicated,
            "condition_videos": {
                key: str(path.relative_to(self.project_root))
                for key, path in videos.items()
            },
        }
        (condition_dir / "condition.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {"videos": videos, **metadata}

    def prepare_actions(self, actions: Any, *, output_dir: Path) -> Path:
        try:
            import numpy as np
        except ImportError as error:
            raise ValidationError("numpy is required to prepare A2World actions") from error
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 7:
            raise ValidationError(
                f"A2World LIBERO expects future actions shaped [T,7], got {array.shape}"
            )
        path = output_dir / "future_actions.npz"
        np.savez_compressed(path, actions=array)
        return path

    def generate(
        self,
        *,
        condition: dict[str, Any],
        actions_path: Path,
        output_dir: Path,
    ) -> dict[str, Path]:
        status = self.validate_rollout(self.config["_rollout"])
        if not status["available"]:
            raise ValidationError("; ".join(status["unavailable_reasons"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = project_path(self.project_root, status["checkpoint"])
        base_checkpoints = project_path(self.project_root, status["base_checkpoints"])
        output = output_dir / "a2world_rollout.mp4"
        command = [
            str(status["command"]),
            "--variant", "libero",
            "--checkpoint", str(checkpoint),
            "--input",
            str(condition["videos"]["agentview"]),
            str(condition["videos"]["eye_in_hand"]),
            "--actions", str(actions_path),
            "--base-checkpoints", str(base_checkpoints),
            "--output", str(output),
            "--autoregressive",
        ]
        completed = subprocess.run(
            command,
            cwd=str(self.project_root / "repos" / "A2World" / "world_model")
            if (self.project_root / "repos" / "A2World" / "world_model").is_dir()
            else str(self.project_root),
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"A2World exited with code {completed.returncode}")

        # LF3R's result contract is per physical view.  The upstream CLI may be
        # wrapped locally to emit these files; common sidecar names are also
        # accepted.  We never invent a second camera by copying generated video.
        candidates = {
            "cam_high": [
                output_dir / "cam_high.mp4",
                output_dir / "a2world_rollout.agentview.mp4",
                output_dir / "a2world_rollout_agentview.mp4",
            ],
            "cam_wrist": [
                output_dir / "cam_wrist.mp4",
                output_dir / "a2world_rollout.eye_in_hand.mp4",
                output_dir / "a2world_rollout_eye_in_hand.mp4",
            ],
        }
        generated = {
            camera: next((path for path in paths if path.is_file()), None)
            for camera, paths in candidates.items()
        }
        if all(path is None for path in generated.values()) and output.is_file():
            # Preserve the upstream artifact for debugging, but do not pretend
            # a tiled/combined video is either physical camera.
            raise ValidationError(
                "A2World produced only a combined rollout video. LF3R requires "
                "per-view cam_high/cam_wrist outputs; configure an A2World bridge "
                "that emits per-view sidecars using the documented names."
            )
        missing = [camera for camera, path in generated.items() if path is None]
        if missing:
            raise ValidationError(
                "A2World generation is missing per-view output(s): " + ", ".join(missing)
            )
        return {camera: path for camera, path in generated.items() if path is not None}
