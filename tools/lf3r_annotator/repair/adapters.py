"""World-model adapter contracts for LF3R Repair."""

from __future__ import annotations

import abc
import json
import os
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
    @staticmethod
    def _transform_rgb(frame: Any, transform: str) -> Any:
        try:
            import numpy as np
        except ImportError as error:
            raise ValidationError("numpy is required for A2World RGB adaptation") from error
        array = np.asarray(frame)
        if transform == "raw":
            adapted = array
        elif transform == "horizontal_flip":
            adapted = array[:, ::-1]
        elif transform == "vertical_flip":
            adapted = array[::-1, :]
        elif transform == "rotate_180":
            adapted = array[::-1, ::-1]
        else:
            raise ValidationError(f"Unknown A2World RGB transform: {transform}")
        return np.ascontiguousarray(adapted)

    def configure_alignment_rgb(self, smoke: dict[str, Any]) -> dict[str, Any]:
        comparisons = smoke.get("comparisons") if isinstance(smoke, dict) else {}
        if not isinstance(comparisons, dict):
            comparisons = {}
        mapping: dict[str, str] = {}
        provenance: dict[str, Any] = {}
        for manifest_view in ("cam_high", "cam_wrist"):
            item = comparisons.get(manifest_view)
            if not isinstance(item, dict):
                continue
            sim_to_manifest = str(item.get("orientation_transform") or "")
            if sim_to_manifest not in self.MANIFEST_TO_A2WORLD_RGB:
                raise ValidationError(
                    f"Alignment did not provide a supported RGB orientation for {manifest_view}"
                )
            manifest_to_a2world = self.MANIFEST_TO_A2WORLD_RGB[sim_to_manifest]
            mapping[manifest_view] = manifest_to_a2world
            provenance[manifest_view] = {
                "sim_to_manifest": sim_to_manifest,
                "sim_to_a2world_training": self.A2WORLD_LIBERO_SIM_TO_TRAINING_RGB,
                "manifest_to_a2world": manifest_to_a2world,
                "a2world_to_manifest": manifest_to_a2world,
            }
        self.config["_manifest_to_a2world_rgb"] = mapping
        self.config["_rgb_adapter_provenance"] = provenance
        return provenance

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

    def save_result(
        self,
        generated: dict[str, Path],
        output_dir: Path,
    ) -> dict[str, str]:
        del output_dir
        return {
            camera: str(path.resolve().relative_to(self.project_root))
            for camera, path in generated.items()
        }


class A2WorldAdapter(WorldModelAdapter):
    """A2World adapter following the upstream LIBERO inference contract.

    A2World's public LIBERO path uses two input views in this order:
      agentview (embedding id 2), eye_in_hand (embedding id 0)

    Its public action loader applies libero_servo_actions to raw LIBERO
    controls. LF3R performs the same transform explicitly and invokes
    a2world.rollout with --action-format precomputed so the exact action
    representation is captured in our provenance.
    """

    name = "a2world"
    DEFAULT_CAMERA_MAPPING = {
        "agentview": "cam_high",
        "eye_in_hand": "cam_wrist",
    }
    VIEW_IDS = {
        "agentview": 2,
        "eye_in_hand": 0,
    }
    ACTION_DIM_PER_ARM = 7
    ACTION_DIM = 14
    ACTION_CHUNK_SIZE = 20
    LIBERO_SERVO_SCALE = (
        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0,
        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0,
    )
    ACTION_ADAPTER = "a2world.actions.libero_servo_actions"
    GENERIC_ACTION_ADAPTER = "a2world.actions.normalize_action_shape"
    A2WORLD_LIBERO_SIM_TO_TRAINING_RGB = "horizontal_flip"
    MANIFEST_TO_A2WORLD_RGB = {
        "raw": "horizontal_flip",
        "horizontal_flip": "raw",
        "vertical_flip": "rotate_180",
        "rotate_180": "vertical_flip",
    }

    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        super().__init__(project_root, config)
        self._validation: dict[str, Any] | None = None

    def _camera_mapping(
        self,
        rollout: dict[str, Any],
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        cameras = rollout.get("camera_video_paths")
        if not isinstance(cameras, dict) or not cameras:
            raise ValidationError("Selected rollout has no camera_video_paths")
        requested = self.config.get("camera_mapping")
        mapping = (
            {str(key): str(value) for key, value in requested.items()}
            if isinstance(requested, dict) and requested
            else dict(self.DEFAULT_CAMERA_MAPPING)
        )
        expected = set(self.DEFAULT_CAMERA_MAPPING)
        if set(mapping) != expected:
            raise ValidationError(
                "A2World camera_mapping must define exactly agentview and eye_in_hand"
            )
        duplicated: list[dict[str, str]] = []
        allow_duplicate = bool(self.config.get("duplicate_missing_views", False))
        for consumer_view, manifest_view in list(mapping.items()):
            if manifest_view in cameras:
                continue
            if allow_duplicate and "cam_high" in cameras:
                mapping[consumer_view] = "cam_high"
                duplicated.append(
                    {
                        "consumer_view": consumer_view,
                        "requested_manifest_view": manifest_view,
                        "used_manifest_view": "cam_high",
                    }
                )
                continue
            raise ValidationError(
                f"A2World view {consumer_view} requires manifest camera "
                f"{manifest_view}; enable duplicate_missing_views explicitly "
                "to substitute cam_high inside the adapter"
            )
        return mapping, duplicated

    def _checkpoint(self, checkpoint_type: str) -> Path:
        raw = str(self.config.get("checkpoint") or "").strip()
        if checkpoint_type == "custom" and not raw:
            raise ValidationError("Custom A2World checkpoint requires a path")
        if not raw:
            raw = (
                "checkpoints/a2world-pretrained.pt"
                if checkpoint_type == "generic_pretrained"
                else "checkpoints/a2world-libero.pt"
            )
        return project_path(self.project_root, raw)

    def _source_root(self) -> Path:
        raw = str(
            self.config.get("source_root")
            or os.environ.get("LF3R_A2WORLD_SOURCE")
            or "repos/A2World/world_model"
        ).strip()
        return project_path(self.project_root, raw)

    def _python(self) -> Path:
        configured = str(
            self.config.get("python")
            or os.environ.get("LF3R_A2WORLD_PYTHON")
            or ""
        ).strip()
        candidates: list[Path] = []
        if configured:
            candidates.append(project_path(self.project_root, configured))
        candidates.extend(
            [
                self.project_root / "conda_envs" / "LF3R-a2world" / "bin" / "python",
                self.project_root / "repos" / "A2World" / ".venv" / "bin" / "python",
                self.project_root / "conda_envs" / "A2World" / "bin" / "python",
            ]
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        # Keep A2World reproducible and project-local.  Do not silently use the
        # WebUI's system interpreter when the dedicated environment is absent.
        return (self.project_root / "conda_envs" / "LF3R-a2world" / "bin" / "python").resolve()

    def validate_rollout(self, rollout: dict[str, Any]) -> dict[str, Any]:
        mapping, duplicated = self._camera_mapping(rollout)
        checkpoint_type = str(
            self.config.get("checkpoint_type") or "libero_adapted"
        ).strip()
        if checkpoint_type not in {
            "generic_pretrained",
            "libero_adapted",
            "custom",
        }:
            raise ValidationError(
                "A2World checkpoint_type must be generic_pretrained, "
                "libero_adapted, or custom"
            )
        checkpoint = self._checkpoint(checkpoint_type)
        base_path = project_path(
            self.project_root,
            str(self.config.get("base_checkpoints") or "checkpoints").strip(),
        )
        source_root = self._source_root()
        python = self._python()
        variant = (
            "pretrained"
            if checkpoint_type == "generic_pretrained"
            else str(self.config.get("variant") or "libero").strip()
        )
        if variant not in {"libero", "pretrained"}:
            raise ValidationError("A2World variant must be libero or pretrained")
        if checkpoint_type == "libero_adapted" and variant != "libero":
            raise ValidationError("LIBERO-adapted checkpoint requires variant=libero")

        reasons: list[str] = []
        if not checkpoint.is_file():
            reasons.append(
                "A2World checkpoint is not installed inside PROJECT_ROOT: "
                + str(checkpoint.relative_to(self.project_root))
            )
        if not base_path.is_dir():
            reasons.append(
                "A2World base checkpoints directory is unavailable: "
                + str(base_path.relative_to(self.project_root))
            )
        rollout_module = source_root / "a2world" / "rollout.py"
        if not rollout_module.is_file():
            reasons.append(
                "A2World source is unavailable: "
                + str(source_root.relative_to(self.project_root))
            )
        if not python.is_file():
            reasons.append("A2World Python interpreter is unavailable: " + str(python))

        # Input trajectories are LIBERO servo controls regardless of which
        # checkpoint variant is selected. Upstream A2World's auto loader also
        # applies libero_servo_actions to 3D LIBERO action arrays for the
        # pretrained variant, so preprocessing is data-domain specific.
        action_adapter = self.ACTION_ADAPTER
        result = {
            "adapter": self.name,
            "available": not reasons,
            "unavailable_reasons": reasons,
            "checkpoint": str(checkpoint.relative_to(self.project_root)),
            "checkpoint_type": checkpoint_type,
            "base_checkpoints": str(base_path.relative_to(self.project_root)),
            "source_root": str(source_root.relative_to(self.project_root)),
            "python": (
                str(python.relative_to(self.project_root))
                if python.is_relative_to(self.project_root)
                else str(python)
            ),
            "camera_mapping": mapping,
            "duplicated_camera": duplicated,
            "view_ids": [
                self.VIEW_IDS["agentview"],
                self.VIEW_IDS["eye_in_hand"],
            ],
            "action_adapter": action_adapter,
            "action_chunk_size": self.ACTION_CHUNK_SIZE,
            "variant": variant,
            "rollout_mode": "autoregressive",
            "standardized_output_includes_condition": False,
        }
        self._validation = result
        return result

    @staticmethod
    def _read_video_frame(path: Path, frame_index: int) -> Any:
        try:
            import imageio.v2 as imageio
        except ImportError as error:
            raise ValidationError(
                "imageio is required to prepare A2World condition frames"
            ) from error
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
            raise ValidationError(
                "imageio is required to prepare A2World inputs"
            ) from error

        mapping, duplicated = self._camera_mapping(rollout)
        camera_paths = rollout["camera_video_paths"]
        condition_dir = output_dir / "condition"
        condition_dir.mkdir(parents=True, exist_ok=True)
        images: dict[str, Path] = {}
        shape: tuple[int, int] | None = None
        for consumer_view in ("agentview", "eye_in_hand"):
            manifest_view = mapping[consumer_view]
            source = project_path(
                self.project_root,
                str(camera_paths[manifest_view]),
            )
            frame = self._read_video_frame(source, cut_frame)
            rgb_transform = (
                self.config.get("_manifest_to_a2world_rgb", {})
                or {}
            ).get(manifest_view)
            if not rgb_transform:
                raise ValidationError(
                    "A2World RGB adaptation requires a passed LIBERO alignment smoke test"
                )
            frame = self._transform_rgb(frame, str(rgb_transform))
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValidationError(
                    f"Unexpected condition frame shape for {manifest_view}: "
                    f"{frame.shape}"
                )
            current_shape = (int(frame.shape[0]), int(frame.shape[1]))
            if shape is None:
                shape = current_shape
            elif current_shape != shape:
                raise ValidationError(
                    "A2World condition views must have the same image shape"
                )
            target = condition_dir / f"{consumer_view}.png"
            imageio.imwrite(str(target), frame)
            images[consumer_view] = target

        assert shape is not None
        metadata = {
            "cut_frame": int(cut_frame),
            "camera_mapping": mapping,
            "duplicated_camera": duplicated,
            "rgb_adapter": self.config.get("_rgb_adapter_provenance", {}),
            "condition_images": {
                key: str(path.relative_to(self.project_root))
                for key, path in images.items()
            },
            "height": shape[0],
            "width": shape[1],
            "output_fps": float(rollout.get("fps") or 30.0),
        }
        (condition_dir / "condition.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {"images": images, **metadata}

    def _prepare_action_array(self, actions: Any) -> tuple[Any, str]:
        try:
            import numpy as np
        except ImportError as error:
            raise ValidationError(
                "numpy is required to prepare A2World actions"
            ) from error

        array = np.asarray(actions, dtype=np.float32)
        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        if array.ndim != 2 or array.shape[1] not in {7, 14}:
            raise ValidationError(
                f"A2World expects LIBERO actions shaped [T,7] or [T,14], "
                f"got {array.shape}"
            )
        if array.shape[1] == 7:
            array = np.pad(array, ((0, 0), (0, 7)))
        array = array.astype(np.float32, copy=True)

        array[:, 6] = (1.0 - array[:, 6]) / 2.0
        scale = np.asarray(self.LIBERO_SERVO_SCALE, dtype=np.float32)
        array *= scale
        return array, self.ACTION_ADAPTER

    def prepare_actions(self, actions: Any, *, output_dir: Path) -> Path:
        try:
            import numpy as np
        except ImportError as error:
            raise ValidationError(
                "numpy is required to prepare A2World actions"
            ) from error

        prepared, adapter_name = self._prepare_action_array(actions)
        requested_count = int(len(prepared))
        if requested_count < 1:
            raise ValidationError("A2World received no future actions")

        pad_count = (-requested_count) % self.ACTION_CHUNK_SIZE
        if pad_count:
            padding = np.zeros(
                (pad_count, self.ACTION_DIM),
                dtype=np.float32,
            )
            padding[:, 6] = prepared[-1, 6]
            prepared = np.concatenate([prepared, padding], axis=0)

        path = output_dir / "future_actions_a2world.npz"
        np.savez_compressed(path, actions=prepared)
        metadata = {
            "adapter": adapter_name,
            "raw_future_action_count": requested_count,
            "prepared_action_count": int(len(prepared)),
            "action_dim": self.ACTION_DIM,
            "chunk_size": self.ACTION_CHUNK_SIZE,
            "tail_padding_count": pad_count,
            "tail_padding_strategy": (
                "none"
                if not pad_count
                else "zero_motion_hold_first_arm_gripper"
            ),
            "input_contract": "LIBERO 7D/14D",
            "output_contract": "A2World precomputed 14D",
        }
        (output_dir / "future_actions_a2world.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def _split_standardized_outputs(
        self,
        *,
        combined_path: Path,
        output_dir: Path,
        mapping: dict[str, str],
        requested_frames: int,
        fps: float,
    ) -> dict[str, Path]:
        try:
            import imageio.v2 as imageio
        except ImportError as error:
            raise ValidationError(
                "imageio is required to split A2World multi-view output"
            ) from error

        reader = imageio.get_reader(str(combined_path))
        frames: list[Any] = []
        try:
            for frame in reader:
                frames.append(frame)
        finally:
            reader.close()
        if len(frames) < requested_frames + 1:
            raise ValidationError(
                "A2World output is shorter than the requested suffix: "
                f"frames={len(frames)}, need={requested_frames + 1} "
                "(including the condition frame)"
            )

        suffix_frames = frames[1 : requested_frames + 1]
        first = suffix_frames[0]
        if first.ndim != 3 or first.shape[-1] != 3 or first.shape[1] % 2:
            raise ValidationError(
                f"Unexpected A2World combined frame shape: {first.shape}"
            )
        view_width = first.shape[1] // 2
        slices = {
            "agentview": (0, view_width),
            "eye_in_hand": (view_width, view_width * 2),
        }
        generated: dict[str, Path] = {}
        seen_manifest_views: set[str] = set()
        for consumer_view in ("agentview", "eye_in_hand"):
            manifest_view = mapping[consumer_view]
            start, stop = slices[consumer_view]
            if manifest_view in seen_manifest_views:
                target = (
                    output_dir
                    / f"a2world_{consumer_view}_from_{manifest_view}_duplicate.mp4"
                )
                expose = False
            else:
                target = output_dir / f"{manifest_view}.mp4"
                expose = True
                seen_manifest_views.add(manifest_view)
            writer = imageio.get_writer(str(target), fps=fps)
            try:
                output_transform = (
                    self.config.get("_manifest_to_a2world_rgb", {})
                    or {}
                ).get(manifest_view)
                if not output_transform:
                    raise ValidationError(
                        f"Missing A2World-to-manifest RGB transform for {manifest_view}"
                    )
                for frame in suffix_frames:
                    view_frame = frame[:, start:stop]
                    writer.append_data(
                        self._transform_rgb(view_frame, str(output_transform))
                    )
            finally:
                writer.close()
            if expose:
                generated[manifest_view] = target

        if not generated:
            raise ValidationError("A2World produced no physical-camera output")
        return generated

    def generate(
        self,
        *,
        condition: dict[str, Any],
        actions_path: Path,
        output_dir: Path,
    ) -> dict[str, Path]:
        status = self._validation
        if status is None:
            raise ValidationError(
                "A2World adapter must validate the rollout before generation"
            )
        if not status["available"]:
            raise ValidationError("; ".join(status["unavailable_reasons"]))

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = project_path(self.project_root, status["checkpoint"])
        base_checkpoints = project_path(
            self.project_root,
            status["base_checkpoints"],
        )
        source_root = project_path(self.project_root, status["source_root"])
        python_value = str(status["python"])
        python = (
            project_path(self.project_root, python_value)
            if not Path(python_value).is_absolute()
            else Path(python_value)
        )
        combined = output_dir / "a2world_combined.mp4"
        images = condition["images"]
        command = [
            str(python),
            "-m",
            "a2world.rollout",
            "--checkpoint",
            str(checkpoint),
            "--variant",
            str(status["variant"]),
            "--input",
            str(images["agentview"]),
            str(images["eye_in_hand"]),
            "--actions",
            str(actions_path),
            "--action-format",
            "precomputed",
            "--view-ids",
            str(self.VIEW_IDS["agentview"]),
            str(self.VIEW_IDS["eye_in_hand"]),
            "--output",
            str(combined),
            "--height",
            str(int(condition["height"])),
            "--width",
            str(int(condition["width"])),
            "--num-sampling-steps",
            str(int(self.config.get("num_sampling_steps", 35))),
            "--guidance",
            str(float(self.config.get("guidance", 0.0))),
            "--seed",
            str(int(self.config.get("seed", 0))),
            "--autoregressive",
        ]
        if (
            status["variant"] == "libero"
            and self.config.get("history", True) is False
        ):
            command.append("--no-history")

        environment = os.environ.copy()
        source_value = str(source_root)
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            source_value
            if not existing_pythonpath
            else source_value + os.pathsep + existing_pythonpath
        )
        environment["COSMOS_PREDICT2_ARGS"] = (
            "--checkpoints " + str(base_checkpoints.resolve())
        )
        if self.config.get("gpu_index") is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(int(self.config["gpu_index"]))
        preflight = subprocess.run(
            [*command, "--preflight"],
            cwd=str(source_root),
            env=environment,
            text=True,
            check=False,
        )
        if preflight.returncode != 0:
            raise RuntimeError(
                f"A2World preflight exited with code {preflight.returncode}"
            )
        completed = subprocess.run(
            command,
            cwd=str(source_root),
            env=environment,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"A2World exited with code {completed.returncode}"
            )
        if not combined.is_file():
            raise ValidationError(
                "A2World completed without writing its combined rollout video"
            )

        return self._split_standardized_outputs(
            combined_path=combined,
            output_dir=output_dir,
            mapping=status["camera_mapping"],
            requested_frames=int(condition["future_action_count"]),
            fps=float(condition["output_fps"]),
        )
