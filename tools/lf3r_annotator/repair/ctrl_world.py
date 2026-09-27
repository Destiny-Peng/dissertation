"""Ctrl-World adapter for LF3R Repair / Synthetic Suffix."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from backend_core import ValidationError
from .adapters import WorldModelAdapter
from .alignment import find_trajectory_path, project_path
from .trajectory import load_ctrl_world_pose_states


class CtrlWorldAdapter(WorldModelAdapter):
    """Adapt LF3R LIBERO demonstrations to the released Ctrl-World interface.

    Ctrl-World's released replay path is trained on DROID and conditions on
    sparse history plus a future 7D Cartesian pose/gripper trajectory.  The
    model expects exactly three visual streams.  LF3R keeps its manifest honest
    (cam_high + cam_wrist); the required second exterior stream is duplicated
    only inside this adapter and the duplication is recorded in provenance.
    """

    name = "ctrl_world"
    CHECKPOINT_TYPE = "droid_pretrained"
    VIEW_ORDER = ("exterior_1", "exterior_2", "wrist")
    DEFAULT_CAMERA_MAPPING = {
        "exterior_1": "cam_high",
        "exterior_2": "cam_high",
        "wrist": "cam_wrist",
    }
    TARGET_FPS = 5.0
    WIDTH = 320
    HEIGHT = 192
    NUM_FRAMES = 5
    NUM_HISTORY = 6

    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        self.project_root = project_root.resolve()
        self.config = dict(config)
        self._validation: dict[str, Any] | None = None

    def _path_candidates(
        self,
        key: str,
        env_key: str,
        defaults: tuple[str, ...],
    ) -> Path:
        configured = str(
            self.config.get(key)
            or os.environ.get(env_key)
            or ""
        ).strip()
        if configured:
            return project_path(self.project_root, configured)
        candidates = [
            project_path(self.project_root, value)
            for value in defaults
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    def _python(self) -> Path:
        configured = str(
            self.config.get("python")
            or os.environ.get("LF3R_CTRL_WORLD_PYTHON")
            or ""
        ).strip()
        configured_env = str(
            self.config.get("environment")
            or os.environ.get("LF3R_ENV_CTRL_WORLD")
            or ""
        ).strip()
        candidates: list[Path] = []
        if configured:
            candidates.append(project_path(self.project_root, configured))
        if configured_env:
            candidates.append(
                project_path(self.project_root, configured_env) / "bin" / "python"
            )
        candidates.extend(
            [
                self.project_root / "conda_envs" / "LF3R-ctrl-world" / "bin" / "python",
                self.project_root / "conda_envs" / "LF3R-Ctrl-World" / "bin" / "python",
                self.project_root / "conda_envs" / "Ctrl-World" / "bin" / "python",
                self.project_root / "conda_envs" / "ctrl-world" / "bin" / "python",
                self.project_root / "conda_envs" / "LF3R-ctrl" / "bin" / "python",
                self.project_root / "repos" / "Ctrl-World" / ".venv" / "bin" / "python",
            ]
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return candidates[0].resolve()

    def _checkpoint(self) -> Path:
        raw = str(self.config.get("checkpoint") or "").strip()
        if raw:
            return project_path(self.project_root, raw)
        candidates = [
            self.project_root
            / "checkpoints"
            / "ctrl_world"
            / "Ctrl-World"
            / "checkpoint-10000.pt",
            self.project_root / "checkpoints" / "Ctrl-World" / "checkpoint-10000.pt",
            self.project_root / "checkpoints" / "ctrl-world" / "checkpoint-10000.pt",
            self.project_root / "repos" / "Ctrl-World" / "checkpoint-10000.pt",
            self.project_root / "checkpoints" / "ctrl-world.pt",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return candidates[0].resolve()

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
        if set(mapping) != set(self.VIEW_ORDER):
            raise ValidationError(
                "Ctrl-World camera_mapping must define exterior_1, exterior_2, and wrist"
            )
        for consumer_view, manifest_view in mapping.items():
            if manifest_view not in cameras:
                raise ValidationError(
                    f"Ctrl-World view {consumer_view} references unavailable manifest "
                    f"camera {manifest_view}"
                )

        duplicated: list[dict[str, str]] = []
        seen: dict[str, str] = {}
        for consumer_view in self.VIEW_ORDER:
            manifest_view = mapping[consumer_view]
            if manifest_view in seen:
                duplicated.append(
                    {
                        "consumer_view": consumer_view,
                        "manifest_view": manifest_view,
                        "duplicates_consumer_view": seen[manifest_view],
                        "reason": "Ctrl-World checkpoint requires three views; LF3R LIBERO has two physical views",
                    }
                )
            else:
                seen[manifest_view] = consumer_view
        return mapping, duplicated

    @staticmethod
    def _relative_or_absolute(project_root: Path, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(project_root))
        except ValueError:
            return str(path.resolve())

    def validate_rollout(self, rollout: dict[str, Any]) -> dict[str, Any]:
        mapping, duplicated = self._camera_mapping(rollout)
        source_root = self._path_candidates(
            "source_root",
            "LF3R_CTRL_WORLD_SOURCE",
            ("repos/Ctrl-World", "repos/ctrl-world", "repos/ctrl_world"),
        )
        checkpoint = self._checkpoint()
        svd_path = self._path_candidates(
            "svd_model_path",
            "LF3R_CTRL_WORLD_SVD",
            (
                "checkpoints/stable-video-diffusion-img2vid",
                "checkpoints/stabilityai/stable-video-diffusion-img2vid",
                "repos/Ctrl-World/checkpoints/stable-video-diffusion-img2vid",
            ),
        )
        clip_path = self._path_candidates(
            "clip_model_path",
            "LF3R_CTRL_WORLD_CLIP",
            (
                "checkpoints/clip-vit-base-patch32",
                "checkpoints/openai/clip-vit-base-patch32",
                "repos/Ctrl-World/checkpoints/clip-vit-base-patch32",
            ),
        )
        data_stat_path = self._path_candidates(
            "data_stat_path",
            "LF3R_CTRL_WORLD_DATA_STAT",
            (
                "repos/Ctrl-World/dataset_meta_info/droid/stat.json",
                "repos/ctrl-world/dataset_meta_info/droid/stat.json",
            ),
        )
        python = self._python()
        trajectory = find_trajectory_path(self.project_root, rollout)

        try:
            target_fps = float(self.config.get("target_fps", self.TARGET_FPS))
            inference_steps = int(self.config.get("num_inference_steps", 50))
            guidance_scale = float(self.config.get("guidance_scale", 1.0))
            seed = int(self.config.get("seed", 0))
        except (TypeError, ValueError) as error:
            raise ValidationError(
                "Ctrl-World target_fps, num_inference_steps, guidance_scale, and seed must be numeric"
            ) from error
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValidationError("Ctrl-World target_fps must be positive")
        if inference_steps < 1:
            raise ValidationError("Ctrl-World num_inference_steps must be at least 1")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValidationError("Ctrl-World guidance_scale must be finite and non-negative")

        reasons: list[str] = []
        if not source_root.is_dir() or not (source_root / "models" / "ctrl_world.py").is_file():
            reasons.append(
                "Ctrl-World source is unavailable: "
                + self._relative_or_absolute(self.project_root, source_root)
            )
        if not python.is_file():
            reasons.append(
                "Ctrl-World Python interpreter is unavailable: "
                + self._relative_or_absolute(self.project_root, python)
            )
        if not checkpoint.is_file():
            reasons.append(
                "Ctrl-World checkpoint is unavailable: "
                + self._relative_or_absolute(self.project_root, checkpoint)
            )
        if not svd_path.is_dir():
            reasons.append(
                "Ctrl-World SVD base model is unavailable: "
                + self._relative_or_absolute(self.project_root, svd_path)
            )
        if not clip_path.is_dir():
            reasons.append(
                "Ctrl-World CLIP model is unavailable: "
                + self._relative_or_absolute(self.project_root, clip_path)
            )
        if not data_stat_path.is_file():
            reasons.append(
                "Ctrl-World DROID normalization stats are unavailable: "
                + self._relative_or_absolute(self.project_root, data_stat_path)
            )
        if trajectory is None or trajectory.suffix.lower() not in {".hdf5", ".h5"}:
            reasons.append(
                "Ctrl-World requires the official LIBERO HDF5 trajectory/proprio source"
            )

        source_fps = float(rollout.get("fps") or 0.0)
        frame_step = (
            max(1, int(round(source_fps / target_fps)))
            if source_fps > 0
            else None
        )
        effective_fps = (
            source_fps / frame_step
            if source_fps > 0 and frame_step
            else None
        )

        result = {
            "adapter": self.name,
            "available": not reasons,
            "unavailable_reasons": reasons,
            "checkpoint": self._relative_or_absolute(self.project_root, checkpoint),
            "checkpoint_type": self.CHECKPOINT_TYPE,
            "source_root": self._relative_or_absolute(self.project_root, source_root),
            "python": self._relative_or_absolute(self.project_root, python),
            "svd_model_path": self._relative_or_absolute(self.project_root, svd_path),
            "clip_model_path": self._relative_or_absolute(self.project_root, clip_path),
            "data_stat_path": self._relative_or_absolute(self.project_root, data_stat_path),
            "camera_mapping": mapping,
            "duplicated_camera": duplicated,
            "view_order": list(self.VIEW_ORDER),
            "target_fps": target_fps,
            "source_frame_step": frame_step,
            "effective_fps": effective_fps,
            "num_frames": self.NUM_FRAMES,
            "num_history": self.NUM_HISTORY,
            "num_inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "text_conditioning": bool(self.config.get("text_conditioning", True)),
            "control_adapter": (
                "official LIBERO obs ee_pos + ee_ori(axis-angle->Euler XYZ) + "
                "Panda gripper qpos->DROID closure scalar"
            ),
            "control_semantics": (
                "future absolute Cartesian pose/gripper state trajectory; "
                "not raw LIBERO delta actions"
            ),
            "rollout_mode": "autoregressive",
        }
        self._validation = result
        return result

    @staticmethod
    def _read_frame(path: Path, index: int) -> Any:
        try:
            import imageio.v2 as imageio
        except ImportError as error:
            raise ValidationError(
                "imageio is required to prepare Ctrl-World condition frames"
            ) from error
        reader = imageio.get_reader(str(path))
        try:
            return reader.get_data(index)
        finally:
            reader.close()

    @staticmethod
    def _resize_frame(frame: Any, width: int, height: int) -> Any:
        try:
            from PIL import Image
            import numpy as np
        except ImportError as error:
            raise ValidationError(
                "Pillow and numpy are required to prepare Ctrl-World images"
            ) from error
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        return np.asarray(
            image.resize((int(width), int(height)), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )

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
                "imageio is required to prepare Ctrl-World condition frames"
            ) from error

        status = self._validation or self.validate_rollout(rollout)
        mapping = status["camera_mapping"]
        camera_paths = rollout["camera_video_paths"]
        condition_dir = output_dir / "ctrl_world_condition"
        condition_dir.mkdir(parents=True, exist_ok=True)
        images: dict[str, Path] = {}
        for consumer_view in self.VIEW_ORDER:
            manifest_view = mapping[consumer_view]
            source = project_path(
                self.project_root,
                str(camera_paths[manifest_view]),
            )
            frame = self._read_frame(source, cut_frame)
            frame = self._resize_frame(frame, self.WIDTH, self.HEIGHT)
            target = condition_dir / f"{consumer_view}.png"
            imageio.imwrite(str(target), frame)
            images[consumer_view] = target

        source_shapes: dict[str, list[int]] = {}
        for consumer_view in self.VIEW_ORDER:
            manifest_view = mapping[consumer_view]
            source = project_path(
                self.project_root,
                str(camera_paths[manifest_view]),
            )
            raw_frame = self._read_frame(source, cut_frame)
            source_shapes[consumer_view] = [
                int(raw_frame.shape[0]),
                int(raw_frame.shape[1]),
            ]

        metadata = {
            "cut_frame": int(cut_frame),
            "camera_mapping": mapping,
            "duplicated_camera": status["duplicated_camera"],
            "condition_images": {
                key: str(path.relative_to(self.project_root))
                for key, path in images.items()
            },
            "width": self.WIDTH,
            "height": self.HEIGHT,
            "source_view_shapes": source_shapes,
            "resize_policy": "source RGB -> Ctrl-World native 192x320 bilinear",
        }
        (condition_dir / "condition.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {"images": images, **metadata}

    def prepare_actions(self, actions: Any, *, output_dir: Path) -> Path:
        del actions, output_dir
        raise ValidationError(
            "Ctrl-World uses model-specific absolute pose controls; call prepare_controls"
        )

    def prepare_controls(
        self,
        rollout: dict[str, Any],
        *,
        cut_frame: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        try:
            import numpy as np
        except ImportError as error:
            raise ValidationError(
                "numpy is required to prepare Ctrl-World controls"
            ) from error

        status = self._validation or self.validate_rollout(rollout)
        source_fps = float(rollout.get("fps") or 0.0)
        frame_step = status.get("source_frame_step")
        if source_fps <= 0 or not frame_step:
            raise ValidationError("Ctrl-World requires a positive source rollout FPS")
        poses = load_ctrl_world_pose_states(self.project_root, rollout)
        total = min(int(rollout.get("total_frames") or len(poses)), len(poses))
        if cut_frame < 0 or cut_frame >= total - 1:
            raise ValidationError("Ctrl-World cut leaves no future pose samples")
        indices = np.arange(cut_frame, total, int(frame_step), dtype=np.int64)
        if len(indices) < 2:
            raise ValidationError(
                "Ctrl-World temporal sampling leaves no generated suffix frame"
            )
        controls = np.asarray(poses[indices], dtype=np.float32)

        path = output_dir / "ctrl_world_controls.npz"
        np.savez_compressed(
            path,
            controls=controls,
            source_frame_indices=indices,
        )
        metadata = {
            "path": str(path.relative_to(self.project_root)),
            "source_fps": source_fps,
            "requested_target_fps": status["target_fps"],
            "source_frame_step": int(frame_step),
            "effective_fps": float(source_fps / int(frame_step)),
            "control_points": int(len(controls)),
            "control_dim": 7,
            "source_frame_indices": [int(x) for x in indices.tolist()],
            "generated_real_frame_indices": [
                int(x) for x in indices[1:].tolist()
            ],
            "generated_real_start_frame": int(indices[1]),
            "control_semantics": status["control_semantics"],
            "control_adapter": status["control_adapter"],
        }
        (output_dir / "ctrl_world_controls.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {"path": path, **metadata}

    def generate(
        self,
        *,
        rollout: dict[str, Any],
        condition: dict[str, Any],
        controls: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Path]:
        status = self._validation
        if status is None:
            raise ValidationError(
                "Ctrl-World adapter must validate the rollout before generation"
            )
        if not status["available"]:
            raise ValidationError("; ".join(status["unavailable_reasons"]))

        output_dir.mkdir(parents=True, exist_ok=True)
        runner = Path(__file__).resolve().parent / "ctrl_world_runner.py"
        python = project_path(self.project_root, status["python"])
        source_root = project_path(self.project_root, status["source_root"])
        checkpoint = project_path(self.project_root, status["checkpoint"])
        svd_path = project_path(self.project_root, status["svd_model_path"])
        clip_path = project_path(self.project_root, status["clip_model_path"])
        data_stat_path = project_path(self.project_root, status["data_stat_path"])
        images = condition["images"]

        command = [
            str(python),
            str(runner),
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--svd-model-path",
            str(svd_path),
            "--clip-model-path",
            str(clip_path),
            "--data-stat-path",
            str(data_stat_path),
            "--controls",
            str(controls["path"]),
            "--exterior-1",
            str(images["exterior_1"]),
            "--exterior-2",
            str(images["exterior_2"]),
            "--wrist",
            str(images["wrist"]),
            "--instruction",
            str(rollout.get("task_description") or ""),
            "--output-dir",
            str(output_dir),
            "--output-fps",
            str(float(controls["effective_fps"])),
            "--num-inference-steps",
            str(int(status["num_inference_steps"])),
            "--guidance-scale",
            str(float(status["guidance_scale"])),
            "--seed",
            str(int(status["seed"])),
        ]
        if not status["text_conditioning"]:
            command.append("--no-text-conditioning")

        environment = os.environ.copy()
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(source_root)
            if not existing
            else str(source_root) + os.pathsep + existing
        )
        if self.config.get("gpu_index") is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(int(self.config["gpu_index"]))
        completed = subprocess.run(
            command,
            cwd=str(source_root),
            env=environment,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Ctrl-World runner exited with code {completed.returncode}"
            )

        generated = {
            "cam_high": output_dir / "cam_high.mp4",
            "cam_wrist": output_dir / "cam_wrist.mp4",
        }
        missing = [str(path) for path in generated.values() if not path.is_file()]
        if missing:
            raise ValidationError(
                "Ctrl-World completed without standardized outputs: "
                + ", ".join(missing)
            )
        return generated

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
