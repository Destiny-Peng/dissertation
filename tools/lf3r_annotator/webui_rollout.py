"""WebUI rollout generation with LIBERO preserved and ManiSkill3 added."""

from __future__ import annotations

import datetime as dt
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from backend_core import ValidationError
from rollout_service import RolloutGenerationService


MANISKILL3_ENVS = (
    "PickCube-v1",
    "StackCube-v1",
    "PegInsertionSide-v1",
    "PlugCharger-v1",
    "PlaceSphere-v1",
    "PushCube-v1",
    "PullCubeTool-v1",
    "LiftPegUpright-v1",
    "PullCube-v1",
    "DrawTriangle-v1",
    "DrawSVG-v1",
    "StackPyramid-v1",
)


class WebUIRolloutGenerationService(RolloutGenerationService):
    """Keep the existing LIBERO generator and add success-only ManiSkill3."""

    def __init__(self, project_root, manifest_path, coordinator, tmux) -> None:
        super().__init__(project_root, manifest_path, coordinator, tmux)
        self.maniskill_python = Path(
            os.environ.get(
                "LF3R_MANISKILL3_PYTHON",
                str(self.project_root / "conda_envs/LF3R-maniskill3/bin/python"),
            )
        ).expanduser()
        self.maniskill_runner = (
            self.project_root / "tools/lf3r_annotator/generate_maniskill3_success.py"
        )
        self.maniskill_output_root = self.project_root / "outputs/maniskill3"
        self.maniskill_manifest_path = (
            self.project_root
            / "datasets"
            / "lf3r_failure_rollouts"
            / "v1"
            / "maniskill3_manifest.jsonl"
        )

    @staticmethod
    def _generator(value: Any) -> str:
        generator = str(value or "libero").strip().lower()
        if generator not in {"libero", "maniskill3"}:
            raise ValidationError("generator must be libero or maniskill3")
        return generator

    @staticmethod
    def _maniskill_env(value: Any) -> str:
        env_id = str(value or "PickCube-v1").strip()
        if env_id not in MANISKILL3_ENVS:
            raise ValidationError(
                "maniskill_env_id must be one of: " + ", ".join(MANISKILL3_ENVS)
            )
        return env_id

    @staticmethod
    def _safe_env_slug(env_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "-", env_id).strip("-").lower()

    def _new_maniskill_run_note(self, env_id: str, label: str) -> str:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = f"-{label}" if label else ""
        base = f"lf3r-maniskill3-{self._safe_env_slug(env_id)}-{stamp}{suffix}"
        candidate = base
        while (self.maniskill_output_root / candidate).exists():
            candidate = f"{base}-{uuid.uuid4().hex[:8]}"
        return candidate

    def _count_maniskill_generated(self, job: dict[str, Any]) -> int:
        run_root = self.project_root / str(job["run_root"])
        env_id = str(job["maniskill_env_id"])
        video_dir = run_root / env_id / "motionplanning"
        if not video_dir.is_dir():
            return 0
        return sum(1 for path in video_dir.glob("*.mp4") if path.is_file())

    def start(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValidationError("Rollout generation request must be a JSON object")
        generator = self._generator(payload.get("generator"))
        if generator == "libero":
            libero_payload = dict(payload)
            libero_payload.pop("generator", None)
            return super().start(libero_payload)
        return self._start_maniskill(payload)

    def _start_maniskill(self, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "generator",
            "gpu",
            "maniskill_env_id",
            "trials",
            "seed",
            "run_label",
            "render_width",
            "render_height",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValidationError(
                "Unknown ManiSkill3 generation field(s): " + ", ".join(sorted(unknown))
            )

        if not self.maniskill_python.is_file():
            raise ValidationError(
                "ManiSkill3 environment is unavailable: " + str(self.maniskill_python)
            )
        if not self.maniskill_runner.is_file():
            raise ValidationError(
                "ManiSkill3 LF3R generator is unavailable: " + str(self.maniskill_runner)
            )

        gpu = self._gpu(payload.get("gpu", "0"))
        env_id = self._maniskill_env(payload.get("maniskill_env_id"))
        trials = self._integer(payload.get("trials", 1), "trials", 1, 50)
        seed = self._integer(payload.get("seed", 0), "seed", 0, 2147483647)
        label = self._run_label(payload.get("run_label", ""))
        render_width = self._resolution(
            payload.get("render_width"), "render_width", 512
        )
        render_height = self._resolution(
            payload.get("render_height"), "render_height", 512
        )

        self.maniskill_output_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        run_note = self._new_maniskill_run_note(env_id, label)
        output_dir = self.maniskill_output_root / run_note
        output_dir.mkdir(parents=False, exist_ok=False)

        job_id = "rollout-" + uuid.uuid4().hex[:12]
        command = [
            str(self.maniskill_python),
            str(self.maniskill_runner),
            "--env-id",
            env_id,
            "--num-traj",
            str(trials),
            "--seed",
            str(seed),
            "--record-dir",
            str(output_dir),
            "--render-width",
            str(render_width),
            "--render-height",
            str(render_height),
        ]
        job = {
            "job_id": job_id,
            "job_type": "rollout_generation",
            "generator": "maniskill3",
            "status": "queued",
            "suite_label": "ManiSkill3",
            "maniskill_env_id": env_id,
            "gpu": gpu,
            "trials": trials,
            "seed": seed,
            "run_label": label,
            "run_note": run_note,
            "requested_rollouts": trials,
            "expected_rollouts": trials,
            "completed_rollouts": 0,
            "render_width": render_width,
            "render_height": render_height,
            "generator_script": self._relative(self.maniskill_runner),
            "python": self._relative(self.maniskill_python),
            "output_root": self._relative(self.maniskill_output_root),
            "run_root": self._relative(output_dir),
            "output_dir": self._relative(output_dir),
            "manifest_path": self._relative(self.maniskill_manifest_path),
            "log_path": self._relative(self.log_root / f"{job_id}.log"),
            "command": command,
            "manifest_rebuilt": False,
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "error": None,
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "interpreter": str(self.maniskill_python),
        }

        with self.jobs_lock:
            self.jobs[job_id] = job
        try:
            cache = self.project_root / "cache"
            self.tmux.submit(
                job,
                command,
                self.log_root / f"{job_id}.log",
                interpreter=str(self.maniskill_python),
                environment={
                    "PYTHONUNBUFFERED": "1",
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "MS_ASSET_DIR": str(self.project_root / "datasets/maniskill3"),
                    "SAPIEN_CACHE_DIR": str(cache / "sapien"),
                    "CUDA_CACHE_PATH": str(cache / "cuda"),
                    "__GL_SHADER_DISK_CACHE_PATH": str(cache / "nvidia"),
                    "XDG_CACHE_HOME": str(cache / "xdg"),
                    "MPLBACKEND": "Agg",
                },
                on_poll=self._on_job_poll,
                on_finished=self._on_job_finished,
            )
        except Exception:
            with self.jobs_lock:
                self.jobs.pop(job_id, None)
            raise
        return dict(job)

    def _on_job_loaded(self, job: dict[str, Any]) -> None:
        if job.get("generator") != "maniskill3":
            super()._on_job_loaded(job)
            return
        with self.jobs_lock:
            self.jobs[str(job["job_id"])] = job

    def _on_job_poll(self, job: dict[str, Any]) -> None:
        if job.get("generator") != "maniskill3":
            super()._on_job_poll(job)
            return
        with self.jobs_lock:
            job["completed_rollouts"] = self._count_maniskill_generated(job)

    def _on_job_finished(
        self,
        job: dict[str, Any],
        return_code: int | None,
        reason: str | None,
    ) -> None:
        if job.get("generator") != "maniskill3":
            super()._on_job_finished(job, return_code, reason)
            return

        completed = self._count_maniskill_generated(job)
        manifest_ready = (
            self.maniskill_manifest_path.is_file()
            and self.maniskill_manifest_path.stat().st_size > 0
        )
        error = reason
        with self.jobs_lock:
            job["completed_rollouts"] = completed
            job["manifest_rebuilt"] = bool(
                return_code == 0
                and completed >= int(job.get("expected_rollouts", 0))
                and manifest_ready
            )
            if error:
                job["status"] = "failed"
            elif return_code == 0 and completed >= int(job.get("expected_rollouts", 0)):
                if manifest_ready:
                    job["status"] = "complete"
                else:
                    job["status"] = "failed"
                    error = "ManiSkill3 rollouts completed but its manifest was not rebuilt"
            elif return_code == 0:
                job["status"] = "failed"
                error = (
                    "ManiSkill3 exited successfully but produced fewer successful videos "
                    "than requested"
                )
            else:
                job["status"] = "failed"
                error = f"ManiSkill3 generator exited with code {return_code}"
            job["error"] = error
            job["return_code"] = return_code
            job["finished_at"] = job.get("finished_at") or dt.datetime.now(
                dt.timezone.utc
            ).isoformat()
