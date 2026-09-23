"""Rollout-generation service for the LF3R WebUI."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from backend_core import JobCoordinator, ValidationError
from task_supervisor import TmuxJobSupervisor


GENERATION_CONFIGS = {
    "libero_10": {
        "label": "LIBERO-10",
        "script_name": "generate_libero10_natural.sh",
        "output_root": "outputs/openvla_libero",
        "run_prefix": "lf3r-data-natural-libero10-",
        "max_task": 9,
        "render_resolution": 256,
        "policy_resolution": 224,
        "record_resolution": 224,
    },
    "libero_spatial": {
        "label": "LIBERO-Spatial",
        "script_name": "generate_libero_spatial_native.sh",
        "output_root": "outputs/openvla_libero_spatial_native",
        "run_prefix": "lf3r-data-natural-libero-spatial-256-",
        "max_task": 9,
        "render_resolution": 256,
        "policy_resolution": 224,
        "record_resolution": 256,
    },
}
GENERATION_RUN_PREFIX = GENERATION_CONFIGS["libero_10"]["run_prefix"]
GENERATION_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
GENERATION_VIDEO_RE = re.compile(r"^task[0-9]+--ep[0-9]+--succ[01]\.mp4$")


class RolloutGenerationService:
    """Run an existing natural OpenVLA/LIBERO generator from the web UI."""

    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        coordinator: JobCoordinator,
        tmux: TmuxJobSupervisor,
    ) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        script_root = self.project_root / "tools" / "lf3r_annotator"
        self.scripts = {
            suite: script_root / str(config["script_name"])
            for suite, config in GENERATION_CONFIGS.items()
        }
        # Preserve these attributes for existing callers and test adapters.
        self.script = self.scripts["libero_10"]
        self.output_roots = {
            suite: self.project_root / str(config["output_root"])
            for suite, config in GENERATION_CONFIGS.items()
        }
        self.output_root = self.output_roots["libero_10"]
        self.log_root = self.project_root / "logs" / "rollouts" / "web_runs"
        self.jobs: dict[str, dict[str, Any]] = {}
        self.jobs_lock = threading.Lock()
        self.coordinator = coordinator
        self.tmux = tmux
        self.tmux.register_handler(
            "rollout_generation",
            self._on_job_loaded,
            self._on_job_poll,
            self._on_job_finished,
        )

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(path)

    @staticmethod
    def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
        if isinstance(value, bool):
            raise ValidationError(f"{name} must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValidationError(f"{name} must be an integer")
        try:
            number = int(value)
        except (TypeError, ValueError) as error:
            raise ValidationError(f"{name} must be an integer") from error
        if number < minimum or number > maximum:
            raise ValidationError(f"{name} must be between {minimum} and {maximum}")
        return number

    @staticmethod
    def _gpu(value: Any) -> str:
        gpu = str(value if value is not None else "0").strip()
        if not re.fullmatch(r"[0-9]+", gpu):
            raise ValidationError("gpu must be one numeric CUDA device index")
        return gpu

    @classmethod
    def _resolution(cls, value: Any, name: str, default: int) -> int:
        if value is None or value == "":
            return int(default)
        resolution = cls._integer(value, name, 64, 2048)
        if resolution % 2:
            raise ValidationError(f"{name} must be an even integer between 64 and 2048")
        return resolution

    @staticmethod
    def _task_suite(value: Any) -> str:
        suite = str(value or "libero_10").strip()
        if suite not in GENERATION_CONFIGS:
            raise ValidationError(
                "task_suite must be one of: " + ", ".join(GENERATION_CONFIGS)
            )
        return suite

    @staticmethod
    def _video_view_mode(value: Any) -> str:
        mode = str(value or "single_view").strip()
        if mode not in {"single_view", "libero_three_view"}:
            raise ValidationError(
                "video_view_mode must be single_view or libero_three_view"
            )
        return mode

    @staticmethod
    def _run_label(value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValidationError("run_label must be a string")
        label = value.strip()
        if label and not GENERATION_LABEL_RE.fullmatch(label):
            raise ValidationError(
                "run_label must contain only letters, numbers, dot, underscore, or hyphen"
            )
        return label

    def _output_root(self, task_suite: str) -> Path:
        # Keep the historical output_root override working for LIBERO-10 callers.
        return self.output_root if task_suite == "libero_10" else self.output_roots[task_suite]

    def _new_run_note(self, label: str, task_suite: str = "libero_10") -> str:
        config = GENERATION_CONFIGS[task_suite]
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = f"-{label}" if label else ""
        candidate = f"{config['run_prefix']}{stamp}{suffix}"
        output_dir = self._output_root(task_suite) / candidate
        if output_dir.exists():
            candidate = f"{candidate}-{uuid.uuid4().hex[:8]}"
        return candidate

    def _count_generated(self, run_note: str, task_suite: str = "libero_10") -> int:
        directory = self._output_root(task_suite) / run_note / task_suite
        if not directory.is_dir():
            return 0
        return sum(
            1 for path in directory.glob("*.mp4") if GENERATION_VIDEO_RE.fullmatch(path.name)
        )

    def _manifest_signature(self) -> str | None:
        if not self.manifest_path.is_file():
            return None
        digest = hashlib.sha256()
        with self.manifest_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _job_snapshot(self, job_id: str) -> dict[str, Any]:
        with self.jobs_lock:
            return dict(self.jobs[job_id])

    def start(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValidationError("Rollout generation request must be a JSON object")
        allowed = {
            "task_suite", "gpu", "task_start", "task_end", "trials", "seed",
            "run_label", "log_safe_features", "render_resolution", "record_resolution",
            "video_view_mode",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValidationError("Unknown rollout-generation field(s): " + ", ".join(sorted(unknown)))

        log_safe_features = payload.get("log_safe_features", True)
        if not isinstance(log_safe_features, bool):
            raise ValidationError("log_safe_features must be boolean")

        task_suite = self._task_suite(payload.get("task_suite", "libero_10"))
        config = GENERATION_CONFIGS[task_suite]
        gpu = self._gpu(payload.get("gpu", "0"))
        render_resolution = self._resolution(
            payload.get("render_resolution"), "render_resolution", int(config["render_resolution"])
        )
        record_resolution = self._resolution(
            payload.get("record_resolution"), "record_resolution", int(config["record_resolution"])
        )
        video_view_mode = self._video_view_mode(payload.get("video_view_mode"))
        max_task = int(config["max_task"])
        task_start = self._integer(payload.get("task_start", 0), "task_start", 0, max_task)
        task_end = self._integer(payload.get("task_end", 3), "task_end", 0, max_task)
        if task_start > task_end:
            raise ValidationError("task_start must not be greater than task_end")
        trials = self._integer(payload.get("trials", 1), "trials", 1, 50)
        seed = self._integer(payload.get("seed", 7), "seed", 0, 2147483647)
        label = self._run_label(payload.get("run_label", ""))
        script = self.script if task_suite == "libero_10" else self.scripts[task_suite]
        if not script.is_file():
            raise ValidationError(f"{config['label']} natural rollout generator is not installed")

        output_root = self._output_root(task_suite)
        output_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        run_note = self._new_run_note(label, task_suite)
        output_dir = output_root / run_note
        if output_dir.exists():
            raise ValidationError("Generated run directory already exists; retry with a new label")

        job_id = "rollout-" + uuid.uuid4().hex[:12]
        command = [
            "bash",
            str(script),
            gpu,
            str(task_start),
            str(task_end),
            str(trials),
            str(seed),
            run_note,
        ]
        if log_safe_features:
            command.append("--log-safe-features")
        command.extend(["--video-view-mode", video_view_mode])
        command.extend(
            [
                "--render-resolution",
                str(render_resolution),
                "--record-resolution",
                str(record_resolution),
            ]
        )
        expected = (task_end - task_start + 1) * trials
        job = {
            "job_id": job_id,
            "job_type": "rollout_generation",
            "status": "queued",
            "task_suite": task_suite,
            "suite_label": config["label"],
            "gpu": gpu,
            "task_start": task_start,
            "task_end": task_end,
            "trials": trials,
            "seed": seed,
            "run_label": label,
            "log_safe_features": log_safe_features,
            "run_note": run_note,
            "requested_rollouts": expected,
            "expected_rollouts": expected,
            "completed_rollouts": 0,
            "render_resolution": render_resolution,
            "policy_resolution": config["policy_resolution"],
            "record_resolution": record_resolution,
            "video_view_mode": video_view_mode,
            "camera_video_views": (
                ["cam_high", "cam_wrist"]
                if video_view_mode == "libero_three_view"
                else None
            ),
            "generator_script": self._relative(script),
            "output_root": self._relative(output_root),
            "run_root": self._relative(output_dir),
            "output_dir": self._relative(output_dir),
            "log_path": self._relative(self.log_root / f"{job_id}.log"),
            "command": command,
            "memory_gate": {
                "kind": "memory_only",
                "minimum_free_mib": 30720,
                "maximum_used_fraction": 0.50,
                "gpu_utilization": "informational",
            },
            "manifest_rebuilt": False,
            "manifest_before": self._manifest_signature(),
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "error": None,
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "interpreter": "bash",
        }
        self.coordinator.acquire(job_id, "manifest_writer")
        try:
            with self.jobs_lock:
                self.jobs[job_id] = job
            self.tmux.submit(
                job,
                command,
                self.log_root / f"{job_id}.log",
                interpreter="bash",
                environment={
                    "MPLBACKEND": "Agg",
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                },
                on_poll=self._on_job_poll,
                on_finished=self._on_job_finished,
            )
        except Exception:
            with self.jobs_lock:
                self.jobs.pop(job_id, None)
            self.coordinator.release(job_id)
            raise
        return dict(job)

    def _on_job_loaded(self, job: dict[str, Any]) -> None:
        job_id = str(job["job_id"])
        with self.jobs_lock:
            self.jobs[job_id] = job
        if job.get("status") in {"queued", "running"}:
            self.coordinator.restore(job_id, "manifest_writer")

    def _on_job_poll(self, job: dict[str, Any]) -> None:
        with self.jobs_lock:
            task_suite = self._task_suite(job.get("task_suite", "libero_10"))
            job["completed_rollouts"] = self._count_generated(str(job["run_note"]), task_suite)

    def _on_job_finished(
        self,
        job: dict[str, Any],
        return_code: int | None,
        reason: str | None,
    ) -> None:
        job_id = str(job["job_id"])
        error = reason
        try:
            task_suite = self._task_suite(job.get("task_suite", "libero_10"))
            completed = self._count_generated(str(job["run_note"]), task_suite)
            manifest_after = self._manifest_signature()
            with self.jobs_lock:
                job["completed_rollouts"] = completed
                job["manifest_after"] = manifest_after
                job["manifest_rebuilt"] = (
                    return_code == 0
                    and manifest_after is not None
                    and manifest_after != job.get("manifest_before")
                )
                if error:
                    job["status"] = "failed"
                elif return_code == 75:
                    job["status"] = "memory_blocked"
                    error = "Generator refused to start because the memory-only GPU gate did not pass"
                elif return_code == 0:
                    job["status"] = "complete"
                else:
                    job["status"] = "failed"
                    error = f"Rollout generator exited with code {return_code}"
                job["error"] = error
                job["return_code"] = return_code
                job["finished_at"] = job.get("finished_at") or dt.datetime.now(dt.timezone.utc).isoformat()
        finally:
            self.coordinator.release(job_id)

    def job(self, job_id: str) -> dict[str, Any]:
        with self.jobs_lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return dict(self.jobs[job_id])

    def list_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        return self.tmux.list("rollout_generation", status)

    def log(self, job_id: str, tail: Any = 200) -> dict[str, Any]:
        job = self.job(job_id)
        try:
            count = self._integer(tail, "tail", 1, 2000)
        except ValidationError:
            count = 200
        path = self.project_root / job["log_path"]
        if not path.is_file():
            return {"job_id": job_id, "lines": [], "text": ""}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]
        return {"job_id": job_id, "lines": lines, "text": "\n".join(lines)}
