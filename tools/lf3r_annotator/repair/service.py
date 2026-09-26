"""Repair service and artifact catalog for the LF3R WebUI."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

from backend_core import JobCoordinator, ValidationError
from non_analysis_tools import gpu_status
from task_supervisor import TmuxJobSupervisor
from .adapters import A2WorldAdapter
from .alignment import capability_summary, compute_alignment
from .alignment_runner import run_alignment_subprocess


RUN_ID_RE = re.compile(r"^repair-suffix-[A-Za-z0-9._-]{1,120}$")
HUMAN_USABILITY = {"yes", "no", "uncertain"}
FAILURE_REASONS = {
    "robot_motion",
    "gripper",
    "object_motion",
    "contact",
    "geometry",
    "visual_corruption",
    "temporal_drift",
}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


class RepairService:
    """Synthetic-suffix experiment service.

    This is intentionally independent from failure-localization services.  It
    consumes the same manifest catalog but owns its own API-facing artifacts and
    persistent job type.
    """

    JOB_TYPE = "repair_synthetic_suffix"

    def __init__(
        self,
        project_root: Path,
        coordinator: JobCoordinator,
        tmux: TmuxJobSupervisor,
    ) -> None:
        self.project_root = project_root.resolve()
        self.coordinator = coordinator
        self.tmux = tmux
        self.root = (
            self.project_root
            / "artifacts"
            / "repair"
            / "synthetic_suffix"
            / "runs"
        )
        self.log_root = self.project_root / "logs" / "repair" / "synthetic_suffix"
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.tmux.register_handler(
            self.JOB_TYPE,
            self._on_job_loaded,
            self._on_job_poll,
            self._on_job_finished,
        )

    def _relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.project_root))

    def _run_dir(self, run_id: str) -> Path:
        if not RUN_ID_RE.fullmatch(str(run_id)):
            raise ValidationError("Invalid Repair run id")
        path = (self.root / run_id).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValidationError("Repair run path escapes artifact root") from error
        return path

    @staticmethod
    def _read_json(path: Path, default: Any = None) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    def _new_run_id(self, rollout_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", rollout_id).strip("-")[:48] or "rollout"
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"repair-suffix-{stamp}-{safe}-{uuid.uuid4().hex[:6]}"

    def _existing_runs_by_rollout(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for run in self.list_runs():
            rollout_id = str(run.get("source_rollout") or "")
            if rollout_id:
                grouped.setdefault(rollout_id, []).append(run)
        return grouped

    def catalog(self, rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        existing = self._existing_runs_by_rollout()
        rows: list[dict[str, Any]] = []
        for rollout in rollouts:
            if str(rollout.get("ground_truth_outcome") or "").lower() != "success":
                continue
            capabilities = capability_summary(self.project_root, rollout)
            total_frames = int(rollout.get("total_frames") or 0)
            plan = None
            plan_error = None
            try:
                plan = compute_alignment(
                    total_frames=total_frames,
                    cut_type="progress",
                    cut_progress=0.5,
                ).as_dict()
            except ValidationError as error:
                plan_error = str(error)
            rows.append(
                {
                    "id": rollout["id"],
                    "manifest_source": rollout.get("manifest_source"),
                    "manifest_label": rollout.get("manifest_label"),
                    "task_suite": rollout.get("task_suite"),
                    "task_id": rollout.get("task_id"),
                    "episode_index": rollout.get("episode_index"),
                    "task_description": rollout.get("task_description"),
                    "outcome": rollout.get("ground_truth_outcome"),
                    "frames": total_frames,
                    "fps": rollout.get("fps"),
                    **capabilities,
                    "default_alignment": plan,
                    "alignment_error": plan_error,
                    "wm_runs": [
                        {
                            "run_id": item["run_id"],
                            "status": item.get("status"),
                            "world_model": item.get("world_model"),
                            "checkpoint_type": item.get("checkpoint_type"),
                        }
                        for item in existing.get(str(rollout["id"]), [])
                    ],
                }
            )
        return rows

    def _worker_python(self) -> Path:
        configured = str(os.environ.get("LF3R_ENV_OPENVLA") or "").strip()
        candidates = []
        if configured:
            candidates.append(Path(configured) / "bin" / "python")
        candidates.append(
            self.project_root / "conda_envs" / "LF3R-openvla" / "bin" / "python"
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return candidates[-1].resolve()

    @staticmethod
    def _gpu_plan() -> dict[str, Any]:
        status = gpu_status()
        gpus = status.get("gpus") if isinstance(status, dict) else []
        eligible = [
            gpu
            for gpu in (gpus or [])
            if gpu.get("gpu_utilization_percent") is not None
            and float(gpu["gpu_utilization_percent"]) < 50.0
        ]
        eligible.sort(
            key=lambda gpu: (
                float(gpu.get("gpu_utilization_percent") or 0.0),
                -float(gpu.get("memory_free_mib") or 0.0),
                int(gpu.get("index") or 0),
            )
        )
        return {
            "available": bool(status.get("available")) if isinstance(status, dict) else False,
            "error": status.get("error") if isinstance(status, dict) else "GPU status unavailable",
            "threshold_percent": 50.0,
            "eligible": eligible,
            "selected": eligible[0] if eligible else None,
            "queried_at": status.get("queried_at") if isinstance(status, dict) else None,
        }

    def validate_plan(
        self,
        payload: dict[str, Any],
        rollout_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValidationError("Repair request must be a JSON object")
        rollout_id = str(payload.get("rollout_id") or "").strip()
        rollout = rollout_map.get(rollout_id)
        if rollout is None:
            raise ValidationError("Unknown rollout")
        if str(rollout.get("ground_truth_outcome") or "").lower() != "success":
            raise ValidationError("Synthetic Suffix Phase 1 accepts success rollouts only")

        capabilities = capability_summary(self.project_root, rollout)
        alignment = compute_alignment(
            total_frames=int(rollout.get("total_frames") or 0),
            cut_type=str(payload.get("cut_type") or "progress"),
            cut_progress=payload.get("cut_progress", 0.5),
            cut_frame=payload.get("cut_frame"),
        )
        wm_config = payload.get("world_model") or {}
        if not isinstance(wm_config, dict):
            raise ValidationError("world_model must be an object")
        model_name = str(wm_config.get("name") or "a2world").lower()
        if model_name != "a2world":
            raise ValidationError("Phase 1 supports A2World only")
        adapter = A2WorldAdapter(self.project_root, wm_config)
        adapter_status = adapter.validate_rollout(rollout)
        gpu = self._gpu_plan()
        blockers: list[str] = []
        if not capabilities["actions_available"]:
            blockers.append("GT actions are unavailable in the selected manifest record")
        if not capabilities["sim_state_available"]:
            blockers.append("simulator state trajectory is unavailable in the selected manifest record")
        blockers.extend(adapter_status["unavailable_reasons"])
        worker_python = self._worker_python()
        if not worker_python.is_file():
            blockers.append(
                "Repair worker runtime is unavailable: "
                + str(worker_python.relative_to(self.project_root))
            )
        if gpu["selected"] is None:
            if gpu["available"]:
                blockers.append("No GPU is below the 50% utilization threshold")
            else:
                blockers.append(
                    "GPU status is unavailable: " + str(gpu.get("error") or "unknown error")
                )
        return {
            "ready": not blockers,
            "rollout_id": rollout_id,
            "capabilities": capabilities,
            "alignment": alignment.as_dict(),
            "world_model": adapter_status,
            "worker_python": (
                str(worker_python.relative_to(self.project_root))
                if worker_python.is_relative_to(self.project_root)
                else str(worker_python)
            ),
            "gpu": gpu,
            "blockers": blockers,
            "validation": {
                "metadata": "complete",
                "alignment_smoke_test": "will_run_before_generation",
            },
        }

    def validate_with_alignment(
        self,
        payload: dict[str, Any],
        rollout_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        plan = self.validate_plan(payload, rollout_map)
        rollout = rollout_map[plan["rollout_id"]]
        capabilities = plan["capabilities"]
        smoke: dict[str, Any] | None = None
        smoke_error: str | None = None
        if (
            capabilities["actions_available"]
            and capabilities["sim_state_available"]
            and plan.get("gpu", {}).get("selected") is not None
        ):
            try:
                smoke = run_alignment_subprocess(
                    project_root=self.project_root,
                    rollout=rollout,
                    cut_frame=int(plan["alignment"]["cut_rgb_frame"]),
                    min_psnr=float(payload.get("alignment_min_psnr", 20.0)),
                    gpu_index=int(plan["gpu"]["selected"]["index"]),
                )
                if not smoke["passed"]:
                    smoke_error = (
                        "LIBERO alignment smoke test failed; inspect per-view "
                        "restore/step PSNR before generation"
                    )
            except (ValidationError, OSError, ValueError) as error:
                smoke_error = f"{type(error).__name__}: {error}"
        elif not (
            capabilities["actions_available"] and capabilities["sim_state_available"]
        ):
            smoke_error = "Alignment smoke test requires both GT actions and simulator states"
        else:
            smoke_error = "Alignment smoke test requires an eligible GPU below 50% utilization"

        if smoke_error:
            plan["blockers"].append(smoke_error)
        plan["ready"] = bool(plan["ready"] and smoke is not None and smoke.get("passed"))
        plan["validation"]["alignment_smoke_test"] = (
            smoke
            if smoke is not None
            else {
                "passed": False,
                "error": smoke_error,
            }
        )
        return plan

    def start(
        self,
        payload: dict[str, Any],
        rollout_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        plan = self.validate_plan(payload, rollout_map)
        if not plan["ready"]:
            raise ValidationError("Repair run is blocked: " + "; ".join(plan["blockers"]))
        rollout = dict(rollout_map[plan["rollout_id"]])
        run_id = self._new_run_id(str(rollout["id"]))
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=False)

        wm_config = dict(payload.get("world_model") or {})
        wm_config["name"] = "a2world"
        wm_config["gpu_index"] = int(plan["gpu"]["selected"]["index"])
        config = {
            "schema_version": 1,
            "experiment": "synthetic_suffix",
            "cut_type": plan["alignment"]["cut_type"],
            "cut_progress": plan["alignment"]["cut_progress"],
            "cut_frame": plan["alignment"]["cut_rgb_frame"],
            "alignment_min_psnr": float(payload.get("alignment_min_psnr", 20.0)),
            # The adapter strips A2World's condition frame before publishing
            # generated/<camera>.mp4, so LF3R artifacts are suffix-only.
            "generated_includes_condition": False,
            "world_model": wm_config,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        input_payload = {
            "source_manifest": rollout.get("manifest_source"),
            "rollout": rollout,
            "alignment": plan["alignment"],
        }
        adapter_status = plan["world_model"]
        provenance = {
            "schema_version": 1,
            "source_rollout": rollout["id"],
            "source_manifest": rollout.get("manifest_source"),
            "source_total_frames": rollout.get("total_frames"),
            "source_fps": rollout.get("fps"),
            "task": rollout.get("task_id"),
            "suite": rollout.get("task_suite"),
            "cut_type": plan["alignment"]["cut_type"],
            "cut_progress": plan["alignment"]["cut_progress"],
            "cut_rgb_frame": plan["alignment"]["cut_rgb_frame"],
            "condition_frame": plan["alignment"]["condition_frame"],
            "branch_state_index": plan["alignment"]["branch_state_index"],
            "gt_action_start": plan["alignment"]["gt_action_start"],
            "gt_action_end": plan["alignment"]["gt_action_end"],
            "gt_action_range_semantics": "[start,end)",
            "world_model": "a2world",
            "checkpoint": adapter_status["checkpoint"],
            "checkpoint_type": adapter_status["checkpoint_type"],
            "repair_worker_python": plan["worker_python"],
            "camera_mapping": adapter_status["camera_mapping"],
            "duplicated_camera": adapter_status["duplicated_camera"],
            "action_adapter": adapter_status["action_adapter"],
            "gpu": {
                "index": int(plan["gpu"]["selected"]["index"]),
                "uuid": plan["gpu"]["selected"].get("uuid"),
                "name": plan["gpu"]["selected"].get("name"),
                "utilization_percent_at_submit": plan["gpu"]["selected"].get("gpu_utilization_percent"),
                "memory_free_mib_at_submit": plan["gpu"]["selected"].get("memory_free_mib"),
                "selection_threshold_percent": plan["gpu"]["threshold_percent"],
            },
            "generation_config": {
                "variant": adapter_status["variant"],
                "rollout_mode": "autoregressive",
                "generated_includes_condition": False,
                "view_ids": adapter_status["view_ids"],
                "action_chunk_size": adapter_status["action_chunk_size"],
                "a2world_native_fps": 10,
                "artifact_playback_fps": rollout.get("fps"),
                "artifact_timing_basis": "source action/frame index",
                "tail_policy": "pad final chunk, then trim generated tail",
            },
            "output_paths": {},
            "created_at": config["created_at"],
        }
        status = {
            "run_id": run_id,
            "status": "queued",
            "phase": "queued",
            "progress": 0.0,
            "error": None,
            "updated_at": config["created_at"],
        }
        _atomic_json(run_dir / "config.json", config)
        _atomic_json(run_dir / "input.json", input_payload)
        _atomic_json(run_dir / "provenance.json", provenance)
        _atomic_json(run_dir / "status.json", status)

        worker = Path(__file__).resolve().parent / "worker.py"
        worker_python = self._worker_python()
        log_path = self.log_root / f"{run_id}.log"
        job_id = "repair-" + uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "job_type": self.JOB_TYPE,
            "status": "queued",
            "run_id": run_id,
            "source_rollout": rollout["id"],
            "world_model": "a2world",
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_dir": self._relative(run_dir),
            "log_path": self._relative(log_path),
            "phase": "queued",
            "progress": 0.0,
        }
        self.coordinator.acquire(job_id, "repair")
        try:
            launched = self.tmux.submit_async(
                job,
                [
                    str(worker_python),
                    str(worker),
                    "--project-root",
                    str(self.project_root),
                    "--run-dir",
                    str(run_dir),
                ],
                log_path,
                interpreter=str(worker_python),
                environment={
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                },
                on_poll=self._on_job_poll,
                on_finished=self._on_job_finished,
            )
        except Exception:
            self.coordinator.release(job_id)
            raise
        status["job_id"] = job_id
        _atomic_json(run_dir / "status.json", status)
        return launched

    def _on_job_loaded(self, job: dict[str, Any]) -> None:
        job_id = str(job.get("job_id") or "")
        if job_id and job.get("status") in {"queued", "running"}:
            self.coordinator.restore(job_id, "repair")

    def _on_job_poll(self, job: dict[str, Any]) -> None:
        run_id = str(job.get("run_id") or "")
        if not run_id:
            return
        status = self._read_json(self._run_dir(run_id) / "status.json", {})
        if isinstance(status, dict):
            job["phase"] = status.get("phase", job.get("phase"))
            job["progress"] = status.get("progress", job.get("progress"))
            if status.get("error"):
                job["repair_error"] = status["error"]

    def _on_job_finished(
        self,
        job: dict[str, Any],
        return_code: int | None,
        reason: str | None,
    ) -> None:
        run_id = str(job.get("run_id") or "")
        status = self._read_json(self._run_dir(run_id) / "status.json", {})
        worker_status = status.get("status") if isinstance(status, dict) else None
        if return_code == 0 and worker_status == "complete":
            job["status"] = "complete"
            job["phase"] = "complete"
            job["progress"] = 1.0
        else:
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = (
                (status or {}).get("error")
                if isinstance(status, dict)
                else None
            ) or reason or f"Repair worker exited with code {return_code}"
        job_id = str(job.get("job_id") or "")
        if job_id:
            self.coordinator.release(job_id)

    def job(self, job_id: str) -> dict[str, Any]:
        job = self.tmux.get(job_id)
        if job.get("job_type") != self.JOB_TYPE:
            raise KeyError(job_id)
        self._on_job_poll(job)
        return job

    def log(self, job_id: str, tail: int | str = 240) -> dict[str, Any]:
        job = self.job(job_id)
        try:
            count = max(1, min(5000, int(tail)))
        except (TypeError, ValueError):
            count = 240
        path = self.project_root / str(job["log_path"])
        if not path.is_file():
            return {"text": "", "lines": 0}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[-count:]
        return {"text": "\n".join(selected), "lines": len(lines)}

    def list_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for directory in sorted(self.root.glob("repair-suffix-*"), reverse=True):
            if not directory.is_dir():
                continue
            config = self._read_json(directory / "config.json", {})
            provenance = self._read_json(directory / "provenance.json", {})
            status = self._read_json(directory / "status.json", {})
            if not isinstance(config, dict) or not isinstance(provenance, dict):
                continue
            runs.append(
                {
                    "run_id": directory.name,
                    "status": (status or {}).get("status", "unknown"),
                    "phase": (status or {}).get("phase"),
                    "progress": (status or {}).get("progress"),
                    "error": (status or {}).get("error"),
                    "source_rollout": provenance.get("source_rollout"),
                    "source_manifest": provenance.get("source_manifest"),
                    "suite": provenance.get("suite"),
                    "task": provenance.get("task"),
                    "cut_frame": provenance.get("cut_rgb_frame"),
                    "world_model": provenance.get("world_model"),
                    "checkpoint": provenance.get("checkpoint"),
                    "checkpoint_type": provenance.get("checkpoint_type"),
                    "created_at": provenance.get("created_at"),
                    "completed_at": provenance.get("completed_at"),
                }
            )
        return runs

    def detail(self, run_id: str) -> dict[str, Any]:
        directory = self._run_dir(run_id)
        if not directory.is_dir():
            raise FileNotFoundError(run_id)
        config = self._read_json(directory / "config.json", {})
        provenance = self._read_json(directory / "provenance.json", {})
        status = self._read_json(directory / "status.json", {})
        metrics = self._read_json(directory / "metrics.json", {})
        alignment = self._read_json(directory / "alignment.json", {})
        rollout_id = str(provenance.get("source_rollout") or "")
        generated = {}
        for camera in ("cam_high", "cam_wrist"):
            path = directory / "generated" / f"{camera}.mp4"
            if path.is_file():
                generated[camera] = (
                    "/api/repair/synthetic-suffix/artifact/"
                    + quote(run_id)
                    + "/generated/"
                    + quote(path.name)
                )
        real = {
            camera: (
                "/api/videos/" + quote(rollout_id) + "?camera=" + quote(camera)
            )
            for camera in (provenance.get("camera_mapping") or {}).values()
            if isinstance(camera, str)
        }
        # Deduplicate manifest camera aliases after adapter mapping.
        real = {camera: url for camera, url in real.items()}
        fps = None
        input_payload = self._read_json(directory / "input.json", {})
        if isinstance(input_payload, dict):
            fps = (input_payload.get("rollout") or {}).get("fps")
        cut_frame = provenance.get("cut_rgb_frame")
        cut_time = (
            float(cut_frame) / float(fps)
            if fps and cut_frame is not None and float(fps) > 0
            else None
        )
        real_suffix_start_time = (
            float(int(cut_frame) + 1) / float(fps)
            if fps and cut_frame is not None and float(fps) > 0
            else None
        )
        total_frames = (
            (input_payload.get("rollout") or {}).get("total_frames")
            if isinstance(input_payload, dict)
            else None
        )
        return {
            "run_id": run_id,
            "status": status,
            "config": config,
            "provenance": provenance,
            "alignment": alignment,
            "metrics": metrics,
            "videos": {
                "real": real,
                "generated": generated,
                "fps": fps,
                "total_frames": total_frames,
                "cut_time_seconds": cut_time,
                "real_suffix_start_time_seconds": real_suffix_start_time,
            },
        }

    def artifact(self, run_id: str, relative: str) -> Path:
        directory = self._run_dir(run_id)
        target = (directory / relative).resolve()
        try:
            target.relative_to(directory)
        except ValueError as error:
            raise ValidationError("Repair artifact path escapes run directory") from error
        if not target.is_file():
            raise FileNotFoundError(relative)
        return target

    def save_human_evaluation(
        self,
        run_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        usability = str(payload.get("usable_for_policy_training") or "").strip().lower()
        if usability not in HUMAN_USABILITY:
            raise ValidationError("usable_for_policy_training must be yes, no, or uncertain")
        raw_reasons = payload.get("failure_reasons") or []
        if not isinstance(raw_reasons, list):
            raise ValidationError("failure_reasons must be an array")
        reasons = sorted({str(item).strip() for item in raw_reasons if str(item).strip()})
        invalid = set(reasons) - FAILURE_REASONS
        if invalid:
            raise ValidationError("Unknown failure reason(s): " + ", ".join(sorted(invalid)))
        directory = self._run_dir(run_id)
        if not directory.is_dir():
            raise FileNotFoundError(run_id)
        metrics_path = directory / "metrics.json"
        metrics = self._read_json(metrics_path, {})
        if not isinstance(metrics, dict):
            metrics = {}
        evaluation = {
            "usable_for_policy_training": usability,
            "failure_reasons": reasons,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        metrics["human_evaluation"] = evaluation
        _atomic_json(metrics_path, metrics)
        return evaluation
