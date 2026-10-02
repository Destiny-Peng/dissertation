"""Sequential persistent batch execution for Repair / Synthetic Suffix."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from backend_core import JobCoordinator, ValidationError
from task_supervisor import TmuxJobSupervisor


BATCH_ID_RE = re.compile(r"^repair-batch-[A-Za-z0-9._-]{1,120}$")


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


class RepairBatchService:
    """Run several normal Repair runs sequentially inside one tmux job.

    Every selected rollout still receives its own normal ``repair-suffix-*``
    artifact directory. The batch only owns orchestration/progress, so Compare
    and downstream export keep consuming the existing per-run format.
    """

    def __init__(
        self,
        project_root: Path,
        coordinator: JobCoordinator,
        tmux: TmuxJobSupervisor,
        repair: Any,
    ) -> None:
        self.project_root = project_root.resolve()
        self.coordinator = coordinator
        self.tmux = tmux
        self.repair = repair
        self.root = (
            self.project_root
            / "artifacts"
            / "repair"
            / "synthetic_suffix"
            / "batches"
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(path.resolve())

    def _batch_dir(self, batch_id: str) -> Path:
        if not BATCH_ID_RE.fullmatch(str(batch_id)):
            raise ValidationError("Invalid Repair batch id")
        path = (self.root / batch_id).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValidationError("Repair batch path escapes artifact root") from error
        return path

    @staticmethod
    def _read_json(path: Path, default: Any = None) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    @staticmethod
    def _gpu_provenance(plan: dict[str, Any]) -> dict[str, Any]:
        selected = (plan.get("gpu") or {}).get("selected")
        gpu = plan.get("gpu") or {}
        if selected is not None:
            return {
                "index": int(selected["index"]),
                "uuid": selected.get("uuid"),
                "name": selected.get("name"),
                "utilization_percent_at_submit": selected.get(
                    "gpu_utilization_percent"
                ),
                "memory_free_mib_at_submit": selected.get("memory_free_mib"),
                "selection_policy": gpu.get("selection_policy"),
                "utilization_gate": False,
            }
        return {
            "index": int(gpu["requested_index"]),
            "selection_policy": gpu.get("selection_policy"),
            "utilization_gate": False,
            "status_available": gpu.get("available"),
            "status_error": (
                gpu.get("error")
                or "Selected GPU was not present in the status snapshot"
            ),
        }

    @staticmethod
    def _generation_config(
        plan: dict[str, Any],
        rollout: dict[str, Any],
    ) -> dict[str, Any]:
        adapter = plan["world_model"]
        if plan["model_name"] == "a2world":
            return {
                "variant": adapter["variant"],
                "rollout_mode": "autoregressive",
                "generated_includes_condition": False,
                "view_ids": adapter["view_ids"],
                "action_chunk_size": adapter["action_chunk_size"],
                "num_sampling_steps": adapter["num_sampling_steps"],
                "guidance": adapter["guidance"],
                "seed": adapter["seed"],
                "history": adapter["history"],
                "a2world_native_fps": 10,
                "artifact_playback_fps": rollout.get("fps"),
                "artifact_timing_basis": "source action/frame index",
                "tail_policy": "pad final chunk, then trim generated tail",
            }
        if plan["model_name"] == "wan2_2":
            return {"task": "i2v-A14B", "generated_includes_condition": True,
                    "input_contract": "single cam_high RGB frame + original task instruction"}
        return {
            "rollout_mode": "autoregressive",
            "generated_includes_condition": False,
            "view_order": adapter["view_order"],
            "target_fps": adapter["target_fps"],
            "source_frame_step": adapter["source_frame_step"],
            "effective_fps": adapter["effective_fps"],
            "svd_microcondition_fps": adapter["svd_microcondition_fps"],
            "num_frames": adapter["num_frames"],
            "num_history": adapter["num_history"],
            "num_inference_steps": adapter["num_inference_steps"],
            "guidance_scale": adapter["guidance_scale"],
            "seed": adapter["seed"],
            "text_conditioning": adapter["text_conditioning"],
            "artifact_timing_basis": (
                "source time sampled at Ctrl-World effective FPS"
            ),
        }

    def _prepare_run(
        self,
        *,
        payload: dict[str, Any],
        rollout: dict[str, Any],
        plan: dict[str, Any],
        batch_id: str,
    ) -> dict[str, Any]:
        run_id = self.repair._new_run_id(str(rollout["id"]))
        run_dir = self.repair._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=False)

        wm_config = dict(payload.get("world_model") or {})
        wm_config["name"] = plan["model_name"]
        if plan["model_name"] == "wan2_2":
            wm_config.update({key: plan["world_model"][key] for key in ("python", "checkpoint", "source_root")})
        wm_config["gpu_index"] = int(plan["gpu"]["requested_index"])
        created_at = dt.datetime.now(dt.timezone.utc).isoformat()
        config = {
            "schema_version": 1,
            "experiment": "synthetic_suffix",
            "cut_type": plan["alignment"]["cut_type"],
            "cut_progress": plan["alignment"]["cut_progress"],
            "cut_frame": plan["alignment"]["cut_rgb_frame"],
            "alignment_min_psnr": float(payload.get("alignment_min_psnr", 20.0)),
            "generated_includes_condition": plan["model_name"] == "wan2_2",
            "world_model": wm_config,
            "created_at": created_at,
            "batch_id": batch_id,
        }
        input_payload = {
            "source_manifest": rollout.get("manifest_source"),
            "rollout": rollout,
            "alignment": plan["alignment"],
        }
        adapter = plan["world_model"]
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
            "world_model": plan["model_name"],
            "checkpoint": adapter["checkpoint"],
            "checkpoint_type": adapter["checkpoint_type"],
            "repair_worker_python": plan["worker_python"],
            "runtime_libero_required": plan["model_name"] == "a2world",
            "offline_ctrl_prepared": bool(rollout.get("ctrl_prepared")),
            "camera_mapping": adapter["camera_mapping"],
            "duplicated_camera": adapter["duplicated_camera"],
            "action_adapter": adapter.get("action_adapter"),
            "control_adapter": adapter.get("control_adapter"),
            "control_semantics": adapter.get("control_semantics"),
            "gpu": self._gpu_provenance(plan),
            "generation_config": self._generation_config(plan, rollout),
            "output_paths": {},
            "created_at": created_at,
            "batch_id": batch_id,
        }
        status = {
            "run_id": run_id,
            "status": "queued",
            "phase": "queued",
            "progress": 0.0,
            "error": None,
            "updated_at": created_at,
            "batch_id": batch_id,
        }
        _atomic_json(run_dir / "config.json", config)
        _atomic_json(run_dir / "input.json", input_payload)
        _atomic_json(run_dir / "provenance.json", provenance)
        _atomic_json(run_dir / "status.json", status)

        worker_python_value = str(plan["worker_python"])
        worker_python_path = Path(worker_python_value).expanduser()
        worker_python = (
            worker_python_path.absolute()
            if plan["model_name"] == "wan2_2"
            else (
                worker_python_path.resolve()
                if worker_python_path.is_absolute()
                else (self.project_root / worker_python_path).resolve()
            )
        )
        return {
            "run_id": run_id,
            "rollout_id": rollout["id"],
            "run_dir": self._relative(run_dir),
            "worker_python": str(worker_python),
        }

    def start(
        self,
        payload: dict[str, Any],
        rollout_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValidationError("Repair batch request must be a JSON object")
        raw_ids = payload.get("rollout_ids")
        if not isinstance(raw_ids, list):
            raise ValidationError("rollout_ids must be an array")
        rollout_ids: list[str] = []
        seen: set[str] = set()
        for raw in raw_ids:
            rollout_id = str(raw or "").strip()
            if not rollout_id or rollout_id in seen:
                continue
            if rollout_id not in rollout_map:
                raise ValidationError(f"Unknown rollout in Repair batch: {rollout_id}")
            seen.add(rollout_id)
            rollout_ids.append(rollout_id)
        if not rollout_ids:
            raise ValidationError("Repair batch selection is empty")
        if len(rollout_ids) > 500:
            raise ValidationError("Repair batch supports at most 500 rollouts per job")

        base_payload = dict(payload)
        base_payload.pop("rollout_ids", None)
        base_payload.pop("batch_selection", None)
        prepared: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        blocked: list[str] = []
        for rollout_id in rollout_ids:
            one = dict(base_payload)
            one["rollout_id"] = rollout_id
            plan = self.repair.validate_plan(one, rollout_map)
            if not plan["ready"]:
                blocked.append(
                    rollout_id + ": " + "; ".join(plan.get("blockers") or ["blocked"])
                )
                continue
            prepared.append((one, dict(rollout_map[rollout_id]), plan))
        if blocked:
            preview = blocked[:8]
            suffix = "" if len(blocked) <= 8 else f"; +{len(blocked) - 8} more"
            raise ValidationError(
                "Repair batch preflight failed: " + " | ".join(preview) + suffix
            )

        batch_id = "repair-batch-" + uuid.uuid4().hex[:12]
        batch_dir = self._batch_dir(batch_id)
        batch_dir.mkdir(parents=True, exist_ok=False)
        items = [
            self._prepare_run(
                payload=one,
                rollout=rollout,
                plan=plan,
                batch_id=batch_id,
            )
            for one, rollout, plan in prepared
        ]
        first_plan = prepared[0][2]
        model_name = str(first_plan["model_name"])
        selection_meta = payload.get("batch_selection")
        if not isinstance(selection_meta, dict):
            selection_meta = {}
        plan_payload = {
            "schema_version": 1,
            "batch_id": batch_id,
            "world_model": model_name,
            "gpu_index": int(first_plan["gpu"]["requested_index"]),
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "selection": selection_meta,
            "items": items,
        }
        batch_status = {
            "batch_id": batch_id,
            "status": "queued",
            "phase": "queued",
            "progress": 0.0,
            "expected_runs": len(items),
            "completed_runs": 0,
            "failed_runs": 0,
            "current_rollout": None,
            "current_run_id": None,
            "updated_at": plan_payload["created_at"],
        }
        _atomic_json(batch_dir / "plan.json", plan_payload)
        _atomic_json(batch_dir / "status.json", batch_status)

        job_id = batch_id
        log_path = self.repair.log_root / f"{job_id}.log"
        worker = Path(__file__).resolve().parent / "batch_worker.py"
        job = {
            "job_id": job_id,
            "job_type": self.repair.JOB_TYPE,
            "batch": True,
            "batch_id": batch_id,
            "status": "queued",
            "world_model": model_name,
            "gpu": str(first_plan["gpu"]["requested_index"]),
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_dir": self._relative(batch_dir),
            "log_path": self._relative(log_path),
            "phase": "queued",
            "progress": 0.0,
            "rollout_ids": rollout_ids,
            "selected_rollouts": len(items),
            "expected_runs": len(items),
            "completed_runs": 0,
            "failed_runs": 0,
            "selection": selection_meta,
        }
        self.coordinator.acquire(job_id, "repair")
        try:
            launched = self.tmux.submit_async(
                job,
                [
                    str(Path(sys.executable).resolve()),
                    str(worker),
                    "--project-root",
                    str(self.project_root),
                    "--batch-dir",
                    str(batch_dir),
                ],
                log_path,
                interpreter=str(Path(sys.executable).resolve()),
                environment={
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                },
                on_poll=self.on_poll,
                on_finished=self.on_finished,
            )
        except Exception:
            self.coordinator.release(job_id)
            raise
        return launched

    def on_loaded(self, job: dict[str, Any]) -> None:
        job_id = str(job.get("job_id") or "")
        if job_id and job.get("status") in {"queued", "running"}:
            self.coordinator.restore(job_id, "repair")

    def on_poll(self, job: dict[str, Any]) -> None:
        batch_id = str(job.get("batch_id") or job.get("job_id") or "")
        if not batch_id:
            return
        status = self._read_json(self._batch_dir(batch_id) / "status.json", {})
        if not isinstance(status, dict):
            return
        for key in (
            "phase",
            "progress",
            "expected_runs",
            "completed_runs",
            "failed_runs",
            "current_rollout",
            "current_run_id",
        ):
            if key in status:
                job[key] = status[key]
        if status.get("error"):
            job["repair_error"] = status["error"]

    def on_finished(
        self,
        job: dict[str, Any],
        return_code: int | None,
        reason: str | None,
    ) -> None:
        batch_id = str(job.get("batch_id") or job.get("job_id") or "")
        status = self._read_json(self._batch_dir(batch_id) / "status.json", {})
        worker_status = status.get("status") if isinstance(status, dict) else None
        if return_code == 0 and worker_status in {
            "complete",
            "complete_with_errors",
            "failed",
        }:
            job["status"] = worker_status
            job["phase"] = status.get("phase", worker_status)
            job["progress"] = float(status.get("progress", 1.0))
            job["completed_runs"] = int(status.get("completed_runs", 0))
            job["failed_runs"] = int(status.get("failed_runs", 0))
            if status.get("error"):
                job["error"] = status["error"]
        else:
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = (
                (status or {}).get("error")
                if isinstance(status, dict)
                else None
            ) or reason or f"Repair batch worker exited with code {return_code}"
        job_id = str(job.get("job_id") or "")
        if job_id:
            self.coordinator.release(job_id)
