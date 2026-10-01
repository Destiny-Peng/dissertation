"""Synthetic demonstration repair support for the LF3R WebUI."""

from __future__ import annotations

import threading
import time
from typing import Any

from .batch import RepairBatchService
from .service import RepairService as _SingleRepairService


class RepairService(_SingleRepairService):
    """Single-run Repair service with Runs-style sequential batch support."""

    GPU_PLAN_CACHE_SECONDS = 5.0

    def __init__(self, project_root, coordinator, tmux) -> None:
        # The parent registers callbacks using these overridden bound methods.
        # Batch state is attached before tmux.recover() is called by the app.
        self._gpu_plan_cache_lock = threading.Lock()
        self._gpu_plan_cache: dict[int, tuple[float, dict[str, Any]]] = {}
        super().__init__(project_root, coordinator, tmux)
        self.batch = RepairBatchService(
            self.project_root,
            self.coordinator,
            self.tmux,
            self,
        )

    def _gpu_plan(self, requested_index: int) -> dict[str, Any]:
        """Reuse one nvidia-smi snapshot across a batch preflight."""
        index = int(requested_index)
        now = time.monotonic()
        with self._gpu_plan_cache_lock:
            cached = self._gpu_plan_cache.get(index)
            if cached is not None and now - cached[0] <= self.GPU_PLAN_CACHE_SECONDS:
                return dict(cached[1])
        result = _SingleRepairService._gpu_plan(index)
        with self._gpu_plan_cache_lock:
            self._gpu_plan_cache[index] = (now, dict(result))
        return result

    def start(
        self,
        payload: dict[str, Any],
        rollout_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if isinstance(payload, dict) and isinstance(payload.get("rollout_ids"), list):
            return self.batch.start(payload, rollout_map)
        return super().start(payload, rollout_map)

    def _on_job_loaded(self, job: dict[str, Any]) -> None:
        if job.get("batch"):
            self.batch.on_loaded(job)
            return
        super()._on_job_loaded(job)

    def _on_job_poll(self, job: dict[str, Any]) -> None:
        if job.get("batch"):
            self.batch.on_poll(job)
            return
        super()._on_job_poll(job)

    def _on_job_finished(
        self,
        job: dict[str, Any],
        return_code: int | None,
        reason: str | None,
    ) -> None:
        if job.get("batch"):
            self.batch.on_finished(job, return_code, reason)
            return
        super()._on_job_finished(job, return_code, reason)


__all__ = ["RepairService"]
