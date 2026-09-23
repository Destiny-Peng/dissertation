#!/usr/bin/env python3
"""LF3R annotator entrypoint with live baseline progress and cancellation.

This module builds on ``server_entry.py`` so the existing ProcVLM LoRA and
non-Analysis tool integrations remain intact. It adds two operational fixes for
the Runs console:

- live baseline progress is refreshed from the runner's per-rollout/state files
  instead of relying only on the final ``run.json`` snapshot;
- active baseline jobs can be cancelled explicitly from the WebUI.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import webui_integrations  # noqa: F401  # Install integrations before job extensions.
import server


_original_refresh_progress = server.BaselineService._refresh_progress
_original_baseline_finished = server.BaselineService._on_job_finished


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _latest_job_statuses(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return latest
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        rollout_id = row.get("rollout_id")
        if rollout_id not in (None, ""):
            latest[str(rollout_id)] = row
    return latest


def _refresh_progress_with_live_state(self: server.BaselineService, job_id: str) -> None:
    """Refresh progress from run.json plus any live per-rollout runner state."""
    _original_refresh_progress(self, job_id)
    with self.jobs_lock:
        job = self.jobs.get(job_id)
        if not job:
            return
        run_root_value = job.get("run_root")
        baseline = str(job.get("baseline") or "")
        total = int(
            job.get("unique_selected_rollouts")
            or job.get("selected_rollouts")
            or 0
        )
    if not run_root_value:
        return
    try:
        run_root = self._project_path(str(run_root_value))
    except server.ValidationError:
        return

    live: dict[str, Any] | None = None
    source = None
    if baseline == "procvlm":
        live = _read_json(run_root / "procvlm_state.json")
        if live:
            source = "procvlm_state"

    # Generic fallback: completed rollout rows are already appended to jobs.jsonl
    # by the runner even when run.json is only finalized later.
    if not live:
        latest = _latest_job_statuses(run_root / "jobs.jsonl")
        if latest:
            completed = sum(row.get("status") == "complete" for row in latest.values())
            failed = sum(
                row.get("status") in {"failed", "interrupted", "fatal_engine_failure"}
                for row in latest.values()
            )
            live = {
                "completed_jobs": completed,
                "failed_jobs": failed,
                "pending_jobs": max(total - completed - failed, 0),
            }
            source = "jobs_jsonl"

    if not live:
        return
    with self.jobs_lock:
        job = self.jobs.get(job_id)
        if not job:
            return
        for field in ("completed_jobs", "failed_jobs", "pending_jobs"):
            value = live.get(field)
            if value is None:
                continue
            try:
                job[field] = int(value)
            except (TypeError, ValueError):
                continue
        job["live_progress_source"] = source
        job["live_progress_updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        live_status = live.get("status")
        if live_status not in (None, ""):
            job["live_run_status"] = str(live_status)


def _atomic_update_run_metadata(job: dict[str, Any]) -> None:
    run_root_value = job.get("run_root")
    if not run_root_value:
        return
    run_root = Path(str(run_root_value))
    if not run_root.is_absolute():
        project_root = Path(server.DEFAULT_PROJECT_ROOT)
        run_root = project_root / run_root
    metadata_path = run_root / "run.json"
    payload = _read_json(metadata_path)
    if payload is None:
        return
    payload.update(
        {
            "status": "cancelled",
            "cancelled_at": job.get("cancelled_at"),
            "completed_at": job.get("cancelled_at"),
            "completed_jobs": int(job.get("completed_jobs") or 0),
            "failed_jobs": int(job.get("failed_jobs") or 0),
            "pending_jobs": int(job.get("pending_jobs") or 0),
        }
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{metadata_path.name}.", suffix=".tmp", dir=metadata_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, metadata_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _baseline_finished_with_cancel(
    self: server.BaselineService,
    job: dict[str, Any],
    return_code: int | None,
    reason: str | None,
) -> None:
    if not job.get("cancel_requested_at"):
        return _original_baseline_finished(self, job, return_code, reason)

    job_id = str(job["job_id"])
    try:
        self._refresh_progress(job_id)
        cancelled_at = dt.datetime.now(dt.timezone.utc).isoformat()
        with self.jobs_lock:
            job["status"] = "cancelled"
            job["tmux_state"] = "cancelled"
            job["cancelled_at"] = cancelled_at
            job["finished_at"] = cancelled_at
            job["return_code"] = return_code
            job["error"] = None
            job["run_status"] = "cancelled"
        _atomic_update_run_metadata(job)
    finally:
        self.coordinator.release(job_id)


def _cancel_baseline_job(self: server.BaselineService, job_id: str) -> dict[str, Any]:
    with self.jobs_lock:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.get("status") not in {"queued", "running"}:
            raise server.ValidationError(
                f"Baseline job is not active: {job.get('status') or 'unknown'}"
            )

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    with self.tmux.lock:
        shared = self.tmux.jobs.get(job_id)
        if shared is None:
            raise KeyError(job_id)
        if shared.get("cancel_requested_at"):
            return dict(shared)
        shared["cancel_requested_at"] = now
        shared["cancel_requested_by"] = "webui"
        shared["supervisor_status"] = "cancelling"
        self.tmux.persist(shared)
        session = str(shared.get("tmux_session") or "")

    if session and self.tmux.has_session(session):
        # SIGINT first gives Python/children a chance to release GPU resources;
        # kill-session then guarantees the persistent task does not keep running.
        self.tmux._tmux(["send-keys", "-t", session, "C-c"], check=False)
        time.sleep(0.15)
        self.tmux._tmux(["kill-session", "-t", session], check=False)

    with self.tmux.lock:
        shared = self.tmux.jobs.get(job_id)
        if shared is not None:
            shared["cancel_signal_sent_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            self.tmux.persist(shared)
    self._refresh_progress(job_id)
    with self.jobs_lock:
        return dict(self.jobs[job_id])


server.BaselineService._refresh_progress = _refresh_progress_with_live_state
server.BaselineService._on_job_finished = _baseline_finished_with_cancel
server.BaselineService.cancel_job = _cancel_baseline_job


_previous_do_get = server.LF3RHandler.do_GET


def _do_get_with_live_jobs(self: server.LF3RHandler) -> None:
    parsed = urlparse(self.path)
    path = unquote(parsed.path)
    if path != "/api/jobs":
        return _previous_do_get(self)
    query = parse_qs(parsed.query, keep_blank_values=True)
    job_type = query.get("job_type", [None])[0] or None
    status = query.get("status", [None])[0] or None
    jobs = self.app.tmux.list(job_type=job_type, status=status)
    refreshed = []
    for job in jobs:
        if job.get("job_type") == "baseline":
            try:
                job = self.app.baselines.job(str(job["job_id"]))
            except KeyError:
                pass
        refreshed.append(job)
    self.json_response(
        HTTPStatus.OK,
        {"jobs": refreshed, "tmux_available": self.app.tmux.available},
    )


server.LF3RHandler.do_GET = _do_get_with_live_jobs


_previous_do_post = server.LF3RHandler.do_POST


def _do_post_with_baseline_cancel(self: server.LF3RHandler) -> None:
    path = unquote(urlparse(self.path).path)
    parts = path.strip("/").split("/")
    if not (
        len(parts) == 4
        and parts[0] == "api"
        and parts[1] == "baseline-jobs"
        and parts[3] == "cancel"
    ):
        return _previous_do_post(self)
    try:
        job = self.app.baselines.cancel_job(parts[2])
        self.json_response(HTTPStatus.ACCEPTED, {"job": job})
    except KeyError:
        self.json_error(HTTPStatus.NOT_FOUND, "Unknown baseline job")
    except server.ValidationError as exc:
        self.json_error(HTTPStatus.CONFLICT, str(exc))
    except server.TmuxSupervisorError as exc:
        self.json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
    except OSError as exc:
        self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


server.LF3RHandler.do_POST = _do_post_with_baseline_cancel
