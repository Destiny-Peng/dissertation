#!/usr/bin/env python3
"""Persistent tmux-backed task supervision for the LF3R annotator.

The annotator server is intentionally small and local-only.  This module keeps
long-running commands out of the server process lifecycle while retaining
project-local job records that can be reconciled after a server restart.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable


class TmuxSupervisorError(RuntimeError):
    """Raised when a persistent tmux task cannot be created or inspected."""


class TmuxJobSupervisor:
    """Launch, persist, monitor, and recover project-local tmux jobs.

    ``on_loaded`` receives a recovered job before monitoring starts.  The
    optional ``on_poll`` callback can update service-specific progress fields;
    ``on_finished`` must assign the service-specific terminal status.
    """

    SESSION_PREFIX = "lf3r-annotator-"
    JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,160}$")
    ACTIVE_STATUSES = {"queued", "running"}
    TERMINAL_STATUSES = {"complete", "complete_with_errors", "failed", "memory_blocked"}
    ENV_KEYS = (
        "PROJECT_ROOT",
        "REPOS",
        "DATASETS",
        "CHECKPOINTS",
        "CONDA_ENVS",
        "CACHE",
        "OUTPUTS",
        "LOGS",
        "ENV_REPORTS",
        "CONDA_PKGS_DIRS",
        "PIP_CACHE_DIR",
        "UV_CACHE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "TMPDIR",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "MPLBACKEND",
        "PYTHONPATH",
    )

    def __init__(
        self,
        project_root: Path,
        registry_root: Path | None = None,
        tmux_binary: str | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.registry_root = (
            registry_root or self.project_root / "logs" / "annotator_jobs"
        ).resolve()
        try:
            self.registry_root.relative_to(self.project_root)
        except ValueError as error:
            raise TmuxSupervisorError("Job registry must be inside the project root") from error
        self.tmux_binary = tmux_binary or shutil.which("tmux")
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.monitors: set[str] = set()
        self.handlers: dict[str, dict[str, Callable[..., Any] | None]] = {}

    @property
    def available(self) -> bool:
        if not self.tmux_binary:
            return False
        candidate = Path(str(self.tmux_binary))
        return candidate.is_file() or bool(shutil.which(str(self.tmux_binary)))

    def register_handler(
        self,
        job_type: str,
        on_loaded: Callable[[dict[str, Any]], None],
        on_poll: Callable[[dict[str, Any]], None] | None = None,
        on_finished: Callable[[dict[str, Any], int | None, str | None], None] | None = None,
    ) -> None:
        self.handlers[job_type] = {
            "on_loaded": on_loaded,
            "on_poll": on_poll,
            "on_finished": on_finished,
        }

    def _job_dir(self, job_id: str) -> Path:
        if not self.JOB_ID_RE.fullmatch(job_id):
            raise TmuxSupervisorError("Invalid job id")
        path = (self.registry_root / job_id).resolve()
        try:
            path.relative_to(self.registry_root)
        except ValueError as error:
            raise TmuxSupervisorError("Job path escapes project root") from error
        return path

    def _record_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _wrapper_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "run.sh"

    def _exit_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "exit_code"

    def _finished_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "finished_at"

    def _project_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        resolved = path.resolve() if path.is_absolute() else (self.project_root / path).resolve()
        try:
            resolved.relative_to(self.project_root)
        except ValueError as error:
            raise TmuxSupervisorError("Task path escapes project root") from error
        return resolved

    @staticmethod
    def _atomic_json_write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(path)

    def persist(self, job: dict[str, Any]) -> None:
        path = self._record_path(str(job["job_id"]))
        # A test harness or an operator may remove a temporary project tree
        # while a daemon monitor is finishing. Do not recreate that tree or
        # leak an exception from the monitor thread.
        if not path.parent.is_dir():
            return
        try:
            self._atomic_json_write(path, job)
        except FileNotFoundError:
            if not path.parent.exists():
                return
            raise

    def _tmux(self, args: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
        if not self.available:
            raise TmuxSupervisorError(
                "tmux is required for annotator jobs but was not found on PATH"
            )
        return subprocess.run(
            [str(self.tmux_binary), *args],
            cwd=str(self.project_root),
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            check=check,
        )

    def has_session(self, session_name: str) -> bool:
        result = self._tmux(["has-session", "-t", session_name])
        return result.returncode == 0

    def _write_wrapper(
        self,
        job_id: str,
        command: list[str],
        log_path: Path,
        environment: dict[str, str] | None = None,
    ) -> Path:
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        exports: list[str] = []
        merged_environment = {
            key: os.environ[key]
            for key in self.ENV_KEYS
            if key in os.environ
        }
        merged_environment.update(environment or {})
        for key, value in sorted(merged_environment.items()):
            exports.append(f"export {key}={shlex.quote(str(value))}")
        command_text = shlex.join([str(item) for item in command])
        wrapper = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set +e",
                f"cd -- {shlex.quote(str(self.project_root))}",
                f"source {shlex.quote(str(self.project_root / 'project_env.sh'))}",
                *exports,
                f"{command_text} > {shlex.quote(str(log_path))} 2>&1",
                "status=$?",
                f"printf '%s\\n' \"$status\" > {shlex.quote(str(self._exit_path(job_id)))}",
                f"date -u +%Y-%m-%dT%H:%M:%SZ > {shlex.quote(str(self._finished_path(job_id)))}",
                "exit \"$status\"",
                "",
            ]
        )
        path = self._wrapper_path(job_id)
        path.write_text(wrapper, encoding="utf-8")
        path.chmod(0o755)
        return path

    def submit(
        self,
        job: dict[str, Any],
        command: list[str],
        log_path: Path,
        interpreter: str | None = None,
        environment: dict[str, str] | None = None,
        on_poll: Callable[[dict[str, Any]], None] | None = None,
        on_finished: Callable[[dict[str, Any], int | None, str | None], None] | None = None,
    ) -> dict[str, Any]:
        job_id = str(job["job_id"])
        if not self.available:
            raise TmuxSupervisorError(
                "tmux is required for annotator jobs but was not found on PATH"
            )
        log_path = self._project_path(log_path)
        with self.lock:
            if job_id in self.jobs or self._record_path(job_id).exists():
                raise TmuxSupervisorError(f"Job already exists: {job_id}")
            session = self.SESSION_PREFIX + job_id
            wrapper = self._write_wrapper(job_id, command, log_path, environment)
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            job.update(
                {
                    "tmux_session": session,
                    "tmux_state": "starting",
                    "persistent": True,
                    "job_record_path": self._relative(self._record_path(job_id)),
                    "wrapper_path": self._relative(wrapper),
                    "exit_code_path": self._relative(self._exit_path(job_id)),
                    "finished_at_path": self._relative(self._finished_path(job_id)),
                    "argv": [str(item) for item in command],
                    "working_directory": str(self.project_root),
                    "tmux_created_at": now,
                    "supervisor_status": "submitted",
                }
            )
            if interpreter:
                job["interpreter"] = interpreter
            self.jobs[job_id] = job
            self.persist(job)
            try:
                self._tmux(
                    ["new-session", "-d", "-s", session, "bash", str(wrapper)],
                    check=True,
                )
            except Exception as error:
                job["status"] = "failed"
                job["tmux_state"] = "launch_failed"
                job["supervisor_status"] = "launch_failed"
                job["error"] = str(error)
                self.persist(job)
                self.jobs.pop(job_id, None)
                raise TmuxSupervisorError(f"Could not start tmux session {session}: {error}") from error
            job["status"] = "running"
            job["tmux_state"] = "running"
            job["supervisor_status"] = "monitoring"
            job["started_at"] = job.get("started_at") or now
            self.persist(job)
        self._start_monitor(job_id, on_poll, on_finished)
        return dict(job)

    def _read_exit_code(self, job_id: str) -> int | None:
        path = self._exit_path(job_id)
        if not path.is_file():
            return None
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def _start_monitor(
        self,
        job_id: str,
        on_poll: Callable[[dict[str, Any]], None] | None,
        on_finished: Callable[[dict[str, Any], int | None, str | None], None] | None,
    ) -> None:
        with self.lock:
            if job_id in self.monitors:
                return
            self.monitors.add(job_id)
        threading.Thread(
            target=self._monitor,
            args=(job_id, on_poll, on_finished),
            name=f"lf3r-tmux-{job_id}",
            daemon=True,
        ).start()

    def _monitor(
        self,
        job_id: str,
        on_poll: Callable[[dict[str, Any]], None] | None,
        on_finished: Callable[[dict[str, Any], int | None, str | None], None] | None,
    ) -> None:
        return_code: int | None = None
        reason: str | None = None
        try:
            while True:
                with self.lock:
                    job = self.jobs.get(job_id)
                if job is None:
                    return
                return_code = self._read_exit_code(job_id)
                if return_code is not None:
                    job["tmux_state"] = "exited"
                    job["supervisor_status"] = "finalizing"
                    break
                try:
                    active = self.has_session(job["tmux_session"])
                except (TmuxSupervisorError, OSError) as error:
                    # A server/test project can disappear while a daemon monitor
                    # is finishing; treat that as an orphaned session instead of
                    # leaking an exception from the monitor thread.
                    reason = str(error) or "tmux session could not be queried"
                    break
                if not active:
                    # The wrapper writes the exit marker immediately before it
                    # exits, so tmux can report a vanished session a few
                    # milliseconds before the marker becomes visible. Re-read
                    # it briefly before treating the session as orphaned.
                    for _ in range(10):
                        return_code = self._read_exit_code(job_id)
                        if return_code is not None:
                            break
                        time.sleep(0.05)
                    if return_code is None:
                        reason = "tmux session disappeared before its exit marker was written"
                        job["tmux_state"] = "missing"
                        job["supervisor_status"] = "orphaned"
                    else:
                        job["tmux_state"] = "exited"
                        job["supervisor_status"] = "finalizing"
                    break
                job["tmux_state"] = "running"
                if on_poll:
                    try:
                        on_poll(job)
                    except Exception as error:
                        job["supervisor_status"] = "poll_error"
                        job["poll_error"] = str(error)
                self.persist(job)
                time.sleep(0.5)
            if on_finished:
                try:
                    on_finished(job, return_code, reason)
                except Exception as error:
                    job["status"] = "failed"
                    job["error"] = str(error)
            if reason and job.get("status") in self.ACTIVE_STATUSES:
                job["status"] = "failed"
                job["error"] = reason
            job["return_code"] = return_code
            job["finished_at"] = job.get("finished_at") or dt.datetime.now(dt.timezone.utc).isoformat()
            job["supervisor_status"] = "finished"
            self.persist(job)
        finally:
            with self.lock:
                self.monitors.discard(job_id)

    def recover(self) -> None:
        """Load known jobs and resume monitoring without touching unknown sessions."""
        self.registry_root.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        for path in sorted(self.registry_root.glob("*/job.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            job_id = payload.get("job_id")
            job_type = payload.get("job_type")
            if not isinstance(job_id, str) or not self.JOB_ID_RE.fullmatch(job_id):
                continue
            if job_type not in self.handlers:
                continue
            records.append(payload)
        for job in records:
            job_id = str(job["job_id"])
            handler = self.handlers[str(job["job_type"])]
            with self.lock:
                self.jobs[job_id] = job
            on_loaded = handler["on_loaded"]
            if on_loaded:
                on_loaded(job)
            if job.get("status") not in self.ACTIVE_STATUSES:
                continue
            exit_code = self._read_exit_code(job_id)
            session = str(job.get("tmux_session") or "")
            if exit_code is not None:
                self._start_monitor(job_id, handler["on_poll"], handler["on_finished"])
                continue
            try:
                active = bool(session) and self.has_session(session)
            except TmuxSupervisorError as error:
                active = False
                job["error"] = str(error)
            if active:
                job["tmux_state"] = "reattached"
                job["reattached_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                job["supervisor_status"] = "reattached"
                self.persist(job)
                self._start_monitor(job_id, handler["on_poll"], handler["on_finished"])
            else:
                job["status"] = "failed"
                job["tmux_state"] = "missing"
                job["supervisor_status"] = "orphaned"
                job["error"] = job.get("error") or "Recorded tmux session is no longer active"
                job["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                self.persist(job)
                on_finished = handler["on_finished"]
                if on_finished:
                    on_finished(job, None, job["error"])

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return dict(self.jobs[job_id])

    def list(self, job_type: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            jobs = list(self.jobs.values())
        if job_type:
            jobs = [job for job in jobs if job.get("job_type") == job_type]
        if status:
            jobs = [job for job in jobs if job.get("status") == status]
        jobs.sort(key=lambda job: str(job.get("submitted_at") or job.get("tmux_created_at") or job.get("job_id")), reverse=True)
        return [dict(job) for job in jobs]

