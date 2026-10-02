"""Indexed persistent tmux task supervision for fast WebUI startup."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

from task_supervisor import TmuxJobSupervisor, TmuxSupervisorError


class IndexedTmuxJobSupervisor(TmuxJobSupervisor):
    """Tmux supervisor backed by a lightweight persistent job index.

    Job JSON files remain authoritative. The SQLite database is only an index so
    startup can recover queued/running jobs without scanning every historical
    ``logs/annotator_jobs/*/job.json`` directory.
    """

    def __init__(
        self,
        project_root: Path,
        registry_root: Path | None = None,
        tmux_binary: str | None = None,
    ) -> None:
        super().__init__(
            project_root,
            registry_root=registry_root,
            tmux_binary=tmux_binary,
        )
        self.index_path = (
            self.project_root
            / "cache"
            / "lf3r_annotator"
            / "persistent_jobs.sqlite3"
        )
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._indexed_state: dict[str, tuple[str, str, str, str]] = {}
        self._initialize_index()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.index_path, timeout=2.0)
        connection.execute("PRAGMA busy_timeout = 2000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _initialize_index(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS persistent_jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sort_time TEXT NOT NULL,
                    record_path TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS persistent_jobs_status_type
                ON persistent_jobs (status, job_type, sort_time DESC)
                """
            )

    def _index_state(self, job: dict[str, Any]) -> tuple[str, str, str, str]:
        job_id = str(job["job_id"])
        return (
            str(job.get("job_type") or ""),
            str(job.get("status") or ""),
            str(
                job.get("submitted_at")
                or job.get("tmux_created_at")
                or job.get("started_at")
                or job_id
            ),
            self._relative(self._record_path(job_id)),
        )

    def _index_job(self, job: dict[str, Any], *, force: bool = False) -> None:
        job_id = str(job.get("job_id") or "")
        if not self.JOB_ID_RE.fullmatch(job_id):
            return
        state = self._index_state(job)
        if not force and self._indexed_state.get(job_id) == state:
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO persistent_jobs (
                    job_id, job_type, status, sort_time, record_path, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    job_type = excluded.job_type,
                    status = excluded.status,
                    sort_time = excluded.sort_time,
                    record_path = excluded.record_path,
                    updated_at = excluded.updated_at
                """,
                (
                    job_id,
                    state[0],
                    state[1],
                    state[2],
                    state[3],
                    dt.datetime.now(dt.timezone.utc).isoformat(),
                ),
            )
        self._indexed_state[job_id] = state

    def persist(self, job: dict[str, Any]) -> None:
        super().persist(job)
        job_id = str(job.get("job_id") or "")
        if job_id and self._record_path(job_id).is_file():
            self._index_job(job)

    def _indexed_rows(
        self,
        *,
        job_type: str | None = None,
        status: str | None = None,
        active_only: bool = False,
    ) -> list[tuple[str, str]]:
        clauses: list[str] = []
        values: list[str] = []
        if job_type:
            clauses.append("job_type = ?")
            values.append(job_type)
        if status:
            clauses.append("status = ?")
            values.append(status)
        elif active_only:
            placeholders = ",".join("?" for _ in sorted(self.ACTIVE_STATUSES))
            clauses.append(f"status IN ({placeholders})")
            values.extend(sorted(self.ACTIVE_STATUSES))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT job_id, record_path FROM persistent_jobs"
                + where
                + " ORDER BY sort_time DESC",
                values,
            ).fetchall()
        return [(str(row[0]), str(row[1])) for row in rows]

    def _read_record(self, job_id: str, record_path: str | None = None) -> dict[str, Any] | None:
        if not self.JOB_ID_RE.fullmatch(job_id):
            return None
        path = (
            (self.project_root / record_path).resolve()
            if record_path
            else self._record_path(job_id)
        )
        try:
            path.relative_to(self.project_root)
        except ValueError:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("job_id") != job_id:
            return None
        return payload

    def _active_tmux_sessions(self) -> set[str]:
        if not self.available:
            return set()
        try:
            result = self._tmux(["list-sessions", "-F", "#{session_name}"])
        except TmuxSupervisorError:
            return set()
        if result.returncode != 0:
            return set()
        return {
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip().startswith(self.SESSION_PREFIX)
        }

    def recover(self) -> dict[str, int]:
        """Recover active jobs without recursively scanning historical jobs."""
        self.registry_root.mkdir(parents=True, exist_ok=True)
        sessions = self._active_tmux_sessions()
        candidate_paths = {
            job_id: record_path
            for job_id, record_path in self._indexed_rows(active_only=True)
        }

        # Bootstrap pre-index jobs from live tmux sessions only. This gives old
        # active jobs an upgrade path without touching terminal history.
        for session in sessions:
            job_id = session[len(self.SESSION_PREFIX) :]
            if self.JOB_ID_RE.fullmatch(job_id):
                candidate_paths.setdefault(
                    job_id,
                    self._relative(self._record_path(job_id)),
                )

        recovered = 0
        orphaned = 0
        for job_id, record_path in sorted(candidate_paths.items()):
            job = self._read_record(job_id, record_path)
            if job is None:
                continue
            job_type = str(job.get("job_type") or "")
            if job_type not in self.handlers:
                self._index_job(job, force=True)
                continue
            if job.get("status") not in self.ACTIVE_STATUSES:
                self._index_job(job, force=True)
                continue

            handler = self.handlers[job_type]
            with self.lock:
                self.jobs[job_id] = job
            on_loaded = handler["on_loaded"]
            if on_loaded:
                on_loaded(job)

            exit_code = self._read_exit_code(job_id)
            session = str(job.get("tmux_session") or "")
            if exit_code is not None:
                self._index_job(job, force=True)
                self._start_monitor(job_id, handler["on_poll"], handler["on_finished"])
                recovered += 1
                continue

            if session and session in sessions:
                job["tmux_state"] = "reattached"
                job["reattached_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                job["supervisor_status"] = "reattached"
                self.persist(job)
                self._start_monitor(job_id, handler["on_poll"], handler["on_finished"])
                recovered += 1
                continue

            job["status"] = "failed"
            job["tmux_state"] = "missing"
            job["supervisor_status"] = "orphaned"
            job["error"] = job.get("error") or "Recorded tmux session is no longer active"
            job["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            self.persist(job)
            on_finished = handler["on_finished"]
            if on_finished:
                on_finished(job, None, job["error"])
            orphaned += 1

        return {
            "indexed_active": len(candidate_paths),
            "live_sessions": len(sessions),
            "recovered": recovered,
            "orphaned": orphaned,
        }

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            current = self.jobs.get(job_id)
        if current is not None:
            return dict(current)
        rows = self._indexed_rows()
        record_path = next((path for key, path in rows if key == job_id), None)
        if record_path is None:
            raise KeyError(job_id)
        payload = self._read_record(job_id, record_path)
        if payload is None:
            raise KeyError(job_id)
        return dict(payload)

    def list(
        self,
        job_type: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        jobs: dict[str, dict[str, Any]] = {}
        for job_id, record_path in self._indexed_rows(job_type=job_type, status=status):
            payload = self._read_record(job_id, record_path)
            if payload is not None:
                jobs[job_id] = payload
        with self.lock:
            for job_id, payload in self.jobs.items():
                jobs[job_id] = dict(payload)
        values = list(jobs.values())
        if job_type:
            values = [job for job in values if job.get("job_type") == job_type]
        if status:
            values = [job for job in values if job.get("status") == status]
        values.sort(
            key=lambda job: str(
                job.get("submitted_at")
                or job.get("tmux_created_at")
                or job.get("job_id")
            ),
            reverse=True,
        )
        return [dict(job) for job in values]

    def rebuild_history_index(self) -> dict[str, int]:
        """Explicit maintenance operation; never called during startup."""
        scanned = 0
        indexed = 0
        self.registry_root.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.registry_root.glob("*/job.json")):
            scanned += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            job_id = str(payload.get("job_id") or "")
            if not self.JOB_ID_RE.fullmatch(job_id):
                continue
            self._index_job(payload, force=True)
            indexed += 1
        return {"scanned": scanned, "indexed": indexed}


__all__ = ["IndexedTmuxJobSupervisor"]
