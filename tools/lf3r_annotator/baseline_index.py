"""Persistent baseline-run metadata index."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Mapping

from backend_core import ValidationError


class BaselineRunIndex:
    """Persistent catalog for baseline run metadata.

    The filesystem remains authoritative. The index only replaces repeated
    recursive discovery under outputs/baselines. A full recursive scan is done
    once when no catalog exists, or explicitly through rebuild().
    """

    def __init__(
        self,
        project_root: Path,
        baseline_root: Path,
    ) -> None:
        self.project_root = project_root.resolve()
        self.baseline_root = baseline_root.resolve()
        self.path = (
            self.project_root
            / "cache"
            / "lf3r_annotator"
            / "baseline_runs.sqlite3"
        )
        self.lock = threading.RLock()
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=5.0,
        )
        connection.execute(
            "PRAGMA busy_timeout = 5000"
        )
        return connection

    def _initialize(self) -> None:
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS baseline_run_index_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS baseline_runs (
                    run_root TEXT PRIMARY KEY,
                    baseline TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sort_time TEXT NOT NULL,
                    selected_rollouts INTEGER NOT NULL,
                    metadata_mtime_ns INTEGER NOT NULL,
                    metadata_size INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS baseline_runs_method_status
                ON baseline_runs (
                    baseline,
                    status,
                    sort_time DESC
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS baseline_run_rollout_cache (
                    run_root TEXT PRIMARY KEY,
                    jobs_mtime_ns INTEGER NOT NULL,
                    jobs_size INTEGER NOT NULL,
                    rollout_ids_json TEXT NOT NULL,
                    robo_signal_ids_json TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _relative_run_root(
        self,
        run_path: Path,
    ) -> str:
        resolved = run_path.resolve()
        try:
            resolved.relative_to(
                self.baseline_root
            )
            return str(
                resolved.relative_to(
                    self.project_root
                )
            )
        except ValueError as error:
            raise ValidationError(
                "Baseline run index path must be inside outputs/baselines"
            ) from error

    @staticmethod
    def _selected_rollouts(
        metadata: dict[str, Any],
    ) -> int:
        try:
            return int(
                metadata.get(
                    "selected_rollouts"
                )
                or 0
            )
        except (TypeError, ValueError):
            return 0

    def _built(self) -> bool:
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM baseline_run_index_meta "
                "WHERE key = 'scan_complete'"
            ).fetchone()
        return bool(row and row[0] == "1")

    def ensure_built(self) -> None:
        if not self._built():
            self.rebuild()

    def upsert(
        self,
        run_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        run_path = run_path.resolve()
        metadata_path = run_path / "run.json"
        if not metadata_path.is_file():
            return False
        relative_run_root = self._relative_run_root(
            run_path
        )
        try:
            stat = metadata_path.stat()
        except OSError:
            return False

        with self.lock, self._connect() as connection:
            existing = connection.execute(
                """
                SELECT metadata_mtime_ns, metadata_size
                FROM baseline_runs
                WHERE run_root = ?
                """,
                (relative_run_root,),
            ).fetchone()
            if (
                existing is not None
                and int(existing[0])
                == stat.st_mtime_ns
                and int(existing[1])
                == stat.st_size
            ):
                return False

        if metadata is None:
            try:
                value = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )
            except (
                OSError,
                json.JSONDecodeError,
            ):
                return False
            if not isinstance(value, dict):
                return False
            metadata = value

        baseline = str(
            metadata.get("baseline") or ""
        ).strip()
        status = str(
            metadata.get("status") or ""
        ).strip()
        if not baseline:
            return False
        sort_time = str(
            metadata.get("completed_at")
            or metadata.get("created_at")
            or ""
        )
        payload = json.dumps(
            metadata,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO baseline_runs (
                    run_root,
                    baseline,
                    status,
                    sort_time,
                    selected_rollouts,
                    metadata_mtime_ns,
                    metadata_size,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_root) DO UPDATE SET
                    baseline = excluded.baseline,
                    status = excluded.status,
                    sort_time = excluded.sort_time,
                    selected_rollouts = excluded.selected_rollouts,
                    metadata_mtime_ns = excluded.metadata_mtime_ns,
                    metadata_size = excluded.metadata_size,
                    metadata_json = excluded.metadata_json
                """,
                (
                    relative_run_root,
                    baseline,
                    status,
                    sort_time,
                    self._selected_rollouts(
                        metadata
                    ),
                    stat.st_mtime_ns,
                    stat.st_size,
                    payload,
                ),
            )
        return True

    def rebuild(self) -> dict[str, int]:
        rows: list[
            tuple[
                str,
                str,
                str,
                str,
                int,
                int,
                int,
                str,
            ]
        ] = []
        scanned = 0
        if self.baseline_root.is_dir():
            for metadata_path in (
                self.baseline_root.rglob(
                    "run.json"
                )
            ):
                scanned += 1
                try:
                    run_path = (
                        metadata_path
                        .parent
                        .resolve()
                    )
                    relative_run_root = (
                        self._relative_run_root(
                            run_path
                        )
                    )
                    stat = (
                        metadata_path.stat()
                    )
                    metadata = json.loads(
                        metadata_path.read_text(
                            encoding="utf-8"
                        )
                    )
                except (
                    OSError,
                    json.JSONDecodeError,
                    ValidationError,
                ):
                    continue
                if not isinstance(
                    metadata,
                    dict,
                ):
                    continue
                baseline = str(
                    metadata.get(
                        "baseline"
                    )
                    or ""
                ).strip()
                if not baseline:
                    continue
                status = str(
                    metadata.get(
                        "status"
                    )
                    or ""
                ).strip()
                sort_time = str(
                    metadata.get(
                        "completed_at"
                    )
                    or metadata.get(
                        "created_at"
                    )
                    or ""
                )
                rows.append(
                    (
                        relative_run_root,
                        baseline,
                        status,
                        sort_time,
                        self._selected_rollouts(
                            metadata
                        ),
                        stat.st_mtime_ns,
                        stat.st_size,
                        json.dumps(
                            metadata,
                            ensure_ascii=False,
                            separators=(
                                ",",
                                ":",
                            ),
                        ),
                    )
                )

        with self.lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM baseline_runs"
            )
            if rows:
                connection.executemany(
                    """
                    INSERT INTO baseline_runs (
                        run_root,
                        baseline,
                        status,
                        sort_time,
                        selected_rollouts,
                        metadata_mtime_ns,
                        metadata_size,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            connection.execute(
                """
                INSERT INTO baseline_run_index_meta (
                    key,
                    value
                )
                VALUES ('scan_complete', '1')
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value
                """
            )
            connection.execute(
                """
                INSERT INTO baseline_run_index_meta (
                    key,
                    value
                )
                VALUES ('last_scan_at', ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value
                """,
                (
                    dt.datetime.now(
                        dt.timezone.utc
                    ).isoformat(),
                ),
            )
        return {
            "scanned": scanned,
            "indexed": len(rows),
        }

    def cached_rollout_details(
        self,
        run_path: Path,
    ) -> dict[str, Any] | None:
        """Return cached rollout/signal inventory when jobs.jsonl is unchanged."""
        run_path = run_path.resolve()
        jobs_path = run_path / "jobs.jsonl"
        if not jobs_path.is_file():
            return None
        try:
            stat = jobs_path.stat()
            run_root = self._relative_run_root(
                run_path
            )
        except (OSError, ValidationError):
            return None

        with self.lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    jobs_mtime_ns,
                    jobs_size,
                    rollout_ids_json,
                    robo_signal_ids_json
                FROM baseline_run_rollout_cache
                WHERE run_root = ?
                """,
                (run_root,),
            ).fetchone()

        if row is None:
            return None
        if (
            int(row[0]) != stat.st_mtime_ns
            or int(row[1]) != stat.st_size
        ):
            return None

        try:
            rollout_ids = json.loads(row[2])
            robo_signal_ids = (
                json.loads(row[3])
                if row[3]
                else None
            )
        except (
            TypeError,
            json.JSONDecodeError,
        ):
            return None

        if (
            not isinstance(
                rollout_ids,
                list,
            )
            or not all(
                isinstance(value, str)
                for value in rollout_ids
            )
        ):
            return None
        if (
            robo_signal_ids is not None
            and not isinstance(
                robo_signal_ids,
                dict,
            )
        ):
            return None

        return {
            "rollout_ids": rollout_ids,
            "robo_signal_ids": robo_signal_ids,
        }

    def store_rollout_details(
        self,
        run_path: Path,
        rollout_ids: set[str],
        robo_signal_ids: (
            Mapping[str, set[str]] | None
        ) = None,
    ) -> bool:
        """Persist rollout inventory so catalog reads avoid raw-tree probing."""
        run_path = run_path.resolve()
        jobs_path = run_path / "jobs.jsonl"
        if not jobs_path.is_file():
            return False
        try:
            stat = jobs_path.stat()
            run_root = self._relative_run_root(
                run_path
            )
        except (OSError, ValidationError):
            return False

        signal_payload = (
            json.dumps(
                {
                    str(mode): sorted(
                        set(ids)
                    )
                    for mode, ids
                    in robo_signal_ids.items()
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if robo_signal_ids
            is not None
            else None
        )

        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO baseline_run_rollout_cache (
                    run_root,
                    jobs_mtime_ns,
                    jobs_size,
                    rollout_ids_json,
                    robo_signal_ids_json,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_root) DO UPDATE SET
                    jobs_mtime_ns = excluded.jobs_mtime_ns,
                    jobs_size = excluded.jobs_size,
                    rollout_ids_json = excluded.rollout_ids_json,
                    robo_signal_ids_json = excluded.robo_signal_ids_json,
                    updated_at = excluded.updated_at
                """,
                (
                    run_root,
                    stat.st_mtime_ns,
                    stat.st_size,
                    json.dumps(
                        sorted(
                            set(
                                rollout_ids
                            )
                        ),
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":",
                        ),
                    ),
                    signal_payload,
                    dt.datetime.now(
                        dt.timezone.utc
                    ).isoformat(),
                ),
            )
        return True

    def clear_rollout_cache(self) -> None:
        with self.lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM "
                "baseline_run_rollout_cache"
            )

    def _delete_roots(
        self,
        run_roots: list[str],
    ) -> None:
        if not run_roots:
            return
        with self.lock, self._connect() as connection:
            connection.executemany(
                "DELETE FROM baseline_runs "
                "WHERE run_root = ?",
                [
                    (run_root,)
                    for run_root
                    in run_roots
                ],
            )
            connection.executemany(
                "DELETE FROM "
                "baseline_run_rollout_cache "
                "WHERE run_root = ?",
                [
                    (run_root,)
                    for run_root
                    in run_roots
                ],
            )

    def candidates(
        self,
        method: str,
        statuses: set[str],
    ) -> list[
        tuple[
            Path,
            dict[str, Any],
        ]
    ]:
        self.ensure_built()
        if not statuses:
            return []

        placeholders = ",".join(
            "?" for _ in statuses
        )
        parameters: list[Any] = [
            method,
            *sorted(statuses),
        ]
        query = f"""
            SELECT
                run_root,
                metadata_mtime_ns,
                metadata_size,
                metadata_json
            FROM baseline_runs
            WHERE baseline = ?
              AND status IN ({placeholders})
            ORDER BY
                sort_time DESC,
                selected_rollouts DESC,
                run_root DESC
        """

        with self.lock, self._connect() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        result: list[
            tuple[
                Path,
                dict[str, Any],
            ]
        ] = []
        remove_roots: list[str] = []

        for (
            run_root,
            mtime_ns,
            size,
            metadata_json,
        ) in rows:
            run_path = (
                self.project_root
                / str(run_root)
            ).resolve()
            metadata_path = (
                run_path / "run.json"
            )
            if not metadata_path.is_file():
                remove_roots.append(
                    str(run_root)
                )
                continue
            try:
                stat = metadata_path.stat()
            except OSError:
                continue

            metadata: (
                dict[str, Any] | None
            ) = None
            if (
                stat.st_mtime_ns
                == int(mtime_ns)
                and stat.st_size
                == int(size)
            ):
                try:
                    cached = json.loads(
                        metadata_json
                    )
                except json.JSONDecodeError:
                    cached = None
                if isinstance(cached, dict):
                    metadata = cached
            else:
                try:
                    current = json.loads(
                        metadata_path.read_text(
                            encoding="utf-8"
                        )
                    )
                except (
                    OSError,
                    json.JSONDecodeError,
                ):
                    try:
                        cached = json.loads(
                            metadata_json
                        )
                    except json.JSONDecodeError:
                        cached = None
                    if isinstance(
                        cached,
                        dict,
                    ):
                        metadata = cached
                else:
                    if isinstance(
                        current,
                        dict,
                    ):
                        self.upsert(
                            run_path,
                            current,
                        )
                        metadata = current
                    else:
                        remove_roots.append(
                            str(run_root)
                        )

            if metadata is None:
                continue
            if (
                metadata.get("baseline")
                != method
            ):
                continue
            if (
                str(
                    metadata.get(
                        "status"
                    )
                    or ""
                )
                not in statuses
            ):
                continue
            result.append(
                (
                    run_path,
                    metadata,
                )
            )

        self._delete_roots(
            remove_roots
        )
        result.sort(
            key=lambda item: (
                str(
                    item[1].get(
                        "completed_at"
                    )
                    or item[1].get(
                        "created_at"
                    )
                    or ""
                ),
                self._selected_rollouts(
                    item[1]
                ),
                str(item[0]),
            ),
            reverse=True,
        )
        return result
