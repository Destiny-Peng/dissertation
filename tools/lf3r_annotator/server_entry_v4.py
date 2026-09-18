#!/usr/bin/env python3
"""LF3R annotator entrypoint with concurrent generation and progressive baseline results.

Builds on ``server_entry_v3``. Rollout generation still has exclusive ownership
of manifest *writes* relative to another rollout-generation job, but it no
longer blocks independent baseline/analysis compute for the full generation
lifetime. While a generation job is active, the multi-manifest catalog is held
at its last stable snapshot; newly generated rollouts become visible after the
writer finishes and the next catalog refresh runs.

Baseline runs are also readable while they are still running. Only rollout
outputs whose per-rollout job record is already ``complete`` are exposed; the
currently executing rollout is never parsed from a partially written raw file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import server
import server_entry_v3


def _acquire_concurrent_manifest_writer(
    self: server.JobCoordinator,
    job_id: str,
    role: str = "compute",
) -> None:
    """Serialize manifest writers only; allow independent compute concurrently."""
    with self.lock:
        if role == "manifest_writer" and any(
            active_role == "manifest_writer"
            for active_role in self.active_jobs.values()
        ):
            raise server.JobConflictError(
                "Another rollout-generation job is already rebuilding the manifest"
            )
        self.active_jobs[job_id] = role


server.JobCoordinator.acquire = _acquire_concurrent_manifest_writer


_original_refresh_manifest_catalog = (
    server_entry_v3.MultiManifestApplication._refresh_manifest_catalog
)


def _refresh_manifest_catalog_from_stable_snapshot(
    self: server_entry_v3.MultiManifestApplication,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Do not rebuild the aggregate catalog while generation owns the manifest writer.

    Baselines launched during generation therefore read the same stable manifest
    snapshot that existed before generation began. The primary manifest may be
    rebuilt by the generator in the background, but the aggregate manifest used
    by Baseline/Analysis is not touched until the writer releases its role.
    """
    coordinator = getattr(self, "job_coordinator", None)
    if not force and coordinator is not None:
        active = coordinator.active()
        if any(role == "manifest_writer" for role in active.values()):
            with self._manifest_catalog_lock:
                return [dict(row) for row in self._manifest_records_cache]
    return _original_refresh_manifest_catalog(self, force=force)


server_entry_v3.MultiManifestApplication._refresh_manifest_catalog = (
    _refresh_manifest_catalog_from_stable_snapshot
)


# A finished rollout is authoritative as soon as its worker records status=complete.
# Persistent ProcVLM / Robo-Dopamine workers write worker-local jobs.jsonl files
# during a run, while the generic/Rynn runner may also update the run-level file.
def _progressive_run_rollout_ids(run_path: Path) -> set[str]:
    result: set[str] = set()
    paths = [run_path / "jobs.jsonl"]
    workers_root = run_path / "workers"
    if workers_root.is_dir():
        paths.extend(sorted(workers_root.glob("worker-*/jobs.jsonl")))

    for jobs_path in paths:
        if not jobs_path.is_file():
            continue
        try:
            with jobs_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # A reader can race the final append of one JSONL row.
                        # Ignore only that incomplete row and keep prior results.
                        continue
                    if row.get("status") != "complete":
                        continue
                    rollout_id = row.get("rollout_id", row.get("id"))
                    if (
                        isinstance(rollout_id, str)
                        and server.ROLLOUT_ID_RE.fullmatch(rollout_id)
                    ):
                        result.add(rollout_id)
        except OSError:
            continue
    return result


server.run_rollout_ids = _progressive_run_rollout_ids

_PROGRESSIVE_RUN_STATUSES = set(server.BASELINE_RUN_STATUSES) | {"running"}


def _progressive_run_candidates(
    self: server.BaselineService,
    method: str,
) -> list[tuple[Path, dict[str, Any]]]:
    return self._indexed_run_candidates(method, _PROGRESSIVE_RUN_STATUSES)


server.BaselineService._run_candidates = _progressive_run_candidates


def _progressive_explicit_run_candidate(
    self: server.BaselineService,
    method: str,
    run_root: Any,
    allowed_conditions: set[str],
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(run_root, (str, Path)) or not str(run_root).strip():
        raise server.ValidationError(
            f"{method} run selection must be a project-relative run path"
        )
    path = self._project_path(str(run_root))
    try:
        path.relative_to(self.baseline_root.resolve())
    except ValueError as exc:
        raise server.ValidationError(
            f"Selected {method} run must be inside outputs/baselines"
        ) from exc
    metadata_path = path / "run.json"
    if not metadata_path.is_file():
        raise server.ValidationError(f"Selected {method} run.json is missing")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise server.ValidationError(
            f"Selected {method} run metadata is invalid"
        ) from exc
    if metadata.get("baseline") != method:
        raise server.ValidationError(f"Selected run is not a {method} run")
    if metadata.get("status") not in _PROGRESSIVE_RUN_STATUSES:
        raise server.ValidationError(
            f"Selected {method} run has no readable completed rollout results"
        )
    condition = self._run_instruction_condition(metadata)
    if condition not in allowed_conditions:
        allowed = ", ".join(sorted(allowed_conditions))
        raise server.ValidationError(
            f"Selected {method} run is for instruction condition {condition!r}; "
            f"expected {allowed}"
        )
    return path, metadata


server.BaselineService._explicit_run_candidate = _progressive_explicit_run_candidate


_original_read_method = server.BaselineService._read_method


def _read_only_completed_progressive_rollout(
    self: server.BaselineService,
    method: str,
    run_path: Path,
    rollout: dict[str, Any],
    run_summary: dict[str, Any],
) -> dict[str, Any]:
    # A running worker may already have created the raw output file for the
    # rollout it is still writing. Do not expose it until the worker has emitted
    # the authoritative per-rollout complete record.
    if run_summary.get("status") == "running":
        rollout_id = str(rollout.get("id") or "")
        if rollout_id not in server.run_rollout_ids(run_path):
            raise FileNotFoundError(run_path / "raw" / rollout_id)
    return _original_read_method(self, method, run_path, rollout, run_summary)


server.BaselineService._read_method = _read_only_completed_progressive_rollout


def main() -> None:
    server_entry_v3.main()


if __name__ == "__main__":
    main()
