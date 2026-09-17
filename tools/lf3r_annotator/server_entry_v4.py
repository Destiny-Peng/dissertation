#!/usr/bin/env python3
"""LF3R annotator entrypoint with concurrent rollout generation and baseline runs.

Builds on ``server_entry_v3``. Rollout generation still has exclusive ownership
of manifest *writes* relative to another rollout-generation job, but it no
longer blocks independent baseline/analysis compute for the full generation
lifetime. While a generation job is active, the multi-manifest catalog is held
at its last stable snapshot; newly generated rollouts become visible after the
writer finishes and the next catalog refresh runs.
"""

from __future__ import annotations

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


def main() -> None:
    server_entry_v3.main()


if __name__ == "__main__":
    main()
