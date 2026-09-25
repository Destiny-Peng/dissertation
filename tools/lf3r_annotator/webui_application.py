#!/usr/bin/env python3
"""Multi-manifest WebUI application composition."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import server
from non_analysis_tools import NonAnalysisToolService
from webui_baseline import WebUIBaselineService


def _atomic_jsonl_write(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class WebUIApplication(server.LF3RApplication):
    """LF3R WebUI application exposing several manifests as one catalog."""

    baseline_service_class = WebUIBaselineService
    project_tool_service_class = NonAnalysisToolService

    def __init__(
        self,
        project_root: Path,
        manifest_paths: list[Path],
        annotation_root: Path,
        analysis_python: Path | None = None,
        tmux_binary: str | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        resolved: list[Path] = []
        for raw in manifest_paths:
            candidate = Path(raw).expanduser()
            path = (
                candidate.resolve()
                if candidate.is_absolute()
                else (self.project_root / candidate).resolve()
            )
            if path not in resolved:
                resolved.append(path)
        if not resolved:
            raise ValueError("At least one manifest path is required")

        self.manifest_paths = sorted(resolved, key=lambda path: str(path))
        source_key = "\0".join(str(path) for path in self.manifest_paths)
        digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
        self.aggregate_manifest_path = (
            self.project_root
            / "cache"
            / "lf3r_annotator"
            / "manifests"
            / f"catalog-{digest}.jsonl"
        )
        self.canonical_manifest_path = (
            self.project_root
            / "datasets"
            / "lf3r_failure_rollouts"
            / "v1"
            / "manifest.jsonl"
        )
        self._manifest_catalog_lock = threading.RLock()
        self._manifest_catalog_signature: (
            tuple[tuple[str, bool, int, int], ...] | None
        ) = None
        self._manifest_records_cache: list[dict[str, Any]] = []
        self._manifest_info_cache: list[dict[str, Any]] = []
        self._manifest_source_by_id: dict[str, Path] = {}
        self._refresh_manifest_catalog(force=True)

        super().__init__(
            self.project_root,
            self.aggregate_manifest_path,
            annotation_root,
            analysis_python=analysis_python,
            tmux_binary=tmux_binary,
        )

        # Catalog sources are read-only peers. LIBERO rollout generation owns
        # the canonical rollout manifest explicitly; rebuild_manifest uses the
        # same canonical target in non_analysis_tools.py.
        self.rollout_jobs.manifest_path = self.canonical_manifest_path

    def _relative_manifest_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)

    def _manifest_source_signature(self) -> tuple[tuple[str, bool, int, int], ...]:
        signature: list[tuple[str, bool, int, int]] = []
        for source_path in self.manifest_paths:
            try:
                stat = source_path.stat()
            except OSError:
                signature.append((str(source_path), False, 0, 0))
                continue
            signature.append(
                (
                    str(source_path),
                    source_path.is_file(),
                    stat.st_mtime_ns,
                    stat.st_size,
                )
            )
        return tuple(signature)

    def _validate_manifest_rows(
        self,
        source_path: Path,
        rows: list[dict[str, Any]],
    ) -> None:
        source_label = self._relative_manifest_path(source_path)
        for row in rows:
            rollout_id = str(row.get("id") or "")
            camera_paths = row.get("camera_video_paths")
            if not isinstance(camera_paths, dict) or not camera_paths:
                raise server.ValidationError(
                    f"{source_label}: rollout {rollout_id} has no camera_video_paths"
                )
            for camera, value in camera_paths.items():
                if not isinstance(camera, str) or not camera.strip():
                    raise server.ValidationError(
                        f"{source_label}: rollout {rollout_id} has an invalid camera key"
                    )
                if not isinstance(value, str) or not value.strip():
                    raise server.ValidationError(
                        f"{source_label}: rollout {rollout_id} has an invalid camera path"
                    )
                self.resolve_project_file(value, ".mp4")

    def _refresh_manifest_catalog(
        self,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        # While generation owns the manifest writer, keep serving the last
        # stable aggregate snapshot. The next refresh after writer release will
        # incorporate newly generated rollouts.
        coordinator = getattr(self, "job_coordinator", None)
        if not force and coordinator is not None:
            active = coordinator.active()
            if any(role == "manifest_writer" for role in active.values()):
                with self._manifest_catalog_lock:
                    return [dict(row) for row in self._manifest_records_cache]

        with self._manifest_catalog_lock:
            signature = self._manifest_source_signature()
            if not force and signature == self._manifest_catalog_signature:
                return [dict(row) for row in self._manifest_records_cache]

            records: list[dict[str, Any]] = []
            aggregate_rows: list[dict[str, Any]] = []
            source_info_by_path: dict[Path, dict[str, Any]] = {}
            parsed_rows: dict[Path, list[dict[str, Any]]] = {}

            for source_path in self.manifest_paths:
                source_exists = source_path.is_file()
                source_relative = self._relative_manifest_path(source_path)
                try:
                    source_rows = (
                        server.load_manifest_records(source_path)
                        if source_exists
                        else []
                    )
                    if not source_rows:
                        raise server.ValidationError(
                            f"{source_relative}: manifest has no rollout records"
                        )
                    self._validate_manifest_rows(source_path, source_rows)
                except (OSError, json.JSONDecodeError, server.ValidationError) as error:
                    source_info_by_path[source_path] = {
                        "path": source_relative,
                        "label": source_path.stem,
                        "exists": source_exists,
                        "valid": False,
                        "rollouts": 0,
                        "error": str(error),
                    }
                    continue

                parsed_rows[source_path] = source_rows
                source_info_by_path[source_path] = {
                    "path": source_relative,
                    "label": source_path.stem,
                    "exists": source_exists,
                    "valid": True,
                    "rollouts": len(source_rows),
                    "error": None,
                }

            sources_by_rollout_id: dict[str, list[Path]] = {}
            for source_path, source_rows in parsed_rows.items():
                for row in source_rows:
                    sources_by_rollout_id.setdefault(str(row["id"]), []).append(source_path)

            conflicting_sources: dict[Path, list[str]] = {}
            for rollout_id, source_paths in sources_by_rollout_id.items():
                if len(source_paths) < 2:
                    continue
                for source_path in source_paths:
                    conflicting_sources.setdefault(source_path, []).append(rollout_id)

            for source_path, rollout_ids in conflicting_sources.items():
                source_info_by_path[source_path].update(
                    {
                        "valid": False,
                        "rollouts": 0,
                        "error": (
                            "Duplicate rollout ids across peer manifests: "
                            + ", ".join(sorted(rollout_ids)[:8])
                            + (" ..." if len(rollout_ids) > 8 else "")
                        ),
                    }
                )

            source_by_id: dict[str, Path] = {}
            for source_path in self.manifest_paths:
                if source_path not in parsed_rows or source_path in conflicting_sources:
                    continue
                source_relative = self._relative_manifest_path(source_path)
                for row in parsed_rows[source_path]:
                    rollout_id = str(row["id"])
                    source_by_id[rollout_id] = source_path
                    aggregate_rows.append(row)
                    enriched = dict(row)
                    enriched["manifest_source"] = source_relative
                    enriched["manifest_label"] = source_path.stem
                    records.append(enriched)

            source_info = [
                source_info_by_path[path]
                for path in self.manifest_paths
                if path in source_info_by_path
            ]
            _atomic_jsonl_write(self.aggregate_manifest_path, aggregate_rows)

            self._manifest_records_cache = records
            self._manifest_info_cache = source_info
            self._manifest_source_by_id = source_by_id
            self._manifest_catalog_signature = self._manifest_source_signature()
            return [dict(row) for row in records]

    def refresh_manifest_catalog(
        self,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        return self._refresh_manifest_catalog(force=force)

    def manifest_info(self) -> list[dict[str, Any]]:
        self._refresh_manifest_catalog()
        return [dict(item) for item in self._manifest_info_cache]

    def dataset_groups(self) -> dict[str, Any]:
        records = self._refresh_manifest_catalog()
        suite_counts: dict[str, int] = {}
        controlled = 0
        for record in records:
            if server._is_controlled_record(record):
                controlled += 1
                continue
            suite = str(record.get("task_suite") or "").strip()
            if suite:
                suite_counts[suite] = suite_counts.get(suite, 0) + 1
        return {
            "task_suites": [
                {"value": suite, "count": count}
                for suite, count in suite_counts.items()
            ],
            "controlled_count": controlled,
            "total_count": len(records),
        }

    def load_rollouts(self) -> list[dict[str, Any]]:
        records = self._refresh_manifest_catalog()
        for record in records:
            camera_paths = record.get("camera_video_paths")
            if not isinstance(camera_paths, dict) or not camera_paths:
                raise server.ValidationError(
                    "Manifest record has invalid camera_video_paths: "
                    + str(record.get("id"))
                )
            for camera, value in camera_paths.items():
                if not isinstance(camera, str) or not camera.strip():
                    raise server.ValidationError(
                        "Manifest record has an invalid camera key: "
                        + str(record.get("id"))
                    )
                if not isinstance(value, str) or not value.strip():
                    raise server.ValidationError(
                        "Manifest record has an invalid camera video path: "
                        + str(record.get("id"))
                    )
                self.resolve_project_file(value, ".mp4")
        return records


def discover_default_manifests(project_root: Path) -> list[Path]:
    """Return all top-level rollout manifests as peer catalog sources."""
    manifest_dir = project_root / "datasets/lf3r_failure_rollouts/v1"
    manifests = sorted(
        (path for path in manifest_dir.glob("*manifest*.jsonl") if path.is_file()),
        key=lambda path: path.name,
    )
    if manifests:
        return manifests
    return [manifest_dir / "manifest.jsonl"]
