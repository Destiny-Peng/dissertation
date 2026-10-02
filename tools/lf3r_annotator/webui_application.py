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
from repair import RepairService
from tactile_service import FailRecoveryTactileService
from webui_baseline import WebUIBaselineService


MANIFEST_CACHE_SCHEMA_VERSION = 1


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


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
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
    repair_service_class = RepairService

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
        self.aggregate_manifest_meta_path = self.aggregate_manifest_path.with_suffix(
            ".meta.json"
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

        # Normal startup is a cache lookup. Full parsing/validation only runs
        # when one of the source manifest signatures actually changes.
        self._refresh_manifest_catalog(force=False)

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
        self.tactile = FailRecoveryTactileService(self.project_root)

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

    @staticmethod
    def _signature_payload(
        signature: tuple[tuple[str, bool, int, int], ...]
    ) -> list[list[Any]]:
        return [list(item) for item in signature]

    def _validate_manifest_rows(
        self,
        source_path: Path,
        rows: list[dict[str, Any]],
    ) -> None:
        """Validate manifest structure without resolving every video on disk."""
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
                candidate = Path(value)
                if candidate.is_absolute() or ".." in candidate.parts:
                    raise server.ValidationError(
                        f"{source_label}: rollout {rollout_id} camera path escapes project root"
                    )
                if candidate.suffix.lower() != ".mp4":
                    raise server.ValidationError(
                        f"{source_label}: rollout {rollout_id} camera path is not an mp4"
                    )

    def _cache_records(
        self,
        *,
        signature: tuple[tuple[str, bool, int, int], ...],
        aggregate_rows: list[dict[str, Any]],
        source_info: list[dict[str, Any]],
        source_by_id: dict[str, Path],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in aggregate_rows:
            rollout_id = str(row.get("id") or "")
            source_path = source_by_id.get(rollout_id)
            if source_path is None:
                raise server.ValidationError(
                    f"Cached aggregate is missing source mapping for {rollout_id}"
                )
            source_relative = self._relative_manifest_path(source_path)
            enriched = dict(row)
            enriched["manifest_source"] = source_relative
            enriched["manifest_label"] = source_path.stem
            records.append(enriched)

        self._manifest_records_cache = records
        self._manifest_info_cache = [dict(item) for item in source_info]
        self._manifest_source_by_id = dict(source_by_id)
        self._manifest_catalog_signature = signature
        return [dict(row) for row in records]

    def _write_manifest_cache_meta(
        self,
        *,
        signature: tuple[tuple[str, bool, int, int], ...],
        source_info: list[dict[str, Any]],
        source_by_id: dict[str, Path],
    ) -> None:
        _atomic_json_write(
            self.aggregate_manifest_meta_path,
            {
                "schema_version": MANIFEST_CACHE_SCHEMA_VERSION,
                "signature": self._signature_payload(signature),
                "source_info": source_info,
                "source_by_id": {
                    rollout_id: self._relative_manifest_path(source_path)
                    for rollout_id, source_path in source_by_id.items()
                },
            },
        )

    def _restore_manifest_catalog(
        self,
        signature: tuple[tuple[str, bool, int, int], ...],
    ) -> list[dict[str, Any]] | None:
        if not self.aggregate_manifest_path.is_file() or not self.aggregate_manifest_meta_path.is_file():
            return None
        try:
            metadata = json.loads(
                self.aggregate_manifest_meta_path.read_text(encoding="utf-8")
            )
            if metadata.get("schema_version") != MANIFEST_CACHE_SCHEMA_VERSION:
                return None
            if metadata.get("signature") != self._signature_payload(signature):
                return None
            source_info = metadata.get("source_info")
            raw_source_by_id = metadata.get("source_by_id")
            if not isinstance(source_info, list) or not isinstance(raw_source_by_id, dict):
                return None
            aggregate_rows = server.load_manifest_records(self.aggregate_manifest_path)
            source_by_id: dict[str, Path] = {}
            for rollout_id, raw_path in raw_source_by_id.items():
                if not isinstance(rollout_id, str) or not isinstance(raw_path, str):
                    return None
                candidate = Path(raw_path)
                source_by_id[rollout_id] = (
                    candidate
                    if candidate.is_absolute()
                    else self.project_root / candidate
                )
            return self._cache_records(
                signature=signature,
                aggregate_rows=aggregate_rows,
                source_info=source_info,
                source_by_id=source_by_id,
            )
        except (OSError, json.JSONDecodeError, server.ValidationError, TypeError):
            return None

    def _bootstrap_legacy_aggregate(
        self,
        signature: tuple[tuple[str, bool, int, int], ...],
    ) -> list[dict[str, Any]] | None:
        """Trust a newer existing aggregate once and create the sidecar cache.

        This avoids one expensive legacy startup immediately after upgrading.
        Source manifests are parsed only to recover source-id mapping; camera
        paths are not resolved and the aggregate is not rewritten.
        """
        if not self.aggregate_manifest_path.is_file():
            return None
        if any(not exists for _path, exists, _mtime, _size in signature):
            return None
        try:
            aggregate_stat = self.aggregate_manifest_path.stat()
        except OSError:
            return None
        newest_source = max((mtime for _path, _exists, mtime, _size in signature), default=0)
        if aggregate_stat.st_mtime_ns < newest_source:
            return None

        parsed_rows: dict[Path, list[dict[str, Any]]] = {}
        source_info: list[dict[str, Any]] = []
        source_by_id: dict[str, Path] = {}
        try:
            for source_path in self.manifest_paths:
                rows = server.load_manifest_records(source_path)
                if not rows:
                    return None
                parsed_rows[source_path] = rows
                source_info.append(
                    {
                        "path": self._relative_manifest_path(source_path),
                        "label": source_path.stem,
                        "exists": True,
                        "valid": True,
                        "rollouts": len(rows),
                        "error": None,
                    }
                )
                for row in rows:
                    rollout_id = str(row.get("id") or "")
                    if not rollout_id or rollout_id in source_by_id:
                        return None
                    source_by_id[rollout_id] = source_path

            aggregate_rows = server.load_manifest_records(self.aggregate_manifest_path)
            aggregate_ids = {str(row.get("id") or "") for row in aggregate_rows}
            if aggregate_ids != set(source_by_id):
                return None
            self._write_manifest_cache_meta(
                signature=signature,
                source_info=source_info,
                source_by_id=source_by_id,
            )
            return self._cache_records(
                signature=signature,
                aggregate_rows=aggregate_rows,
                source_info=source_info,
                source_by_id=source_by_id,
            )
        except (OSError, json.JSONDecodeError, server.ValidationError, TypeError):
            return None

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

            if not force:
                restored = self._restore_manifest_catalog(signature)
                if restored is not None:
                    return restored
                bootstrapped = self._bootstrap_legacy_aggregate(signature)
                if bootstrapped is not None:
                    return bootstrapped

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
            self._write_manifest_cache_meta(
                signature=signature,
                source_info=source_info,
                source_by_id=source_by_id,
            )

            self._manifest_records_cache = records
            self._manifest_info_cache = source_info
            self._manifest_source_by_id = source_by_id
            self._manifest_catalog_signature = signature
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
        role_counts: dict[str, int] = {}
        controlled = 0
        for record in records:
            role = str(record.get("dataset_role") or "")
            if role:
                role_counts[role] = role_counts.get(role, 0) + 1
            if server._is_controlled_record(record):
                controlled += 1
                continue
            suite = str(record.get("task_suite") or "").strip()
            if suite:
                suite_counts[suite] = suite_counts.get(suite, 0) + 1
        return {
            "dataset_roles": [
                {"value": role, "count": count}
                for role, count in role_counts.items()
            ],
            "task_suites": [
                {"value": suite, "count": count}
                for suite, count in suite_counts.items()
            ],
            "controlled_count": controlled,
            "total_count": len(records),
        }

    def load_rollouts(self) -> list[dict[str, Any]]:
        # Structural validation is performed only when source manifests change.
        # Normal requests reuse the in-memory/persistent catalog directly.
        return self._refresh_manifest_catalog()


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
