#!/usr/bin/env python3
"""Multi-manifest application and raw video serving for the WebUI.

Loaded after ``webui_jobs`` so integrations, project tools, live progress, and
cancellation behavior are already installed.

Runtime video probing/transcoding is deliberately absent. ``/api/videos/<id>``
serves canonical ``video_path`` by default; ``?camera=<slot>`` explicitly
serves one declared ``camera_video_paths`` entry. Browser-incompatible
canonical videos can be converted explicitly through the manual WebUI
project-tool action.

Dataset scopes are discovered from the currently loaded manifests. Any
non-controlled ``task_suite`` value becomes a valid suite scope automatically;
``controlled_analysis`` and ``all`` remain the two reserved aggregate scopes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import tempfile
import threading
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import webui_jobs  # noqa: F401  # Install integrations and job extensions first.
import non_analysis_tools
import server


_DYNAMIC_SCOPE_RE = re.compile(r"^[A-Za-z0-9._-]{1,160}$")
_RESERVED_SCOPES = {"all", "controlled_analysis"}


def _is_controlled_record(record: dict[str, Any]) -> bool:
    return (
        record.get("analysis_partition") == "controlled_analysis"
        or record.get("source_kind") == "controlled_injected"
    )


def _validate_dynamic_run_scope(scope: Any) -> str:
    value = str(scope or "all").strip()
    if value in _RESERVED_SCOPES:
        return value
    if not _DYNAMIC_SCOPE_RE.fullmatch(value):
        raise server.ValidationError(
            "scope must be 'all', 'controlled_analysis', or a task_suite name "
            "present in the loaded manifests"
        )
    return value


def _record_matches_dynamic_scope(record: dict[str, Any], scope: str) -> bool:
    scope = _validate_dynamic_run_scope(scope)
    if scope == "all":
        return True
    if scope == "controlled_analysis":
        return _is_controlled_record(record)
    return str(record.get("task_suite") or "") == scope and not _is_controlled_record(record)


def _select_dynamic_scope_records(
    records: list[dict[str, Any]], scope: str
) -> list[dict[str, Any]]:
    scope = _validate_dynamic_run_scope(scope)
    return [record for record in records if _record_matches_dynamic_scope(record, scope)]


# Replace the legacy fixed LIBERO scope list. Existing server services resolve
# these globals at call time, so Baseline and Analysis immediately gain support
# for any task_suite loaded from a manifest.
server.validate_run_scope = _validate_dynamic_run_scope
server.record_matches_scope = _record_matches_dynamic_scope
server.select_scope_records = _select_dynamic_scope_records


_original_scoped_baseline_command = server.BaselineService._baseline_command


def _remove_cli_pairs(command: list[str], flags: set[str]) -> list[str]:
    cleaned: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token in flags:
            index += 2
            continue
        cleaned.append(token)
        index += 1
    return cleaned


def _baseline_command_with_dynamic_suite(
    self: server.BaselineService,
    baseline: str,
    scope: str,
    gpu: str,
    utilization: float,
    run_parent: Path,
    options: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> list[str]:
    command = _original_scoped_baseline_command(
        self,
        baseline,
        scope,
        gpu,
        utilization,
        run_parent,
        options,
        *args,
        **kwargs,
    )

    instruction_condition = kwargs.get("instruction_condition")
    if instruction_condition is None and len(args) > 6:
        instruction_condition = args[6]
    instruction_condition = str(instruction_condition or "full_instruction")
    # Base command already emitted an explicit rollout-ID selection for
    # instruction variants or result-coverage filtering. Do not replace it
    # with the entire dynamic suite here.
    if (
        instruction_condition != "full_instruction"
        or scope in _RESERVED_SCOPES
        or kwargs.get("rollout_ids") is not None
    ):
        return command

    # A suite scope is defined by the actual selected manifest records, not by a
    # fixed partition assumption. This is important for real-robot manifests,
    # whose analysis_partition need not be natural_observation. Repeated
    # --rollout-id filters preserve the exact scope and keep controlled injected
    # records separate even when they share the same task_suite.
    selected = self._condition_records("full_instruction", scope)
    command = _remove_cli_pairs(command, {"--partition", "--task-suite", "--rollout-id"})
    command.extend(["--partition", "all"])
    for record in selected:
        command.extend(["--rollout-id", str(record["id"])])
    return command


server.BaselineService._baseline_command = _baseline_command_with_dynamic_suite


def _atomic_jsonl_write(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _manual_transcode_command(payload: dict[str, Any]) -> list[str]:
    value = str(payload.get("video_path") or "").strip()
    if not value:
        raise ValueError("video_path is required")
    video = non_analysis_tools.project_path(value)
    if video.suffix.lower() != ".mp4":
        raise ValueError("video_path must point to an .mp4 file")
    if not video.is_file():
        raise ValueError(f"video does not exist: {value}")
    backup = video.with_name(video.name[:-4] + ".orig.mp4")
    if backup.exists():
        raise ValueError(
            "original backup already exists; refusing to overwrite: "
            + str(backup.relative_to(non_analysis_tools.PROJECT_ROOT))
        )
    script = non_analysis_tools.PROJECT_ROOT / "tools/lf3r_annotator/transcode_video_h264.sh"
    return ["/usr/bin/bash", str(script), str(video)]


# Extend the existing persistent project-tool service. The tool runs in its own
# tmux job, not in an HTTP request thread, so ffmpeg cannot block server shutdown.
non_analysis_tools.TOOL_BUILDERS["transcode_video"] = _manual_transcode_command
non_analysis_tools.TOOL_LABELS["transcode_video"] = "Transcode selected video to H.264"


class MultiManifestApplication(server.LF3RApplication):
    """LF3R application that exposes several manifests as one catalog."""

    def __init__(
        self,
        project_root: Path,
        manifest_paths: list[Path],
        annotation_root: Path,
    ) -> None:
        self.project_root = project_root.resolve()
        resolved: list[Path] = []
        for raw in manifest_paths:
            path = Path(raw).expanduser().resolve()
            if path not in resolved:
                resolved.append(path)
        if not resolved:
            raise ValueError("At least one manifest path is required")

        self.manifest_paths = resolved
        self.primary_manifest_path = resolved[0]
        source_key = "\0".join(str(path) for path in resolved)
        digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
        self.aggregate_manifest_path = (
            self.primary_manifest_path
            if len(resolved) == 1
            else self.project_root
            / "cache"
            / "lf3r_annotator"
            / "manifests"
            / f"manifest-{digest}.jsonl"
        )
        self._manifest_catalog_lock = threading.RLock()
        self._manifest_catalog_signature: tuple[tuple[str, bool, int, int], ...] | None = None
        self._manifest_records_cache: list[dict[str, Any]] = []
        self._manifest_info_cache: list[dict[str, Any]] = []
        self._manifest_source_by_id: dict[str, Path] = {}
        self._refresh_manifest_catalog(force=True)

        super().__init__(self.project_root, self.aggregate_manifest_path, annotation_root)

        # Rollout generation owns and updates only the primary manifest. Baseline
        # and Analysis services keep the aggregate manifest supplied above so
        # newly loaded task suites, including real-robot data, are discoverable.
        self.rollout_jobs.manifest_path = self.primary_manifest_path

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
                (str(source_path), source_path.is_file(), stat.st_mtime_ns, stat.st_size)
            )
        return tuple(signature)

    def _refresh_manifest_catalog(self, force: bool = False) -> list[dict[str, Any]]:
        with self._manifest_catalog_lock:
            signature = self._manifest_source_signature()
            if not force and signature == self._manifest_catalog_signature:
                return [dict(row) for row in self._manifest_records_cache]

            records: list[dict[str, Any]] = []
            aggregate_rows: list[dict[str, Any]] = []
            source_info: list[dict[str, Any]] = []
            source_by_id: dict[str, Path] = {}
            for source_path in self.manifest_paths:
                source_exists = source_path.is_file()
                source_rows = server.load_manifest_records(source_path) if source_exists else []
                source_info.append(
                    {
                        "path": self._relative_manifest_path(source_path),
                        "label": source_path.stem,
                        "primary": source_path == self.primary_manifest_path,
                        "exists": source_exists,
                        "rollouts": len(source_rows),
                    }
                )
                for row in source_rows:
                    rollout_id = str(row["id"])
                    previous = source_by_id.get(rollout_id)
                    if previous is not None:
                        raise server.ValidationError(
                            "Duplicate rollout id across manifests: "
                            + rollout_id
                            + " ("
                            + self._relative_manifest_path(previous)
                            + " and "
                            + self._relative_manifest_path(source_path)
                            + ")"
                        )
                    source_by_id[rollout_id] = source_path
                    aggregate_rows.append(row)
                    enriched = dict(row)
                    enriched["manifest_source"] = self._relative_manifest_path(source_path)
                    enriched["manifest_label"] = source_path.stem
                    enriched["manifest_primary"] = source_path == self.primary_manifest_path
                    records.append(enriched)

            if self.aggregate_manifest_path != self.primary_manifest_path:
                _atomic_jsonl_write(self.aggregate_manifest_path, aggregate_rows)

            self._manifest_records_cache = records
            self._manifest_info_cache = source_info
            self._manifest_source_by_id = source_by_id
            self._manifest_catalog_signature = self._manifest_source_signature()
            return [dict(row) for row in records]

    def refresh_manifest_catalog(self, force: bool = False) -> list[dict[str, Any]]:
        return self._refresh_manifest_catalog(force=force)

    def manifest_info(self) -> list[dict[str, Any]]:
        self._refresh_manifest_catalog()
        return [dict(item) for item in self._manifest_info_cache]

    def dataset_groups(self) -> dict[str, Any]:
        records = self._refresh_manifest_catalog()
        suite_counts: dict[str, int] = {}
        controlled = 0
        for record in records:
            if _is_controlled_record(record):
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
            video_path = record.get("video_path")
            if not isinstance(video_path, str) or not video_path:
                raise server.ValidationError(
                    "Manifest record has an invalid video_path: " + str(record.get("id"))
                )
            self.resolve_project_file(video_path, ".mp4")
        return records


_previous_do_get = server.LF3RHandler.do_GET


def _do_get_with_multi_manifest(self: server.LF3RHandler) -> None:
    path = unquote(urlparse(self.path).path)

    if path == "/api/manifests":
        self.json_response(
            HTTPStatus.OK,
            {
                "manifests": self.app.manifest_info(),
                "primary_manifest": self.app._relative_manifest_path(self.app.primary_manifest_path),
                "dataset_groups": self.app.dataset_groups(),
            },
        )
        return

    if path == "/api/rollouts":
        records = []
        for record in self.app.load_rollouts():
            annotation = self.app.store.read(record["id"])
            enriched = {
                **record,
                "annotation": annotation,
                "annotation_status": annotation["review_status"] if annotation else "unreviewed",
            }
            options = self.app.instruction_variant_options(record)
            enriched["instruction_variants"] = options
            enriched["instruction_variant_conditions"] = list(options)
            if self.app.instruction_variant_manifest_path.is_file():
                enriched["instruction_variant_manifest"] = str(
                    self.app.instruction_variant_manifest_path.relative_to(self.app.project_root)
                )
            records.append(enriched)
        self.json_response(
            HTTPStatus.OK,
            {
                "rollouts": records,
                "manifests": self.app.manifest_info(),
                "dataset_groups": self.app.dataset_groups(),
            },
        )
        return

    if path.startswith("/api/videos/"):
        rollout_id = path.rsplit("/", 1)[-1]
        rollout = self.app.rollout_map().get(rollout_id)
        if not rollout:
            self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
            return
        query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
        camera = str(query.get("camera", [""])[0] or "").strip()
        if camera:
            camera_paths = rollout.get("camera_video_paths")
            value = camera_paths.get(camera) if isinstance(camera_paths, dict) else None
            if not isinstance(value, str) or not value:
                self.json_error(HTTPStatus.NOT_FOUND, "Camera video is unavailable")
                return
            video = self.app.resolve_project_file(value, ".mp4")
            if not video.is_file():
                self.json_error(HTTPStatus.NOT_FOUND, "Camera video file is unavailable")
                return
        else:
            video = self.app.resolve_project_file(rollout["video_path"], ".mp4")
        self.serve_video(video)
        return

    return _previous_do_get(self)


server.LF3RHandler.do_GET = _do_get_with_multi_manifest

_previous_do_post = server.LF3RHandler.do_POST


def _do_post_with_manifest_refresh(self: server.LF3RHandler) -> None:
    path = unquote(urlparse(self.path).path)
    if path == "/api/baselines/run-batch" or path.startswith("/api/baselines/run/"):
        self.app.refresh_manifest_catalog()
    return _previous_do_post(self)


server.LF3RHandler.do_POST = _do_post_with_manifest_refresh


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the LF3R annotation tool locally")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; keep loopback for SSH forwarding")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project-root", type=Path, default=server.DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--manifest",
        dest="manifest_paths",
        type=Path,
        action="append",
        help=(
            "Explicit manifest to load; repeat to select several. When omitted, "
            "all top-level *manifest*.jsonl files in datasets/lf3r_failure_rollouts/v1 are loaded."
        ),
    )
    parser.add_argument("--annotations", type=Path)
    return parser.parse_args()


def _discover_default_manifests(project_root: Path) -> list[Path]:
    """Return the standard manifest first, then every other active manifest beside it."""
    manifest_dir = project_root / "datasets/lf3r_failure_rollouts/v1"
    primary = manifest_dir / "manifest.jsonl"
    others = sorted(
        (
            path
            for path in manifest_dir.glob("*manifest*.jsonl")
            if path.is_file() and path.resolve() != primary.resolve()
        ),
        key=lambda path: path.name,
    )
    return [primary, *others]


def main() -> None:
    args = _parse_args()
    project_root = args.project_root.resolve()
    manifest_paths = (
        list(args.manifest_paths)
        if args.manifest_paths is not None
        else _discover_default_manifests(project_root)
    )
    annotations = args.annotations or project_root / "annotations/failure_annotations/v1"

    app = MultiManifestApplication(project_root, manifest_paths, annotations)
    http_server = ThreadingHTTPServer((args.host, args.port), server.make_handler(app))
    print(f"LF3R annotator: http://{args.host}:{http_server.server_port}")
    print(f"Project root: {project_root}")
    if args.manifest_paths is None:
        print("Manifest discovery: datasets/lf3r_failure_rollouts/v1/*manifest*.jsonl")
    for source_path in app.manifest_paths:
        print(f"Manifest source: {source_path}")
    if app.aggregate_manifest_path != app.primary_manifest_path:
        print(f"Aggregate manifest: {app.aggregate_manifest_path}")
    groups = app.dataset_groups()
    suites = ", ".join(
        f"{item['value']} ({item['count']})" for item in groups["task_suites"]
    ) or "none"
    print(f"Dataset task suites: {suites}")
    print(f"Controlled rollouts: {groups['controlled_count']}")
    print(f"Annotations: {annotations}")
    print("Video serving: canonical by default; ?camera=<name> serves manifest camera_video_paths; runtime transcoding disabled")

    def stop_server(_signum: int, _frame: Any) -> None:
        print("Shutdown requested; stopping LF3R annotator...")
        threading.Thread(target=http_server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http_server.server_close()
