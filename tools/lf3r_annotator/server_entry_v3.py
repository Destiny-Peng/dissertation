#!/usr/bin/env python3
"""LF3R annotator entrypoint with multi-manifest support and non-blocking video compatibility.

Builds on server_entry_v2 so the ProcVLM LoRA, project tools, live baseline
progress, and cancellation patches remain intact.

Key differences from the experimental main-branch implementation:
- multiple manifests are merged into a cached aggregate without replacing the
  primary manifest used by rollout generation;
- browser-compatibility transcoding never runs on the HTTP request thread;
- large multi-manifest queues are exposed with provenance metadata.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import threading
import uuid
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import server_entry_v2  # noqa: F401  # Apply all previous WebUI compatibility patches first.
import server


VIDEO_CACHE_VARIANT = "h264-baseline-v2-async"


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


class AsyncVideoCompatibilityCache:
    """Prepare H.264 browser copies in a background thread.

    The request path always returns immediately with either the already-cached
    H.264 file or the original source. Browser-side retry logic reloads the
    video after conversion finishes if the original codec cannot be decoded.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.cache_root = self.project_root / "cache" / "lf3r_annotator" / "video_cache"
        self.lock = threading.Lock()
        self.encode_slot = threading.Semaphore(1)
        self.states: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _now() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    def _key(self, path: Path) -> tuple[str, Path]:
        stat = path.stat()
        key = hashlib.sha256(
            f"{VIDEO_CACHE_VARIANT}:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:24]
        return key, self.cache_root / f"{key}.mp4"

    def request_path(self, path: Path) -> Path:
        if not path.is_file():
            return path
        try:
            key, cached = self._key(path)
        except OSError:
            return path
        if cached.is_file():
            try:
                if cached.stat().st_size > 0:
                    with self.lock:
                        self.states[key] = {
                            "status": "ready",
                            "codec": "h264",
                            "cached": str(cached),
                            "updated_at": self._now(),
                        }
                    return cached
            except OSError:
                pass

        with self.lock:
            state = self.states.get(key)
            if state and state.get("status") == "ready":
                ready_path = Path(str(state.get("cached") or cached))
                if ready_path.is_file():
                    return ready_path
            if not state or state.get("status") in {"failed", "stale"}:
                self.states[key] = {
                    "status": "queued",
                    "source": str(path),
                    "cached": str(cached),
                    "updated_at": self._now(),
                }
                threading.Thread(
                    target=self._prepare,
                    args=(key, path, cached),
                    name=f"lf3r-video-cache-{key}",
                    daemon=True,
                ).start()
        return path

    def status(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {"status": "missing"}
        try:
            key, cached = self._key(path)
        except OSError:
            return {"status": "missing"}
        if cached.is_file():
            try:
                if cached.stat().st_size > 0:
                    return {
                        "status": "ready",
                        "codec": "h264",
                        "cached": str(cached),
                    }
            except OSError:
                pass
        with self.lock:
            state = dict(self.states.get(key) or {})
        if not state:
            return {"status": "idle"}
        state.pop("source", None)
        state.pop("cached", None)
        return state

    def _set_state(self, key: str, **updates: Any) -> None:
        with self.lock:
            state = dict(self.states.get(key) or {})
            state.update(updates)
            state["updated_at"] = self._now()
            self.states[key] = state

    def _prepare(self, key: str, source: Path, cached: Path) -> None:
        with self.encode_slot:
            self._set_state(key, status="probing")
            try:
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name", "-of", "json", str(source),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                payload = json.loads(probe.stdout) if probe.returncode == 0 else {}
                streams = payload.get("streams") if isinstance(payload, dict) else None
                codec = str((streams or [{}])[0].get("codec_name", ""))
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError, IndexError, AttributeError) as exc:
                self._set_state(key, status="failed", error=f"ffprobe failed: {exc}")
                return

            if codec in {"", "h264"}:
                self._set_state(key, status="passthrough", codec=codec or "unknown")
                return

            self.cache_root.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_root / f".{key}.{uuid.uuid4().hex}.tmp.mp4"
            self._set_state(key, status="transcoding", codec=codec)
            try:
                result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-v", "error", "-i", str(source),
                        "-an", "-c:v", "libx264", "-profile:v", "baseline",
                        "-level", "3.1", "-preset", "veryfast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-threads", "2",
                        "-movflags", "+faststart", str(temporary),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                if result.returncode == 0 and temporary.is_file() and temporary.stat().st_size > 0:
                    os.replace(temporary, cached)
                    self._set_state(key, status="ready", codec="h264")
                    return
                error = (result.stderr or result.stdout or f"ffmpeg exited {result.returncode}").strip()
                self._set_state(key, status="failed", error=error[-2000:])
            except (OSError, subprocess.SubprocessError) as exc:
                self._set_state(key, status="failed", error=str(exc))
            finally:
                try:
                    if temporary.exists():
                        temporary.unlink()
                except OSError:
                    pass


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
            else self.project_root / "cache" / "lf3r_annotator" / "manifests" / f"manifest-{digest}.jsonl"
        )
        self._manifest_catalog_lock = threading.RLock()
        self._manifest_catalog_signature: tuple[tuple[str, bool, int, int], ...] | None = None
        self._manifest_records_cache: list[dict[str, Any]] = []
        self._manifest_info_cache: list[dict[str, Any]] = []
        self._manifest_source_by_id: dict[str, Path] = {}
        self._refresh_manifest_catalog(force=True)

        super().__init__(self.project_root, self.aggregate_manifest_path, annotation_root)

        # Rollout generation still owns and updates the primary natural manifest.
        self.rollout_jobs.manifest_path = self.primary_manifest_path
        # Analysis remains scoped to the established primary dataset.
        if hasattr(self.analysis, "manifest_path"):
            self.analysis.manifest_path = self.primary_manifest_path
        if hasattr(self.analysis_jobs, "manifest_path"):
            self.analysis_jobs.manifest_path = self.primary_manifest_path

        self.video_compat = AsyncVideoCompatibilityCache(self.project_root)

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
    parsed = urlparse(self.path)
    path = unquote(parsed.path)
    query = parse_qs(parsed.query, keep_blank_values=True)

    if path == "/api/manifests":
        self.json_response(
            HTTPStatus.OK,
            {
                "manifests": self.app.manifest_info(),
                "primary_manifest": self.app._relative_manifest_path(self.app.primary_manifest_path),
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
            {"rollouts": records, "manifests": self.app.manifest_info()},
        )
        return

    if path.startswith("/api/video-compatibility/"):
        rollout_id = path.rsplit("/", 1)[-1]
        rollout = self.app.rollout_map().get(rollout_id)
        if not rollout:
            self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
            return
        video = self.app.resolve_project_file(rollout["video_path"], ".mp4")
        self.json_response(
            HTTPStatus.OK,
            {"rollout_id": rollout_id, **self.app.video_compat.status(video)},
        )
        return

    if path.startswith("/api/videos/"):
        rollout_id = path.rsplit("/", 1)[-1]
        rollout = self.app.rollout_map().get(rollout_id)
        if not rollout:
            self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
            return
        video = self.app.resolve_project_file(rollout["video_path"], ".mp4")
        serve_path = self.app.video_compat.request_path(video)
        self.serve_video(serve_path)
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
        help="Manifest to load; repeat this option to show multiple manifests",
    )
    parser.add_argument("--annotations", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = args.project_root.resolve()
    default_manifest = project_root / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
    manifest_paths = list(args.manifest_paths or [default_manifest])
    if args.manifest_paths is None:
        optional_realrobot = project_root / "datasets/lf3r_failure_rollouts/v1/realrobot_manifest.jsonl"
        if optional_realrobot.is_file():
            manifest_paths.append(optional_realrobot)
    annotations = args.annotations or project_root / "annotations/failure_annotations/v1"

    app = MultiManifestApplication(project_root, manifest_paths, annotations)
    http_server = ThreadingHTTPServer((args.host, args.port), server.make_handler(app))
    print(f"LF3R annotator: http://{args.host}:{http_server.server_port}")
    for source_path in app.manifest_paths:
        print(f"Manifest source: {source_path}")
    if app.aggregate_manifest_path != app.primary_manifest_path:
        print(f"Aggregate manifest: {app.aggregate_manifest_path}")
    print(f"Annotations: {annotations}")
    print("Video compatibility: asynchronous H.264 cache (non-blocking request path)")

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


if __name__ == "__main__":
    main()
