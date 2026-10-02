#!/usr/bin/env python3
"""Compose and run the LF3R WebUI."""

from __future__ import annotations

import argparse
import signal
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import server
from webui_application import WebUIApplication, discover_default_manifests
from webui_handler import WebUIHandler
from webui_rollout import WebUIRolloutGenerationService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the LF3R annotation tool locally"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address; keep loopback for SSH forwarding",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=server.DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument(
        "--manifest",
        dest="manifest_paths",
        type=Path,
        action="append",
        help=(
            "Explicit manifest to load; repeat to select several. When omitted, "
            "all top-level *manifest*.jsonl files in "
            "datasets/lf3r_failure_rollouts/v1 are loaded."
        ),
    )
    parser.add_argument("--annotations", type=Path)
    return parser.parse_args()


def _make_handler(app: WebUIApplication) -> type[WebUIHandler]:
    class BoundWebUIHandler(WebUIHandler):
        pass

    BoundWebUIHandler.app = app
    return BoundWebUIHandler


def main() -> None:
    startup_started = time.perf_counter()
    args = _parse_args()
    project_root = args.project_root.resolve()

    discovery_started = time.perf_counter()
    manifest_paths = (
        [
            path.expanduser().resolve()
            if path.expanduser().is_absolute()
            else (project_root / path.expanduser()).resolve()
            for path in args.manifest_paths
        ]
        if args.manifest_paths is not None
        else discover_default_manifests(project_root)
    )
    discovery_seconds = time.perf_counter() - discovery_started
    annotations = (
        args.annotations
        or project_root / "annotations/failure_annotations/v1"
    )

    # Keep the core application default on the historical LIBERO service; the
    # WebUI explicitly opts into the additive ManiSkill3-aware subclass.
    WebUIApplication.rollout_service_class = WebUIRolloutGenerationService
    print(
        f"[startup] manifest discovery {discovery_seconds:.3f}s; "
        f"initializing application with {len(manifest_paths)} manifest(s)...",
        flush=True,
    )
    app_started = time.perf_counter()
    app = WebUIApplication(
        project_root,
        manifest_paths,
        annotations,
    )
    app_seconds = time.perf_counter() - app_started
    print(f"[startup] application initialized in {app_seconds:.3f}s", flush=True)

    server_started = time.perf_counter()
    http_server = ThreadingHTTPServer(
        (args.host, args.port),
        _make_handler(app),
    )
    server_seconds = time.perf_counter() - server_started
    total_seconds = time.perf_counter() - startup_started

    print(
        f"LF3R annotator: http://{args.host}:{http_server.server_port}"
    )
    print(
        f"[startup] HTTP bind {server_seconds:.3f}s; total {total_seconds:.3f}s"
    )
    print(f"Project root: {project_root}")
    if args.manifest_paths is None:
        print(
            "Manifest discovery: "
            "datasets/lf3r_failure_rollouts/v1/*manifest*.jsonl"
        )
    for source_path in app.manifest_paths:
        print(f"Manifest source: {source_path}")
    print(f"Aggregate catalog: {app.aggregate_manifest_path}")
    groups = app.dataset_groups()
    suites = ", ".join(
        f"{item['value']} ({item['count']})"
        for item in groups["dataset_roles"]
    ) or "none"
    print(f"Dataset roles: {suites}")
    print(f"Controlled rollouts: {groups['controlled_count']}")
    print(f"Annotations: {annotations}")
    print(
        "Video serving: preferred available camera by default; "
        "?camera=<name> serves manifest camera_video_paths; "
        "runtime transcoding disabled"
    )

    def stop_server(_signum: int, _frame: Any) -> None:
        print("Shutdown requested; stopping LF3R annotator...")
        threading.Thread(
            target=http_server.shutdown,
            daemon=True,
        ).start()

    signal.signal(signal.SIGTERM, stop_server)
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http_server.server_close()
