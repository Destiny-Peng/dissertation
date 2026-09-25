#!/usr/bin/env python3
"""Compose and run the LF3R WebUI."""

from __future__ import annotations

import argparse
import signal
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import server
from webui_application import WebUIApplication, discover_default_manifests
from webui_handler import WebUIHandler


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
    args = _parse_args()
    project_root = args.project_root.resolve()
    manifest_paths = (
        list(args.manifest_paths)
        if args.manifest_paths is not None
        else discover_default_manifests(project_root)
    )
    annotations = (
        args.annotations
        or project_root / "annotations/failure_annotations/v1"
    )

    app = WebUIApplication(
        project_root,
        manifest_paths,
        annotations,
    )
    http_server = ThreadingHTTPServer(
        (args.host, args.port),
        _make_handler(app),
    )

    print(
        f"LF3R annotator: http://{args.host}:{http_server.server_port}"
    )
    print(f"Project root: {project_root}")
    if args.manifest_paths is None:
        print(
            "Manifest discovery: "
            "datasets/lf3r_failure_rollouts/v1/*manifest*.jsonl"
        )
    for source_path in app.manifest_paths:
        print(f"Manifest source: {source_path}")
    if app.aggregate_manifest_path != app.primary_manifest_path:
        print(f"Aggregate manifest: {app.aggregate_manifest_path}")
    groups = app.dataset_groups()
    suites = ", ".join(
        f"{item['value']} ({item['count']})"
        for item in groups["task_suites"]
    ) or "none"
    print(f"Dataset task suites: {suites}")
    print(f"Controlled rollouts: {groups['controlled_count']}")
    print(f"Annotations: {annotations}")
    print(
        "Video serving: preferred manifest camera by default; "
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
