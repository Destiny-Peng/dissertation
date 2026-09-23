#!/usr/bin/env python3
"""Compatibility facade and standalone entrypoint for the LF3R annotator backend.

Business logic lives in focused backend modules. This module intentionally
re-exports the historical public symbols so existing tests, scripts, and WebUI
extensions can continue to import server without depending on file layout.
"""

from __future__ import annotations

import argparse
import signal
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from analysis_constants import (
    ANALYSIS_ARTIFACT_NAMES,
    ANALYSIS_BASELINE_METHODS,
    ANALYSIS_DETAIL_FILTERS,
    ANALYSIS_DETAIL_KINDS,
    ANALYSIS_DETAIL_SORTS,
    ANALYSIS_EVENT_ARRAY_FIELDS,
    ANALYSIS_EVENT_FIELDS,
    ANALYSIS_LOCALIZATION_EVENT_FIELDS,
    ANALYSIS_LOCALIZATION_TABLE_FILES,
    ANALYSIS_RECORD_FIELDS,
    ANALYSIS_TABLE_FILES,
    CHANGEPOINT_EVENT_FIELDS,
    CHANGEPOINT_TABLE_FILES,
    EVENT_TRIGGERED_TABLE_FILES,
    ROBO_HOP_EXTENDED_FILES,
    ROBO_HOP_REQUIRED_FILES,
    ROBO_HOP_TABLE_FILES,
    ROBO_LABEL_LOSS_REQUIRED_FILES,
    ROBO_LOCALIZATION_HEAD_REQUIRED_FILES,
)
from analysis_jobs import AnalysisJobService
from analysis_service import AnalysisService
from application import LF3RApplication
from backend_core import (
    AnalysisEnvironmentError,
    INSTRUCTION_VARIANT_CONDITIONS,
    JobConflictError,
    JobCoordinator,
    ROLLOUT_ID_RE,
    RESERVED_RUN_SCOPES,
    ValidationError,
    _is_controlled_record,
    atomic_json_write,
    load_manifest_records,
    record_matches_scope,
    run_rollout_ids,
    select_scope_records,
    validate_instruction_condition,
    validate_run_scope,
)
from baseline_constants import (
    BASELINE_ADVANCED_FIELDS,
    BASELINE_LABELS,
    BASELINE_METHOD_OPTION_FIELDS,
    BASELINE_METHODS,
    BASELINE_RESULT_FILTERS,
    BASELINE_RUN_STATUSES,
    INSTRUCTION_VARIANT_LABELS,
)
from baseline_index import BaselineRunIndex
from baseline_service import BaselineService
from http_handler import CONTROL_PREFIX_REQUEST_RE, LF3RHandler
from rollout_service import (
    GENERATION_CONFIGS,
    GENERATION_LABEL_RE,
    GENERATION_RUN_PREFIX,
    GENERATION_VIDEO_RE,
    RolloutGenerationService,
)
from stores import (
    AnnotationStore,
    DEFAULT_SETTINGS,
    FAILURE_TYPES,
    OUTCOME_LABELS,
    REVIEW_STATUSES,
    SETTING_COLOR_FIELDS,
    SETTING_COLOR_RE,
    SETTING_DENSITIES,
    SettingsStore,
    optional_frame,
    validate_annotation,
    validate_failure_event,
)
from task_supervisor import TmuxJobSupervisor, TmuxSupervisorError


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_handler(app: LF3RApplication) -> type[LF3RHandler]:
    class BoundHandler(LF3RHandler):
        pass

    BoundHandler.app = app
    return BoundHandler


def parse_args() -> argparse.Namespace:
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
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--annotations", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    manifest = (
        args.manifest
        or project_root
        / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
    )
    annotations = (
        args.annotations
        or project_root
        / "annotations/failure_annotations/v1"
    )
    app = LF3RApplication(
        project_root,
        manifest,
        annotations,
    )
    http_server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(app),
    )
    print(
        f"LF3R annotator: "
        f"http://{args.host}:{http_server.server_port}"
    )
    print(f"Manifest: {manifest}")
    print(f"Annotations: {annotations}")

    def stop_server(
        _signum: int,
        _frame: Any,
    ) -> None:
        print(
            "Shutdown requested; "
            "stopping LF3R annotator..."
        )
        threading.Thread(
            target=http_server.shutdown,
            daemon=True,
        ).start()

    signal.signal(
        signal.SIGTERM,
        stop_server,
    )
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http_server.server_close()


if __name__ == "__main__":
    main()
