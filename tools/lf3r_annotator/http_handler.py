"""HTTP handler for the LF3R annotator backend."""

from __future__ import annotations

import datetime as dt
import json
import mimetypes
import re
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from application import LF3RApplication
from backend_core import (
    AnalysisEnvironmentError,
    JobConflictError,
    ValidationError,
)
from baseline_constants import (
    BASELINE_METHODS,
    INSTRUCTION_VARIANT_LABELS,
)
from task_supervisor import TmuxSupervisorError


CONTROL_PREFIX_REQUEST_RE = re.compile(
    rb"^(?P<prefix>[\x00-\x1f]{1,32})"
    rb"(?P<request>(?:GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH) [^\r\n]+ HTTP/1\.[01]\r?\n)$"
)


class LF3RHandler(BaseHTTPRequestHandler):
    app: LF3RApplication
    server_version = "LF3RAnnotator/1.0"
    protocol_version = "HTTP/1.0"
    CLIENT_DISCONNECT_ERRORS = (
        BrokenPipeError,
        ConnectionResetError,
        ConnectionAbortedError,
    )

    def handle(self) -> None:
        try:
            super().handle()
        except self.CLIENT_DISCONNECT_ERRORS:
            # Browsers routinely cancel stale fetch/video requests during
            # navigation and manifest refresh. The response can no longer be
            # delivered, but this is not a server-side failure.
            self.close_connection = True

    def _safe_end_headers(self) -> bool:
        self.close_connection = True
        try:
            self.end_headers()
        except self.CLIENT_DISCONNECT_ERRORS:
            return False
        return True

    def _safe_write(self, data: bytes) -> bool:
        try:
            self.wfile.write(data)
        except self.CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True
            return False
        return True

    @staticmethod
    def _sanitize_raw_requestline(raw: bytes) -> tuple[bytes, int]:
        match = CONTROL_PREFIX_REQUEST_RE.fullmatch(raw)
        if match is None:
            return raw, 0
        prefix = match.group("prefix")
        return match.group("request"), len(prefix)

    def parse_request(self) -> bool:
        sanitized, stripped = self._sanitize_raw_requestline(self.raw_requestline)
        if stripped:
            self.raw_requestline = sanitized
            method = sanitized.split(b" ", 1)[0].decode("ascii", errors="replace")
            self.log_error(
                "sanitized %d leading control byte(s) before HTTP method %s",
                stripped,
                method,
            )
        return super().parse_request()

    def log_message(self, fmt: str, *args: Any) -> None:
        super().log_message(fmt, *args)

    def json_response(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        if not self._safe_end_headers():
            return
        self._safe_write(body)

    def json_error(self, status: int, message: str) -> None:
        self.json_response(status, {"error": message})

    def file_response(self, path: Path) -> None:
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", 'attachment; filename="' + path.name + '"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        if not self._safe_end_headers():
            return
        self._safe_write(body)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            if path == "/api/health":
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "manifest": str(self.app.manifest_path.relative_to(self.app.project_root)),
                        "rollouts": len(self.app.load_rollouts()),
                        "tmux": {
                            "available": self.app.tmux.available,
                            "persistent": True,
                            "persistent_jobs": len(self.app.tmux.list()),
                            "error": None if self.app.tmux.available else "tmux is required for persistent annotator jobs",
                        },
                        "analysis_environment": self.app.analysis_jobs.environment_status(),
                    },
                )
                return
            if path == "/api/settings":
                self.json_response(HTTPStatus.OK, self.app.settings.response())
                return
            if path == "/api/analysis/localization/presets":
                self.json_response(
                    HTTPStatus.OK,
                    {"presets": self.app.analysis_jobs.localization_presets()},
                )
                return
            if path == "/api/analysis/localization":
                self.json_response(
                    HTTPStatus.OK,
                    {"localization": self.app.analysis_jobs.localization_results()},
                )
                return
            if path == "/api/analysis/localization/challenge-sets":
                self.json_response(
                    HTTPStatus.OK,
                    {"challenge_sets": self.app.analysis_jobs.localization_challenge_sets()},
                )
                return
            if path.startswith("/api/analysis/localization/result/"):
                run_name = path.rsplit("/", 1)[-1]
                try:
                    result = self.app.analysis_jobs.localization_result_detail(run_name)
                except FileNotFoundError:
                    self.json_error(HTTPStatus.NOT_FOUND, "Localization result not found")
                    return
                self.json_response(HTTPStatus.OK, {"result": result})
                return
            if path.startswith("/api/analysis/localization/challenge/"):
                run_name = path.rsplit("/", 1)[-1]
                self.json_response(
                    HTTPStatus.OK,
                    {"challenge": self.app.analysis_jobs.localization_challenge_data(run_name)},
                )
                return
            if path.startswith("/api/analysis/localization/artifacts/"):
                relative = path[len("/api/analysis/localization/artifacts/"):]
                parts = relative.split("/", 1)
                if len(parts) != 2:
                    self.json_error(HTTPStatus.NOT_FOUND, "Localization artifact not found")
                    return
                try:
                    artifact = self.app.analysis_jobs.localization_artifact(parts[0], parts[1])
                except FileNotFoundError:
                    self.json_error(HTTPStatus.NOT_FOUND, "Localization artifact not found")
                    return
                self.file_response(artifact)
                return
            if path == "/api/analysis":
                requested_view = (query.get("view", ["dashboard"])[0] or "dashboard").lower()
                compact = requested_view not in {"full", "legacy", "compatibility"}
                if query.get("include_details", [""])[0] in {"1", "true", "yes"}:
                    compact = False
                self.json_response(
                    HTTPStatus.OK,
                    {"analysis": self.app.analysis.response(compact=compact)},
                )
                return
            if path == "/api/analysis/robo-localization-head":
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "localization_head": (
                            self.app.analysis.robo_localization_head_response()
                        )
                    },
                )
                return
            if path.startswith("/api/analysis/robo-localization-head/artifacts/"):
                name = path.rsplit("/", 1)[-1]
                try:
                    artifact = self.app.analysis.robo_localization_head_artifact_path(name)
                except FileNotFoundError:
                    self.json_error(
                        HTTPStatus.NOT_FOUND,
                        "Localization-head artifact is unavailable",
                    )
                    return
                self.file_response(artifact)
                return
            if path == "/api/analysis/robo-label-loss":
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "label_loss": (
                            self.app.analysis.robo_label_loss_response()
                        )
                    },
                )
                return
            if path.startswith("/api/analysis/robo-label-loss/artifacts/"):
                name = path.rsplit("/", 1)[-1]
                try:
                    artifact = self.app.analysis.robo_label_loss_artifact_path(name)
                except FileNotFoundError:
                    self.json_error(
                        HTTPStatus.NOT_FOUND,
                        "Label/loss ablation artifact is unavailable",
                    )
                    return
                self.file_response(artifact)
                return
            if path == "/api/analysis/robo-hop":
                self.json_response(
                    HTTPStatus.OK,
                    {"robo_hop": self.app.analysis.robo_hop_response()},
                )
                return
            if path == "/api/analysis/details":
                detail_query = {
                    key: values[0] if values else ""
                    for key, values in query.items()
                }
                self.json_response(
                    HTTPStatus.OK,
                    self.app.analysis.details(detail_query),
                )
                return
            if path.startswith("/api/analysis/artifacts/"):
                name = path[len("/api/analysis/artifacts/"):]
                try:
                    artifact = self.app.analysis.artifact_path(name)
                except FileNotFoundError:
                    self.json_error(HTTPStatus.NOT_FOUND, "Analysis artifact is unavailable")
                    return
                self.file_response(artifact)
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
                self.json_response(HTTPStatus.OK, {"rollouts": records})
                return
            if path == "/api/baselines/result-coverage":
                baseline = query.get("baseline", [""])[0]
                scope = query.get("scope", ["libero_10"])[0]
                condition = query.get("condition", ["full_instruction"])[0]
                self.json_response(
                    HTTPStatus.OK,
                    self.app.baselines.result_coverage(
                        baseline,
                        scope,
                        condition,
                    ),
                )
                return
            if path == "/api/baselines/runs":
                scope = query.get("scope", ["libero_10"])[0]
                condition = query.get("condition", ["full_instruction"])[0]
                rescan = str(query.get("rescan", ["0"])[0]).lower() in {
                    "1", "true", "yes"
                }
                index_refresh = (
                    self.app.baselines.rebuild_run_index()
                    if rescan
                    else None
                )
                payload = {
                    "scope": scope,
                    "condition": condition,
                    "runs": self.app.baselines.list_runs(scope, condition),
                }
                if index_refresh is not None:
                    payload["index_refresh"] = index_refresh
                self.json_response(HTTPStatus.OK, payload)
                return
            if path == "/api/jobs":
                job_type = query.get("job_type", [None])[0] or None
                status = query.get("status", [None])[0] or None
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "jobs": self.app.tmux.list(job_type=job_type, status=status),
                        "tmux_available": self.app.tmux.available,
                    },
                )
                return
            if path.startswith("/api/rollout-jobs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[3] == "log":
                    job_id = parts[2]
                    tail = query.get("tail", ["200"])[0]
                    try:
                        log = self.app.rollout_jobs.log(job_id, tail)
                    except KeyError:
                        self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout-generation job")
                        return
                    self.json_response(HTTPStatus.OK, {"log": log})
                    return
                if len(parts) != 3:
                    self.json_error(HTTPStatus.NOT_FOUND, "Not found")
                    return
                job_id = parts[2]
                try:
                    job = self.app.rollout_jobs.job(job_id)
                except KeyError:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout-generation job")
                    return
                self.json_response(HTTPStatus.OK, {"job": job})
                return
            if path.startswith("/api/baseline-jobs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[3] == "log":
                    job_id = parts[2]
                    tail = query.get("tail", ["200"])[0]
                    try:
                        log = self.app.baselines.log(job_id, tail)
                    except KeyError:
                        self.json_error(HTTPStatus.NOT_FOUND, "Unknown baseline job")
                        return
                    self.json_response(HTTPStatus.OK, {"log": log})
                    return
                if len(parts) != 3:
                    self.json_error(HTTPStatus.NOT_FOUND, "Not found")
                    return
                job_id = parts[2]
                try:
                    job = self.app.baselines.job(job_id)
                except KeyError:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown baseline job")
                    return
                self.json_response(HTTPStatus.OK, {"job": job})
                return
            if path.startswith("/api/analysis-jobs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[3] == "log":
                    job_id = parts[2]
                    tail = query.get("tail", ["200"])[0]
                    try:
                        log = self.app.analysis_jobs.log(job_id, tail)
                    except KeyError:
                        self.json_error(HTTPStatus.NOT_FOUND, "Unknown analysis job")
                        return
                    self.json_response(HTTPStatus.OK, {"log": log})
                    return
                if len(parts) != 3:
                    self.json_error(HTTPStatus.NOT_FOUND, "Not found")
                    return
                job_id = parts[2]
                try:
                    job = self.app.analysis_jobs.job(job_id)
                except KeyError:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown analysis job")
                    return
                self.json_response(HTTPStatus.OK, {"job": job})
                return
            if path.startswith("/api/baselines/"):
                rollout_id = path.rsplit("/", 1)[-1]
                rollout = self.app.rollout_map().get(rollout_id)
                if not rollout:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                    return
                condition = query.get("condition", ["full_instruction"])[0] or "full_instruction"
                run_overrides = {
                    method: query.get("run_" + method, [""])[0]
                    for method in BASELINE_METHODS
                    if query.get("run_" + method, [""])[0]
                }
                variant = self.app.instruction_variant_for(rollout, condition)
                if variant is None:
                    evaluation = self.app.baselines.evaluation(
                        rollout,
                        condition=condition,
                        source_rollout_id=rollout_id,
                        variant_available=False,
                        unavailable_reason=(
                            "No prepared " + INSTRUCTION_VARIANT_LABELS[condition]
                            + " variant exists for this rollout."
                        ),
                        run_overrides=run_overrides,
                    )
                    evaluation.update({
                        "variant_id": None,
                        "instruction": None,
                        "instruction_type": None,
                    })
                else:
                    evaluation = self.app.baselines.evaluation(
                        variant,
                        condition=condition,
                        source_rollout_id=rollout_id,
                        variant_available=True,
                        run_overrides=run_overrides,
                    )
                    evaluation.update({
                        "variant_id": variant["id"],
                        "instruction": variant.get("task_description") or variant.get("instruction", ""),
                        "instruction_type": variant.get("instruction_type"),
                    })
                self.json_response(HTTPStatus.OK, {"evaluation": evaluation})
                return
            if path.startswith("/api/annotations/"):
                rollout_id = path.rsplit("/", 1)[-1]
                if rollout_id not in self.app.rollout_map():
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                    return
                self.json_response(
                    HTTPStatus.OK,
                    {"annotation": self.app.store.read(rollout_id)},
                )
                return
            if path.startswith("/api/videos/"):
                rollout_id = path.rsplit("/", 1)[-1]
                rollout = self.app.rollout_map().get(rollout_id)
                if not rollout:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                    return
                camera_paths = rollout.get("camera_video_paths")
                if not isinstance(camera_paths, dict) or not camera_paths:
                    self.json_error(HTTPStatus.NOT_FOUND, "Rollout has no camera videos")
                    return
                camera = str(query.get("camera", [""])[0] or "").strip()
                if not camera:
                    declared = rollout.get("primary_camera")
                    if isinstance(declared, str) and declared in camera_paths:
                        camera = declared
                    else:
                        preferred = (
                            "cam_high",
                            "cam_wrist",
                            "cam_left_wrist",
                            "cam_right_wrist",
                        )
                        camera = next(
                            (name for name in preferred if name in camera_paths),
                            next(iter(camera_paths)),
                        )
                value = camera_paths.get(camera)
                if not isinstance(value, str) or not value:
                    self.json_error(HTTPStatus.NOT_FOUND, "Camera video is unavailable")
                    return
                video = self.app.resolve_project_file(value, ".mp4")
                if not video.is_file():
                    self.json_error(HTTPStatus.NOT_FOUND, "Camera video file is unavailable")
                    return
                self.serve_video(video)
                return
            if path == "/":
                self.serve_static(self.app.static_dir / "index.html")
                return
            if path.startswith("/static/"):
                relative = path[len("/static/") :]
                target = (self.app.static_dir / relative).resolve()
                try:
                    target.relative_to(self.app.static_dir)
                except ValueError:
                    self.json_error(HTTPStatus.FORBIDDEN, "Invalid static path")
                    return
                self.serve_static(target)
                return
            self.json_error(HTTPStatus.NOT_FOUND, "Not found")
        except JobConflictError as exc:
            self.json_error(HTTPStatus.CONFLICT, str(exc))
        except TmuxSupervisorError as exc:
            self.json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        except (ValidationError, json.JSONDecodeError) as exc:
            self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
        except subprocess.SubprocessError as exc:
            self.json_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Subprocess check failed: " + str(exc),
            )
        except OSError as exc:
            self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_PUT(self) -> None:
        try:
            path = unquote(urlparse(self.path).path)
            if path != "/api/settings":
                self.json_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.json_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                return
            if length <= 0 or length > 100_000:
                self.json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size")
                return
            payload = json.loads(self.rfile.read(length))
            settings = self.app.settings.write(payload)
            self.json_response(
                HTTPStatus.OK,
                {
                    "settings": settings,
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
        except (ValidationError, json.JSONDecodeError) as exc:
            self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
        except OSError as exc:
            self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        try:
            path = unquote(urlparse(self.path).path)
            if path in {
                "/api/analysis/localization/run",
                "/api/analysis/localization/presets/save",
                "/api/analysis/localization/presets/delete",
                "/api/analysis/localization/challenge/save",
            }:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.json_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                    return
                if length <= 0 or length > 1_000_000:
                    self.json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size")
                    return
                payload = json.loads(self.rfile.read(length))
                if path == "/api/analysis/localization/run":
                    job = self.app.analysis_jobs.start_localization_spec_run(payload)
                    self.json_response(HTTPStatus.ACCEPTED, {"job": job})
                elif path == "/api/analysis/localization/presets/save":
                    preset = self.app.analysis_jobs.save_localization_preset(payload)
                    self.json_response(HTTPStatus.OK, {"preset": preset})
                elif path == "/api/analysis/localization/presets/delete":
                    result = self.app.analysis_jobs.delete_localization_preset(payload)
                    self.json_response(HTTPStatus.OK, result)
                else:
                    manifest = self.app.analysis_jobs.save_localization_challenge_set(payload)
                    self.json_response(HTTPStatus.OK, {"challenge_set": manifest})
                return
            if path.startswith("/api/baselines/posthoc-localization/"):
                rollout_id = path.rsplit("/", 1)[-1]
                rollout = self.app.rollout_map().get(rollout_id)
                if not rollout:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.json_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                    return
                if length <= 0 or length > 100_000:
                    self.json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size")
                    return
                payload = json.loads(self.rfile.read(length))
                result = self.app.baselines.run_posthoc_localization(
                    rollout,
                    payload,
                )
                self.json_response(HTTPStatus.OK, {"posthoc_localization": result})
                return
            if path in {"/api/baselines/run-batch", "/api/analysis/run", "/api/rollouts/generate"}:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.json_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                    return
                if length <= 0 or length > 100_000:
                    self.json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size")
                    return
                payload = json.loads(self.rfile.read(length))
                if path == "/api/baselines/run-batch":
                    job = self.app.baselines.start_batch(payload)
                elif path == "/api/analysis/run":
                    job = self.app.analysis_jobs.start_run(payload)
                else:
                    job = self.app.rollout_jobs.start(payload)
                self.json_response(HTTPStatus.ACCEPTED, {"job": job})
                return
            if path.startswith("/api/baselines/run/"):
                rollout_id = path.rsplit("/", 1)[-1]
                rollout = self.app.rollout_map().get(rollout_id)
                if not rollout:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.json_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                    return
                if length > 100_000:
                    self.json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size")
                    return
                payload = json.loads(self.rfile.read(length)) if length else {}
                job = self.app.baselines.start_run(
                    rollout,
                    str(payload.get("baseline", "")),
                    str(payload.get("gpu", "0")),
                    payload.get("memory_utilization", 0.80),
                    payload.get(
                        "instruction_condition",
                        payload.get("condition", "full_instruction"),
                    ),
                    options=payload.get("options"),
                )
                self.json_response(HTTPStatus.ACCEPTED, {"job": job})
                return
            if not path.startswith("/api/annotations/"):
                self.json_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            rollout_id = path.rsplit("/", 1)[-1]
            rollout = self.app.rollout_map().get(rollout_id)
            if not rollout:
                self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.json_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                return
            if length <= 0 or length > 1_000_000:
                self.json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid request size")
                return
            payload = json.loads(self.rfile.read(length))
            record = self.app.store.write(rollout, payload)
            self.json_response(HTTPStatus.OK, {"annotation": record})
        except JobConflictError as exc:
            self.json_error(HTTPStatus.CONFLICT, str(exc))
        except AnalysisEnvironmentError as exc:
            self.json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        except TmuxSupervisorError as exc:
            self.json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        except (ValidationError, json.JSONDecodeError) as exc:
            self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
        except OSError as exc:
            self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def serve_static(self, path: Path) -> None:
        if not path.is_file():
            self.json_error(HTTPStatus.NOT_FOUND, "Static file not found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        if not self._safe_end_headers():
            return
        self._safe_write(body)

    def serve_video(self, path: Path) -> None:
        if not path.is_file():
            self.json_error(HTTPStatus.NOT_FOUND, "Video not found")
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            left, right = match.groups()
            if left:
                start = int(left)
                end = int(right) if right else end
            elif right:
                count = int(right)
                start = max(size - count, 0)
            if start >= size or start > end:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Connection", "close")
        if not self._safe_end_headers():
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                if not self._safe_write(chunk):
                    break
                remaining -= len(chunk)
