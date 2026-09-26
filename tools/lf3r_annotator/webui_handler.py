#!/usr/bin/env python3
"""Explicit HTTP routes layered on top of the core LF3R handler."""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import server
from non_analysis_tools import gpu_status
from webui_application import WebUIApplication


class WebUIHandler(server.LF3RHandler):
    """WebUI-only routes without mutating the core handler class."""

    app: WebUIApplication

    def _read_json_body(self, *, maximum: int = 100_000) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0 or length > maximum:
            raise ValueError("Invalid request size")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query, keep_blank_values=True)

        try:
            if path == "/api/repair/synthetic-suffix/rollouts":
                self.json_response(
                    HTTPStatus.OK,
                    {"rollouts": self.app.repair.catalog(self.app.load_rollouts())},
                )
                return

            if path == "/api/repair/synthetic-suffix/runs":
                self.json_response(
                    HTTPStatus.OK,
                    {"runs": self.app.repair.list_runs()},
                )
                return

            if path.startswith("/api/repair/synthetic-suffix/artifact/"):
                relative = path[len("/api/repair/synthetic-suffix/artifact/"):]
                parts = relative.split("/", 1)
                if len(parts) != 2:
                    self.json_error(HTTPStatus.NOT_FOUND, "Repair artifact not found")
                    return
                artifact = self.app.repair.artifact(parts[0], parts[1])
                if artifact.suffix.lower() == ".mp4":
                    self.serve_video(artifact)
                else:
                    self.file_response(artifact)
                return

            if path.startswith("/api/repair/synthetic-suffix/run/"):
                run_id = path[len("/api/repair/synthetic-suffix/run/"):].strip("/")
                if "/" in run_id or not run_id:
                    self.json_error(HTTPStatus.NOT_FOUND, "Repair run not found")
                    return
                self.json_response(
                    HTTPStatus.OK,
                    {"run": self.app.repair.detail(run_id)},
                )
                return

            if path.startswith("/api/repair/synthetic-suffix/jobs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 6 and parts[5] == "log":
                    job_id = parts[4]
                    tail = query.get("tail", ["240"])[0]
                    self.json_response(
                        HTTPStatus.OK,
                        {"log": self.app.repair.log(job_id, tail)},
                    )
                    return
                if len(parts) == 5:
                    self.json_response(
                        HTTPStatus.OK,
                        {"job": self.app.repair.job(parts[4])},
                    )
                    return
                self.json_error(HTTPStatus.NOT_FOUND, "Repair job not found")
                return

            if path == "/api/gpu-status":
                self.json_response(
                    HTTPStatus.OK,
                    {"gpu_status": gpu_status()},
                )
                return

            if path == "/api/manifests":
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "manifests": self.app.manifest_info(),
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
                        "annotation_status": (
                            annotation["review_status"]
                            if annotation
                            else "unreviewed"
                        ),
                    }
                    options = self.app.instruction_variant_options(record)
                    enriched["instruction_variants"] = options
                    enriched["instruction_variant_conditions"] = list(options)
                    if self.app.instruction_variant_manifest_path.is_file():
                        enriched["instruction_variant_manifest"] = str(
                            self.app.instruction_variant_manifest_path.relative_to(
                                self.app.project_root
                            )
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

            if path == "/api/jobs":
                job_type = query.get("job_type", [None])[0] or None
                status = query.get("status", [None])[0] or None
                jobs = self.app.tmux.list(job_type=job_type, status=status)
                refreshed = []
                for job in jobs:
                    if job.get("job_type") == "baseline":
                        try:
                            job = self.app.baselines.job(str(job["job_id"]))
                        except KeyError:
                            pass
                    refreshed.append(job)
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "jobs": refreshed,
                        "tmux_available": self.app.tmux.available,
                    },
                )
                return

            if path == "/api/tool-jobs":
                status = query.get("status", [None])[0] or None
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "jobs": self.app.project_tools.list(status=status),
                        "project_tools_protocol": "immediate-registry-v1",
                    },
                )
                return

            if path.startswith("/api/tool-jobs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[3] == "log":
                    job_id = parts[2]
                    tail = query.get("tail", ["240"])[0]
                    self.json_response(
                        HTTPStatus.OK,
                        {
                            "log": self.app.project_tools.log(
                                job_id,
                                int(tail),
                            )
                        },
                    )
                    return
                if len(parts) == 3:
                    self.json_response(
                        HTTPStatus.OK,
                        {"job": self.app.project_tools.get(parts[2])},
                    )
                    return
                self.json_error(HTTPStatus.NOT_FOUND, "Not found")
                return
        except KeyError:
            self.json_error(HTTPStatus.NOT_FOUND, "Unknown project-tool job")
            return
        except server.ValidationError as exc:
            self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except (TypeError, ValueError) as exc:
            self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except FileNotFoundError:
            self.json_error(HTTPStatus.NOT_FOUND, "Repair resource not found")
            return
        except OSError as exc:
            self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return

        super().do_GET()

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)

        if path in {
            "/api/repair/synthetic-suffix/validate",
            "/api/repair/synthetic-suffix/run",
        }:
            try:
                payload = self._read_json_body(maximum=300_000)
                if path.endswith("/validate"):
                    plan = self.app.repair.validate_with_alignment(
                        payload,
                        self.app.rollout_map(),
                    )
                    self.json_response(HTTPStatus.OK, {"validation": plan})
                else:
                    job = self.app.repair.start(
                        payload,
                        self.app.rollout_map(),
                    )
                    self.json_response(HTTPStatus.ACCEPTED, {"job": job})
            except server.TmuxSupervisorError as exc:
                self.json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            except server.ValidationError as exc:
                self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
            except OSError as exc:
                self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return

        if path.startswith("/api/repair/synthetic-suffix/run/") and path.endswith("/review"):
            try:
                prefix = "/api/repair/synthetic-suffix/run/"
                run_id = path[len(prefix):-len("/review")].strip("/")
                if "/" in run_id or not run_id:
                    raise server.ValidationError("Invalid Repair run id")
                payload = self._read_json_body(maximum=100_000)
                evaluation = self.app.repair.save_human_evaluation(run_id, payload)
                self.json_response(HTTPStatus.OK, {"human_evaluation": evaluation})
            except FileNotFoundError:
                self.json_error(HTTPStatus.NOT_FOUND, "Repair run not found")
            except server.ValidationError as exc:
                self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
            except OSError as exc:
                self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return

        if path == "/api/tools/run":
            try:
                payload = self._read_json_body()
                action = str(payload.get("action") or "").strip()
                options = payload.get("options") or {}
                if not isinstance(options, dict):
                    raise ValueError("options must be a JSON object")
                client_request_id = str(
                    payload.get("client_request_id") or ""
                ).strip()
                if client_request_id:
                    if len(client_request_id) > 120 or not all(
                        character.isalnum() or character in "._:-"
                        for character in client_request_id
                    ):
                        raise ValueError("invalid client_request_id")
                self.log_message(
                    "project-tool submit received action=%s client_request_id=%s",
                    action,
                    client_request_id or "-",
                )
                job = self.app.project_tools.submit(
                    action,
                    options,
                    client_request_id=client_request_id or None,
                )
                self.log_message(
                    "project-tool submit registered action=%s job_id=%s status=%s",
                    action,
                    job.get("job_id"),
                    job.get("status"),
                )
                self.json_response(
                    HTTPStatus.ACCEPTED,
                    {
                        "job": job,
                        "project_tools_protocol": "immediate-registry-v1",
                    },
                )
            except server.TmuxSupervisorError as exc:
                self.json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    str(exc),
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
            except OSError as exc:
                self.json_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    str(exc),
                )
            return

        parts = path.strip("/").split("/")
        if (
            len(parts) == 4
            and parts[0] == "api"
            and parts[1] == "baseline-jobs"
            and parts[3] == "cancel"
        ):
            try:
                job = self.app.baselines.cancel_job(parts[2])
                self.json_response(
                    HTTPStatus.ACCEPTED,
                    {"job": job},
                )
            except KeyError:
                self.json_error(
                    HTTPStatus.NOT_FOUND,
                    "Unknown baseline job",
                )
            except server.ValidationError as exc:
                self.json_error(HTTPStatus.CONFLICT, str(exc))
            except server.TmuxSupervisorError as exc:
                self.json_error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    str(exc),
                )
            except OSError as exc:
                self.json_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    str(exc),
                )
            return

        if (
            path == "/api/baselines/run-batch"
            or path.startswith("/api/baselines/run/")
        ):
            try:
                self.app.refresh_manifest_catalog()
            except server.ValidationError as exc:
                self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except OSError as exc:
                self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return

        super().do_POST()
