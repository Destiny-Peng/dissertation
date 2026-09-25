"""Persistent analysis-job service composition."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from analysis_constants import (
    ANALYSIS_BASELINE_METHODS,
    ANALYSIS_TABLE_FILES,
    ROLLOUT_OUTCOME_TABLE_FILES,
    ROLLOUT_OUTCOME_REQUIRED_FILES,
    ROBO_HOP_EXTENDED_FILES,
    ROBO_HOP_REQUIRED_FILES,
    ROBO_LABEL_LOSS_REQUIRED_FILES,
    ROBO_LOCALIZATION_HEAD_REQUIRED_FILES,
)
from analysis_localization import AnalysisLocalizationMixin
from analysis_robo_jobs import AnalysisRoboJobsMixin
from backend_core import (
    AnalysisEnvironmentError,
    JobCoordinator,
    ValidationError,
    atomic_json_write,
    load_manifest_records,
    run_rollout_ids,
    select_scope_records,
    validate_run_scope,
)
from baseline_constants import BASELINE_RUN_STATUSES
from baseline_service import BaselineService
from task_supervisor import TmuxJobSupervisor


class AnalysisJobService(
    AnalysisRoboJobsMixin,
    AnalysisLocalizationMixin,
):
    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        annotation_root: Path,
        baselines: BaselineService,
        coordinator: JobCoordinator,
        tmux: TmuxJobSupervisor,
        analysis_python: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.annotation_root = annotation_root.resolve()
        self.baselines = baselines
        self.coordinator = coordinator
        self.tmux = tmux
        self.analysis_root = self.project_root / "outputs" / "baseline_signal_analysis"
        self.robo_hop_root = (
            self.project_root / "outputs" / "robo_dopamine_incremental_hop"
        )
        self.robo_localization_head_root = (
            self.project_root / "outputs" / "robo_dopamine_localization_head"
        )
        self.robo_label_loss_root = (
            self.project_root / "outputs" / "robo_dopamine_label_loss_ablation"
        )
        self.localization_root = self.project_root / "outputs" / "robo_localization"
        self.localization_preset_root = (
            self.project_root / "config" / "robo_localization_presets"
        )
        self.localization_challenge_root = (
            self.project_root / "config" / "robo_localization_challenge_sets"
        )
        self.log_root = self.project_root / "logs" / "baselines" / "analysis_web"
        configured_python = (
            analysis_python
            if analysis_python is not None
            else self.project_root / "conda_envs" / "LF3R-ananlyse" / "bin" / "python"
        )
        # Preserve the venv entrypoint symlink. Resolving it points at uv's
        # cache interpreter and bypasses the venv's site-packages.
        configured_python = Path(configured_python).expanduser()
        if not configured_python.is_absolute():
            configured_python = self.project_root / configured_python
        self.analysis_python = Path(os.path.abspath(configured_python))
        robo_python = os.environ.get("LF3R_ROBODOPAMINE_PYTHON")
        if robo_python:
            self.robo_python = Path(os.path.abspath(Path(robo_python).expanduser()))
        else:
            self.robo_python = (
                self.project_root / "conda_envs" / "LF3R-robo-dopamine" / "bin" / "python"
            )
        self.jobs: dict[str, dict[str, Any]] = {}
        self.jobs_lock = threading.Lock()
        self.tmux.register_handler(
            "analysis",
            self._on_job_loaded,
            self._on_job_poll,
            self._on_job_finished,
        )

    def _project_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        resolved = path.resolve() if path.is_absolute() else (self.project_root / path).resolve()
        try:
            resolved.relative_to(self.project_root)
        except ValueError as exc:
            raise ValidationError("Analysis path escapes project root") from exc
        return resolved

    def _relative(self, path: Path) -> str:
        try:
            return str(Path(os.path.abspath(path)).relative_to(self.project_root))
        except ValueError:
            return str(path)

    def environment_status(self) -> dict[str, Any]:
        path = self.analysis_python
        status: dict[str, Any] = {
            "path": self._relative(path),
            "absolute_path": str(path),
            "exists": path.is_file(),
            "executable": path.is_file() and os.access(path, os.X_OK),
            "ready": False,
            "dependencies": ["matplotlib", "numpy", "pandas"],
            "error": None,
        }
        if not status["executable"]:
            status["error"] = "Analysis environment Python executable is missing"
            return status
        try:
            result = subprocess.run(
                [str(path), "-c", "import matplotlib, numpy, pandas"],
                cwd=str(self.project_root),
                env={**os.environ, "MPLBACKEND": "Agg"},
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            status["error"] = str(error)
            return status
        if result.returncode != 0:
            status["error"] = (
                result.stderr or result.stdout or "Analysis dependency import failed"
            ).strip()[-2000:]
            return status
        status["ready"] = True
        return status

    def require_environment(self) -> None:
        status = self.environment_status()
        if not status["ready"]:
            raise AnalysisEnvironmentError(
                "Analysis environment unavailable: "
                + str(status.get("error") or status["path"])
            )

    @staticmethod
    def _integer(value: Any, name: str, minimum: int = 1, maximum: int = 1000000) -> int:
        if isinstance(value, bool):
            raise ValidationError(f"{name} must be an integer")
        try:
            number = int(value)
        except (TypeError, ValueError) as error:
            raise ValidationError(f"{name} must be an integer") from error
        if number < minimum or number > maximum:
            raise ValidationError(f"{name} must be between {minimum} and {maximum}")
        return number

    def _manifest_records(self) -> list[dict[str, Any]]:
        return load_manifest_records(self.manifest_path)

    def _validate_runs(
        self,
        raw_runs: Any,
        selected_ids: set[str],
        allow_partial_coverage: bool = False,
    ) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
        if not isinstance(raw_runs, dict):
            raise ValidationError("runs must be an object containing all baseline methods")
        missing_methods = [method for method in ANALYSIS_BASELINE_METHODS if not raw_runs.get(method)]
        if missing_methods:
            raise ValidationError("Missing analysis run(s): " + ", ".join(missing_methods))
        validated: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
        for method in ANALYSIS_BASELINE_METHODS:
            raw_value = raw_runs[method]
            raw_values = raw_value if isinstance(raw_value, list) else [raw_value]
            if method != "rynnvalue" and len(raw_values) != 1:
                raise ValidationError(f"{method} accepts one run root")
            if not raw_values or any(not isinstance(value, (str, Path)) for value in raw_values):
                raise ValidationError(f"{method} run must be a path or, for RynnValue, a list of paths")
            method_runs: list[tuple[Path, dict[str, Any]]] = []
            seen_ids: set[str] = set()
            duplicate_ids: set[str] = set()
            for raw_path in raw_values:
                run_path = self._project_path(str(raw_path))
                try:
                    run_path.relative_to(self.baselines.baseline_root.resolve())
                except ValueError as exc:
                    raise ValidationError(f"{method} run must be inside outputs/baselines") from exc
                metadata_path = run_path / "run.json"
                if not metadata_path.is_file():
                    raise ValidationError(f"{method} run.json is missing")
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ValidationError(f"Invalid {method} run metadata") from error
                if metadata.get("baseline") != method:
                    raise ValidationError(f"Selected run is not a {method} run")
                if metadata.get("status") not in BASELINE_RUN_STATUSES:
                    raise ValidationError(f"{method} run is not complete")
                run_ids = run_rollout_ids(run_path)
                duplicate_ids.update(seen_ids.intersection(run_ids))
                seen_ids.update(run_ids)
                try:
                    int(metadata.get("selected_rollouts") or 0)
                except (TypeError, ValueError) as error:
                    raise ValidationError(f"{method} run has an invalid selected_rollouts count") from error
                method_runs.append((run_path, metadata))
            if duplicate_ids:
                raise ValidationError(
                    f"Duplicate {method} rollout IDs across selected runs: {sorted(duplicate_ids)[:5]}"
                )
            missing = sorted(selected_ids - seen_ids)
            if missing and not allow_partial_coverage:
                raise ValidationError(f"{method} run does not cover {len(missing)} selected rollout(s)")
            if not allow_partial_coverage and sum(
                int(metadata.get("selected_rollouts") or 0)
                for _, metadata in method_runs
            ) < len(selected_ids):
                raise ValidationError(f"{method} run selected fewer rollouts than the requested scope")
            validated[method] = method_runs
        return validated

    def start_outcome_evaluation_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.require_environment()
        allowed_fields = {"analysis_kind", "scope", "output_label"}
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            raise ValidationError(
                "Unknown outcome-evaluation field(s): "
                + ", ".join(sorted(unknown_fields))
            )
        scope = validate_run_scope(payload.get("scope"))
        sources = self.baselines.outcome_evaluation_sources(scope)
        evaluation_ids = list(sources["evaluation_rollout_ids"])
        if not evaluation_ids:
            raise ValidationError(
                f"No completed annotations matched scope {scope}"
            )
        if not any(row["available_rollouts"] for row in sources["coverage"]):
            raise ValidationError(
                f"No saved baseline outputs matched completed annotations in scope {scope}"
            )

        label = str(payload.get("output_label") or "web_outcome").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", label):
            raise ValidationError(
                "output_label must contain only letters, numbers, dot, underscore, or hyphen"
            )
        script = self.project_root / "tools" / "analyze_baseline_rollout_outcomes.py"
        if not script.is_file():
            raise ValidationError("Rollout outcome evaluator is not installed")

        job_id = "analysis-outcome-" + uuid.uuid4().hex[:12]
        workspace = self.analysis_root / ".web_jobs" / job_id
        output_temp = workspace / "output"
        output_final = self.analysis_root / (
            "outcome_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            + "_" + label + "_" + job_id[-8:]
        )
        selection_path = workspace / "selection.json"
        source_map_path = workspace / "source_map.json"
        selection_doc = {
            "schema_version": 1,
            "scope": scope,
            "selection": [{"id": rollout_id} for rollout_id in evaluation_ids],
        }
        source_map_doc = {
            "schema_version": 1,
            "scope": scope,
            "source_resolution": "newest_parseable_output_per_rollout",
            "source_maps": sources["source_maps"],
            "coverage": sources["coverage"],
        }
        command = [
            str(self.analysis_python), str(script),
            "--selection", str(selection_path),
            "--source-map", str(source_map_path),
            "--manifest", str(self.manifest_path),
            "--annotations-dir", str(self.annotation_root / "records"),
            "--output-dir", str(output_temp),
        ]

        self.analysis_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        (self.analysis_root / ".web_jobs").mkdir(parents=True, exist_ok=True)
        self.coordinator.acquire(job_id, "analysis")
        try:
            workspace.mkdir(parents=True, exist_ok=False)
            atomic_json_write(selection_path, selection_doc)
            atomic_json_write(source_map_path, source_map_doc)
            job = {
                "job_id": job_id,
                "job_type": "analysis",
                "analysis_kind": "rollout_outcome_evaluation",
                "status": "queued",
                "scope": scope,
                "selected_rollouts": len(evaluation_ids),
                "scope_rollouts": int(sources["scope_rollouts"]),
                "incomplete_annotation_rollouts": int(
                    sources["incomplete_annotation_rollouts"]
                ),
                "coverage": sources["coverage"],
                "command": command,
                "output_dir": self._relative(output_final),
                "output_temp": self._relative(output_temp),
                "selection_path": self._relative(selection_path),
                "source_map_path": self._relative(source_map_path),
                "log_path": self._relative(self.log_root / f"{job_id}.log"),
                "started_at": None,
                "finished_at": None,
                "return_code": None,
                "error": None,
                "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "interpreter": str(self.analysis_python),
            }
            with self.jobs_lock:
                self.jobs[job_id] = job
            self.tmux.submit(
                job,
                command,
                self.log_root / f"{job_id}.log",
                interpreter=str(self.analysis_python),
                environment={"MPLBACKEND": "Agg"},
                on_poll=self._on_job_poll,
                on_finished=self._on_job_finished,
            )
        except Exception:
            with self.jobs_lock:
                self.jobs.pop(job_id, None)
            self.coordinator.release(job_id)
            raise
        return dict(job)

    def start_run(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValidationError("Analysis request must be a JSON object")
        analysis_kind = payload.get("analysis_kind")
        if analysis_kind == "rollout_outcome_evaluation":
            return self.start_outcome_evaluation_run(payload)
        if analysis_kind == "robo_localization_experiment":
            return self.start_localization_spec_run(payload)
        if analysis_kind == "robo_bilstm_success_ablation":
            return self.start_robo_localization_head_run(payload)
        if analysis_kind == "robo_bilstm_label_loss_ablation":
            return self.start_robo_label_loss_run(payload)
        if analysis_kind in {
            "robo_incremental_hop",
            "robo_hop_comparison",
        }:
            return self.start_robo_hop_run(payload)
        raise ValidationError(
            "analysis_kind is required; the legacy temporal analysis runner has been removed"
        )

    def _on_job_loaded(self, job: dict[str, Any]) -> None:
        job_id = str(job["job_id"])
        with self.jobs_lock:
            self.jobs[job_id] = job
        if job.get("status") in {"queued", "running"}:
            self.coordinator.restore(job_id, "analysis")

    def _on_job_poll(self, job: dict[str, Any]) -> None:
        job["last_polled_at"] = dt.datetime.now(dt.timezone.utc).isoformat()

    def _on_job_finished(
        self,
        job: dict[str, Any],
        return_code: int | None,
        reason: str | None,
    ) -> None:
        job_id = str(job["job_id"])
        error = reason
        try:
            if error is None and return_code == 0:
                output_temp = self._project_path(str(job["output_temp"]))
                output_final = self._project_path(str(job["output_dir"]))
                if job.get("analysis_kind") == "rollout_outcome_evaluation":
                    required = ROLLOUT_OUTCOME_REQUIRED_FILES
                    missing_message = (
                        "Outcome evaluation completed without all required artifacts"
                    )
                elif job.get("analysis_kind") == "robo_localization_experiment":
                    required = (
                        "metadata.json", "config.json", "experiment_manifest.json",
                        "training_records.json", "summary.csv", "per_rollout_predictions.csv",
                        "all_failure_predictions.csv",
                    )
                    missing_message = "Localization experiment completed without all required artifacts"
                elif job.get("analysis_kind") == "robo_bilstm_success_ablation":
                    required = ROBO_LOCALIZATION_HEAD_REQUIRED_FILES
                    missing_message = (
                        "BiLSTM localization-head training completed without all required artifacts"
                    )
                elif job.get("analysis_kind") == "robo_bilstm_label_loss_ablation":
                    required = ROBO_LABEL_LOSS_REQUIRED_FILES
                    missing_message = (
                        "BiLSTM label/loss ablation completed without all required artifacts"
                    )
                elif job.get("analysis_kind") in {
                    "robo_incremental_hop",
                    "robo_hop_comparison",
                }:
                    required = ROBO_HOP_REQUIRED_FILES
                    if job.get("analysis_kind") == "robo_hop_comparison":
                        required = (
                            *required,
                            *ROBO_HOP_EXTENDED_FILES,
                        )
                    missing_message = (
                        "Robo-Dopamine hop comparison completed without all required artifacts"
                    )
                else:
                    required = (
                        "metadata.json",
                        "event_metrics.jsonl",
                        *ANALYSIS_TABLE_FILES.values(),
                        *ROLLOUT_OUTCOME_TABLE_FILES.values(),
                    )
                    missing_message = (
                        "Temporal analysis completed without all required artifacts"
                    )
                if not all((output_temp / name).is_file() for name in required):
                    raise OSError(missing_message)
                if (
                    job.get("analysis_kind") == "robo_localization_experiment"
                    and not (output_temp / "checkpoints").is_dir()
                ):
                    raise OSError("Localization experiment completed without checkpoints")
                if output_final.exists():
                    raise OSError(f"Analysis output already exists: {output_final}")
                os.replace(output_temp, output_final)
                job["status"] = "complete"
            else:
                job["status"] = "failed"
                if job.get("analysis_kind") == "rollout_outcome_evaluation":
                    label = "Outcome evaluation"
                elif job.get("analysis_kind") == "robo_localization_experiment":
                    label = "Localization experiment"
                elif job.get("analysis_kind") == "robo_bilstm_success_ablation":
                    label = "BiLSTM localization-head training"
                elif job.get("analysis_kind") == "robo_bilstm_label_loss_ablation":
                    label = "BiLSTM label/loss ablation"
                else:
                    label = (
                        "Robo-Dopamine hop comparison"
                        if job.get("analysis_kind") in {
                            "robo_incremental_hop",
                            "robo_hop_comparison",
                        }
                        else "Temporal analysis"
                    )
                error = error or f"{label} exited with code {return_code}"
        except Exception as exc:
            job["status"] = "failed"
            error = str(exc)
        finally:
            with self.jobs_lock:
                job["return_code"] = return_code
                job["finished_at"] = job.get("finished_at") or dt.datetime.now(dt.timezone.utc).isoformat()
                job["error"] = error
            self.coordinator.release(job_id)

    def list_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        return self.tmux.list("analysis", status)

    def job(self, job_id: str) -> dict[str, Any]:
        with self.jobs_lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return dict(self.jobs[job_id])

    def log(self, job_id: str, tail: Any = 200) -> dict[str, Any]:
        job = self.job(job_id)
        try:
            count = self._integer(tail, "tail", 1, 2000)
        except ValidationError:
            count = 200
        path = self._project_path(job["log_path"])
        if not path.is_file():
            return {"job_id": job_id, "lines": [], "text": ""}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]
        return {"job_id": job_id, "lines": lines, "text": "\n".join(lines)}

