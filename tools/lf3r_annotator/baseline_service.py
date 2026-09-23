"""Baseline service composition.

Implementation is split between result/catalog behavior and launch/job behavior.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any

from backend_core import JobCoordinator, ValidationError
from baseline_index import BaselineRunIndex
from baseline_jobs import BaselineJobsMixin
from baseline_results import BaselineResultsMixin
from stores import AnnotationStore
from task_supervisor import TmuxJobSupervisor


class BaselineService(BaselineResultsMixin, BaselineJobsMixin):
    """Read baseline outputs and launch bounded baseline jobs."""

    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        coordinator: JobCoordinator | None = None,
        tmux: TmuxJobSupervisor | None = None,
        annotation_store: AnnotationStore | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.annotation_store = annotation_store
        self.baseline_root = self.project_root / "outputs" / "baselines"
        self.web_output_root = self.baseline_root / "web_runs"
        self.variant_manifest_path = (
            self.project_root
            / "tools"
            / "lf3r_annotator"
            / "instruction_variants"
            / "libero_10_v1"
            / "manifest.jsonl"
        )
        self.variant_output_root = (
            self.baseline_root / "instruction_variants" / "libero_10"
        )
        self.web_logs_root = (
            self.project_root / "logs" / "baselines" / "web_runs"
        )
        robo_python = os.environ.get("LF3R_ROBODOPAMINE_PYTHON")
        if robo_python:
            self.robo_python = Path(
                os.path.abspath(Path(robo_python).expanduser())
            )
        else:
            self.robo_python = (
                self.project_root
                / "conda_envs"
                / "LF3R-robo-dopamine"
                / "bin"
                / "python"
            )

        self.run_index = BaselineRunIndex(
            self.project_root,
            self.baseline_root,
        )
        self.jobs: dict[str, dict[str, Any]] = {}
        self.jobs_lock = threading.Lock()
        self.coordinator = coordinator or JobCoordinator()
        self.tmux = tmux or TmuxJobSupervisor(
            self.project_root
        )
        self.tmux.register_handler(
            "baseline",
            self._on_job_loaded,
            self._on_job_poll,
            self._on_job_finished,
        )

    def _project_path(
        self,
        value: str | Path,
    ) -> Path:
        path = Path(value).expanduser()
        resolved = (
            path.resolve()
            if path.is_absolute()
            else (self.project_root / path).resolve()
        )
        try:
            resolved.relative_to(self.project_root)
        except ValueError as exc:
            raise ValidationError(
                "Baseline path escapes project root"
            ) from exc
        return resolved

    def _relative(self, path: Path) -> str:
        try:
            return str(
                path.resolve().relative_to(
                    self.project_root
                )
            )
        except ValueError:
            return str(path)

    def _number(
        self,
        value: Any,
    ) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return (
            number
            if math.isfinite(number)
            else None
        )
