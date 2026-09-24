"""Core LF3R application composition."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from analysis_jobs import AnalysisJobService
from analysis_service import AnalysisService
from backend_core import (
    INSTRUCTION_VARIANT_CONDITIONS,
    JobCoordinator,
    ROLLOUT_ID_RE,
    ValidationError,
    load_manifest_records,
    record_camera_video_paths,
)
from baseline_constants import INSTRUCTION_VARIANT_LABELS
from baseline_service import BaselineService
from rollout_service import RolloutGenerationService
from stores import AnnotationStore, SettingsStore
from task_supervisor import TmuxJobSupervisor


class LF3RApplication:
    coordinator_class = JobCoordinator
    baseline_service_class = BaselineService
    project_tool_service_class = None

    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        annotation_root: Path,
        analysis_python: Path | None = None,
        tmux_binary: str | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.static_dir = (Path(__file__).resolve().parent / "static").resolve()
        self.instruction_variant_manifest_path = (
            self.project_root
            / "tools"
            / "lf3r_annotator"
            / "instruction_variants"
            / "libero_10_v1"
            / "manifest.jsonl"
        )
        self._instruction_variant_index: dict[str, dict[str, dict[str, Any]]] | None = None
        self.store = AnnotationStore(annotation_root)
        self.settings = SettingsStore(self.project_root)
        self.analysis = AnalysisService(self.project_root, self.manifest_path, annotation_root)
        self.job_coordinator = self.coordinator_class()
        self.tmux = TmuxJobSupervisor(self.project_root, tmux_binary=tmux_binary)
        self.baselines = self.baseline_service_class(
            self.project_root,
            self.manifest_path,
            self.job_coordinator,
            self.tmux,
            annotation_store=self.store,
        )
        configured_analysis_python = (
            Path(os.environ["LF3R_ANALYSIS_PYTHON"]).expanduser()
            if analysis_python is None and os.environ.get("LF3R_ANALYSIS_PYTHON")
            else analysis_python
        )
        self.analysis_jobs = AnalysisJobService(
            self.project_root,
            self.manifest_path,
            annotation_root,
            self.baselines,
            self.job_coordinator,
            self.tmux,
            analysis_python=configured_analysis_python,
        )
        self.rollout_jobs = RolloutGenerationService(
            self.project_root, self.manifest_path, self.job_coordinator, self.tmux
        )
        if self.project_tool_service_class is not None:
            self.project_tools = self.project_tool_service_class(
                self.project_root,
                self.tmux,
            )
        self.tmux.recover()

    def load_instruction_variant_records(self) -> dict[str, dict[str, dict[str, Any]]]:
        if self._instruction_variant_index is not None:
            return self._instruction_variant_index
        path = self.instruction_variant_manifest_path
        if not path.is_file():
            self._instruction_variant_index = {}
            return self._instruction_variant_index
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for row in load_manifest_records(path):
            source_id = row.get("source_rollout_id") or row.get("source_id")
            condition = row.get("instruction_variant") or row.get("condition")
            if not isinstance(source_id, str) or not ROLLOUT_ID_RE.fullmatch(source_id):
                raise ValidationError("Invalid source rollout id in instruction variant manifest")
            if condition not in {"subtask_a", "subtask_b"}:
                continue
            if row.get("id") != source_id + "--" + condition:
                raise ValidationError("Instruction variant id does not match its source and condition")
            if condition in grouped.setdefault(source_id, {}):
                raise ValidationError("Duplicate instruction variant for " + source_id + ": " + condition)
            grouped[source_id][condition] = row
        self._instruction_variant_index = grouped
        return grouped

    def instruction_variant_options(self, record: dict[str, Any]) -> dict[str, dict[str, Any]]:
        source_id = str(record["id"])
        options: dict[str, dict[str, Any]] = {
            "full_instruction": {
                "id": source_id,
                "condition": "full_instruction",
                "label": INSTRUCTION_VARIANT_LABELS["full_instruction"],
                "instruction": record.get("task_description", ""),
                "instruction_type": "original_full_instruction",
                "available": True,
                "counterfactual": False,
            }
        }
        variants = self.load_instruction_variant_records().get(source_id, {})
        for condition in ("subtask_a", "subtask_b"):
            row = variants.get(condition)
            if row is None:
                continue
            if row.get("camera_video_paths") != record.get("camera_video_paths"):
                raise ValidationError("Instruction variant cameras do not match source rollout: " + source_id)
            options[condition] = {
                "id": row["id"],
                "condition": condition,
                "label": row.get("subtask_label") or INSTRUCTION_VARIANT_LABELS[condition],
                "instruction": row.get("task_description") or row.get("instruction", ""),
                "instruction_type": row.get("instruction_type", "counterfactual_single_subtask"),
                "subtask_label": row.get("subtask_label"),
                "subtask_subject": row.get("subtask_subject"),
                "subtask_target": row.get("subtask_target"),
                "available": True,
                "counterfactual": True,
            }
        return options

    def instruction_variant_for(
        self,
        record: dict[str, Any],
        condition: str,
    ) -> dict[str, Any] | None:
        if condition not in INSTRUCTION_VARIANT_CONDITIONS:
            raise ValidationError(
                "condition must be one of: " + ", ".join(INSTRUCTION_VARIANT_CONDITIONS)
            )
        if condition == "full_instruction":
            variant = dict(record)
            variant.update({
                "condition": "full_instruction",
                "instruction_variant": "full_instruction",
                "instruction_type": "original_full_instruction",
                "instruction": record.get("task_description", ""),
                "original_full_instruction": record.get("task_description", ""),
                "source_rollout_id": record["id"],
            })
            return variant
        row = self.load_instruction_variant_records().get(record["id"], {}).get(condition)
        if row is None:
            return None
        if row.get("camera_video_paths") != record.get("camera_video_paths"):
            raise ValidationError("Instruction variant cameras do not match source rollout: " + record["id"])
        return dict(row)

    def load_rollouts(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                rollout_id = record.get("id")
                if not isinstance(rollout_id, str) or not ROLLOUT_ID_RE.fullmatch(rollout_id):
                    raise ValidationError(f"Invalid rollout id at manifest line {line_number}")
                if rollout_id in seen:
                    raise ValidationError(f"Duplicate rollout id: {rollout_id}")
                seen.add(rollout_id)
                for video_path in record_camera_video_paths(record).values():
                    self.resolve_project_file(video_path, ".mp4")
                records.append(record)
        return records

    def rollout_map(self) -> dict[str, dict[str, Any]]:
        return {record["id"]: record for record in self.load_rollouts()}

    def resolve_project_file(self, relative: str, suffix: str | None = None) -> Path:
        path = (self.project_root / relative).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValidationError("Manifest path escapes project root") from exc
        if suffix and path.suffix.lower() != suffix:
            raise ValidationError(f"Expected a {suffix} file")
        return path
