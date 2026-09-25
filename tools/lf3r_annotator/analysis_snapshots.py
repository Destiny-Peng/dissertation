"""Snapshot discovery and primary analysis responses."""

from __future__ import annotations

import copy
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

from analysis_constants import (
    ANALYSIS_BASELINE_METHODS,
    ANALYSIS_EVENT_ARRAY_FIELDS,
    ANALYSIS_EVENT_FIELDS,
    ANALYSIS_LOCALIZATION_EVENT_FIELDS,
    ANALYSIS_RECORD_FIELDS,
    ANALYSIS_TABLE_FILES,
    CHANGEPOINT_EVENT_FIELDS,
    CHANGEPOINT_TABLE_FILES,
    EVENT_TRIGGERED_TABLE_FILES,
    ROBO_HOP_REQUIRED_FILES,
    ROBO_HOP_TABLE_FILES,
    ROLLOUT_OUTCOME_TABLE_FILES,
    ROLLOUT_OUTCOME_REQUIRED_FILES,
)


class AnalysisSnapshotsMixin:
    def _latest_snapshot(self) -> tuple[Path, dict[str, Any]] | None:
        if not self.analysis_root.is_dir():
            return None
        candidates = []
        required = ("metadata.json", "event_metrics.jsonl", *ANALYSIS_TABLE_FILES.values())
        for metadata_path in self.analysis_root.glob("*/metadata.json"):
            directory = metadata_path.parent
            if not all((directory / name).is_file() for name in required):
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            candidates.append((metadata_path.stat().st_mtime, directory, metadata))
        if not candidates:
            return None
        _, directory, metadata = max(candidates, key=lambda item: item[0])
        return directory, metadata

    def _latest_annotation_update(self) -> str | None:
        records_dir = self.annotation_root / "records"
        latest = None
        if not records_dir.is_dir():
            return None
        for path in records_dir.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8")).get("updated_at")
            except (OSError, json.JSONDecodeError):
                continue
            if value and (latest is None or str(value) > latest):
                latest = str(value)
        return latest

    def _event_metrics(self, path: Path, manifest: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("is_background"):
                    continue
                rollout_id = raw.get("rollout_id")
                row = {field: self._coerce(raw.get(field)) for field in ANALYSIS_EVENT_FIELDS}
                for field, value in raw.items():
                    if field in row or field in ANALYSIS_EVENT_ARRAY_FIELDS or isinstance(value, (list, dict, tuple)):
                        continue
                    row[field] = self._coerce(value)
                row["rollout_id"] = rollout_id
                record = manifest.get(str(rollout_id), {})
                row.update({field: record.get(field) for field in ANALYSIS_RECORD_FIELDS})
                rows.append(row)
        return rows

    def _localization_event_metrics(
        self,
        path: Path,
        manifest: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                row = {
                    field: self._coerce(raw.get(field))
                    for field in ANALYSIS_LOCALIZATION_EVENT_FIELDS
                    if field in raw
                }
                for field, value in raw.items():
                    if field in row or isinstance(value, (list, dict, tuple)):
                        continue
                    row[field] = self._coerce(value)
                rollout_id = raw.get("rollout_id")
                row["rollout_id"] = rollout_id
                record = manifest.get(str(rollout_id), {})
                for field in ANALYSIS_RECORD_FIELDS:
                    if row.get(field) is None and record.get(field) is not None:
                        row[field] = record.get(field)
                rows.append(row)
        return rows

    def _latest_event_triggered_snapshot(self) -> tuple[Path, dict[str, Any]] | None:
        if not self.analysis_root.is_dir():
            return None
        required = (
            "metadata.json",
            *EVENT_TRIGGERED_TABLE_FILES.values(),
        )
        candidates = []
        for metadata_path in self.analysis_root.glob("*/metadata.json"):
            directory = metadata_path.parent
            if not all((directory / name).is_file() for name in required):
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("analysis") != "lf3r_event_triggered_signal_analysis":
                continue
            candidates.append((metadata_path.stat().st_mtime, directory, metadata))
        if not candidates:
            return None
        _, directory, metadata = max(candidates, key=lambda item: item[0])
        return directory, metadata

    def _event_triggered_response(self) -> dict[str, Any]:
        selected = self._latest_event_triggered_snapshot()
        if selected is None:
            return {
                "available": False,
                "message": (
                    "No complete event-triggered signal snapshot found under "
                    "outputs/baseline_signal_analysis."
                ),
            }
        directory, metadata = selected
        metadata_path = directory / "metadata.json"
        manifest = self._manifest()
        generated_at = str(
            metadata.get("generated_at")
            or metadata.get("completed_at")
            or self._iso_mtime(metadata_path)
        )
        current_manifest_hash = self._sha256(self.manifest_path)
        snapshot_manifest_hash = metadata.get("manifest_sha256")
        counts = metadata.get("counts") or {}
        snapshot_rollout_count = int(
            counts.get("rollouts")
            or len(metadata.get("rollouts") or [])
            or 0
        )
        latest_annotation_update = self._latest_annotation_update()
        selection_path = None
        if metadata.get("selection"):
            selection_path = Path(str(metadata["selection"]))
            if not selection_path.is_absolute():
                selection_path = self.project_root / selection_path
        stale = bool(
            snapshot_manifest_hash
            and snapshot_manifest_hash != current_manifest_hash
        )
        if latest_annotation_update and latest_annotation_update > generated_at:
            stale = True
        tables = {
            name: self._read_csv(directory / filename)
            for name, filename in EVENT_TRIGGERED_TABLE_FILES.items()
        }
        return {
            "available": True,
            "source": {
                "directory": self._relative(directory),
                "metadata": self._relative(metadata_path),
                "generated_at": generated_at,
                "selection": (
                    self._relative(selection_path) if selection_path else None
                ),
                "selection_count": snapshot_rollout_count,
                "plots": [
                    self._relative(directory / str(plot))
                    for plot in (metadata.get("plots") or [])
                    if (directory / str(plot)).is_file()
                ],
            },
            "freshness": {
                "stale": stale,
                "manifest_matches": snapshot_manifest_hash == current_manifest_hash,
                "snapshot_manifest_sha256": snapshot_manifest_hash,
                "current_manifest_sha256": current_manifest_hash,
                "snapshot_rollout_count": snapshot_rollout_count,
                "current_rollout_count": len(manifest),
                "latest_annotation_update": latest_annotation_update,
            },
            "parameters": metadata.get("parameters") or {},
            "methods": metadata.get("methods") or list(ANALYSIS_BASELINE_METHODS),
            "signals_by_method": metadata.get("signals_by_method") or {},
            "event_group_counts": metadata.get("event_group_counts") or {},
            "counts": counts,
            "native_sampling": metadata.get("native_sampling") or {},
            "source_runs": metadata.get("source_runs") or {},
            "method_coverage": tables["method_coverage"],
            "curves": tables["curves"],
            "change_scores": tables["change_scores"],
            "separation": tables["separation"],
            "summary": tables["summary"],
            "peak_events": tables["peak_events"],
            "controls": tables["controls"],
        }

    @staticmethod
    def _is_robo_hop_metadata(metadata: Mapping[str, Any]) -> bool:
        """Recognize this analysis semantically instead of by one display label."""
        script = str(metadata.get("script") or "")
        signal = metadata.get("signal") or {}
        signal_name = str(signal.get("name") or "")
        return (
            script.endswith("tools/analyze_robo_dopamine_incremental_hop.py")
            or signal_name in {
                "Robo-Dopamine incremental hop",
                "Robo-Dopamine four-mode hop comparison",
                "Robo-Dopamine fused hop failure detection",
            }
            or (
                "Robo-Dopamine" in signal_name
                and "hop" in signal_name.lower()
                and isinstance(metadata.get("input"), dict)
            )
        )

    def _robo_hop_snapshot_candidates(
        self,
    ) -> tuple[list[tuple[float, Path, dict[str, Any]]], dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "root": self._relative(self.robo_hop_root),
            "metadata_found": 0,
            "invalid_metadata": 0,
            "wrong_analysis_type": 0,
            "incomplete_artifacts": [],
        }
        if not self.robo_hop_root.is_dir():
            diagnostics["root_missing"] = True
            return [], diagnostics

        candidates: list[tuple[float, Path, dict[str, Any]]] = []
        # Include finalized top-level snapshots and complete web-job outputs.
        # metadata.json is written only after the analysis tables, so a nested
        # .web_jobs/<id>/output directory with the complete required artifact
        # set is safe to render even if server finalization/move was interrupted.
        for metadata_path in self.robo_hop_root.rglob("metadata.json"):
            diagnostics["metadata_found"] += 1
            directory = metadata_path.parent
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                diagnostics["invalid_metadata"] += 1
                continue
            if not isinstance(metadata, dict) or not self._is_robo_hop_metadata(metadata):
                diagnostics["wrong_analysis_type"] += 1
                continue
            missing = [
                name
                for name in ROBO_HOP_REQUIRED_FILES
                if not (directory / name).is_file()
            ]
            if missing:
                diagnostics["incomplete_artifacts"].append({
                    "directory": self._relative(directory),
                    "missing": missing,
                })
                continue
            try:
                mtime = metadata_path.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, directory, metadata))

        diagnostics["eligible_snapshots"] = len(candidates)
        return candidates, diagnostics

    def _latest_robo_hop_snapshot(self) -> tuple[Path, dict[str, Any]] | None:
        candidates, _diagnostics = self._robo_hop_snapshot_candidates()
        if not candidates:
            return None
        _, directory, metadata = max(candidates, key=lambda item: item[0])
        return directory, metadata

    def _robo_hop_response(self) -> dict[str, Any]:
        selected = self._latest_robo_hop_snapshot()
        if selected is None:
            _candidates, discovery = self._robo_hop_snapshot_candidates()
            details: list[str] = []
            if discovery.get("root_missing"):
                details.append("analysis output root does not exist")
            metadata_found = int(discovery.get("metadata_found") or 0)
            if metadata_found:
                details.append(f"{metadata_found} metadata file(s) discovered")
            incomplete = discovery.get("incomplete_artifacts") or []
            if incomplete:
                preview = incomplete[0]
                missing = ", ".join(preview.get("missing") or [])
                details.append(
                    f"incomplete snapshot {preview.get('directory')}: missing {missing}"
                )
                if len(incomplete) > 1:
                    details.append(f"{len(incomplete) - 1} additional incomplete snapshot(s)")
            wrong_type = int(discovery.get("wrong_analysis_type") or 0)
            if wrong_type:
                details.append(f"{wrong_type} metadata file(s) belong to another analysis type")
            invalid = int(discovery.get("invalid_metadata") or 0)
            if invalid:
                details.append(f"{invalid} invalid metadata file(s)")
            suffix = (" · " + " · ".join(details)) if details else ""
            return {
                "available": False,
                "message": (
                    "No complete Robo-Dopamine fused-hop failure analysis snapshot found."
                    + suffix
                ),
                "discovery": discovery,
            }
        directory, metadata = selected
        metadata_path = directory / "metadata.json"
        generated_at = str(
            metadata.get("generated_at")
            or self._iso_mtime(metadata_path)
        )
        current_manifest_hash = self._sha256(self.manifest_path)
        input_metadata = metadata.get("input") or {}
        snapshot_manifest_hash = input_metadata.get("manifest_sha256")
        latest_annotation_update = self._latest_annotation_update()
        stale = bool(
            snapshot_manifest_hash
            and snapshot_manifest_hash != current_manifest_hash
        )
        if latest_annotation_update and latest_annotation_update > generated_at:
            stale = True
        best_configs = self._read_csv(directory / ROBO_HOP_TABLE_FILES["best_configs"])
        sweep_summary = self._read_csv(directory / ROBO_HOP_TABLE_FILES["sweep_summary"])
        recovery_results = self._read_csv(directory / ROBO_HOP_TABLE_FILES["recovery_results"])
        breakdown_summary = self._read_csv(directory / ROBO_HOP_TABLE_FILES["breakdown_summary"])
        ensemble_sweep_path = directory / ROBO_HOP_TABLE_FILES["ensemble_sweep"]
        ensemble_selected_path = directory / ROBO_HOP_TABLE_FILES["ensemble_selected"]
        ensemble_failure_path = (
            directory / ROBO_HOP_TABLE_FILES["ensemble_by_failure_type"]
        )
        ensemble_sweep = (
            self._read_csv(ensemble_sweep_path)
            if ensemble_sweep_path.is_file()
            else []
        )
        ensemble_selected = (
            self._read_csv(ensemble_selected_path)
            if ensemble_selected_path.is_file()
            else []
        )
        ensemble_by_failure_type = (
            self._read_csv(ensemble_failure_path)
            if ensemble_failure_path.is_file()
            else []
        )
        interval_localization_path = (
            directory / ROBO_HOP_TABLE_FILES["interval_localization"]
        )
        interval_localization_rows = (
            self._read_csv(interval_localization_path)
            if interval_localization_path.is_file()
            else []
        )
        interval_localization_top = [
            row
            for row in interval_localization_rows
            if row.get("rank_mse") is not None
            and int(row["rank_mse"]) <= 10
        ]
        task_cv_path = directory / "task_cv_results.csv"
        selected_configs = [
            row for row in best_configs
            if row.get("selection_status") == "selected"
        ]
        families = sorted({
            str(row.get("detector_family"))
            for row in best_configs
            if row.get("detector_family")
        })
        signal_modes = sorted({
            str(row.get("signal_mode"))
            for row in best_configs
            if row.get("signal_mode")
        })
        return {
            "available": True,
            "source": {
                "directory": self._relative(directory),
                "metadata": self._relative(metadata_path),
                "generated_at": generated_at,
                "run_root": input_metadata.get("run_root"),
                "selection": input_metadata.get("selection"),
            },
            "freshness": {
                "stale": stale,
                "manifest_matches": snapshot_manifest_hash == current_manifest_hash,
                "snapshot_manifest_sha256": snapshot_manifest_hash,
                "current_manifest_sha256": current_manifest_hash,
                "latest_annotation_update": latest_annotation_update,
            },
            "signal": metadata.get("signal") or {},
            "counts": metadata.get("counts") or {},
            "counts_by_signal_mode": metadata.get("counts_by_signal_mode") or {},
            "generalization": metadata.get("generalization") or {},
            "detector_config_n": (
                metadata.get("detector_config_n_total")
                or metadata.get("detector_config_n")
            ),
            "detector_config_n_per_signal": metadata.get(
                "detector_config_n_per_signal"
            ),
            "selected_config_n": metadata.get("selected_config_n"),
            "families": families,
            "signal_modes": signal_modes,
            "best_configs": best_configs,
            "selected_configs": selected_configs,
            "sweep_summary": sweep_summary,
            "recovery_results": recovery_results,
            "breakdown_summary": breakdown_summary,
            "ensemble_sweep": ensemble_sweep,
            "ensemble_selected": ensemble_selected,
            "ensemble_by_failure_type": ensemble_by_failure_type,
            "interval_localization_ranking": metadata.get(
                "interval_localization_ranking"
            ) or {},
            "interval_localization_rows": interval_localization_rows,
            "interval_localization_top": interval_localization_top,
            "search_cache": metadata.get("search_cache") or {},
            "phenotype_detector": metadata.get("phenotype_detector") or {},
            "grasp_failure_diagnosis": metadata.get(
                "grasp_failure_diagnosis"
            ) or {},
            "task_cv_available": task_cv_path.is_file(),
            "artifacts": [
                {
                    "name": name,
                    "url": "/api/analysis/artifacts/" + name,
                }
                for name in (
                    "sweep_summary.csv",
                    "event_results.csv",
                    "clean_rollout_results.csv",
                    "best_configs.csv",
                    "recovery_results.csv",
                    "breakdown_summary.csv",
                    "no_event_failure_results.csv",
                    "ensemble_sweep.csv",
                    "ensemble_selected.csv",
                    "ensemble_by_failure_type.csv",
                                                                                                                    "interval_localization_ranking.csv",
                    "offline_localization_diagnostics.csv",
                    "grasp_event_features.csv",
                    "grasp_detected_vs_missed.csv",
                    "grasp_matched_control.csv",
                    "grasp_matched_control_summary.csv",
                    "grasp_failure_categories.csv",
                    "grasp_failure_category_summary.csv",
                    "grasp_event_heatmap.png",
                    "task_cv_results.csv",
                )
                if (directory / name).is_file()
            ],
        }

    def _latest_outcome_snapshot(self) -> tuple[Path, dict[str, Any]] | None:
        if not self.analysis_root.is_dir():
            return None
        candidates = []
        for metadata_path in self.analysis_root.glob("*/metadata.json"):
            directory = metadata_path.parent
            if not all((directory / name).is_file() for name in ROLLOUT_OUTCOME_REQUIRED_FILES):
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("analysis") != "lf3r_rollout_outcome_evaluation":
                continue
            candidates.append((metadata_path.stat().st_mtime, directory, metadata))
        if not candidates:
            return None
        _, directory, metadata = max(candidates, key=lambda item: item[0])
        return directory, metadata

    def _outcome_response(self) -> dict[str, Any]:
        selected = self._latest_outcome_snapshot()
        if selected is None:
            return {
                "available": False,
                "message": "No dedicated rollout-outcome snapshot is available.",
            }
        directory, metadata = selected
        metadata_path = directory / "metadata.json"
        generated_at = str(
            metadata.get("generated_at")
            or metadata.get("completed_at")
            or self._iso_mtime(metadata_path)
        )
        current_manifest_hash = self._sha256(self.manifest_path)
        snapshot_manifest_hash = metadata.get("manifest_sha256")
        latest_annotation_update = self._latest_annotation_update()
        stale = bool(
            snapshot_manifest_hash
            and snapshot_manifest_hash != current_manifest_hash
        )
        if latest_annotation_update and latest_annotation_update > generated_at:
            stale = True
        summary_path = directory / ROLLOUT_OUTCOME_TABLE_FILES["summary"]
        predictions_path = directory / ROLLOUT_OUTCOME_TABLE_FILES["predictions"]
        coverage_path = directory / "method_coverage.csv"
        summary = self._read_csv(summary_path)
        coverage = self._read_csv(coverage_path)
        return {
            "available": bool(summary),
            "source": {
                "directory": self._relative(directory),
                "metadata": self._relative(metadata_path),
                "generated_at": generated_at,
                "selection_count": int(
                    (metadata.get("counts") or {}).get("rollouts")
                    or len(metadata.get("rollouts") or [])
                    or 0
                ),
            },
            "freshness": {
                "stale": stale,
                "manifest_matches": snapshot_manifest_hash == current_manifest_hash,
                "snapshot_manifest_sha256": snapshot_manifest_hash,
                "current_manifest_sha256": current_manifest_hash,
                "latest_annotation_update": latest_annotation_update,
            },
            "summary": summary,
            "predictions_available": predictions_path.is_file(),
            "method_coverage": coverage,
            "config": metadata.get("rollout_outcome_classification") or {},
        }

    def _latest_change_point_snapshot(self) -> tuple[Path, dict[str, Any]] | None:
        if not self.analysis_root.is_dir():
            return None
        required = (
            "metadata.json",
            "changepoint_event_metrics.jsonl",
            *CHANGEPOINT_TABLE_FILES.values(),
        )
        candidates = []
        for metadata_path in self.analysis_root.glob("*/metadata.json"):
            directory = metadata_path.parent
            if not all((directory / name).is_file() for name in required):
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("analysis") != "lf3r_baseline_change_points":
                continue
            candidates.append((metadata_path.stat().st_mtime, directory, metadata))
        if not candidates:
            return None
        _, directory, metadata = max(candidates, key=lambda item: item[0])
        return directory, metadata

    def _change_point_event_metrics(
        self,
        path: Path,
        manifest: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                row = {
                    field: self._coerce(raw.get(field))
                    for field in CHANGEPOINT_EVENT_FIELDS
                    if field in raw
                }
                rollout_id = raw.get("rollout_id")
                row["rollout_id"] = rollout_id
                record = manifest.get(str(rollout_id), {})
                for field in ANALYSIS_RECORD_FIELDS:
                    if row.get(field) is None and record.get(field) is not None:
                        row[field] = record.get(field)
                rows.append(row)
        return rows

    def _change_point_response(self) -> dict[str, Any]:
        selected = self._latest_change_point_snapshot()
        if selected is None:
            return {
                "available": False,
                "message": "No complete local change-point snapshot found under outputs/baseline_signal_analysis.",
            }
        directory, metadata = selected
        manifest = self._manifest()
        metadata_path = directory / "metadata.json"
        generated_at = str(
            metadata.get("generated_at")
            or metadata.get("completed_at")
            or self._iso_mtime(metadata_path)
        )
        current_manifest_hash = self._sha256(self.manifest_path)
        snapshot_manifest_hash = metadata.get("manifest_sha256")
        counts = metadata.get("counts") or {}
        snapshot_rollout_count = int(
            counts.get("rollouts")
            or len(metadata.get("rollouts") or [])
            or 0
        )
        latest_annotation_update = self._latest_annotation_update()
        selection_path = None
        if metadata.get("selection"):
            selection_path = Path(str(metadata["selection"]))
            if not selection_path.is_absolute():
                selection_path = self.project_root / selection_path
        stale = bool(snapshot_manifest_hash and snapshot_manifest_hash != current_manifest_hash)
        if latest_annotation_update and latest_annotation_update > generated_at:
            stale = True
        tables = {
            name: self._read_csv(directory / filename)
            for name, filename in CHANGEPOINT_TABLE_FILES.items()
        }
        events = self._change_point_event_metrics(
            directory / "changepoint_event_metrics.jsonl",
            manifest,
        )
        localization_tables = {
            "summary": (
                self._read_csv(directory / "localization_summary.csv")
                if (directory / "localization_summary.csv").is_file()
                else tables["summary"]
            ),
            "thresholds": (
                self._read_csv(directory / "localization_thresholds.csv")
                if (directory / "localization_thresholds.csv").is_file()
                else tables["scales"]
            ),
            "by_failure_type": (
                self._read_csv(directory / "localization_by_failure_type.csv")
                if (directory / "localization_by_failure_type.csv").is_file()
                else tables["by_failure_type"]
            ),
        }
        comparison_path = directory / "comparison_with_full_136_20260827.csv"
        comparison = self._read_csv(comparison_path) if comparison_path.is_file() else []
        rollout_outcome_summary_path = directory / ROLLOUT_OUTCOME_TABLE_FILES["summary"]
        rollout_outcome_predictions_path = directory / ROLLOUT_OUTCOME_TABLE_FILES["predictions"]
        rollout_outcome_summary = (
            self._read_csv(rollout_outcome_summary_path)
            if rollout_outcome_summary_path.is_file()
            else []
        )
        return {
            "available": True,
            "source": {
                "directory": self._relative(directory),
                "metadata": self._relative(metadata_path),
                "generated_at": generated_at,
                "selection": self._relative(selection_path) if selection_path else None,
                "selection_count": snapshot_rollout_count,
                "comparison": self._relative(comparison_path) if comparison_path.is_file() else None,
            },
            "freshness": {
                "stale": stale,
                "manifest_matches": snapshot_manifest_hash == current_manifest_hash,
                "snapshot_manifest_sha256": snapshot_manifest_hash,
                "current_manifest_sha256": current_manifest_hash,
                "snapshot_rollout_count": snapshot_rollout_count,
                "current_rollout_count": len(manifest),
                "latest_annotation_update": latest_annotation_update,
            },
            "parameters": {
                "pre_window_frames": metadata.get("pre_window_frames"),
                "post_window_frames": metadata.get("post_window_frames"),
                "min_samples_per_side": metadata.get("min_samples_per_side"),
                "reference_exclusion_radius_frames": metadata.get("reference_exclusion_radius_frames"),
                "frame_coordinate": metadata.get("frame_coordinate"),
                "native_sampling_preserved": metadata.get("native_sampling_preserved"),
                "threshold_calibration": metadata.get("threshold_calibration"),
            },
            "methods": metadata.get("methods") or list(ANALYSIS_BASELINE_METHODS),
            "features": metadata.get("features") or [],
            "feature_labels": metadata.get("feature_labels") or {},
            "local_scales_frames": metadata.get("local_scales_frames") or [],
            "thresholds": metadata.get("thresholds") or [],
            "tolerances_frames": metadata.get("tolerances_frames") or [],
            "direction_used_after_detection": metadata.get("direction_used_after_detection") or {},
            "counts": counts,
            "method_coverage": tables["method_coverage"],
            "summary": tables["summary"],
            "reference_summary": tables["reference_summary"],
            "by_failure_type": tables["by_failure_type"],
            "scales": tables["scales"],
            "localization_available": True,
            "localization_summary": localization_tables["summary"],
            "localization_thresholds": localization_tables["thresholds"],
            "localization_by_failure_type": localization_tables["by_failure_type"],
            "localization_event_metrics": events,
            "localization": {
                "available": True,
                "summary": localization_tables["summary"],
                "thresholds": localization_tables["thresholds"],
                "by_failure_type": localization_tables["by_failure_type"],
                "event_metrics": events,
            },
            "event_metrics": events,
            "rollout_outcome_available": bool(rollout_outcome_summary),
            "rollout_outcome_summary": rollout_outcome_summary,
            "rollout_outcome_predictions_available": rollout_outcome_predictions_path.is_file(),
            "rollout_outcome": metadata.get("rollout_outcome_classification") or {},
            "comparison": comparison,
        }

