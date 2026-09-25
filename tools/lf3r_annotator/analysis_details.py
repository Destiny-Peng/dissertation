"""Dashboard detail queries, artifacts, and specialized analysis responses."""

from __future__ import annotations

import copy
import json
import math
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
    ANALYSIS_LOCALIZATION_TABLE_FILES,
    ANALYSIS_RECORD_FIELDS,
    ANALYSIS_TABLE_FILES,
    ROLLOUT_OUTCOME_TABLE_FILES,
    ROBO_LABEL_LOSS_REQUIRED_FILES,
    ROBO_LOCALIZATION_HEAD_REQUIRED_FILES,
)
from backend_core import ValidationError


class AnalysisDetailsMixin:
    @staticmethod
    def _query_value(query: dict[str, Any], key: str, default: str = "") -> str:
        value = query.get(key, default)
        if isinstance(value, list):
            value = value[0] if value else default
        return str(value) if value is not None else default

    @staticmethod
    def _same_detail_value(value: Any, wanted: str) -> bool:
        if wanted in {"", "all"}:
            return True
        if value is None:
            return False
        left = str(value)
        right = str(wanted)
        if left == right:
            return True
        try:
            return float(left) == float(right)
        except (TypeError, ValueError):
            return left.lower() == right.lower()

    def _latest_for_detail_kind(
        self, kind: str
    ) -> tuple[str, Path, dict[str, Any]] | None:
        snapshot_type, _filename = ANALYSIS_DETAIL_KINDS[kind]
        if snapshot_type == "change_point":
            selected = self._latest_change_point_snapshot()
        elif snapshot_type == "event_triggered":
            selected = self._latest_event_triggered_snapshot()
        else:
            selected = self._latest_snapshot()
        if selected is None:
            return None
        directory, metadata = selected
        return snapshot_type, directory, metadata

    def _detail_rows(
        self,
        kind: str,
        directory: Path,
        manifest: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        _snapshot_type, filename = ANALYSIS_DETAIL_KINDS[kind]
        path = directory / filename
        if not path.is_file():
            return []
        if filename.endswith(".csv"):
            return self._read_csv(path)
        if kind == "changepoint_events":
            return self._change_point_event_metrics(path, manifest)
        if kind == "localization_events":
            return self._localization_event_metrics(path, manifest)
        return self._event_metrics(path, manifest)

    @classmethod
    def _detail_matches(
        cls,
        row: dict[str, Any],
        filters: dict[str, str],
    ) -> bool:
        for field, wanted in filters.items():
            if wanted in {"", "all"}:
                continue
            if field == "task":
                values = (
                    row.get("task_id"),
                    row.get("task"),
                    row.get("task_description"),
                )
            elif field == "outcome":
                values = (row.get("outcome_group"), row.get("event_group"), row.get("outcome"))
            elif field == "failure_type":
                values = (row.get("failure_type"), row.get("failure"))
            elif field == "scale":
                values = (row.get("scale_frames"), row.get("scale"))
            else:
                values = (row.get(field),)
            if not any(cls._same_detail_value(value, wanted) for value in values):
                return False
        return True

    @staticmethod
    def _detail_sort_value(row: dict[str, Any], sort: str) -> Any:
        aliases = {
            "score": (
                "event_score", "onset_score", "peak_score",
                "normalized_response_magnitude", "strongest_separation_z",
            ),
            "event_score": ("event_score", "onset_score", "peak_score"),
            "localization_error": (
                "absolute_localization_error_frames",
                "localization_error_frames",
                "first_exceedance_absolute_error_frames",
                "peak_distance_frames",
            ),
            "peak_distance": ("peak_distance_frames", "median_peak_distance_frames"),
            "recall": ("recall", "event_hit_rate"),
            "false_alarm": (
                "false_alarm_rate", "clean_success_false_alarm_rate",
                "clean_success_false_alarm_count",
            ),
            "f1": ("f1",),
            "auroc": ("auroc",),
            "rollout": ("rollout_id",),
            "task": ("task_id", "task_description"),
        }
        for field in aliases.get(sort, ()):
            value = row.get(field)
            if value not in (None, ""):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return str(value).lower()
        return float("-inf") if sort != "rollout" else ""

    @staticmethod
    def _detail_columns(rows: list[dict[str, Any]]) -> list[str]:
        preferred = [
            "method", "signal", "feature", "scale_frames", "threshold",
            "outcome_group", "failure_type", "task_suite", "task_id",
            "rollout_id", "event_frame", "observable_onset_frame",
            "event_score", "recall", "false_alarm_rate",
            "clean_success_false_alarm_rate",
            "absolute_localization_error_frames", "f1", "auroc",
        ]
        keys = set()
        for row in rows:
            keys.update(
                key for key, value in row.items()
                if not isinstance(value, (list, dict, tuple))
            )
        return [key for key in preferred if key in keys] + sorted(
            keys.difference(preferred)
        )

    @classmethod
    def _dashboard_rows(
        cls,
        rows: list[dict[str, Any]],
        *,
        limit: int = 64,
        feature: str = "level",
        scale: str = "16",
        threshold: str = "q95",
        outcome: str = "all_events",
    ) -> list[dict[str, Any]]:
        if len(rows) <= limit:
            return rows
        exact = [
            row for row in rows
            if cls._same_detail_value(row.get("feature"), feature)
            and cls._same_detail_value(row.get("scale_frames"), scale)
            and cls._same_detail_value(
                row.get("threshold") or row.get("threshold_name"), threshold
            )
            and (
                outcome == "all_events"
                or cls._same_detail_value(row.get("outcome_group"), outcome)
            )
        ]
        return exact[:limit] or rows[:limit]

    def _live_summary(self) -> dict[str, Any]:
        records = self._manifest()
        outcomes = {
            "clean_success": 0,
            "recovered_success": 0,
            "terminal_failure": 0,
            "uncertain": 0,
        }
        by_partition: dict[str, int] = {}
        annotated = 0
        failure_events = 0
        observable_events = 0
        for record in records.values():
            partition = str(record.get("analysis_partition") or "unknown")
            by_partition[partition] = by_partition.get(partition, 0) + 1
            annotation_path = self.annotation_root / "records" / f"{record['id']}.json"
            annotation: dict[str, Any] = {}
            try:
                if annotation_path.is_file():
                    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                annotation = {}
            if annotation:
                annotated += 1
            label = str(annotation.get("outcome_label") or record.get("ground_truth_outcome") or "uncertain")
            if label in {"success", "clean_success"}:
                outcome = "clean_success"
            elif label == "recovered_success":
                outcome = "recovered_success"
            elif label in {"failure", "terminal_failure"}:
                outcome = "terminal_failure"
            else:
                outcome = "uncertain"
            outcomes[outcome] += 1
            events = annotation.get("failure_events") or []
            failure_events += len(events)
            observable_events += sum(
                1 for event in events
                if event.get("observable_onset_frame") is not None
            )
        resolved = outcomes["clean_success"] + outcomes["recovered_success"] + outcomes["terminal_failure"]
        return {
            "rollouts": len(records),
            "annotated": annotated,
            "resolved": resolved,
            "resolved_success_rate": (
                (outcomes["clean_success"] + outcomes["recovered_success"]) / resolved
                if resolved else None
            ),
            "outcomes": outcomes,
            "failure_events": failure_events,
            "observable_onset_events": observable_events,
            "by_partition": by_partition,
        }

    def _artifact_links(self) -> list[dict[str, Any]]:
        selected = [
            ("rollout_outcome", self._latest_outcome_snapshot()),
            ("change_point", self._latest_change_point_snapshot()),
        ]
        links: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source_type, result in selected:
            if result is None:
                continue
            directory, _metadata = result
            for name in sorted(ANALYSIS_ARTIFACT_NAMES):
                if name in seen:
                    continue
                candidate = directory / name
                if candidate.is_file():
                    seen.add(name)
                    links.append({
                        "name": name,
                        "source": source_type,
                        "path": self._relative(candidate),
                        "url": "/api/analysis/artifacts/" + name,
                    })
        return links

    def _compact_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact = copy.deepcopy(payload)
        compact["dashboard"] = True
        compact["default_scope"] = "libero_10"
        compact["available_tabs"] = ["outcome", "localization"]
        compact["artifact_links"] = self._artifact_links()
        compact["live"] = self._live_summary()
        detail_counts: dict[str, int] = {}
        for kind, (snapshot_type, _filename) in ANALYSIS_DETAIL_KINDS.items():
            if snapshot_type == "change_point":
                node = compact.get("change_point") or {}
                count_key = {
                    "changepoint_summary": "summary",
                    "changepoint_failure_types": "by_failure_type",
                    "changepoint_events": "event_metrics",
                    "localization_summary": "localization_summary",
                    "localization_thresholds": "localization_thresholds",
                    "localization_failure_types": "localization_by_failure_type",
                    "localization_events": "localization_event_metrics",
                    "changepoint_comparison": "comparison",
                }.get(kind)
            elif snapshot_type == "event_triggered":
                node = compact.get("event_triggered") or {}
                count_key = {
                    "event_triggered_summary": "summary",
                    "event_triggered_peaks": "peak_events",
                    "event_triggered_curves": "curves",
                    "event_triggered_change_scores": "change_scores",
                }.get(kind)
            else:
                node = compact
                count_key = "event_metrics"
            value = node.get(count_key, []) if count_key else []
            detail_counts[kind] = len(value) if isinstance(value, list) else 0
        compact["detail_counts"] = detail_counts

        change_point = compact.get("change_point")
        if isinstance(change_point, dict):
            for key in ("summary", "reference_summary", "by_failure_type", "scales",
                        "localization_summary", "localization_thresholds",
                        "localization_by_failure_type"):
                if isinstance(change_point.get(key), list):
                    change_point[key] = self._dashboard_rows(change_point[key])
            localization = change_point.get("localization")
            if isinstance(localization, dict):
                for key in ("summary", "thresholds", "by_failure_type"):
                    if isinstance(localization.get(key), list):
                        localization[key] = self._dashboard_rows(localization[key])
            for key in ("event_metrics", "localization_event_metrics"):
                if isinstance(change_point.get(key), list) and len(change_point[key]) > 64:
                    change_point.pop(key, None)
            if isinstance(localization, dict) and isinstance(localization.get("event_metrics"), list):
                if len(localization["event_metrics"]) > 64:
                    localization.pop("event_metrics", None)

        for key in ("event_metrics", "localization_event_metrics"):
            if isinstance(compact.get(key), list) and len(compact[key]) > 64:
                compact.pop(key, None)
        event_triggered = compact.get("event_triggered")
        if isinstance(event_triggered, dict):
            for key in ("summary", "peak_events"):
                if isinstance(event_triggered.get(key), list):
                    event_triggered[key] = event_triggered[key][:64]
            for key in ("curves", "change_scores", "separation", "controls"):
                if isinstance(event_triggered.get(key), list) and len(event_triggered[key]) > 240:
                    event_triggered.pop(key, None)
        for key in (
            "summary_by_method_signal_outcome",
            "summary_by_method_outcome",
            "onset_signal_statistics",
            "clean_background_summary",
        ):
            if isinstance(compact.get(key), list) and len(compact[key]) > 64:
                compact[key] = compact[key][:64]
        compact["compatibility"] = {
            "full_endpoint": "/api/analysis?view=full",
            "note": "The full compatibility response contains event-level arrays; dashboard mode keeps them behind /api/analysis/details.",
        }
        return compact

    def details(self, query: dict[str, Any]) -> dict[str, Any]:
        allowed = {"kind", "page", "page_size", "sort"} | ANALYSIS_DETAIL_FILTERS
        unknown = set(query) - allowed
        if unknown:
            raise ValidationError("Unknown analysis details field(s): " + ", ".join(sorted(unknown)))
        kind = self._query_value(query, "kind", "changepoint_summary")
        if kind not in ANALYSIS_DETAIL_KINDS:
            raise ValidationError("Unsupported analysis details kind")
        try:
            page = int(self._query_value(query, "page", "1"))
            page_size = int(self._query_value(query, "page_size", "25"))
        except ValueError as exc:
            raise ValidationError("page and page_size must be integers") from exc
        if page < 1:
            raise ValidationError("page must be at least 1")
        if page_size < 1 or page_size > 100:
            raise ValidationError("page_size must be between 1 and 100")
        sort = self._query_value(query, "sort", "score")
        if sort not in ANALYSIS_DETAIL_SORTS:
            raise ValidationError("Unsupported analysis details sort")
        filters = {
            field: self._query_value(query, field)
            for field in ANALYSIS_DETAIL_FILTERS
            if self._query_value(query, field) not in {"", "all"}
        }
        selected = self._latest_for_detail_kind(kind)
        if selected is None:
            return {
                "available": False,
                "kind": kind,
                "page": page,
                "page_size": page_size,
                "total": 0,
                "page_count": 0,
                "items": [],
                "columns": [],
                "message": "No compatible analysis snapshot is available.",
            }
        snapshot_type, directory, metadata = selected
        rows = self._detail_rows(kind, directory, self._manifest())
        rows = [row for row in rows if self._detail_matches(row, filters)]
        reverse = sort not in {"rollout", "task"}
        rows.sort(
            key=lambda row: self._detail_sort_value(row, sort),
            reverse=reverse,
        )
        total = len(rows)
        start = (page - 1) * page_size
        items = rows[start:start + page_size]
        source = {
            "directory": self._relative(directory),
            "generated_at": metadata.get("generated_at") or self._iso_mtime(directory / "metadata.json"),
            "kind": kind,
            "snapshot_type": snapshot_type,
        }
        return {
            "available": bool((directory / ANALYSIS_DETAIL_KINDS[kind][1]).is_file()),
            "kind": kind,
            "source": source,
            "page": page,
            "page_size": page_size,
            "page_count": math.ceil(total / page_size) if total else 0,
            "total": total,
            "filters": filters,
            "sort": sort,
            "columns": self._detail_columns(items or rows[:1]),
            "items": items,
        }

    def artifact_path(self, name: str) -> Path:
        if name not in ANALYSIS_ARTIFACT_NAMES or "/" in name or "\\" in name:
            raise ValidationError("Unsupported analysis artifact")
        for _source_type, result in (
            ("change_point", self._latest_change_point_snapshot()),
            ("event_triggered", self._latest_event_triggered_snapshot()),
            ("legacy", self._latest_snapshot()),
            ("robo_incremental_hop", self._latest_robo_hop_snapshot()),
        ):
            if result is None:
                continue
            directory, _metadata = result
            candidate = (directory / name).resolve()
            try:
                candidate.relative_to(directory.resolve())
            except ValueError as exc:
                raise ValidationError("Analysis artifact escapes its snapshot") from exc
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(name)

    def _full_response(self) -> dict[str, Any]:
        """Return only the analysis families still exposed by the current UI.

        Outcome Evaluation uses its dedicated snapshot. Rule-based failure
        localization uses the latest change-point snapshot. Learned
        localization and Robo-Dopamine fused-hop already have dedicated API
        endpoints, so legacy temporal/event-triggered snapshots are deliberately
        not loaded here.
        """
        outcome_snapshot = self._outcome_response()
        change_point = self._change_point_response()

        dedicated_outcome_summary = outcome_snapshot.get("summary", [])
        change_point_outcome_summary = change_point.get(
            "rollout_outcome_summary", []
        )
        rollout_outcome_summary = (
            dedicated_outcome_summary
            if dedicated_outcome_summary
            else change_point_outcome_summary
        )
        rollout_outcome_metadata = (
            outcome_snapshot.get("config", {})
            if dedicated_outcome_summary
            else change_point.get("rollout_outcome", {})
        )
        rollout_outcome_predictions_available = (
            bool(outcome_snapshot.get("predictions_available"))
            if dedicated_outcome_summary
            else bool(change_point.get("rollout_outcome_predictions_available"))
        )

        localization = change_point.get("localization")
        if not isinstance(localization, dict):
            localization = {
                "available": False,
                "summary": [],
                "thresholds": [],
                "by_failure_type": [],
                "event_metrics": [],
            }

        has_analysis_artifact = bool(
            outcome_snapshot.get("available")
            or change_point.get("available")
        )
        primary_source = (
            change_point.get("source")
            if change_point.get("available")
            else outcome_snapshot.get("source")
        )
        primary_freshness = (
            change_point.get("freshness")
            if change_point.get("available")
            else outcome_snapshot.get("freshness")
        )

        return {
            "available": has_analysis_artifact,
            "temporal_available": False,
            "legacy_temporal_available": False,
            "message": (
                None
                if has_analysis_artifact
                else "No current outcome or change-point analysis snapshot is available."
            ),
            "source": primary_source,
            "freshness": primary_freshness or {},
            "parameters": change_point.get("parameters", {}),
            "methods": list(ANALYSIS_BASELINE_METHODS),
            "orientation": {},
            "signal_units": {},
            "method_coverage": change_point.get("method_coverage", []),
            "summary_by_method_signal_outcome": [],
            "summary_by_method_outcome": [],
            "onset_signal_statistics": [],
            "clean_background_summary": [],
            "event_metrics": [],
            "rollout_outcome_available": bool(rollout_outcome_summary),
            "rollout_outcome_summary": rollout_outcome_summary,
            "rollout_outcome_predictions_available": (
                rollout_outcome_predictions_available
            ),
            "rollout_outcome": rollout_outcome_metadata,
            "rollout_outcome_snapshot": outcome_snapshot,
            "localization_available": bool(localization.get("available")),
            "localization_summary": localization.get("summary", []),
            "localization_thresholds": localization.get("thresholds", []),
            "localization_by_failure_type": localization.get(
                "by_failure_type", []
            ),
            "localization_event_metrics": localization.get(
                "event_metrics", []
            ),
            "localization": localization,
            "change_point_available": bool(change_point.get("available")),
            "change_point": change_point,
            "event_triggered_available": False,
            "event_triggered": {
                "available": False,
                "message": "Legacy event-triggered analysis is not part of the current Analysis UI.",
            },
            "robo_hop_available": False,
            "robo_hop": {
                "available": False,
                "message": "Robo-Dopamine fused-hop uses /api/analysis/robo-hop.",
            },
            "robo_incremental_hop_available": False,
            "robo_incremental_hop": {
                "available": False,
                "message": "Robo-Dopamine fused-hop uses /api/analysis/robo-hop.",
            },
            "primary_analysis_type": (
                "change_point"
                if change_point.get("available")
                else (
                    "rollout_outcome"
                    if outcome_snapshot.get("available")
                    else None
                )
            ),
            "primary_analysis_source": primary_source,
        }


    def _latest_robo_localization_head_snapshot(
        self,
    ) -> tuple[Path, dict[str, Any]] | None:
        root = self.robo_localization_head_root
        if not root.is_dir():
            return None
        candidates: list[tuple[float, Path, dict[str, Any]]] = []
        for metadata_path in root.rglob("metadata.json"):
            directory = metadata_path.parent
            if not all(
                (directory / name).is_file()
                for name in ROBO_LOCALIZATION_HEAD_REQUIRED_FILES
            ):
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("analysis") != "robo_dopamine_bilstm_success_negative_ablation":
                continue
            candidates.append((metadata_path.stat().st_mtime, directory, metadata))
        if not candidates:
            return None
        _, directory, metadata = max(candidates, key=lambda item: item[0])
        return directory, metadata

    def robo_localization_head_response(self) -> dict[str, Any]:
        selected = self._latest_robo_localization_head_snapshot()
        if selected is None:
            return {
                "available": False,
                "message": "No completed BiLSTM success-negative training snapshot yet.",
            }
        directory, metadata = selected
        return {
            "available": True,
            "source": {
                "directory": self._relative(directory),
                "generated_at": metadata.get("generated_at"),
                "source_root": (
                    metadata.get("source_root")
                    or metadata.get("run_root")
                ),
                "selection_mode": metadata.get("selection_mode"),
            },
            "metadata": metadata,
            "comparison": self._read_csv(directory / "ablation_comparison.csv"),
            "delta": self._read_csv(directory / "ablation_delta.csv"),
            "per_split": self._read_csv(directory / "per_split_metrics.csv"),
            "artifacts": [
                {
                    "name": name,
                    "url": "/api/analysis/robo-localization-head/artifacts/" + name,
                }
                for name in ROBO_LOCALIZATION_HEAD_REQUIRED_FILES
                if (directory / name).is_file()
            ],
        }

    def robo_localization_head_artifact_path(self, name: str) -> Path:
        if name not in ROBO_LOCALIZATION_HEAD_REQUIRED_FILES:
            raise ValidationError("Unsupported localization-head artifact")
        selected = self._latest_robo_localization_head_snapshot()
        if selected is None:
            raise FileNotFoundError(name)
        directory, _metadata = selected
        path = (directory / name).resolve()
        path.relative_to(directory.resolve())
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    def _latest_robo_label_loss_snapshot(
        self,
    ) -> tuple[Path, dict[str, Any]] | None:
        root = self.robo_label_loss_root
        if not root.is_dir():
            return None
        candidates: list[tuple[float, Path, dict[str, Any]]] = []
        for metadata_path in root.rglob("metadata.json"):
            directory = metadata_path.parent
            if not all(
                (directory / name).is_file()
                for name in ROBO_LABEL_LOSS_REQUIRED_FILES
            ):
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("analysis") != "robo_dopamine_bilstm_label_loss_ablation":
                continue
            candidates.append((metadata_path.stat().st_mtime, directory, metadata))
        if not candidates:
            return None
        _, directory, metadata = max(candidates, key=lambda item: item[0])
        return directory, metadata

    def robo_label_loss_response(self) -> dict[str, Any]:
        selected = self._latest_robo_label_loss_snapshot()
        if selected is None:
            return {
                "available": False,
                "message": "No completed BiLSTM label/loss ablation snapshot yet.",
            }
        directory, metadata = selected
        summary_path = directory / "dataset_target_summary.json"
        best_path = directory / "best_configuration.json"
        try:
            dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            dataset_summary = {}
        try:
            best_configuration = json.loads(best_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            best_configuration = {}
        return {
            "available": True,
            "source": {
                "directory": self._relative(directory),
                "generated_at": metadata.get("generated_at"),
                "source_root": metadata.get("source_root"),
                "selection_mode": metadata.get("selection_mode"),
            },
            "metadata": metadata,
            "dataset_summary": dataset_summary,
            "best_configuration": best_configuration,
            "label_ablation": self._read_csv(directory / "label_ablation.csv"),
            "loss_ablation": self._read_csv(directory / "loss_ablation.csv"),
            "per_split": self._read_csv(directory / "per_split_metrics.csv"),
            "artifacts": [
                {
                    "name": name,
                    "url": "/api/analysis/robo-label-loss/artifacts/" + name,
                }
                for name in ROBO_LABEL_LOSS_REQUIRED_FILES
                if (directory / name).is_file()
            ],
        }

    def robo_label_loss_artifact_path(self, name: str) -> Path:
        if name not in ROBO_LABEL_LOSS_REQUIRED_FILES:
            raise ValidationError("Unsupported label/loss ablation artifact")
        selected = self._latest_robo_label_loss_snapshot()
        if selected is None:
            raise FileNotFoundError(name)
        directory, _metadata = selected
        path = (directory / name).resolve()
        path.relative_to(directory.resolve())
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    def robo_hop_response(self) -> dict[str, Any]:
        """Return the latest Robo-Dopamine fused-hop snapshot directly."""
        return self._robo_hop_response()

    def response(self, compact: bool = True) -> dict[str, Any]:
        payload = self._full_response()
        return self._compact_response(payload) if compact else payload
