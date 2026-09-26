"""Baseline manifest coverage, run catalogs, and Robo signal inventory."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend_core import (
    INSTRUCTION_VARIANT_CONDITIONS,
    ValidationError,
    load_manifest_records,
    run_rollout_ids,
    select_scope_records,
    validate_instruction_condition,
    validate_run_scope,
)
from analysis_constants import ANALYSIS_BASELINE_METHODS, OUTCOME_EVALUATION_METHODS
from baseline_constants import (
    BASELINE_METHODS,
    BASELINE_RESULT_FILTERS,
    BASELINE_RUN_STATUSES,
    INSTRUCTION_VARIANT_LABELS,
)


class BaselineCatalogMixin:
    def _manifest_records(self) -> list[dict[str, Any]]:
        return load_manifest_records(self.manifest_path)

    def _manifest_for_condition(self, condition: str) -> Path:
        condition = validate_instruction_condition(condition)
        if condition == "full_instruction":
            return self.manifest_path
        if not self.variant_manifest_path.is_file():
            raise ValidationError(
                "Instruction-variant manifest is unavailable; run "
                "tools/prepare_libero10_instruction_variants.py first"
            )
        return self.variant_manifest_path

    def _condition_records(self, condition: str, scope: str) -> list[dict[str, Any]]:
        condition = validate_instruction_condition(condition)
        source_records = select_scope_records(self._manifest_records(), scope)
        if condition == "full_instruction":
            return source_records

        variant_records = load_manifest_records(self._manifest_for_condition(condition))
        by_source: dict[str, dict[str, Any]] = {}
        for record in variant_records:
            row_condition = record.get("instruction_variant") or record.get("condition")
            if row_condition != condition:
                continue
            source_id = record.get("source_rollout_id") or record.get("source_id")
            if isinstance(source_id, str):
                by_source[source_id] = record
        return [
            by_source[record["id"]]
            for record in source_records
            if record["id"] in by_source
        ]

    @staticmethod
    def _validate_result_filter(value: Any) -> str:
        result_filter = str(value or "all").strip()
        if result_filter not in BASELINE_RESULT_FILTERS:
            raise ValidationError(
                "result_filter must be one of: " + ", ".join(sorted(BASELINE_RESULT_FILTERS))
            )
        return result_filter

    @staticmethod
    def _source_rollout_id(record: dict[str, Any], condition: str) -> str:
        if condition == "full_instruction":
            return str(record["id"])
        return str(
            record.get("source_rollout_id")
            or record.get("source_id")
            or record["id"]
        )

    def _complete_annotation_records(
        self,
        records: list[dict[str, Any]],
        condition: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Keep only source rollouts whose annotation review_status is complete."""
        condition = validate_instruction_condition(condition)
        if self.annotation_store is None:
            raise ValidationError("Annotation store is unavailable for result filtering")
        complete: list[dict[str, Any]] = []
        incomplete_source_ids: list[str] = []
        for record in records:
            source_id = self._source_rollout_id(record, condition)
            annotation = self.annotation_store.read(source_id)
            if annotation and annotation.get("review_status") == "complete":
                complete.append(record)
            else:
                incomplete_source_ids.append(source_id)
        return complete, incomplete_source_ids

    def _result_source_map(
        self,
        baseline: str,
        condition: str,
        records: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Resolve each rollout to its newest parseable completed baseline run.

        Run candidates are already ordered newest-first by BaselineRunIndex.
        A rollout is therefore assigned once, to the first parseable output,
        and later/older runs only fill still-missing rollout IDs.
        """
        if baseline not in BASELINE_METHODS:
            raise ValidationError("Invalid baseline method")
        condition = validate_instruction_condition(condition)
        allowed_conditions = (
            {condition, "unknown"} if condition == "full_instruction" else {condition}
        )
        remaining = {str(record["id"]): record for record in records}
        resolved: dict[str, str] = {}
        for run_path, metadata in self._run_candidates(baseline):
            if self._run_instruction_condition(metadata) not in allowed_conditions:
                continue
            completed_ids = run_rollout_ids(run_path)
            overlap = set(remaining).intersection(completed_ids)
            if not overlap:
                continue
            run_summary = self._run_summary(run_path, metadata)
            for rollout_id in list(overlap):
                record = remaining[rollout_id]
                try:
                    self._read_method(
                        baseline,
                        run_path,
                        record,
                        dict(run_summary),
                    )
                except (
                    FileNotFoundError,
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    json.JSONDecodeError,
                ):
                    continue
                resolved[rollout_id] = self._relative(run_path)
                remaining.pop(rollout_id, None)
            if not remaining:
                break
        return resolved

    def _valid_result_rollout_ids(
        self,
        baseline: str,
        condition: str,
        records: list[dict[str, Any]],
    ) -> set[str]:
        """Return record IDs with at least one parseable completed baseline output."""
        return set(self._result_source_map(baseline, condition, records))

    def outcome_evaluation_sources(
        self,
        scope: Any = "libero_10",
    ) -> dict[str, Any]:
        """Resolve latest per-rollout outputs independently for each baseline."""
        scope = validate_run_scope(scope)
        condition = "full_instruction"
        scope_records = self._condition_records(condition, scope)
        records, incomplete_source_ids = self._complete_annotation_records(
            scope_records,
            condition,
        )
        source_maps: dict[str, dict[str, str]] = {}
        coverage: list[dict[str, Any]] = []
        for baseline in OUTCOME_EVALUATION_METHODS:
            mapping = self._result_source_map(baseline, condition, records)
            source_maps[baseline] = mapping
            coverage.append({
                "method": baseline,
                "evaluation_population": len(records),
                "available_rollouts": len(mapping),
                "missing_rollouts": len(records) - len(mapping),
            })
        return {
            "scope": scope,
            "condition": condition,
            "scope_rollouts": len(scope_records),
            "evaluation_population": len(records),
            "incomplete_annotation_rollouts": len(incomplete_source_ids),
            "evaluation_rollout_ids": [str(record["id"]) for record in records],
            "source_maps": source_maps,
            "coverage": coverage,
        }

    def result_coverage(
        self,
        baseline: Any,
        scope: Any = "libero_10",
        condition: Any = "full_instruction",
    ) -> dict[str, Any]:
        baseline = str(baseline or "")
        if baseline not in BASELINE_METHODS:
            raise ValidationError("Invalid baseline method")
        scope = validate_run_scope(scope)
        condition = validate_instruction_condition(condition)
        scope_records = self._condition_records(condition, scope)
        records, incomplete_source_ids = self._complete_annotation_records(
            scope_records,
            condition,
        )
        valid_ids = self._valid_result_rollout_ids(baseline, condition, records)
        valid_source_ids = [
            self._source_rollout_id(record, condition)
            for record in records
            if str(record["id"]) in valid_ids
        ]
        missing_source_ids = [
            self._source_rollout_id(record, condition)
            for record in records
            if str(record["id"]) not in valid_ids
        ]
        return {
            "baseline": baseline,
            "scope": scope,
            "condition": condition,
            "scope_rollouts": len(scope_records),
            "matched_rollouts": len(records),
            "complete_annotation_rollouts": len(records),
            "incomplete_annotation_rollouts": len(incomplete_source_ids),
            "incomplete_source_rollout_ids": incomplete_source_ids,
            "valid_result_rollouts": len(valid_source_ids),
            "missing_valid_result_rollouts": len(missing_source_ids),
            "valid_source_rollout_ids": valid_source_ids,
            "missing_source_rollout_ids": missing_source_ids,
        }

    def _variant_record_for_source(
        self,
        source_rollout_id: str,
        condition: str,
    ) -> dict[str, Any]:
        records = self._condition_records(condition, "all")
        for record in records:
            source_id = record.get("source_rollout_id") or record.get("source_id")
            if source_id == source_rollout_id:
                return record
        raise ValidationError(
            f"No prepared {INSTRUCTION_VARIANT_LABELS[condition]} variant exists "
            f"for rollout {source_rollout_id}"
        )

    ROBO_HOP_SIGNAL_MODES = ("incremental", "forward", "backward", "fused")

    def _resolve_robo_signal_prediction(
        self,
        run_path: Path,
        rollout_id: str,
        signal_mode: str,
    ) -> Path | None:
        """Resolve one saved Robo-Dopamine hop source without changing its semantics."""
        if signal_mode not in self.ROBO_HOP_SIGNAL_MODES:
            return None
        worker_dir = run_path / "raw" / rollout_id
        result_path = worker_dir / "worker_result.json"
        if not result_path.is_file():
            return None
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(result, dict):
            return None

        recorded: Any = None
        perspectives = result.get("perspective_outputs")
        if signal_mode != "fused" and isinstance(perspectives, dict):
            mode_output = perspectives.get(signal_mode)
            if isinstance(mode_output, dict):
                recorded = mode_output.get("raw_model_output")

        fusion = result.get("fusion")
        if (
            recorded in (None, "")
            and signal_mode != "fused"
            and isinstance(fusion, dict)
        ):
            source_paths = fusion.get("source_prediction_paths")
            if isinstance(source_paths, dict):
                recorded = source_paths.get(signal_mode)

        eval_mode = str(result.get("eval_mode") or "").lower()
        eval_modes = [
            str(value).lower()
            for value in (result.get("eval_modes") or [])
        ]
        if (
            recorded in (None, "")
            and signal_mode != "fused"
            and (eval_mode == signal_mode or eval_modes == [signal_mode])
        ):
            recorded = result.get("raw_model_output")

        if signal_mode == "fused" and recorded in (None, ""):
            if result.get("fused_model_output"):
                recorded = result["fused_model_output"]
            elif isinstance(fusion, dict) and fusion.get("output_path"):
                recorded = fusion["output_path"]
            elif (
                result.get("multi_perspective")
                or eval_mode == "fused"
                or set(eval_modes) >= {"incremental", "forward", "backward"}
            ) and result.get("raw_model_output"):
                recorded = result["raw_model_output"]

        if recorded in (None, ""):
            metadata_path = worker_dir / "multi_perspective" / "metadata.json"
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metadata = None
                if isinstance(metadata, dict):
                    if signal_mode == "fused":
                        recorded = metadata.get("fused_path")
                    else:
                        prediction_paths = metadata.get("prediction_paths")
                        if isinstance(prediction_paths, dict):
                            recorded = prediction_paths.get(signal_mode)

        candidates: list[Path] = []
        if recorded not in (None, ""):
            path = Path(str(recorded)).expanduser()
            if path.is_absolute():
                candidates.append(path)
                parts = path.parts
                for anchor in ("outputs", "datasets", "annotations", "tools", "repos"):
                    if anchor in parts:
                        index = parts.index(anchor)
                        candidates.append(self.project_root.joinpath(*parts[index:]))
                        break
            else:
                candidates.extend(
                    (
                        worker_dir / path,
                        run_path / path,
                        self.project_root / path,
                    )
                )

        candidates.extend(
            path
            for path in worker_dir.rglob("pred_vllm.json")
            if signal_mode in path.parts
        )

        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                resolved.relative_to(self.project_root)
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                return resolved
        return None

    def _robo_run_signal_ids(
        self,
        run_path: Path,
        run_ids: set[str],
        signal_mode: str,
    ) -> set[str]:
        return {
            rollout_id
            for rollout_id in run_ids
            if self._resolve_robo_signal_prediction(
                run_path, rollout_id, signal_mode
            ) is not None
        }

    def _scan_robo_run_four_signal_ids(
        self,
        run_path: Path,
        run_ids: set[str],
    ) -> tuple[set[str], dict[str, set[str]]]:
        by_mode = {
            mode: self._robo_run_signal_ids(run_path, run_ids, mode)
            for mode in self.ROBO_HOP_SIGNAL_MODES
        }
        common = set(run_ids)
        for mode in self.ROBO_HOP_SIGNAL_MODES:
            common.intersection_update(by_mode[mode])
        return common, by_mode

    def _run_inventory(
        self,
        run_path: Path,
        method: str,
    ) -> tuple[set[str], dict[str, set[str]], set[str]]:
        """Read run membership from cache; probe only fused Robo output once."""
        cached = self.run_index.cached_rollout_details(run_path)
        if cached is not None:
            run_ids = set(cached["rollout_ids"])
            if method != "robo_dopamine":
                return run_ids, {}, set()
            raw_signal_ids = cached.get("robo_signal_ids")
            if isinstance(raw_signal_ids, dict) and "fused" in raw_signal_ids:
                fused_ids = {
                    str(value)
                    for value in raw_signal_ids.get("fused", [])
                    if isinstance(value, str)
                }
                return run_ids, {"fused": fused_ids}, fused_ids

        run_ids = run_rollout_ids(run_path)
        by_mode: dict[str, set[str]] = {}
        fused_ids: set[str] = set()
        if method == "robo_dopamine":
            fused_ids = self._robo_run_signal_ids(
                run_path, run_ids, "fused"
            )
            by_mode = {"fused": fused_ids}
        self.run_index.store_rollout_details(
            run_path,
            run_ids,
            by_mode if method == "robo_dopamine" else None,
        )
        return run_ids, by_mode, fused_ids

    def _robo_run_four_signal_ids(
        self,
        run_path: Path,
        run_ids: set[str],
    ) -> tuple[set[str], dict[str, set[str]]]:
        """Legacy explicit four-signal probe; not used by fused-only catalog."""
        return self._scan_robo_run_four_signal_ids(run_path, run_ids)

    def _robo_run_incremental_ids(
        self,
        run_path: Path,
        run_ids: set[str],
    ) -> set[str]:
        """Legacy explicit incremental probe; not used by fused-only catalog."""
        return self._robo_run_signal_ids(run_path, run_ids, "incremental")

    def list_runs(
        self,
        scope: Any = "libero_10",
        condition: Any = "full_instruction",
    ) -> list[dict[str, Any]]:
        scope = validate_run_scope(scope)
        condition = validate_instruction_condition(condition)
        selected_ids = {
            record["id"] for record in self._condition_records(condition, scope)
        }
        variant_by_id: dict[str, dict[str, Any]] = {}
        if condition != "full_instruction":
            variant_by_id = {
                record["id"]: record
                for record in load_manifest_records(self._manifest_for_condition(condition))
            }
        summaries = []
        for method in BASELINE_METHODS:
            for run_path, metadata in self._run_candidates(method):
                run_condition = self._run_instruction_condition(metadata)
                if condition == "full_instruction":
                    if run_condition not in {"full_instruction", "unknown"}:
                        continue
                elif run_condition != condition:
                    continue
                run_ids, signal_ids_by_mode, fused_inventory_ids = self._run_inventory(
                    run_path, method
                )
                missing = selected_ids - run_ids if run_ids else selected_ids
                source_ids = {
                    (
                        variant_by_id[rollout_id].get("source_rollout_id")
                        or variant_by_id[rollout_id].get("source_id")
                        or rollout_id
                    )
                    for rollout_id in run_ids
                    if rollout_id in variant_by_id
                }
                if condition == "full_instruction":
                    source_ids = set(run_ids)
                summary = self._run_summary(run_path, metadata)
                fused_ids: set[str] = set()
                if method == "robo_dopamine":
                    fused_ids = signal_ids_by_mode.get(
                        "fused", fused_inventory_ids
                    )
                summary.update({
                    "created_at": metadata.get("created_at"),
                    "manifest_sha256": metadata.get("manifest_sha256"),
                    "run_rollout_count": len(run_ids),
                    "run_rollout_ids": sorted(run_ids),
                    "run_source_rollout_ids": sorted(source_ids),
                    "selected_scope_rollouts": len(selected_ids),
                    "compatible": bool(selected_ids)
                    and not missing
                    and summary["selected_rollouts"] >= len(selected_ids),
                    "partial_compatible": bool(selected_ids.intersection(run_ids)),
                    "missing_rollouts": len(missing),
                    "scope": scope,
                    "instruction_condition": run_condition,
                    "fused_rollout_count": (
                        len(fused_ids) if method == "robo_dopamine" else None
                    ),
                    "fused_scope_rollout_count": (
                        len(selected_ids.intersection(fused_ids))
                        if method == "robo_dopamine"
                        else None
                    ),
                    "fused_missing_rollouts": (
                        len(selected_ids - fused_ids)
                        if method == "robo_dopamine"
                        else None
                    ),
                    "fused_scope_coverage": (
                        (
                            len(selected_ids.intersection(fused_ids))
                            / len(selected_ids)
                        )
                        if method == "robo_dopamine" and selected_ids
                        else None
                    ),
                    "fused_compatible": (
                        bool(selected_ids)
                        and selected_ids.issubset(fused_ids)
                        if method == "robo_dopamine"
                        else None
                    ),
                    "hop_signal_rollout_counts": (
                        {"fused": len(fused_ids)}
                        if method == "robo_dopamine"
                        else None
                    ),
                })
                summaries.append(summary)
        summaries.sort(
            key=lambda item: (
                str(item.get("baseline")),
                str(item.get("completed_at") or item.get("created_at") or ""),
            ),
            reverse=True,
        )
        return summaries

