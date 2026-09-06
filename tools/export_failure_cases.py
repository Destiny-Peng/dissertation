#!/usr/bin/env python3
"""Export selected LF3R annotations and their rollout media into a share package.

The default selection is complete primary_natural annotations whose human
outcome_label is "failure". Selection is joined to the versioned manifest by
rollout_id; evaluator filename suffixes such as succ0/succ1 are not used as
human labels.

The generated package contains:
  manifest.jsonl       package-local manifest with rewritten video_path values
  selection.json       analyzer-compatible list of selected rollout IDs
  index.json           provenance and per-case file mapping
  annotations/         selected annotation JSON files
  rollouts/<id>/       the matching video and same-stem sidecar files

The source project is never modified. Existing output directories are never
overwritten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "annotations/failure_annotations/v1/records"
DEFAULT_SHARE_ROOT = PROJECT_ROOT / "outputs/shares"

OUTCOME_LABELS = ("success", "failure", "recovered_success", "uncertain")
DATASET_ROLES = (
    "primary_natural",
    "reference_natural",
    "controlled_analysis",
    "all",
)
REVIEW_STATUSES = ("complete", "in_progress", "unreviewed", "all")
TIMING_FIELDS = (
    "causal_onset_frame",
    "observable_onset_frame",
    "terminal_failure_frame",
    "recovery_frame",
)


class ExportError(RuntimeError):
    """Raised when the requested share package cannot be created safely."""


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ExportError(f"Path must be inside the project root: {path}") from error
    return path


def relative_project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError as error:
        raise ExportError(f"Path is outside the project root: {path}") from error


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ExportError(f"Manifest does not exist: {relative_project_path(path)}")

    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExportError(f"Invalid manifest JSON at line {line_number}") from error
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ExportError(f"Manifest line {line_number} has no valid id")
        rollout_id = record["id"]
        if rollout_id in records:
            raise ExportError(f"Manifest contains duplicate rollout id: {rollout_id}")
        records[rollout_id] = record
    return records


def effective_event_count(annotation: dict[str, Any]) -> int:
    events = annotation.get("failure_events")
    if isinstance(events, list) and events:
        return len(events)
    if any(annotation.get(field) is not None for field in TIMING_FIELDS):
        return 1
    return 0


def load_cases(
    manifest: dict[str, dict[str, Any]],
    annotations_dir: Path,
    outcomes: set[str],
    dataset_role: str | None,
    review_status: str | None,
) -> list[dict[str, Any]]:
    if not annotations_dir.is_dir():
        raise ExportError(
            f"Annotation directory does not exist: {relative_project_path(annotations_dir)}"
        )

    cases: list[dict[str, Any]] = []
    seen_annotation_ids: set[str] = set()

    for annotation_path in sorted(annotations_dir.glob("*.json")):
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ExportError(
                f"Invalid annotation JSON: {relative_project_path(annotation_path)}"
            ) from error
        if not isinstance(annotation, dict):
            raise ExportError(
                f"Annotation is not an object: {relative_project_path(annotation_path)}"
            )

        rollout_id = annotation.get("rollout_id")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ExportError(
                f"Annotation has no valid rollout_id: {relative_project_path(annotation_path)}"
            )
        if rollout_id in seen_annotation_ids:
            raise ExportError(f"Duplicate annotation rollout_id: {rollout_id}")
        seen_annotation_ids.add(rollout_id)

        record = manifest.get(rollout_id)
        if record is None:
            raise ExportError(
                f"Annotation is not present in the manifest: {rollout_id}"
            )
        if annotation.get("outcome_label") not in outcomes:
            continue
        if review_status is not None and annotation.get("review_status") != review_status:
            continue
        if dataset_role is not None and record.get("dataset_role") != dataset_role:
            continue

        raw_video_path = record.get("video_path")
        if not isinstance(raw_video_path, str) or not raw_video_path:
            raise ExportError(f"Manifest record has no video_path: {rollout_id}")
        video_path = project_path(raw_video_path)
        if not video_path.is_file():
            raise ExportError(
                f"Video for {rollout_id} does not exist: {relative_project_path(video_path)}"
            )

        sidecars: list[Path] = []
        for candidate in sorted(video_path.parent.glob(video_path.stem + ".*")):
            if not candidate.is_file():
                continue
            resolved_candidate = candidate.resolve()
            try:
                resolved_candidate.relative_to(PROJECT_ROOT)
            except ValueError as error:
                raise ExportError(
                    f"Sidecar points outside project root: {candidate}"
                ) from error
            sidecars.append(resolved_candidate)
        if video_path.resolve() not in sidecars:
            sidecars.insert(0, video_path.resolve())

        cases.append(
            {
                "rollout_id": rollout_id,
                "annotation_path": annotation_path.resolve(),
                "annotation": annotation,
                "record": record,
                "video_path": video_path.resolve(),
                "sidecars": sidecars,
            }
        )

    cases.sort(key=lambda case: case["rollout_id"])
    return cases


def json_write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def copy_case(case: dict[str, Any], package_root: Path) -> dict[str, Any]:
    rollout_id = case["rollout_id"]
    case_root = package_root / "rollouts" / rollout_id
    case_root.mkdir(parents=True, exist_ok=False)

    copied_files: list[str] = []
    shared_video: str | None = None
    for source in case["sidecars"]:
        destination = case_root / source.name
        shutil.copy2(source, destination)
        relative_destination = destination.relative_to(package_root).as_posix()
        copied_files.append(relative_destination)
        if source == case["video_path"]:
            shared_video = relative_destination

    if shared_video is None:
        raise ExportError(f"Selected video was not copied for {rollout_id}")

    annotation_destination = package_root / "annotations" / f"{rollout_id}.json"
    shutil.copy2(case["annotation_path"], annotation_destination)

    manifest_record = dict(case["record"])
    manifest_record["video_path"] = shared_video

    return {
        "rollout_id": rollout_id,
        "task_suite": case["record"].get("task_suite"),
        "task_id": case["record"].get("task_id"),
        "episode_index": case["record"].get("episode_index"),
        "dataset_role": case["record"].get("dataset_role"),
        "analysis_partition": case["record"].get("analysis_partition"),
        "outcome_label": case["annotation"].get("outcome_label"),
        "failure_type": case["annotation"].get("failure_type"),
        "failure_event_count": effective_event_count(case["annotation"]),
        "annotation": annotation_destination.relative_to(package_root).as_posix(),
        "video": shared_video,
        "files": copied_files,
        "source_annotation": relative_project_path(case["annotation_path"]),
        "source_video": relative_project_path(case["video_path"]),
        "manifest_record": manifest_record,
    }


def write_package(
    output_dir: Path,
    cases: list[dict[str, Any]],
    manifest_path: Path,
    annotations_dir: Path,
    outcomes: set[str],
    dataset_role: str | None,
    review_status: str | None,
) -> None:
    if output_dir.exists():
        raise ExportError(
            f"Refusing to overwrite existing output directory: "
            f"{relative_project_path(output_dir)}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=str(output_dir.parent),
        )
    )
    try:
        (temporary_dir / "annotations").mkdir()
        (temporary_dir / "rollouts").mkdir()

        index_rows: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        for case in cases:
            copied = copy_case(case, temporary_dir)
            manifest_rows.append(copied.pop("manifest_record"))
            index_rows.append(copied)

        (temporary_dir / "manifest.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in manifest_rows
            ),
            encoding="utf-8",
        )
        json_write(
            temporary_dir / "selection.json",
            {
                "schema_version": 1,
                "scope": dataset_role or "all",
                "selection": [{"id": row["rollout_id"]} for row in index_rows],
            },
        )

        generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
        criteria = {
            "outcome_labels": sorted(outcomes),
            "dataset_role": dataset_role or "all",
            "review_status": review_status or "all",
        }
        json_write(
            temporary_dir / "index.json",
            {
                "schema_version": 1,
                "generated_at": generated_at,
                "criteria": criteria,
                "count": len(index_rows),
                "failure_event_count": sum(
                    int(row["failure_event_count"]) for row in index_rows
                ),
                "source_manifest": relative_project_path(manifest_path),
                "source_annotations": relative_project_path(annotations_dir),
                "cases": index_rows,
            },
        )

        readme = (
            "# LF3R selected rollout share\n\n"
            f"Generated at: {generated_at}\n"
            f"Selected rollouts: {len(index_rows)}\n"
            f"Selection criteria: {json.dumps(criteria, ensure_ascii=False)}\n\n"
            "Files:\n"
            "- manifest.jsonl: package-local manifest; video_path values point into this package.\n"
            "- selection.json: selected rollout IDs in analyzer-compatible format.\n"
            "- annotations/<rollout-id>.json: human annotation records.\n"
            "- rollouts/<rollout-id>/: rollout video and matching same-stem sidecar files.\n"
            "- index.json: source-to-package provenance and per-case mapping.\n\n"
            "The package contains selected annotation/media files only. It does not "
            "include baseline model raw outputs.\n"
        )
        (temporary_dir / "README.md").write_text(readme, encoding="utf-8")

        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export selected complete LF3R annotations and their rollout media. "
            "Default: primary_natural + outcome_label=failure."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Versioned manifest JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="Annotation record directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Share package directory. Defaults to "
            "outputs/shares/failure_primary_natural_<timestamp>."
        ),
    )
    parser.add_argument(
        "--outcome",
        action="append",
        choices=OUTCOME_LABELS,
        default=None,
        help=(
            "Human outcome label to include; repeat for multiple labels "
            "(default: failure)."
        ),
    )
    parser.add_argument(
        "--dataset-role",
        choices=DATASET_ROLES,
        default="primary_natural",
        help="Dataset role to include (default: primary_natural).",
    )
    parser.add_argument(
        "--review-status",
        choices=REVIEW_STATUSES,
        default="complete",
        help="Annotation review status (default: complete).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report the selection; do not create a package.",
    )
    parser.add_argument(
        "--print-ids",
        action="store_true",
        help="Print selected rollout IDs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outcomes = set(args.outcome or ["failure"])
    dataset_role = None if args.dataset_role == "all" else args.dataset_role
    review_status = None if args.review_status == "all" else args.review_status

    try:
        manifest_path = project_path(args.manifest)
        annotations_dir = project_path(args.annotations_dir)
        manifest = load_manifest(manifest_path)
        cases = load_cases(
            manifest,
            annotations_dir,
            outcomes,
            dataset_role,
            review_status,
        )
        if not cases:
            raise ExportError("No annotations matched the requested criteria")

        if args.output_dir is None:
            labels = "_".join(sorted(outcomes))
            role = dataset_role or "all"
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = DEFAULT_SHARE_ROOT / f"{labels}_{role}_{timestamp}"
        else:
            output_dir = project_path(args.output_dir)

        print(f"selected_rollouts={len(cases)}")
        print(
            "failure_events="
            + str(sum(effective_event_count(case["annotation"]) for case in cases))
        )
        print(f"output_dir={relative_project_path(output_dir)}")
        if args.print_ids:
            for case in cases:
                print(case["rollout_id"])

        if not args.dry_run:
            write_package(
                output_dir,
                cases,
                manifest_path,
                annotations_dir,
                outcomes,
                dataset_role,
                review_status,
            )
            print("export_status=complete")
        else:
            print("export_status=dry_run")
        return 0
    except (ExportError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
