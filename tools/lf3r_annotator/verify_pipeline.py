#!/usr/bin/env python3
"""Requirement-level verifier for the LF3R failure-data pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"


def check(name: str, passed: bool, details: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def media_is_valid(project_root: Path, record: dict[str, Any]) -> bool:
    try:
        video = (project_root / record["video_path"]).resolve()
        video.relative_to(project_root.resolve())
    except (KeyError, ValueError):
        return False
    if not video.is_file():
        return False
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    try:
        frames = int(subprocess.check_output(command, text=True).strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False
    return frames == int(record.get("total_frames", -1))


def verify(project_root: Path, manifest_path: Path) -> dict[str, Any]:
    records = read_manifest(manifest_path)
    primary = [record for record in records if record.get("dataset_role") == "primary_natural"]
    controlled = [
        record for record in records if record.get("source_kind") == "controlled_injected"
    ]
    natural = [record for record in records if record.get("source_kind") == "natural_policy"]
    outcomes = {record.get("ground_truth_outcome") for record in primary}
    required_tool_files = [
        project_root / "tools/lf3r_annotator/server.py",
        project_root / "tools/lf3r_annotator/static/index.html",
        project_root / "tools/lf3r_annotator/static/app.js",
        project_root / "tools/lf3r_annotator/build_manifest.py",
        project_root / "tools/lf3r_annotator/generate_libero10_natural.sh",
        project_root / "tools/lf3r_annotator/generate_libero_spatial_native.sh",
        project_root / "tools/lf3r_annotator/annotation.schema.json",
        project_root / "tools/lf3r_annotator/README.md",
    ]
    checkpoint_config = (
        project_root / "checkpoints/openvla-7b-finetuned-libero-10/config.json"
    )
    spatial_checkpoint_config = (
        project_root / "checkpoints/openvla-7b-finetuned-libero-spatial/config.json"
    )
    checkpoint_ok = False
    spatial_checkpoint_ok = False
    if checkpoint_config.is_file():
        checkpoint_ok = "libero_10" in json.loads(
            checkpoint_config.read_text(encoding="utf-8")
        ).get("norm_stats", {})
    if spatial_checkpoint_config.is_file():
        spatial_checkpoint_ok = "libero_spatial" in json.loads(
            spatial_checkpoint_config.read_text(encoding="utf-8")
        ).get("norm_stats", {})
    spatial_assets_ok = (
        project_root / "repos/LIBERO/libero/libero/bddl_files/libero_spatial"
    ).is_dir() and (
        project_root / "repos/LIBERO/libero/libero/init_files/libero_spatial"
    ).is_dir()
    partition_ok = all(
        record.get("analysis_partition") == "controlled_analysis"
        and record.get("dataset_role") == "controlled_analysis"
        and isinstance(record.get("injection"), dict)
        for record in controlled
    ) and all(
        record.get("analysis_partition") == "natural_observation"
        and record.get("injection") is None
        for record in natural
    )
    results = [
        check(
            "primary_backbone",
            bool(primary) and all(record.get("task_suite") == "libero_10" for record in primary),
            f"{len(primary)} primary natural LIBERO-10 rollouts",
        ),
        check(
            "small_natural_set",
            3 <= len(primary) <= 12,
            f"expected 3-12 primary natural rollouts, found {len(primary)}",
        ),
        check(
            "natural_success_and_failure",
            outcomes == {"success", "failure"},
            "primary outcomes: " + ", ".join(sorted(str(value) for value in outcomes)),
        ),
        check(
            "strict_injection_partition",
            partition_ok and bool(controlled),
            f"{len(natural)} natural and {len(controlled)} controlled records checked",
        ),
        check(
            "media_integrity",
            bool(records) and all(media_is_valid(project_root, record) for record in records),
            f"{len(records)} manifest media records checked",
        ),
        check(
            "libero10_checkpoint",
            checkpoint_ok,
            "dedicated checkpoint contains libero_10 normalization statistics",
        ),
        check(
            "libero_spatial_native_pipeline",
            spatial_checkpoint_ok and spatial_assets_ok,
            "Spatial checkpoint contains libero_spatial normalization statistics and LIBERO-Spatial assets are installed",
        ),
        check(
            "annotation_tool",
            all(path.is_file() for path in required_tool_files),
            f"{sum(path.is_file() for path in required_tool_files)}/{len(required_tool_files)} required files present",
        ),
        check(
            "project_local_storage",
            manifest_path.resolve().is_relative_to(project_root.resolve())
            and (project_root / "annotations/failure_annotations/v1").is_dir(),
            "manifest and annotation root are under PROJECT_ROOT",
        ),
    ]
    return {
        "status": "complete" if all(item["passed"] for item in results) else "incomplete",
        "manifest": str(manifest_path),
        "rollouts": len(records),
        "requirements": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify LF3R failure-data pipeline")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify(args.project_root.resolve(), args.manifest.resolve())
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
