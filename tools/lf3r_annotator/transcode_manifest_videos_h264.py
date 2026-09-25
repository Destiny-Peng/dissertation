#!/usr/bin/env python3
"""Batch-transcode all manifest camera_video_paths to browser-compatible H.264."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        required=True,
        help="Manifest JSONL to scan; repeat for multiple loaded manifests.",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    raw = Path(value).expanduser()
    path = raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {value}") from exc
    return path


def read_manifest_video_paths(
    project_root: Path,
    manifests: list[Path],
) -> list[Path]:
    videos: list[Path] = []
    seen: set[Path] = set()
    for manifest in manifests:
        path = resolve_project_path(project_root, manifest)
        if not path.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest}")
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    row: Any = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {path} line {line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"manifest row must be an object: {path} line {line_number}"
                    )
                camera_paths = row.get("camera_video_paths")
                if not isinstance(camera_paths, dict) or not camera_paths:
                    raise ValueError(
                        f"manifest row has no valid camera_video_paths: "
                        f"{path} line {line_number}"
                    )
                for camera, value in camera_paths.items():
                    if not isinstance(camera, str) or not camera.strip():
                        raise ValueError(
                            f"manifest row has an invalid camera key: "
                            f"{path} line {line_number}"
                        )
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(
                            f"manifest row has no path for camera {camera!r}: "
                            f"{path} line {line_number}"
                        )
                    video = resolve_project_path(project_root, value.strip())
                    if video.suffix.lower() != ".mp4":
                        raise ValueError(
                            f"camera_video_paths[{camera!r}] is not an .mp4 file: {value}"
                        )
                    if video not in seen:
                        seen.add(video)
                        videos.append(video)
    return videos


def probe_codec(video: Path) -> str:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or "ffprobe failed"
        raise RuntimeError(message)
    return completed.stdout.strip().splitlines()[0].strip().lower() if completed.stdout.strip() else ""


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()

    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            print(f"ERROR missing required executable: {binary}", file=sys.stderr)
            return 2

    manifests = [resolve_project_path(project_root, path) for path in args.manifest]
    videos = read_manifest_video_paths(project_root, manifests)
    transcode_script = project_root / "tools" / "lf3r_annotator" / "transcode_video_h264.sh"
    if not transcode_script.is_file():
        print(f"ERROR transcode script is missing: {transcode_script}", file=sys.stderr)
        return 2

    print(f"BATCH_H264_MANIFESTS count={len(manifests)}")
    for manifest in manifests:
        print(f"BATCH_H264_MANIFEST path={manifest.relative_to(project_root)}")
    print(f"BATCH_H264_SELECTED unique_videos={len(videos)}")

    converted = 0
    already_h264 = 0
    conflicts = 0
    missing = 0
    failed = 0

    for index, video in enumerate(videos, 1):
        relative = video.relative_to(project_root)
        print(f"BATCH_H264_VIDEO index={index}/{len(videos)} path={relative}", flush=True)

        if not video.is_file():
            missing += 1
            print(f"BATCH_H264_MISSING path={relative}", file=sys.stderr, flush=True)
            continue

        try:
            codec = probe_codec(video)
        except Exception as exc:
            failed += 1
            print(
                f"BATCH_H264_PROBE_FAILED path={relative} error={exc}",
                file=sys.stderr,
                flush=True,
            )
            continue

        if codec == "h264":
            already_h264 += 1
            print(f"BATCH_H264_SKIP_H264 path={relative}", flush=True)
            continue

        backup = video.with_name(video.stem + ".orig.mp4")
        if backup.exists():
            conflicts += 1
            print(
                f"BATCH_H264_CONFLICT path={relative} "
                f"backup={backup.relative_to(project_root)}",
                file=sys.stderr,
                flush=True,
            )
            continue

        completed = subprocess.run(
            ["/usr/bin/bash", str(transcode_script), str(video)],
            cwd=str(project_root),
            check=False,
        )
        if completed.returncode == 0:
            converted += 1
            print(f"BATCH_H264_CONVERTED path={relative}", flush=True)
        else:
            failed += 1
            print(
                f"BATCH_H264_FAILED path={relative} returncode={completed.returncode}",
                file=sys.stderr,
                flush=True,
            )

    print(
        "BATCH_H264_SUMMARY "
        f"selected={len(videos)} "
        f"converted={converted} "
        f"already_h264={already_h264} "
        f"conflicts={conflicts} "
        f"missing={missing} "
        f"failed={failed}",
        flush=True,
    )

    # Existing backups are intentional conflicts and missing files can happen
    # after a manifest changes; neither should discard successful conversions.
    # Only an actual ffprobe/transcode failure marks the project-tool job failed.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
