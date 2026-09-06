#!/usr/bin/env python3
"""Build a versioned LF3R rollout manifest from OpenVLA/LIBERO outputs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAN_ROOTS = (
    PROJECT_ROOT / "outputs/openvla_libero",
    PROJECT_ROOT / "outputs/openvla_libero_spatial_native",
)
ROLLOUT_RE = re.compile(r"task(?P<task>\d+)--ep(?P<episode>\d+)--succ(?P<success>[01])\.mp4$")
INJECTIONS = {
    "lf3r-feasibility-freeze-t50": {
        "type": "action_freeze",
        "causal_onset_frame": 40,
        "environment_timestep": 50,
        "end_frame": None,
    },
    "lf3r-feasibility-gripper-stuck-open-t58": {
        "type": "gripper_stuck_open",
        "causal_onset_frame": 48,
        "environment_timestep": 58,
        "end_frame": None,
    },
    "lf3r-feasibility-translation-inverted-t50": {
        "type": "translation_inversion",
        "causal_onset_frame": 40,
        "environment_timestep": 50,
        "end_frame": None,
    },
    "lf3r-feasibility-delayed-release-t60": {
        "type": "temporary_gripper_release",
        "causal_onset_frame": 50,
        "environment_timestep": 60,
        "end_frame": 69,
    },
}


def probe_video(path: Path) -> tuple[int, float, float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    frame_text = stream.get("nb_read_frames") or stream.get("nb_frames")
    if not frame_text or frame_text == "N/A":
        raise RuntimeError(f"Could not determine frame count for {path}")
    numerator, denominator = stream["r_frame_rate"].split("/", 1)
    fps = float(numerator) / float(denominator)
    duration = float(stream.get("duration") or int(frame_text) / fps)
    return int(frame_text), fps, duration


def csv_timesteps(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "action/timestep" not in rows[0]:
        return None, None
    return int(float(rows[0]["action/timestep"])), int(float(rows[-1]["action/timestep"]))


def load_task_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(suite): {str(task_id): str(description) for task_id, description in tasks.items()}
        for suite, tasks in payload.items()
    }


def classify_run(run_name: str) -> tuple[str, str, dict[str, Any] | None]:
    if run_name in INJECTIONS:
        return "controlled_injected", "controlled_analysis", INJECTIONS[run_name]
    if "natural" in run_name:
        return "natural_policy", "natural_observation", None
    raise ValueError("unrecognized run provenance")


def stable_id(relative_video: str, suite: str, task: int, episode: int, source_kind: str) -> str:
    digest = hashlib.sha1(relative_video.encode("utf-8")).hexdigest()[:10]
    origin = "natural" if source_kind == "natural_policy" else "controlled"
    return f"{suite}-task{task:02d}-ep{episode:03d}-{origin}-{digest}"


def build_record(video: Path, project_root: Path, task_metadata: dict[str, dict[str, str]]) -> dict[str, Any] | None:
    match = ROLLOUT_RE.fullmatch(video.name)
    if not match:
        return None
    try:
        relative = video.resolve().relative_to(project_root.resolve())
    except ValueError:
        raise RuntimeError(f"Video escapes project root: {video}")
    suite = video.parent.name
    run_name = video.parent.parent.name
    try:
        source_kind, partition, injection = classify_run(run_name)
    except ValueError:
        print(f"Skipping unrecognized run provenance: {run_name}", file=sys.stderr)
        return None
    task = int(match.group("task"))
    episode = int(match.group("episode"))
    outcome = "success" if match.group("success") == "1" else "failure"
    frames, fps, duration = probe_video(video)
    csv_path = video.with_suffix(".csv")
    first_timestep, last_timestep = csv_timesteps(csv_path)
    if source_kind == "controlled_injected":
        dataset_role = "controlled_analysis"
    elif suite == "libero_10":
        dataset_role = "primary_natural"
    else:
        dataset_role = "reference_natural"
    description = task_metadata.get(suite, {}).get(str(task), f"{suite} task {task}")
    record = {
        "schema_version": 1,
        "id": stable_id(str(relative), suite, task, episode, source_kind),
        "task_suite": suite,
        "task_id": task,
        "episode_index": episode,
        "task_description": description,
        "ground_truth_outcome": outcome,
        "source_kind": source_kind,
        "analysis_partition": partition,
        "dataset_role": dataset_role,
        "video_path": str(relative),
        "csv_path": str(csv_path.resolve().relative_to(project_root.resolve())) if csv_path.exists() else None,
        "total_frames": frames,
        "fps": round(fps, 6),
        "duration_seconds": round(duration, 6),
        "first_environment_timestep": first_timestep,
        "last_environment_timestep": last_timestep,
        "run_name": run_name,
        "policy_family": "OpenVLA-7B",
        "policy_checkpoint": (
            "openvla-7b-finetuned-libero-10"
            if suite == "libero_10"
            else "openvla-7b-finetuned-libero-spatial"
        ),
        "generation_entrypoint": "experiments/robot/libero/run_libero_eval.py",
        "generated_at": dt.datetime.fromtimestamp(
            video.stat().st_mtime, tz=dt.timezone.utc
        ).isoformat(),
        "injection": injection,
    }
    return record


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LF3R rollout manifest")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        default=None,
        help="Output root to scan; repeat for multiple roots (defaults to both LF3R OpenVLA roots)",
    )
    parser.add_argument(
        "--task-metadata",
        type=Path,
        default=PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/task_metadata.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    scan_roots = (
        [path.resolve() for path in args.scan_root]
        if args.scan_root
        else [path.resolve() for path in DEFAULT_SCAN_ROOTS]
    )
    task_metadata = load_task_metadata(args.task_metadata)
    records = []
    seen_videos: set[Path] = set()
    for scan_root in scan_roots:
        for video in sorted(scan_root.rglob("*.mp4")):
            resolved_video = video.resolve()
            if resolved_video in seen_videos:
                continue
            seen_videos.add(resolved_video)
            record = build_record(video, project_root, task_metadata)
            if record:
                records.append(record)
    records.sort(
        key=lambda item: (
            item["dataset_role"],
            item["task_suite"],
            item["task_id"],
            item["episode_index"],
            item["id"],
        )
    )
    manifest_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    atomic_write(args.output, manifest_text)
    counts = Counter(
        (record["dataset_role"], record["ground_truth_outcome"]) for record in records
    )
    summary = {
        "schema_version": 1,
        "manifest": str(args.output.resolve().relative_to(project_root)),
        "total_rollouts": len(records),
        "counts": {
            f"{role}:{outcome}": count
            for (role, outcome), count in sorted(counts.items())
        },
        "strict_partitioning": {
            "natural_observation": "No action or environment intervention; use for natural success/failure analysis.",
            "controlled_analysis": "Known injected intervention; exclude from natural failure rates.",
        },
    }
    atomic_write(
        args.output.with_name("summary.json"),
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
