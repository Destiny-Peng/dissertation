#!/usr/bin/env python3
"""Prepare official LIBERO demonstrations for Repair / Synthetic Suffix.

This is an interruption-safe incremental importer. Each successfully prepared
or resumed demo is published to the manifest immediately, so Ctrl-World can use
completed demos while the remaining dataset is still being processed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py

import prepare_libero_manifest_core as core


PROJECT_ROOT = core.PROJECT_ROOT
DEFAULT_INPUT_ROOT = core.DEFAULT_INPUT_ROOT
DEFAULT_MANIFEST = core.DEFAULT_MANIFEST
DEFAULT_VIDEO_ROOT = core.DEFAULT_VIDEO_ROOT
CAMERA_DATASETS = core.CAMERA_DATASETS
DemoImportError = core.DemoImportError


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record_id = str(record.get("id") or "")
        if record_id:
            records[record_id] = record
    return records


def _sorted_records(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records.values(),
        key=lambda item: (
            str(item.get("task_suite") or ""),
            int(item.get("task_id") or 0),
            int(item.get("episode_index") or 0),
            str(item.get("id") or ""),
        ),
    )


def _publish_manifest(path: Path, records: dict[str, dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in _sorted_records(records)
    )
    core._atomic_text(path, text)


def _publish_summary(
    *,
    project_root: Path,
    input_root: Path,
    output: Path,
    task_suite: str,
    records: dict[str, dict[str, Any]],
) -> None:
    ordered = _sorted_records(records)
    suite_records = [r for r in ordered if r.get("task_suite") == task_suite]
    summary = {
        "schema_version": 1,
        "manifest": core._project_relative(project_root, output, "Repair manifest"),
        "source_root": core._project_relative(
            project_root, input_root, "Official LIBERO input root"
        ),
        "task_suite": task_suite,
        "tasks_found": len({int(r["task_id"]) for r in suite_records}),
        "demonstrations": len(suite_records),
        "ground_truth_outcome": "success",
        "alignment_rule": (
            "rgb[c] ~= observation(states[c+1]); continue with actions[c+1:]"
        ),
        "camera_views": list(CAMERA_DATASETS),
        "ctrl_prepared": True,
        "ctrl_controls": "one frame-aligned absolute pose/gripper control per RGB frame",
        "publication": "incremental_after_each_demo",
    }
    summary_path = output.with_name(output.stem.replace("manifest", "summary") + ".json")
    core._atomic_text(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )


def _record_for_demo(
    *,
    project_root: Path,
    task_suite: str,
    task: Any,
    task_id: int,
    demo_index: int,
    group: Any,
    group_name: str,
    source_relative: str,
    camera_paths: dict[str, Path],
    ctrl_controls_path: Path,
    alignment_path: Path,
    alignment_validation: dict[str, Any],
    frame_count: int,
    fps: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": f"{task_suite}-task{task_id:02d}-demo{demo_index:03d}-official",
        "task_suite": task_suite,
        "task_id": task_id,
        "episode_index": demo_index,
        "task_description": str(task.language),
        "ground_truth_outcome": "success",
        "source_kind": "official_demonstration",
        "analysis_partition": "official_demonstration",
        "dataset_role": f"{task_suite}_official_success",
        "camera_video_paths": {
            camera: core._project_relative(project_root, path, f"{camera} video")
            for camera, path in camera_paths.items()
        },
        "source_hdf5_path": source_relative,
        "trajectory_group": group_name,
        "trajectory_format": "libero_official_processed_hdf5",
        "total_frames": frame_count,
        "fps": float(fps),
        "duration_seconds": float(frame_count) / float(fps),
        "first_environment_timestep": 0,
        "last_environment_timestep": frame_count - 1,
        "official_demo": True,
        "model_xml_available": "model_file" in group.attrs,
        "rgb_alignment": {
            "condition_frame": "rgb[c]",
            "branch_state": "states[c+1]",
            "future_actions": "actions[c+1:]",
            "ctrl_control": "ctrl_controls[c]",
            "validated": bool(alignment_validation.get("passed")),
            "validation_method": alignment_validation["method"],
            "validation_path": core._project_relative(
                project_root, alignment_path, "offline alignment validation"
            ),
            "minimum_psnr": alignment_validation["minimum_psnr"],
            "sampled_cut_frames": alignment_validation["sampled_cut_frames"],
        },
        "ctrl_prepared": True,
        "ctrl_controls_path": core._project_relative(
            project_root, ctrl_controls_path, "prepared Ctrl controls"
        ),
        "ctrl_controls_format": "lf3r_ctrl_absolute_pose_v1",
        "ctrl_control_semantics": (
            "xyz + Euler XYZ + DROID-style gripper closure; "
            "one control aligned to every source RGB frame"
        ),
    }


def build_manifest(
    *,
    project_root: Path,
    input_root: Path,
    task_suite: str,
    output: Path,
    video_root: Path,
    task_id: int | None = None,
    max_demos: int | None = None,
    fps_override: float | None = None,
    alignment_min_psnr: float = 0.0,
    resume: bool = False,
    reset_manifest: bool = False,
) -> list[dict[str, Any]]:
    project_root = project_root.expanduser().resolve()
    input_root = input_root.expanduser().resolve()
    output = output.expanduser().resolve()
    video_root = video_root.expanduser().resolve()

    if not input_root.is_dir():
        raise DemoImportError(f"Official LIBERO input root does not exist: {input_root}")
    core._project_relative(project_root, input_root, "Official LIBERO input root")
    core._project_relative(project_root, output, "Repair manifest output")
    core._project_relative(project_root, video_root, "Repair source-video root")

    suite = core._load_suite(task_suite, project_root)
    task_count = int(suite.get_num_tasks())
    if task_id is not None and not 0 <= int(task_id) < task_count:
        raise DemoImportError(f"--task-id must be in [0, {task_count - 1}]")
    if max_demos is not None and int(max_demos) < 1:
        raise DemoImportError("--max-demos must be at least 1")

    records = {} if reset_manifest else _load_existing(output)
    task_ids = [int(task_id)] if task_id is not None else list(range(task_count))
    completed_this_run = 0

    if reset_manifest:
        _publish_manifest(output, records)

    for current_task_id in task_ids:
        task = suite.get_task(current_task_id)
        expected_relative = suite.get_task_demonstration(current_task_id)
        hdf5_path = core._find_task_hdf5(input_root, expected_relative)
        if hdf5_path is None:
            continue
        source_relative = core._project_relative(
            project_root, hdf5_path, "Official LIBERO HDF5"
        )

        with h5py.File(hdf5_path, "r") as handle:
            if "data" not in handle:
                raise DemoImportError(f"Official LIBERO HDF5 has no data group: {hdf5_path}")
            data = handle["data"]
            fps = core._fps_from_hdf5(data, fps_override)
            demo_names = sorted(
                [
                    name
                    for name in data.keys()
                    if hasattr(data[name], "keys")
                    and "states" in data[name]
                    and "actions" in data[name]
                ],
                key=core._demo_sort_key,
            )
            if not demo_names:
                raise DemoImportError(f"No demonstrations found in {hdf5_path}")

            for fallback_index, demo_name in enumerate(demo_names):
                if max_demos is not None and completed_this_run >= int(max_demos):
                    return _sorted_records(records)

                group = data[demo_name]
                frame_count = core._validate_demo_group(
                    group, hdf5_path, f"data/{demo_name}"
                )
                demo_index = core._demo_index(demo_name, fallback_index)
                group_name = f"data/{demo_name}"

                camera_paths = core._materialize_demo_videos(
                    group=group,
                    video_root=video_root,
                    suite_name=task_suite,
                    task_id=current_task_id,
                    demo_index=demo_index,
                    source_relative=source_relative,
                    group_name=group_name,
                    frame_count=frame_count,
                    fps=fps,
                    resume=resume,
                )
                ctrl_controls_path, alignment_path, alignment_validation = (
                    core._materialize_ctrl_preparation(
                        project_root=project_root,
                        group=group,
                        camera_paths=camera_paths,
                        suite_name=task_suite,
                        task_id=current_task_id,
                        demo_index=demo_index,
                        frame_count=frame_count,
                        fps=fps,
                        alignment_min_psnr=alignment_min_psnr,
                        resume=resume,
                    )
                )

                record = _record_for_demo(
                    project_root=project_root,
                    task_suite=task_suite,
                    task=task,
                    task_id=current_task_id,
                    demo_index=demo_index,
                    group=group,
                    group_name=group_name,
                    source_relative=source_relative,
                    camera_paths=camera_paths,
                    ctrl_controls_path=ctrl_controls_path,
                    alignment_path=alignment_path,
                    alignment_validation=alignment_validation,
                    frame_count=frame_count,
                    fps=fps,
                )
                records[record["id"]] = record
                _publish_manifest(output, records)
                _publish_summary(
                    project_root=project_root,
                    input_root=input_root,
                    output=output,
                    task_suite=task_suite,
                    records=records,
                )
                completed_this_run += 1
                print(
                    "LF3R_REPAIR_LIBERO_DEMO_READY "
                    f"id={record['id']} manifest_records={len(records)}"
                )

    if completed_this_run == 0 and not records:
        raise DemoImportError(
            f"No official {task_suite} demonstrations were found under {input_root}"
        )
    return _sorted_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, help="Process only one LIBERO task ID.")
    parser.add_argument(
        "--max-demos",
        type=int,
        help="Stop after this many completed demos in the current invocation.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--fps", type=float)
    parser.add_argument(
        "--alignment-min-psnr",
        type=float,
        default=0.0,
        help="Optional RGB replay diagnostic threshold. Default 0 so it does not block preparation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed video/control/alignment artifacts when present.",
    )
    parser.add_argument(
        "--reset-manifest",
        action="store_true",
        help="Start a new manifest before processing selected demos; existing media/control artifacts are untouched.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_manifest(
        project_root=args.project_root,
        input_root=args.input_root,
        task_suite=str(args.task_suite),
        output=args.output,
        video_root=args.video_root,
        task_id=args.task_id,
        max_demos=args.max_demos,
        fps_override=args.fps,
        alignment_min_psnr=float(args.alignment_min_psnr),
        resume=bool(args.resume),
        reset_manifest=bool(args.reset_manifest),
    )
    print(
        "LF3R_REPAIR_LIBERO_MANIFEST "
        f"suite={args.task_suite} records={len(records)} output={args.output}"
    )


if __name__ == "__main__":
    main()
