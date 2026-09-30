#!/usr/bin/env python3
"""Incrementally prepare official LIBERO demos for Repair / Ctrl-World.

This wrapper reuses prepare_libero_manifest.py helpers, but publishes the
manifest after every successfully prepared or resumed demo. It is intended for
quick Ctrl-World iteration and interruption-safe preprocessing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py

import prepare_libero_manifest as base


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
    base._atomic_text(path, text)


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
            camera: base._project_relative(project_root, path, f"{camera} video")
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
            "validation_path": base._project_relative(
                project_root, alignment_path, "offline alignment validation"
            ),
            "minimum_psnr": alignment_validation["minimum_psnr"],
            "sampled_cut_frames": alignment_validation["sampled_cut_frames"],
        },
        "ctrl_prepared": True,
        "ctrl_controls_path": base._project_relative(
            project_root, ctrl_controls_path, "prepared Ctrl controls"
        ),
        "ctrl_controls_format": "lf3r_ctrl_absolute_pose_v1",
        "ctrl_control_semantics": (
            "xyz + Euler XYZ + DROID-style gripper closure; "
            "one control aligned to every source RGB frame"
        ),
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    project_root = args.project_root.expanduser().resolve()
    input_root = args.input_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()

    if not input_root.is_dir():
        raise base.DemoImportError(f"Official LIBERO input root does not exist: {input_root}")

    base._project_relative(project_root, input_root, "Official LIBERO input root")
    base._project_relative(project_root, output, "Repair manifest output")
    base._project_relative(project_root, video_root, "Repair source-video root")

    suite = base._load_suite(args.task_suite, project_root)
    existing = _load_existing(output)
    processed_this_run = 0

    task_ids = list(range(int(suite.get_num_tasks())))
    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= int(suite.get_num_tasks()):
            raise base.DemoImportError(
                f"--task-id must be in [0, {int(suite.get_num_tasks()) - 1}]"
            )
        task_ids = [int(args.task_id)]

    for task_id in task_ids:
        task = suite.get_task(task_id)
        expected_relative = suite.get_task_demonstration(task_id)
        hdf5_path = base._find_task_hdf5(input_root, expected_relative)
        if hdf5_path is None:
            continue
        source_relative = base._project_relative(
            project_root, hdf5_path, "Official LIBERO HDF5"
        )

        with h5py.File(hdf5_path, "r") as handle:
            if "data" not in handle:
                raise base.DemoImportError(
                    f"Official LIBERO HDF5 has no data group: {hdf5_path}"
                )
            data = handle["data"]
            fps = base._fps_from_hdf5(data, args.fps)
            demo_names = sorted(
                [
                    name
                    for name in data.keys()
                    if hasattr(data[name], "keys")
                    and "states" in data[name]
                    and "actions" in data[name]
                ],
                key=base._demo_sort_key,
            )

            for fallback_index, demo_name in enumerate(demo_names):
                if args.max_demos is not None and processed_this_run >= args.max_demos:
                    return _sorted_records(existing)

                group = data[demo_name]
                frame_count = base._validate_demo_group(
                    group, hdf5_path, f"data/{demo_name}"
                )
                demo_index = base._demo_index(demo_name, fallback_index)
                group_name = f"data/{demo_name}"

                camera_paths = base._materialize_demo_videos(
                    group=group,
                    video_root=video_root,
                    suite_name=args.task_suite,
                    task_id=task_id,
                    demo_index=demo_index,
                    source_relative=source_relative,
                    group_name=group_name,
                    frame_count=frame_count,
                    fps=fps,
                    resume=args.resume,
                )
                ctrl_controls_path, alignment_path, alignment_validation = (
                    base._materialize_ctrl_preparation(
                        project_root=project_root,
                        group=group,
                        camera_paths=camera_paths,
                        suite_name=args.task_suite,
                        task_id=task_id,
                        demo_index=demo_index,
                        frame_count=frame_count,
                        fps=fps,
                        alignment_min_psnr=args.alignment_min_psnr,
                        resume=args.resume,
                    )
                )

                record = _record_for_demo(
                    project_root=project_root,
                    task_suite=args.task_suite,
                    task=task,
                    task_id=task_id,
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
                existing[record["id"]] = record
                _publish_manifest(output, existing)
                processed_this_run += 1
                print(
                    "LF3R_REPAIR_LIBERO_DEMO_READY "
                    f"id={record['id']} manifest_records={len(existing)}"
                )

    return _sorted_records(existing)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=base.PROJECT_ROOT)
    parser.add_argument("--input-root", type=Path, default=base.DEFAULT_INPUT_ROOT)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int)
    parser.add_argument(
        "--max-demos",
        type=int,
        help="Maximum number of demos to prepare/recover in this invocation.",
    )
    parser.add_argument("--output", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=base.DEFAULT_VIDEO_ROOT)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--alignment-min-psnr", type=float, default=20.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_demos is not None and args.max_demos < 1:
        parser.error("--max-demos must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    records = run(args)
    print(
        "LF3R_REPAIR_LIBERO_MANIFEST_INCREMENTAL "
        f"suite={args.task_suite} records={len(records)} output={args.output}"
    )


if __name__ == "__main__":
    main()
