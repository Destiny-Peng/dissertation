#!/usr/bin/env python3
"""Prepare official LIBERO demonstrations for Repair / Synthetic Suffix.

This is an interruption-safe incremental importer. Each successfully prepared
or resumed demo is published to the manifest immediately, so Ctrl-World can use
completed demos while the remaining dataset is still being processed.

Ctrl preparation does not reuse the square RGB frames stored in the official
processed HDF5. Instead, each demonstration is replayed once from the official
simulator state/action trajectory. That replay simultaneously materializes the
three Ctrl-World camera streams at the model-native 320x192 resolution and the
frame-aligned 7D Cartesian pose/gripper controls.
"""

from __future__ import annotations

import argparse
import json
import os
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
_ensure_project_libero_on_sys_path = core._ensure_project_libero_on_sys_path

CTRL_WORLD_WIDTH = 320
CTRL_WORLD_HEIGHT = 192
CTRL_CAMERA_SOURCES = {
    "cam_high": "agentview",
    "cam_front": "frontview",
    "cam_wrist": "robot0_eye_in_hand",
}


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
        "schema_version": 2,
        "manifest": core._project_relative(project_root, output, "Repair manifest"),
        "source_root": core._project_relative(
            project_root, input_root, "Official LIBERO input root"
        ),
        "task_suite": task_suite,
        "tasks_found": len({int(r["task_id"]) for r in suite_records}),
        "demonstrations": len(suite_records),
        "ground_truth_outcome": "success",
        "alignment_rule": (
            "rgb[c] = replay observation restored from states[c+1] / reached by action[c]; "
            "continue with actions[c+1:]"
        ),
        "camera_views": list(CTRL_CAMERA_SOURCES),
        "camera_source_names": dict(CTRL_CAMERA_SOURCES),
        "ctrl_replay_resolution": {
            "width": CTRL_WORLD_WIDTH,
            "height": CTRL_WORLD_HEIGHT,
        },
        "ctrl_prepared": True,
        "ctrl_controls": "one frame-aligned absolute pose/gripper control per replayed RGB frame",
        "publication": "incremental_after_each_demo",
    }
    summary_path = output.with_name(output.stem.replace("manifest", "summary") + ".json")
    core._atomic_text(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )


def _ctrl_replay_paths(
    *,
    video_root: Path,
    suite_name: str,
    task_id: int,
    demo_index: int,
) -> tuple[dict[str, Path], Path, Path, Path]:
    stem = f"{suite_name}-task{task_id:02d}-demo{demo_index:03d}"
    directory = (
        video_root
        / suite_name
        / f"task{task_id:02d}"
        / f"ctrl_{CTRL_WORLD_WIDTH}x{CTRL_WORLD_HEIGHT}"
    )
    camera_paths = {
        slot: directory / f"{stem}.{slot}.mp4"
        for slot in CTRL_CAMERA_SOURCES
    }
    return (
        camera_paths,
        directory / f"{stem}.ctrl_controls.npz",
        directory / f"{stem}.alignment.json",
        directory / f"{stem}.ctrl_replay.json",
    )


def _ctrl_replay_spec(
    *,
    source_relative: str,
    group_name: str,
    frame_count: int,
    fps: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_hdf5_path": source_relative,
        "trajectory_group": group_name,
        "frame_count": int(frame_count),
        "fps": float(fps),
        "width": CTRL_WORLD_WIDTH,
        "height": CTRL_WORLD_HEIGHT,
        "camera_source_names": dict(CTRL_CAMERA_SOURCES),
        "rgb_transform": "rotate_180",
        "restore_state_index_for_frame_0": 1,
        "first_replayed_action_index": 1,
        "alignment_rule": (
            "replay frame 0 from states[1]; for frame k>0 step actions[k]; "
            "frame k therefore matches the official RGB/action index k"
        ),
        "control_semantics": "xyz + Euler XYZ + DROID-style gripper closure",
    }


def _resume_ctrl_replay(
    *,
    camera_paths: dict[str, Path],
    controls_path: Path,
    validation_path: Path,
    replay_path: Path,
    expected: dict[str, Any],
    frame_count: int,
) -> dict[str, Any] | None:
    required = [*camera_paths.values(), controls_path, validation_path, replay_path]
    if not all(path.is_file() for path in required):
        return None
    try:
        import numpy as np

        actual = json.loads(replay_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        with np.load(controls_path, allow_pickle=False) as payload:
            controls = np.asarray(payload["controls"])
            indices = np.asarray(payload["source_frame_indices"])
    except (ImportError, OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    if actual != expected:
        return None
    if controls.shape != (frame_count, 7):
        return None
    if indices.shape != (frame_count,):
        return None
    if not np.array_equal(indices, np.arange(frame_count, dtype=np.int64)):
        return None
    if not isinstance(validation, dict) or not validation.get("passed"):
        return None
    if int(validation.get("frame_count") or -1) != int(frame_count):
        return None
    return validation


def _materialize_ctrl_replay(
    *,
    project_root: Path,
    group: Any,
    task: Any,
    video_root: Path,
    suite_name: str,
    task_id: int,
    demo_index: int,
    source_relative: str,
    group_name: str,
    frame_count: int,
    fps: float,
    alignment_min_psnr: float,
    resume: bool,
) -> tuple[dict[str, Path], Path, Path, Path, dict[str, Any]]:
    """Replay one official demo once and publish Ctrl RGB + 7D controls.

    The replay begins at states[1], which is the state aligned with official
    RGB[0]. For every subsequent source frame k, action[k] is stepped once. The
    camera frames and Ctrl pose are sampled from the same observation, so their
    indexing cannot drift relative to one another.
    """

    try:
        import imageio.v2 as imageio
        import numpy as np
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from repair.trajectory import (
            _ctrl_world_pose_from_observation,
            run_libero_alignment_smoke,
        )
    except ImportError as error:
        raise DemoImportError(
            "LIBERO, imageio, numpy, and Repair trajectory helpers are required "
            "for native Ctrl replay preparation"
        ) from error

    camera_paths, controls_path, validation_path, replay_path = _ctrl_replay_paths(
        video_root=video_root,
        suite_name=suite_name,
        task_id=task_id,
        demo_index=demo_index,
    )
    expected = _ctrl_replay_spec(
        source_relative=source_relative,
        group_name=group_name,
        frame_count=frame_count,
        fps=fps,
    )
    if resume:
        cached = _resume_ctrl_replay(
            camera_paths=camera_paths,
            controls_path=controls_path,
            validation_path=validation_path,
            replay_path=replay_path,
            expected=expected,
            frame_count=frame_count,
        )
        if cached is not None:
            return camera_paths, controls_path, validation_path, replay_path, cached

    occupied = [
        path
        for path in [*camera_paths.values(), controls_path, validation_path, replay_path]
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(
            "Ctrl replay artifacts already exist but do not match the requested "
            "preparation; use the matching --resume invocation or remove the "
            f"ctrl_{CTRL_WORLD_WIDTH}x{CTRL_WORLD_HEIGHT} directory: "
            + ", ".join(str(path) for path in occupied)
        )

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault(
        "LIBERO_CONFIG_PATH",
        str(project_root / "cache" / "libero"),
    )

    states = np.asarray(group["states"][...])
    actions = np.asarray(group["actions"][...], dtype=np.float64)
    if len(states) < 2 or len(actions) != frame_count:
        raise DemoImportError(
            "Official LIBERO trajectory cannot seed Ctrl replay: "
            f"states={len(states)} actions={len(actions)} frames={frame_count}"
        )

    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    for path in camera_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    temporary_paths = {
        slot: path.with_name(f".{path.stem}.tmp{path.suffix}")
        for slot, path in camera_paths.items()
    }
    for path in temporary_paths.values():
        if path.exists():
            path.unlink()

    env = None
    writers: dict[str, Any] = {}
    controls: list[Any] = []
    published: list[Path] = []
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_names=list(CTRL_CAMERA_SOURCES.values()),
            camera_heights=CTRL_WORLD_HEIGHT,
            camera_widths=CTRL_WORLD_WIDTH,
        )
        env.seed(0)
        env.reset()
        obs = env.set_init_state(states[1])

        writers = {
            slot: imageio.get_writer(
                str(temporary_paths[slot]),
                fps=float(fps),
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=None,
            )
            for slot in CTRL_CAMERA_SOURCES
        }

        for frame_index in range(frame_count):
            controls.append(_ctrl_world_pose_from_observation(obs))
            for slot, camera in CTRL_CAMERA_SOURCES.items():
                key = camera + "_image"
                if key not in obs:
                    available = ", ".join(
                        sorted(name for name in obs if name.endswith("_image"))
                    )
                    raise DemoImportError(
                        f"Ctrl replay observation is missing {key}; available: {available}"
                    )
                frame = np.asarray(obs[key])
                if frame.shape != (CTRL_WORLD_HEIGHT, CTRL_WORLD_WIDTH, 3):
                    raise DemoImportError(
                        f"Unexpected Ctrl replay {key} shape at frame {frame_index}: "
                        f"{frame.shape}"
                    )
                writers[slot].append_data(
                    np.ascontiguousarray(frame[::-1, ::-1])
                )

            next_action = frame_index + 1
            if next_action < frame_count:
                obs, _, _, _ = env.step(actions[next_action].tolist())

        for writer in writers.values():
            writer.close()
        writers = {}

        control_array = np.stack(controls, axis=0).astype(np.float32)
        indices = np.arange(frame_count, dtype=np.int64)
        if control_array.shape != (frame_count, 7):
            raise DemoImportError(
                "Ctrl native replay did not produce one 7D control per RGB frame: "
                f"{control_array.shape}"
            )

        for slot, target in camera_paths.items():
            os.replace(temporary_paths[slot], target)
            published.append(target)

        temporary_controls = controls_path.with_name(
            f".{controls_path.name}.tmp.npz"
        )
        if temporary_controls.exists():
            temporary_controls.unlink()
        np.savez_compressed(
            temporary_controls,
            controls=control_array,
            source_frame_indices=indices,
            source_fps=np.asarray([float(fps)], dtype=np.float32),
        )
        os.replace(temporary_controls, controls_path)
        published.append(controls_path)

        rollout = {
            "task_suite": suite_name,
            "task_id": int(task_id),
            "total_frames": int(frame_count),
            "fps": float(fps),
            "camera_video_paths": {
                slot: core._project_relative(
                    project_root,
                    path,
                    f"Ctrl {slot} replay video",
                )
                for slot, path in camera_paths.items()
            },
        }
        model_xml_raw = group.attrs.get("model_file")
        model_xml = (
            model_xml_raw.decode("utf-8")
            if isinstance(model_xml_raw, bytes)
            else (str(model_xml_raw) if model_xml_raw is not None else None)
        )
        sample_cuts = sorted(
            {
                0,
                min(frame_count - 2, frame_count // 2),
                frame_count - 2,
            }
        )
        smoke_results = [
            run_libero_alignment_smoke(
                project_root=project_root,
                rollout=rollout,
                states=states,
                actions=actions,
                cut_frame=int(cut_frame),
                min_psnr=float(alignment_min_psnr),
                model_xml=model_xml,
            )
            for cut_frame in sample_cuts
        ]
        passed = bool(smoke_results) and all(
            bool(item.get("passed")) for item in smoke_results
        )
        validation = {
            "schema_version": 2,
            "passed": passed,
            "method": "offline_libero_native_ctrl_replay_sampled_cuts",
            "frame_count": int(frame_count),
            "fps": float(fps),
            "minimum_psnr": float(alignment_min_psnr),
            "sampled_cut_frames": sample_cuts,
            "samples": smoke_results,
            "index_contract": {
                "condition_frame": "ctrl_replay_rgb[c]",
                "branch_state": "states[c+1]",
                "future_actions": "actions[c+1:]",
                "ctrl_control": "ctrl_controls[c]",
            },
            "camera_replay": {
                "width": CTRL_WORLD_WIDTH,
                "height": CTRL_WORLD_HEIGHT,
                "camera_source_names": dict(CTRL_CAMERA_SOURCES),
                "rgb_transform": "rotate_180",
            },
            "control_semantics": "xyz + Euler XYZ + DROID-style gripper closure",
            "control_derivation": (
                "restore states[1], replay GT actions[1:], sample RGB and 7D pose "
                "from the same simulator observation at every source frame"
            ),
            "future_recorded_proprio_used": False,
        }
        if not passed:
            raise DemoImportError(
                "Offline LIBERO Ctrl replay alignment validation failed; prepared "
                "Ctrl artifacts were not published"
            )

        core._atomic_text(
            validation_path,
            json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        )
        published.append(validation_path)
        core._atomic_text(
            replay_path,
            json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
        )
        published.append(replay_path)
        return camera_paths, controls_path, validation_path, replay_path, validation
    except Exception:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        for path in temporary_paths.values():
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
        temporary_controls = controls_path.with_name(
            f".{controls_path.name}.tmp.npz"
        )
        try:
            if temporary_controls.exists():
                temporary_controls.unlink()
        except OSError:
            pass
        for path in reversed(published):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
        raise
    finally:
        if env is not None:
            env.close()


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
    ctrl_replay_path: Path,
    alignment_validation: dict[str, Any],
    frame_count: int,
    fps: float,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
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
        "camera_source_names": dict(CTRL_CAMERA_SOURCES),
        "camera_materialization": "native_simulator_replay_from_official_gt_actions",
        "camera_width": CTRL_WORLD_WIDTH,
        "camera_height": CTRL_WORLD_HEIGHT,
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
            "condition_frame": "ctrl_replay_rgb[c]",
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
        "ctrl_replay_path": core._project_relative(
            project_root, ctrl_replay_path, "prepared Ctrl replay metadata"
        ),
        "ctrl_controls_format": "lf3r_ctrl_absolute_pose_v1",
        "ctrl_control_semantics": (
            "xyz + Euler XYZ + DROID-style gripper closure; "
            "one control aligned to every replayed RGB frame"
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

                (
                    camera_paths,
                    ctrl_controls_path,
                    alignment_path,
                    ctrl_replay_path,
                    alignment_validation,
                ) = _materialize_ctrl_replay(
                    project_root=project_root,
                    group=group,
                    task=task,
                    video_root=video_root,
                    suite_name=task_suite,
                    task_id=current_task_id,
                    demo_index=demo_index,
                    source_relative=source_relative,
                    group_name=group_name,
                    frame_count=frame_count,
                    fps=fps,
                    alignment_min_psnr=alignment_min_psnr,
                    resume=resume,
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
                    ctrl_replay_path=ctrl_replay_path,
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
                    f"id={record['id']} manifest_records={len(records)} "
                    f"ctrl_replay={CTRL_WORLD_WIDTH}x{CTRL_WORLD_HEIGHT}"
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
        help="Optional replay diagnostic threshold. Default 0 so it does not block preparation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed native Ctrl replay/control/alignment artifacts when present.",
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
        f"suite={args.task_suite} records={len(records)} output={args.output} "
        f"ctrl_replay={CTRL_WORLD_WIDTH}x{CTRL_WORLD_HEIGHT}"
    )


if __name__ == "__main__":
    main()
