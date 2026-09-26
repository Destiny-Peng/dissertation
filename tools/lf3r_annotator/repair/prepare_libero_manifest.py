#!/usr/bin/env python3
"""Index official LIBERO success demonstrations for Repair / Synthetic Suffix.

The official processed LIBERO HDF5 dataset has the alignment used by LF3R:
RGB[k] is captured after action[k] and therefore corresponds to the simulator
state reached from states[k] by action[k], i.e. approximately states[k + 1].
This importer does not rewrite states/actions.  It only materializes the two
physical RGB streams as MP4 files and writes a manifest that points back to the
original HDF5 demo group.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "libero_official"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "datasets"
    / "lf3r_failure_rollouts"
    / "v1"
    / "libero_official_success_manifest.jsonl"
)
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "datasets" / "libero_official_success" / "v1"
CAMERA_DATASETS = {
    "cam_high": "obs/agentview_rgb",
    "cam_wrist": "obs/eye_in_hand_rgb",
}


class ImportError(RuntimeError):
    """Official-demo indexing error."""


def _project_relative(project_root: Path, path: Path, label: str) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError as error:
        raise ImportError(f"{label} must live under the LF3R project root: {resolved}") from error


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _json_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return (int(name.rsplit("_", 1)[-1]), name)
    except (TypeError, ValueError):
        return (10**9, name)


def _demo_index(name: str, fallback: int) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return fallback


def _fps_from_hdf5(data_group: Any, override: float | None) -> float:
    if override is not None:
        if override <= 0:
            raise ImportError("--fps must be positive")
        return float(override)

    raw_env_args = data_group.attrs.get("env_args")
    if raw_env_args is not None:
        try:
            payload = json.loads(_json_text(raw_env_args))
            env_kwargs = payload.get("env_kwargs") if isinstance(payload, dict) else None
            value = env_kwargs.get("control_freq") if isinstance(env_kwargs, dict) else None
            if value is not None and float(value) > 0:
                return float(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    # LIBERO's official dataset conversion configures control_freq=20.
    return 20.0


def _find_task_hdf5(input_root: Path, expected_relative: str) -> Path | None:
    expected = Path(expected_relative)
    direct_candidates = [
        input_root / expected,
        input_root / expected.name,
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = sorted(
        {
            path.resolve()
            for path in input_root.rglob(expected.name)
            if path.is_file()
        }
    )
    if not matches:
        return None
    if len(matches) > 1:
        raise ImportError(
            f"Multiple HDF5 files match {expected.name}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _validate_demo_group(group: Any, source: Path, group_name: str) -> int:
    for key in ("states", "actions", *CAMERA_DATASETS.values()):
        if key not in group:
            raise ImportError(f"{source}:{group_name} is missing {key}")
    count = int(len(group["actions"]))
    lengths = {
        "states": int(len(group["states"])),
        "actions": count,
        **{
            camera: int(len(group[path]))
            for camera, path in CAMERA_DATASETS.items()
        },
    }
    if count < 2 or len(set(lengths.values())) != 1:
        raise ImportError(
            f"{source}:{group_name} has incompatible trajectory lengths: {lengths}"
        )
    action_shape = tuple(int(item) for item in group["actions"].shape)
    if len(action_shape) != 2 or action_shape[1] != 7:
        raise ImportError(
            f"{source}:{group_name} expected LIBERO [T,7] actions, got {action_shape}"
        )
    return count


def _source_sidecar(
    *,
    source_hdf5: str,
    group_name: str,
    frame_count: int,
    fps: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_hdf5_path": source_hdf5,
        "trajectory_group": group_name,
        "frame_count": int(frame_count),
        "fps": float(fps),
        "camera_datasets": dict(CAMERA_DATASETS),
        "rgb_transform": "none",
        "alignment_rule": (
            "processed RGB[c] is observation after action[c]; "
            "Repair branches from states[c+1] with actions[c+1:]"
        ),
    }


def _can_resume(
    sidecar_path: Path,
    camera_paths: dict[str, Path],
    expected: dict[str, Any],
) -> bool:
    if not sidecar_path.is_file() or not all(path.is_file() for path in camera_paths.values()):
        return False
    try:
        actual = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def _write_camera_video(dataset: Any, target: Path, fps: float) -> None:
    try:
        import imageio.v2 as imageio
        import numpy as np
    except ImportError as error:
        raise ImportError(
            "imageio and numpy are required to materialize official LIBERO RGB videos"
        ) from error

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing Repair source video: {target}")

    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    if temporary.exists():
        temporary.unlink()
    writer = None
    try:
        writer = imageio.get_writer(
            str(temporary),
            fps=float(fps),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        for index in range(len(dataset)):
            frame = np.asarray(dataset[index])
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ImportError(
                    f"Unexpected RGB frame shape in {target.name} at {index}: {frame.shape}"
                )
            writer.append_data(np.ascontiguousarray(frame))
        writer.close()
        writer = None
        os.replace(temporary, target)
    except Exception:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
        raise


def _materialize_demo_videos(
    *,
    group: Any,
    video_root: Path,
    suite_name: str,
    task_id: int,
    demo_index: int,
    source_relative: str,
    group_name: str,
    frame_count: int,
    fps: float,
    resume: bool,
) -> dict[str, Path]:
    stem = f"{suite_name}-task{task_id:02d}-demo{demo_index:03d}"
    directory = video_root / suite_name / f"task{task_id:02d}"
    camera_paths = {
        camera: directory / f"{stem}.{camera}.mp4"
        for camera in CAMERA_DATASETS
    }
    sidecar_path = directory / f"{stem}.source.json"
    expected = _source_sidecar(
        source_hdf5=source_relative,
        group_name=group_name,
        frame_count=frame_count,
        fps=fps,
    )

    if resume and _can_resume(sidecar_path, camera_paths, expected):
        return camera_paths

    occupied = [path for path in [*camera_paths.values(), sidecar_path] if path.exists()]
    if occupied:
        raise FileExistsError(
            "Official LIBERO demo artifacts already exist; use --resume only when "
            "the matching source sidecar is intact: "
            + ", ".join(str(path) for path in occupied)
        )

    created: list[Path] = []
    try:
        for camera, dataset_path in CAMERA_DATASETS.items():
            target = camera_paths[camera]
            _write_camera_video(group[dataset_path], target, fps)
            created.append(target)
        _atomic_text(
            sidecar_path,
            json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
        )
        created.append(sidecar_path)
    except Exception:
        for path in created:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return camera_paths


def _load_suite(name: str) -> Any:
    try:
        from libero.libero import benchmark
    except ImportError as error:
        raise ImportError(
            "LIBERO is required to map official HDF5 files to benchmark task IDs; "
            "run this importer in LF3R-openvla"
        ) from error
    mapping = benchmark.get_benchmark_dict()
    if name not in mapping:
        raise ImportError(
            f"Unknown LIBERO suite {name!r}; available: {', '.join(sorted(mapping))}"
        )
    return mapping[name]()


def build_manifest(
    *,
    project_root: Path,
    input_root: Path,
    task_suite: str,
    output: Path,
    video_root: Path,
    fps_override: float | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    try:
        import h5py
    except ImportError as error:
        raise ImportError("h5py is required to index official LIBERO demonstrations") from error

    project_root = project_root.expanduser().resolve()
    input_root = input_root.expanduser().resolve()
    output = output.expanduser().resolve()
    video_root = video_root.expanduser().resolve()

    if not input_root.is_dir():
        raise ImportError(f"Official LIBERO input root does not exist: {input_root}")
    _project_relative(project_root, input_root, "Official LIBERO input root")
    _project_relative(project_root, output, "Repair manifest output")
    _project_relative(project_root, video_root, "Repair source-video root")

    suite = _load_suite(task_suite)
    records: list[dict[str, Any]] = []
    discovered_tasks = 0

    for task_id in range(int(suite.get_num_tasks())):
        task = suite.get_task(task_id)
        expected_relative = suite.get_task_demonstration(task_id)
        hdf5_path = _find_task_hdf5(input_root, expected_relative)
        if hdf5_path is None:
            continue
        discovered_tasks += 1
        source_relative = _project_relative(project_root, hdf5_path, "Official LIBERO HDF5")

        with h5py.File(hdf5_path, "r") as handle:
            if "data" not in handle:
                raise ImportError(f"Official LIBERO HDF5 has no data group: {hdf5_path}")
            data = handle["data"]
            fps = _fps_from_hdf5(data, fps_override)
            demo_names = sorted(
                [
                    name
                    for name in data.keys()
                    if hasattr(data[name], "keys")
                    and "states" in data[name]
                    and "actions" in data[name]
                ],
                key=_demo_sort_key,
            )
            if not demo_names:
                raise ImportError(f"No demonstrations found in {hdf5_path}")

            for fallback_index, demo_name in enumerate(demo_names):
                group = data[demo_name]
                frame_count = _validate_demo_group(group, hdf5_path, f"data/{demo_name}")
                demo_index = _demo_index(demo_name, fallback_index)
                group_name = f"data/{demo_name}"
                camera_paths = _materialize_demo_videos(
                    group=group,
                    video_root=video_root,
                    suite_name=task_suite,
                    task_id=task_id,
                    demo_index=demo_index,
                    source_relative=source_relative,
                    group_name=group_name,
                    frame_count=frame_count,
                    fps=fps,
                    resume=resume,
                )
                records.append(
                    {
                        "schema_version": 1,
                        "id": (
                            f"{task_suite}-task{task_id:02d}-"
                            f"demo{demo_index:03d}-official"
                        ),
                        "task_suite": task_suite,
                        "task_id": task_id,
                        "episode_index": demo_index,
                        "task_description": str(task.language),
                        "ground_truth_outcome": "success",
                        "source_kind": "official_demonstration",
                        "analysis_partition": "official_demonstration",
                        "dataset_role": f"{task_suite}_official_success",
                        "camera_video_paths": {
                            camera: _project_relative(
                                project_root,
                                path,
                                f"{camera} video",
                            )
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
                        },
                    }
                )

    if discovered_tasks == 0 or not records:
        raise ImportError(
            f"No official {task_suite} demonstrations were found under {input_root}"
        )

    records.sort(
        key=lambda item: (
            int(item["task_id"]),
            int(item["episode_index"]),
            str(item["id"]),
        )
    )
    manifest_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    _atomic_text(output, manifest_text)
    summary = {
        "schema_version": 1,
        "manifest": _project_relative(project_root, output, "Repair manifest"),
        "source_root": _project_relative(project_root, input_root, "Official LIBERO input root"),
        "task_suite": task_suite,
        "tasks_found": discovered_tasks,
        "demonstrations": len(records),
        "ground_truth_outcome": "success",
        "alignment_rule": (
            "rgb[c] ~= observation(states[c+1]); continue with actions[c+1:]"
        ),
        "camera_views": list(CAMERA_DATASETS),
    }
    _atomic_text(
        output.with_name(output.stem.replace("manifest", "summary") + ".json"),
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Mounted/downloaded official LIBERO dataset root.",
    )
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument(
        "--fps",
        type=float,
        help="Override source playback FPS; default reads LIBERO control_freq, then 20.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only videos whose source sidecar exactly matches the HDF5 demo.",
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
        fps_override=args.fps,
        resume=bool(args.resume),
    )
    print(
        "LF3R_REPAIR_LIBERO_MANIFEST "
        f"suite={args.task_suite} demos={len(records)} output={args.output}"
    )


if __name__ == "__main__":
    main()
