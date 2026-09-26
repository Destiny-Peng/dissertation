"""Trajectory loading and LIBERO alignment smoke tests for Repair."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from backend_core import ValidationError
from .alignment import action_source, find_trajectory_path, project_path


ACTION_FIELDS = (
    "action/dx",
    "action/dy",
    "action/dz",
    "action/droll",
    "action/dpitch",
    "action/dyaw",
    "action/dgripper",
)
CAMERA_TO_LIBERO = {
    "cam_high": "agentview",
    "cam_wrist": "robot0_eye_in_hand",
}


def _read_csv_actions(path: Path) -> Any:
    import numpy as np

    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in ACTION_FIELDS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValidationError(
                "Rollout CSV is missing executed-action fields: " + ", ".join(missing)
            )
        for row in reader:
            rows.append([float(row[name]) for name in ACTION_FIELDS])
    if not rows:
        raise ValidationError("Rollout action CSV is empty")
    return np.asarray(rows, dtype=np.float64)


def _h5_group(handle: Any, rollout: dict[str, Any]) -> Any:
    explicit = rollout.get("trajectory_group") or rollout.get("demo_key")
    if isinstance(explicit, str) and explicit.strip():
        key = explicit.strip().strip("/")
        if key not in handle:
            raise ValidationError(f"Trajectory group not found in HDF5: {key}")
        return handle[key]
    if "states" in handle and "actions" in handle:
        return handle
    episode = int(rollout.get("episode_index") or 0)
    candidates = (
        f"data/demo_{episode}",
        f"data/demo_{episode + 1}",
        f"demo_{episode}",
        f"demo_{episode + 1}",
    )
    for key in candidates:
        if key in handle and "states" in handle[key] and "actions" in handle[key]:
            return handle[key]
    raise ValidationError(
        "Could not locate states/actions in HDF5; set trajectory_group in the manifest"
    )


def load_actions(project_root: Path, rollout: dict[str, Any]) -> Any:
    import numpy as np

    source = action_source(project_root, rollout)
    if source is None:
        raise ValidationError("Selected rollout has no action source in the manifest")
    if source.suffix.lower() == ".csv":
        return _read_csv_actions(source)
    if source.suffix.lower() == ".npz":
        payload = np.load(source, allow_pickle=False)
        for key in ("actions", "action"):
            if key in payload:
                return np.asarray(payload[key], dtype=np.float64)
        raise ValidationError("NPZ action source has no actions array")
    if source.suffix.lower() == ".npy":
        return np.asarray(np.load(source, allow_pickle=False), dtype=np.float64)
    if source.suffix.lower() in {".hdf5", ".h5"}:
        try:
            import h5py
        except ImportError as error:
            raise ValidationError("h5py is required to read LIBERO HDF5 trajectories") from error
        with h5py.File(source, "r") as handle:
            group = _h5_group(handle, rollout)
            return np.asarray(group["actions"][...], dtype=np.float64)
    raise ValidationError(f"Unsupported action source: {source.suffix}")


def load_states(project_root: Path, rollout: dict[str, Any]) -> Any:
    import numpy as np

    source = find_trajectory_path(project_root, rollout)
    if source is None:
        raise ValidationError(
            "Selected rollout has no simulator-state trajectory path in the manifest"
        )
    if source.suffix.lower() == ".npz":
        payload = np.load(source, allow_pickle=False)
        for key in ("states", "sim_states", "state"):
            if key in payload:
                return np.asarray(payload[key])
        raise ValidationError("NPZ trajectory has no states array")
    if source.suffix.lower() == ".npy":
        return np.asarray(np.load(source, allow_pickle=False))
    if source.suffix.lower() in {".hdf5", ".h5"}:
        try:
            import h5py
        except ImportError as error:
            raise ValidationError("h5py is required to read LIBERO HDF5 trajectories") from error
        with h5py.File(source, "r") as handle:
            group = _h5_group(handle, rollout)
            return np.asarray(group["states"][...])
    raise ValidationError(f"Unsupported simulator-state source: {source.suffix}")


def _video_frame(path: Path, index: int) -> Any:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise ValidationError("imageio is required for alignment validation") from error
    reader = imageio.get_reader(str(path))
    try:
        return reader.get_data(index)
    finally:
        reader.close()


def _orient(image: Any) -> Any:
    import numpy as np

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValidationError(f"Unexpected LIBERO RGB shape: {array.shape}")
    return np.ascontiguousarray(array[::-1, ::-1])


def _psnr(a: Any, b: Any) -> float:
    import numpy as np

    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.shape != bb.shape:
        raise ValidationError(f"Alignment image shape mismatch: {aa.shape} vs {bb.shape}")
    mse = float(np.mean((aa - bb) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def run_libero_alignment_smoke(
    *,
    project_root: Path,
    rollout: dict[str, Any],
    states: Any,
    actions: Any,
    cut_frame: int,
    min_psnr: float = 20.0,
) -> dict[str, Any]:
    """Restore state[c+1], compare to RGB[c], then step actions[c+1]."""

    if str(rollout.get("task_suite") or "") not in {
        "libero_10",
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_90",
    }:
        raise ValidationError("Alignment smoke test currently supports LIBERO suites only")
    branch = cut_frame + 1
    action_index = cut_frame + 1
    if len(states) <= branch:
        raise ValidationError(
            f"Trajectory has {len(states)} states; branch state {branch} is unavailable"
        )
    if len(actions) <= action_index:
        raise ValidationError(
            f"Trajectory has {len(actions)} actions; future action {action_index} is unavailable"
        )

    try:
        import os
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as error:
        raise ValidationError(
            "LIBERO is required for the Repair alignment smoke test in the selected runtime"
        ) from error

    suite_name = str(rollout["task_suite"])
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValidationError(f"LIBERO benchmark is unavailable: {suite_name}")
    suite = benchmark_dict[suite_name]()
    task_id = int(rollout.get("task_id") or 0)
    task = suite.get_task(task_id)
    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    camera_paths = rollout.get("camera_video_paths") or {}
    physical_views = [
        view for view in ("cam_high", "cam_wrist") if view in camera_paths
    ]
    if not physical_views:
        raise ValidationError("Alignment validation requires cam_high or cam_wrist")
    libero_cameras = [CAMERA_TO_LIBERO[view] for view in physical_views]
    reference_view = physical_views[0]
    reference_path = project_path(
        project_root,
        str(camera_paths[reference_view]),
    )
    reference_frame = _video_frame(reference_path, cut_frame)
    if reference_frame.ndim != 3 or reference_frame.shape[-1] != 3:
        raise ValidationError(
            f"Unexpected reference RGB shape: {reference_frame.shape}"
        )
    height = int(reference_frame.shape[0])
    width = int(reference_frame.shape[1])
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=libero_cameras,
        camera_heights=height,
        camera_widths=width,
    )
    comparisons: dict[str, Any] = {}
    try:
        env.seed(0)
        env.reset()
        obs0 = env.set_init_state(states[branch])
        obs1, _, _, _ = env.step(actions[action_index].tolist())
        for view in physical_views:
            camera = CAMERA_TO_LIBERO[view]
            real0 = _video_frame(
                project_path(project_root, str(camera_paths[view])),
                cut_frame,
            )
            real1 = _video_frame(
                project_path(project_root, str(camera_paths[view])),
                cut_frame + 1,
            )
            render0 = _orient(obs0[camera + "_image"])
            render1 = _orient(obs1[camera + "_image"])
            p0 = _psnr(real0, render0)
            p1 = _psnr(real1, render1)
            comparisons[view] = {
                "restore_state_index": branch,
                "reference_rgb_frame": cut_frame,
                "restore_psnr": p0,
                "step_action_index": action_index,
                "next_reference_rgb_frame": cut_frame + 1,
                "step_psnr": p1,
                "passed": p0 >= min_psnr and p1 >= min_psnr,
            }
    finally:
        env.close()

    passed = bool(comparisons) and all(item["passed"] for item in comparisons.values())
    return {
        "passed": passed,
        "minimum_psnr": float(min_psnr),
        "cut_rgb_frame": int(cut_frame),
        "branch_state_index": branch,
        "future_action_start": action_index,
        "comparisons": comparisons,
    }
