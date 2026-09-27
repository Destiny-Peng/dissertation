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


def load_model_xml(project_root: Path, rollout: dict[str, Any]) -> str | None:
    source = find_trajectory_path(project_root, rollout)
    if source is None or source.suffix.lower() not in {".hdf5", ".h5"}:
        return None
    try:
        import h5py
    except ImportError as error:
        raise ValidationError("h5py is required to read LIBERO HDF5 trajectories") from error
    with h5py.File(source, "r") as handle:
        group = _h5_group(handle, rollout)
        value = group.attrs.get("model_file")
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)


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



def _ctrl_world_pose_from_observation(obs: dict[str, Any]) -> Any:
    """Convert one LIBERO observation to Ctrl-World's DROID-style 7D pose."""

    import numpy as np
    try:
        from scipy.spatial.transform import Rotation
        import robosuite.utils.transform_utils as transform_utils
    except ImportError as error:
        raise ValidationError(
            "scipy and robosuite are required for Ctrl-World pose conversion"
        ) from error

    try:
        position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        quaternion = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    except KeyError as error:
        raise ValidationError(
            f"LIBERO observation is missing Ctrl-World proprio field: {error}"
        ) from error

    if position.shape != (3,) or quaternion.shape != (4,) or gripper_qpos.size < 2:
        raise ValidationError(
            "Unexpected LIBERO proprio shapes for Ctrl-World: "
            f"eef_pos={position.shape}, eef_quat={quaternion.shape}, "
            f"gripper={gripper_qpos.shape}"
        )
    rotvec = np.asarray(
        transform_utils.quat2axisangle(quaternion),
        dtype=np.float64,
    )
    euler_xyz = Rotation.from_rotvec(rotvec).as_euler("xyz")
    opening_width = float(
        np.clip(gripper_qpos[0] - gripper_qpos[1], 0.0, 0.08)
    )
    gripper_closed = 1.0 - opening_width / 0.08
    pose = np.concatenate(
        [position, euler_xyz, np.asarray([gripper_closed])],
        axis=0,
    )
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValidationError(f"Invalid Ctrl-World pose from LIBERO obs: {pose}")
    return pose.astype(np.float32)


def replay_ctrl_world_pose_controls(
    *,
    project_root: Path,
    rollout: dict[str, Any],
    states: Any,
    actions: Any,
    cut_frame: int,
    frame_step: int,
    model_xml: str | None = None,
) -> dict[str, Any]:
    """Derive Ctrl controls only by replaying GT LIBERO actions from the cut.

    The first control point corresponds to RGB[c] / states[c+1]. Subsequent
    points are simulator observations reached by actions[c+1:], sampled at the
    requested source-frame stride. No future recorded proprio is read.
    """

    import numpy as np

    if frame_step < 1:
        raise ValidationError("Ctrl-World source frame step must be at least 1")
    branch = int(cut_frame) + 1
    action_start = int(cut_frame) + 1
    total_frames = int(rollout.get("total_frames") or len(actions))
    last_frame = min(total_frames - 1, len(actions) - 1)
    if branch >= len(states) or action_start >= len(actions):
        raise ValidationError(
            "Ctrl-World replay cut leaves no valid branch state/future action"
        )

    try:
        import os
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as error:
        raise ValidationError(
            "LIBERO is required to derive Ctrl-World controls from GT actions"
        ) from error

    suite_name = str(rollout.get("task_suite") or "")
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
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=["agentview"],
        camera_heights=64,
        camera_widths=64,
    )
    controls: list[Any] = []
    source_indices: list[int] = []
    try:
        env.seed(0)
        env.reset()
        if model_xml:
            env.reset_from_xml_string(model_xml)
            env.sim.reset()
        obs = env.set_init_state(states[branch])
        controls.append(_ctrl_world_pose_from_observation(obs))
        source_indices.append(int(cut_frame))

        for action_index in range(action_start, last_frame + 1):
            obs, _, _, _ = env.step(actions[action_index].tolist())
            source_frame = int(action_index)
            if (source_frame - int(cut_frame)) % int(frame_step) != 0:
                continue
            controls.append(_ctrl_world_pose_from_observation(obs))
            source_indices.append(source_frame)
    finally:
        env.close()

    if len(controls) < 2:
        raise ValidationError(
            "Ctrl-World GT-action replay produced no future sampled control point"
        )
    return {
        "controls": np.stack(controls, axis=0).astype(np.float32),
        "source_frame_indices": np.asarray(source_indices, dtype=np.int64),
        "branch_state_index": branch,
        "action_start": action_start,
        "derivation": (
            "restore states[c+1], replay GT actions[c+1:], convert resulting "
            "LIBERO observations to DROID-style Cartesian pose/gripper"
        ),
        "future_recorded_proprio_used": False,
    }



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


def _orient(image: Any, transform: str) -> Any:
    import numpy as np

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValidationError(f"Unexpected LIBERO RGB shape: {array.shape}")
    if transform == "raw":
        oriented = array
    elif transform == "horizontal_flip":
        oriented = array[:, ::-1]
    elif transform == "vertical_flip":
        oriented = array[::-1, :]
    elif transform == "rotate_180":
        oriented = array[::-1, ::-1]
    else:
        raise ValidationError(f"Unknown RGB orientation transform: {transform}")
    return np.ascontiguousarray(oriented)


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


def _best_orientation(reference: Any, rendered: Any) -> tuple[str, float]:
    candidates = (
        "raw",
        "horizontal_flip",
        "vertical_flip",
        "rotate_180",
    )
    scored = [
        (transform, _psnr(reference, _orient(rendered, transform)))
        for transform in candidates
    ]
    return max(scored, key=lambda item: item[1])


def run_libero_alignment_smoke(
    *,
    project_root: Path,
    rollout: dict[str, Any],
    states: Any,
    actions: Any,
    cut_frame: int,
    min_psnr: float = 20.0,
    model_xml: str | None = None,
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
        if model_xml:
            env.reset_from_xml_string(model_xml)
            env.sim.reset()
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
            transform, p0 = _best_orientation(
                real0,
                obs0[camera + "_image"],
            )
            render1 = _orient(obs1[camera + "_image"], transform)
            p1 = _psnr(real1, render1)
            comparisons[view] = {
                "restore_state_index": branch,
                "reference_rgb_frame": cut_frame,
                "orientation_transform": transform,
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
        "model_xml_used": bool(model_xml),
        "comparisons": comparisons,
    }
