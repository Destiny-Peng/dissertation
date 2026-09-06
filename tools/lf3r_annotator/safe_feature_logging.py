"""LF3R-owned postprocessor for official SAFE/OpenVLA rollout artifacts."""

from __future__ import annotations

import csv
import json
import os
import pickle
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - OpenVLA environment provides torch.
    torch = None


SAFE_ACTION_TOKEN_COUNT = 7
SAFE_HIDDEN_SIZE = 4096
SAFE_HIDDEN_STATE_SHAPE = (SAFE_ACTION_TOKEN_COUNT, SAFE_HIDDEN_SIZE)
SAFE_FEATURE_SCHEMA = "lf3r.openvla.safe_features.v1"
SAFE_FEATURE_SELECTION = 'generated_outputs["hidden_states"][token][-1][0, -1, :]'
ACTION_COLUMNS = (
    "action/dx",
    "action/dy",
    "action/dz",
    "action/droll",
    "action/dpitch",
    "action/dyaw",
    "action/dgripper",
)
EPISODE_NAME_RE = re.compile(
    r"^task(?P<task_id>[0-9]+)--ep(?P<episode_idx>[0-9]+)--succ(?P<success>[01])\.pkl$"
)


def extract_safe_hidden_states(generated_outputs: Mapping[str, Any]) -> Any:
    """Extract the official SAFE representation for one policy call."""

    if torch is None:
        raise RuntimeError("torch is required to extract model hidden states")
    all_hidden_states = generated_outputs["hidden_states"]
    if all_hidden_states is None or len(all_hidden_states) == 0:
        raise ValueError("model output contains no generated-token hidden states")
    selected = []
    for token_index, token_states in enumerate(all_hidden_states):
        if token_states is None or len(token_states) == 0:
            raise ValueError(f"hidden_states token {token_index} contains no layers")
        selected.append(token_states[-1][0, -1, :])
    return torch.stack(selected, dim=0).detach().cpu()


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"root": str(repo_root), "commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if commit.returncode != 0:
            return result
        result["commit"] = commit.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
        result["dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except OSError:
        pass
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent)
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _hidden_states_to_numpy(value: Any) -> tuple[np.ndarray, str]:
    if isinstance(value, list):
        if torch is None:
            value = np.asarray(value)
        else:
            value = torch.stack(value, dim=0)
    if hasattr(value, "dtype"):
        dtype = str(value.dtype).removeprefix("torch.")
    else:
        dtype = str(np.asarray(value).dtype)
    if torch is not None and isinstance(value, torch.Tensor):
        array = value.detach().cpu().float().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    return np.asarray(array, dtype=np.float32), dtype


def _probe_video_frame_count(path: Path) -> int | None:
    """Read a replay frame count when ffprobe is available."""

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    if not value.isdigit():
        return None
    return int(value)


def _load_official_pkl(path: Path) -> tuple[np.ndarray, str, dict[str, Any]]:
    with path.open("rb") as handle:
        record = pickle.load(handle)
    if not isinstance(record, dict) or "hidden_states" not in record:
        raise ValueError(f"SAFE artifact is missing hidden_states: {path}")
    hidden_states, source_dtype = _hidden_states_to_numpy(record["hidden_states"])
    if hidden_states.ndim != 3 or tuple(hidden_states.shape[1:]) != SAFE_HIDDEN_STATE_SHAPE:
        raise ValueError(
            "Official SAFE/OpenVLA hidden states must have shape (T, 7, 4096), "
            f"got {tuple(hidden_states.shape)} in {path}"
        )
    if not np.isfinite(hidden_states).all():
        raise ValueError(f"Official SAFE hidden states contain NaN/Inf: {path}")
    return hidden_states, source_dtype, record


def _load_logged_actions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = ("action/timestep",) + ACTION_COLUMNS
    missing = [column for column in required if not rows or column not in rows[0]]
    if missing:
        raise ValueError(f"OpenVLA CSV is missing required action columns {missing}: {path}")
    timesteps = []
    actions = []
    for row_number, row in enumerate(rows):
        try:
            timesteps.append(int(float(row["action/timestep"])))
            actions.append([float(row[column]) for column in ACTION_COLUMNS])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid action row {row_number} in {path}") from error
    timestep_array = np.asarray(timesteps, dtype=np.int64)
    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.size and not np.isfinite(action_array).all():
        raise ValueError(f"OpenVLA actions contain NaN/Inf: {path}")
    if timestep_array.size and np.any(np.diff(timestep_array) <= 0):
        raise ValueError(f"OpenVLA policy timesteps are not strictly increasing: {path}")
    return timestep_array, action_array


def _episode_ids(pkl_path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    match = EPISODE_NAME_RE.match(pkl_path.name)
    if match is None:
        raise ValueError(f"Unexpected OpenVLA episode filename: {pkl_path.name}")
    task_id = int(record.get("task_id", match.group("task_id")))
    episode_idx = int(record.get("episode_idx", record.get("eposide_idx", match.group("episode_idx"))))
    filename_success = int(match.group("success"))
    success = int(bool(record.get("episode_success", filename_success)))
    if success != filename_success:
        raise ValueError(f"Success label mismatch between pkl metadata and filename: {pkl_path}")
    return {
        "task_suite_name": str(record.get("task_suite_name", "")),
        "task_id": task_id,
        "task_description": str(record.get("task_description", "")),
        "episode_idx": episode_idx,
        "episode_success": success,
    }


def postprocess_episode(
    *,
    pkl_path: Path,
    checkpoint: Path,
    project_root: Path,
    task_suite_name: str,
    policy_input_resolution: int = 224,
    render_resolution: int = 256,
    record_resolution: int = 224,
    center_crop: bool = True,
    n_samples: int = 1,
) -> tuple[Path, Path]:
    """Merge one official SAFE pkl and OpenVLA CSV without touching either input."""

    pkl_path = pkl_path.resolve()
    csv_path = pkl_path.with_suffix(".csv")
    mp4_path = pkl_path.with_suffix(".mp4")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing OpenVLA action CSV for {pkl_path}")
    if not mp4_path.is_file():
        raise FileNotFoundError(f"Missing OpenVLA replay video for {pkl_path}")

    hidden_states, source_dtype, record = _load_official_pkl(pkl_path)
    environment_timesteps, actions = _load_logged_actions(csv_path)
    if hidden_states.shape[0] != environment_timesteps.shape[0]:
        raise ValueError(
            "SAFE hidden-state count and policy-action count differ for "
            f"{pkl_path}: {hidden_states.shape[0]} vs {environment_timesteps.shape[0]}"
        )
    video_frame_count = _probe_video_frame_count(mp4_path)
    if video_frame_count is not None and video_frame_count != hidden_states.shape[0]:
        raise ValueError(
            "SAFE hidden-state count and replay-video frame count differ for "
            f"{pkl_path}: {hidden_states.shape[0]} vs {video_frame_count}"
        )

    ids = _episode_ids(pkl_path, record)
    if ids["task_suite_name"] and ids["task_suite_name"] != task_suite_name:
        raise ValueError(
            f"Task-suite mismatch for {pkl_path}: {ids['task_suite_name']} vs {task_suite_name}"
        )
    ids["task_suite_name"] = task_suite_name
    step_count = int(hidden_states.shape[0])
    npz_path = mp4_path.with_suffix(".safe_features.npz")
    metadata_path = mp4_path.with_suffix(".safe_features.json")
    arrays = {
        "hidden_states": hidden_states,
        "policy_step_index": np.arange(step_count, dtype=np.int64),
        "environment_timestep": environment_timesteps,
        "frame_index": np.arange(step_count, dtype=np.int64),
        "actions": actions,
        "task_id": np.asarray(ids["task_id"], dtype=np.int64),
        "episode_idx": np.asarray(ids["episode_idx"], dtype=np.int64),
        "episode_success": np.asarray(ids["episode_success"], dtype=np.int8),
    }
    _atomic_npz(npz_path, arrays)

    safe_openvla_root = project_root / "repos" / "safe-openvla"
    metadata = {
        "schema": SAFE_FEATURE_SCHEMA,
        "artifact": {
            "numeric_file": npz_path.name,
            "official_safe_file": pkl_path.name,
            "action_log_file": csv_path.name,
            "replay_video_file": mp4_path.name,
            "metadata_file": metadata_path.name,
        },
        "representation": {
            "name": "SAFE OpenVLA generated action-token hidden states",
            "selection": SAFE_FEATURE_SELECTION,
            "layer": "last transformer layer",
            "sequence_position": "last sequence position",
            "action_token_count": SAFE_ACTION_TOKEN_COUNT,
            "hidden_size": SAFE_HIDDEN_SIZE,
            "shape": list(hidden_states.shape),
            "official_torch_dtype": source_dtype,
            "numeric_dtype": str(hidden_states.dtype),
            "safe_default_input_shape": [step_count, SAFE_HIDDEN_SIZE],
            "safe_default_token_idx_rel": 1.0,
        },
        "alignment": {
            "one_row_per_policy_call": True,
            "policy_step_index": "zero-based row order in the official action CSV",
            "environment_timestep": "action/timestep from the official evaluator CSV",
            "frame_index": "zero-based replay-video frame order; official evaluator appends one frame per policy call",
            "action": "post-normalization and post-gripper-inversion vector logged immediately before env.step",
            "count_validation": {
                "hidden_state_steps": step_count,
                "action_log_steps": int(environment_timesteps.shape[0]),
                "replay_video_frames": video_frame_count,
            },
        },
        "identifiers": {
            **ids,
            "mp4_path": str(mp4_path),
        },
        "policy": {
            "checkpoint": str(checkpoint),
            "model_family": "openvla",
            "n_samples": int(n_samples),
            "center_crop": bool(center_crop),
            "policy_input_resolution": int(policy_input_resolution),
            "render_resolution": int(render_resolution),
            "record_resolution": int(record_resolution),
            "image_preprocessing": "official get_libero_image resize followed by OpenVLA preprocess_image RGB conversion; policy input remains 224x224",
        },
        "provenance": {
            "safe_openvla": _git_provenance(safe_openvla_root),
            "lf3r_project": _git_provenance(project_root),
        },
        "compatibility": {
            "safe_loader": "repos/SAFE/failure_prob/data/openvla.py",
            "loader_behavior": "loads same-basename .pkl, casts hidden_states to float, then applies dataset.token_idx_rel",
            "dense_tensor_json": False,
            "official_pkl_unchanged": True,
        },
    }
    _atomic_json(metadata_path, metadata)
    return npz_path, metadata_path


def postprocess_run(
    *,
    run_output_dir: Path,
    checkpoint: Path,
    project_root: Path,
    task_suite_name: str,
    record_resolution: int,
) -> list[tuple[Path, Path]]:
    pkl_paths = sorted(run_output_dir.glob("*.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(f"No official SAFE .pkl artifacts found under {run_output_dir}")
    return [
        postprocess_episode(
            pkl_path=pkl_path,
            checkpoint=checkpoint,
            project_root=project_root,
            task_suite_name=task_suite_name,
            record_resolution=record_resolution,
        )
        for pkl_path in pkl_paths
    ]
