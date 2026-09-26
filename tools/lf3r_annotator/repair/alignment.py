"""LIBERO trajectory/RGB alignment helpers used by Repair.

LF3R uses the official playback convention:
    state[k] --action[k]--> state[k + 1]
while processed RGB frame k is the observation after action[k], i.e. it
corresponds to state[k + 1].  A cut at RGB frame c therefore continues from
state[c + 1] with actions[c + 1:].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend_core import ValidationError


TRAJECTORY_PATH_FIELDS = (
    "trajectory_path",
    "sim_state_path",
    "states_path",
    "state_action_path",
    "hdf5_path",
    "source_hdf5_path",
    "demo_path",
)


@dataclass(frozen=True)
class AlignmentPlan:
    cut_type: str
    cut_progress: float | None
    cut_rgb_frame: int
    condition_frame: int
    branch_state_index: int
    gt_action_start: int
    gt_action_end: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cut_type": self.cut_type,
            "cut_progress": self.cut_progress,
            "cut_rgb_frame": self.cut_rgb_frame,
            "condition_frame": self.condition_frame,
            "branch_state_index": self.branch_state_index,
            "gt_action_start": self.gt_action_start,
            "gt_action_end": self.gt_action_end,
            "alignment_rule": "rgb[c] ~= observation(state[c+1]); continue with actions[c+1:]",
        }


def compute_alignment(
    *,
    total_frames: int,
    cut_type: str = "progress",
    cut_progress: float = 0.5,
    cut_frame: int | None = None,
    action_count: int | None = None,
) -> AlignmentPlan:
    if total_frames < 2:
        raise ValidationError("Repair requires at least two RGB frames")
    cut_type = str(cut_type or "progress").strip().lower()
    if cut_type not in {"progress", "frame"}:
        raise ValidationError("cut_type must be progress or frame")

    progress: float | None
    if cut_type == "progress":
        try:
            progress = float(cut_progress)
        except (TypeError, ValueError) as error:
            raise ValidationError("cut_progress must be a number") from error
        if not 0.0 < progress < 1.0:
            raise ValidationError("cut_progress must be strictly between 0 and 1")
        # Frame indices are zero-based.  Use the observable trajectory span
        # instead of total_frames so c+1 always exists.
        c = int((total_frames - 1) * progress)
    else:
        progress = None
        if isinstance(cut_frame, bool):
            raise ValidationError("cut_frame must be an integer")
        try:
            c = int(cut_frame)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ValidationError("cut_frame must be an integer") from error

    if c < 0 or c >= total_frames - 1:
        raise ValidationError(
            f"cut RGB frame must be in [0, {total_frames - 2}] so a continuation exists"
        )

    action_end = int(action_count) if action_count is not None else None
    if action_end is not None and c + 1 >= action_end:
        raise ValidationError(
            "cut leaves no future action after LIBERO alignment; "
            f"cut={c}, actions={action_end}"
        )
    return AlignmentPlan(
        cut_type=cut_type,
        cut_progress=progress,
        cut_rgb_frame=c,
        condition_frame=c,
        branch_state_index=c + 1,
        gt_action_start=c + 1,
        gt_action_end=action_end,
    )


def project_path(project_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValidationError("Repair input path escapes project root") from error
    return path


def find_trajectory_path(project_root: Path, rollout: dict[str, Any]) -> Path | None:
    for field in TRAJECTORY_PATH_FIELDS:
        value = rollout.get(field)
        if isinstance(value, str) and value.strip():
            path = project_path(project_root, value.strip())
            if path.is_file():
                return path
    return None


def action_source(project_root: Path, rollout: dict[str, Any]) -> Path | None:
    for field in (
        "actions_path",
        "trajectory_path",
        "state_action_path",
        "hdf5_path",
        "source_hdf5_path",
        "demo_path",
        "csv_path",
    ):
        value = rollout.get(field)
        if isinstance(value, str) and value.strip():
            path = project_path(project_root, value.strip())
            if path.is_file():
                return path
    return None


def capability_summary(project_root: Path, rollout: dict[str, Any]) -> dict[str, Any]:
    cameras = rollout.get("camera_video_paths")
    views = sorted(cameras) if isinstance(cameras, dict) else []
    actions = action_source(project_root, rollout)
    trajectory = find_trajectory_path(project_root, rollout)
    return {
        "views": views,
        "actions_available": actions is not None,
        "actions_source": (
            str(actions.relative_to(project_root)) if actions is not None else None
        ),
        "sim_state_available": trajectory is not None,
        "trajectory_source": (
            str(trajectory.relative_to(project_root)) if trajectory is not None else None
        ),
    }
