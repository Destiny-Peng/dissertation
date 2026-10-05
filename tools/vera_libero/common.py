"""Shared evaluation contracts; no VERA or simulator imports."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ('agentview_rgb', 'eye_in_hand_rgb')
DIMENSIONS = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_delta(pos, rotvec, gripper):
    """Official VERA target: [translation of T1 inv(T0), rotvec, left finger delta]."""
    r = Rotation.from_rotvec(rotvec)
    dr = r[1:] * r[:-1].inv()
    v = pos[1:] - dr.apply(pos[:-1])
    return np.concatenate((v, dr.as_rotvec(), np.diff(gripper[:, :1], axis=0)), axis=1)


def metric_to_osc(metric, current_pos, current_gripper, normalized_grip, controller):
    """Convert state-space SE3 prediction to OSC goal deltas; no fitted gains.

    Finger delta does not uniquely identify the binary command during contact or
    holds. Use sign/deadband and current finger width, without GT commands.
    """
    dr = Rotation.from_rotvec(metric[:, 3:6])
    dx = dr.apply(current_pos) + metric[:, :3] - current_pos
    raw = np.zeros_like(metric)
    low = np.asarray(controller['output_min'], dtype=np.float64)
    high = np.asarray(controller['output_max'], dtype=np.float64)
    assert np.allclose(low, -high)
    raw[:, :3] = dx / high[:3]
    raw[:, 3:6] = metric[:, 3:6] / high[3:6]
    # VERA's left-finger displacement grows on opening; LIBERO +1 closes.
    raw[:, 6] = np.where(current_gripper[:, 0] <= 0.02, 1., -1.)
    raw[normalized_grip > 0.18, 6] = -1.
    raw[normalized_grip < -0.18, 6] = 1.
    return raw, np.clip(raw, -1., 1.)


def metrics(pred, gt):
    error = np.asarray(pred) - np.asarray(gt)
    return {'count': int(len(error)), 'mae': float(np.abs(error).mean()),
            'mse': float(np.square(error).mean()),
            'per_dimension_mae': np.abs(error).mean(axis=0).tolist(),
            'per_dimension_mse': np.square(error).mean(axis=0).tolist()}


def read_plan(output):
    return json.loads((Path(output) / 'plan.json').read_text())
