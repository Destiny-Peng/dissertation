"""Create a tiny official SAFE/OpenVLA-shaped mock dataset."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import torch

ACTION_COLUMNS = (
    "action/dx", "action/dy", "action/dz", "action/droll",
    "action/dpitch", "action/dyaw", "action/dgripper",
)


def create_mock_dataset(output: Path, task_count: int = 4, episodes_per_task: int = 4, steps: int = 5) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to reuse non-empty mock dataset: {output}")
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(20260907)
    rows = []
    for task_id in range(task_count):
        for episode_idx in range(episodes_per_task):
            success = int(episode_idx < episodes_per_task // 2)
            stem = f"task{task_id}--ep{episode_idx}--succ{success}"
            hidden = torch.randn(steps, 7, 4096, generator=generator) * 0.02
            hidden += 0.2 if success else -0.2
            hidden[:, -1, :] += 0.4 if success else -0.4
            record = {
                "task_suite_name": "openvla_mock",
                "task_id": task_id,
                "task_description": f"Mock task {task_id}",
                "episode_idx": episode_idx,
                "episode_success": success,
                "hidden_states": hidden.float(),
            }
            with (output / f"{stem}.pkl").open("wb") as handle:
                pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["action/timestep", *ACTION_COLUMNS])
                writer.writeheader()
                for step in range(steps):
                    writer.writerow({"action/timestep": step, **{column: step / 10.0 for column in ACTION_COLUMNS}})
            rows.append({"task_id": task_id, "episode_idx": episode_idx, "episode_success": success, "steps": steps})
    metadata = {
        "schema_version": 1,
        "kind": "synthetic_official_safe_openvla",
        "hidden_states_shape": [steps, 7, 4096],
        "rollouts": rows,
    }
    (output / "mock_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-count", type=int, default=4)
    parser.add_argument("--episodes-per-task", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.task_count < 2 or args.episodes_per_task < 2 or args.steps < 2:
        raise SystemExit("task-count, episodes-per-task, and steps must all be at least 2")
    print(json.dumps(create_mock_dataset(Path(args.output), args.task_count, args.episodes_per_task, args.steps), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
