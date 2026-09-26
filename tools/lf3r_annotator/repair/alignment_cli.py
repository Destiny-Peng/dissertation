#!/usr/bin/env python3
"""Run one LIBERO Repair alignment smoke test in the OpenVLA/LIBERO environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend_core import ValidationError
from repair.trajectory import load_actions, load_states, run_libero_alignment_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--rollout-json", type=Path, required=True)
    parser.add_argument("--cut-frame", type=int, required=True)
    parser.add_argument("--min-psnr", type=float, default=20.0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    rollout = json.loads(args.rollout_json.read_text(encoding="utf-8"))
    if not isinstance(rollout, dict):
        raise ValidationError("rollout-json must contain one manifest record")
    actions = load_actions(project_root, rollout)
    states = load_states(project_root, rollout)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValidationError(f"LIBERO actions must be [T,7], got {actions.shape}")
    result = run_libero_alignment_smoke(
        project_root=project_root,
        rollout=rollout,
        states=states,
        actions=actions,
        cut_frame=args.cut_frame,
        min_psnr=args.min_psnr,
    )
    result["action_count"] = int(len(actions))
    result["state_count"] = int(len(states))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
