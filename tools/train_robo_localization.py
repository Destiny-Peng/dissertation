#!/usr/bin/env python3
"""Run a Localization Lab experiment spec on saved Robo-Dopamine fused signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from robo_incremental_hop.io import PROJECT_ROOT
from robo_localization_head.spec_runner import run_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Experiment spec JSON file.")
    parser.add_argument(
        "--run-pool-root",
        default=str(PROJECT_ROOT / "outputs" / "baselines"),
        help="Historical baseline output pool; latest usable fused result is selected per rollout.",
    )
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "datasets" / "lf3r_failure_rollouts" / "v1" / "manifest.jsonl"),
    )
    parser.add_argument(
        "--annotations",
        default=str(PROJECT_ROOT / "annotations" / "failure_annotations" / "v1" / "records"),
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(args.spec.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment spec must be a JSON object")
    run_spec(
        spec=payload,
        run_pool_root=args.run_pool_root,
        manifest_path=args.manifest,
        annotation_dir=args.annotations,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
