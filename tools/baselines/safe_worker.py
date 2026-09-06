#!/usr/bin/env python3
"""Extract SAFE's official handcrafted OpenVLA signals from one rollout CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from failure_prob.data.openvla import compute_hand_crafted_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--rollout-id", required=True)
    args = parser.parse_args()

    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    source = pd.read_csv(args.input_csv)
    features = compute_hand_crafted_metrics(source)
    if features.empty:
        raise ValueError(f"SAFE produced no features from {args.input_csv}")
    if len(features) != len(source):
        raise ValueError("SAFE feature rows do not match source rows")
    if "action/timestep" in source:
        features.insert(0, "action_timestep", source["action/timestep"].to_numpy())

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_csv, index=False)
    metadata = {
        "schema_version": 1,
        "baseline": "safe_handcrafted_openvla",
        "rollout_id": args.rollout_id,
        "source_csv": str(args.input_csv.resolve()),
        "raw_output_csv": str(args.output_csv.resolve()),
        "rows": len(features),
        "columns": list(features.columns),
        "scope_note": "Official SAFE handcrafted OpenVLA features; no trained SAFE detector checkpoint is available locally.",
    }
    args.output_csv.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(features)} SAFE feature rows to {args.output_csv}")


if __name__ == "__main__":
    main()
