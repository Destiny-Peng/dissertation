"""Prepare LF3R latent rollouts for the official SAFE OpenVLA loader.

The upstream loader scans a flat prefix for matching CSV files and loads a
same-stem official SAFE pickle. This wrapper selects manifest rows, links
existing official files, and converts only LF3R safe-feature sidecars into
that official pickle record shape. Source rollout files are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

HIDDEN_SHAPE = (7, 4096)
ACTION_COLUMNS = (
    "action/dx", "action/dy", "action/dz", "action/droll",
    "action/dpitch", "action/dyaw", "action/dgripper",
)


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {value}") from exc
    return resolved


def relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError(f"Manifest line {line_number} has no rollout id")
            rows.append(row)
    return rows


def success_label(row: dict[str, Any]) -> int:
    if row.get("episode_success") is not None:
        return int(bool(row["episode_success"]))
    return int(row.get("ground_truth_outcome") == "success")


def source_candidates(root: Path, row: dict[str, Any]) -> tuple[Path, list[Path], list[Path], list[Path], Path]:
    if not row.get("csv_path"):
        raise ValueError(f"Manifest row {row['id']} has no csv_path")
    csv_path = project_path(root, row["csv_path"])
    video_path = project_path(root, row["video_path"]) if row.get("video_path") else csv_path.with_suffix(".mp4")
    pkl = [csv_path.with_suffix(".pkl")]
    npz = [csv_path.with_suffix(".safe_features.npz"), video_path.with_suffix(".safe_features.npz")]
    metadata = [csv_path.with_suffix(".safe_features.json"), video_path.with_suffix(".safe_features.json")]
    for key in ("safe_features_pkl", "official_safe_pkl", "pkl_path"):
        if row.get(key):
            pkl.insert(0, project_path(root, row[key]))
    for key in ("safe_features_npz", "safe_features_path", "features_npz"):
        if row.get(key):
            npz.insert(0, project_path(root, row[key]))
    for key in ("safe_features_json", "safe_features_metadata", "metadata_path"):
        if row.get(key):
            metadata.insert(0, project_path(root, row[key]))
    return csv_path, pkl, npz, metadata, video_path


def first_file(paths: list[Path]) -> Path | None:
    seen = set()
    for path in paths:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def hidden_shape(value: Any, source: Path) -> tuple[int, ...]:
    if isinstance(value, list):
        value = torch.stack(value, dim=0)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    shape = tuple(int(item) for item in value.shape)
    if len(shape) != 3 or shape[1:] != HIDDEN_SHAPE:
        raise ValueError(f"{source} must contain hidden_states shaped (T,7,4096), got {shape}")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"{source} contains non-finite hidden_states")
    return shape


def validate_pkl(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        record = pickle.load(handle)
    if not isinstance(record, dict) or "hidden_states" not in record:
        raise ValueError(f"Official SAFE pickle has no hidden_states: {path}")
    return hidden_shape(record["hidden_states"], path)


def convert_npz(path: Path, destination: Path, row: dict[str, Any]) -> tuple[int, ...]:
    with np.load(path, allow_pickle=False) as arrays:
        if "hidden_states" not in arrays:
            raise ValueError(f"SAFE sidecar has no hidden_states: {path}")
        hidden = np.asarray(arrays["hidden_states"])
    shape = tuple(int(item) for item in hidden.shape)
    if len(shape) != 3 or shape[1:] != HIDDEN_SHAPE:
        raise ValueError(f"{path} must contain hidden_states shaped (T,7,4096), got {shape}")
    if not np.isfinite(hidden.astype(np.float32, copy=False)).all():
        raise ValueError(f"{path} contains non-finite hidden_states")
    record = {
        "task_suite_name": row.get("task_suite", "openvla"),
        "task_id": int(row.get("task_id", 0)),
        "task_description": row.get("task_description", f"Task {row.get('task_id', 0)}"),
        "episode_idx": int(row.get("episode_index", row.get("episode_idx", 0))),
        "episode_success": success_label(row),
        "hidden_states": torch.from_numpy(hidden.astype(np.float32, copy=False)),
        "lf3r_source_safe_features_npz": str(path),
    }
    with destination.open("wb") as handle:
        pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return shape


def validate_metadata(path: Path, shape: tuple[int, ...]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid SAFE sidecar metadata JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"SAFE sidecar metadata must be an object: {path}")
    schema = payload.get("schema")
    if schema is not None and schema != "lf3r.openvla.safe_features.v1":
        raise ValueError(f"Unsupported SAFE sidecar metadata schema {schema!r}: {path}")
    representation = payload.get("representation")
    if isinstance(representation, dict) and representation.get("shape") is not None:
        declared = tuple(int(item) for item in representation["shape"])
        if declared != shape:
            raise ValueError(f"SAFE metadata shape {declared} disagrees with NPZ shape {shape}: {path}")


def materialize(source: Path, destination: Path) -> str:
    try:
        destination.symlink_to(source)
        return "symlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def csv_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows:
        missing = [column for column in ACTION_COLUMNS if column not in rows[0]]
        if missing:
            raise ValueError(f"{path} is missing action columns: {missing}")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="datasets/lf3r_failure_rollouts/v1/manifest.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-role", choices=("all", "primary_natural", "reference_natural", "controlled_analysis"), default="all")
    parser.add_argument("--partition", choices=("all", "natural_observation", "controlled_analysis"), default="all")
    parser.add_argument("--run-name")
    parser.add_argument("--rollout-id", action="append", default=[])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = project_path(root, args.manifest)
    output = project_path(root, args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    if args.start_index < 0 or (args.end_index is not None and args.end_index < args.start_index):
        raise SystemExit("Invalid selection range")
    rows = []
    ids = set(args.rollout_id)
    for row in read_manifest(manifest):
        if ids and row["id"] not in ids:
            continue
        if args.dataset_role != "all" and row.get("dataset_role") != args.dataset_role:
            continue
        if args.partition != "all" and row.get("analysis_partition") != args.partition:
            continue
        if args.run_name and row.get("run_name") != args.run_name:
            continue
        rows.append(row)
    rows = rows[args.start_index:args.end_index]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("Selection is empty")
    output.mkdir(parents=True, exist_ok=True)
    seen_stems = set()
    records = []
    for row in rows:
        csv_path, pkl_candidates, npz_candidates, metadata_candidates, video_path = source_candidates(root, row)
        if not csv_path.is_file():
            raise SystemExit(f"Missing source CSV for {row['id']}: {csv_path}")
        task_id = int(row.get("task_id", 0))
        episode_idx = int(row.get("episode_index", row.get("episode_idx", 0)))
        success = success_label(row)
        stem = f"task{task_id}--ep{episode_idx}--succ{success}"
        if stem in seen_stems:
            raise SystemExit(f"Duplicate official filename {stem}; select one --run-name or explicit IDs")
        seen_stems.add(stem)
        pkl_source = first_file(pkl_candidates)
        npz_source = first_file(npz_candidates)
        metadata_source = first_file(metadata_candidates)
        if pkl_source is None and npz_source is None:
            raise SystemExit(f"No official .pkl or .safe_features.npz for {row['id']}")
        action_steps = csv_count(csv_path)
        csv_mode = materialize(csv_path, output / f"{stem}.csv")
        if pkl_source is not None:
            shape = validate_pkl(pkl_source)
            pkl_mode = materialize(pkl_source, output / f"{stem}.pkl")
            source_kind, feature_source = "official_pkl", pkl_source
        else:
            shape = convert_npz(npz_source, output / f"{stem}.pkl", row)
            pkl_mode, source_kind, feature_source = "converted_safe_sidecar", "safe_features_npz", npz_source
        metadata_mode = None
        if metadata_source is not None:
            validate_metadata(metadata_source, shape)
            metadata_mode = materialize(metadata_source, output / f"{stem}.safe_features.json")
        if action_steps != shape[0]:
            raise SystemExit(f"Count mismatch for {row['id']}: csv={action_steps}, hidden_states={shape[0]}")
        video_mode = None
        if video_path.is_file():
            video_mode = materialize(video_path, output / f"{stem}.mp4")
        records.append({
            "rollout_id": row["id"],
            "official_filename": f"{stem}.pkl",
            "task_id": task_id,
            "episode_idx": episode_idx,
            "episode_success": success,
            "steps": shape[0],
            "source_kind": source_kind,
            "source_feature_file": relative(root, feature_source),
            "source_metadata_file": relative(root, metadata_source) if metadata_source is not None else None,
            "source_csv": relative(root, csv_path),
            "source_video": relative(root, video_path) if video_path.is_file() else None,
            "csv_materialization": csv_mode,
            "pkl_materialization": pkl_mode,
            "metadata_materialization": metadata_mode,
            "video_materialization": video_mode,
        })
    selection = {
        "schema_version": 1,
        "loader": "repos/SAFE/failure_prob/data/openvla.py",
        "token_idx_rel": 1.0,
        "source_hidden_state_shape": ["T", 7, 4096],
        "dataset_directory": relative(root, output),
        "manifest": relative(root, manifest),
        "filters": {"dataset_role": args.dataset_role, "partition": args.partition, "run_name": args.run_name, "rollout_ids": args.rollout_id, "start_index": args.start_index, "end_index": args.end_index, "limit": args.limit},
        "rollouts": records,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": relative(root, output), "rollouts": len(records), "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
