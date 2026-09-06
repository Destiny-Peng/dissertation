#!/usr/bin/env python3
"""Run a small 2/5/10 frame-interval Robo-Dopamine sweep.

The checkpoint is initialized once in this process. Each interval gets an
independent run root so its native raw outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from robo_dopamine_multi_perspective import PERSPECTIVE_MODES
from robo_dopamine_persistent_worker import infer_rollout, initialize_robo_model
from run_lf3r_baseline import (
    BASELINES,
    file_sha256,
    git_revision,
    iso_now,
    load_jsonl,
    resolve_record_path,
    resolve_vllm_memory_budget,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--rollout-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--vllm-free-memory-fraction", type=float, default=0.6)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--goal-image", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frame-interval", action="append", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    manifest = args.manifest.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_path = (args.model_path or BASELINES["robo_dopamine"]["checkpoint"]).expanduser().resolve()
    goal_image = (args.goal_image or BASELINES["robo_dopamine"]["repo"] / "examples/blank_goal.png").expanduser().resolve()
    intervals = args.frame_interval or [2, 5, 10]
    if any(interval < 1 for interval in intervals):
        raise SystemExit("--frame-interval values must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    records_by_id = {str(row["id"]): row for row in load_jsonl(manifest)}
    missing = [rollout_id for rollout_id in args.rollout_id if rollout_id not in records_by_id]
    if missing:
        raise SystemExit(f"Rollout IDs are missing from manifest: {missing}")

    # The runner's existing conversion keeps the user-facing value relative
    # to currently free memory. This script deliberately does not inspect GPU
    # utilization or add another admission rule.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    total_fraction, memory_budget = resolve_vllm_memory_budget(
        str(args.gpu), float(args.vllm_free_memory_fraction)
    )
    worker_args = SimpleNamespace(
        repo=BASELINES["robo_dopamine"]["repo"],
        model_path=model_path,
        goal_image=goal_image,
        frame_interval=intervals[0],
        batch_size=args.batch_size,
        eval_mode="forward",
        eval_modes=list(PERSPECTIVE_MODES),
        render_video=False,
        vllm_total_memory_fraction=total_fraction,
    )
    run_root = output_dir / f"interval_sweep_{timestamp()}"
    run_root.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "status": "initializing",
        "created_at": iso_now(),
        "baseline": "robo_dopamine",
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "rollout_ids": list(args.rollout_id),
        "intervals": intervals,
        "batch_size": args.batch_size,
        "gpu": str(args.gpu),
        "vllm_free_memory_fraction": args.vllm_free_memory_fraction,
        "vllm_total_memory_fraction": total_fraction,
        "memory_budget": memory_budget,
        "checkpoint": str(model_path),
        "goal_image": str(goal_image),
        "eval_modes": list(PERSPECTIVE_MODES),
        "fusion_rule": "arithmetic_mean_of_official_progress",
        "source_commit": git_revision(BASELINES["robo_dopamine"]["repo"]),
    }
    (run_root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    init_started = time.perf_counter()
    model = initialize_robo_model(worker_args)
    metadata["engine_initialization_seconds"] = time.perf_counter() - init_started
    metadata["status"] = "running"
    (run_root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    results = []
    for interval in intervals:
        interval_root = run_root / f"interval_{interval}"
        for rollout_id in args.rollout_id:
            record = records_by_id[rollout_id]
            video = resolve_record_path(str(record["video_path"]), data_root)
            task = record.get("task", record.get("task_description"))
            if task is None:
                raise SystemExit(f"Manifest record has no task description: {rollout_id}")
            job = {
                "rollout_id": rollout_id,
                "video_path": str(video),
                "task": str(task),
                "raw_output_dir": str(interval_root / "raw" / rollout_id),
                "goal_image": str(goal_image),
                "frame_interval": interval,
                "batch_size": args.batch_size,
                "eval_mode": "forward",
                "eval_modes": list(PERSPECTIVE_MODES),
                "render_video": False,
            }
            print(f"SWEEP interval={interval} rollout={rollout_id}", flush=True)
            result = infer_rollout(job, worker_args, model)
            results.append({"interval": interval, "rollout_id": rollout_id, **result})
    metadata["status"] = "complete"
    metadata["results"] = results
    metadata["completed_at"] = iso_now()
    (run_root / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"SWEEP_ROOT {run_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

