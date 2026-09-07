#!/usr/bin/env python3
"""Run the official DenseReward 3-frame model persistently over rollout jobs."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import traceback
from datetime import datetime
from typing import Any


REASON_RE = re.compile(r"<think>\s*([A-Za-z][A-Za-z _-]*)\s*</think>", re.IGNORECASE)
REWARD_RE = re.compile(r"</think>\s*([01](?:\.\d+)?)", re.IGNORECASE)


def iso_now() -> str:
    return datetime.now().astimezone().isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_output(raw: str) -> tuple[str, float, bool]:
    reason_match = REASON_RE.search(raw)
    reward_match = REWARD_RE.search(raw)
    if reason_match is None:
        raise ValueError(f"DenseReward output has no <think> reason: {raw!r}")
    if reward_match is None:
        raise ValueError(f"DenseReward output has no scalar reward: {raw!r}")
    reason = " ".join(reason_match.group(1).lower().split())
    reward = float(reward_match.group(1))
    if not math.isfinite(reward):
        raise ValueError(f"DenseReward reward is not finite: {raw!r}")
    clipped = not 0.0 <= reward <= 1.0
    reward = min(1.0, max(0.0, reward))
    return reason, reward, clipped


def job_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "completed_jobs": sum(record.get("status") == "complete" for record in records),
        "failed_jobs": sum(record.get("status") == "failed" for record in records),
        "pending_jobs": sum(record.get("status") not in {"complete", "failed"} for record in records),
    }


def refresh_state(
    path: Path,
    *,
    status: str,
    total_jobs: int,
    records: list[dict[str, Any]],
    model_load_seconds: float | None,
    last_rollout_id: str | None = None,
    error: str | None = None,
) -> None:
    counts = job_counts(records)
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "updated_at": iso_now(),
        "total_jobs": total_jobs,
        **counts,
        "model_load_seconds": model_load_seconds,
        "last_rollout_id": last_rollout_id,
    }
    if error:
        state["error"] = error
    atomic_json(path, state)


class DenseRewardEngine:
    """One official DenseReward processor/model pair reused by one worker."""

    def __init__(self, model_path: Path, max_new_tokens: int) -> None:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.max_new_tokens = int(max_new_tokens)
        self.model_path = model_path.resolve()
        prompt_path = self.model_path / "system_prompt.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"DenseReward system prompt is missing: {prompt_path}")
        self.system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        self.system_prompt_sha256 = hashlib.sha256(self.system_prompt.encode()).hexdigest()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        try:
            self.input_device = self.model.device
        except Exception:
            self.input_device = next(self.model.parameters()).device

    def infer(self, frames: list[Any], task: str) -> dict[str, Any]:
        if len(frames) != 3:
            raise ValueError(f"DenseReward requires exactly three frames, got {len(frames)}")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frames[0]},
                    {"type": "image", "image": frames[1]},
                    {"type": "image", "image": frames[2]},
                    {"type": "text", "text": task},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.input_device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        raw = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        reason, reward, clipped = parse_output(raw)
        return {
            "input_tokens": int(inputs.input_ids.shape[-1]),
            "generated_tokens": int(generated_ids.shape[-1]),
            "raw_text": raw,
            "reason": reason,
            "reward": reward,
            "reward_was_clipped": clipped,
        }


def decode_frames(video_path: Path):
    import av

    container = av.open(str(video_path))
    try:
        for frame_index, frame in enumerate(container.decode(video=0)):
            yield frame_index, frame.to_image().convert("RGB")
    finally:
        container.close()


def infer_job(
    job: dict[str, Any],
    engine: DenseRewardEngine,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rollout_id = str(job["rollout_id"])
    video_path = Path(str(job["video_path"])).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    output_dir = Path(str(job["raw_output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "densereward_raw.jsonl"
    result_path = output_dir / "worker_result.json"
    interval = int(job.get("frame_interval", args.frame_interval))
    max_new_tokens = int(job.get("max_new_tokens", args.max_new_tokens))
    if interval < 1:
        raise ValueError("DenseReward frame_interval must be positive")
    if max_new_tokens < 1:
        raise ValueError("DenseReward max_new_tokens must be positive")
    task = str(job.get("task", job.get("task_description", "")))
    if not task:
        raise ValueError(f"DenseReward rollout {rollout_id} has no task description")

    existing_rows = load_jsonl(output_path) if args.resume else []
    existing_frames = {
        int(row["frame_index"])
        for row in existing_rows
        if row.get("frame_index") is not None
    }
    if not args.resume:
        output_path.write_text("", encoding="utf-8")
    rows = list(existing_rows)
    sampled_indices: list[int] = [int(row["frame_index"]) for row in existing_rows if row.get("frame_index") is not None]
    started = time.perf_counter()
    frame_window: deque[tuple[int, Any]] = deque(maxlen=3)
    for frame_index, image in decode_frames(video_path):
        frame_window.append((frame_index, image))
        if len(frame_window) < 3 or (frame_index - 2) % interval != 0:
            continue
        if frame_index in existing_frames:
            continue
        sampled_frame_indices = [item[0] for item in frame_window]
        inference = engine.infer([item[1] for item in frame_window], task)
        row = {
            "schema_version": 1,
            "baseline": "densereward",
            "rollout_id": rollout_id,
            "frame_index": frame_index,
            "sampled_frame_indices": sampled_frame_indices,
            "frame_interval": interval,
            "task": task,
            **inference,
        }
        append_jsonl(output_path, row)
        rows.append(row)
        sampled_indices.append(frame_index)
        print(
            f"DENSEREWARD_SAMPLE rollout={rollout_id} frame={frame_index} "
            f"window={sampled_frame_indices} reward={inference['reward']}",
            flush=True,
        )

    if not rows:
        raise ValueError(f"DenseReward produced no usable samples for {rollout_id}")
    rows.sort(key=lambda row: int(row.get("frame_index", 0)))
    result = {
        "schema_version": 1,
        "baseline": "densereward",
        "rollout_id": rollout_id,
        "video_path": str(video_path),
        "task": task,
        "raw_model_output": str(output_path),
        "frame_interval": interval,
        "sampled_frame_indices": [int(row["frame_index"]) for row in rows],
        "sampled_frame_count": len(rows),
        "sampled_window_size": 3,
        "sampling_mode": "three_consecutive_rgb_frames_current_at_frame_2_plus_interval",
        "checkpoint": str(engine.model_path),
        "system_prompt_sha256": engine.system_prompt_sha256,
        "dtype": "bfloat16",
        "decoding": {"do_sample": False, "max_new_tokens": max_new_tokens},
        "inference_seconds": time.perf_counter() - started,
        "signal_names": ["reward"],
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "raw_output_dir": str(output_dir),
        "raw_output_files": [str(output_path), str(result_path)],
        "raw_model_output": str(output_path),
        "sampled_frame_count": len(rows),
        "sampled_frame_min": min(int(row["frame_index"]) for row in rows),
        "sampled_frame_max": max(int(row["frame_index"]) for row in rows),
        "inference_seconds": result["inference_seconds"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--jobs-output-file", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.frame_interval < 1:
        parser.error("--frame-interval must be positive")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    return args


def main() -> int:
    args = parse_args()
    plan = load_jsonl(args.jobs_file)
    if not plan:
        raise SystemExit(f"DenseReward job plan is empty: {args.jobs_file}")
    for required in (args.model_path, args.jobs_file):
        if not required.exists():
            raise SystemExit(f"Required DenseReward path is missing: {required}")
    jobs = load_jsonl(args.jobs_output_file)
    total_jobs = len(plan)
    model_load_seconds: float | None = None
    refresh_state(
        args.state_file,
        status="starting",
        total_jobs=total_jobs,
        records=jobs,
        model_load_seconds=None,
    )
    try:
        started = time.perf_counter()
        engine = DenseRewardEngine(args.model_path, args.max_new_tokens)
        model_load_seconds = time.perf_counter() - started
        append_jsonl(args.progress_file, {
            "event": "engine_initialized",
            "at": iso_now(),
            "initialization_seconds": model_load_seconds,
            "model_path": str(args.model_path.resolve()),
            "dtype": "bfloat16",
            "device": str(engine.input_device),
            "persistent_model": True,
        })
    except Exception as error:
        details = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        print(details, end="", flush=True)
        refresh_state(
            args.state_file,
            status="failed",
            total_jobs=total_jobs,
            records=jobs,
            model_load_seconds=None,
            error=str(error),
        )
        return 1

    refresh_state(
        args.state_file,
        status="running",
        total_jobs=total_jobs,
        records=jobs,
        model_load_seconds=model_load_seconds,
    )
    existing = {
        str(record.get("rollout_id")): record
        for record in jobs
        if record.get("rollout_id") is not None
    }
    for index, job in enumerate(plan):
        rollout_id = str(job["rollout_id"])
        if args.resume and existing.get(rollout_id, {}).get("status") == "complete":
            continue
        started_at = iso_now()
        print(f"DENSEREWARD_RUN {index + 1}/{total_jobs} rollout={rollout_id}", flush=True)
        try:
            result = infer_job(job, engine, args)
            job_result = {
                **job,
                **result,
                "status": "complete",
                "return_code": 0,
                "started_at": started_at,
                "completed_at": iso_now(),
            }
        except Exception as error:
            details = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            print(details, end="", flush=True)
            job_result = {
                **job,
                "status": "failed",
                "return_code": 1,
                "error": str(error),
                "traceback": details[-12000:],
                "started_at": started_at,
                "completed_at": iso_now(),
                "raw_output_dir": str(Path(str(job["raw_output_dir"])).resolve()),
                "raw_output_files": [],
            }
        existing[rollout_id] = job_result
        jobs = list(existing.values())
        jobs.sort(key=lambda record: int(record.get("job_index", 0)))
        write_jsonl_atomic(args.jobs_output_file, jobs)
        refresh_state(
            args.state_file,
            status="running",
            total_jobs=total_jobs,
            records=jobs,
            model_load_seconds=model_load_seconds,
            last_rollout_id=rollout_id,
        )

    counts = job_counts(jobs)
    status = "complete" if counts["failed_jobs"] == 0 and counts["pending_jobs"] == 0 else "complete_with_errors"
    refresh_state(
        args.state_file,
        status=status,
        total_jobs=total_jobs,
        records=jobs,
        model_load_seconds=model_load_seconds,
        last_rollout_id=str(jobs[-1].get("rollout_id")) if jobs else None,
    )
    append_jsonl(args.progress_file, {
        "event": "worker_complete",
        "at": iso_now(),
        "status": status,
        **counts,
        "model_load_seconds": model_load_seconds,
    })
    print(
        f"DENSEREWARD_END status={status} completed={counts['completed_jobs']} "
        f"failed={counts['failed_jobs']}",
        flush=True,
    )
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
