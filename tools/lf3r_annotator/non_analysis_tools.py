#!/usr/bin/env python3
"""Small LF3R-owned helpers for WebUI non-analysis tools.

This module intentionally avoids the Analysis pipeline. It exposes bounded,
project-local command builders and lightweight GPU inspection that the
annotator server can use without duplicating tool semantics in JavaScript.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    resolved.relative_to(PROJECT_ROOT)
    return resolved


def gpu_status() -> dict[str, Any]:
    """Return one-shot NVIDIA GPU status; never start or mutate GPU workloads."""
    binary = shutil.which("nvidia-smi")
    if not binary:
        return {"available": False, "error": "nvidia-smi is unavailable", "gpus": []}
    query = (
        "index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,"
        "utilization.memory,temperature.gpu,power.draw,power.limit"
    )
    try:
        completed = subprocess.run(
            [binary, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc), "gpus": []}
    rows = []
    reader = csv.reader(completed.stdout.splitlines())
    for raw in reader:
        if len(raw) != 11:
            continue
        values = [item.strip() for item in raw]
        def number(text: str) -> float | None:
            try:
                return float(text)
            except (TypeError, ValueError):
                return None
        total = number(values[3])
        used = number(values[4])
        free = number(values[5])
        rows.append({
            "index": int(values[0]),
            "name": values[1],
            "uuid": values[2],
            "memory_total_mib": total,
            "memory_used_mib": used,
            "memory_free_mib": free,
            "memory_free_fraction": (free / total) if total and free is not None else None,
            "gpu_utilization_percent": number(values[6]),
            "memory_utilization_percent": number(values[7]),
            "temperature_c": number(values[8]),
            "power_draw_w": number(values[9]),
            "power_limit_w": number(values[10]),
        })
    return {"available": True, "error": None, "gpus": rows}


def export_command(payload: dict[str, Any]) -> list[str]:
    script = PROJECT_ROOT / "tools/export_failure_cases.py"
    command = ["/usr/bin/python3", str(script)]
    outcomes = payload.get("outcomes") or ["failure"]
    for outcome in outcomes:
        if outcome not in {"success", "failure", "recovered_success", "uncertain"}:
            raise ValueError(f"invalid outcome: {outcome}")
        command.extend(["--outcome", outcome])
    role = str(payload.get("dataset_role") or "primary_natural")
    if role not in {"primary_natural", "reference_natural", "controlled_analysis", "all"}:
        raise ValueError("invalid dataset_role")
    command.extend(["--dataset-role", role])
    review = str(payload.get("review_status") or "complete")
    if review not in {"complete", "in_progress", "unreviewed", "all"}:
        raise ValueError("invalid review_status")
    command.extend(["--review-status", review])
    if payload.get("output_dir"):
        command.extend(["--output-dir", str(project_path(str(payload["output_dir"])))])
    if bool(payload.get("dry_run", False)):
        command.append("--dry-run")
    return command


def safe_prepare_command(payload: dict[str, Any]) -> list[str]:
    python = Path(os.environ.get("LF3R_SAFE_PYTHON", PROJECT_ROOT / "conda_envs/LF3R-safe/bin/python"))
    script = PROJECT_ROOT / "tools/safe_training/prepare_dataset.py"
    output = project_path(str(payload.get("output") or "outputs/safe_training/datasets/web_prepare"))
    command = [
        str(python), str(script), "--output", str(output),
        "--dataset-role", str(payload.get("dataset_role") or "primary_natural"),
        "--partition", str(payload.get("partition") or "natural_observation"),
    ]
    run_name = str(payload.get("run_name") or "").strip()
    if run_name:
        command.extend(["--run-name", run_name])
    for rollout_id in payload.get("rollout_ids") or []:
        command.extend(["--rollout-id", str(rollout_id)])
    return command


def safe_train_command(payload: dict[str, Any]) -> list[str]:
    python = Path(os.environ.get("LF3R_SAFE_PYTHON", PROJECT_ROOT / "conda_envs/LF3R-safe/bin/python"))
    script = PROJECT_ROOT / "tools/safe_training/run_safe_training.py"
    model = str(payload.get("model") or "mlp")
    if model not in {"mlp", "lstm"}:
        raise ValueError("SAFE model must be mlp or lstm")
    dataset = project_path(str(payload.get("dataset_dir") or "outputs/safe_training/datasets/web_prepare"))
    logs = project_path(str(payload.get("logs_root") or "outputs/safe_training/logs/web"))
    command = [
        str(python), str(script),
        "--dataset-dir", str(dataset),
        "--model", model,
        "--gpu", str(payload.get("gpu") or "0"),
        "--logs-root", str(logs),
        "--epochs", str(int(payload.get("epochs") or 1000)),
        "--batch-size", str(int(payload.get("batch_size") or 512)),
        "--hidden-dim", str(int(payload.get("hidden_dim") or 256)),
        "--seed", str(payload.get("seed") or "0"),
        "--token-idx-rel", str(payload.get("token_idx_rel") or "1.0"),
    ]
    if bool(payload.get("normalize", False)):
        command.append("--normalize")
    return command


def safe_conformal_command(payload: dict[str, Any]) -> list[str]:
    python = Path(os.environ.get("LF3R_SAFE_PYTHON", PROJECT_ROOT / "conda_envs/LF3R-safe/bin/python"))
    script = PROJECT_ROOT / "tools/safe_training/run_safe_functional_conformal_eval.py"
    command = [str(python), str(script)]
    mapping = {
        "dataset_dir": "--dataset",
        "mlp_checkpoint": "--mlp",
        "lstm_checkpoint": "--lstm",
        "output_dir": "--output",
    }
    for key, flag in mapping.items():
        value = payload.get(key)
        if value:
            command.extend([flag, str(project_path(str(value)))])
    if payload.get("gpu") not in (None, ""):
        command.extend(["--gpu", str(payload["gpu"])])
    return command


def robo_interval_sweep_command(payload: dict[str, Any]) -> list[str]:
    script = PROJECT_ROOT / "tools/baselines/run_robo_dopamine_interval_sanity.py"
    command = [
        "/usr/bin/python3", str(script),
        "--manifest", str(PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"),
        "--data-root", str(PROJECT_ROOT),
        "--output-dir", str(project_path(str(payload.get("output_dir") or "outputs/baselines/robo_interval_sweeps"))),
        "--gpu", str(payload.get("gpu") or "0"),
        "--vllm-free-memory-fraction", str(float(payload.get("memory_utilization") or 0.6)),
        "--batch-size", str(int(payload.get("batch_size") or 1)),
    ]
    rollout_ids = payload.get("rollout_ids") or []
    if not rollout_ids:
        raise ValueError("at least one rollout_id is required")
    for rollout_id in rollout_ids:
        command.extend(["--rollout-id", str(rollout_id)])
    for interval in payload.get("intervals") or [2, 5, 10]:
        command.extend(["--frame-interval", str(int(interval))])
    if payload.get("model_path"):
        command.extend(["--model-path", str(project_path(str(payload["model_path"])))])
    if payload.get("goal_image"):
        command.extend(["--goal-image", str(project_path(str(payload["goal_image"])))])
    return command


def validate_instruction_variants_command() -> list[str]:
    return ["/usr/bin/python3", str(PROJECT_ROOT / "tools/prepare_libero10_instruction_variants.py"), "--check-only"]


def rebuild_manifest_command() -> list[str]:
    return ["/usr/bin/python3", str(PROJECT_ROOT / "tools/lf3r_annotator/build_manifest.py")]


def baseline_pipeline_validation_command(check_environments: bool = True) -> list[str]:
    command = ["/usr/bin/python3", str(PROJECT_ROOT / "tools/baselines/validate_pipeline.py")]
    if check_environments:
        command.append("--check-environments")
    return command
