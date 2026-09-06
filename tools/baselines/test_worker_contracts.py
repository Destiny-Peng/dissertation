#!/usr/bin/env python3
"""Test raw-output adapters without loading baseline checkpoints."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> None:
    print("CONTRACT", json.dumps(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="lf3r-worker-contracts-") as temporary:
        temp = Path(temporary)
        rynn_repo = temp / "fake-rynn"
        rynn_infer = rynn_repo / "rynn_infer"
        rynn_infer.mkdir(parents=True)
        (rynn_infer / "inference.py").write_text("""
import argparse
import numpy as np
from pathlib import Path

def build_output_path(args):
    return str(Path(args.output_path) / "fake_official_run")

def parse_analysis(text):
    return {"description": text, "match": "Yes", "success": "No"}

def save_video_with_trend(**kwargs):
    raise AssertionError("rendering should be disabled in the contract test")

def main():
    parser = argparse.ArgumentParser()
    for name in ("model_path", "video_path", "instruction", "output_path", "num_frames",
                 "num_steps", "batch_size", "max_image_side", "max_new_tokens",
                 "robot_description", "camera_description"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    assert int(args.num_frames) == 16
    assert int(args.num_steps) == 3
    assert int(args.batch_size) == 4
    Path(build_output_path(args)).mkdir(parents=True, exist_ok=True)
    parse_analysis("unmodified generated analysis")
    save_video_with_trend(images=[], value=[1.25, 0.75],
                          output_path="unused.mp4", sampled_indices=[0, 9])
""")
        rynn_model = temp / "fake-rynn-model"
        rynn_model.mkdir()
        video = temp / "fake.mp4"
        video.touch()
        rynn_output = temp / "rynn-output"
        run([
            str(PROJECT_ROOT / "repos/RynnValue/.venv/bin/python"),
            str(PROJECT_ROOT / "tools/baselines/rynnvalue_worker.py"),
            "--repo", str(rynn_repo),
            "--model-path", str(rynn_model),
            "--video-path", str(video),
            "--instruction", "fake task",
            "--output-dir", str(rynn_output),
            "--num-frames", "16",
            "--num-steps", "3",
            "--evaluation-interval", "8",
            "--batch-size", "4",
            "--robot-description", "fake robot",
            "--camera-description", "fake camera",
        ])
        rynn_raw = json.loads(
            (rynn_output / "fake_official_run/raw_model_outputs.json").read_text()
        )
        assert rynn_raw["values"] == [1.25, 0.75]
        assert rynn_raw["sampled_indices"] == [0, 9]
        assert rynn_raw["analysis_text"] == "unmodified generated analysis"
        assert rynn_raw["num_frames"] == 16
        assert rynn_raw["num_frames_mode"] == "uniform_subsample"
        assert rynn_raw["sampling_mode"] == "approximate_fixed_interval_prefix_uniform"
        assert rynn_raw["evaluation_interval"] == 8
        assert rynn_raw["num_steps"] == 3
        assert rynn_raw["requested_batch_size"] == 4
        assert rynn_raw["effective_batch_size"] == 4

        robo_repo = temp / "fake-robo"
        examples = robo_repo / "examples"
        examples.mkdir(parents=True)
        (examples / "__init__.py").touch()
        (examples / "inference.py").write_text("""
import json
from pathlib import Path

class LLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class GRMInference:
    def __init__(self, model_path):
        self.model_path = model_path
    def run_pipeline(self, *, out_root, **kwargs):
        result = Path(out_root) / "fake_official_run"
        result.mkdir(parents=True, exist_ok=True)
        raw = [{"pred": "<score>+25%</score>"}]
        (result / "pred_vllm.json").write_text(json.dumps(raw))
        return str(result)
""")
        robo_model = temp / "fake-robo-model"
        robo_model.mkdir()
        goal = temp / "goal.png"
        goal.touch()
        robo_output = temp / "robo-output"
        robo_jobs = temp / "robo-jobs.jsonl"
        robo_jobs.write_text(json.dumps({
            "job_index": 0,
            "rollout_id": "fake-rollout",
            "video_path": str(video),
            "task": "fake task",
            "raw_output_dir": str(robo_output),
            "goal_image": str(goal),
        }) + chr(10))
        run([
            str(PROJECT_ROOT / "conda_envs/LF3R-robo-dopamine/bin/python"),
            str(PROJECT_ROOT / "tools/baselines/robo_dopamine_persistent_worker.py"),
            "--repo", str(robo_repo),
            "--model-path", str(robo_model),
            "--jobs-file", str(robo_jobs),
            "--jobs-output-file", str(robo_output / "jobs.jsonl"),
            "--progress-file", str(robo_output / "progress.jsonl"),
            "--state-file", str(robo_output / "state.json"),
            "--goal-image", str(goal),
            "--memory-budget-json", json.dumps({
                "scope": "free_gpu_memory",
                "requested_free_fraction": 0.8,
                "resolved_total_fraction": 0.55,
            }),
            "--vllm-total-memory-fraction", "0.55",
        ])
        robo_result = json.loads(
            (robo_output / "worker_result.json").read_text()
        )
        raw_prediction = json.loads(Path(robo_result["raw_model_output"]).read_text())
        assert raw_prediction == [{"pred": "<score>+25%</score>"}]

    print("BASELINE_WORKER_CONTRACTS_OK")


if __name__ == "__main__":
    main()
