#!/usr/bin/env python3
"""Bounded validation for LF3R baseline command planning and SAFE extraction."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve().parent / "run_lf3r_baseline.py"
sys.path.insert(0, str(RUNNER.parent))
from run_lf3r_baseline import convert_free_memory_fraction, approximate_num_steps_for_interval

BASELINES = ("safe", "procvlm", "rynnvalue", "robo_dopamine")


def latest_run(parent: Path, baseline: str) -> Path:
    matches = sorted(parent.glob(f"{baseline}_*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {baseline} run under {parent}, found {len(matches)}")
    return matches[0]


def run_checked(command: list[str]) -> None:
    print("VALIDATE", json.dumps(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-environments", action="store_true")
    parser.add_argument("--execute-safe-smoke", action="store_true")
    args = parser.parse_args()
    manifest = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
    converted, ratio = convert_free_memory_fraction(0.80, [{"gpu": 0, "total_mib": 100, "free_mib": 75}])
    assert converted == 0.6 and ratio == 0.75
    assert approximate_num_steps_for_interval(414, 8) == 53
    assert approximate_num_steps_for_interval(20, 8) == 4
    assert approximate_num_steps_for_interval(1, 8) == 1

    with tempfile.TemporaryDirectory(prefix="lf3r-baseline-validation-") as temporary:
        temp = Path(temporary)
        for baseline in BASELINES:
            output = temp / "dry" / baseline
            logs = temp / "logs" / baseline
            command = [
                "/usr/bin/python3", str(RUNNER),
                "--baseline", baseline,
                "--manifest", str(manifest),
                "--data-root", str(PROJECT_ROOT),
                "--output-dir", str(output),
                "--logs-dir", str(logs),
                "--limit", "1",
                "--dry-run",
            ]
            if baseline == "rynnvalue":
                command.extend(["--rynn-evaluation-interval", "8", "--rynn-batch-size", "4"])
            if args.check_environments:
                command.append("--validate-environment")
            run_checked(command)
            run_root = latest_run(output, baseline)
            metadata = json.loads((run_root / "run.json").read_text())
            commands = [json.loads(line) for line in (run_root / "commands.jsonl").read_text().splitlines()]
            jobs = [json.loads(line) for line in (run_root / "jobs.jsonl").read_text().splitlines()]
            assert metadata["status"] == "dry_run_complete"
            assert metadata["selected_rollouts"] == 1
            assert len(commands) == len(jobs) == 1
            assert jobs[0]["status"] == "planned"
            assert not (run_root / "raw").exists()
            argv = commands[0]["argv"]
            if baseline == "procvlm":
                assert Path(argv[1]).name == "procvlm_worker.py"
                assert argv[argv.index("--window-size") + 1] == "4"
                assert "--max-sampled-frames" not in argv
                assert "--dry-run" in argv
                assert argv[argv.index("--vllm-free-memory-fraction") + 1] == "0.8"
                assert commands[0]["execution_scope"] == "persistent_procvlm_worker"
                assert metadata["vllm_memory_scope"] == "free_gpu_memory"
                assert metadata["vllm_requested_free_fraction"] == 0.8
                assert commands[0]["vllm_memory_budget"]["resolution"] == "deferred_until_execution"
            elif baseline == "rynnvalue":
                assert Path(argv[1]).name == "rynnvalue_worker.py"
                assert argv[argv.index("--num-frames") + 1] == "16"
                assert argv[argv.index("--num-steps") + 1] == "53"
                assert argv[argv.index("--evaluation-interval") + 1] == "8"
                assert argv[argv.index("--batch-size") + 1] == "4"
                assert commands[0]["sampling"] == {
                    "total_frames": 414,
                    "target_interval": 8,
                    "approximate_num_steps": 53,
                    "endpoint_sampler": "official_uniform_prefix",
                }
            elif baseline == "robo_dopamine":
                assert Path(argv[1]).name == "robo_dopamine_persistent_worker.py"
                assert argv[argv.index("--frame-interval") + 1] == "4"
                assert argv[argv.index("--vllm-free-memory-fraction") + 1] == "0.8"
                assert "--gpu-memory-utilization" not in argv
                assert commands[0]["execution_scope"] == "persistent_robo_dopamine_worker"
                assert metadata["vllm_memory_scope"] == "free_gpu_memory"
                assert metadata["vllm_requested_free_fraction"] == 0.8
                assert commands[0]["vllm_memory_budget"]["resolution"] == "deferred_until_execution"

        if args.execute_safe_smoke:
            output = temp / "safe-smoke"
            logs = temp / "safe-smoke-logs"
            run_checked([
                "/usr/bin/python3", str(RUNNER),
                "--baseline", "safe",
                "--manifest", str(manifest),
                "--data-root", str(PROJECT_ROOT),
                "--output-dir", str(output),
                "--logs-dir", str(logs),
                "--limit", "1",
            ])
            run_root = latest_run(output, "safe")
            metadata = json.loads((run_root / "run.json").read_text())
            jobs = [json.loads(line) for line in (run_root / "jobs.jsonl").read_text().splitlines()]
            assert metadata["status"] == "complete"
            assert metadata["completed_jobs"] == 1 and metadata["failed_jobs"] == 0
            raw_csv = Path(jobs[0]["raw_output_dir"]) / "safe_features.csv"
            with raw_csv.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert rows and "max_token_entropy" in rows[0]

    print("BASELINE_PIPELINE_VALIDATION_OK")


if __name__ == "__main__":
    main()
