#!/usr/bin/env python3
"""Run the official OpenVLA LIBERO evaluator without any intervention."""

from __future__ import annotations

import argparse
import importlib.machinery
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

from safe_feature_logging import postprocess_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT_ROOT / "checkpoints/openvla-7b-finetuned-libero-10"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/openvla_libero"
MAX_TRIALS_PER_TASK = 50  # LIBERO supplies 50 indexed initial states per task.

SUITE_CONFIGS = {
    "libero_10": {
        "label": "LIBERO-10",
        "checkpoint": CHECKPOINT,
        "output_root": OUTPUT_ROOT,
        "run_prefix": "lf3r-data-natural-libero10-",
        "max_task": 9,
        "render_resolution": 256,
        "record_resolution": 224,
    },
    "libero_spatial": {
        "label": "LIBERO-Spatial",
        "checkpoint": PROJECT_ROOT / "checkpoints/openvla-7b-finetuned-libero-spatial",
        "output_root": PROJECT_ROOT / "outputs/openvla_libero_spatial_native",
        "run_prefix": "lf3r-data-natural-libero-spatial-256-",
        "max_task": 9,
        "render_resolution": 256,
        "record_resolution": 256,
    },
}


def install_wandb_stub() -> None:
    module = types.ModuleType("wandb")
    module.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    module.init = lambda *args, **kwargs: None
    module.log = lambda *args, **kwargs: None
    module.finish = lambda *args, **kwargs: None
    module.Image = type("Image", (), {"__init__": lambda self, *args, **kwargs: None})
    sys.modules["wandb"] = module


def import_openvla_evaluator():
    """Import OpenVLA while redirecting robosuite's hard-coded debug log."""
    target = os.environ.get("ROBOSUITE_LOG_PATH")
    if not target:
        import experiments.robot.libero.run_libero_eval as evaluator

        return evaluator

    target_path = Path(target).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    original_file_handler = logging.FileHandler

    class RobosuiteRedirectFileHandler(original_file_handler):
        def __init__(self, filename, *args, **kwargs):
            try:
                candidate = os.fspath(filename)
            except TypeError:
                candidate = filename
            if candidate == "/tmp/robosuite.log":
                filename = target_path
            super().__init__(filename, *args, **kwargs)

    logging.FileHandler = RobosuiteRedirectFileHandler
    try:
        import experiments.robot.libero.run_libero_eval as evaluator
    finally:
        logging.FileHandler = original_file_handler
    return evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate natural OpenVLA rollouts for a supported LIBERO suite")
    parser.add_argument(
        "--task-suite",
        choices=tuple(SUITE_CONFIGS),
        default="libero_10",
        help="LIBERO task suite; defaults use suite-specific render/replay resolutions and 224x224 policy preprocessing",
    )
    parser.add_argument("--task-start", type=int, required=True)
    parser.add_argument("--task-end", type=int, required=True)
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Episodes per selected task (1-50; LF3R defaults to 1, official evaluator defaults to 50)",
    )
    parser.add_argument("--run-note", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--render-resolution",
        type=int,
        default=None,
        help="Square simulator render resolution; defaults to the suite configuration",
    )
    parser.add_argument(
        "--record-resolution",
        type=int,
        default=None,
        help="Square replay-video resolution; defaults to the suite configuration",
    )
    parser.add_argument(
        "--video-view-mode",
        choices=("single_view", "libero_three_view"),
        default="single_view",
        help=(
            "Recording mode. libero_three_view keeps the canonical single-view rollout "
            "and adds separate post-run Robo-Dopamine camera-slot videos."
        ),
    )
    parser.add_argument(
        "--log-safe-features",
        action="store_true",
        help=(
            "Enable official SAFE OpenVLA hidden-state logging and write "
            "per-step compressed numeric sidecars. The default keeps rollout "
            "generation unchanged."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = SUITE_CONFIGS[args.task_suite]
    render_resolution = (
        int(args.render_resolution)
        if args.render_resolution is not None
        else int(suite["render_resolution"])
    )
    record_resolution = (
        int(args.record_resolution)
        if args.record_resolution is not None
        else int(suite["record_resolution"])
    )
    for name, value in (
        ("render-resolution", render_resolution),
        ("record-resolution", record_resolution),
    ):
        if value < 64 or value > 2048 or value % 2:
            raise SystemExit(f"{name} must be an even integer between 64 and 2048")
    checkpoint = Path(suite["checkpoint"])
    output_root = Path(suite["output_root"])
    max_task = int(suite["max_task"])
    if not 0 <= args.task_start <= args.task_end <= max_task:
        raise SystemExit(f"task range must satisfy 0 <= start <= end <= {max_task}")
    if not 1 <= args.trials <= MAX_TRIALS_PER_TASK:
        raise SystemExit(
            f"trials must be between 1 and {MAX_TRIALS_PER_TASK}; "
            "LIBERO indexes one of 50 predefined initial states for each episode"
        )
    if not args.run_note.startswith(str(suite["run_prefix"])):
        raise SystemExit(
            f"run-note must preserve natural {suite['label']} provenance "
            f"with prefix {suite['run_prefix']}"
        )
    if not checkpoint.is_dir():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    output_dir = output_root / args.run_note
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {output_dir}")

    install_wandb_stub()
    torch.serialization.add_safe_globals(
        [
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.dtypes.Float64DType,
        ]
    )
    evaluator = import_openvla_evaluator()

    print("LF3R_PROVENANCE source_kind=natural_policy intervention=none")
    print(f"LF3R_TASK_SUITE suite={args.task_suite} label={suite['label']}")
    print(f"LF3R_TASK_RANGE start={args.task_start} end={args.task_end} trials={args.trials}")
    print(
        f"LF3R_RESOLUTION render={render_resolution} policy=224 record={record_resolution}"
    )
    print(f"LF3R_SAFE_FEATURES enabled={args.log_safe_features}")
    print(f"LF3R_VIDEO_VIEW_MODE mode={args.video_view_mode}")
    print(f"LF3R_ROBOSUITE_LOG path={os.environ.get('ROBOSUITE_LOG_PATH', '/tmp/robosuite.log')}")
    sys.argv = [
        "run_libero_eval.py",
        f"--pretrained_checkpoint={checkpoint}",
        f"--task_suite_name={args.task_suite}",
        f"--num_trials_per_task={args.trials}",
        f"--task_start_index={args.task_start}",
        f"--task_end_index={args.task_end}",
        f"--run_id_note={args.run_note}",
        f"--save_root={output_root}",
        f"--render_resolution={render_resolution}",
        f"--record_resolution={record_resolution}",
        "--use_wandb=False",
        "--save_logs=True",
        f"--output_hidden_states={args.log_safe_features}",
        "--attn_implementation=eager",
        f"--seed={args.seed}",
    ]
    evaluator.eval_libero()

    if args.log_safe_features:
        run_output_dir = output_root / args.run_note / args.task_suite
        sidecars = postprocess_run(
            run_output_dir=run_output_dir,
            checkpoint=checkpoint,
            project_root=PROJECT_ROOT,
            task_suite_name=args.task_suite,
            record_resolution=record_resolution,
        )
        print(f"LF3R_SAFE_FEATURES_POSTPROCESSED episodes={len(sidecars)}")
        for npz_path, metadata_path in sidecars:
            print(f"LF3R_SAFE_FEATURES_NUMERIC path={npz_path}")
            print(f"LF3R_SAFE_FEATURES_METADATA path={metadata_path}")

    if args.video_view_mode == "libero_three_view":
        run_output_dir = output_root / args.run_note / args.task_suite
        multiview_script = PROJECT_ROOT / "tools" / "lf3r_annotator" / "generate_libero_multiview.py"
        subprocess.run(
            [
                sys.executable,
                str(multiview_script),
                "--run-dir",
                str(run_output_dir),
                "--task-suite",
                args.task_suite,
                "--record-resolution",
                str(record_resolution),
                "--fps",
                "30",
            ],
            cwd=str(PROJECT_ROOT),
            check=True,
        )


if __name__ == "__main__":
    main()
