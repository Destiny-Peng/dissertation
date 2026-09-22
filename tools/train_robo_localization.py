#!/usr/bin/env python3
"""Unified CLI entry point for Robo-Dopamine BiLSTM localization training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from robo_localization_head.experiments import EXPERIMENTS, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=False,
    )
    parser.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENTS),
        required=True,
        help="Localization experiment preset.",
    )
    parser.add_argument("-h", "--help", action="store_true")
    known, remaining = parser.parse_known_args()
    if known.help:
        preset = EXPERIMENTS[known.experiment]
        module = __import__(preset.module)
        module.build_parser().print_help()
        return 0
    return run_experiment(known.experiment, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
