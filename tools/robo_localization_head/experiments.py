"""Experiment registry for the unified Robo-Dopamine localization trainer."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from typing import Sequence


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    module: str
    description: str
    default_batch_size: int = 32


EXPERIMENTS = {
    "success_negative": ExperimentPreset(
        name="success_negative",
        module="train_robo_dopamine_localization_head",
        description="Success-negative ratio ablation with h16/h32 BiLSTM heads.",
    ),
    "label_loss": ExperimentPreset(
        name="label_loss",
        module="train_robo_dopamine_label_loss_ablation",
        description="Failure-only weighted multi-event label/loss ablation.",
    ),
}


def names() -> tuple[str, ...]:
    return tuple(EXPERIMENTS)


def run_experiment(name: str, argv: Sequence[str]) -> int:
    try:
        preset = EXPERIMENTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown localization experiment: {name}") from exc
    module = importlib.import_module(preset.module)
    previous = list(sys.argv)
    try:
        sys.argv = [preset.module, *argv]
        return int(module.main())
    finally:
        sys.argv = previous
