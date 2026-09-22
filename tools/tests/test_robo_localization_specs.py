#!/usr/bin/env python3
"""Tests for the Localization Lab experiment-spec engine."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from robo_localization_head import specs


class LocalizationSpecTests(unittest.TestCase):
    def test_cartesian_sweep_estimate(self) -> None:
        spec = {
            "name": "matrix",
            "base": specs.DEFAULT_BASE,
            "repeats": 5,
            "sweep": [
                {"path": "model.hidden", "values": [16, 32]},
                {"path": "target.kind", "values": ["hard", "gaussian"]},
                {"path": "training.batch_size", "values": [16, 32, 64]},
            ],
            "stages": [],
        }
        normalized = specs.normalize_spec(spec)
        configs = specs.expand(normalized["base"], normalized["sweep"])
        self.assertEqual(len(configs), 12)
        self.assertEqual(specs.estimate_runs(spec)["training_runs"], 60)

    def test_empty_variants_do_not_create_base_variant(self) -> None:
        configs = specs.expand(
            specs.DEFAULT_BASE,
            [{"path": "loss.name", "values": ["bce", "temporal_softmax_ce"]}],
            [],
        )
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0]["loss"]["name"], "bce")
        self.assertEqual(configs[1]["loss"]["name"], "temporal_softmax_ce")

    def test_parallel_workers_default_and_validation(self) -> None:
        normalized = specs.normalize_spec({
            "name": "workers",
            "base": {},
            "repeats": 1,
            "sweep": [],
            "stages": [],
        })
        self.assertEqual(normalized["base"]["training"]["parallel_workers"], 4)

        invalid = specs.deep_merge(
            specs.DEFAULT_BASE,
            {"training": {"parallel_workers": 9}},
        )
        with self.assertRaises(ValueError):
            specs.validate_config(invalid)

    def test_stage_spec_is_valid(self) -> None:
        spec = specs.BUILTIN_PRESETS["label_loss_default"]
        normalized = specs.normalize_spec(spec)
        self.assertEqual(len(normalized["stages"]), 2)
        self.assertEqual(
            specs.estimate_runs(spec),
            {"configurations": 11, "training_runs": 55},
        )

    def test_success_negative_rejects_temporal_softmax(self) -> None:
        config = specs.deep_merge(
            specs.DEFAULT_BASE,
            {
                "data": {"population": "failure_success", "success_ratio": 1.0},
                "loss": {"name": "temporal_softmax_ce"},
            },
        )
        with self.assertRaises(ValueError):
            specs.validate_config(config)

    def test_coupled_target_variants_move_together(self) -> None:
        variants = [
            {
                "name": "hard",
                "set": {
                    "target.kind": "hard",
                    "target.sigma_pre": 3.0,
                    "target.sigma_post": 3.0,
                    "target.tau_event": 20.0,
                },
            },
            {
                "name": "gaussian_asymmetric",
                "set": {
                    "target.kind": "gaussian",
                    "target.sigma_pre": 3.0,
                    "target.sigma_post": 1.0,
                    "target.tau_event": 20.0,
                },
            },
        ]
        configs = specs.expand(specs.DEFAULT_BASE, [], variants)
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0]["target"]["kind"], "hard")
        self.assertEqual(configs[1]["target"]["kind"], "gaussian")
        self.assertEqual(configs[1]["target"]["sigma_post"], 1.0)

    def test_independent_sweep_crosses_coupled_variants(self) -> None:
        configs = specs.expand(
            specs.DEFAULT_BASE,
            [{"path": "model.hidden", "values": [16, 32]}],
            [
                {"name": "hard", "set": {"target.kind": "hard"}},
                {
                    "name": "gaussian_sigma_3",
                    "set": {
                        "target.kind": "gaussian",
                        "target.sigma_pre": 3.0,
                        "target.sigma_post": 3.0,
                    },
                },
            ],
        )
        self.assertEqual(len(configs), 4)


if __name__ == "__main__":
    unittest.main()
