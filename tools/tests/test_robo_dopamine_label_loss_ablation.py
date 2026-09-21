#!/usr/bin/env python3
"""Synthetic tests for the weighted multi-event BiLSTM label/loss ablation."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

SCRIPT = TOOLS_DIR / "train_robo_dopamine_label_loss_ablation.py"
SPEC = importlib.util.spec_from_file_location("robo_label_loss_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class RoboLabelLossAblationTests(unittest.TestCase):
    def signals(self):
        return {
            "multi": {
                "frames": [0, 1, 2, 3, 4, 5, 6],
                "progress": [0.0, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0],
                "hops": [0.0, 0.1, -0.2, 0.0, -0.3, 0.0, 0.0],
                "task_key": "libero_10:task0",
                "task_id": 0,
            },
            "pseudo": {
                "frames": [0, 2, 4, 6],
                "progress": [0.0, 0.0, 0.0, 0.0],
                "hops": [0.0, 0.0, 0.0, 0.0],
                "task_key": "libero_10:task1",
                "task_id": 1,
            },
        }

    def events(self):
        return [
            {
                "rollout_id": "multi",
                "event_id": "multi::event0",
                "event_index": 0,
                "outcome": "terminal_failure",
                "failure_type": "grasp_failure",
                "task_key": "libero_10:task0",
                "task_id": 0,
                "causal_onset_frame": 1,
                "observable_onset_frame": 2,
            },
            {
                "rollout_id": "multi",
                "event_id": "multi::event1",
                "event_index": 1,
                "outcome": "terminal_failure",
                "failure_type": "grasp_failure",
                "task_key": "libero_10:task0",
                "task_id": 0,
                "causal_onset_frame": 5,
                "observable_onset_frame": 5,
            },
        ]

    def test_multi_event_weights_decay_from_first_event(self) -> None:
        dataset = probe.build_failure_targets(
            self.signals(),
            self.events(),
            [],
            tau_event=2.0,
        )
        row = dataset["multi"]
        self.assertEqual(row["event_count"], 2)
        self.assertEqual(row["intervals"], [(1, 2), (5, 5)])
        self.assertAlmostEqual(row["events"][0]["event_weight"], 1.0)
        self.assertAlmostEqual(
            row["events"][1]["event_weight"],
            math.exp(-(5 - 1) / 2.0),
        )

    def test_hard_labels_keep_later_events_with_decayed_weight(self) -> None:
        dataset = probe.build_failure_targets(
            self.signals(),
            self.events(),
            [],
            tau_event=2.0,
        )
        row = dataset["multi"]
        labels = probe.make_labels(row, kind="hard")
        expected_later = math.exp(-2.0)
        np.testing.assert_allclose(
            labels,
            [0.0, 1.0, 1.0, 0.0, 0.0, expected_later, 0.0],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_gaussian_labels_use_max_not_sum(self) -> None:
        dataset = probe.build_failure_targets(
            self.signals(),
            self.events(),
            [],
            tau_event=2.0,
        )
        row = dataset["multi"]
        labels = probe.make_labels(
            row,
            kind="gaussian",
            sigma_pre=1.0,
            sigma_post=1.0,
        )
        self.assertLessEqual(float(labels.max()), 1.0)
        self.assertAlmostEqual(float(labels[1]), 1.0)
        self.assertAlmostEqual(float(labels[2]), 1.0)
        self.assertGreater(float(labels[3]), 0.0)
        self.assertGreater(float(labels[5]), 0.0)

    def test_eventless_terminal_failure_becomes_frame0_pseudo_event(self) -> None:
        no_event = [
            {
                "rollout_id": "pseudo",
                "outcome": "terminal_failure",
                "failure_type": "timeout_no_progress",
                "task_key": "libero_10:task1",
                "task_id": 1,
            }
        ]
        dataset = probe.build_failure_targets(
            self.signals(),
            [],
            no_event,
            tau_event=20.0,
        )
        row = dataset["pseudo"]
        self.assertTrue(row["has_pseudo_event"])
        self.assertEqual(row["intervals"], [(0, 0)])
        self.assertEqual(
            row["events"][0]["target_source"],
            "pseudo_no_event_frame0",
        )
        np.testing.assert_allclose(
            probe.make_labels(row, kind="hard"),
            [1.0, 0.0, 0.0, 0.0],
        )

    def test_nearest_interval_error_uses_closest_event(self) -> None:
        intervals = [(2, 3), (8, 9)]
        self.assertEqual(probe.nearest_interval_error(2, intervals), (0, 0))
        self.assertEqual(probe.nearest_interval_error(8, intervals), (0, 1))
        self.assertEqual(probe.nearest_interval_error(6, intervals), (-2, 1))
        self.assertEqual(probe.nearest_interval_error(5, intervals), (2, 0))

    def test_prediction_reports_any_and_first_event_hits_separately(self) -> None:
        dataset = probe.build_failure_targets(
            self.signals(),
            self.events(),
            [],
            tau_event=2.0,
        )
        row = probe.prediction_row(dataset, "multi", 5, 0.5)
        self.assertTrue(row["in_interval"])
        self.assertFalse(row["first_event_in_interval"])
        self.assertEqual(row["nearest_event_rank"], 1)

    def test_all_requested_losses_are_finite(self) -> None:
        dataset = probe.build_failure_targets(
            self.signals(),
            self.events(),
            [],
            tau_event=2.0,
        )
        config = {
            "name": "gaussian_sigma_2",
            "kind": "gaussian",
            "sigma_pre": 2.0,
            "sigma_post": 2.0,
        }
        row = probe.dataset_with_labels(dataset, config)["multi"]
        logits = torch.tensor(
            [[0.0, 0.2, 0.3, -0.1, 0.1, 0.15, -0.2]],
            dtype=torch.float32,
        )
        labels = torch.from_numpy(row["labels"]).unsqueeze(0)
        pos_weight = torch.tensor(2.0)
        for loss_name in probe.LOSS_NAMES:
            value = probe.loss_value(
                logits,
                row,
                labels,
                loss_name=loss_name,
                pos_weight=pos_weight,
                distance_weight=1.0,
                ranking_weight=1.0,
                ranking_margin=1.0,
            )
            self.assertTrue(torch.isfinite(value), msg=loss_name)

    def test_soft_improvement_gate_requires_both_hit_and_mae(self) -> None:
        hard = {
            "in_interval_rate_mean": 0.4,
            "mae_samples_mean": 4.0,
        }
        self.assertTrue(
            probe.soft_clearly_improves(
                hard,
                {
                    "in_interval_rate_mean": 0.5,
                    "mae_samples_mean": 3.0,
                },
            )
        )
        self.assertFalse(
            probe.soft_clearly_improves(
                hard,
                {
                    "in_interval_rate_mean": 0.5,
                    "mae_samples_mean": 5.0,
                },
            )
        )


if __name__ == "__main__":
    unittest.main()
