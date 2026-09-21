#!/usr/bin/env python3
"""Synthetic tests for the BiLSTM success-negative localization ablation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

SCRIPT = TOOLS_DIR / "train_robo_dopamine_localization_head.py"
SPEC = importlib.util.spec_from_file_location("robo_localization_head_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class RoboLocalizationHeadTests(unittest.TestCase):
    def failure_dataset(self, count: int = 18):
        dataset = {}
        for index in range(count):
            task_id = index % 3
            rollout_id = f"f{index:02d}"
            sequence = np.stack(
                [
                    np.asarray(
                        [0.0, 0.1, 0.2, 0.3, 0.8, 0.5, 0.4, 0.3, 0.2],
                        dtype=np.float32,
                    ),
                    np.asarray(
                        [0.2, 0.2, 0.2, 0.2, -0.4, -0.4, 0.2, 0.2, 0.2],
                        dtype=np.float32,
                    ),
                ],
                axis=1,
            )
            labels = np.zeros(9, dtype=np.float32)
            labels[4:6] = 1.0
            dataset[rollout_id] = {
                "rollout_id": rollout_id,
                "kind": "failure",
                "event": {
                    "event_id": rollout_id + "::event0",
                    "task_key": f"libero_10:task{task_id}",
                    "task_id": task_id,
                    "failure_type": "grasp_failure",
                },
                "task_key": f"libero_10:task{task_id}",
                "task_id": task_id,
                "frames": np.arange(9, dtype=np.int64),
                "labels": labels,
                "causal_index": 4,
                "observable_index": 5,
                "sequence": sequence,
            }
        return dataset

    def success_dataset(self, count: int = 9):
        dataset = {}
        for index in range(count):
            task_id = index % 3
            rollout_id = f"s{index:02d}"
            sequence = np.stack(
                [
                    np.linspace(0.0, 1.0, 9, dtype=np.float32),
                    np.full(9, 0.1, dtype=np.float32),
                ],
                axis=1,
            )
            dataset[rollout_id] = {
                "rollout_id": rollout_id,
                "kind": "clean_success",
                "task_key": f"libero_10:task{task_id}",
                "task_id": task_id,
                "frames": np.arange(9, dtype=np.int64),
                "labels": np.zeros(9, dtype=np.float32),
                "sequence": sequence,
            }
        return dataset

    def test_build_failure_and_success_labels(self) -> None:
        signals = {
            "failure": {
                "frames": [0, 2, 4, 6, 8],
                "progress": [0.0, 0.1, 0.2, 0.3, 0.4],
                "hops": [0.0, 0.1, 0.1, -0.2, 0.1],
                "task_key": "libero_10:task0",
                "task_id": "0",
            },
            "success": {
                "frames": [0, 2, 4, 6, 8],
                "progress": [0.0, 0.2, 0.4, 0.7, 1.0],
                "hops": [0.0, 0.2, 0.2, 0.3, 0.3],
                "task_key": "libero_10:task0",
                "task_id": "0",
            },
        }
        events = [{
            "rollout_id": "failure",
            "event_id": "failure::event0",
            "event_index": 0,
            "outcome": "terminal_failure",
            "failure_type": "grasp_failure",
            "task_key": "libero_10:task0",
            "task_id": "0",
            "causal_onset_frame": 2,
            "observable_onset_frame": 4,
        }]
        failures = probe.build_failure_dataset(signals, events)
        successes = probe.build_success_dataset(
            signals,
            [{
                "rollout_id": "success",
                "task_key": "libero_10:task0",
                "task_id": "0",
            }],
        )
        self.assertEqual(
            failures["failure"]["labels"].tolist(),
            [0.0, 1.0, 1.0, 0.0, 0.0],
        )
        self.assertEqual(successes["success"]["labels"].tolist(), [0.0] * 5)
        self.assertEqual(failures["failure"]["sequence"].shape, (5, 2))

    def test_rollout_split_has_no_overlap(self) -> None:
        dataset = self.failure_dataset()
        split = probe.rollout_split(
            dataset,
            seed=17,
            train_fraction=0.70,
            val_fraction=0.15,
        )
        train, val, test = map(
            set, (split["train"], split["val"], split["test"])
        )
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(train | val | test, set(dataset))

    def test_success_ratio_subsets_are_nested_and_same_task_first(self) -> None:
        failures = self.failure_dataset(30)
        successes = self.success_dataset(30)
        failure_train = ["f00", "f03", "f06", "f09", "f12"]
        subsets, meta = probe.build_success_ratio_subsets(
            successes,
            failures,
            failure_train,
            seed=1701,
        )
        self.assertEqual(len(subsets[0.0]), 0)
        self.assertEqual(len(subsets[0.5]), 2)
        self.assertEqual(len(subsets[1.0]), 5)
        self.assertEqual(len(subsets[2.0]), 10)
        self.assertTrue(set(subsets[0.5]) <= set(subsets[1.0]))
        self.assertTrue(set(subsets[1.0]) <= set(subsets[2.0]))
        same_task_n = meta["same_task_available_n"]
        prefix = subsets[2.0][: min(same_task_n, len(subsets[2.0]))]
        self.assertTrue(
            all(successes[rid]["task_key"] == "libero_10:task0" for rid in prefix)
        )

    def test_success_ratio_floor_matches_requested_example(self) -> None:
        failures = self.failure_dataset(30)
        successes = self.success_dataset(60)
        failure_train = [f"f{index:02d}" for index in range(25)]
        subsets, _ = probe.build_success_ratio_subsets(
            successes,
            failures,
            failure_train,
            seed=17,
        )
        self.assertEqual(len(subsets[0.0]), 0)
        self.assertEqual(len(subsets[0.5]), 12)
        self.assertEqual(len(subsets[1.0]), 25)
        self.assertEqual(len(subsets[2.0]), 50)

    def test_task_holdout_excludes_success_from_heldout_task(self) -> None:
        failures = self.failure_dataset(18)
        successes = self.success_dataset(18)
        failure_train = ["f01", "f02", "f04", "f05", "f07", "f08"]
        subsets, _ = probe.build_success_ratio_subsets(
            successes,
            failures,
            failure_train,
            seed=17,
            held_out_task="libero_10:task0",
        )
        self.assertTrue(
            all(
                successes[rid]["task_key"] != "libero_10:task0"
                for ids in subsets.values()
                for rid in ids
            )
        )

    def test_bilstm_forward_shape(self) -> None:
        model = probe.TinyBiLSTM(hidden=16)
        x = torch.randn(1, 11, 2)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (1, 11))

    def test_shared_normalization_and_pos_weight_ignore_success(self) -> None:
        failures = self.failure_dataset(8)
        successes = self.success_dataset(6)
        train_ids = ["f00", "f01", "f02", "f03"]
        mean_a, std_a = probe.standardization_stats(failures, train_ids)
        pos_a = probe.positive_weight_from_failures(failures, train_ids)

        # Mutating success data must not affect either controlled statistic.
        for row in successes.values():
            row["sequence"][:] = 999.0
            row["labels"][:] = 0.0
        mean_b, std_b = probe.standardization_stats(failures, train_ids)
        pos_b = probe.positive_weight_from_failures(failures, train_ids)
        np.testing.assert_allclose(mean_a, mean_b)
        np.testing.assert_allclose(std_a, std_b)
        self.assertEqual(pos_a, pos_b)

    def test_success_ratio_ablation_keeps_failure_split_fixed(self) -> None:
        failures = self.failure_dataset(18)
        successes = self.success_dataset(18)
        split = probe.rollout_split(
            failures,
            seed=17,
            train_fraction=0.70,
            val_fraction=0.15,
        )
        subsets, _ = probe.build_success_ratio_subsets(
            successes,
            failures,
            split["train"],
            seed=1701,
        )
        self.assertTrue(set(split["train"]).isdisjoint(split["test"]))
        self.assertTrue(set(split["val"]).isdisjoint(split["test"]))
        self.assertEqual(
            probe.parse_success_ratios("0.5,1,2"),
            (0.0, 0.5, 1.0, 2.0),
        )
        self.assertTrue(set(subsets[0.5]) <= set(subsets[1.0]))
        self.assertTrue(set(subsets[1.0]) <= set(subsets[2.0]))

    def test_success_ratio_parser_is_configurable(self) -> None:
        self.assertEqual(
            probe.parse_success_ratios("1.5,0.25,0.5,0.5"),
            (0.0, 0.25, 0.5, 1.5),
        )
        with self.assertRaises(ValueError):
            probe.parse_success_ratios("0,-1")

    def test_interval_metrics_match_definition(self) -> None:
        dataset = self.failure_dataset(1)
        rows = [
            probe.prediction_row(dataset, "f00", 2, 0.5),
            probe.prediction_row(dataset, "f00", 4, 0.5),
            probe.prediction_row(dataset, "f00", 8, 0.5),
        ]
        summary = probe.metric_summary(rows)
        self.assertAlmostEqual(summary["in_interval_rate"], 1 / 3)
        self.assertAlmostEqual(summary["before_interval_rate"], 1 / 3)
        self.assertAlmostEqual(summary["after_interval_rate"], 1 / 3)
        self.assertAlmostEqual(summary["mae_samples"], 5 / 3)
        self.assertAlmostEqual(summary["mse_samples"], 13 / 3)
        self.assertAlmostEqual(summary["within_1"], 1 / 3)
        self.assertAlmostEqual(summary["within_3"], 1.0)


if __name__ == "__main__":
    unittest.main()
