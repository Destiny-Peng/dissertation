#!/usr/bin/env python3
"""Synthetic tests for the lightweight Robo-Dopamine localization head."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

SCRIPT = TOOLS_DIR / "train_robo_dopamine_localization_head.py"
SPEC = importlib.util.spec_from_file_location("robo_localization_head_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class RoboLocalizationHeadTests(unittest.TestCase):
    def synthetic_dataset(self, count: int = 18):
        dataset = {}
        for index in range(count):
            task_id = index % 3
            rollout_id = f"r{index:02d}"
            context = np.zeros((9, 2, 7), dtype=np.float64)
            context[:, 0, :] = np.linspace(0.0, 1.0, 9)[:, None]
            context[:, 1, :] = 0.2
            context[4:6, 1, :] = -0.4
            labels = np.zeros(9, dtype=np.float64)
            labels[4:6] = 1.0
            dataset[rollout_id] = {
                "event": {
                    "event_id": rollout_id + "::event0",
                    "task_key": f"libero_10:{task_id}",
                    "task_id": task_id,
                    "failure_type": "grasp_failure",
                },
                "frames": np.arange(9, dtype=np.int64),
                "context": context,
                "labels": labels,
                "causal_index": 4,
                "observable_index": 5,
                "progress": np.asarray(
                    [0.0, 0.1, 0.2, 0.3, 0.8, 0.5, 0.4, 0.3, 0.2],
                    dtype=np.float64,
                ),
                "sequence": np.stack(
                    [
                        np.asarray(
                            [0.0, 0.1, 0.2, 0.3, 0.8, 0.5, 0.4, 0.3, 0.2],
                            dtype=np.float64,
                        ),
                        np.asarray(
                            [0.2, 0.2, 0.2, 0.2, -0.4, -0.4, 0.2, 0.2, 0.2],
                            dtype=np.float64,
                        ),
                    ],
                    axis=1,
                ),
            }
        return dataset

    def test_primary_event_and_context_semantics(self) -> None:
        signals = {
            "r": {
                "frames": [0, 2, 4, 6, 8],
                "progress": [0.0, 0.1, 0.2, 0.3, 0.4],
                "hops": [0.0, 0.1, 0.1, 0.1, 0.1],
            }
        }
        events = [
            {
                "rollout_id": "r",
                "event_id": "r::event1",
                "event_index": 1,
                "outcome": "terminal_failure",
                "failure_type": "grasp_failure",
                "task_key": "libero_10:0",
                "task_id": 0,
                "causal_onset_frame": 6,
                "observable_onset_frame": 8,
            },
            {
                "rollout_id": "r",
                "event_id": "r::event0",
                "event_index": 0,
                "outcome": "terminal_failure",
                "failure_type": "grasp_failure",
                "task_key": "libero_10:0",
                "task_id": 0,
                "causal_onset_frame": 2,
                "observable_onset_frame": 4,
            },
        ]
        dataset = probe.build_dataset(signals, events, radius=2)
        row = dataset["r"]
        self.assertEqual(row["event"]["event_id"], "r::event0")
        self.assertEqual(row["context"].shape, (5, 2, 5))
        self.assertEqual(row["sequence"].shape, (5, 2))
        self.assertEqual(row["causal_index"], 1)
        self.assertEqual(row["observable_index"], 2)
        self.assertEqual(row["labels"].tolist(), [0.0, 1.0, 1.0, 0.0, 0.0])

    def test_rollout_split_never_splits_timesteps(self) -> None:
        dataset = self.synthetic_dataset()
        split = probe.rollout_split(
            dataset,
            seed=17,
            train_fraction=0.70,
            val_fraction=0.15,
        )
        train, val, test = map(set, (split["train"], split["val"], split["test"]))
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(train | val | test, set(dataset))
        self.assertGreaterEqual(len(val), 2)
        self.assertGreaterEqual(len(test), 2)

    def test_task_holdout_uses_entire_task_only_for_test(self) -> None:
        dataset = self.synthetic_dataset()
        splits = probe.task_splits(dataset, seed=17, min_test_rollouts=2)
        self.assertEqual(len(splits), 3)
        for split in splits:
            held = split["held_out_task"]
            self.assertTrue(
                all(
                    dataset[rid]["event"]["task_key"] == held
                    for rid in split["test"]
                )
            )
            self.assertTrue(
                all(
                    dataset[rid]["event"]["task_key"] != held
                    for rid in split["train"] + split["val"]
                )
            )

    def test_lightweight_models_train_without_extra_dependencies(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(size=(320, 2, 7))
        y = (x[:, 1, 3] < -0.2).astype(np.float64)
        xv = rng.normal(size=(100, 2, 7))
        yv = (xv[:, 1, 3] < -0.2).astype(np.float64)

        local_models = [
            name for name in probe.MODEL_NAMES
            if name not in probe.SEQUENCE_MODEL_NAMES
        ]
        for index, name in enumerate(local_models):
            model = probe.make_model(
                name,
                seed=10 + index,
                epochs=100,
                patience=20,
            )
            training = model.fit(x, y, xv, yv)
            logits = model.logits(xv)
            self.assertEqual(logits.shape, (len(xv),))
            self.assertTrue(np.isfinite(logits).all())
            self.assertGreater(training["best_epoch"], 0)


    def test_bilstm_uses_full_rollout_sequence(self) -> None:
        dataset = self.synthetic_dataset(8)
        train_ids = ["r00", "r01", "r02", "r03"]
        val_ids = ["r04", "r05"]
        test_ids = ["r06", "r07"]
        metrics, rows, training = probe.train_and_evaluate_sequence_model(
            "tiny_bilstm_h16",
            dataset,
            train_ids,
            val_ids,
            test_ids,
            seed=23,
            epochs=8,
            patience=4,
            split={
                "kind": "rollout_random",
                "split_id": "synthetic",
            },
            train_size_label="4",
        )
        self.assertEqual(metrics["test_rollout_n"], 2)
        self.assertEqual(len(rows), 2)
        self.assertTrue(training["sequence_model"])
        self.assertEqual(training["hidden"], 16)
        self.assertTrue(all(np.isfinite(row["score"]) for row in rows))

    def test_interval_metrics_match_requested_definition(self) -> None:
        dataset = self.synthetic_dataset(1)
        rows = [
            probe.prediction_row(dataset, "r00", 2, 0.5),
            probe.prediction_row(dataset, "r00", 4, 0.5),
            probe.prediction_row(dataset, "r00", 8, 0.5),
        ]
        summary = probe.metric_summary(rows)
        self.assertAlmostEqual(summary["in_interval_rate"], 1 / 3)
        self.assertAlmostEqual(summary["before_interval_rate"], 1 / 3)
        self.assertAlmostEqual(summary["after_interval_rate"], 1 / 3)
        self.assertAlmostEqual(summary["mae_samples"], 5 / 3)
        self.assertAlmostEqual(summary["mse_samples"], 13 / 3)
        self.assertAlmostEqual(summary["within_1"], 1 / 3)
        self.assertAlmostEqual(summary["within_3"], 1.0)

    def test_baseline_selection_uses_validation_only(self) -> None:
        dataset = self.synthetic_dataset(6)
        val_ids = ["r00", "r01"]
        bank = {
            ("good",): {"r00": 4, "r01": 5, "r02": 0},
            ("bad",): {"r00": 0, "r01": 0, "r02": 4},
        }
        winner, metrics = probe.choose_baseline_candidate(
            bank,
            dataset,
            val_ids,
        )
        self.assertEqual(winner, ("good",))
        self.assertAlmostEqual(metrics["in_interval_rate"], 1.0)

    def test_baseline_selection_does_not_reward_partial_coverage(self) -> None:
        dataset = self.synthetic_dataset(6)
        val_ids = ["r00", "r01"]
        bank = {
            ("partial_perfect",): {"r00": 4},
            ("full",): {"r00": 4, "r01": 2},
        }
        winner, metrics = probe.choose_baseline_candidate(
            bank,
            dataset,
            val_ids,
        )
        self.assertEqual(winner, ("full",))
        self.assertAlmostEqual(metrics["coverage"], 1.0)
        self.assertAlmostEqual(metrics["coverage_floor"], 1.0)

    def test_learning_curve_subsets_are_nested_for_same_seed(self) -> None:
        dataset = self.synthetic_dataset(30)
        ids = sorted(dataset)
        ten = set(probe.training_subset(dataset, ids, 10, seed=1707))
        twenty = set(probe.training_subset(dataset, ids, 20, seed=1707))
        thirty = set(probe.training_subset(dataset, ids, 30, seed=1707))
        self.assertEqual(len(ten), 10)
        self.assertEqual(len(twenty), 20)
        self.assertTrue(ten < twenty)
        self.assertTrue(twenty <= thirty)


if __name__ == "__main__":
    unittest.main()
