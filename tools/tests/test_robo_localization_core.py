#!/usr/bin/env python3
"""Synthetic tests for the shared Robo-Dopamine localization training core."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from robo_localization_head import core


class LocalizationCoreTests(unittest.TestCase):
    def dataset(self):
        result = {}
        for index, length in enumerate((5, 7, 4, 6, 8, 5)):
            labels = np.zeros(length, dtype=np.float32)
            labels[min(2, length - 1)] = 1.0
            result[f"r{index}"] = {
                "task_key": f"task{index % 2}",
                "sequence": np.stack(
                    [
                        np.linspace(0.0, 1.0, length, dtype=np.float32),
                        np.linspace(0.2, -0.2, length, dtype=np.float32),
                    ],
                    axis=1,
                ),
                "labels": labels,
            }
        return result

    def test_variable_length_collate_and_packed_forward(self) -> None:
        dataset = self.dataset()
        mean, std = core.standardization_stats(dataset, list(dataset))
        prepared = core.prepare_tensor_dataset(dataset, mean, std)
        x, y, lengths = core.collate_rollout_batch(
            prepared,
            ["r0", "r1", "r2"],
            torch.device("cpu"),
        )
        self.assertEqual(tuple(x.shape), (3, 7, 2))
        self.assertEqual(tuple(y.shape), (3, 7))
        self.assertEqual(lengths.tolist(), [5, 7, 4])
        logits = core.TinyBiLSTM(hidden=16)(x, lengths)
        self.assertEqual(tuple(logits.shape), (3, 7))

    def test_rollout_split_has_no_overlap(self) -> None:
        dataset = self.dataset()
        split = core.rollout_split(
            dataset,
            seed=17,
            train_fraction=0.6,
            val_fraction=0.2,
        )
        train, val, test = map(set, (split["train"], split["val"], split["test"]))
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(train | val | test, set(dataset))

    def test_shared_trainer_runs_minibatches(self) -> None:
        dataset = self.dataset()
        train_ids = ["r0", "r1", "r2", "r3"]
        val_ids = ["r4", "r5"]
        mean, std = core.standardization_stats(dataset, train_ids)

        def row_loss(logits, _row, labels, pos_weight):
            return F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=pos_weight,
            )

        model, metadata = core.train_bilstm(
            dataset=dataset,
            train_ids=train_ids,
            val_ids=val_ids,
            mean=mean,
            std=std,
            hidden=8,
            pos_weight=2.0,
            seed=17,
            epochs=2,
            patience=2,
            learning_rate=0.003,
            weight_decay=1e-4,
            grad_clip=5.0,
            batch_size=2,
            device=torch.device("cpu"),
            row_loss_fn=row_loss,
            progress_label="test",
            logger=lambda _message: None,
        )
        self.assertIsInstance(model, core.TinyBiLSTM)
        self.assertEqual(metadata["effective_train_batch_size"], 2)
        self.assertEqual(metadata["optimizer_steps_per_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
