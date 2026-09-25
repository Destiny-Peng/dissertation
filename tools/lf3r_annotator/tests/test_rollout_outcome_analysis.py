from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import analyze_baseline_temporal_signals as analysis  # noqa: E402


def rollout(outcome: str, progress: float, *, rollout_id: str) -> tuple[str, dict]:
    return rollout_id, {
        "record": {
            "id": rollout_id,
            "task_suite": "libero_10",
            "task_id": 0,
            "total_frames": 10,
        },
        "annotation": {"outcome_label": outcome},
        "methods": {
            "procvlm": {
                "signals": {
                    "progress": {
                        "frames": np.asarray([0, 9], dtype=int),
                        "values": np.asarray([0.0, progress], dtype=float),
                        "direction": -1.0,
                    }
                }
            }
        },
        "method_errors": {},
    }


class RolloutOutcomeClassificationTests(unittest.TestCase):
    def test_q95_terminal_progress_metrics_and_uncertain_exclusion(self) -> None:
        rows = dict([
            rollout("success", 0.9, rollout_id="success-high"),
            rollout("recovered_success", 0.8, rollout_id="success-recovered"),
            rollout("failure", 0.2, rollout_id="failure-a"),
            rollout("failure", 0.1, rollout_id="failure-b"),
            rollout("uncertain", 0.3, rollout_id="uncertain"),
        ])

        summary, predictions = analysis.compute_rollout_outcome_classification(rows)
        q95 = summary[
            (summary["method"] == "procvlm")
            & (summary["threshold"] == "q95")
        ].iloc[0]

        self.assertEqual(int(q95["n_resolved"]), 4)
        self.assertEqual(int(q95["n_final_success"]), 2)
        self.assertEqual(int(q95["n_terminal_failure"]), 2)
        self.assertEqual((int(q95["tp"]), int(q95["fn"])), (2, 0))
        self.assertEqual((int(q95["fp"]), int(q95["tn"])), (1, 1))
        self.assertAlmostEqual(float(q95["accuracy"]), 0.75)
        self.assertAlmostEqual(float(q95["failure_recall"]), 1.0)
        self.assertAlmostEqual(float(q95["specificity"]), 0.5)
        self.assertAlmostEqual(float(q95["false_positive_rate"]), 0.5)
        self.assertAlmostEqual(float(q95["balanced_accuracy"]), 0.75)
        self.assertAlmostEqual(float(q95["precision"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(q95["f1"]), 0.8)
        self.assertAlmostEqual(float(q95["auroc"]), 1.0)
        self.assertEqual(q95["failure_rule"], "progress <= 0.805")

        q95_predictions = predictions[
            (predictions["method"] == "procvlm")
            & (predictions["threshold"] == "q95")
        ]
        self.assertEqual(len(q95_predictions), 5)
        uncertain = q95_predictions[q95_predictions["rollout_id"] == "uncertain"].iloc[0]
        self.assertFalse(bool(uncertain["included_in_metrics"]))
        self.assertTrue(
            uncertain["prediction_correct"] is None
            or (
                isinstance(uncertain["prediction_correct"], float)
                and math.isnan(uncertain["prediction_correct"])
            )
        )


if __name__ == "__main__":
    unittest.main()
