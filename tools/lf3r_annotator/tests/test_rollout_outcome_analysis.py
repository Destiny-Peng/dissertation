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
import analyze_baseline_rollout_outcomes as outcome_analysis  # noqa: E402


def rollout(
    outcome: str,
    progress: float,
    *,
    rollout_id: str,
    maximum_progress: float | None = None,
) -> tuple[str, dict]:
    return rollout_id, {
        "record": {
            "id": rollout_id,
            "task_suite": "libero_10",
            "task_id": 0,
            "total_frames": 10,
        },
        "annotation": {"outcome_label": outcome},
        "methods": {
            "safe": {
                "signals": {
                    "max_token_prob": {
                        "frames": np.asarray([0, 9], dtype=int),
                        "values": np.asarray([0.1, 0.9], dtype=float),
                        "direction": 1.0,
                    }
                }
            },
            "procvlm": {
                "signals": {
                    "progress": {
                        "frames": np.asarray([0, 5, 9], dtype=int),
                        "values": np.asarray(
                            [
                                0.0,
                                progress if maximum_progress is None else maximum_progress,
                                progress,
                            ],
                            dtype=float,
                        ),
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
        self.assertEqual((int(q95["tp"]), int(q95["fn"])), (1, 1))
        self.assertEqual((int(q95["fp"]), int(q95["tn"])), (0, 2))
        self.assertAlmostEqual(float(q95["accuracy"]), 0.75)
        self.assertAlmostEqual(float(q95["success_recall"]), 0.5)
        self.assertAlmostEqual(float(q95["failure_recall"]), 1.0)
        self.assertAlmostEqual(float(q95["specificity"]), 1.0)
        self.assertAlmostEqual(float(q95["false_positive_rate"]), 0.0)
        self.assertAlmostEqual(float(q95["balanced_accuracy"]), 0.75)
        self.assertAlmostEqual(float(q95["precision"]), 1.0)
        self.assertAlmostEqual(float(q95["f1"]), 2.0 / 3.0)
        self.assertNotIn("auroc", q95.index)
        self.assertEqual(q95["success_rule"], "progress > 0.805")
        self.assertEqual(q95["failure_rule"], "progress <= 0.805")
        self.assertEqual(q95["positive_class"], "clean_success+recovered_success")
        self.assertEqual(q95["negative_class"], "terminal_failure")

        self.assertNotIn("safe", set(summary["method"]))
        self.assertNotIn("safe", set(predictions["method"]))

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

        sweep = outcome_analysis._progress_threshold_sweep(predictions, rows)
        proc_80 = sweep[
            (sweep["method"] == "procvlm")
            & (sweep["score_aggregation"] == "final")
            & (sweep["raw_progress_threshold"] == 0.8)
        ].iloc[0]
        self.assertEqual(int(proc_80["n_resolved"]), 4)
        self.assertAlmostEqual(float(proc_80["accuracy"]), 0.75)
        self.assertAlmostEqual(float(proc_80["failure_recall"]), 1.0)
        self.assertNotIn("safe", set(sweep["method"]))
        self.assertEqual(
            set(sweep["score_aggregation"]),
            {"final", "maximum"},
        )

    def test_maximum_progress_sweep_uses_peak_value(self) -> None:
        rows = dict([
            rollout(
                "success",
                0.9,
                maximum_progress=0.95,
                rollout_id="success-high",
            ),
            rollout(
                "recovered_success",
                0.8,
                maximum_progress=0.85,
                rollout_id="success-recovered",
            ),
            rollout(
                "failure",
                0.2,
                maximum_progress=0.6,
                rollout_id="failure-a",
            ),
            rollout(
                "failure",
                0.1,
                maximum_progress=0.4,
                rollout_id="failure-b",
            ),
        ])
        _summary, predictions = analysis.compute_rollout_outcome_classification(rows)
        sweep = outcome_analysis._progress_threshold_sweep(predictions, rows)

        final_80 = sweep[
            (sweep["method"] == "procvlm")
            & (sweep["score_aggregation"] == "final")
            & (sweep["raw_progress_threshold"] == 0.8)
        ].iloc[0]
        maximum_80 = sweep[
            (sweep["method"] == "procvlm")
            & (sweep["score_aggregation"] == "maximum")
            & (sweep["raw_progress_threshold"] == 0.8)
        ].iloc[0]

        self.assertAlmostEqual(float(final_80["accuracy"]), 0.75)
        self.assertAlmostEqual(float(maximum_80["accuracy"]), 1.0)
        self.assertAlmostEqual(float(maximum_80["success_recall"]), 1.0)
        self.assertAlmostEqual(float(maximum_80["failure_recall"]), 1.0)


if __name__ == "__main__":
    unittest.main()
