#!/usr/bin/env python3
"""Synthetic tests for Robo-Dopamine incremental-hop detector logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from robo_incremental_hop.core import (
    detect_hop_scale,
    detector_mask,
    evaluate_event,
    make_config,
    positive_episode_count,
    recovery_metrics,
)


class IncrementalHopDetectorTests(unittest.TestCase):
    def config(
        self,
        family: str,
        **kwargs,
    ):
        return make_config(
            "synthetic",
            family,
            **kwargs,
        )

    def test_always_positive_progress(self) -> None:
        hops = [
            0.20,
            0.15,
            0.08,
            0.12,
            0.04,
        ]
        mask = detector_mask(
            hops,
            self.config(
                "consecutive",
                epsilon=0.0,
                n=2,
            ),
        )
        self.assertEqual(
            mask,
            [
                False,
                False,
                False,
                False,
                False,
            ],
        )
        event = evaluate_event(
            [0, 4, 8, 12, 16],
            mask,
            5,
        )
        self.assertFalse(
            event["detected"]
        )
        self.assertFalse(
            event["recall_at_3"]
        )

    def test_isolated_negative_mistake(self) -> None:
        hops = [
            0.20,
            -0.20,
            0.15,
            0.10,
        ]
        mask = detector_mask(
            hops,
            self.config(
                "consecutive",
                epsilon=0.0,
                n=2,
            ),
        )
        self.assertFalse(any(mask))
        self.assertEqual(
            positive_episode_count(mask),
            0,
        )

    def test_sustained_negative_regression(self) -> None:
        frames = [
            0,
            4,
            8,
            12,
            16,
            20,
        ]
        hops = [
            0.20,
            0.10,
            -0.15,
            -0.20,
            -0.30,
            -0.25,
        ]
        mask = detector_mask(
            hops,
            self.config(
                "consecutive",
                epsilon=-0.05,
                n=3,
            ),
        )
        self.assertEqual(
            mask,
            [
                False,
                False,
                False,
                False,
                True,
                True,
            ],
        )
        event = evaluate_event(
            frames,
            mask,
            observable_onset_frame=9,
        )
        self.assertEqual(
            event["detection_frame"],
            16,
        )
        self.assertEqual(
            event["delay_frames"],
            7,
        )
        self.assertEqual(
            event["delay_samples"],
            2,
        )
        self.assertTrue(
            event["recall_at_3"]
        )

    def test_near_zero_stagnation(self) -> None:
        hops = [
            0.30,
            0.01,
            -0.01,
            0.00,
            0.02,
        ]
        stagnation = detector_mask(
            hops,
            self.config(
                "stagnation_consecutive",
                delta=0.02,
                n=3,
            ),
        )
        regression = detector_mask(
            hops,
            self.config(
                "consecutive",
                epsilon=-0.05,
                n=1,
            ),
        )
        self.assertEqual(
            stagnation,
            [
                False,
                False,
                False,
                True,
                True,
            ],
        )
        self.assertFalse(
            any(regression)
        )

    def test_one_positive_mistake_inside_negative_run(self) -> None:
        hops = [
            -0.20,
            -0.15,
            0.05,
            -0.30,
            -0.25,
        ]
        strict = detector_mask(
            hops,
            self.config(
                "consecutive",
                epsilon=0.0,
                n=3,
            ),
        )
        relaxed = detector_mask(
            hops,
            self.config(
                "k_of_m",
                epsilon=0.0,
                m=5,
                k=4,
            ),
        )
        self.assertFalse(any(strict))
        self.assertEqual(
            relaxed,
            [
                False,
                False,
                False,
                False,
                True,
            ],
        )

    def test_regression_followed_by_recovery(self) -> None:
        frames = [
            0,
            4,
            8,
            12,
            16,
            20,
            24,
        ]
        hops = [
            0.2,
            -0.2,
            -0.3,
            -0.2,
            0.25,
            0.30,
            0.20,
        ]
        mask = detector_mask(
            hops,
            self.config(
                "consecutive",
                epsilon=0.0,
                n=2,
            ),
        )
        metrics = recovery_metrics(
            frames,
            hops,
            hops,
            mask,
            recovery_frame=13,
            window_samples=2,
        )
        self.assertTrue(
            metrics[
                "detector_positive_at_recovery_state"
            ]
        )
        self.assertEqual(
            metrics[
                "recovery_state_frame"
            ],
            12,
        )
        self.assertEqual(
            metrics[
                "first_detector_negative_frame_at_or_after_recovery"
            ],
            16,
        )
        self.assertEqual(
            metrics[
                "clearance_delay_samples"
            ],
            1,
        )
        self.assertEqual(
            metrics[
                "clearance_delay_frames"
            ],
            3,
        )
        self.assertLess(
            metrics[
                "hop_before_recovery_mean"
            ],
            0,
        )
        self.assertGreater(
            metrics[
                "hop_after_recovery_mean"
            ],
            0,
        )

    def test_early_alarm_and_hop_scale_contracts(self) -> None:
        event = evaluate_event(
            [0, 4, 8, 12, 16],
            [
                False,
                True,
                True,
                False,
                True,
            ],
            observable_onset_frame=10,
        )
        self.assertTrue(
            event["early_positive_any"]
        )
        self.assertEqual(
            event["detection_frame"],
            16,
        )
        self.assertEqual(
            event["delay_samples"],
            2,
        )

        official = detect_hop_scale(
            [
                {
                    "hop": 0.03,
                    "pred": (
                        "<score>+3%</score>"
                    ),
                },
                {
                    "hop": -0.10,
                    "pred": (
                        "<score>-10%</score>"
                    ),
                },
            ]
        )
        percentage = detect_hop_scale(
            [
                {
                    "hop": 3.0,
                    "pred": (
                        "<score>+3%</score>"
                    ),
                },
                {
                    "hop": -10.0,
                    "pred": (
                        "<score>-10%</score>"
                    ),
                },
            ]
        )
        self.assertEqual(
            official["source_scale"],
            "normalized_-1_to_1",
        )
        self.assertEqual(
            percentage[
                "source_scale"
            ],
            (
                "percentage_points_"
                "-100_to_100"
            ),
        )
        self.assertEqual(
            percentage[
                "normalized_hops"
            ],
            [0.03, -0.10],
        )


if __name__ == "__main__":
    unittest.main()
