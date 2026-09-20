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
from robo_incremental_hop.report import build_pairwise_overlap_rows


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

    def test_delay_profile_and_eventual_recall_stop_at_recovery(self) -> None:
        frames = [0, 4, 8, 12, 16, 20, 24]
        mask = [False, False, False, True, False, True, True]

        detected = evaluate_event(
            frames,
            mask,
            observable_onset_frame=9,
            episode_end_frame=17,
            episode_end_source="recovery_frame",
        )
        self.assertTrue(detected["recall_at_1"])
        self.assertTrue(detected["recall_at_3"])
        self.assertTrue(detected["recall_at_5"])
        self.assertTrue(detected["recall_at_10"])
        self.assertTrue(detected["recall_at_20"])
        self.assertTrue(detected["eventual_recall"])
        self.assertEqual(detected["detection_frame"], 12)
        self.assertEqual(detected["episode_end_source"], "recovery_frame")

        late_only = evaluate_event(
            frames,
            [False, False, False, False, False, True, True],
            observable_onset_frame=9,
            episode_end_frame=17,
            episode_end_source="recovery_frame",
        )
        self.assertFalse(late_only["eventual_recall"])
        self.assertFalse(late_only["recall_at_20"])
        self.assertIsNone(late_only["detection_frame"])

    def test_recall_at_20_extends_delay_profile(self) -> None:
        frames = list(range(0, 100, 4))
        mask = [False] * len(frames)
        mask[18] = True
        event = evaluate_event(
            frames,
            mask,
            observable_onset_frame=4,
        )
        self.assertFalse(event["recall_at_10"])
        self.assertTrue(event["recall_at_20"])
        self.assertTrue(event["eventual_recall"])

    def test_pairwise_overlap_tracks_tp_and_clean_fp_complementarity(self) -> None:
        best_rows = [
            {
                "signal_mode": "incremental",
                "clean_fpr_constraint": 0.20,
                "selection_status": "selected",
                "detector_family": "consecutive",
                "config_id": "a",
            },
            {
                "signal_mode": "incremental",
                "clean_fpr_constraint": 0.20,
                "selection_status": "selected",
                "detector_family": "stagnation_consecutive",
                "config_id": "b",
            },
        ]

        def event_row(
            config_id: str,
            event_id: str,
            failure_type: str,
            detected: bool,
            delay: int | None,
        ) -> dict:
            return {
                "signal_mode": "incremental",
                "config_id": config_id,
                "event_id": event_id,
                "rollout_id": event_id.split("::")[0],
                "event_index": int(event_id.rsplit("event", 1)[1]),
                "failure_type": failure_type,
                "recall_at_1": bool(detected and delay is not None and delay <= 1),
                "recall_at_3": bool(detected and delay is not None and delay <= 3),
                "recall_at_5": bool(detected and delay is not None and delay <= 5),
                "recall_at_10": bool(detected and delay is not None and delay <= 10),
                "recall_at_20": bool(detected and delay is not None and delay <= 20),
                "eventual_recall": detected,
                "delay_samples": delay,
                "delay_frames": None if delay is None else delay * 4,
            }

        event_rows = [
            event_row("a", "r0::event0", "timeout_no_progress", True, 1),
            event_row("a", "r1::event0", "grasp_failure", True, 2),
            event_row("a", "r2::event0", "grasp_failure", False, None),
            event_row("b", "r0::event0", "timeout_no_progress", False, None),
            event_row("b", "r1::event0", "grasp_failure", True, 3),
            event_row("b", "r2::event0", "grasp_failure", True, 2),
        ]
        clean_rows = [
            {"signal_mode": "incremental", "config_id": "a", "rollout_id": "c0", "any_positive": True},
            {"signal_mode": "incremental", "config_id": "a", "rollout_id": "c1", "any_positive": False},
            {"signal_mode": "incremental", "config_id": "b", "rollout_id": "c0", "any_positive": True},
            {"signal_mode": "incremental", "config_id": "b", "rollout_id": "c1", "any_positive": True},
        ]

        summary, by_failure = build_pairwise_overlap_rows(
            best_rows,
            event_rows,
            clean_rows,
        )
        at3 = next(row for row in summary if row["horizon"] == "3")
        self.assertEqual(at3["event_n"], 3)
        self.assertEqual(at3["a_detected_n"], 2)
        self.assertEqual(at3["b_detected_n"], 2)
        self.assertEqual(at3["overlap_n"], 1)
        self.assertEqual(at3["a_only_n"], 1)
        self.assertEqual(at3["b_only_n"], 1)
        self.assertEqual(at3["or_detected_n"], 3)
        self.assertAlmostEqual(at3["or_recall"], 1.0)
        self.assertAlmostEqual(at3["tp_jaccard"], 1.0 / 3.0)
        self.assertAlmostEqual(at3["a_fpr"], 0.5)
        self.assertAlmostEqual(at3["b_fpr"], 1.0)
        self.assertAlmostEqual(at3["or_fpr"], 1.0)
        self.assertAlmostEqual(at3["fp_jaccard"], 0.5)

        grasp = next(
            row
            for row in by_failure
            if row["horizon"] == "3"
            and row["failure_type"] == "grasp_failure"
        )
        self.assertEqual(grasp["event_n"], 2)
        self.assertEqual(grasp["overlap_n"], 1)
        self.assertEqual(grasp["a_only_n"], 0)
        self.assertEqual(grasp["b_only_n"], 1)
        self.assertAlmostEqual(grasp["or_recall"], 1.0)

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
