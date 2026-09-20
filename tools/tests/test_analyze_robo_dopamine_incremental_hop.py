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
    evaluate_failure_rollout_from_start,
    make_config,
    positive_episode_count,
    recovery_metrics,
)
from robo_incremental_hop.report import build_pairwise_ensemble_rows


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

    def test_no_event_failure_uses_rollout_start_latency(self) -> None:
        detected = evaluate_failure_rollout_from_start(
            [4, 8, 12],
            [False, True, True],
        )
        self.assertTrue(detected["eventual_recall"])
        self.assertFalse(detected["recall_at_1"])
        self.assertTrue(detected["recall_at_3"])
        self.assertEqual(detected["delay_samples"], 2)
        self.assertEqual(detected["first_alarm_frame"], 8)
        self.assertEqual(detected["delay_frames"], 8)

        missed = evaluate_failure_rollout_from_start(
            [4, 8, 12],
            [False, False, False],
        )
        self.assertFalse(missed["eventual_recall"])
        self.assertFalse(missed["recall_at_20"])
        self.assertIsNone(missed["first_alarm_frame"])

    def test_joint_pairwise_or_sweep_uses_true_fp_union(self) -> None:
        configs = [
            make_config(
                "a1",
                "stagnation_consecutive",
                delta=0.02,
                n=2,
            ),
            make_config(
                "a2",
                "stagnation_consecutive",
                delta=0.05,
                n=3,
            ),
            make_config(
                "b1",
                "window_mean",
                m=3,
                theta=-0.02,
            ),
            make_config(
                "b2",
                "window_mean",
                m=5,
                theta=0.0,
            ),
        ]

        def detection_row(
            config_id: str,
            row_id: str,
            detected: bool,
            delay: int | None,
            *,
            event: bool,
            failure_type: str = "other",
        ) -> dict:
            base = {
                "config_id": config_id,
                "rollout_id": row_id.split("::")[0],
                "recall_at_1": bool(
                    detected and delay is not None and delay <= 1
                ),
                "recall_at_3": bool(
                    detected and delay is not None and delay <= 3
                ),
                "recall_at_5": bool(
                    detected and delay is not None and delay <= 5
                ),
                "recall_at_10": bool(
                    detected and delay is not None and delay <= 10
                ),
                "recall_at_20": bool(
                    detected and delay is not None and delay <= 20
                ),
                "eventual_recall": detected,
                "detected": detected,
                "delay_samples": delay,
                "delay_frames": None if delay is None else delay * 4,
                "outcome": "terminal_failure",
            }
            if event:
                base.update(
                    {
                        "event_id": row_id,
                        "event_index": int(
                            row_id.rsplit("event", 1)[1]
                        ),
                        "failure_type": failure_type,
                    }
                )
            return base

        event_pattern = {
            "a1": (True, False),
            "a2": (False, True),
            "b1": (True, True),
            "b2": (False, False),
        }
        event_rows = []
        for config_id, (first, second) in event_pattern.items():
            event_rows.extend(
                [
                    detection_row(
                        config_id,
                        "r0::event0",
                        first,
                        1 if first else None,
                        event=True,
                        failure_type="grasp_failure",
                    ),
                    detection_row(
                        config_id,
                        "r1::event0",
                        second,
                        2 if second else None,
                        event=True,
                        failure_type="timeout_no_progress",
                    ),
                ]
            )

        no_event_pattern = {
            "a1": False,
            "a2": True,
            "b1": False,
            "b2": True,
        }
        no_event_rows = [
            detection_row(
                config_id,
                "r2",
                detected,
                3 if detected else None,
                event=False,
            )
            for config_id, detected in no_event_pattern.items()
        ]

        clean_ids = ["c0", "c1", "c2", "c3", "c4"]
        false_positive = {
            "a1": {"c0"},
            "a2": {"c1"},
            "b1": set(),
            "b2": set(),
        }
        clean_rows = [
            {
                "config_id": config_id,
                "rollout_id": rollout_id,
                "any_positive": rollout_id in false_positive[config_id],
            }
            for config_id in false_positive
            for rollout_id in clean_ids
        ]

        sweep, selected, by_failure = build_pairwise_ensemble_rows(
            configs,
            event_rows,
            no_event_rows,
            clean_rows,
        )
        self.assertEqual(len(sweep), 4)

        combo = next(
            row
            for row in sweep
            if row["a_config_id"] == "a2"
            and row["b_config_id"] == "b1"
        )
        self.assertAlmostEqual(combo["clean_rollout_fpr"], 0.2)
        self.assertEqual(combo["fp_overlap_n"], 0)
        self.assertAlmostEqual(combo["event_recall_eventual"], 1.0)
        self.assertAlmostEqual(combo["no_event_recall_eventual"], 1.0)
        self.assertAlmostEqual(
            combo["overall_failed_rollout_coverage"],
            1.0,
        )
        self.assertEqual(combo["event_overlap_at_eventual_n"], 1)
        self.assertEqual(combo["event_a_only_at_eventual_n"], 0)
        self.assertEqual(combo["event_b_only_at_eventual_n"], 1)
        self.assertEqual(combo["no_event_a_only_at_eventual_n"], 1)

        chosen = next(
            row
            for row in selected
            if row["selection_status"] == "selected"
            and row["clean_fpr_constraint"] == 0.20
            and row["selection_target"]
            == "overall_failed_rollout_coverage"
            and row["detector_a_family"]
            == "stagnation_consecutive"
            and row["detector_b_family"] == "window_mean"
        )
        self.assertEqual(chosen["a_config_id"], "a2")
        self.assertEqual(chosen["b_config_id"], "b1")
        self.assertAlmostEqual(chosen["selection_value"], 1.0)
        self.assertTrue(
            any(
                row["failure_type"] == "timeout_no_progress"
                and row["horizon"] == "eventual"
                and row["b_only_n"] == 0
                for row in by_failure
            )
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
