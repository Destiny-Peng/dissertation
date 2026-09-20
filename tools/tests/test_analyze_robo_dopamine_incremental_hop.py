#!/usr/bin/env python3
"""Synthetic tests for Robo-Dopamine incremental-hop detector logic."""

from __future__ import annotations

import sys
import tempfile
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
from robo_incremental_hop.diagnosis import (
    build_grasp_event_features,
    build_matched_clean_pairs,
    category_summary,
    choose_reference_ensemble,
    summarize_detected_vs_missed,
    summarize_matched_controls,
)
from robo_incremental_hop.phenotypes import build_phenotype_detector_configs
from robo_incremental_hop.oracle import build_oracle_analysis
from robo_incremental_hop.progress_peak import (
    evaluate_progress_peak_localization,
    summarize_progress_peak_localization,
)
from robo_incremental_hop.localization_ranking import (
    rank_existing_sweep_interval_localization,
    rank_existing_sweep_localization,
)
from robo_incremental_hop.report import build_pairwise_ensemble_rows
import robo_incremental_hop.search_cache as search_cache


class IncrementalHopDetectorTests(unittest.TestCase):
    def test_interval_localization_uses_causal_observable_interval(self) -> None:
        signals = {
            "r0": {"frames": [0, 1, 2, 3, 4]},
            "r1": {"frames": [0, 1, 2, 3, 4]},
        }
        base = {
            "detector_family": "consecutive",
            "epsilon": 0.0,
            "n": 1,
            "parameters_json": '{"epsilon":0.0,"n":1}',
            "outcome": "terminal_failure",
            "failure_type": "grasp_failure",
        }
        event_rows = [
            {
                **base, "config_id": "c", "rollout_id": "r0",
                "event_id": "r0::event0", "event_index": 0,
                "causal_onset_frame": 1, "observable_onset_frame": 3,
                "earliest_early_positive_frame": 2, "detection_frame": 3,
            },
            {
                **base, "config_id": "c", "rollout_id": "r1",
                "event_id": "r1::event0", "event_index": 0,
                "causal_onset_frame": 2, "observable_onset_frame": 3,
                "earliest_early_positive_frame": 0, "detection_frame": 3,
            },
        ]
        ranking = rank_existing_sweep_interval_localization(
            event_rows=event_rows,
            ensemble_sweep=[],
            signals=signals,
        )
        row = next(
            item for item in ranking
            if item["population"] == "all_eligible_failure_events"
        )
        self.assertEqual(row["eligible_event_n"], 2)
        self.assertAlmostEqual(row["in_interval_rate"], 0.5)
        self.assertAlmostEqual(row["within_1"], 0.5)
        self.assertAlmostEqual(row["within_3"], 1.0)
        self.assertAlmostEqual(row["before_interval_rate"], 0.5)
        self.assertAlmostEqual(row["after_interval_rate"], 0.0)
        self.assertAlmostEqual(row["median_signed_interval_error_samples"], -1.0)
        self.assertAlmostEqual(row["median_absolute_interval_error_samples"], 1.0)
        self.assertAlmostEqual(row["mae_samples"], 1.0)
        self.assertAlmostEqual(row["mse_samples"], 2.0)

    def test_existing_sweep_ranking_uses_first_trigger_without_new_search(self) -> None:
        signals = {
            "r1": {"frames": [0, 1, 2, 3, 4, 5]},
            "r2": {"frames": [0, 1, 2, 3, 4, 5]},
        }
        base = {
            "detector_family": "consecutive",
            "epsilon": 0.0,
            "n": 1,
            "m": None,
            "k": None,
            "theta": None,
            "A": None,
            "delta": None,
            "parameters_json": '{"epsilon":0.0,"n":1}',
            "outcome": "terminal_failure",
            "failure_type": "grasp_failure",
        }
        event_rows = [
            {
                **base,
                "config_id": "a",
                "rollout_id": "r1",
                "event_id": "r1::event0",
                "event_index": 0,
                "observable_onset_frame": 3,
                "earliest_early_positive_frame": 0,
                "detection_frame": 3,
            },
            {
                **base,
                "config_id": "a",
                "rollout_id": "r2",
                "event_id": "r2::event0",
                "event_index": 0,
                "observable_onset_frame": 2,
                "earliest_early_positive_frame": None,
                "detection_frame": 2,
            },
            {
                **base,
                "config_id": "b",
                "rollout_id": "r1",
                "event_id": "r1::event0",
                "event_index": 0,
                "observable_onset_frame": 3,
                "earliest_early_positive_frame": None,
                "detection_frame": 3,
            },
            {
                **base,
                "config_id": "b",
                "rollout_id": "r2",
                "event_id": "r2::event0",
                "event_index": 0,
                "observable_onset_frame": 2,
                "earliest_early_positive_frame": None,
                "detection_frame": 3,
            },
        ]
        ensemble_sweep = [
            {"a_config_id": "a", "b_config_id": "b"}
        ]
        ranking = rank_existing_sweep_localization(
            event_rows=event_rows,
            ensemble_sweep=ensemble_sweep,
            signals=signals,
        )
        first_population = [
            row
            for row in ranking
            if row["population"] == "first_event_per_failed_rollout"
        ]
        by_id = {row["config_id"]: row for row in first_population}
        self.assertEqual(by_id["a"]["median_signed_offset_samples"], -1.5)
        self.assertEqual(by_id["b"]["median_signed_offset_samples"], 0.5)
        self.assertEqual(by_id["b"]["rank_rmse"], 1)
        self.assertAlmostEqual(by_id["b"]["rmse_samples"], 2 ** -0.5)
        self.assertAlmostEqual(by_id["b"]["within_1"], 1.0)
        self.assertEqual(
            by_id["OR:a|b"]["median_signed_offset_samples"],
            -1.5,
        )
        self.assertAlmostEqual(by_id["OR:a|b"]["trigger_coverage"], 1.0)

    def test_progress_peak_uses_earliest_argmax_and_first_failure_onset(self) -> None:
        signals = {
            "fail_a": {
                "frames": [0, 4, 8, 12],
                "progress": [0.1, 0.8, 0.8, 0.2],
                "prediction_path": Path("/tmp/fail_a.json"),
            },
            "fail_b": {
                "frames": [0, 4, 8, 12],
                "progress": [0.1, 0.2, 0.3, 0.9],
                "prediction_path": Path("/tmp/fail_b.json"),
            },
            "recovered": {
                "frames": [0, 4, 8],
                "progress": [0.1, 0.9, 0.2],
                "prediction_path": Path("/tmp/recovered.json"),
            },
        }
        events = [
            {
                "event_id": "fail_a::event0",
                "rollout_id": "fail_a",
                "event_index": 0,
                "failure_type": "grasp_failure",
                "observable_onset_frame": 8,
                "outcome": "terminal_failure",
                "task_key": "libero_10:0",
            },
            {
                "event_id": "fail_a::event1",
                "rollout_id": "fail_a",
                "event_index": 1,
                "failure_type": "timeout_no_progress",
                "observable_onset_frame": 12,
                "outcome": "terminal_failure",
                "task_key": "libero_10:0",
            },
            {
                "event_id": "fail_b::event0",
                "rollout_id": "fail_b",
                "event_index": 0,
                "failure_type": "grasp_failure",
                "observable_onset_frame": 4,
                "outcome": "terminal_failure",
                "task_key": "libero_10:1",
            },
            {
                "event_id": "recovered::event0",
                "rollout_id": "recovered",
                "event_index": 0,
                "failure_type": "grasp_failure",
                "observable_onset_frame": 4,
                "outcome": "recovered_success",
                "task_key": "libero_10:2",
            },
        ]
        rows = evaluate_progress_peak_localization(signals, events)
        self.assertEqual(len(rows), 2)
        by_id = {row["rollout_id"]: row for row in rows}
        self.assertEqual(by_id["fail_a"]["annotated_event_n"], 2)
        self.assertEqual(by_id["fail_a"]["first_observable_onset_frame"], 8)
        self.assertEqual(by_id["fail_a"]["t_star_frame"], 4)
        self.assertEqual(by_id["fail_a"]["t_star_minus_onset_samples"], -1)
        self.assertEqual(by_id["fail_a"]["t_star_minus_onset_frames"], -4)
        self.assertTrue(by_id["fail_a"]["within_1_samples"])
        self.assertEqual(by_id["fail_b"]["t_star_frame"], 12)
        self.assertEqual(by_id["fail_b"]["t_star_minus_onset_samples"], 2)

        summary = summarize_progress_peak_localization(rows)
        overall = next(
            row
            for row in summary
            if row["group"] == "overall" and row["value"] == "all"
        )
        self.assertEqual(overall["rollout_n"], 2)
        self.assertAlmostEqual(overall["within_1_samples_fraction"], 0.5)
        self.assertAlmostEqual(overall["within_3_samples_fraction"], 1.0)
        self.assertAlmostEqual(overall["before_onset_fraction"], 0.5)
        self.assertAlmostEqual(overall["after_onset_fraction"], 0.5)

    def test_search_cache_fingerprint_and_roundtrip(self) -> None:
        signals = {
            "r0": {
                "frames": [0, 4, 8],
                "hops": [0.1, -0.2, 0.0],
            }
        }
        events = [
            {
                "event_id": "r0::event0",
                "rollout_id": "r0",
                "event_index": 0,
                "failure_type": "grasp_failure",
                "observable_onset_frame": 4,
            }
        ]
        fingerprint = search_cache.search_fingerprint(
            signals,
            events,
            [],
            [],
        )
        self.assertEqual(
            fingerprint,
            search_cache.search_fingerprint(signals, events, [], []),
        )
        changed = {
            "r0": {
                "frames": [0, 4, 8],
                "hops": [0.1, -0.21, 0.0],
            }
        }
        self.assertNotEqual(
            fingerprint,
            search_cache.search_fingerprint(changed, events, [], []),
        )

        original_root = search_cache.SEARCH_CACHE_ROOT
        with tempfile.TemporaryDirectory() as temporary:
            search_cache.SEARCH_CACHE_ROOT = Path(temporary)
            try:
                search_cache.write_search_cache(
                    fingerprint,
                    configs=[make_config("c0", "regression_window_min", m=1, theta=-0.2)],
                    oracle_configs=[],
                    phenotype_grid={"example": True},
                    summary_rows=[{"config_id": "c0", "value": 1.0}],
                    event_rows=[{"config_id": "c0", "detected": True}],
                    no_event_rows=[],
                    clean_rows=[],
                    ensemble_sweep=[{"pair_priority": 1}],
                    oracle_global_best=[],
                    oracle_event_detectability=[],
                    oracle_summary=[],
                )
                cached = search_cache.load_search_cache(fingerprint)
                self.assertIsNotNone(cached)
                self.assertEqual(cached["summary_rows"][0]["value"], 1.0)
                self.assertTrue(cached["event_rows"][0]["detected"])
                self.assertEqual(cached["ensemble_sweep"][0]["pair_priority"], 1)
            finally:
                search_cache.SEARCH_CACHE_ROOT = original_root

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

    def test_empirical_regression_grid_comes_from_saved_hop_values(self) -> None:
        signals = {
            "failure": {
                "frames": [0, 4, 8, 12],
                "hops": [0.1, -0.30, 0.05, -0.10],
            },
            "clean0": {"frames": [0, 4, 8, 12], "hops": [0.2, 0.1, 0.1, 0.1]},
            "clean1": {"frames": [0, 4, 8, 12], "hops": [0.2, 0.1, 0.1, 0.1]},
            "clean2": {"frames": [0, 4, 8, 12], "hops": [0.2, 0.1, 0.1, 0.1]},
            "clean3": {"frames": [0, 4, 8, 12], "hops": [0.2, 0.1, 0.1, 0.1]},
            "clean4": {"frames": [0, 4, 8, 12], "hops": [0.2, -0.25, 0.1, 0.1]},
        }
        events = [
            {
                "rollout_id": "failure",
                "event_index": 0,
                "failure_type": "grasp_failure",
                "observable_onset_frame": 0,
                "episode_end_frame": None,
            }
        ]
        clean = [
            {"rollout_id": f"clean{index}"}
            for index in range(5)
        ]
        configs, oracle_configs, metadata = build_phenotype_detector_configs(
            signals,
            events,
            [],
            clean,
        )
        regression = [
            config
            for config in configs
            if config["detector_family"] == "regression_window_min"
        ]
        self.assertTrue(regression)
        self.assertTrue(all(float(config["theta"]) < 0.0 for config in regression))
        oracle_regression = [
            config
            for config in oracle_configs
            if config["detector_family"] == "regression_window_min"
        ]
        self.assertGreaterEqual(len(oracle_regression), len(regression))
        self.assertTrue(
            any(abs(float(config["theta"]) + 0.30) < 1e-12 for config in regression)
        )
        self.assertEqual(
            metadata["regression_family"],
            "regression_window_min",
        )
        self.assertIn("oracle_grid", metadata)
        self.assertIn("operational_prefilter", metadata)

    def test_oracle_rejects_long_pre_onset_positive_episode(self) -> None:
        frames = [0, 4, 8, 12, 16]
        signals = {
            "tolerated": {
                "frames": frames,
                "hops": [0.2, 0.2, -0.3, -0.2, 0.1],
            },
            "long_early": {
                "frames": frames,
                "hops": [-0.3, -0.3, -0.3, -0.3, -0.3],
            },
            "persistent": {
                "frames": frames,
                "hops": [-0.3, 0.1, 0.1, 0.1, 0.1],
            },
        }
        events = [
            {
                "event_id": "tolerated::event0",
                "rollout_id": "tolerated",
                "event_index": 0,
                "failure_type": "grasp_failure",
                "outcome": "terminal_failure",
                "observable_onset_frame": 12,
                "episode_end_frame": None,
            },
            {
                "event_id": "long_early::event0",
                "rollout_id": "long_early",
                "event_index": 0,
                "failure_type": "grasp_failure",
                "outcome": "terminal_failure",
                "observable_onset_frame": 12,
                "episode_end_frame": None,
            },
        ]
        no_event = [
            {
                "rollout_id": "persistent",
                "outcome": "terminal_failure",
            }
        ]
        configs = [
            make_config(
                "st",
                "stagnation_consecutive",
                delta=1.0,
                n=1,
            ),
            make_config(
                "reg",
                "regression_window_min",
                m=1,
                theta=-0.1,
            ),
        ]
        global_best, detectability, summary = build_oracle_analysis(
            configs,
            signals,
            events,
            no_event,
            [],
            early_tolerance_samples=1,
        )

        regression_events = {
            row["event_id"]: row
            for row in detectability
            if row["signal_family"] == "regression"
            and row["record_kind"] == "annotated_event"
        }
        self.assertTrue(regression_events["tolerated::event0"]["oracle_detectable"])
        self.assertFalse(
            regression_events["tolerated::event0"]["oracle_detectable_strict"]
        )
        self.assertEqual(
            regression_events["tolerated::event0"]["best_start_offset_samples"],
            -1,
        )
        self.assertEqual(
            regression_events["tolerated::event0"]["best_delay_samples"],
            0,
        )
        self.assertFalse(
            regression_events["long_early::event0"]["oracle_detectable"]
        )

        grasp_regression = next(
            row
            for row in summary
            if row["signal_family"] == "regression"
            and row["population"] == "grasp_failure"
        )
        self.assertAlmostEqual(
            grasp_regression["oracle_recall_eventual"],
            0.5,
        )
        self.assertAlmostEqual(
            grasp_regression["strict_recall_eventual"],
            0.0,
        )
        persistent_regression = next(
            row
            for row in summary
            if row["signal_family"] == "regression"
            and row["population"] == "no_event_failure_rollouts"
        )
        self.assertAlmostEqual(
            persistent_regression["oracle_recall_at_1"],
            1.0,
        )
        self.assertTrue(
            any(
                row["signal_family"] == "combined"
                and row["population"] == "grasp_failure"
                and row["selection_target"] == "eventual"
                for row in global_best
            )
        )

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
                "regression_window_min",
                m=3,
                theta=-0.02,
            ),
            make_config(
                "b2",
                "regression_window_min",
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
                        failure_type="grasp_failure",
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
        self.assertAlmostEqual(combo["grasp_recall_eventual"], 1.0)
        self.assertAlmostEqual(combo["grasp_recall_at_10"], 1.0)
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
            == "grasp_recall_eventual"
            and row["detector_a_family"]
            == "stagnation_consecutive"
            and row["detector_b_family"] == "regression_window_min"
        )
        self.assertEqual(chosen["a_config_id"], "a2")
        self.assertEqual(chosen["b_config_id"], "b1")
        self.assertAlmostEqual(chosen["selection_value"], 1.0)
        self.assertTrue(
            any(
                row["failure_type"] == "grasp_failure"
                and row["horizon"] == "eventual"
                and row["or_recall"] == 1.0
                for row in by_failure
            )
        )

    def test_grasp_failure_diagnosis_taxonomy_and_matched_controls(self) -> None:
        frames = [0, 4, 8, 12, 16, 20, 24, 28]
        signals = {
            "immediate": {
                "frames": frames,
                "hops": [0.2, 0.1, -0.2, -0.1, 0.1, 0.1, 0.1, 0.1],
            },
            "delayed": {
                "frames": frames,
                "hops": [0.2, 0.1, 0.2, 0.2, 0.2, -0.2, -0.1, 0.1],
            },
            "stagnation": {
                "frames": frames,
                "hops": [0.2, 0.1, 0.005, 0.004, 0.003, 0.002, 0.1, 0.1],
            },
            "clear": {
                "frames": frames,
                "hops": [0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            },
            "clean": {
                "frames": frames,
                "hops": [0.2, 0.1, 0.15, 0.12, 0.1, 0.1, 0.1, 0.1],
            },
        }
        events = [
            {
                "event_id": rollout_id + "::event0",
                "rollout_id": rollout_id,
                "event_index": 0,
                "task_key": "libero_10:0",
                "task_suite": "libero_10",
                "task_id": 0,
                "outcome": "terminal_failure",
                "failure_type": "grasp_failure",
                "observable_onset_frame": 8,
                "episode_end_frame": None,
                "episode_end_source": "rollout_end",
            }
            for rollout_id in ("immediate", "delayed", "stagnation", "clear")
        ]
        configs = [
            make_config(
                "a",
                "stagnation_consecutive",
                delta=0.01,
                n=2,
            ),
            make_config(
                "b",
                "regression_window_min",
                m=2,
                theta=-0.05,
            ),
        ]
        reference = {
            "selection_status": "selected",
            "selection_target": "grasp_recall_eventual",
            "clean_fpr_constraint": 0.20,
            "overall_failed_rollout_coverage": 0.75,
            "clean_rollout_fpr": 0.0,
            "pair_priority": 1,
            "a_config_id": "a",
            "b_config_id": "b",
        }
        features = build_grasp_event_features(
            signals,
            events,
            configs,
            reference,
        )
        by_id = {row["rollout_id"]: row for row in features}
        self.assertEqual(
            by_id["immediate"]["failure_mode_category"],
            "immediate_regression",
        )
        self.assertEqual(
            by_id["delayed"]["failure_mode_category"],
            "delayed_regression",
        )
        self.assertEqual(
            by_id["stagnation"]["failure_mode_category"],
            "stagnation",
        )
        self.assertEqual(
            by_id["clear"]["failure_mode_category"],
            "no_clear_hop_response",
        )
        self.assertTrue(by_id["immediate"]["oracle_visible_abnormality"])
        self.assertTrue(by_id["stagnation"]["oracle_visible_abnormality"])
        self.assertFalse(by_id["clear"]["oracle_visible_abnormality"])

        detected_summary = summarize_detected_vs_missed(features)
        self.assertTrue(
            any(row["feature"] == "negative_fraction_20" for row in detected_summary)
        )

        matched = build_matched_clean_pairs(
            features,
            signals,
            [
                {
                    "rollout_id": "clean",
                    "task_key": "libero_10:0",
                }
            ],
        )
        self.assertEqual(len(matched), 4)
        self.assertTrue(
            all(row["control_rollout_id"] == "clean" for row in matched)
        )
        matched_summary = summarize_matched_controls(matched)
        self.assertTrue(
            any(row["feature"] == "min_hop_20" for row in matched_summary)
        )
        categories = category_summary(features)
        self.assertEqual(
            sum(
                row["event_n"]
                for row in categories
                if not str(row["failure_mode_category"]).startswith("__")
            ),
            4,
        )

    def test_reference_ensemble_prefers_highest_cap_then_best_grasp_recall(self) -> None:
        selected = [
            {
                "selection_status": "selected",
                "selection_target": "grasp_recall_eventual",
                "clean_fpr_constraint": 0.10,
                "grasp_recall_eventual": 0.7,
                "grasp_recall_at_10": 0.7,
                "overall_failed_rollout_coverage": 1.0,
                "clean_rollout_fpr": 0.1,
                "a_config_id": "low",
                "b_config_id": "cap",
            },
            {
                "selection_status": "selected",
                "selection_target": "grasp_recall_eventual",
                "clean_fpr_constraint": 0.20,
                "grasp_recall_eventual": 0.8,
                "grasp_recall_at_10": 0.7,
                "overall_failed_rollout_coverage": 0.8,
                "clean_rollout_fpr": 0.15,
                "pair_priority": 2,
                "a_config_id": "worse",
                "b_config_id": "coverage",
            },
            {
                "selection_status": "selected",
                "selection_target": "grasp_recall_eventual",
                "clean_fpr_constraint": 0.20,
                "grasp_recall_eventual": 0.9,
                "grasp_recall_at_10": 0.8,
                "overall_failed_rollout_coverage": 0.9,
                "clean_rollout_fpr": 0.18,
                "pair_priority": 3,
                "a_config_id": "reference",
                "b_config_id": "winner",
            },
        ]
        reference = choose_reference_ensemble(selected)
        self.assertIsNotNone(reference)
        self.assertEqual(reference["a_config_id"], "reference")
        self.assertEqual(reference["b_config_id"], "winner")

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
