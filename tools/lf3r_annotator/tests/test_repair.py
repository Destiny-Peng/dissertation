from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend_core import ValidationError
from repair.adapters import A2WorldAdapter
from repair.alignment import capability_summary, compute_alignment


class RepairAlignmentTest(unittest.TestCase):
    def test_progress_cut_uses_official_libero_continuation_alignment(self) -> None:
        plan = compute_alignment(total_frames=10, cut_type="progress", cut_progress=0.5)
        self.assertEqual(plan.cut_rgb_frame, 4)
        self.assertEqual(plan.condition_frame, 4)
        self.assertEqual(plan.branch_state_index, 5)
        self.assertEqual(plan.gt_action_start, 5)

    def test_fixed_frame_cannot_consume_last_rgb_frame(self) -> None:
        with self.assertRaises(ValidationError):
            compute_alignment(total_frames=10, cut_type="frame", cut_frame=9)

    def test_manifest_capabilities_report_only_real_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "high.mp4").write_bytes(b"video")
            (root / "actions.csv").write_text("action/dx\n0\n", encoding="utf-8")
            (root / "trajectory.npz").write_bytes(b"state")
            rollout = {
                "camera_video_paths": {"cam_high": "high.mp4"},
                "csv_path": "actions.csv",
                "trajectory_path": "trajectory.npz",
            }
            summary = capability_summary(root, rollout)
            self.assertEqual(summary["views"], ["cam_high"])
            self.assertTrue(summary["actions_available"])
            self.assertTrue(summary["sim_state_available"])


class A2WorldAdapterTest(unittest.TestCase):
    def _config(self, root: Path, duplicate: bool) -> dict:
        checkpoint = root / "checkpoints" / "a2world-libero.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        source = root / "repos" / "A2World" / "world_model" / "a2world"
        source.mkdir(parents=True, exist_ok=True)
        (source / "rollout.py").write_text("# test stub\n", encoding="utf-8")
        return {
            "checkpoint_type": "libero_adapted",
            "checkpoint": "checkpoints/a2world-libero.pt",
            "base_checkpoints": "checkpoints",
            "source_root": "repos/A2World/world_model",
            "duplicate_missing_views": duplicate,
        }

    def test_missing_wrist_is_not_silently_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = A2WorldAdapter(root, self._config(root, duplicate=False))
            with self.assertRaises(ValidationError):
                adapter.validate_rollout(
                    {"camera_video_paths": {"cam_high": "high.mp4"}}
                )


    def test_libero_action_transform_matches_upstream_contract(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = A2WorldAdapter(root, self._config(root, duplicate=False))
            raw = np.asarray(
                [[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, -1.0]],
                dtype=np.float32,
            )
            prepared, adapter_name = adapter._prepare_action_array(raw)
            np.testing.assert_allclose(
                prepared[0, :7],
                np.asarray(
                    [0.05, -0.10, 0.15, -0.20, 0.25, -0.30, 1.0],
                    dtype=np.float32,
                ),
            )
            np.testing.assert_allclose(prepared[0, 7:], 0.0)
            self.assertEqual(
                adapter_name,
                "a2world.actions.libero_servo_actions",
            )

    def test_generic_checkpoint_keeps_libero_action_preprocessing(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root, duplicate=False)
            config["checkpoint_type"] = "generic_pretrained"
            config["checkpoint"] = "checkpoints/a2world-pretrained.pt"
            (root / "checkpoints" / "a2world-pretrained.pt").write_bytes(b"checkpoint")
            adapter = A2WorldAdapter(root, config)
            raw = np.asarray(
                [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            prepared, adapter_name = adapter._prepare_action_array(raw)
            self.assertEqual(
                adapter_name,
                "a2world.actions.libero_servo_actions",
            )
            self.assertAlmostEqual(float(prepared[0, 0]), 0.01, places=6)
            self.assertAlmostEqual(float(prepared[0, 6]), 0.0, places=6)

    def test_rgb_adapter_is_derived_from_alignment_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = A2WorldAdapter(root, self._config(root, duplicate=False))
            provenance = adapter.configure_alignment_rgb(
                {
                    "comparisons": {
                        "cam_high": {"orientation_transform": "rotate_180"},
                        "cam_wrist": {"orientation_transform": "horizontal_flip"},
                    }
                }
            )
            self.assertEqual(
                provenance["cam_high"]["manifest_to_a2world"],
                "vertical_flip",
            )
            self.assertEqual(
                provenance["cam_wrist"]["manifest_to_a2world"],
                "raw",
            )

    def test_final_action_chunk_is_padded_and_recorded(self) -> None:
        import json
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = A2WorldAdapter(root, self._config(root, duplicate=False))
            output = root / "prepared"
            output.mkdir()
            path = adapter.prepare_actions(
                np.zeros((21, 7), dtype=np.float32),
                output_dir=output,
            )
            with np.load(path) as data:
                self.assertEqual(data["actions"].shape, (40, 14))
            metadata = json.loads(
                (output / "future_actions_a2world.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["raw_future_action_count"], 21)
            self.assertEqual(metadata["tail_padding_count"], 19)

    def test_explicit_duplication_is_recorded_in_provenance_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = A2WorldAdapter(root, self._config(root, duplicate=True))
            status = adapter.validate_rollout(
                {"camera_video_paths": {"cam_high": "high.mp4"}}
            )
            self.assertEqual(status["camera_mapping"]["eye_in_hand"], "cam_high")
            self.assertEqual(len(status["duplicated_camera"]), 1)
            self.assertEqual(
                status["duplicated_camera"][0]["requested_manifest_view"],
                "cam_wrist",
            )


class OfficialLiberoManifestContractTest(unittest.TestCase):
    def test_importer_preserves_official_physical_views_and_c_plus_one_alignment(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "repair"
            / "prepare_libero_manifest.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"cam_high": "obs/agentview_rgb"', source)
        self.assertIn('"cam_wrist": "obs/eye_in_hand_rgb"', source)
        self.assertIn('"source_hdf5_path": source_relative', source)
        self.assertIn('"trajectory_group": group_name', source)
        self.assertIn('"branch_state": "states[c+1]"', source)
        self.assertIn('"future_actions": "actions[c+1:]"', source)
        self.assertIn('"rgb_transform": "none"', source)
        self.assertNotIn("cam_left_wrist", source)
        self.assertNotIn("cam_right_wrist", source)


class RepairFrontendContractTest(unittest.TestCase):
    def test_repair_is_top_level_and_not_an_analysis_subtab(self) -> None:
        static_root = Path(__file__).resolve().parents[1] / "static"
        html = (static_root / "index.html").read_text(encoding="utf-8")
        router = (static_root / "workspace" / "router.js").read_text(encoding="utf-8")
        repair_js = (static_root / "repair" / "synthetic-suffix.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('href="#/repair" data-route="repair"', html)
        self.assertIn('id="repairView"', html)
        self.assertNotIn('data-analysis-tab="repair"', html)
        self.assertIn('"repair"', router)
        self.assertIn("LF3RRepairSyntheticSuffix.refresh", router)
        self.assertIn("row.repair_eligible", repair_js)
        self.assertIn('" disabled"', repair_js)
        self.assertIn("Official LIBERO demonstration", repair_js)
        self.assertIn("/api/repair/synthetic-suffix/", repair_js)


if __name__ == "__main__":
    unittest.main()
