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
        return {
            "checkpoint_type": "libero_adapted",
            "checkpoint": "checkpoints/a2world-libero.pt",
            "base_checkpoints": "checkpoints",
            "duplicate_missing_views": duplicate,
            "command": "true",
        }

    def test_missing_wrist_is_not_silently_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = A2WorldAdapter(root, self._config(root, duplicate=False))
            with self.assertRaises(ValidationError):
                adapter.validate_rollout(
                    {"camera_video_paths": {"cam_high": "high.mp4"}}
                )

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
        self.assertIn("/api/repair/synthetic-suffix/", repair_js)


if __name__ == "__main__":
    unittest.main()
