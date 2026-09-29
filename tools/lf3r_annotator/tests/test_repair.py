from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend_core import ValidationError
from repair.adapters import A2WorldAdapter
from repair.ctrl_world import CtrlWorldAdapter
from repair.alignment import capability_summary, compute_alignment, project_path
from repair.prepare_libero_manifest import _ensure_project_libero_on_sys_path


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

    def test_project_relative_official_libero_path_resolves_under_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            relative = (
                "datasets/libero_official/libero_10/"
                "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_"
                "tomato_sauce_in_the_basket_demo.hdf5"
            )
            expected = root / relative
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"hdf5")
            resolved = project_path(
                root,
                relative,
                label="trajectory field source_hdf5_path",
            )
            self.assertEqual(resolved, expected.resolve())
            self.assertTrue(resolved.is_relative_to(root))

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

    def test_generation_parameters_are_resolved_into_validation_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root, duplicate=False)
            config.update(
                {
                    "num_sampling_steps": 41,
                    "guidance": 0.7,
                    "seed": 13,
                    "history": False,
                }
            )
            adapter = A2WorldAdapter(root, config)
            status = adapter.validate_rollout(
                {
                    "camera_video_paths": {
                        "cam_high": "high.mp4",
                        "cam_wrist": "wrist.mp4",
                    }
                }
            )
            self.assertEqual(status["num_sampling_steps"], 41)
            self.assertAlmostEqual(status["guidance"], 0.7)
            self.assertEqual(status["seed"], 13)
            self.assertFalse(status["history"])

    def test_invalid_generation_parameters_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root, duplicate=False)
            config["num_sampling_steps"] = 0
            adapter = A2WorldAdapter(root, config)
            with self.assertRaises(ValidationError):
                adapter.validate_rollout(
                    {
                        "camera_video_paths": {
                            "cam_high": "high.mp4",
                            "cam_wrist": "wrist.mp4",
                        }
                    }
                )


class CtrlWorldAdapterTest(unittest.TestCase):
    def _config(self, root: Path) -> dict:
        source = root / "repos" / "Ctrl-World"
        (source / "models").mkdir(parents=True, exist_ok=True)
        (source / "models" / "ctrl_world.py").write_text(
            "# test stub\n",
            encoding="utf-8",
        )
        for name in [
            "pipeline_ctrl_world.py",
            "pipeline_stable_video_diffusion.py",
            "unet_spatio_temporal_condition.py",
        ]:
            (source / "models" / name).write_text(
                "# test stub\n",
                encoding="utf-8",
            )
        (source / "config.py").write_text("# test stub\n", encoding="utf-8")
        python = root / "conda_envs" / "LF3R-ctrl-world" / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"python")
        checkpoint = (
            root
            / "checkpoints"
            / "ctrl_world"
            / "Ctrl-World"
            / "checkpoint-10000.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        (root / "checkpoints" / "stable-video-diffusion-img2vid").mkdir(
            parents=True,
            exist_ok=True,
        )
        (root / "checkpoints" / "clip-vit-base-patch32").mkdir(
            parents=True,
            exist_ok=True,
        )
        stats = source / "dataset_meta_info" / "droid" / "stat.json"
        stats.parent.mkdir(parents=True, exist_ok=True)
        stats.write_text(
            '{"state_01":[0,0,0,0,0,0,0],"state_99":[1,1,1,1,1,1,1]}\n',
            encoding="utf-8",
        )
        trajectory = root / "official.hdf5"
        trajectory.write_bytes(b"hdf5")
        return {
            "source_root": "repos/Ctrl-World",
            "checkpoint": (
                "checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt"
            ),
            "svd_model_path": "checkpoints/stable-video-diffusion-img2vid",
            "clip_model_path": "checkpoints/clip-vit-base-patch32",
            "data_stat_path": "repos/Ctrl-World/dataset_meta_info/droid/stat.json",
        }

    def test_ctrl_world_uses_adapter_local_third_view_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = CtrlWorldAdapter(root, self._config(root))
            status = adapter.validate_rollout(
                {
                    "camera_video_paths": {
                        "cam_high": "high.mp4",
                        "cam_wrist": "wrist.mp4",
                    },
                    "source_hdf5_path": "official.hdf5",
                    "fps": 20.0,
                }
            )
            self.assertTrue(status["available"])
            self.assertEqual(
                status["camera_mapping"],
                {
                    "exterior_1": "cam_high",
                    "exterior_2": "cam_high",
                    "wrist": "cam_wrist",
                },
            )
            self.assertEqual(len(status["duplicated_camera"]), 1)
            self.assertEqual(status["source_frame_step"], 4)
            self.assertEqual(status["effective_fps"], 5.0)
            self.assertIn("absolute Cartesian pose", status["control_semantics"])

    def test_ctrl_world_requires_real_wrist_view(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = CtrlWorldAdapter(root, self._config(root))
            with self.assertRaises(ValidationError):
                adapter.validate_rollout(
                    {
                        "camera_video_paths": {"cam_high": "high.mp4"},
                        "source_hdf5_path": "official.hdf5",
                        "fps": 20.0,
                    }
                )

    def test_ctrl_pose_adapter_documents_libero_to_droid_conversion(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "repair"
            / "trajectory.py"
        ).read_text(encoding="utf-8")
        self.assertIn('Rotation.from_rotvec(rotvec).as_euler("xyz")', source)
        self.assertIn("gripper_qpos[0] - gripper_qpos[1]", source)
        self.assertIn("1.0 - opening_width / 0.08", source)
        self.assertIn("def replay_ctrl_world_pose_controls(", source)
        self.assertIn("replay GT actions[c+1:]", source)


class OfficialLiberoManifestContractTest(unittest.TestCase):
    def test_importer_auto_exposes_project_local_libero_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            libero_root = root / "repos" / "LIBERO"
            libero_root.mkdir(parents=True)
            original = list(sys.path)
            try:
                resolved = _ensure_project_libero_on_sys_path(root)
                self.assertEqual(resolved, libero_root.resolve())
                self.assertEqual(sys.path[0], str(libero_root.resolve()))
            finally:
                sys.path[:] = original

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
        repair_page = (static_root / "repair" / "page.js").read_text(
            encoding="utf-8"
        )
        ctrl_adapter = (
            Path(__file__).resolve().parents[1] / "repair" / "ctrl_world.py"
        ).read_text(encoding="utf-8")
        worker = (
            Path(__file__).resolve().parents[1] / "repair" / "worker.py"
        ).read_text(encoding="utf-8")
        service = (
            Path(__file__).resolve().parents[1] / "repair" / "service.py"
        ).read_text(encoding="utf-8")
        alignment_runner = (
            Path(__file__).resolve().parents[1] / "repair" / "alignment_runner.py"
        ).read_text(encoding="utf-8")
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
        self.assertIn('id=\\"repairWorldModel\\"', repair_page)
        self.assertIn('value=\\"ctrl_world\\"', repair_page)
        self.assertIn("Ctrl-World", repair_page)
        self.assertIn('name: "ctrl_world"', repair_js)
        self.assertIn("CtrlWorldAdapter", ctrl_adapter)
        self.assertIn('"ctrl" in env_dir.name.lower()', ctrl_adapter)
        self.assertIn("future_recorded_proprio_used", ctrl_adapter)
        self.assertIn('model_name in {"ctrl", "ctrl_world"}', worker)
        self.assertIn("replay_ctrl_world_pose_controls", worker)
        self.assertIn('"future_recorded_proprio_used": False', worker)
        self.assertNotIn("ensure_gpu_below_threshold", worker)
        self.assertNotIn("below the 50% utilization threshold", service)
        self.assertNotIn("threshold_percent", service)
        self.assertIn('"utilization_gate": False', service)
        self.assertIn('"selection_policy": "user_selected_device_no_hard_threshold"', service)
        self.assertNotIn("least_utilized_reported_device", service)
        self.assertIn('id=\\"repairGpu\\"', repair_page)
        self.assertIn('value=\\"\\" placeholder=\\"e.g. 0\\"', repair_page)
        self.assertNotIn('id=\\"repairGpu\\" type=\\"text\\" inputmode=\\"numeric\\" value=\\"0\\"', repair_page)
        self.assertIn('gpu_index: Number(gpu)', repair_js)
        self.assertIn("user selected · no utilization gate", repair_js)
        self.assertIn("Select a GPU before validating or running Repair", service)
        self.assertIn("resolve_repair_python", service)
        self.assertIn('if model_name == "ctrl_world":', service)
        self.assertIn('runtime_label="Ctrl-World"', service)
        self.assertIn("preferred_python=python", service)
        self.assertIn("python_override=", service)
        self.assertIn('plan["worker_python"]', service)
        self.assertIn('"numpy"', alignment_runner)
        self.assertIn('"h5py"', alignment_runner)
        self.assertIn('"libero.libero"', alignment_runner)
        self.assertIn("LF3R_ENV_REPAIR", alignment_runner)
        self.assertIn("No project-local Repair/LIBERO Python", alignment_runner)
        self.assertIn("Repair runtime candidate:", repair_js)
        self.assertIn("Runtime preflight error:", repair_js)
        self.assertIn('"runtime_candidate": runtime_candidate', service)
        self.assertIn('"runtime_error": runtime_error', service)
        self.assertNotIn("User-managed CUDA device index, matching Runs.", repair_page)
        for control_id in [
            "repairSamplingSteps",
            "repairGuidance",
            "repairSeed",
            "repairHistory",
            "repairGpu",
            "repairCtrlCheckpoint",
            "repairCtrlSourceRoot",
            "repairCtrlPython",
            "repairCtrlSvd",
            "repairCtrlClip",
            "repairCtrlDataStat",
            "repairCtrlTargetFps",
            "repairCtrlInferenceSteps",
            "repairCtrlGuidance",
            "repairCtrlSeed",
            "repairCtrlTextConditioning",
        ]:
            self.assertIn(control_id, repair_page)
            self.assertIn(control_id, repair_js)
        self.assertIn("num_sampling_steps", repair_js)
        self.assertIn("/api/repair/synthetic-suffix/", repair_js)


if __name__ == "__main__":
    unittest.main()
