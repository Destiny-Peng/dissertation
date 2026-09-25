from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "tools" / "export_failure_cases.py"
SPEC = importlib.util.spec_from_file_location("export_failure_cases", SCRIPT)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportFailureCasesCameraTests(unittest.TestCase):
    def test_same_named_camera_files_are_kept_in_separate_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left = root / "source" / "left" / "rollout.mp4"
            right = root / "source" / "right" / "rollout.mp4"
            left.parent.mkdir(parents=True)
            right.parent.mkdir(parents=True)
            left.write_bytes(b"LEFT")
            right.write_bytes(b"RIGHT")
            (left.parent / "rollout.json").write_text("left", encoding="utf-8")
            (right.parent / "rollout.json").write_text("right", encoding="utf-8")

            annotation = root / "annotation.json"
            annotation.write_text("{}", encoding="utf-8")
            package = root / "package"
            (package / "rollouts").mkdir(parents=True)
            (package / "annotations").mkdir(parents=True)

            case = {
                "rollout_id": "r1",
                "annotation_path": annotation,
                "annotation": {
                    "outcome_label": "failure",
                    "failure_type": "grasp_failure",
                    "failure_events": [],
                },
                "record": {
                    "id": "r1",
                    "task_suite": "test",
                    "task_id": 0,
                    "episode_index": 0,
                    "dataset_role": "test",
                    "analysis_partition": "natural_observation",
                    "camera_video_paths": {
                        "cam_left_wrist": "source/left/rollout.mp4",
                        "cam_right_wrist": "source/right/rollout.mp4",
                    },
                },
                "camera_paths": {
                    "cam_left_wrist": left,
                    "cam_right_wrist": right,
                },
                "camera_files": {
                    "cam_left_wrist": [
                        left,
                        left.parent / "rollout.json",
                    ],
                    "cam_right_wrist": [
                        right,
                        right.parent / "rollout.json",
                    ],
                },
            }

            with mock.patch.object(
                exporter,
                "relative_project_path",
                side_effect=lambda value: str(Path(value)),
            ):
                result = exporter.copy_case(case, package)

            left_rel = result["camera_videos"]["cam_left_wrist"]
            right_rel = result["camera_videos"]["cam_right_wrist"]
            self.assertNotEqual(left_rel, right_rel)
            self.assertEqual((package / left_rel).read_bytes(), b"LEFT")
            self.assertEqual((package / right_rel).read_bytes(), b"RIGHT")
            self.assertIn("/cam_left_wrist/", "/" + left_rel)
            self.assertIn("/cam_right_wrist/", "/" + right_rel)
            self.assertEqual(
                result["manifest_record"]["camera_video_paths"],
                result["camera_videos"],
            )


if __name__ == "__main__":
    unittest.main()
