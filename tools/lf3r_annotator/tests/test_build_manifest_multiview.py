from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_manifest  # noqa: E402


class BuildManifestMultiviewTests(unittest.TestCase):
    def make_rollout(self, root: Path) -> tuple[Path, dict[str, Path]]:
        suite_dir = root / "outputs" / "lf3r-data-natural-test" / "libero_10"
        suite_dir.mkdir(parents=True)
        canonical = suite_dir / "task0--ep0--succ1.mp4"
        canonical.write_bytes(b"canonical")
        cameras = {
            camera: suite_dir / f"task0--ep0--succ1.{camera}.mp4"
            for camera in build_manifest.MULTIVIEW_CAMERAS
        }
        for path in cameras.values():
            path.write_bytes(b"camera")
        metadata = {
            "schema_version": 2,
            "video_view_mode": "libero_three_view",
            "multiview_layout": "separate_videos",
            "multiview_cameras": list(build_manifest.MULTIVIEW_CAMERAS),
            "camera_video_paths": {
                camera: path.name for camera, path in cameras.items()
            },
        }
        canonical.with_name(canonical.stem + ".multiview.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        return canonical, cameras

    def test_build_record_emits_separate_camera_video_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical, cameras = self.make_rollout(root)
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ):
                record = build_manifest.build_record(canonical, root, {})

            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record["video_path"], str(canonical.relative_to(root)))
            self.assertIsNone(record["multiview_video_path"])
            self.assertEqual(record["video_view_mode"], "libero_three_view")
            self.assertEqual(record["multiview_layout"], "separate_videos")
            self.assertEqual(
                record["multiview_cameras"],
                list(build_manifest.MULTIVIEW_CAMERAS),
            )
            self.assertEqual(
                record["camera_video_paths"],
                {
                    camera: str(path.relative_to(root))
                    for camera, path in cameras.items()
                },
            )

    def test_build_record_rejects_partial_camera_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical, cameras = self.make_rollout(root)
            cameras["sideview"].unlink()
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ):
                with self.assertRaisesRegex(RuntimeError, "Incomplete multiview camera set"):
                    build_manifest.build_record(canonical, root, {})


if __name__ == "__main__":
    unittest.main()
