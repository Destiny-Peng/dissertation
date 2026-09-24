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


class BuildManifestCameraVideoTests(unittest.TestCase):
    def make_rollout(self, root: Path) -> tuple[Path, Path]:
        suite_dir = root / "outputs" / "lf3r-data-natural-test" / "libero_10"
        suite_dir.mkdir(parents=True)
        canonical = suite_dir / "task0--ep0--succ1.mp4"
        canonical.write_bytes(b"canonical")
        # Deliberately does not encode the camera name in the filename.
        wrist = suite_dir / "extra-view-17.mp4"
        wrist.write_bytes(b"wrist")
        metadata = {
            "schema_version": 2,
            "camera_video_paths": {
                "cam_high": canonical.name,
                "cam_wrist": wrist.name,
            },
        }
        canonical.with_name(canonical.stem + ".camera_videos.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        return canonical, wrist

    def test_build_record_emits_only_camera_video_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical, wrist = self.make_rollout(root)
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ) as probe:
                record = build_manifest.build_record(canonical, root, {})

            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record["schema_version"], 2)
            self.assertNotIn("video_path", record)
            self.assertEqual(
                record["camera_video_paths"],
                {
                    "cam_high": str(canonical.relative_to(root)),
                    "cam_wrist": str(wrist.relative_to(root)),
                },
            )
            self.assertEqual(probe.call_count, 2)

    def test_build_record_does_not_infer_camera_from_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite_dir = root / "outputs" / "lf3r-data-natural-test" / "libero_10"
            suite_dir.mkdir(parents=True)
            canonical = suite_dir / "task0--ep0--succ1.mp4"
            canonical.write_bytes(b"canonical")
            # This filename looks like a wrist view, but without explicit
            # metadata it must not be added to the manifest.
            (suite_dir / "task0--ep0--succ1.cam_wrist.mp4").write_bytes(b"ignored")

            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ) as probe:
                record = build_manifest.build_record(canonical, root, {})

            assert record is not None
            self.assertEqual(
                record["camera_video_paths"],
                {"cam_high": str(canonical.relative_to(root))},
            )
            self.assertEqual(probe.call_count, 1)

    def test_build_record_rejects_declared_missing_camera_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical, wrist = self.make_rollout(root)
            wrist.unlink()
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ):
                with self.assertRaisesRegex(RuntimeError, "Camera video is missing"):
                    build_manifest.build_record(canonical, root, {})

    def test_build_record_rejects_cam_high_alias_to_different_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical, wrist = self.make_rollout(root)
            metadata_path = canonical.with_name(canonical.stem + ".camera_videos.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["camera_video_paths"]["cam_high"] = wrist.name
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ):
                with self.assertRaisesRegex(RuntimeError, "cam_high must reference"):
                    build_manifest.build_record(canonical, root, {})


if __name__ == "__main__":
    unittest.main()
