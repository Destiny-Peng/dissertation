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
    def make_rollout(self, root: Path) -> tuple[Path, dict[str, Path]]:
        suite_dir = root / "outputs" / "lf3r-data-natural-test" / "libero_10"
        suite_dir.mkdir(parents=True)
        canonical = suite_dir / "task0--ep0--succ1.mp4"
        canonical.write_bytes(b"canonical")
        cameras = {
            slot: suite_dir / f"task0--ep0--succ1.{slot}.mp4"
            for slot in build_manifest.ROBO_DOPAMINE_CAMERA_SLOTS
        }
        for path in cameras.values():
            path.write_bytes(b"camera")
        metadata = {
            "schema_version": 1,
            "camera_video_paths": {
                slot: path.name for slot, path in cameras.items()
            },
            "camera_source_names": dict(
                build_manifest.ROBO_DOPAMINE_CAMERA_SOURCES
            ),
            "consumer_interface": "robo_dopamine_three_view",
        }
        canonical.with_name(canonical.stem + ".camera_videos.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        return canonical, cameras

    def test_build_record_emits_robo_dopamine_camera_video_paths(self) -> None:
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
            self.assertEqual(
                record["camera_video_paths"],
                {
                    slot: str(path.relative_to(root))
                    for slot, path in cameras.items()
                },
            )
            self.assertEqual(
                record["camera_source_names"],
                build_manifest.ROBO_DOPAMINE_CAMERA_SOURCES,
            )
            for removed in (
                "multiview_video_path",
                "multiview_layout",
                "multiview_cameras",
            ):
                self.assertNotIn(removed, record)

    def test_build_record_rejects_partial_robo_camera_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical, cameras = self.make_rollout(root)
            cameras["cam_right_wrist"].unlink()
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Incomplete Robo-Dopamine camera set",
                ):
                    build_manifest.build_record(canonical, root, {})


if __name__ == "__main__":
    unittest.main()
