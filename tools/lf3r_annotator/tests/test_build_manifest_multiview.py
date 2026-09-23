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
        high = suite_dir / "task0--ep0--succ1.cam_high.mp4"
        wrist = suite_dir / "task0--ep0--succ1.cam_wrist.mp4"
        high.write_bytes(b"high")
        wrist.write_bytes(b"wrist")
        cameras = {
            "cam_high": high,
            "cam_wrist": wrist,
        }
        metadata = {
            "schema_version": 1,
            "camera_video_paths": {
                slot: path.name for slot, path in cameras.items()
            },
        }
        canonical.with_name(canonical.stem + ".camera_videos.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        return canonical, cameras

    def test_build_record_emits_physical_camera_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical, cameras = self.make_rollout(root)
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ) as probe:
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
            self.assertNotIn("camera_source_names", record)
            # canonical + two unique camera files; shared wrist is probed once.
            self.assertEqual(probe.call_count, 3)

    def test_build_record_normalizes_legacy_shared_wrist_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical, cameras = self.make_rollout(root)
            new_wrist = cameras["cam_wrist"]
            legacy_wrist = new_wrist.with_name(
                new_wrist.name.replace(".cam_wrist.mp4", ".cam_left_wrist.mp4")
            )
            new_wrist.rename(legacy_wrist)
            metadata_path = canonical.with_name(canonical.stem + ".camera_videos.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["camera_video_paths"] = {
                "cam_high": cameras["cam_high"].name,
                "cam_left_wrist": legacy_wrist.name,
                "cam_right_wrist": legacy_wrist.name,
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ):
                record = build_manifest.build_record(canonical, root, {})

            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(
                record["camera_video_paths"],
                {
                    "cam_high": str(cameras["cam_high"].relative_to(root)),
                    "cam_wrist": str(legacy_wrist.relative_to(root)),
                },
            )
            self.assertNotIn("cam_left_wrist", record["camera_video_paths"])
            self.assertNotIn("cam_right_wrist", record["camera_video_paths"])

    def test_build_record_rejects_missing_shared_wrist_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical, cameras = self.make_rollout(root)
            cameras["cam_wrist"].unlink()
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Incomplete camera video set",
                ):
                    build_manifest.build_record(canonical, root, {})


if __name__ == "__main__":
    unittest.main()
