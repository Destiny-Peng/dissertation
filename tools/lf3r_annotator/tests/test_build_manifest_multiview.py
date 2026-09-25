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
        rollout = suite_dir / "task0--ep0--succ1.mp4"
        rollout.write_bytes(b"canonical")
        high = suite_dir / "high-view-any-name.mp4"
        wrist = suite_dir / "wrist-view-any-name.mp4"
        high.write_bytes(b"high")
        wrist.write_bytes(b"wrist")
        cameras = {"cam_high": high, "cam_wrist": wrist}
        metadata = {
            "schema_version": 1,
            "camera_video_paths": {
                slot: path.name for slot, path in cameras.items()
            },
        }
        rollout.with_name(rollout.stem + ".camera_videos.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        return rollout, cameras

    def test_build_record_emits_only_declared_physical_camera_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rollout, cameras = self.make_rollout(root)
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ) as probe:
                record = build_manifest.build_record(rollout, root, {})

            self.assertIsNotNone(record)
            assert record is not None
            self.assertNotIn("video_path", record)
            self.assertEqual(
                record["camera_video_paths"],
                {
                    slot: str(path.relative_to(root))
                    for slot, path in cameras.items()
                },
            )
            self.assertNotIn("camera_source_names", record)
            self.assertEqual(probe.call_count, 3)

    def test_plain_single_view_rollout_is_cam_high(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite_dir = root / "outputs" / "lf3r-data-natural-test" / "libero_10"
            suite_dir.mkdir(parents=True)
            rollout = suite_dir / "task0--ep0--succ1.mp4"
            rollout.write_bytes(b"single")
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ) as probe:
                record = build_manifest.build_record(rollout, root, {})

            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(
                record["camera_video_paths"],
                {"cam_high": str(rollout.relative_to(root))},
            )
            self.assertNotIn("video_path", record)
            self.assertEqual(probe.call_count, 1)

    def test_camera_paths_do_not_depend_on_filename_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rollout, cameras = self.make_rollout(root)
            self.assertFalse(cameras["cam_high"].name.endswith(".cam_high.mp4"))
            self.assertFalse(cameras["cam_wrist"].name.endswith(".cam_wrist.mp4"))
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ):
                record = build_manifest.build_record(rollout, root, {})
            assert record is not None
            self.assertEqual(set(record["camera_video_paths"]), {"cam_high", "cam_wrist"})

    def test_duplicate_wrist_aliases_are_preserved_without_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rollout, cameras = self.make_rollout(root)
            metadata_path = rollout.with_name(rollout.stem + ".camera_videos.json")
            metadata = {
                "schema_version": 1,
                "camera_video_paths": {
                    "cam_high": cameras["cam_high"].name,
                    "cam_left_wrist": cameras["cam_wrist"].name,
                    "cam_right_wrist": cameras["cam_wrist"].name,
                },
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ):
                record = build_manifest.build_record(rollout, root, {})

            assert record is not None
            self.assertEqual(
                record["camera_video_paths"]["cam_left_wrist"],
                record["camera_video_paths"]["cam_right_wrist"],
            )
            self.assertNotIn("cam_wrist", record["camera_video_paths"])
            self.assertNotIn("primary_camera", record)

    def test_empty_scan_refuses_to_overwrite_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scan_root = root / "outputs"
            scan_root.mkdir()
            output = root / "manifest.jsonl"
            output.write_text("keep-existing\n", encoding="utf-8")
            args = mock.Mock(
                project_root=root,
                scan_root=[scan_root],
                task_metadata=root / "missing-task-metadata.json",
                output=output,
                refresh_instruction_variants=False,
            )

            with mock.patch.object(build_manifest, "parse_args", return_value=args):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "No rollout records discovered; refusing to overwrite",
                ):
                    build_manifest.main()

            self.assertEqual(output.read_text(encoding="utf-8"), "keep-existing\n")

    def test_existing_camera_sidecar_must_declare_camera_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite_dir = root / "outputs" / "lf3r-data-natural-test" / "libero_10"
            suite_dir.mkdir(parents=True)
            rollout = suite_dir / "task0--ep0--succ1.mp4"
            rollout.write_bytes(b"single")
            rollout.with_name(rollout.stem + ".camera_videos.json").write_text(
                json.dumps({"schema_version": 1}),
                encoding="utf-8",
            )
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "has no camera_video_paths",
                ):
                    build_manifest.build_record(rollout, root, {})

    def test_declared_missing_camera_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rollout, cameras = self.make_rollout(root)
            cameras["cam_wrist"].unlink()
            with mock.patch.object(
                build_manifest,
                "probe_video",
                return_value=(42, 30.0, 1.4),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Declared camera video is missing",
                ):
                    build_manifest.build_record(rollout, root, {})


if __name__ == "__main__":
    unittest.main()
