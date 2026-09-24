from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TOOL_ROOT / "transcode_manifest_videos_h264.py"
SPEC = importlib.util.spec_from_file_location("transcode_manifest_videos_h264", SCRIPT)
assert SPEC and SPEC.loader
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class ManifestBatchH264Tests(unittest.TestCase):
    def test_manifest_reader_uses_all_camera_paths_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            high = root / "outputs" / "high.mp4"
            wrist = root / "outputs" / "wrist.mp4"
            high.parent.mkdir(parents=True)
            high.write_bytes(b"high")
            wrist.write_bytes(b"wrist")

            first = root / "first.jsonl"
            second = root / "second.jsonl"
            row = {
                "schema_version": 2,
                "id": "r1",
                "camera_video_paths": {
                    "cam_high": "outputs/high.mp4",
                    "cam_wrist": "outputs/wrist.mp4",
                },
            }
            first.write_text(json.dumps(row) + "\n", encoding="utf-8")
            second.write_text(
                json.dumps({**row, "id": "r2"}) + "\n",
                encoding="utf-8",
            )

            videos = batch.read_manifest_video_paths(root.resolve(), [first, second])

            self.assertEqual(videos, [high.resolve(), wrist.resolve()])

    def test_manifest_reader_rejects_non_mp4_camera_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({
                    "schema_version": 2,
                    "id": "r1",
                    "camera_video_paths": {"cam_high": "outputs/video.avi"},
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not an .mp4"):
                batch.read_manifest_video_paths(root.resolve(), [manifest])


if __name__ == "__main__":
    unittest.main()
