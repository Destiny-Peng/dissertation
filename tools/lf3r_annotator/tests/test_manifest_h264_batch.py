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
    def test_manifest_reader_uses_only_canonical_video_path_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = root / "outputs" / "canonical.mp4"
            cam_high = root / "outputs" / "canonical.cam_high.mp4"
            canonical.parent.mkdir(parents=True)
            canonical.write_bytes(b"canonical")
            cam_high.write_bytes(b"cam-high")

            first = root / "first.jsonl"
            second = root / "second.jsonl"
            row = {
                "id": "r1",
                "video_path": "outputs/canonical.mp4",
                "camera_video_paths": {
                    "cam_high": "outputs/canonical.cam_high.mp4",
                },
            }
            first.write_text(json.dumps(row) + "\n", encoding="utf-8")
            second.write_text(
                json.dumps({**row, "id": "r2"}) + "\n",
                encoding="utf-8",
            )

            videos = batch.read_manifest_video_paths(root.resolve(), [first, second])

            self.assertEqual(videos, [canonical.resolve()])
            self.assertNotIn(cam_high.resolve(), videos)

    def test_manifest_reader_rejects_non_mp4_video_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"id": "r1", "video_path": "outputs/video.avi"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not an .mp4"):
                batch.read_manifest_video_paths(root.resolve(), [manifest])


if __name__ == "__main__":
    unittest.main()
