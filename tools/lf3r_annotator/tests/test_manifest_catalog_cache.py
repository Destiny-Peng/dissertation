from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from webui_application import WebUIApplication  # noqa: E402


class ManifestCatalogCacheTest(unittest.TestCase):
    @staticmethod
    def _make_app(root: Path, sources: list[Path]) -> WebUIApplication:
        app = WebUIApplication.__new__(WebUIApplication)
        app.project_root = root.resolve()
        app.manifest_paths = [path.resolve() for path in sources]
        app.aggregate_manifest_path = root / "cache" / "catalog-test.jsonl"
        app.aggregate_manifest_meta_path = root / "cache" / "catalog-test.meta.json"
        app._manifest_catalog_lock = threading.RLock()
        app._manifest_catalog_signature = None
        app._manifest_records_cache = []
        app._manifest_info_cache = []
        app._manifest_source_by_id = {}
        return app

    @staticmethod
    def _write_manifest(path: Path, rollout_id: str, role: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "id": rollout_id,
                    "camera_video_paths": {"cam_high": f"videos/{rollout_id}.mp4"},
                    "dataset_role": role,
                    "task_suite": role,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_second_process_restores_catalog_without_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "datasets" / "a_manifest.jsonl"
            source_b = root / "datasets" / "b_manifest.jsonl"
            self._write_manifest(source_a, "rollout-a", "a")
            self._write_manifest(source_b, "rollout-b", "b")

            first = self._make_app(root, [source_a, source_b])
            records = first._refresh_manifest_catalog(force=False)
            self.assertEqual({row["id"] for row in records}, {"rollout-a", "rollout-b"})
            self.assertTrue(first.aggregate_manifest_meta_path.is_file())

            second = self._make_app(root, [source_a, source_b])
            with mock.patch.object(
                second,
                "_validate_manifest_rows",
                side_effect=AssertionError("cache hit must not validate source rows"),
            ):
                restored = second._refresh_manifest_catalog(force=False)

            self.assertEqual({row["id"] for row in restored}, {"rollout-a", "rollout-b"})
            self.assertEqual(
                {row["manifest_source"] for row in restored},
                {
                    str(source_a.relative_to(root)),
                    str(source_b.relative_to(root)),
                },
            )

    def test_manifest_change_invalidates_persistent_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "datasets" / "manifest.jsonl"
            self._write_manifest(source, "rollout-a", "a")

            first = self._make_app(root, [source])
            first._refresh_manifest_catalog(force=False)

            self._write_manifest(source, "rollout-b", "b")
            second = self._make_app(root, [source])
            records = second._refresh_manifest_catalog(force=False)
            self.assertEqual([row["id"] for row in records], ["rollout-b"])


if __name__ == "__main__":
    unittest.main()
