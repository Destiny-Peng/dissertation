from __future__ import annotations

import unittest

import server
import webui_runtime  # noqa: F401  # installs the runtime coordinator semantics


class GenerationConcurrencyTest(unittest.TestCase):
    def test_baseline_and_manifest_writer_can_overlap(self) -> None:
        coordinator = server.JobCoordinator()
        coordinator.acquire("baseline-a", "baseline")
        coordinator.acquire("generation-a", "manifest_writer")
        self.assertEqual(
            coordinator.active(),
            {"baseline-a": "baseline", "generation-a": "manifest_writer"},
        )

    def test_manifest_writer_and_baseline_can_overlap_in_reverse_order(self) -> None:
        coordinator = server.JobCoordinator()
        coordinator.acquire("generation-a", "manifest_writer")
        coordinator.acquire("baseline-a", "baseline")
        self.assertEqual(
            coordinator.active(),
            {"generation-a": "manifest_writer", "baseline-a": "baseline"},
        )

    def test_two_manifest_writers_remain_exclusive(self) -> None:
        coordinator = server.JobCoordinator()
        coordinator.acquire("generation-a", "manifest_writer")
        with self.assertRaises(server.JobConflictError):
            coordinator.acquire("generation-b", "manifest_writer")


if __name__ == "__main__":
    unittest.main()
