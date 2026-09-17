from __future__ import annotations

import unittest
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]


class ProgressiveBaselineResultsContractTest(unittest.TestCase):
    def test_server_exposes_only_completed_rollouts_from_running_runs(self) -> None:
        source = (TOOL_ROOT / "server_entry_v4.py").read_text(encoding="utf-8")
        self.assertIn('_PROGRESSIVE_RUN_STATUSES = set(server.BASELINE_RUN_STATUSES) | {"running"}', source)
        self.assertIn('workers_root.glob("worker-*/jobs.jsonl")', source)
        self.assertIn('row.get("status") != "complete"', source)
        self.assertIn('rollout_id not in server.run_rollout_ids(run_path)', source)
        self.assertIn('server.BaselineService._run_candidates = _progressive_run_candidates', source)
        self.assertIn('server.BaselineService._read_method = _read_only_completed_progressive_rollout', source)

    def test_frontend_refreshes_catalog_when_completed_count_advances(self) -> None:
        script = (TOOL_ROOT / "static/progressive-baseline-results.js").read_text(encoding="utf-8")
        workspace = (TOOL_ROOT / "static/workspace.js").read_text(encoding="utf-8")
        self.assertIn('nextCompleted > previousCompleted', script)
        self.assertIn('loadCatalog(condition, true)', script)
        self.assertIn('matchingProgressiveRun(job, record, condition)', script)
        self.assertIn('currentDisplayedRunRoot(job.baseline)', script)
        self.assertIn('window.loadEvaluation(rolloutId)', script)
        self.assertIn('progressive-baseline-results.js?v=progressive-baseline-results-v1-20260917', workspace)


if __name__ == "__main__":
    unittest.main()
