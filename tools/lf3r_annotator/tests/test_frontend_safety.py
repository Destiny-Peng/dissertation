from __future__ import annotations

import re
import unittest
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = TOOL_ROOT / "static"


class FrontendSafetyContractTest(unittest.TestCase):
    def test_observer_guard_loads_before_enhancement_scripts(self) -> None:
        workspace = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")
        guard = workspace.index("/static/frontend-loop-guard.js")
        first_enhancement = workspace.index("/static/raw-video-source.js")
        self.assertLess(guard, first_enhancement)
        self.assertTrue((STATIC_ROOT / "frontend-loop-guard.js").is_file())

    def test_no_document_wide_childlist_subtree_observer(self) -> None:
        dangerous = re.compile(
            r"\.observe\(\s*document\.(?:body|documentElement)\s*,\s*\{"
            r"(?=[^}]*childList\s*:\s*true)"
            r"(?=[^}]*subtree\s*:\s*true)[^}]*\}\s*\)",
            re.DOTALL,
        )
        offenders: list[str] = []
        for path in sorted(STATIC_ROOT.rglob("*.js")):
            javascript = path.read_text(encoding="utf-8")
            if dangerous.search(javascript):
                offenders.append(path.name)
        self.assertEqual(
            offenders,
            [],
            "Document-wide childList+subtree observers can self-trigger and freeze the tab",
        )

    def test_known_observers_are_narrowed(self) -> None:
        procvlm = (STATIC_ROOT / "baselines" / "procvlm.js").read_text(encoding="utf-8")
        self.assertNotIn("MutationObserver", procvlm)
        self.assertNotIn("window.fetch =", procvlm)
        self.assertIn("applyOptions: applyOptions", procvlm)

        results_config = (STATIC_ROOT / "results" / "run-config.js").read_text(encoding="utf-8")
        self.assertNotIn("MutationObserver", results_config)
        self.assertNotIn("stopImmediatePropagation", results_config)

        results_layout = (STATIC_ROOT / "results" / "layout.js").read_text(encoding="utf-8")
        self.assertNotIn("MutationObserver", results_layout)
        self.assertIn("window.lf3rResultsLayoutRefresh = refresh", results_layout)

        runs_jobs = (STATIC_ROOT / "runs" / "jobs.js").read_text(encoding="utf-8")
        self.assertNotIn("MutationObserver", runs_jobs)
        self.assertNotRegex(runs_jobs, r"window\\.renderPersistentJobLists\\s*=(?!=)")
        self.assertNotIn("window.loadBaselineBatchLog =", runs_jobs)

    def test_runs_log_refresh_stays_bound_to_selected_job(self) -> None:
        runs_jobs = (STATIC_ROOT / "runs" / "jobs.js").read_text(encoding="utf-8")
        app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            'if (!channel || !channel.log || visibleJobId(channel) !== jobId) return;',
            runs_jobs,
        )
        self.assertIn(
            'if (requestSerial !== channel.requestSerial || visibleJobId(channel) !== jobId) return;',
            runs_jobs,
        )
        self.assertIn('refreshSelectedLog: refreshSelectedLog', runs_jobs)
        self.assertIn('window.LF3RRunsJobs.refreshSelectedLog("baseline", jobId)', app)
        self.assertIn('window.LF3RRunsJobs.refreshSelectedLog("rollout_generation", jobId)', app)

        workspace = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")
        self.assertIn("runs/jobs.js", workspace)
        self.assertNotIn("runs-log-ui.js", workspace)

    def test_procvlm_model_path_is_visible_in_single_run_config(self) -> None:
        procvlm = (STATIC_ROOT / "baselines" / "procvlm.js").read_text(encoding="utf-8")
        self.assertIn("var modelInput = drawer.querySelector('[data-option=\"model_path\"]')", procvlm)
        self.assertIn("modelLabel.parentNode !== core", procvlm)
        self.assertIn("core.insertBefore(modelLabel, modeLabel.nextSibling)", procvlm)

    def test_workspace_retries_transient_script_load_failures(self) -> None:
        workspace = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")
        self.assertIn("if (retry < 1)", workspace)
        self.assertIn("lf3r_retry=", workspace)
        self.assertIn("continuing with remaining modules", workspace)
        self.assertIn("if (onload) onload();", workspace)

    def test_workspace_uses_stable_module_names_without_manual_versions(self) -> None:
        workspace = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")
        self.assertNotIn("?v=", workspace)
        for stale in [
            "workspace-legacy.js",
            "manifest-support-v2.js",
            "results-run-config-v3.js",
            "results-run-click-bridge.js",
            "runs-submit-fix.js",
        ]:
            self.assertNotIn(stale, workspace)
        for stable in [
            "analysis/live.js",
            "analysis/snapshot.js",
            "analysis/localization.js",
            "analysis/change-point.js",
            "analysis/event-triggered.js",
            "analysis/runs.js",
            "analysis/dashboard.js",
            "analysis/signals.js",
            "analysis/details.js",
            "workspace-core.js",
            "manifest-support.js",
            "results/layout.js",
            "results/run-config.js",
            "runs/layout.js",
            "runs/jobs.js",
            "runs/scope.js",
            "baselines/procvlm.js",
        ]:
            self.assertIn(stable, workspace)

        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("/static/results/core.js", html)
        for obsolete in [
            "results-axis-scale.js",
            "results-current-values.js",
            "results-run-actions.js",
            "progressive-baseline-results.js",
            "results-layout.js",
            "results-run-config.js",
            "baseline-job-filter.js",
            "runs-log-ui.js",
            "runs-job-control.js",
            "runs-submit.js",
            "procvlm-mode-ui.js",
            "runs-semantics.js",
            "dataset-scope-ui.js",
            "runs-layout.js",
        ]:
            self.assertNotIn(obsolete, workspace)

    def test_stale_results_configurator_is_removed(self) -> None:
        self.assertFalse((STATIC_ROOT / "results-run-config-v2.js").exists())
        self.assertFalse((STATIC_ROOT / "results-run-config.js").exists())
        self.assertTrue((STATIC_ROOT / "results" / "run-config.js").is_file())
        workspace = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")
        self.assertNotIn("results-run-config-v2.js", workspace)


if __name__ == "__main__":
    unittest.main()
