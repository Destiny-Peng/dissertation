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
        for path in sorted(STATIC_ROOT.glob("*.js")):
            javascript = path.read_text(encoding="utf-8")
            if dangerous.search(javascript):
                offenders.append(path.name)
        self.assertEqual(
            offenders,
            [],
            "Document-wide childList+subtree observers can self-trigger and freeze the tab",
        )

    def test_known_observers_are_narrowed(self) -> None:
        procvlm = (STATIC_ROOT / "procvlm-mode-ui.js").read_text(encoding="utf-8")
        self.assertIn("drawerObserver.observe(singleCore, { childList: true })", procvlm)
        self.assertNotIn("drawerObserver.observe(document.body", procvlm)

        results_config = (STATIC_ROOT / "results-run-config-v3.js").read_text(encoding="utf-8")
        self.assertIn("observer.observe(methodsHost, { childList: true })", results_config)
        self.assertNotIn(
            "observer.observe(methodsHost, { childList: true, subtree: true })",
            results_config,
        )

        runs_log = (STATIC_ROOT / "runs-log-ui.js").read_text(encoding="utf-8")
        self.assertNotIn(
            "new MutationObserver(syncButtons).observe(view, { childList: true, subtree: true })",
            runs_log,
        )
        self.assertIn(
            "new MutationObserver(syncButtons).observe(channel.jobs, { childList: true })",
            runs_log,
        )

        runs_control = (STATIC_ROOT / "runs-job-control.js").read_text(encoding="utf-8")
        self.assertIn(
            "new MutationObserver(decorateCards).observe(jobs, { childList: true })",
            runs_control,
        )

    def test_stale_results_configurator_is_removed(self) -> None:
        self.assertFalse((STATIC_ROOT / "results-run-config-v2.js").exists())
        workspace = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")
        self.assertNotIn("results-run-config-v2.js", workspace)


if __name__ == "__main__":
    unittest.main()
