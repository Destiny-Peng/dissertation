from __future__ import annotations

import unittest
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = TOOL_ROOT / "static"


class WebUiArchitectureContractTest(unittest.TestCase):
    def test_single_stable_server_entrypoint(self) -> None:
        self.assertTrue((TOOL_ROOT / "server_entry.py").is_file())
        self.assertEqual(list(TOOL_ROOT.glob("server_entry_v*.py")), [])
        entry = (TOOL_ROOT / "server_entry.py").read_text(encoding="utf-8")
        self.assertIn("import webui_runtime", entry)
        self.assertIn("webui_runtime.main()", entry)

    def test_semantic_backend_modules_exist(self) -> None:
        for name in [
            "webui_baseline.py",
            "webui_application.py",
            "webui_handler.py",
            "webui_runtime.py",
        ]:
            path = TOOL_ROOT / name
            self.assertTrue(path.is_file(), name)
            self.assertNotIn('if __name__ == "__main__"', path.read_text(encoding="utf-8"))

    def test_backend_modules_do_not_monkey_patch_server_classes(self) -> None:
        forbidden = [
            "server.BaselineService.",
            "server.LF3RApplication.__init__ =",
            "server.LF3RHandler.do_GET =",
            "server.LF3RHandler.do_POST =",
            "server.JobCoordinator.acquire =",
        ]
        for name in [
            "webui_baseline.py",
            "webui_application.py",
            "webui_handler.py",
            "webui_runtime.py",
        ]:
            source = (TOOL_ROOT / name).read_text(encoding="utf-8")
            for needle in forbidden:
                self.assertNotIn(needle, source, f"{name}: {needle}")

    def test_obsolete_patch_modules_are_removed(self) -> None:
        for name in [
            "webui_integrations.py",
            "webui_jobs.py",
            "webui_manifests.py",
        ]:
            self.assertFalse((TOOL_ROOT / name).exists(), name)

    def test_application_wires_explicit_service_subclass(self) -> None:
        application = (TOOL_ROOT / "webui_application.py").read_text(encoding="utf-8")
        handler = (TOOL_ROOT / "webui_handler.py").read_text(encoding="utf-8")
        runtime = (TOOL_ROOT / "webui_runtime.py").read_text(encoding="utf-8")
        self.assertIn("baseline_service_class = WebUIBaselineService", application)
        self.assertIn("project_tool_service_class = NonAnalysisToolService", application)
        self.assertIn("class WebUIHandler(server.LF3RHandler):", handler)
        self.assertIn("WebUIApplication(", runtime)
        self.assertIn("_make_handler(app)", runtime)

    def test_scripts_use_only_stable_entrypoint(self) -> None:
        run_server = (TOOL_ROOT / "run_server.sh").read_text(encoding="utf-8")
        stop_server = (TOOL_ROOT / "stop_server.sh").read_text(encoding="utf-8")
        self.assertIn("server_entry.py", run_server)
        self.assertIn("server_entry.py", stop_server)
        self.assertNotRegex(run_server, r"server_entry_v\d+\.py")
        self.assertNotRegex(stop_server, r"server_entry_v\d+\.py")

    def test_frontend_loader_has_no_manual_version_query(self) -> None:
        workspace = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")
        self.assertNotIn("?v=", workspace)
        self.assertNotRegex(workspace, r"/static/[^\"']+-v\d+\.js")

    def test_patch_era_frontend_files_are_removed(self) -> None:
        for name in [
            "workspace-legacy.js",
            "manifest-support-v2.js",
            "results-run-config-v3.js",
            "results-run-click-bridge.js",
            "runs-submit-fix.js",
        ]:
            self.assertFalse((STATIC_ROOT / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
