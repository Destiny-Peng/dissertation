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

    def test_server_is_thin_compatibility_facade(self) -> None:
        server = (TOOL_ROOT / "server.py").read_text(encoding="utf-8")
        self.assertLess(len(server.splitlines()), 250)
        for module in [
            "baseline_service",
            "analysis_service",
            "analysis_jobs",
            "rollout_service",
            "application",
            "http_handler",
        ]:
            self.assertIn("from " + module + " import", server)
        for class_name in [
            "BaselineService",
            "AnalysisService",
            "AnalysisJobService",
            "RolloutGenerationService",
            "LF3RApplication",
            "LF3RHandler",
        ]:
            self.assertNotIn("class " + class_name + ":", server)

    def test_backend_service_modules_exist(self) -> None:
        for name in [
            "backend_core.py",
            "stores.py",
            "baseline_constants.py",
            "baseline_index.py",
            "baseline_readers.py",
            "baseline_catalog.py",
            "baseline_results.py",
            "baseline_jobs.py",
            "baseline_service.py",
            "analysis_constants.py",
            "analysis_snapshots.py",
            "analysis_details.py",
            "analysis_service.py",
            "analysis_robo_jobs.py",
            "analysis_localization_results.py",
            "analysis_localization_challenge.py",
            "analysis_localization.py",
            "analysis_jobs.py",
            "rollout_service.py",
            "application.py",
            "http_handler.py",
        ]:
            self.assertTrue((TOOL_ROOT / name).is_file(), name)

    def test_baseline_results_is_thin_composition(self) -> None:
        results = (TOOL_ROOT / "baseline_results.py").read_text(encoding="utf-8")
        readers = (TOOL_ROOT / "baseline_readers.py").read_text(encoding="utf-8")
        catalog = (TOOL_ROOT / "baseline_catalog.py").read_text(encoding="utf-8")
        self.assertLess(len(results.splitlines()), 60)
        self.assertIn("BaselineReadersMixin", results)
        self.assertIn("BaselineCatalogMixin", results)
        self.assertIn("class BaselineReadersMixin:", readers)
        self.assertIn("class BaselineCatalogMixin:", catalog)

    def test_baseline_service_is_thin_composition(self) -> None:
        service = (TOOL_ROOT / "baseline_service.py").read_text(encoding="utf-8")
        results = (TOOL_ROOT / "baseline_results.py").read_text(encoding="utf-8")
        jobs = (TOOL_ROOT / "baseline_jobs.py").read_text(encoding="utf-8")
        self.assertLess(len(service.splitlines()), 180)
        self.assertIn(
            "class BaselineService(BaselineResultsMixin, BaselineJobsMixin):",
            service,
        )
        self.assertIn("class BaselineResultsMixin:", results)
        self.assertIn("class BaselineJobsMixin:", jobs)
        self.assertNotIn("baseline_service", results)
        self.assertNotIn("baseline_service", jobs)

    def test_analysis_service_is_thin_composition(self) -> None:
        service = (TOOL_ROOT / "analysis_service.py").read_text(encoding="utf-8")
        snapshots = (TOOL_ROOT / "analysis_snapshots.py").read_text(encoding="utf-8")
        details = (TOOL_ROOT / "analysis_details.py").read_text(encoding="utf-8")
        self.assertLess(len(service.splitlines()), 140)
        self.assertIn("AnalysisSnapshotsMixin", service)
        self.assertIn("AnalysisDetailsMixin", service)
        self.assertIn("class AnalysisSnapshotsMixin:", snapshots)
        self.assertIn("class AnalysisDetailsMixin:", details)
        self.assertNotIn("analysis_service", snapshots)
        self.assertNotIn("analysis_service", details)

    def test_localization_workflow_is_thin_composition(self) -> None:
        workflow = (TOOL_ROOT / "analysis_localization.py").read_text(encoding="utf-8")
        results = (TOOL_ROOT / "analysis_localization_results.py").read_text(encoding="utf-8")
        challenge = (TOOL_ROOT / "analysis_localization_challenge.py").read_text(encoding="utf-8")
        self.assertLess(len(workflow.splitlines()), 60)
        self.assertIn("AnalysisLocalizationResultsMixin", workflow)
        self.assertIn("AnalysisLocalizationChallengeMixin", workflow)
        self.assertIn("class AnalysisLocalizationResultsMixin:", results)
        self.assertIn("class AnalysisLocalizationChallengeMixin:", challenge)

    def test_analysis_job_service_is_thin_composition(self) -> None:
        service = (TOOL_ROOT / "analysis_jobs.py").read_text(encoding="utf-8")
        robo = (TOOL_ROOT / "analysis_robo_jobs.py").read_text(encoding="utf-8")
        localization = (TOOL_ROOT / "analysis_localization.py").read_text(encoding="utf-8")
        self.assertLess(len(service.splitlines()), 650)
        self.assertIn("AnalysisRoboJobsMixin", service)
        self.assertIn("AnalysisLocalizationMixin", service)
        self.assertIn("class AnalysisRoboJobsMixin:", robo)
        self.assertIn("class AnalysisLocalizationMixin:", localization)
        self.assertNotIn("analysis_jobs", robo)
        self.assertNotIn("analysis_jobs", localization)

    def test_scripts_use_only_stable_entrypoint(self) -> None:
        run_server = (TOOL_ROOT / "run_server.sh").read_text(encoding="utf-8")
        stop_server = (TOOL_ROOT / "stop_server.sh").read_text(encoding="utf-8")
        self.assertIn("server_entry.py", run_server)
        self.assertIn("server_entry.py", stop_server)
        self.assertNotRegex(run_server, r"server_entry_v\d+\.py")
        self.assertNotRegex(stop_server, r"server_entry_v\d+\.py")

    def test_results_frontend_has_semantic_modules(self) -> None:
        results_root = STATIC_ROOT / "results"
        for name in ["charts.js", "core.js", "layout.js", "run-config.js"]:
            self.assertTrue((results_root / name).is_file(), name)
        for obsolete in ["results-layout.js", "results-run-config.js"]:
            self.assertFalse((STATIC_ROOT / obsolete).exists(), obsolete)

        charts = (results_root / "charts.js").read_text(encoding="utf-8")
        core = (results_root / "core.js").read_text(encoding="utf-8")
        layout = (results_root / "layout.js").read_text(encoding="utf-8")
        config = (results_root / "run-config.js").read_text(encoding="utf-8")
        self.assertIn("LF3RResultsCharts.renderSignalChart", core)
        self.assertIn("renderSignalChart", charts)
        self.assertIn("lf3rOpenSingleBaselineConfig", core)
        self.assertIn("lf3rResultsLayoutRefresh", core)
        self.assertNotIn("MutationObserver", layout)
        self.assertNotIn("MutationObserver", config)

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
