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

    def test_runs_jobs_frontend_is_canonical(self) -> None:
        jobs = STATIC_ROOT / "runs" / "jobs.js"
        self.assertTrue(jobs.is_file())
        source = jobs.read_text(encoding="utf-8")
        self.assertIn("window.LF3RRunsJobs", source)
        self.assertNotIn("MutationObserver", source)
        self.assertNotRegex(source, r"window\\.renderPersistentJobLists\\s*=(?!=)")
        for obsolete in [
            "baseline-job-filter.js",
            "runs-log-ui.js",
            "runs-job-control.js",
            "runs-submit.js",
        ]:
            self.assertFalse((STATIC_ROOT / obsolete).exists(), obsolete)

    def test_procvlm_mode_does_not_patch_fetch(self) -> None:
        controller = STATIC_ROOT / "baselines" / "procvlm.js"
        self.assertTrue(controller.is_file())
        source = controller.read_text(encoding="utf-8")
        app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        run_config = (STATIC_ROOT / "results" / "run-config.js").read_text(encoding="utf-8")
        workspace = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")

        self.assertIn("applyOptions: applyOptions", source)
        self.assertNotIn("MutationObserver", source)
        self.assertNotIn("window.fetch =", source)
        self.assertIn('LF3RProcvlmMode.applyOptions(batchOptions, "batch")', app)
        self.assertIn('LF3RProcvlmMode.applyOptions(collected.options, "single")', run_config)
        self.assertIn('LF3RProcvlmMode.refreshSingle(method)', run_config)
        self.assertIn("/static/baselines/procvlm.js", workspace)
        self.assertNotIn("procvlm-mode-ui.js", workspace)
        self.assertFalse((STATIC_ROOT / "procvlm-mode-ui.js").exists())

    def test_dataset_scope_frontend_is_explicit(self) -> None:
        controller = STATIC_ROOT / "runs" / "scope.js"
        self.assertTrue(controller.is_file())
        source = controller.read_text(encoding="utf-8")
        app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        analysis_live = (STATIC_ROOT / "analysis" / "live.js").read_text(encoding="utf-8")
        workspace_router = (STATIC_ROOT / "workspace" / "router.js").read_text(encoding="utf-8")
        workspace = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")

        self.assertIn("window.LF3RDatasetScopes", source)
        self.assertIn("matchesBaseline: matchesBaseline", source)
        self.assertIn("matchesPartition: matchesPartition", source)
        self.assertIn("scopeLabel: scopeLabel", source)
        self.assertIn("decorateHelp: decorateHelp", source)
        self.assertNotIn("window.baselineBatchMatchesScope =", source)
        self.assertNotIn("window.workspacePartitionMatches =", source)
        self.assertNotIn("window.persistentJobScope =", source)
        self.assertNotIn("window.cliHelpEntry =", source)

        self.assertIn("LF3RDatasetScopes.matchesBaseline", app)
        self.assertIn("LF3RDatasetScopes.scopeLabel", app)
        self.assertIn("LF3RDatasetScopes.decorateHelp", app)
        self.assertIn("LF3RDatasetScopes.matchesPartition", analysis_live)
        self.assertIn("LF3RDatasetScopes.refresh", workspace_router)
        self.assertIn("/static/runs/scope.js", workspace)

        for obsolete in ["runs-semantics.js", "dataset-scope-ui.js"]:
            self.assertFalse((STATIC_ROOT / obsolete).exists(), obsolete)

    def test_runs_core_is_extracted_from_app_shell(self) -> None:
        runs_core = STATIC_ROOT / "runs" / "core.js"
        runs_layout = STATIC_ROOT / "runs" / "layout.js"
        self.assertTrue(runs_core.is_file())
        self.assertTrue(runs_layout.is_file())
        self.assertFalse((STATIC_ROOT / "runs-layout.js").exists())
        source = runs_core.read_text(encoding="utf-8")
        app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

        self.assertIn("function startBaselineBatch(", source)
        self.assertIn("function updateBaselineBatchSelection(", source)
        self.assertIn("function startRolloutGeneration(", source)
        self.assertIn("function rolloutGenerationJobMessage(", source)
        self.assertNotIn("function startBaselineBatch(", app)
        self.assertNotIn("function startRolloutGeneration(", app)
        self.assertLess(
            html.index("/static/runs/core.js"),
            html.index("/static/app.js"),
        )

    def test_analysis_frontend_is_split_from_workspace_shell(self) -> None:
        analysis_root = STATIC_ROOT / "analysis"
        analysis_module_names = [
            "live.js",
            "snapshot.js",
            "localization.js",
            "change-point.js",
            "event-triggered.js",
            "runs.js",
            "dashboard.js",
            "signals.js",
            "details.js",
        ]
        workspace_root = STATIC_ROOT / "workspace"
        workspace_module_names = ["settings.js", "router.js", "events.js"]
        shell = STATIC_ROOT / "workspace-core.js"
        loader = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")

        sources = {}
        for name in analysis_module_names:
            path = analysis_root / name
            self.assertTrue(path.is_file(), name)
            sources["analysis/" + name] = path.read_text(encoding="utf-8")
        self.assertFalse((analysis_root / "core.js").exists())

        workspace_sources = {}
        for name in workspace_module_names:
            path = workspace_root / name
            self.assertTrue(path.is_file(), name)
            workspace_sources["workspace/" + name] = path.read_text(encoding="utf-8")

        shell_source = shell.read_text(encoding="utf-8")
        self.assertIn("function workspaceRenderLiveAnalysis(", sources["analysis/live.js"])
        self.assertIn("function workspaceRenderCoverageChart(", sources["analysis/snapshot.js"])
        self.assertIn("function workspaceRenderLocalization(", sources["analysis/localization.js"])
        self.assertIn("function workspaceRenderChangePoint(", sources["analysis/change-point.js"])
        self.assertIn("function workspaceRenderEventTriggered(", sources["analysis/event-triggered.js"])
        self.assertIn("function workspaceLoadBaselineRuns(", sources["analysis/runs.js"])
        self.assertIn("function workspaceDashboardRenderSnapshot(", sources["analysis/dashboard.js"])
        self.assertIn("function workspaceDashboardRenderComparison(", sources["analysis/dashboard.js"])
        self.assertIn("function workspaceRenderSnapshot(", sources["analysis/dashboard.js"])
        self.assertIn("function workspaceDashboardRenderSignalShape(", sources["analysis/signals.js"])
        self.assertIn("function workspaceDashboardRenderDetails(", sources["analysis/details.js"])
        self.assertIn("function workspaceLoadAnalysisDetails(", sources["analysis/details.js"])

        self.assertIn("function workspaceLoadSettings(", workspace_sources["workspace/settings.js"])
        self.assertIn("function workspaceRenderRoute(", workspace_sources["workspace/router.js"])
        self.assertIn("function workspaceInstallEvents(", workspace_sources["workspace/events.js"])

        self.assertNotIn("function workspaceRenderLocalization(", shell_source)
        self.assertNotIn("function workspaceDashboardRenderSnapshot(", shell_source)
        self.assertNotIn("function workspaceLoadSettings(", shell_source)
        self.assertNotIn("function workspaceRenderRoute(", shell_source)
        self.assertNotIn("function workspaceInstallEvents(", shell_source)
        self.assertLess(len(shell_source), 8000)

        combined = "\n".join(sources.values()) + "\n" + "\n".join(workspace_sources.values()) + "\n" + shell_source
        for name in [
            "workspaceRenderTaskChart",
            "workspaceRenderSnapshot",
            "workspaceParseRoute",
            "workspaceRenderRoute",
            "workspaceDataChanged",
            "workspaceLoadSettings",
            "workspaceInstallEvents",
        ]:
            self.assertEqual(combined.count("function " + name + "("), 1, name)

        for name in analysis_module_names:
            self.assertLess(
                loader.index("/static/analysis/" + name),
                loader.index("/static/workspace-core.js"),
            )
        for name in workspace_module_names:
            self.assertLess(
                loader.index("/static/workspace/" + name),
                loader.index("/static/workspace-core.js"),
            )


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
