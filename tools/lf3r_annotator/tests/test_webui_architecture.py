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
        self.assertIn("repair_service_class = RepairService", application)
        self.assertIn("class WebUIHandler(server.LF3RHandler):", handler)
        self.assertIn("WebUIApplication(", runtime)
        self.assertIn("_make_handler(app)", runtime)

    def test_repair_is_top_level_semantic_module(self) -> None:
        repair_root = TOOL_ROOT / "repair"
        static_repair = STATIC_ROOT / "repair"
        for name in [
            "__init__.py",
            "alignment.py",
            "alignment_cli.py",
            "alignment_runner.py",
            "prepare_libero_manifest.py",
            "trajectory.py",
            "adapters.py",
            "service.py",
            "worker.py",
        ]:
            self.assertTrue((repair_root / name).is_file(), name)
        for name in ["styles.css", "page.js", "synthetic-suffix.js"]:
            self.assertTrue((static_repair / name).is_file(), name)

        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        router = (STATIC_ROOT / "workspace" / "router.js").read_text(encoding="utf-8")
        loader = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")
        handler = (TOOL_ROOT / "webui_handler.py").read_text(encoding="utf-8")

        self.assertIn('href="#/repair" data-route="repair"', html)
        self.assertIn('id="repairView"', html)
        self.assertIn('id="repairMount"', html)
        self.assertNotIn('id="repairRolloutSelect"', html)
        self.assertIn('"repair"', router)
        self.assertIn('/static/repair/page.js', loader)
        self.assertIn('/static/repair/synthetic-suffix.js', loader)
        self.assertIn('/static/repair/styles.css', loader)
        self.assertIn('/api/repair/synthetic-suffix/', handler)
        self.assertNotIn('/api/analysis/repair', handler)

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
        self.assertIn("@classmethod\n    def _dashboard_rows(", details)
        self.assertNotIn("AnalysisService.", details)
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
        module_names = [
            "charts.js",
            "model.js",
            "catalog.js",
            "view.js",
            "actions.js",
            "core.js",
            "layout.js",
            "run-config.js",
        ]
        for name in module_names:
            self.assertTrue((results_root / name).is_file(), name)
        for obsolete in ["results-layout.js", "results-run-config.js"]:
            self.assertFalse((STATIC_ROOT / obsolete).exists(), obsolete)

        charts = (results_root / "charts.js").read_text(encoding="utf-8")
        model = (results_root / "model.js").read_text(encoding="utf-8")
        catalog = (results_root / "catalog.js").read_text(encoding="utf-8")
        view = (results_root / "view.js").read_text(encoding="utf-8")
        actions = (results_root / "actions.js").read_text(encoding="utf-8")
        core = (results_root / "core.js").read_text(encoding="utf-8")
        layout = (results_root / "layout.js").read_text(encoding="utf-8")
        config = (results_root / "run-config.js").read_text(encoding="utf-8")
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

        self.assertIn("renderSignalChart", charts)
        self.assertIn("function sampleOutputText(", model)
        self.assertIn("var evaluationSampleIndexCache", model)
        self.assertIn("function evaluationSampleIndex(", model)
        self.assertIn("while (high - low > 1)", model)
        self.assertNotIn("return samples.reduce(function (nearest, sample)", model)
        self.assertIn("plot.dataset.playheadFrame", model)
        self.assertIn("if (plot.dataset.playheadFrame === frameKey) return;", model)
        self.assertIn("function loadBaselineRunCatalog(", catalog)
        self.assertIn("function resultsOnPersistentJobChanged(", catalog)
        self.assertIn("LF3RResultsCharts.renderSignalChart", view)
        self.assertIn("function renderEvaluationPanel(", view)
        self.assertIn('if (output.textContent !== nextText) output.textContent = nextText;', view)
        self.assertIn('if (output.hidden !== nextHidden) output.hidden = nextHidden;', view)
        self.assertIn('if (output.classList.contains("has-current-numeric-values")) return;', view)
        self.assertIn("function runPosthocLocalization(", actions)
        self.assertIn("function loadEvaluation(", core)
        self.assertIn("function bindResultsEvents(", core)
        self.assertIn("lf3rResultsLayoutRefresh", view)
        self.assertNotIn("function renderEvaluationPanel(", core)
        self.assertNotIn("function loadBaselineRunCatalog(", core)
        self.assertLess(len(core), 8000)
        self.assertNotIn("MutationObserver", layout)
        self.assertNotIn("MutationObserver", config)
        self.assertIn("document.createDocumentFragment()", layout)
        self.assertIn("output.parentNode.appendChild(fragment)", layout)
        self.assertNotIn("output.parentNode.appendChild(measurer)", layout)

        ordered = [
            "/static/results/charts.js",
            "/static/results/model.js",
            "/static/results/catalog.js",
            "/static/results/view.js",
            "/static/results/actions.js",
            "/static/results/core.js",
        ]
        for left, right in zip(ordered, ordered[1:]):
            self.assertLess(html.index(left), html.index(right))


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
        runs_baseline = (STATIC_ROOT / "runs" / "baseline.js").read_text(encoding="utf-8")
        run_config = (STATIC_ROOT / "results" / "run-config.js").read_text(encoding="utf-8")
        workspace = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")

        self.assertIn("applyOptions: applyOptions", source)
        self.assertNotIn("MutationObserver", source)
        self.assertNotIn("window.fetch =", source)
        self.assertIn('LF3RProcvlmMode.applyOptions(batchOptions, "batch")', runs_baseline)
        self.assertIn('LF3RProcvlmMode.applyOptions(collected.options, "single")', run_config)
        self.assertIn('LF3RProcvlmMode.refreshSingle(method)', run_config)
        self.assertIn("/static/baselines/procvlm.js", workspace)
        self.assertNotIn("procvlm-mode-ui.js", workspace)
        self.assertFalse((STATIC_ROOT / "procvlm-mode-ui.js").exists())

    def test_dataset_scope_frontend_is_explicit(self) -> None:
        controller = STATIC_ROOT / "runs" / "scope.js"
        self.assertTrue(controller.is_file())
        source = controller.read_text(encoding="utf-8")
        runs_baseline_selection = (STATIC_ROOT / "runs" / "baseline-selection.js").read_text(encoding="utf-8")
        app_jobs = (STATIC_ROOT / "app" / "jobs.js").read_text(encoding="utf-8")
        app_help = (STATIC_ROOT / "app" / "help.js").read_text(encoding="utf-8")
        analysis_live = (STATIC_ROOT / "analysis" / "live.js").read_text(encoding="utf-8")
        workspace_router = (STATIC_ROOT / "workspace" / "router.js").read_text(encoding="utf-8")
        workspace = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")

        self.assertIn("window.LF3RDatasetScopes", source)
        self.assertIn("matchesBaseline: matchesBaseline", source)
        self.assertIn("matchesPartition: matchesPartition", source)
        self.assertIn("scopeLabel: scopeLabel", source)
        self.assertIn("decorateHelp: decorateHelp", source)
        self.assertIn('"analysisOutcomeRunScope"', source)
        self.assertIn('"analysisHopScope"', source)
        self.assertNotIn('"analysisRunScope"', source)
        self.assertNotIn("window.baselineBatchMatchesScope =", source)
        self.assertNotIn("window.workspacePartitionMatches =", source)
        self.assertNotIn("window.persistentJobScope =", source)
        self.assertNotIn("window.cliHelpEntry =", source)

        self.assertIn("LF3RDatasetScopes.matchesBaseline", runs_baseline_selection)
        self.assertIn("LF3RDatasetScopes.scopeLabel", app_jobs)
        self.assertIn("LF3RDatasetScopes.decorateHelp", app_help)
        self.assertIn("LF3RDatasetScopes.matchesPartition", analysis_live)
        self.assertIn("LF3RDatasetScopes.refresh", workspace_router)
        self.assertIn("lf3rOutcomeCoverageChanged", workspace_router)
        self.assertIn("/static/runs/scope.js", workspace)

        outcome_ui = (STATIC_ROOT / "analysis-outcome.js").read_text(encoding="utf-8")
        self.assertIn("labels[row.method]", outcome_ui)
        self.assertNotIn("labels[row.baseline]", outcome_ui)
        self.assertIn("available_scope_rollouts", outcome_ui)
        self.assertIn("scope_population", outcome_ui)

        for obsolete in ["runs-semantics.js", "dataset-scope-ui.js"]:
            self.assertFalse((STATIC_ROOT / obsolete).exists(), obsolete)

    def test_runs_core_is_extracted_from_app_shell(self) -> None:
        runs_root = STATIC_ROOT / "runs"
        runs_core = runs_root / "core.js"
        runs_baseline_selection = runs_root / "baseline-selection.js"
        runs_baseline = runs_root / "baseline.js"
        runs_rollout = runs_root / "rollout.js"
        runs_layout = runs_root / "layout.js"
        tools_layout = runs_root / "tools-layout.js"
        gpu = runs_root / "gpu.js"
        project_tool_client = runs_root / "project-tool-client.js"
        project_tool_actions = runs_root / "project-tool-actions.js"
        manifest_tools = runs_root / "manifest-tools.js"
        project_tools = runs_root / "project-tools.js"
        for path in [
            runs_core,
            runs_baseline_selection,
            runs_baseline,
            runs_rollout,
            runs_layout,
            tools_layout,
            gpu,
            project_tool_client,
            project_tool_actions,
            manifest_tools,
            project_tools,
        ]:
            self.assertTrue(path.is_file(), path.name)
        self.assertFalse((STATIC_ROOT / "runs-layout.js").exists())

        core_source = runs_core.read_text(encoding="utf-8")
        baseline_selection_source = runs_baseline_selection.read_text(encoding="utf-8")
        baseline_source = runs_baseline.read_text(encoding="utf-8")
        rollout_source = runs_rollout.read_text(encoding="utf-8")
        layout_source = runs_layout.read_text(encoding="utf-8")
        tools_layout_source = tools_layout.read_text(encoding="utf-8")
        gpu_source = gpu.read_text(encoding="utf-8")
        project_tool_client_source = project_tool_client.read_text(encoding="utf-8")
        project_tool_actions_source = project_tool_actions.read_text(encoding="utf-8")
        manifest_tools_source = manifest_tools.read_text(encoding="utf-8")
        project_tools_source = project_tools.read_text(encoding="utf-8")
        app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        loader = (STATIC_ROOT / "workspace.js").read_text(encoding="utf-8")

        self.assertIn("function updateBaselineBatchSelection(", baseline_selection_source)
        self.assertIn("function baselineBatchWorkerSummary(", baseline_selection_source)
        self.assertIn("function startBaselineBatch(", baseline_source)
        self.assertIn("function startRolloutGeneration(", rollout_source)
        self.assertIn("function rolloutGenerationJobMessage(", rollout_source)
        self.assertNotIn("function updateBaselineBatchSelection(", baseline_source)
        self.assertNotIn("function startBaselineBatch(", core_source)
        self.assertNotIn("function startRolloutGeneration(", core_source)
        self.assertLess(len(core_source), 2000)
        self.assertLess(len(baseline_source), 10000)
        self.assertNotIn("function startBaselineBatch(", app)
        self.assertNotIn("function startRolloutGeneration(", app)

        ordered_core = [
            "/static/runs/baseline-selection.js",
            "/static/runs/baseline.js",
            "/static/runs/rollout.js",
            "/static/runs/core.js",
            "/static/app.js",
        ]
        for left, right in zip(ordered_core, ordered_core[1:]):
            self.assertLess(html.index(left), html.index(right))

        self.assertLess(len(layout_source), 12000)
        self.assertIn("LF3RRunsToolsLayout.createPanels", layout_source)
        self.assertIn("LF3RRunsGpu.createStrip", layout_source)
        self.assertIn("LF3RProjectTools.install", layout_source)
        self.assertIn("window.LF3RRunsToolsLayout", tools_layout_source)
        self.assertIn("window.LF3RRunsGpu", gpu_source)
        self.assertIn("window.LF3RProjectToolClient", project_tool_client_source)
        self.assertIn("recoverSubmission", project_tool_client_source)
        self.assertIn("AbortController", project_tool_client_source)
        self.assertIn("immediate-registry-v1", project_tool_client_source)
        self.assertIn("lastRegistryRefreshAt", project_tool_client_source)
        self.assertIn("Date.now() - lastRegistryRefreshAt >= 5000", project_tool_client_source)
        self.assertIn("window.LF3RProjectToolActions", project_tool_actions_source)
        self.assertIn('client.submit("safe_prepare"', project_tool_actions_source)
        self.assertIn('client.submit("robo_interval_sweep"', project_tool_actions_source)
        self.assertIn("Array.isArray(state.rollouts)", project_tool_actions_source)
        self.assertIn("renderRolloutOptions(cached)", project_tool_actions_source)
        self.assertIn("window.LF3RManifestTools", manifest_tools_source)
        self.assertIn("loadBatchManifestOptions", manifest_tools_source)
        self.assertIn("Array.isArray(state.manifests)", manifest_tools_source)
        self.assertIn("if (!forceNetwork && cached.length)", manifest_tools_source)
        self.assertIn("await loadBatchManifestOptions(true, true);", manifest_tools_source)
        self.assertIn('client.submit("rebuild_manifest"', manifest_tools_source)
        self.assertIn("window.LF3RProjectTools", project_tools_source)
        self.assertIn("actionsModule.install(client)", project_tools_source)
        self.assertIn("manifestModule.install(client", project_tools_source)
        self.assertIn("client.install({", project_tools_source)
        self.assertNotIn('fetch("/api/tools/run"', project_tools_source)
        self.assertLess(len(project_tools_source), 3000)
        self.assertIn("safePrepareRun", tools_layout_source)
        self.assertIn("refreshGpuStatus", gpu_source)
        self.assertNotIn("styles-tools.css?v=", layout_source)

        ordered_layout = [
            "/static/runs/tools-layout.js",
            "/static/runs/gpu.js",
            "/static/runs/project-tool-client.js",
            "/static/runs/project-tool-actions.js",
            "/static/runs/manifest-tools.js",
            "/static/runs/project-tools.js",
            "/static/runs/layout.js",
        ]
        for left, right in zip(ordered_layout, ordered_layout[1:]):
            self.assertLess(loader.index(left), loader.index(right))


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


    def test_app_shell_and_annotate_frontend_are_split(self) -> None:
        app_shell = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        app_jobs = (STATIC_ROOT / "app" / "jobs.js").read_text(encoding="utf-8")
        app_help = (STATIC_ROOT / "app" / "help.js").read_text(encoding="utf-8")
        annotate_catalog = (STATIC_ROOT / "annotate" / "catalog.js").read_text(encoding="utf-8")
        annotate_failure_events = (STATIC_ROOT / "annotate" / "failure-events.js").read_text(encoding="utf-8")
        annotate_timeline = (STATIC_ROOT / "annotate" / "timeline.js").read_text(encoding="utf-8")
        annotate_core = (STATIC_ROOT / "annotate" / "core.js").read_text(encoding="utf-8")
        annotate_events = (STATIC_ROOT / "annotate" / "events.js").read_text(encoding="utf-8")
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

        self.assertLess(len(app_shell), 8000)
        self.assertIn("var state = {", app_shell)
        self.assertIn("installEvents();", app_shell)
        self.assertIn("window.lf3rInitialRolloutsPromise = loadRollouts();", app_shell)
        self.assertIn("window.lf3rInitialRolloutsPromise.catch", app_shell)

        self.assertIn("function renderPersistentJobCards(", app_jobs)
        self.assertIn("function pollPersistentJob(", app_jobs)
        self.assertIn("function pollBaselineJob(", app_jobs)
        self.assertIn("var cliHelpState = {", app_help)
        self.assertIn("function showCliHelp(", app_help)

        self.assertIn('var INSTRUCTION_CONDITION_ORDER = ["full_instruction", "subtask_a", "subtask_b"];', annotate_catalog)
        self.assertIn("function loadRollouts(", annotate_catalog)
        self.assertIn("function renderRolloutList(", annotate_catalog)
        self.assertIn("function updateRolloutListSelection(id)", annotate_catalog)
        self.assertIn("updateRolloutListSelection(id);", annotate_catalog)
        self.assertNotIn('button.addEventListener("click", function () {\n      maybeSelectRollout', annotate_catalog)
        self.assertIn("var failureTypeChoices = [", annotate_failure_events)
        self.assertIn("function renderFailureEvents(", annotate_failure_events)
        self.assertIn("function setActiveEventFrame(", annotate_failure_events)
        self.assertIn("var TIMELINE_MARKER_DEFINITIONS = [", annotate_timeline)
        self.assertIn("function seekFrame(", annotate_timeline)
        self.assertIn("function renderTimelineMarkers(", annotate_timeline)
        self.assertIn("function saveAnnotation(", annotate_core)
        self.assertIn("function installEvents(", annotate_events)
        self.assertIn('rolloutList.addEventListener("click"', annotate_events)
        self.assertIn('node.dataset.primaryFilterHook = "true"', annotate_events)
        self.assertIn("filterRenderPending", annotate_events)
        self.assertIn("requestAnimationFrame", annotate_events)
        self.assertLess(len(annotate_core), 8000)

        self.assertNotIn("function loadRollouts(", annotate_core)
        self.assertNotIn("function renderFailureEvents(", annotate_core)
        self.assertNotIn("function seekFrame(", annotate_core)
        self.assertNotIn("function loadRollouts(", app_shell)
        self.assertNotIn("function installEvents(", app_shell)
        self.assertNotIn("function persistentJobEndpoint(", app_shell)
        self.assertNotIn("function startBaselineRun(", app_shell)
        self.assertNotIn("function startBaselineRun(", app_jobs)
        self.assertNotIn("function startBaselineRun(", annotate_core)

        ordered = [
            "/static/app/jobs.js",
            "/static/app/help.js",
            "/static/annotate/catalog.js",
            "/static/annotate/failure-events.js",
            "/static/annotate/timeline.js",
            "/static/annotate/core.js",
            "/static/annotate/events.js",
            "/static/app.js",
        ]
        for left, right in zip(ordered, ordered[1:]):
            self.assertLess(html.index(left), html.index(right))


    def test_manifest_support_reuses_initial_rollout_metadata(self) -> None:
        application = (TOOL_ROOT / "webui_application.py").read_text(encoding="utf-8")
        manifest = (STATIC_ROOT / "manifest-support.js").read_text(encoding="utf-8")
        catalog = (STATIC_ROOT / "annotate" / "catalog.js").read_text(encoding="utf-8")
        self.assertNotIn("primary_manifest_path", application)
        self.assertNotIn("manifest_primary", application)
        self.assertNotIn("manifest_primary", manifest)
        self.assertNotIn("manifest_primary", catalog)
        self.assertIn('if (origin === "real_robot") return isRealRobotManifest(record);', catalog)
        self.assertIn('sourceKind.indexOf("realrobot") === 0', catalog)
        self.assertIn('if (origin === "real_robot") return isRealRobotRecord(record);', manifest)
        self.assertIn('sourceKind.indexOf("realrobot") === 0', manifest)
        self.assertIn("window.lf3rInitialRolloutsPromise", manifest)
        dashboard = (STATIC_ROOT / "analysis" / "dashboard.js").read_text(encoding="utf-8")
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("workspaceDashboardRenderRolloutOutcome", dashboard)
        self.assertIn("Failure recall", dashboard)
        self.assertIn("Balanced acc.", dashboard)
        self.assertIn('id="analysisRolloutOutcomeTable"', html)
        self.assertIn("Array.isArray(state.manifests)", manifest)
        self.assertIn("if (!manifests.length)", manifest)
        self.assertIn('fetch("/api/manifests"', manifest)
        self.assertIn("function initializeManifestMetadata()", manifest)
        self.assertIn("bootstrap.then(", manifest)

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
