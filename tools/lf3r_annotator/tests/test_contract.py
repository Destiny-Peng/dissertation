from __future__ import annotations

import csv
import json
import math
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOL_ROOT = Path(__file__).resolve().parents[1]


class DatasetAndFrontendContractTest(unittest.TestCase):
    def test_manifest_keeps_controlled_data_separate(self) -> None:
        manifest = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
        records = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertGreater(len(records), 0)
        for record in records:
            if record["source_kind"] == "controlled_injected":
                self.assertEqual(record["analysis_partition"], "controlled_analysis")
                self.assertEqual(record["dataset_role"], "controlled_analysis")
                self.assertIsInstance(record["injection"], dict)
                self.assertIn("causal_onset_frame", record["injection"])
            else:
                self.assertEqual(record["source_kind"], "natural_policy")
                self.assertEqual(record["analysis_partition"], "natural_observation")
                self.assertIsNone(record["injection"])

    def test_frontend_exposes_required_annotation_controls(self) -> None:
        html = (TOOL_ROOT / "static/index.html").read_text(encoding="utf-8")
        app_javascript = (TOOL_ROOT / "static/app.js").read_text(encoding="utf-8")
        app_jobs = (TOOL_ROOT / "static/app/jobs.js").read_text(encoding="utf-8")
        app_help = (TOOL_ROOT / "static/app/help.js").read_text(encoding="utf-8")
        annotate_catalog = (TOOL_ROOT / "static/annotate/catalog.js").read_text(encoding="utf-8")
        annotate_failure_events = (TOOL_ROOT / "static/annotate/failure-events.js").read_text(encoding="utf-8")
        annotate_timeline = (TOOL_ROOT / "static/annotate/timeline.js").read_text(encoding="utf-8")
        annotate_core = (TOOL_ROOT / "static/annotate/core.js").read_text(encoding="utf-8")
        annotate_events = (TOOL_ROOT / "static/annotate/events.js").read_text(encoding="utf-8")
        results_core = (TOOL_ROOT / "static/results/core.js").read_text(encoding="utf-8")
        results_charts = (TOOL_ROOT / "static/results/charts.js").read_text(encoding="utf-8")
        results_model = (TOOL_ROOT / "static/results/model.js").read_text(encoding="utf-8")
        results_catalog = (TOOL_ROOT / "static/results/catalog.js").read_text(encoding="utf-8")
        results_view = (TOOL_ROOT / "static/results/view.js").read_text(encoding="utf-8")
        results_actions = (TOOL_ROOT / "static/results/actions.js").read_text(encoding="utf-8")
        runs_baseline_selection = (TOOL_ROOT / "static/runs/baseline-selection.js").read_text(encoding="utf-8")
        runs_baseline = (TOOL_ROOT / "static/runs/baseline.js").read_text(encoding="utf-8")
        runs_rollout = (TOOL_ROOT / "static/runs/rollout.js").read_text(encoding="utf-8")
        runs_javascript = (TOOL_ROOT / "static/runs/core.js").read_text(encoding="utf-8")
        javascript = "\n".join([
            app_javascript,
            app_jobs,
            app_help,
            annotate_catalog,
            annotate_failure_events,
            annotate_timeline,
            annotate_core,
            annotate_events,
            results_core,
            results_charts,
            results_model,
            results_catalog,
            results_view,
            results_actions,
            runs_baseline_selection,
            runs_baseline,
            runs_rollout,
            runs_javascript,
        ])
        workspace = (TOOL_ROOT / "static/workspace.js").read_text(encoding="utf-8")
        frontend_javascript = javascript + "\n" + workspace
        for element_id in [
            "rolloutVideo",
            "frameSlider",
            "failureType",
            "failureEvents",
            "addFailureEvent",
            "saveButton",
            "evaluationMethods",
            "evaluationStatus",
            "instructionVariantControl",
            "instructionCondition",
            "instructionVariantNote",
            "reloadEvaluation",
            "baselineBatchForm",
            "baselineBatchMethod",
            "baselineBatchScope",
            "baselineBatchCondition",
            "baselineBatchResultFilter",
            "baselineBatchGpu",
            "baselineBatchMemoryUtilization",
            "baselineBatchStartIndex",
            "baselineBatchLimit",
            "baselineBatchEndIndex",
            "baselineBatchRynnParallel",
            "baselineBatchWorkers",
            "baselineBatchAddWorker",
            "baselineBatchWorkerSummary",
            "baselineBatchRun",
            "baselineBatchStatus",
            "baselineBatchJobs",
            "baselineBatchLog",
            "rolloutGenerationForm",
            "rolloutGenerationSuite",
            "rolloutGenerationRenderResolution",
            "rolloutGenerationRecordResolution",
            "rolloutGenerationVideoViewMode",
            "rolloutGenerationGpu",
            "rolloutGenerationTaskStart",
            "rolloutGenerationTaskEnd",
            "rolloutGenerationTrials",
            "rolloutGenerationSeed",
            "rolloutGenerationLabel",
            "rolloutGenerationLogSafeFeatures",
            "rolloutGenerationSelection",
            "rolloutGenerationRun",
            "rolloutGenerationBadge",
            "rolloutGenerationStatus",
            "rolloutGenerationJobs",
            "rolloutGenerationLog",
        ]:
            self.assertIn('id="' + element_id + '"', html)
        self.assertIn("/api/videos/", javascript)
        self.assertIn("/api/annotations/", javascript)
        self.assertNotIn('id="baselineGpu"', html)
        self.assertNotIn('id="baselineMemoryUtilization"', html)
        self.assertIn("first_environment_timestep", javascript)
        self.assertIn("sessionStorage", javascript)
        for endpoint in ["/api/baselines/", "/api/baselines/run/", "/api/baselines/run-batch", "/api/baselines/posthoc-localization/", "/api/baseline-jobs/", "/api/baselines/runs", "/api/baselines/result-coverage", "/api/rollouts/generate", "/api/rollout-jobs/", "/api/jobs"]:
            self.assertIn(endpoint, frontend_javascript)
        for marker in ["model_output", "renderSignalChart", "loadEvaluation", "data-run-baseline", "parameter_help.json", "cliHelpPopover", "startRolloutGeneration", "pollRolloutGenerationJob", "loadPersistentJobs", "persistentJobPollTimers", "latestPersistentJob", "latestGeneration", "tmux_session", "baselineBatchRebalanceWorkers", "worker-spec", "persistentWorkerSummary", "rolloutGenerationSuite", "rolloutGenerationRenderResolution", "rolloutGenerationRecordResolution", "rolloutGenerationVideoViewMode", "render_resolution", "record_resolution", "video_view_mode", "libero_three_view", "task_suite", "libero_spatial", "renderResolution", "recordResolution", "instruction_variants", "instructionCondition", "baselineBatchCondition", "baselineBatchResultFilter", "instruction_condition", "result_filter", "missing_valid", "run_source_rollout_ids", "variant baseline outputs", "condition_label", "data-evaluation-run-select", "data-apply-baseline-run", "baselineRunAll", "baselineRunSelections", "loadBaselineRunCatalog", "Automatic · newest available", "Apply to all", "run_", "relative_value", "relative temporal displacement"]:
            self.assertIn(marker, javascript)

        for marker in [
            "TIMELINE_MARKER_DEFINITIONS",
            "collectTimelineMarkers",
            "evaluation-onset-pin",
            "data-evaluation-onset-markers",
            "data-evaluation-signal-toggle",
            "data-evaluation-signal-path",
            "evaluationSignalVisibility",
            "click labels to show/hide",
            "frameDomainMax",
            "evaluation-current-body",
            "textEntries",
            "video frames 0-",
        ]:
            self.assertIn(marker, javascript)

        self.assertIn("var width = 100;", javascript)
        self.assertNotIn("var width = 760;", workspace)
        self.assertIn("analysis-threshold-table", workspace)
        self.assertIn("onset_signal_statistics", workspace)
        self.assertIn("n_onset_events", workspace)
        self.assertIn("direction_consistency_fraction_all", workspace)
        self.assertIn("threshold-direction", workspace)
        self.assertIn("preserveAspectRatio", javascript)
        self.assertIn("evaluation-chart-plot", javascript)
        self.assertIn("failure_events", javascript)
        self.assertIn("recovery_frame", javascript)
        self.assertIn("data-event-field", javascript)
        self.assertIn("recovered_success", html)
        self.assertIn("baseline-worker-panel", html)
        self.assertIn("baselineBatchUsesWorkers", javascript)
        self.assertIn("parallel_workers", javascript)
        self.assertIn("workers", javascript)
        self.assertIn("renderRepeatDetails", javascript)
        self.assertIn("Show all ", javascript)
        self.assertIn('data-batch-option="robo_localization_ckpt"', html)
        self.assertIn('data-batch-option="robo_camera_mode"', html)
        self.assertIn('value="multi_view"', html)
        self.assertIn("renderLocalizationPredictionMarker", javascript)
        self.assertIn("renderLocalizationPredictionCurve", javascript)
        self.assertIn("data-localization-curve", javascript)
        self.assertIn("sigmoid_scores", javascript)
        self.assertIn("Full checkpoint output curve", javascript)
        self.assertIn("renderPosthocLocalizationControls", javascript)
        self.assertIn("data-run-posthoc-localization", javascript)
        self.assertIn("data-toggle-baseline-card", javascript)
        self.assertIn("baselineCollapsed", javascript)
        self.assertIn("renderLocalizationPredictionSummary", javascript)
        for method in ["safe", "procvlm", "rynnvalue", "robo_dopamine", "densereward"]:
            self.assertIn('value="' + method + '"', html)

        styles = (TOOL_ROOT / "static/styles.css").read_text(encoding="utf-8")
        for marker in [
            ".evaluation-chart-plot",
            ".evaluation-chart-markers",
            ".evaluation-onset-pin",
            ".evaluation-signal-toggle",
            "--timeline-track-inset",
            "inset: 8px 0 18px",
            ".timeline-track",
            "clip-path: inset(0)",
            ".timeline-track",
            "clip-path: inset(0)",
            ".rynn-worker-row",
            ".persistent-job-workers",
            ".batch-resource-warning",
            ".evaluation-run-controls",
            ".evaluation-localization-summary",
            ".evaluation-localization-pin",
            ".localization-repeat-details",
            ".localization-repeat-table",
        ]:
            self.assertIn(marker, styles)

    def test_workspace_navigation_settings_and_analysis_contract(self) -> None:
        html = (TOOL_ROOT / "static/index.html").read_text(encoding="utf-8")
        workspace = "\n".join(
            (TOOL_ROOT / path).read_text(encoding="utf-8")
            for path in [
                "static/workspace.js",
                "static/workspace/settings.js",
                "static/workspace/router.js",
                "static/workspace/events.js",
                "static/workspace-core.js",
                "static/analysis/live.js",
                "static/analysis/snapshot.js",
                "static/analysis/localization.js",
                "static/analysis/change-point.js",
                "static/analysis/event-triggered.js",
                "static/analysis/runs.js",
                "static/analysis/dashboard.js",
                "static/analysis/signals.js",
                "static/analysis/details.js",
            ]
        )
        hop_analysis = (TOOL_ROOT / "static/analysis-robo-hop.js").read_text(encoding="utf-8")
        label_loss_analysis = (TOOL_ROOT / "static/analysis-robo-label-loss.js").read_text(encoding="utf-8")
        server = "\n".join(
            (TOOL_ROOT / name).read_text(encoding="utf-8")
            for name in [
                "server.py",
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
            ]
        )
        styles = (TOOL_ROOT / "static/styles.css").read_text(encoding="utf-8")

        for route in ["#/review", "#/analysis", "#/settings"]:
            self.assertIn('href="' + route + '"', html)
        for element_id in [
            "pageTitle",
            "analysisView",
            "analysisStatus",
            "analysisKpis",
            "analysisPartitionFilter",
            "analysisSuiteFilter",
            "analysisTaskFilter",
            "analysisOutcomeFilter",
            "analysisOutcomeChart",
            "analysisFailureTypeChart",
            "analysisTimingChart",
            "analysisTaskChart",
            "analysisRunForm",
            "analysisRunScope",
            "analysisRunSafe",
            "analysisRunProcvlm",
            "analysisRunRynnvalue",
            "analysisRunRoboDopamine",
            "analysisPreWindow",
            "analysisPostWindow",
            "analysisBackgroundStride",
            "analysisOutputLabel",
            "analysisRunButton",
            "analysisRunBadge",
            "analysisEnvironmentStatus",
            "analysisRunLog",
            "analysisRunJobs",
            "analysisHopForm",
            "analysisHopScope",
            "analysisHopRun",
            "analysisHopOutputLabel",
            "analysisHopTaskCv",
            "analysisHopRunButton",
            "analysisHopBadge",
            "analysisHopSelection",
            "analysisHopLog",
            "analysisHopResults",
            "analysisHopArtifacts",
            "analysisLabelLossForm",
            "analysisLabelLossDevice",
            "analysisLabelLossTauEvent",
            "analysisLabelLossRepeats",
            "analysisLabelLossEpochs",
            "analysisLabelLossPatience",
            "analysisLabelLossLearningRate",
            "analysisLabelLossWeightDecay",
            "analysisLabelLossGradClip",
            "analysisLabelLossDistanceWeight",
            "analysisLabelLossRankingWeight",
            "analysisLabelLossRankingMargin",
            "analysisLabelLossAsymmetric",
            "analysisLabelLossOutputLabel",
            "analysisLabelLossRunButton",
            "analysisLabelLossBadge",
            "analysisLabelLossSelection",
            "analysisLabelLossLog",
            "analysisLabelLossResults",
            "analysisLabelLossArtifacts",
            "analysisSnapshotMethod",
            "analysisSnapshotOutcome",
            "analysisSnapshotTask",
            "analysisCoverageChart",
            "analysisThresholdChart",
            "analysisResponseChart",
            "analysisPersistenceChart",
            "analysisRecoveryChart",
            "analysisAnomalies",
            "analysisLocalizationMethod",
            "analysisLocalizationSignal",
            "analysisLocalizationThreshold",
            "analysisLocalizationOutcome",
            "analysisLocalizationTask",
            "analysisLocalizationRecallChart",
            "analysisLocalizationErrorChart",
            "analysisLocalizationFailureType",
            "analysisLocalizationEvents",
            "analysisLocalizationProvenance",
            "analysisChangePointMethod",
            "analysisChangePointSignal",
            "analysisChangePointFeature",
            "analysisChangePointScale",
            "analysisChangePointThreshold",
            "analysisChangePointOutcome",
            "analysisChangePointTask",
            "analysisChangePointRecallChart",
            "analysisChangePointErrorChart",
            "analysisChangePointFailureType",
            "analysisChangePointComparison",
            "analysisChangePointEvents",
            "analysisChangePointProvenance",
            "analysisEventTriggeredMethod",
            "analysisEventTriggeredSignal",
            "analysisEventTriggeredGroup",
            "analysisEventTriggeredScale",
            "analysisEventTriggeredBadge",
            "analysisEventTriggeredStatus",
            "analysisEventTriggeredProvenance",
            "analysisEventTriggeredSignalChart",
            "analysisEventTriggeredChangeChart",
            "analysisEventTriggeredSummary",
            "analysisEventTriggeredPeaks",
            "settingsView",
            "settingsForm",
            "settingsBackground",
            "settingsSurface",
            "settingsRaised",
            "settingsControl",
            "settingsText",
            "settingsMuted",
            "settingsAccent",
            "settingsFontScale",
            "settingsFontScaleValue",
            "settingsReviewFontScale",
            "settingsReviewFontScaleValue",
            "settingsAnalysisFontScale",
            "settingsAnalysisFontScaleValue",
            "settingsControlFontScale",
            "settingsControlFontScaleValue",
            "settingsDensity",
            "settingsReset",
            "settingsSave",
        ]:
            self.assertIn('id="' + element_id + '"', html)
        self.assertIn('/static/workspace.js', html)
        self.assertIn('/static/analysis-robo-hop.js', workspace)
        self.assertIn('/static/analysis-robo-label-loss.js', workspace)
        for marker in [
            "/api/settings",
            "/api/analysis",
            "/api/baselines/runs",
            "/api/analysis/run",
            "/api/analysis-jobs/",
            "workspaceLoadBaselineRuns",
            "workspaceStartAnalysisRun",
            "workspaceRenderRoute",
            "workspaceLoadAnalysisEnvironment",
            "workspaceJobsChanged",
            "workspaceSelectedAnalysisRuns",
            "partial_compatible",
            "localization_event_metrics",
            "change_point",
            "event_triggered",
            "workspaceRenderChangePoint",
            "workspaceRenderEventTriggered",
            "workspaceRenderEventTriggeredSignalChart",
            "eventTriggered.curves",
            "eventTriggered.change_scores",
            "eventTriggered.summary",
            "data-analysis-rollout",
            "resolvedSuccessRate",
            "Observable coverage",
            "viewBox",
            '<title>',
            'aria-label=',
        ]:
            self.assertIn(marker, workspace)
        for marker in ["data-analysis-live-filter", "data-analysis-snapshot-filter", "data-analysis-localization-filter", "data-analysis-changepoint-filter", "data-analysis-event-triggered-filter", "comparison_with_full_136_20260827", "primary-analysis-card", "legacy-analysis-card", "event-triggered-card"]:
            self.assertIn(marker, html)
        self.assertIn('multiple size="5"', html)
        for endpoint in [
            'path == "/api/settings"', 'path == "/api/analysis"',
            'path == "/api/analysis/robo-label-loss"',
            'path == "/api/baselines/runs"', 'path == "/api/baselines/result-coverage"', '"/api/baselines/run-batch"',
            '"/api/analysis/run"', 'analysis-jobs', 'rollout-jobs', '"/api/rollouts/generate"', '"/api/jobs"', 'worker_assignments', 'parallel_workers', 'CHANGEPOINT_TABLE_FILES', 'changepoint_summary.csv', 'comparison_with_full_136_20260827', 'primary_analysis_type', 'event_triggered_available', 'EVENT_TRIGGERED_TABLE_FILES', 'event_triggered_curves.csv', 'os.replace(temp_name, self.path)'
        ]:
            self.assertIn(endpoint, server)
        for marker in [
            "run_rollout_ids",
            "partial_compatible",
            "_complete_annotation_records",
            "_valid_result_rollout_ids",
            "complete_annotation_rollouts",
            "incomplete_annotation_rollouts",
            "BASELINE_RESULT_FILTERS",
            "missing_valid",
            "localization_event_metrics",
            "localization_summary",
            "ROBO_HOP_REQUIRED_FILES",
            "ROBO_HOP_EXTENDED_FILES",
            "ROBO_LABEL_LOSS_REQUIRED_FILES",
            "start_robo_label_loss_run",
            "robo_bilstm_label_loss_ablation",
            "robo_label_loss_response",
            "robo_label_loss_artifact_path",
            "start_robo_hop_run",
            "robo_hop_comparison",
            "fused_scope_rollout_count",
            "sweep_summary.csv",
            "best_configs.csv",
            "no_event_failure_results.csv",
            "ensemble_sweep.csv",
            "ensemble_selected.csv",
            "ensemble_by_failure_type.csv",
            "interval_localization_ranking.csv",
            '"interval_localization_rows": interval_localization_rows',
            "grasp_event_features.csv",
            "grasp_detected_vs_missed.csv",
            "grasp_matched_control.csv",
            "grasp_failure_categories.csv",
            "grasp_event_heatmap.png",
            "cpu_limit",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            '"search_cache": metadata.get("search_cache") or {}',
        ]:
            self.assertIn(marker, server)
        self.assertNotIn("window.workspaceLoadBaselineRuns", hop_analysis)
        self.assertIn("/api/baselines/runs?scope=", hop_analysis)
        for marker in [
            'analysis_kind: "robo_hop_comparison"',
            "/api/analysis/run",
            "analysisHopRunButton",
            "analysisHopCpuLimit",
            "cpu_limit: cpuLimit",
            "fused_scope_rollout_count",
            "Failed-rollout interval localization",
            "analysisHopIntervalRankPopulation",
            "analysisHopIntervalRankSort",
            "analysisHopIntervalRankDirection",
            "analysisHopIntervalRankLimit",
            "sortedRankingRows",
            "rankingControlsHtml",
            "Stagnation OR regression",
            "localizationFamilyLabel",
            "localizationConfigLabel",
            "stagnation: ",
            "regression: ",
            "in_interval_rate",
            "median_signed_interval_error_samples",
            "trigger_coverage",
            "within_3",
            "search cache HIT",
            "reused prior search",
            "search computed",
            "workspaceLoadAnalysis(true)",
        ]:
            self.assertIn(marker, hop_analysis)
        for marker in [
            'analysis_kind: "robo_bilstm_label_loss_ablation"',
            "/api/analysis/run",
            "/api/analysis/robo-label-loss",
            "analysisLabelLossRunButton",
            "analysisLabelLossTauEvent",
            "analysisLabelLossDistanceWeight",
            "analysisLabelLossRankingWeight",
            "analysisLabelLossRankingMargin",
            "label_ablation",
            "loss_ablation",
            "first_event_in_interval_rate_mean",
            "pseudo_no_event_frame0_rollout_n",
            "recoverLatestJob",
            "fetchJson",
            "readAnalysisJob",
            "/api/jobs?job_type=analysis",
            "returned non-JSON",
        ]:
            self.assertIn(marker, label_loss_analysis)
        self.assertNotIn("workspaceLoadBaselineRuns", label_loss_analysis)
        self.assertNotIn("modeOrder", hop_analysis)
        self.assertNotIn("pairwise_overlap", hop_analysis)
        for obsolete in [
            "analysisHopPointRank",
            "unconstrained_localization",
            "progress_peak_localization",
            "Earliest global progress maximum",
            "oracle_recall_at_",
            "FPR-unconstrained oracle",
            "event_recall_at_",
            "grasp_recall_at_",
            "grasp_gain_vs_best_",
            "overall_failed_rollout_coverage",
            "event_median_delay_samples",
            "tp_jaccard",
        ]:
            self.assertNotIn(obsolete, hop_analysis)
        baseline_module = (TOOL_ROOT / "webui_baseline.py").read_text(encoding="utf-8")
        self.assertIn('rollout_ids is not None', baseline_module)
        for marker in [
            "--font-scale",
            "--review-font-scale",
            "--analysis-font-scale",
            "--control-font-scale",
            "--accent",
            "font-size: calc(100% * var(--font-scale)",
            ".analysis-threshold-table",
            ".cli-help-popover",
            ".rollout-generation-grid",
            ".persistent-job-card",
            ".persistent-job-workers",
            ".rynn-worker-row",
            ".analysis-run-field-help",
            ".localization-heat-table",
            ".page-navigation",
            ".analysis-grid",
            ".settings-color-grid",
            "--causal:",
            "--observable:",
            "--terminal:",
            "--recovery:",
        ]:
            self.assertIn(marker, styles)
        self.assertNotRegex(styles, r"font-size:\s*[0-9]+px")
        self.assertNotRegex(styles, r"font:\s*[0-9]+px")
        help_path = TOOL_ROOT / "static/parameter_help.json"
        help_data = json.loads(help_path.read_text(encoding="utf-8"))
        for section, keys in {
            "baseline": ["gpu", "vllm_free_memory_fraction", "start_index", "end_index", "limit", "parallel_workers", "worker_spec", "result_filter", "procvlm_window_size", "procvlm_frame_stride", "rynn_num_frames", "rynn_evaluation_interval", "robo_eval_mode", "robo_camera_mode", "densereward_frame_interval", "densereward_max_new_tokens"],
            "rollout": ["task_suite", "gpu", "task_start", "task_end", "trials", "seed", "run_note", "log_safe_features", "render_resolution", "record_resolution", "video_view_mode"],
            "settings": ["font_scale", "review_font_scale", "analysis_font_scale", "control_font_scale"],
        }.items():
            for key in keys:
                self.assertIn(key, help_data[section])
                self.assertIn("description", help_data[section][key])
                self.assertIn("default", help_data[section][key])
        self.assertEqual(help_data["baseline"]["vllm_free_memory_fraction"]["cli"], "--vllm-free-memory-fraction VALUE")
        self.assertIn("--gpu_memory_utilization", help_data["baseline"]["vllm_free_memory_fraction"]["forwarded_as"])
        self.assertEqual(help_data["rollout"]["task_end"]["cli"], "--task-end N")
        self.assertEqual(help_data["rollout"]["task_suite"]["cli"], "--task-suite {libero_10,libero_spatial}")
        self.assertEqual(help_data["rollout"]["render_resolution"]["cli"], "--render-resolution N")
        self.assertEqual(help_data["rollout"]["record_resolution"]["cli"], "--record-resolution N")
        self.assertEqual(help_data["rollout"]["video_view_mode"]["cli"], "--video-view-mode MODE")
        self.assertEqual(help_data["baseline"]["robo_eval_mode"]["default"], "fused")
        self.assertEqual(help_data["baseline"]["robo_camera_mode"]["default"], "auto")
        self.assertEqual(help_data["baseline"]["densereward_frame_interval"]["default"], "1")
        self.assertIn('value="fused" selected', html)

        self.assertNotIn("margin: 8px -26px", styles)
        self.assertNotIn("margin: -11px var(--timeline-track-inset) 0", styles)
        self.assertNotIn("margin-inline: calc(var(--timeline-track-inset) - 1px)", styles)


    def test_analysis_dashboard_contract(self) -> None:
        html = (TOOL_ROOT / "static/index.html").read_text(encoding="utf-8")
        workspace = "\n".join(
            (TOOL_ROOT / path).read_text(encoding="utf-8")
            for path in [
                "static/workspace.js",
                "static/workspace/settings.js",
                "static/workspace/router.js",
                "static/workspace/events.js",
                "static/workspace-core.js",
                "static/analysis/live.js",
                "static/analysis/snapshot.js",
                "static/analysis/localization.js",
                "static/analysis/change-point.js",
                "static/analysis/event-triggered.js",
                "static/analysis/runs.js",
                "static/analysis/dashboard.js",
                "static/analysis/signals.js",
                "static/analysis/details.js",
            ]
        )
        hop_analysis = (TOOL_ROOT / "static/analysis-robo-hop.js").read_text(encoding="utf-8")
        styles = (TOOL_ROOT / "static/styles.css").read_text(encoding="utf-8")
        for route in [
            "#/analysis/overview",
            "#/analysis/comparison",
            "#/analysis/failures",
            "#/analysis/events",
            "#/analysis/signals",
            "#/analysis/archive",
        ]:
            self.assertIn('href="' + route + '"', html)
        for element_id in [
            "analysisTabOverview",
            "analysisTabComparison",
            "analysisTabFailures",
            "analysisTabEvents",
            "analysisTabSignals",
            "analysisTabArchive",
            "analysisFailureMetric",
            "analysisDetailsTable",
            "analysisDetailsPrevious",
            "analysisDetailsNext",
            "analysisArchiveLinks",
            "analysisHopForm",
            "analysisHopResults",
            "analysisHopArtifacts",
        ]:
            self.assertIn('id="' + element_id + '"', html)
        for marker in [
            "data-analysis-panel",
            "data-analysis-tab",
            "libero_10",
            "Q95 (primary)",
            "16 frames (default)",
            "/api/analysis/details",
            "/api/analysis/artifacts/",
            "analysis-horizontal-list",
            "analysis-heatmap-table",
            "analysis-details-table",
            "analysis-download-grid",
            "clipPath",
            "workspaceDashboardRenderSnapshot",
            "Failure detection and ensemble analysis",
            "analysisHopCpuLimit",
            "fused",
        ]:
            self.assertIn(marker, html + workspace + hop_analysis)
        self.assertNotIn('transform="rotate(', workspace)
        self.assertNotIn(".analysis-chart > .analysis-svg {" + chr(10) + "  min-width: 42rem", styles)
        self.assertNotIn(".analysis-chart-scroll > .analysis-svg {" + chr(10) + "  min-width: 42rem", styles)


    def test_rollout_generation_uses_egl_and_keeps_persistent_logs_open(self) -> None:
        for filename in [
            "generate_libero10_natural.sh",
            "generate_libero_spatial_native.sh",
        ]:
            script = (TOOL_ROOT / filename).read_text(encoding="utf-8")
            self.assertIn("MUJOCO_GL=egl", script)
            self.assertIn("PYOPENGL_PLATFORM=egl", script)
            self.assertNotIn("MUJOCO_GL=osmesa", script)
        javascript = "\n".join([
            (TOOL_ROOT / "static/app.js").read_text(encoding="utf-8"),
            (TOOL_ROOT / "static/app/jobs.js").read_text(encoding="utf-8"),
        ])
        self.assertIn("persistentJobLogOpen", javascript)
        self.assertIn("persistentJobLogText", javascript)
        self.assertIn('aria-expanded=\"false\"', javascript)
        self.assertIn('button.textContent = \"Hide log\"', javascript)
        self.assertNotIn("output.hidden = false;\n      window.setTimeout", javascript)


    def test_openvla_safe_feature_wrapper_contract(self) -> None:
        wrapper = (TOOL_ROOT / "run_openvla_libero10_natural.py").read_text(encoding="utf-8")
        helper = (TOOL_ROOT / "safe_feature_logging.py").read_text(encoding="utf-8")
        shell = (TOOL_ROOT / "generate_libero10_natural.sh").read_text(encoding="utf-8")
        readme = (TOOL_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("--log-safe-features", wrapper)
        self.assertIn("--render-resolution", wrapper)
        self.assertIn("--record-resolution", wrapper)
        self.assertIn("--video-view-mode", wrapper)
        self.assertIn("generate_libero_multiview.py", wrapper)
        self.assertIn("output_hidden_states=", wrapper)
        self.assertIn("postprocess_run(", wrapper)
        self.assertIn('generated_outputs["hidden_states"][token][-1][0, -1, :]', helper)
        self.assertIn("SAFE_HIDDEN_STATE_SHAPE = (SAFE_ACTION_TOKEN_COUNT, SAFE_HIDDEN_SIZE)", helper)
        self.assertIn(".safe_features.npz", helper)
        self.assertIn(".safe_features.json", helper)
        self.assertIn("--log-safe-features", shell)
        self.assertIn("--render-resolution", shell)
        self.assertIn("--record-resolution", shell)
        self.assertIn("--video-view-mode", shell)
        spatial_shell = (TOOL_ROOT / "generate_libero_spatial_native.sh").read_text(encoding="utf-8")
        self.assertIn("--log-safe-features", spatial_shell)
        self.assertIn("--video-view-mode", spatial_shell)
        robo_worker = (PROJECT_ROOT / "tools/baselines/robo_dopamine_persistent_worker.py").read_text(encoding="utf-8")
        robo_reader = (TOOL_ROOT / "baseline_readers.py").read_text(encoding="utf-8")
        self.assertIn('"frames"', robo_worker)
        self.assertIn('"logits"', robo_worker)
        self.assertIn('"sigmoid_scores"', robo_worker)
        self.assertIn('"frames"', robo_reader)
        self.assertIn('"logits"', robo_reader)
        self.assertIn('"sigmoid_scores"', robo_reader)
        multiview = (TOOL_ROOT / "generate_libero_multiview.py").read_text(encoding="utf-8")
        self.assertNotIn('HIGH_CAMERA = "agentview"', multiview)
        self.assertIn('WRIST_CAMERA = "robot0_eye_in_hand"', multiview)
        self.assertIn('"cam_high": video.name', multiview)
        self.assertIn('"cam_wrist": wrist_path.name', multiview)
        self.assertIn("ACTION_FIELDS", multiview)
        self.assertIn('"camera_video_paths"', multiview)
        self.assertNotIn('.cam_high.mp4', multiview)
        self.assertIn('.cam_wrist.mp4', multiview)
        self.assertIn('.camera_videos.json', multiview)
        self.assertNotIn("sideview", multiview)
        self.assertNotIn("camera_source_names", multiview)
        self.assertNotIn(".cam_left_wrist.mp4", multiview)
        self.assertNotIn(".cam_right_wrist.mp4", multiview)
        self.assertNotIn("multiview_video_path", multiview)
        self.assertNotIn("multiview_layout", multiview)
        self.assertNotIn("multiview_cameras", multiview)
        self.assertNotIn(".multiview.mp4", multiview)
        self.assertNotIn("save_safe_features=True", readme)
        self.assertIn("official .pkl", readme)

    def test_generated_change_point_snapshot_contract(self) -> None:
        root = PROJECT_ROOT / "outputs/baseline_signal_analysis/changepoint_primary_20260830"
        self.assertTrue(root.is_dir())
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["analysis"], "lf3r_baseline_change_points")
        self.assertEqual(metadata["counts"]["rollouts"], 125)
        self.assertEqual(metadata["counts"]["observable_events"], 58)
        self.assertEqual(metadata["local_scales_frames"], [8, 16, 32, 64])
        self.assertEqual(metadata["features"], ["level", "variance", "slope"])
        self.assertEqual(metadata["method_coverage"]["rynnvalue"]["available_rollouts"], 124)
        self.assertEqual(
            metadata["method_coverage"]["rynnvalue"]["missing_rollout_ids"],
            ["libero_10-task02-ep005-natural-e8fc18cf25"],
        )
        for filename in [
            "changepoint_event_metrics.jsonl",
            "changepoint_summary.csv",
            "changepoint_reference_summary.csv",
            "changepoint_by_failure_type.csv",
            "changepoint_scales.csv",
            "localization_event_metrics.jsonl",
            "localization_summary.csv",
            "localization_by_failure_type.csv",
            "localization_thresholds.csv",
            "method_coverage.csv",
            "comparison_with_full_136_20260827.csv",
            "REPORT.md",
        ]:
            self.assertTrue((root / filename).is_file(), filename)
        event = json.loads((root / "changepoint_event_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(event["native_samples_only"])
        self.assertNotIn("frame_values", event)
        for field in ["precision", "f1", "auroc", "average_precision", "false_alarm_rate"]:
            self.assertIn(field, event)
        summary_header = (root / "localization_summary.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
        for field in ["precision", "f1", "auroc", "average_precision", "false_alarm_rate"]:
            self.assertIn(field, summary_header)
        threshold_header = (root / "localization_thresholds.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
        for field in ["threshold_q90", "threshold_q95", "threshold_q99"]:
            self.assertIn(field, threshold_header)
        with (root / "localization_summary.csv").open(newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))
        self.assertTrue(any(
            row.get("threshold") == "q95"
            and row.get("outcome_group") == "all_events"
            and math.isfinite(float(row["auroc"]))
            and math.isfinite(float(row["average_precision"]))
            for row in summary_rows
        ))
        self.assertTrue(any("local change-point" in line.lower() for line in (root / "REPORT.md").read_text(encoding="utf-8").splitlines()))

if __name__ == "__main__":
    unittest.main()
