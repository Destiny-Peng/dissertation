"use strict";

/* Workspace event wiring. */

function workspaceInstallEvents() {
  window.addEventListener("hashchange", workspaceRenderRoute);
  window.addEventListener("beforeunload", function (event) {
    if (workspaceState.settingsDirty) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  byId("analysisRefresh").addEventListener("click", function () { workspaceLoadAnalysis(true); });
  byId("analysisResetFilters").addEventListener("click", function () {
    workspaceState.liveFilters = Object.assign({}, ANALYSIS_DEFAULT_LIVE_FILTERS);
    workspaceState.localizationFilters = {
      method: "all", signal: "all", threshold: "q95", outcome: "all", task: "all"
    };
    workspaceState.changePointFilters = {
      method: "all", signal: "all", feature: "level", scale: "16", threshold: "q95", outcome: "all", task: "all"
    };
    workspaceState.eventTriggeredFilters = {
      method: "all", signal: "all", event_group: "terminal_failure", scale: "all"
    };
    workspaceState.analysisDashboardRows = null;
    workspaceState.analysisSignalRows = null;
    workspaceRenderLiveAnalysis();
    workspaceRenderSnapshot();
  });
  byId("analysisView").addEventListener("change", function (event) {
    if (event.target.id === "analysisRunScope") {
      workspaceState.analysisRunScope = event.target.value;
      workspaceLoadBaselineRuns(event.target.value, true);
      return;
    }
    if (event.target.dataset.analysisRunMethod) {
      workspaceRenderAnalysisRunPanel();
      return;
    }
    var liveField = event.target.dataset.analysisLiveFilter;
    if (liveField) {
      workspaceState.liveFilters[liveField] = event.target.value;
      workspaceRenderLiveAnalysis();
      return;
    }
    var snapshotField = event.target.dataset.analysisSnapshotFilter;
    if (snapshotField) {
      workspaceState.snapshotFilters[snapshotField] = event.target.value;
      workspaceRenderSnapshot();
      return;
    }
    var localizationField = event.target.dataset.analysisLocalizationFilter;
    if (localizationField) {
      workspaceState.localizationFilters[localizationField] = event.target.value;
      workspaceRenderSnapshot();
      return;
    }
    var changePointField = event.target.dataset.analysisChangepointFilter;
    if (changePointField) {
      workspaceState.changePointFilters[changePointField] = event.target.value;
      workspaceRenderSnapshot();
      return;
    }
    var eventTriggeredField = event.target.dataset.analysisEventTriggeredFilter;
    if (eventTriggeredField) {
      workspaceState.eventTriggeredFilters[eventTriggeredField] = event.target.value;
      workspaceRenderSnapshot();
    }
  });
  byId("analysisView").addEventListener("click", function (event) {
    var taskBar = event.target.closest("[data-analysis-task]");
    if (taskBar) {
      workspaceState.liveFilters.task = taskBar.dataset.analysisTask;
      workspaceState.liveFilters.suite = taskBar.dataset.analysisSuite || "all";
      workspaceRenderLiveAnalysis();
      return;
    }
    var outcomeBar = event.target.closest("[data-analysis-outcome]");
    if (outcomeBar) {
      workspaceState.liveFilters.outcome = outcomeBar.dataset.analysisOutcome;
      byId("analysisOutcomeFilter").value = workspaceState.liveFilters.outcome;
      workspaceRenderLiveAnalysis();
      return;
    }
    var rolloutButton = event.target.closest("[data-analysis-rollout]");
    if (rolloutButton) {
      event.preventDefault();
      window.location.hash = "#/review/" + encodeURIComponent(rolloutButton.dataset.analysisRollout);
    }
  });
  byId("analysisRunForm").addEventListener("submit", workspaceStartAnalysisRun);
  byId("settingsForm").addEventListener("input", workspaceHandleSettingsInput);
  byId("settingsForm").addEventListener("change", workspaceHandleSettingsInput);
  byId("settingsForm").addEventListener("submit", workspaceSaveSettings);
  byId("settingsView").addEventListener("click", function (event) {
    var preset = event.target.closest("[data-settings-preset]");
    if (preset) workspaceHandlePreset(preset.dataset.settingsPreset);
    if (event.target.closest("#settingsReset")) workspaceResetSettings();
  });
}

function workspaceInstallAnalysisDashboardEvents() {
  var view = byId("analysisView");
  if (!view) return;
  view.addEventListener("change", function (event) {
    var failureField = event.target.dataset.analysisFailureFilter;
    if (failureField === "metric") {
      workspaceState.analysisFailureMetric = event.target.value;
      workspaceState.analysisDashboardRows = workspaceState.analysisDashboardRows || {};
      workspaceState.analysisDashboardRows.failure = null;
      workspaceDashboardRenderSnapshot();
      return;
    }
    var changePointField = event.target.dataset.analysisChangepointFilter;
    if (changePointField) {
      workspaceState.analysisDashboardRows = workspaceState.analysisDashboardRows || {};
      workspaceState.analysisDashboardRows.summary = null;
      workspaceState.analysisDashboardRows.failure = null;
      return;
    }
    var localizationField = event.target.dataset.analysisLocalizationFilter;
    if (localizationField) {
      workspaceState.analysisDashboardRows = workspaceState.analysisDashboardRows || {};
      workspaceState.analysisDashboardRows.failure = null;
      return;
    }
    var eventTriggeredField = event.target.dataset.analysisEventTriggeredFilter;
    if (eventTriggeredField) {
      workspaceState.analysisSignalRows = null;
      return;
    }
    var detailsField = event.target.dataset.analysisDetailsFilter;
    if (detailsField) {
      var details = workspaceState.analysisDetails;
      if (detailsField === "kind") details.kind = event.target.value;
      else if (detailsField === "sort") details.sort = event.target.value;
      else details.filters[detailsField] = event.target.value;
      details.page = 1;
      details.payload = null;
      if (workspaceState.analysisTab === "events") workspaceLoadAnalysisDetails();
    }
  });
  byId("analysisDetailsPrevious").addEventListener("click", function () {
    if (workspaceState.analysisDetails.page > 1) {
      workspaceState.analysisDetails.page -= 1;
      workspaceLoadAnalysisDetails();
    }
  });
  byId("analysisDetailsNext").addEventListener("click", function () {
    var payload = workspaceState.analysisDetails.payload;
    if (payload && workspaceState.analysisDetails.page < payload.page_count) {
      workspaceState.analysisDetails.page += 1;
      workspaceLoadAnalysisDetails();
    }
  });
}
