"use strict";

var SETTINGS_DEFAULTS = {
  background_color: "#0d1117",
  surface_color: "#121820",
  surface_raised_color: "#18202a",
  control_color: "#0f151c",
  text_color: "#e7edf3",
  muted_color: "#929eaa",
  accent_color: "#79aa9e",
  font_scale: 1.0,
  review_font_scale: 1.0,
  analysis_font_scale: 1.0,
  control_font_scale: 1.0,
  density: "comfortable"
};

var SETTINGS_PRESETS = {
  midnight: {
    background_color: "#0d1117",
    surface_color: "#121820",
    surface_raised_color: "#18202a",
    control_color: "#0f151c",
    text_color: "#e7edf3",
    muted_color: "#929eaa",
    accent_color: "#79aa9e",
    font_scale: 1.0,
    review_font_scale: 1.0,
    analysis_font_scale: 1.0,
    control_font_scale: 1.0,
    density: "comfortable"
  },
  slate: {
    background_color: "#111821",
    surface_color: "#17212b",
    surface_raised_color: "#202c38",
    control_color: "#0c1218",
    text_color: "#eef4f8",
    muted_color: "#a9b6c1",
    accent_color: "#8eb8ff",
    font_scale: 1.0,
    review_font_scale: 1.0,
    analysis_font_scale: 1.0,
    control_font_scale: 1.0,
    density: "comfortable"
  },
  warm: {
    background_color: "#17120f",
    surface_color: "#211a15",
    surface_raised_color: "#2c2119",
    control_color: "#100d0b",
    text_color: "#f6eee6",
    muted_color: "#b9aaa0",
    accent_color: "#efb47f",
    font_scale: 1.0,
    review_font_scale: 1.0,
    analysis_font_scale: 1.0,
    control_font_scale: 1.0,
    density: "comfortable"
  }
};

var SETTINGS_LEGACY_DEFAULT_PALETTE = {
  background_color: "#0b0d10",
  surface_color: "#11151a",
  surface_raised_color: "#171c22",
  control_color: "#0f1318",
  text_color: "#f4f6f7",
  muted_color: "#98a3ad",
  accent_color: "#67d9b5"
};

function workspaceNormalizeLoadedSettings(settings) {
  if (!settings) return Object.assign({}, SETTINGS_DEFAULTS);
  var usesLegacyDefaults = Object.keys(SETTINGS_LEGACY_DEFAULT_PALETTE).every(function (key) {
    return String(settings[key] || "").toLowerCase() === SETTINGS_LEGACY_DEFAULT_PALETTE[key];
  });
  return usesLegacyDefaults
    ? Object.assign({}, settings, {
        background_color: SETTINGS_DEFAULTS.background_color,
        surface_color: SETTINGS_DEFAULTS.surface_color,
        surface_raised_color: SETTINGS_DEFAULTS.surface_raised_color,
        control_color: SETTINGS_DEFAULTS.control_color,
        text_color: SETTINGS_DEFAULTS.text_color,
        muted_color: SETTINGS_DEFAULTS.muted_color,
        accent_color: SETTINGS_DEFAULTS.accent_color
      })
    : settings;
}

var ANALYSIS_OUTCOMES = [
  { value: "clean_success", label: "Clean success", color: "var(--success)" },
  { value: "recovered_success", label: "Recovered success", color: "var(--recovery)" },
  { value: "terminal_failure", label: "Terminal failure", color: "var(--terminal)" },
  { value: "uncertain", label: "Uncertain", color: "var(--muted)" }
];
var ANALYSIS_METHODS = ["safe", "procvlm", "rynnvalue", "robo_dopamine"];
var ANALYSIS_EVENT_GROUPS = ["terminal_failure", "recovered_success", "uncertain"];
var ANALYSIS_EVENT_COLORS = {
  terminal_failure: "#ef7869",
  recovered_success: "#67d9b5",
  uncertain: "#e7c15c"
};
var ANALYSIS_METHOD_LABELS = {
  safe: "SAFE",
  procvlm: "ProcVLM",
  rynnvalue: "RynnValue",
  robo_dopamine: "Robo-Dopamine"
};
var ANALYSIS_DEFAULT_LIVE_FILTERS = {
  partition: "libero_10",
  suite: "all",
  task: "all",
  outcome: "all"
};
var ANALYSIS_DEFAULT_SNAPSHOT_FILTERS = {
  method: "all",
  outcome: "all",
  task: "all"
};
var workspaceState = {
  view: "review",
  lastHash: "",
  settings: null,
  settingsDraft: null,
  settingsLoaded: false,
  settingsLoading: false,
  settingsDirty: false,
  analysisSnapshot: null,
  analysisLoading: false,
  analysisLoaded: false,
  baselineRuns: [],
  baselineRunsLoading: false,
  baselineRunsRequest: 0,
  analysisRunScope: "libero_10",
  analysisRunJob: null,
  analysisRunJobs: {},
  analysisEnvironment: null,
  analysisEnvironmentLoading: false,
  liveFilters: Object.assign({}, ANALYSIS_DEFAULT_LIVE_FILTERS),
  snapshotFilters: Object.assign({}, ANALYSIS_DEFAULT_SNAPSHOT_FILTERS),
  localizationFilters: {
    method: "all",
    signal: "all",
    threshold: "q95",
    outcome: "all",
    task: "all"
  },
  changePointFilters: {
    method: "all",
    signal: "all",
    feature: "level",
    scale: "16",
    threshold: "q95",
    outcome: "all",
    task: "all"
  },
  eventTriggeredFilters: {
    method: "all",
    signal: "all",
    event_group: "terminal_failure",
    scale: "all"
  },
  analysisTab: "localization",
  analysisDetails: {
    kind: "changepoint_events",
    page: 1,
    pageSize: 25,
    sort: "score",
    filters: {
      method: "all",
      signal: "all",
      feature: "all",
      scale: "all",
      threshold: "all",
      outcome: "all",
      failure_type: "all",
      task: "all"
    },
    loading: false,
    payload: null,
    request: 0
  },
  analysisFailureMetric: "recall",
  analysisSignalRows: null,
  analysisSignalRequest: 0
};

function workspaceApplySettings(settings) {
  var root = document.documentElement;
  var mapping = {
    background_color: "--bg",
    surface_color: "--surface",
    surface_raised_color: "--surface-raised",
    control_color: "--control",
    text_color: "--text",
    muted_color: "--muted",
    accent_color: "--accent"
  };
  Object.keys(mapping).forEach(function (field) {
    if (settings && settings[field]) root.style.setProperty(mapping[field], settings[field]);
  });
  root.style.setProperty("--font-scale", String(settings && settings.font_scale || 1));
  root.style.setProperty("--review-font-scale", String(settings && settings.review_font_scale || 1));
  root.style.setProperty("--analysis-font-scale", String(settings && settings.analysis_font_scale || 1));
  root.style.setProperty("--control-font-scale", String(settings && settings.control_font_scale || 1));
  document.body.dataset.density = settings && settings.density || "comfortable";
}

function workspaceSettingsFromForm() {
  var fields = ["background_color", "surface_color", "surface_raised_color", "control_color", "text_color", "muted_color", "accent_color"];
  var settings = {};
  fields.forEach(function (field) {
    var input = document.querySelector('[data-settings-field="' + field + '"]');
    settings[field] = input ? input.value : SETTINGS_DEFAULTS[field];
  });
  var scale = byId("settingsFontScale");
  var reviewScale = byId("settingsReviewFontScale");
  var analysisScale = byId("settingsAnalysisFontScale");
  var controlScale = byId("settingsControlFontScale");
  var density = byId("settingsDensity");
  settings.font_scale = scale ? Number(scale.value) : SETTINGS_DEFAULTS.font_scale;
  settings.review_font_scale = reviewScale ? Number(reviewScale.value) : SETTINGS_DEFAULTS.review_font_scale;
  settings.analysis_font_scale = analysisScale ? Number(analysisScale.value) : SETTINGS_DEFAULTS.analysis_font_scale;
  settings.control_font_scale = controlScale ? Number(controlScale.value) : SETTINGS_DEFAULTS.control_font_scale;
  settings.density = density ? density.value : SETTINGS_DEFAULTS.density;
  return settings;
}

function workspaceUpdateColorOutputs(settings) {
  document.querySelectorAll("[data-settings-value]").forEach(function (output) {
    var field = output.dataset.settingsValue;
    output.value = settings[field] || "";
    output.textContent = settings[field] || "";
  });
  [
    ["settingsFontScale", "settingsFontScaleValue"],
    ["settingsReviewFontScale", "settingsReviewFontScaleValue"],
    ["settingsAnalysisFontScale", "settingsAnalysisFontScaleValue"],
    ["settingsControlFontScale", "settingsControlFontScaleValue"]
  ].forEach(function (pair) {
    var scale = byId(pair[0]);
    var output = byId(pair[1]);
    if (scale && output) output.textContent = Math.round(Number(scale.value) * 100) + "%";
  });
}

function workspacePopulateSettings(settings) {
  var fields = ["background_color", "surface_color", "surface_raised_color", "control_color", "text_color", "muted_color", "accent_color"];
  fields.forEach(function (field) {
    var input = document.querySelector('[data-settings-field="' + field + '"]');
    if (input) input.value = settings[field];
  });
  [
    ["settingsFontScale", "font_scale"],
    ["settingsReviewFontScale", "review_font_scale"],
    ["settingsAnalysisFontScale", "analysis_font_scale"],
    ["settingsControlFontScale", "control_font_scale"]
  ].forEach(function (pair) {
    var input = byId(pair[0]);
    if (input) input.value = settings[pair[1]] == null ? SETTINGS_DEFAULTS[pair[1]] : settings[pair[1]];
  });
  if (byId("settingsDensity")) byId("settingsDensity").value = settings.density;
  workspaceUpdateColorOutputs(settings);
}

function workspaceSettingsStatus(message, kind) {
  var status = byId("settingsStatus");
  status.textContent = message;
  status.className = "analysis-status" + (kind ? " " + kind : "");
}

async function workspaceLoadSettings() {
  if (workspaceState.settingsLoading || workspaceState.settingsLoaded) return;
  workspaceState.settingsLoading = true;
  try {
    var response = await fetch("/api/settings", { cache: "no-store" });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not load shared settings");
    workspaceState.settings = workspaceNormalizeLoadedSettings(
      payload.settings || Object.assign({}, SETTINGS_DEFAULTS)
    );
    workspaceState.settingsDraft = Object.assign({}, workspaceState.settings);
    workspaceState.settingsLoaded = true;
    workspaceState.settingsDirty = false;
    workspaceApplySettings(workspaceState.settings);
    workspacePopulateSettings(workspaceState.settings);
    workspaceSettingsStatus(payload.updated_at ? "Shared settings loaded · updated " + payload.updated_at : "Using default shared settings; save to create the project config.", "");
  } catch (error) {
    workspaceState.settings = Object.assign({}, SETTINGS_DEFAULTS);
    workspaceState.settingsDraft = Object.assign({}, SETTINGS_DEFAULTS);
    workspaceState.settingsLoaded = true;
    workspaceApplySettings(workspaceState.settings);
    workspacePopulateSettings(workspaceState.settings);
    workspaceSettingsStatus("Settings API unavailable; previewing defaults. " + error.message, "warning");
  } finally {
    workspaceState.settingsLoading = false;
  }
}

async function workspaceSaveSettings(event) {
  if (event) event.preventDefault();
  var settings = workspaceSettingsFromForm();
  workspaceApplySettings(settings);
  var button = byId("settingsSave");
  button.disabled = true;
  workspaceSettingsStatus("Saving shared settings…", "");
  try {
    var response = await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings)
    });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not save shared settings");
    workspaceState.settings = payload.settings;
    workspaceState.settingsDraft = Object.assign({}, payload.settings);
    workspaceState.settingsDirty = false;
    workspaceApplySettings(payload.settings);
    workspacePopulateSettings(payload.settings);
    workspaceSettingsStatus("Shared settings saved · updated " + (payload.updated_at || "now"), "");
  } catch (error) {
    workspaceState.settingsDirty = true;
    workspaceSettingsStatus("Settings save failed: " + error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function workspaceHandleSettingsInput() {
  var settings = workspaceSettingsFromForm();
  workspaceState.settingsDraft = settings;
  workspaceState.settingsDirty = true;
  workspaceApplySettings(settings);
  workspaceUpdateColorOutputs(settings);
  workspaceSettingsStatus("Previewing unsaved shared settings; save to publish them.", "warning");
}

function workspaceHandlePreset(name) {
  if (!SETTINGS_PRESETS[name]) return;
  workspaceState.settingsDraft = Object.assign({}, SETTINGS_PRESETS[name]);
  workspacePopulateSettings(workspaceState.settingsDraft);
  workspaceApplySettings(workspaceState.settingsDraft);
  workspaceState.settingsDirty = true;
  workspaceSettingsStatus("Previewing the " + name + " preset; save to publish it.", "warning");
}

function workspaceResetSettings() {
  workspaceState.settingsDraft = Object.assign({}, SETTINGS_DEFAULTS);
  workspacePopulateSettings(workspaceState.settingsDraft);
  workspaceApplySettings(workspaceState.settingsDraft);
  workspaceState.settingsDirty = true;
  workspaceSettingsStatus("Previewing defaults; save to publish them.", "warning");
}

function workspaceHasUnsavedChanges() {
  return Boolean(state.dirty || workspaceState.settingsDirty);
}

function workspaceShowView(view) {
  ["reviewWorkspace", "runsView", "analysisView", "settingsView"].forEach(function (id) {
    var node = byId(id);
    if (node) node.classList.toggle("hidden", (id === "reviewWorkspace" ? ["annotate", "results"].indexOf(view) === -1 : node.dataset.view !== view));
  });
  document.querySelectorAll("[data-route]").forEach(function (link) {
    var active = link.dataset.route === view;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.documentElement.dataset.activeView = view;
  document.body.dataset.view = view;
  if (typeof window.lf3rResultsLayoutRefresh === "function") {
    window.lf3rResultsLayoutRefresh();
  }
  byId("pageTitle").textContent = view === "analysis" ? "Analysis" : view === "settings" ? "Settings" : view === "results" ? "Results" : view === "runs" ? "Runs" : "Annotate";
  document.title = "LF3R " + (view === "review" ? "Failure Review" : labelFor(view));
}

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

window.lf3rWorkspaceDataChanged = workspaceDataChanged;
window.lf3rWorkspaceJobsChanged = workspaceJobsChanged;
window.lf3rWorkspaceJobChanged = workspaceJobChanged;

function workspaceParseRoute() {
  var hash = window.location.hash || "#/annotate";
  var raw = hash.replace(/^#\/?/, "");
  var parts = raw.split("/");
  var view = ["review", "annotate", "results", "runs", "analysis", "settings"].indexOf(parts[0]) === -1 ? "annotate" : parts[0];
  if (view === "review") view = "annotate";
  var analysisTabs = ["localization", "overview", "comparison", "failures", "events", "signals", "archive"];
  var analysisTab = view === "analysis" && analysisTabs.indexOf(parts[1]) !== -1 ? parts[1] : "localization";
  var id = ["annotate", "results"].indexOf(view) !== -1 && parts.length > 1 && parts[1]
    ? decodeURIComponent(parts.slice(1).join("/")) : null;
  return { view: view, id: id, analysisTab: analysisTab, hash: hash };
}

function workspaceRenderAnalysisTabs(tab) {
  var allowed = ["localization", "overview", "comparison", "failures", "events", "signals", "archive"];
  if (allowed.indexOf(tab) === -1) tab = "localization";
  workspaceState.analysisTab = tab;
  document.querySelectorAll("[data-analysis-panel]").forEach(function (panel) {
    panel.classList.toggle("hidden", panel.dataset.analysisPanel !== tab);
  });
  document.querySelectorAll("[data-analysis-tab]").forEach(function (link) {
    var active = link.dataset.analysisTab === tab;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.querySelectorAll("[data-analysis-legacy-global]").forEach(function (node) {
    node.classList.toggle("hidden", tab === "localization");
  });
}

function workspaceRenderRoute() {
  var route = workspaceParseRoute();
  if (workspaceState.lastHash && workspaceState.lastHash !== route.hash && workspaceHasUnsavedChanges()) {
    var sharedReview = ["annotate", "results"].indexOf(workspaceState.view) !== -1 && ["annotate", "results"].indexOf(route.view) !== -1 && (!route.id || route.id === state.selectedId);
    if (!window.confirm(sharedReview ? "Switch page and keep unsaved annotations?" : "Discard unsaved changes and leave this page?")) {
      window.history.replaceState(null, "", workspaceState.lastHash);
      return;
    }
    if (!sharedReview) {
      state.dirty = false;
      workspaceState.settingsDirty = false;
    }
  }
  workspaceState.lastHash = route.hash;
  workspaceState.view = route.view;
  state.view = route.view;
  workspaceShowView(route.view);
  if (["annotate", "results"].indexOf(route.view) !== -1) {
    if (route.id && route.id !== state.selectedId && (state.rollouts || []).some(function (record) { return record.id === route.id; })) {
      selectRollout(route.id);
    } else if (!selectedRollout() && state.filtered && state.filtered.length) {
      selectRollout(state.filtered[0].id);
    }
  } else if (route.view === "analysis") {
    workspaceRenderAnalysisTabs(route.analysisTab);
    if (route.analysisTab === "localization") {
      if (typeof window.localizationLabRefresh === "function") window.localizationLabRefresh();
    } else {
      workspaceRenderLiveAnalysis();
      workspaceRenderAnalysisRunPanel();
      workspaceLoadAnalysisEnvironment();
      workspaceLoadAnalysis(false);
      workspaceLoadBaselineRuns(workspaceState.analysisRunScope || "libero_10");
    }
  } else if (route.view === "settings") {
    workspaceLoadSettings();
  }
}

function workspaceDataChanged() {
  if (window.LF3RDatasetScopes
      && typeof window.LF3RDatasetScopes.refresh === "function") {
    window.LF3RDatasetScopes.refresh();
  }
  if (["annotate", "results"].indexOf(workspaceState.view) !== -1) {
    var route = workspaceParseRoute();
    if (route.id && route.id !== state.selectedId && (state.rollouts || []).some(function (record) { return record.id === route.id; }) && state.selectedId !== route.id) {
      selectRollout(route.id);
    }
  } else if (workspaceState.view === "analysis") {
    workspaceRenderLiveAnalysis();
    workspaceRenderAnalysisRunPanel();
    workspaceDashboardRenderSnapshot();
  }
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

workspaceInstallAnalysisDashboardEvents();

workspaceInstallEvents();
workspaceLoadSettings();
workspaceRenderRoute();
if (typeof window.lf3rRefreshPersistentJobs === "function") {
  window.lf3rRefreshPersistentJobs();
}
