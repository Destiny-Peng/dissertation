"use strict";

var SETTINGS_DEFAULTS = {
  background_color: "#F6F8FB",
  surface_color: "#FFFFFF",
  surface_raised_color: "#F9FBFD",
  control_color: "#FFFFFF",
  text_color: "#1F2A37",
  muted_color: "#5B6B7F",
  accent_color: "#4F7DF3",
  font_scale: 1.0,
  review_font_scale: 1.0,
  analysis_font_scale: 1.0,
  control_font_scale: 1.0,
  density: "comfortable"
};

var SETTINGS_PRESETS = {
  academic_light: {
    background_color: "#F6F8FB",
    surface_color: "#FFFFFF",
    surface_raised_color: "#F9FBFD",
    control_color: "#FFFFFF",
    text_color: "#1F2A37",
    muted_color: "#5B6B7F",
    accent_color: "#4F7DF3",
    font_scale: 1.0,
    review_font_scale: 1.0,
    analysis_font_scale: 1.0,
    control_font_scale: 1.0,
    density: "comfortable"
  },
  midnight: {
    background_color: "#0D1117",
    surface_color: "#121820",
    surface_raised_color: "#18202A",
    control_color: "#0F151C",
    text_color: "#E7EDF3",
    muted_color: "#929EAA",
    accent_color: "#79AA9E",
    font_scale: 1.0,
    review_font_scale: 1.0,
    analysis_font_scale: 1.0,
    control_font_scale: 1.0,
    density: "comfortable"
  },
  slate: {
    background_color: "#111821",
    surface_color: "#17212B",
    surface_raised_color: "#202C38",
    control_color: "#0C1218",
    text_color: "#EEF4F8",
    muted_color: "#A9B6C1",
    accent_color: "#8EB8FF",
    font_scale: 1.0,
    review_font_scale: 1.0,
    analysis_font_scale: 1.0,
    control_font_scale: 1.0,
    density: "comfortable"
  },
  warm: {
    background_color: "#17120F",
    surface_color: "#211A15",
    surface_raised_color: "#2C2119",
    control_color: "#100D0B",
    text_color: "#F6EEE6",
    muted_color: "#B9AAA0",
    accent_color: "#EFB47F",
    font_scale: 1.0,
    review_font_scale: 1.0,
    analysis_font_scale: 1.0,
    control_font_scale: 1.0,
    density: "comfortable"
  }
};

var SETTINGS_PREVIOUS_DEFAULT_PALETTE = {
  background_color: "#0d1117",
  surface_color: "#121820",
  surface_raised_color: "#18202a",
  control_color: "#0f151c",
  text_color: "#e7edf3",
  muted_color: "#929eaa",
  accent_color: "#79aa9e"
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


var ANALYSIS_OUTCOMES = [
  { value: "clean_success", label: "Clean success", color: "var(--success)" },
  { value: "recovered_success", label: "Recovered success", color: "var(--recovery)" },
  { value: "terminal_failure", label: "Terminal failure", color: "var(--terminal)" },
  { value: "uncertain", label: "Uncertain", color: "var(--muted)" }
];
var ANALYSIS_METHODS = ["safe", "procvlm", "rynnvalue", "robo_dopamine"];
var ANALYSIS_EVENT_GROUPS = ["terminal_failure", "recovered_success", "uncertain"];
var ANALYSIS_EVENT_COLORS = {
  terminal_failure: "#D96C6C",
  recovered_success: "#2F9E6F",
  uncertain: "#E6A23C"
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
  analysisTab: "outcome",
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


window.lf3rWorkspaceDataChanged = workspaceDataChanged;
window.lf3rWorkspaceJobsChanged = workspaceJobsChanged;
window.lf3rWorkspaceJobChanged = workspaceJobChanged;


workspaceInstallAnalysisDashboardEvents();

workspaceInstallEvents();
workspaceLoadSettings();
workspaceRenderRoute();
