"use strict";

/* Workspace routing and view lifecycle. */

function workspaceHasUnsavedChanges() {
  return Boolean(state.dirty || workspaceState.settingsDirty);
}

function workspaceShowView(view) {
  ["reviewWorkspace", "runsView", "analysisView", "repairView", "settingsView"].forEach(function (id) {
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
  byId("pageTitle").textContent = view === "analysis" ? "Analysis" : view === "repair" ? "Repair" : view === "settings" ? "Settings" : view === "results" ? "Results" : view === "runs" ? "Runs" : "Annotate";
  document.title = "LF3R " + (view === "review" ? "Failure Review" : labelFor(view));
  window.dispatchEvent(new CustomEvent("lf3r:viewchange", {
    detail: { view: view }
  }));
}

function workspaceParseRoute() {
  var hash = window.location.hash || "#/annotate";
  var raw = hash.replace(/^#\/?/, "");
  var parts = raw.split("/");
  var view = ["review", "annotate", "results", "runs", "analysis", "repair", "settings"].indexOf(parts[0]) === -1 ? "annotate" : parts[0];
  if (view === "review") view = "annotate";
  var analysisTabs = ["outcome", "localization"];
  var analysisTab = view === "analysis" && analysisTabs.indexOf(parts[1]) !== -1 ? parts[1] : "outcome";
  var id = ["annotate", "results"].indexOf(view) !== -1 && parts.length > 1 && parts[1]
    ? decodeURIComponent(parts.slice(1).join("/")) : null;
  return { view: view, id: id, analysisTab: analysisTab, hash: hash };
}

function workspaceRenderAnalysisTabs(tab) {
  var allowed = ["outcome", "localization"];
  if (allowed.indexOf(tab) === -1) tab = "outcome";
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
    workspaceLoadAnalysis(false);
    workspaceLoadAnalysisEnvironment();
    if (route.analysisTab === "localization") {
      if (typeof window.localizationLabRefresh === "function") window.localizationLabRefresh();
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
  if (typeof window.lf3rOutcomeCoverageChanged === "function") {
    window.lf3rOutcomeCoverageChanged();
  }
  if (["annotate", "results"].indexOf(workspaceState.view) !== -1) {
    var route = workspaceParseRoute();
    if (route.id && route.id !== state.selectedId && (state.rollouts || []).some(function (record) { return record.id === route.id; }) && state.selectedId !== route.id) {
      selectRollout(route.id);
    }
  } else if (workspaceState.view === "repair"
      && window.LF3RRepairSyntheticSuffix
      && typeof window.LF3RRepairSyntheticSuffix.refresh === "function") {
    window.LF3RRepairSyntheticSuffix.refresh();
  } else if (workspaceState.view === "analysis") {
    workspaceDashboardRenderSnapshot();
  }
}
