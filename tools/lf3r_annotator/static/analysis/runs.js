"use strict";

/* Shared Analysis snapshot/environment state. Focused tools own job polling. */

function workspaceJobChanged(_job) {}
function workspaceJobsChanged(_jobs) {}

async function workspaceLoadAnalysis(force) {
  if (workspaceState.analysisLoading || (workspaceState.analysisLoaded && !force)) {
    workspaceRenderSnapshot();
    return workspaceState.analysisSnapshot;
  }
  workspaceState.analysisLoading = true;
  workspaceRenderSnapshot();
  try {
    var response = await fetch("/api/analysis", { cache: "no-store" });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not load analysis snapshot");
    workspaceState.analysisSnapshot = payload.analysis || {
      available: false,
      message: "Empty analysis response"
    };
    workspaceState.analysisLoaded = true;
  } catch (error) {
    workspaceState.analysisSnapshot = {
      available: false,
      message: "Analysis snapshot error: " + error.message
    };
    workspaceState.analysisLoaded = true;
  } finally {
    workspaceState.analysisLoading = false;
    workspaceRenderSnapshot();
  }
  return workspaceState.analysisSnapshot;
}

async function workspaceLoadAnalysisEnvironment() {
  workspaceState.analysisEnvironmentLoading = true;
  try {
    var response = await fetch("/api/health", { cache: "no-store" });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not read Analysis environment status");
    workspaceState.analysisEnvironment = payload.analysis_environment || null;
  } catch (error) {
    workspaceState.analysisEnvironment = {
      ready: false,
      error: "Health endpoint unavailable: " + error.message
    };
  } finally {
    workspaceState.analysisEnvironmentLoading = false;
    if (typeof window.lf3rOutcomeEnvironmentChanged === "function") {
      window.lf3rOutcomeEnvironmentChanged();
    }
    if (typeof window.lf3rRoboHopEnvironmentChanged === "function") {
      window.lf3rRoboHopEnvironmentChanged();
    }
  }
}

window.workspaceLoadAnalysis = workspaceLoadAnalysis;
