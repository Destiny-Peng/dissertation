"use strict";

/* Shared Analysis environment state.
 *
 * The legacy temporal-analysis runner was removed. Rule-based fused-hop
 * analysis still uses the dedicated CPU Analysis environment, so this module
 * only keeps the health check that those focused tools share.
 */

function workspaceJobChanged(_job) {
  // Focused Analysis tools own their job polling.
}

function workspaceJobsChanged(_jobs) {
  // Focused Analysis tools own their job polling.
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
    if (typeof window.lf3rRoboHopEnvironmentChanged === "function") {
      window.lf3rRoboHopEnvironmentChanged();
    }
  }
}
