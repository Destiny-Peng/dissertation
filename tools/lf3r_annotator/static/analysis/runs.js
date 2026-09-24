"use strict";

/* Analysis runs module. Runtime state is owned by workspace-core.js. */

function workspaceAnalysisRunScopeCount(scope) {
  return (state.rollouts || []).filter(function (record) {
    return workspacePartitionMatches(record, scope);
  }).length;
}

function workspaceAnalysisRunStatus(message, kind) {
  var status = byId("analysisRunSelection");
  if (status) {
    status.textContent = message;
    status.className = "analysis-run-selection" + (kind ? " " + kind : "");
  }
}

function workspaceAnalysisRunBadge(status) {
  var badge = byId("analysisRunBadge");
  if (!badge) return;
  var normalized = String(status || "Idle").toLowerCase();
  badge.className = "analysis-badge" + (normalized === "complete" ? " ok" : normalized === "failed" ? " missing" : normalized === "running" || normalized === "queued" ? " warning" : "");
  badge.textContent = status || "Idle";
}

function workspaceRenderAnalysisJobs() {
  var jobs = Object.keys(workspaceState.analysisRunJobs).map(function (jobId) {
    return workspaceState.analysisRunJobs[jobId];
  });
  jobs.sort(function (left, right) {
    return String(right.submitted_at || right.job_id).localeCompare(String(left.submitted_at || left.job_id));
  });
  if (typeof window.lf3rRenderJobCards === "function") {
    window.lf3rRenderJobCards(
      "analysisRunJobs",
      jobs,
      "No temporal-analysis jobs recorded."
    );
  }
}

function workspaceJobIsActive(job) {
  return job && (job.status === "queued" || job.status === "running");
}

function workspaceIsTemporalAnalysisJob(job) {
  return Boolean(job && job.job_type === "analysis" && !job.analysis_kind);
}

function workspaceJobChanged(job) {
  if (!workspaceIsTemporalAnalysisJob(job)) return;
  var previous = workspaceState.analysisRunJobs[job.job_id];
  workspaceState.analysisRunJobs[job.job_id] = job;
  var current = workspaceState.analysisRunJob;
  if (!current || current.job_id === job.job_id
      || (workspaceJobIsActive(job) && !workspaceJobIsActive(current))) {
    workspaceState.analysisRunJob = job;
  }
  workspaceRenderAnalysisJobs();
  if (workspaceState.analysisRunJob && workspaceState.analysisRunJob.job_id === job.job_id) {
    workspaceAnalysisRunBadge(job.status);
    if (workspaceJobIsActive(job)) {
      workspaceAnalysisRunStatus(
        "Temporal analysis " + job.status + " / " + (job.selected_rollouts || 0)
          + " rollout(s) / tmux " + (job.tmux_session || "unavailable") + "...",
        ""
      );
    } else if (job.status === "complete") {
      workspaceAnalysisRunStatus("Temporal analysis complete / new snapshot: " + (job.output_dir || "unknown"), "");
    } else if (job.status === "failed") {
      workspaceAnalysisRunStatus("Temporal analysis failed; inspect the job log below.", "error");
    }
  }
  if (previous && previous.status !== job.status && job.status === "complete") {
    workspaceLoadAnalysis(true);
    workspaceLoadBaselineRuns(workspaceState.analysisRunScope || job.scope, true);
  }
}

function workspaceJobsChanged(jobs) {
  (jobs || []).filter(workspaceIsTemporalAnalysisJob).forEach(function (job) {
    workspaceJobChanged(job);
  });
  var analysisJobs = (jobs || []).filter(workspaceIsTemporalAnalysisJob);
  if (!workspaceState.analysisRunJob && analysisJobs.length) {
    workspaceState.analysisRunJob = analysisJobs[0];
  }
  workspaceRenderAnalysisJobs();
  if (workspaceState.analysisRunJob) {
    workspaceAnalysisRunBadge(workspaceState.analysisRunJob.status);
  }
}

async function workspaceLoadAnalysisEnvironment() {
  workspaceState.analysisEnvironmentLoading = true;
  workspaceRenderAnalysisRunPanel();
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
    workspaceRenderAnalysisRunPanel();
  }
}

function workspaceRunLabel(run) {
  var date = run.completed_at || run.created_at || "time unknown";
  var coverage = (run.run_rollout_count || run.selected_rollouts || 0) + " rollout(s)";
  var failed = run.failed_jobs ? " · " + run.failed_jobs + " failed" : "";
  return (run.run_root || "unknown run") + " · " + coverage + failed + " · " + date;
}

function workspaceSelectedAnalysisRuns(method) {
  var select = byId(ANALYSIS_RUN_SELECT_IDS[method]);
  if (!select) return [];
  return Array.prototype.filter.call(select.options, function (option) {
    return option.selected && option.value;
  }).map(function (option) {
    return option.value;
  });
}

function workspaceSetAnalysisRunSelections(select, values) {
  var selected = values || [];
  Array.prototype.forEach.call(select.options, function (option) {
    option.selected = selected.indexOf(option.value) !== -1;
  });
}

function workspaceAnalysisRunCanSelect(method, run) {
  return Boolean(run && (run.compatible || (method === "rynnvalue" && run.partial_compatible)));
}

function workspaceSelectedAnalysisRunObjects(method) {
  var values = workspaceSelectedAnalysisRuns(method);
  return values.map(function (value) {
    return (workspaceState.baselineRuns || []).find(function (run) {
      return run.baseline === method && run.run_root === value;
    });
  }).filter(Boolean);
}

function workspaceAnalysisRynnCoverage(scope) {
  var scopeIds = {};
  (state.rollouts || []).filter(function (record) {
    return workspacePartitionMatches(record, scope);
  }).forEach(function (record) {
    scopeIds[record.id] = true;
  });
  var covered = {};
  workspaceSelectedAnalysisRunObjects("rynnvalue").forEach(function (run) {
    (run.run_rollout_ids || []).forEach(function (rolloutId) {
      if (scopeIds[rolloutId]) covered[rolloutId] = true;
    });
  });
  return {
    selected_runs: workspaceSelectedAnalysisRuns("rynnvalue"),
    covered: Object.keys(covered).length,
    total: Object.keys(scopeIds).length
  };
}

function workspaceRenderAnalysisRunPanel() {
  var scope = workspaceState.analysisRunScope || byId("analysisRunScope").value;
  var count = workspaceAnalysisRunScopeCount(scope);
  var runs = workspaceState.baselineRuns || [];
  var compatible = {};
  runs.forEach(function (run) {
    if (run.compatible || (run.baseline === "rynnvalue" && run.partial_compatible)) {
      compatible[run.baseline] = (compatible[run.baseline] || 0) + 1;
    }
  });
  var rynnCoverage = workspaceAnalysisRynnCoverage(scope);
  var allSelected = ANALYSIS_METHODS.every(function (method) {
    var select = byId(ANALYSIS_RUN_SELECT_IDS[method]);
    return select && !select.disabled && workspaceSelectedAnalysisRuns(method).length > 0;
  });
  var button = byId("analysisRunButton");
  var active = workspaceState.analysisRunJob && ["queued", "running"].indexOf(workspaceState.analysisRunJob.status) !== -1;
  var environmentUnavailable = workspaceState.analysisEnvironment
    && !workspaceState.analysisEnvironment.ready;
  var environmentChecking = workspaceState.analysisEnvironmentLoading
    || !workspaceState.analysisEnvironment;
  if (button) button.disabled = !count || !allSelected || Boolean(active)
    || workspaceState.baselineRunsLoading || Boolean(environmentUnavailable)
    || environmentChecking;
  var environmentStatus = byId("analysisEnvironmentStatus");
  if (environmentStatus) {
    if (workspaceState.analysisEnvironmentLoading) {
      environmentStatus.textContent = "Checking the dedicated Analysis environment...";
      environmentStatus.className = "analysis-run-selection";
    } else if (environmentUnavailable) {
      environmentStatus.textContent = "Analysis environment unavailable: "
        + (workspaceState.analysisEnvironment.error || "install tools/lf3r_annotator/setup_analysis_env.sh");
      environmentStatus.className = "analysis-run-selection error";
    } else if (workspaceState.analysisEnvironment && workspaceState.analysisEnvironment.ready) {
      environmentStatus.textContent = "Analysis environment ready: "
        + (workspaceState.analysisEnvironment.path || "project-local venv")
        + " / CPU-only temporal analysis.";
      environmentStatus.className = "analysis-run-selection";
    }
  }
  if (workspaceState.baselineRunsLoading) {
    workspaceAnalysisRunStatus("Discovering completed runs for " + (ANALYSIS_RUN_SCOPE_LABELS[scope] || scope) + "…", "");
  } else if (environmentChecking) {
    workspaceAnalysisRunStatus("Checking the dedicated Analysis environment...", "warning");
  } else if (environmentUnavailable) {
    workspaceAnalysisRunStatus(
      "Analysis cannot run until the project-local LF3R-ananlyse environment is installed.",
      "error"
    );
  } else if (!count) {
    workspaceAnalysisRunStatus("This scope is empty; choose another scope or create baseline outputs first.", "warning");
  } else if (!allSelected) {
    var missing = ANALYSIS_METHODS.filter(function (method) { return !compatible[method]; }).map(function (method) { return ANALYSIS_METHOD_LABELS[method]; });
    workspaceAnalysisRunStatus(count + " rollout(s) selected. Choose one compatible completed run for each method" + (missing.length ? "; missing: " + missing.join(", ") : "."), "warning");
  } else if (rynnCoverage.selected_runs.length && rynnCoverage.covered < count) {
    workspaceAnalysisRunStatus(
      count + " rollout(s) selected. RynnValue sources cover " + rynnCoverage.covered + "/" + count
        + "; missing raw outputs remain explicit in the snapshot. Add another disjoint source if this is unintended.",
      "warning"
    );
  } else {
    workspaceAnalysisRunStatus(count + " rollout(s) selected. All selected baseline inputs are ready; analysis uses existing files only.", "");
  }
}

function workspacePopulateAnalysisRunSelectors() {
  var runs = workspaceState.baselineRuns || [];
  ANALYSIS_METHODS.forEach(function (method) {
    var select = byId(ANALYSIS_RUN_SELECT_IDS[method]);
    if (!select) return;
    var previous = workspaceSelectedAnalysisRuns(method);
    var choices = runs.filter(function (run) {
      return run.baseline === method && workspaceAnalysisRunCanSelect(method, run);
    });
    var html = "";
    if (!choices.length) {
      html = '<option value="">No compatible completed run</option>';
      select.disabled = true;
    } else {
      choices.forEach(function (run) {
        html += '<option value="' + escapeHtml(run.run_root) + '">' + escapeHtml(workspaceRunLabel(run)) + '</option>';
      });
      select.disabled = false;
    }
    select.innerHTML = html;
    var selectedValues = previous.filter(function (value) {
      return choices.some(function (run) { return run.run_root === value; });
    });
    if (!selectedValues.length && choices.length) {
      var completeChoice = choices.find(function (run) { return run.compatible; });
      selectedValues = method === "rynnvalue" && !completeChoice
        ? choices.map(function (run) { return run.run_root; })
        : [completeChoice ? completeChoice.run_root : choices[0].run_root];
    }
    workspaceSetAnalysisRunSelections(select, selectedValues);
  });
  workspaceRenderAnalysisRunPanel();
}

async function workspaceLoadBaselineRuns(scope, force) {
  scope = scope || workspaceState.analysisRunScope || "natural_observation";
  if (
    workspaceState.baselineRunsLoading
    && !force
    && workspaceState.baselineRunsScope === scope
  ) {
    while (workspaceState.baselineRunsLoading) {
      await new Promise(function (resolve) { window.setTimeout(resolve, 20); });
    }
    return workspaceState.baselineRuns;
  }
  if (!force && workspaceState.baselineRunsScope === scope && workspaceState.baselineRunsLoaded) {
    workspaceRenderAnalysisRunPanel();
    return workspaceState.baselineRuns;
  }
  var requestId = ++workspaceState.baselineRunsRequest;
  workspaceState.analysisRunScope = scope;
  workspaceState.baselineRunsLoading = true;
  workspaceState.baselineRunsScope = scope;
  workspaceRenderAnalysisRunPanel();
  try {
    var response = await fetch("/api/baselines/runs?scope=" + encodeURIComponent(scope), { cache: "no-store" });
    var payload = await response.json();
    if (requestId !== workspaceState.baselineRunsRequest) return;
    if (!response.ok) throw new Error(payload.error || "Could not discover baseline runs");
    workspaceState.baselineRuns = payload.runs || [];
    workspaceState.baselineRunsLoaded = true;
    workspacePopulateAnalysisRunSelectors();
  } catch (error) {
    if (requestId !== workspaceState.baselineRunsRequest) return;
    workspaceState.baselineRuns = [];
    workspaceState.baselineRunsLoaded = true;
    ANALYSIS_METHODS.forEach(function (method) {
      var select = byId(ANALYSIS_RUN_SELECT_IDS[method]);
      if (select) {
        select.innerHTML = '<option value="">Run discovery failed</option>';
        select.disabled = true;
      }
    });
    workspaceAnalysisRunStatus("Baseline run discovery failed: " + error.message, "error");
  } finally {
    if (requestId === workspaceState.baselineRunsRequest) {
      workspaceState.baselineRunsLoading = false;
      workspaceRenderAnalysisRunPanel();
    }
  }
  return workspaceState.baselineRuns;
}

async function workspaceLoadAnalysisRunLog(jobId) {
  try {
    var response = await fetch("/api/analysis-jobs/" + encodeURIComponent(jobId) + "/log?tail=240", { cache: "no-store" });
    var payload = await response.json();
    if (response.ok && workspaceState.analysisRunJob && workspaceState.analysisRunJob.job_id === jobId) {
      byId("analysisRunLog").textContent = payload.log ? payload.log.text : "";
    }
  } catch (_error) {
    // Job state remains available if the log is not created yet.
  }
}

async function workspaceStartAnalysisRun(event) {
  if (event) event.preventDefault();
  var scope = byId("analysisRunScope").value;
  if (workspaceState.analysisEnvironment && !workspaceState.analysisEnvironment.ready) {
    workspaceAnalysisRunStatus(
      "Analysis environment unavailable: "
        + (workspaceState.analysisEnvironment.error || "install the project-local environment first."),
      "error"
    );
    return;
  }
  var runValues = {};
  ANALYSIS_METHODS.forEach(function (method) {
    var selected = workspaceSelectedAnalysisRuns(method);
    runValues[method] = method === "rynnvalue" ? selected : (selected[0] || "");
  });
  if (!workspaceAnalysisRunScopeCount(scope)) {
    workspaceAnalysisRunStatus("This scope is empty; temporal analysis cannot run.", "warning");
    return;
  }
  if (ANALYSIS_METHODS.some(function (method) {
    return !runValues[method] || (Array.isArray(runValues[method]) && !runValues[method].length);
  })) {
    workspaceAnalysisRunStatus("Select a compatible completed run for SAFE, ProcVLM, RynnValue, and Robo-Dopamine.", "warning");
    return;
  }
  var pre = Number(byId("analysisPreWindow").value);
  var post = Number(byId("analysisPostWindow").value);
  var stride = Number(byId("analysisBackgroundStride").value);
  var label = byId("analysisOutputLabel").value.trim();
  if (![pre, post, stride].every(function (value) { return Number.isInteger(value) && value >= 1; })) {
    workspaceAnalysisRunStatus("Window and stride values must be positive integers.", "error");
    return;
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(label)) {
    workspaceAnalysisRunStatus("Output label may contain only letters, numbers, dot, underscore, or hyphen.", "error");
    byId("analysisOutputLabel").focus();
    return;
  }
  var button = byId("analysisRunButton");
  button.disabled = true;
  workspaceAnalysisRunBadge("queued");
  workspaceAnalysisRunStatus("Starting temporal analysis for " + workspaceAnalysisRunScopeCount(scope) + " rollout(s)…", "");
  byId("analysisRunLog").textContent = "";
  try {
    var response = await fetch("/api/analysis/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scope: scope,
        runs: runValues,
        pre_window_frames: pre,
        post_window_frames: post,
        background_stride_frames: stride,
        output_label: label,
        allow_partial_coverage: workspaceAnalysisRynnCoverage(scope).covered < workspaceAnalysisRynnCoverage(scope).total
      })
    });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not start temporal analysis");
    workspaceState.analysisRunJob = payload.job;
    workspaceJobChanged(payload.job);
    await workspaceLoadAnalysisRunLog(payload.job.job_id);
    workspacePollAnalysisRun(payload.job.job_id);
  } catch (error) {
    workspaceState.analysisRunJob = null;
    workspaceAnalysisRunBadge("failed");
    workspaceAnalysisRunStatus("Temporal analysis error: " + error.message, "error");
    button.disabled = false;
  }
}

async function workspacePollAnalysisRun(jobId) {
  try {
    var response = await fetch("/api/analysis-jobs/" + encodeURIComponent(jobId), { cache: "no-store" });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not read temporal-analysis job");
    var job = payload.job;
    if (!workspaceState.analysisRunJob || workspaceState.analysisRunJob.job_id !== jobId) return;
    workspaceState.analysisRunJob = job;
    workspaceJobChanged(job);
    workspaceAnalysisRunBadge(job.status);
    await workspaceLoadAnalysisRunLog(jobId);
    if (job.status === "queued" || job.status === "running") {
      workspaceAnalysisRunStatus("Temporal analysis " + job.status + " · " + job.selected_rollouts + " rollout(s) · output is being written to a temporary task directory…", "");
      window.setTimeout(function () { workspacePollAnalysisRun(jobId); }, 1500);
      return;
    }
    byId("analysisRunButton").disabled = false;
    if (job.status === "complete") {
      workspaceAnalysisRunStatus("Temporal analysis complete · new snapshot: " + (job.output_dir || "unknown"), "");
      workspaceLoadAnalysis(true);
      workspaceLoadBaselineRuns(job.scope, true);
    } else {
      workspaceAnalysisRunStatus("Temporal analysis failed; inspect the log below.", "error");
    }
  } catch (error) {
    if (workspaceState.analysisRunJob && workspaceState.analysisRunJob.job_id === jobId) {
      workspaceAnalysisRunBadge("failed");
      workspaceAnalysisRunStatus("Temporal-analysis job error: " + error.message, "error");
      byId("analysisRunButton").disabled = false;
    }
  }
}

async function workspaceLoadAnalysis(force) {
  if (workspaceState.analysisLoading || (workspaceState.analysisLoaded && !force)) {
    workspaceRenderSnapshot();
    return;
  }
  workspaceState.analysisLoading = true;
  workspaceRenderSnapshot();
  try {
    var response = await fetch("/api/analysis", { cache: "no-store" });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not load analysis snapshot");
    workspaceState.analysisSnapshot = payload.analysis || { available: false, message: "Empty analysis response" };
    workspaceState.analysisLoaded = true;
  } catch (error) {
    workspaceState.analysisSnapshot = { available: false, message: "Analysis snapshot error: " + error.message };
    workspaceState.analysisLoaded = true;
  } finally {
    workspaceState.analysisLoading = false;
    workspaceRenderSnapshot();
  }
}
