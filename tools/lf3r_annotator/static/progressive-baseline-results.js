"use strict";

(function installProgressiveBaselineResults() {
  if (!window.state || window.__lf3rProgressiveBaselineResultsInstalled) return;
  window.__lf3rProgressiveBaselineResultsInstalled = true;

  var originalLoadCatalog = window.loadBaselineRunCatalog;
  var originalRememberJob = window.rememberPersistentJob;
  if (typeof originalLoadCatalog !== "function" || typeof originalRememberJob !== "function") return;

  var catalogSerial = 0;
  var refreshTimer = null;

  function loadCatalog(condition, force) {
    condition = condition || state.instructionCondition || "full_instruction";
    if (!force && Array.isArray(state.baselineRuns) && state.baselineRunsCondition === condition) {
      return Promise.resolve(state.baselineRuns);
    }
    if (!force && state.baselineRunsLoading && state.baselineRunsCondition === condition) {
      return state.baselineRunsLoading;
    }

    var serial = ++catalogSerial;
    state.baselineRunsCondition = condition;
    var request = fetch(
      "/api/baselines/runs?scope=all&condition=" + encodeURIComponent(condition),
      { cache: "no-store" }
    )
      .then(function (response) {
        if (!response.ok) throw new Error("Could not load baseline run catalog");
        return response.json();
      })
      .then(function (payload) {
        var runs = payload.runs || [];
        if (serial === catalogSerial) {
          state.baselineRuns = runs;
          state.baselineRunsCondition = payload.condition || condition;
          state.baselineRunNotice = "";
        }
        return runs;
      })
      .catch(function (error) {
        if (serial === catalogSerial) {
          state.baselineRuns = [];
          state.baselineRunsCondition = condition;
          state.baselineRunNotice = error.message;
        }
        return [];
      })
      .finally(function () {
        if (serial === catalogSerial) state.baselineRunsLoading = null;
      });

    state.baselineRunsLoading = request;
    return request;
  }

  window.loadBaselineRunCatalog = loadCatalog;
  try { loadBaselineRunCatalog = loadCatalog; } catch (_) {}

  function completedCount(job) {
    var value = Number(job && job.completed_jobs);
    return Number.isFinite(value) ? value : 0;
  }

  function matchingProgressiveRun(job, record, condition) {
    if (!job || !record || typeof window.baselineRunOptions !== "function") return null;
    var options = window.baselineRunOptions(job.baseline, record, condition) || [];
    var jobRoot = String(job.run_root || "");
    for (var index = 0; index < options.length; index += 1) {
      var run = options[index];
      if (jobRoot && String(run.run_root || "") !== jobRoot) continue;
      if (!jobRoot && run.status !== "running" && run.status !== "complete" && run.status !== "complete_with_errors") continue;
      return run;
    }
    return null;
  }

  function currentDisplayedRunRoot(method) {
    var result = state.evaluation && state.evaluation.methods
      ? state.evaluation.methods[method]
      : null;
    return String(result && result.run && result.run.run_root || "");
  }

  function scheduleResultRefresh(job) {
    if (refreshTimer) window.clearTimeout(refreshTimer);
    refreshTimer = window.setTimeout(function () {
      refreshTimer = null;
      var rolloutId = state.selectedId;
      var condition = state.instructionCondition || "full_instruction";
      var record = (state.rollouts || []).find(function (item) {
        return item.id === rolloutId;
      }) || null;

      loadCatalog(condition, true).then(function () {
        if (!rolloutId || state.selectedId !== rolloutId || state.instructionCondition !== condition) return;
        var run = matchingProgressiveRun(job, record, condition);
        if (!run) return;
        if (currentDisplayedRunRoot(job.baseline) === String(run.run_root || "")) return;
        if (typeof window.loadEvaluation === "function") {
          window.loadEvaluation(rolloutId);
        }
      });
    }, 180);
  }

  function rememberJob(job) {
    if (!job || !job.job_id) return originalRememberJob(job);
    var previous = (state.persistentJobs || []).find(function (item) {
      return item.job_id === job.job_id;
    }) || null;
    var previousCompleted = completedCount(previous);
    var nextCompleted = completedCount(job);

    originalRememberJob(job);

    // A worker writes its per-rollout result before completed_jobs advances.
    // Refresh the catalog only when a new rollout has actually reached that
    // completed boundary, not on every 1.5 s status poll.
    if (job.job_type === "baseline" && nextCompleted > previousCompleted) {
      scheduleResultRefresh(job);
    }
    return job;
  }

  window.rememberPersistentJob = rememberJob;
  try { rememberPersistentJob = rememberJob; } catch (_) {}
})();
