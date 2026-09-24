"use strict";

/* Results run catalog, selection, and progressive refresh. */

function baselineRunPreferenceKey(method, record, condition) {
  return String(condition || "full_instruction") + "::" + String(record && record.id || "") + "::" + String(method || "");
}

function baselineRunAllKey(method, condition) {
  return String(condition || "full_instruction") + "::" + String(method || "");
}

function hasOwn(object, key) {
  return Object.prototype.hasOwnProperty.call(object, key);
}

function baselineRunOverride(method, record, condition) {
  var allKey = baselineRunAllKey(method, condition);
  if (hasOwn(state.baselineRunAll, allKey) && state.baselineRunAll[allKey] !== BASELINE_AUTO_RUN) {
    return state.baselineRunAll[allKey];
  }
  var localKey = baselineRunPreferenceKey(method, record, condition);
  if (hasOwn(state.baselineRunSelections, localKey)
      && state.baselineRunSelections[localKey] !== BASELINE_AUTO_RUN) {
    return state.baselineRunSelections[localKey];
  }
  return "";
}

function baselineRunSelectionValue(method, record, condition) {
  var allKey = baselineRunAllKey(method, condition);
  if (hasOwn(state.baselineRunAll, allKey)) return state.baselineRunAll[allKey];
  var localKey = baselineRunPreferenceKey(method, record, condition);
  if (hasOwn(state.baselineRunSelections, localKey)) return state.baselineRunSelections[localKey];
  return BASELINE_AUTO_RUN;
}

function baselineRunDisplay(run) {
  var root = String(run && run.run_root || "unknown");
  var parts = root.split("/");
  var shortRoot = parts.slice(Math.max(0, parts.length - 2)).join("/");
  var completed = run && run.completed_jobs != null ? run.completed_jobs : 0;
  var selected = run && run.selected_rollouts != null ? run.selected_rollouts : (run && run.run_rollout_count || 0);
  var when = run && (run.completed_at || run.created_at);
  var date = when ? formatDate(when) : "unknown time";
  var status = run && run.status ? " · " + run.status : "";
  var procedureMode = run && run.procedure_mode
    ? " · " + String(run.procedure_mode)
    : "";
  return date + " · " + completed + "/" + selected + status + procedureMode + " · " + shortRoot;
}

function baselineRunOptions(method, record, condition) {
  if (!Array.isArray(state.baselineRuns) || !record) return [];
  return state.baselineRuns.filter(function (run) {
    if ((run.method || run.baseline) !== method) return false;
    var ids = Array.isArray(run.run_source_rollout_ids)
      ? run.run_source_rollout_ids
      : (Array.isArray(run.run_rollout_ids) ? run.run_rollout_ids : []);
    if (ids.indexOf(record.id) === -1) return false;
    var runCondition = run.instruction_condition || "unknown";
    if (condition === "full_instruction") return runCondition === "full_instruction" || runCondition === "unknown";
    return runCondition === condition;
  });
}

function renderBaselineRunControls(method, result, record) {
  var condition = state.instructionCondition || "full_instruction";
  var selected = baselineRunSelectionValue(method, record, condition);
  var options = baselineRunOptions(method, record, condition);
  var optionHtml = '<option value="' + BASELINE_AUTO_RUN + '"' + (selected === BASELINE_AUTO_RUN ? " selected" : "") + '>Automatic · newest available</option>';
  options.forEach(function (run) {
    var root = String(run.run_root || "");
    optionHtml += '<option value="' + escapeHtml(root) + '" title="' + escapeHtml(root) + '"'
      + (selected === root ? " selected" : "") + '>'
      + escapeHtml(baselineRunDisplay(run)) + '</option>';
  });
  var allKey = baselineRunAllKey(method, condition);
  var applied = hasOwn(state.baselineRunAll, allKey) && state.baselineRunAll[allKey] !== BASELINE_AUTO_RUN;
  var note = applied ? "Applied to all rollouts" : "This rollout";
  var disabled = options.length === 0 ? " disabled" : "";
  return '<div class="evaluation-run-controls">'
    + '<label><span>Result run</span><select data-evaluation-run-select data-evaluation-method="' + escapeHtml(method)
    + '" aria-label="' + escapeHtml((result.label || method) + " result run") + '"' + disabled + '>'
    + optionHtml + '</select></label>'
    + '<button type="button" class="ghost-button" data-apply-baseline-run data-evaluation-method="' + escapeHtml(method)
    + '"' + disabled + '>Apply to all</button>'
    + (options.length ? "" : '<small>No completed run available</small>')
    + '</div>';
}

var resultsCatalogSerial = 0;
var resultsRefreshTimer = null;

function loadBaselineRunCatalog(condition, force) {
  condition = condition || state.instructionCondition || "full_instruction";
  if (!force && Array.isArray(state.baselineRuns) && state.baselineRunsCondition === condition) {
    return Promise.resolve(state.baselineRuns);
  }
  if (!force && state.baselineRunsLoading && state.baselineRunsCondition === condition) {
    return state.baselineRunsLoading;
  }

  var serial = ++resultsCatalogSerial;
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
      if (serial === resultsCatalogSerial) {
        state.baselineRuns = runs;
        state.baselineRunsCondition = payload.condition || condition;
        state.baselineRunNotice = "";
      }
      return runs;
    })
    .catch(function (error) {
      if (serial === resultsCatalogSerial) {
        state.baselineRuns = [];
        state.baselineRunsCondition = condition;
        state.baselineRunNotice = error.message;
      }
      return [];
    })
    .finally(function () {
      if (serial === resultsCatalogSerial) state.baselineRunsLoading = null;
    });

  state.baselineRunsLoading = request;
  return request;
}

function resultsCompletedCount(job) {
  var value = Number(job && job.completed_jobs);
  return Number.isFinite(value) ? value : 0;
}

function resultsMatchingProgressiveRun(job, record, condition) {
  if (!job || !record) return null;
  var options = baselineRunOptions(job.baseline, record, condition) || [];
  var jobRoot = String(job.run_root || "");
  for (var index = 0; index < options.length; index += 1) {
    var run = options[index];
    if (jobRoot && String(run.run_root || "") !== jobRoot) continue;
    if (!jobRoot && run.status !== "running" && run.status !== "complete"
        && run.status !== "complete_with_errors") continue;
    return run;
  }
  return null;
}

function resultsCurrentDisplayedRunRoot(method) {
  var result = state.evaluation && state.evaluation.methods
    ? state.evaluation.methods[method]
    : null;
  return String(result && result.run && result.run.run_root || "");
}

function resultsOnPersistentJobChanged(previous, job) {
  if (!job || job.job_type !== "baseline"
      || resultsCompletedCount(job) <= resultsCompletedCount(previous)) return;
  if (resultsRefreshTimer) window.clearTimeout(resultsRefreshTimer);
  resultsRefreshTimer = window.setTimeout(function () {
    resultsRefreshTimer = null;
    var rolloutId = state.selectedId;
    var condition = state.instructionCondition || "full_instruction";
    var record = (state.rollouts || []).find(function (item) {
      return item.id === rolloutId;
    }) || null;
    loadBaselineRunCatalog(condition, true).then(function () {
      if (!rolloutId || state.selectedId !== rolloutId
          || state.instructionCondition !== condition) return;
      var run = resultsMatchingProgressiveRun(job, record, condition);
      if (!run) return;
      if (resultsCurrentDisplayedRunRoot(job.baseline) === String(run.run_root || "")) return;
      loadEvaluation(rolloutId);
    });
  }, 180);
}
