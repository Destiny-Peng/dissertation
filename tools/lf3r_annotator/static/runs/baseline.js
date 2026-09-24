"use strict";

/* Baseline batch execution for Runs. */

var BASELINE_BATCH_SCOPE_LABELS = {
  all: "All loaded rollouts",
  libero_10: "LIBERO-10",
  libero_spatial: "LIBERO-Spatial",
  controlled_analysis: "Controlled"
};

function baselineBatchMatchesScope(record, scope) {
  if (window.LF3RDatasetScopes
      && typeof window.LF3RDatasetScopes.matchesBaseline === "function") {
    return window.LF3RDatasetScopes.matchesBaseline(record, scope);
  }
  if (scope === "all") return true;
  if (scope === "libero_10" || scope === "libero_spatial") return record.task_suite === scope;
  return record.analysis_partition === scope;
}

function baselineBatchUsesWorkers() {
  // All baseline methods now share the rollout-level worker allocator.
  return true;
}

function baselineBatchIsRynnValue() {
  var method = byId("baselineBatchMethod");
  return Boolean(method && method.value === "rynnvalue");
}

function baselineBatchUnfilteredScopeRecords() {
  var scope = byId("baselineBatchScope");
  if (!scope) return [];
  var conditionNode = byId("baselineBatchCondition");
  var condition = conditionNode ? conditionNode.value : (state.instructionCondition || "full_instruction");
  return (state.rollouts || []).filter(function (record) {
    if (!baselineBatchMatchesScope(record, scope.value)) return false;
    if (condition === "full_instruction") return true;
    return Boolean(record.instruction_variants && record.instruction_variants[condition]);
  });
}

function baselineBatchResultFilterValue() {
  var node = byId("baselineBatchResultFilter");
  return node ? (node.value || "all") : "all";
}

function baselineBatchCoverageKey() {
  var method = byId("baselineBatchMethod");
  var scope = byId("baselineBatchScope");
  var condition = byId("baselineBatchCondition");
  return [
    method ? method.value : "",
    scope ? scope.value : "",
    condition ? condition.value : "full_instruction"
  ].join("::");
}

function baselineBatchScopeRecords() {
  var records = baselineBatchUnfilteredScopeRecords();
  if (baselineBatchResultFilterValue() !== "missing_valid") return records;
  var key = baselineBatchCoverageKey();
  var coverage = state.baselineBatchCoverage;
  if (!coverage || state.baselineBatchCoverageKey !== key) return [];
  var missing = {};
  (coverage.missing_source_rollout_ids || []).forEach(function (rolloutId) {
    missing[String(rolloutId)] = true;
  });
  return records.filter(function (record) {
    return Boolean(missing[String(record.id)]);
  });
}

async function loadBaselineBatchCoverage(force) {
  if (baselineBatchResultFilterValue() !== "missing_valid") {
    state.baselineBatchCoverage = null;
    state.baselineBatchCoverageKey = null;
    state.baselineBatchCoverageLoading = false;
    return null;
  }
  var key = baselineBatchCoverageKey();
  if (!force && state.baselineBatchCoverage && state.baselineBatchCoverageKey === key) {
    return state.baselineBatchCoverage;
  }
  var requestId = ++state.baselineBatchCoverageRequest;
  var method = byId("baselineBatchMethod").value;
  var scope = byId("baselineBatchScope").value;
  var condition = byId("baselineBatchCondition").value || "full_instruction";
  state.baselineBatchCoverageLoading = true;
  var note = byId("baselineBatchSelection");
  if (note) note.textContent = "Checking existing " + method + " results…";
  try {
    var response = await fetch(
      "/api/baselines/result-coverage?baseline=" + encodeURIComponent(method)
      + "&scope=" + encodeURIComponent(scope)
      + "&condition=" + encodeURIComponent(condition),
      { cache: "no-store" }
    );
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not inspect baseline result coverage");
    if (requestId !== state.baselineBatchCoverageRequest || key !== baselineBatchCoverageKey()) return null;
    var previousIds = state.baselineBatchCoverage && state.baselineBatchCoverageKey === key
      ? JSON.stringify(state.baselineBatchCoverage.missing_source_rollout_ids || [])
      : null;
    var nextIds = JSON.stringify(payload.missing_source_rollout_ids || []);
    state.baselineBatchCoverage = payload;
    state.baselineBatchCoverageKey = key;
    state.baselineBatchCoverageLoading = false;
    if (previousIds !== nextIds) baselineBatchScopeChanged();
    else updateBaselineBatchSelection();
    return payload;
  } catch (error) {
    if (requestId !== state.baselineBatchCoverageRequest) return null;
    state.baselineBatchCoverage = null;
    state.baselineBatchCoverageKey = key;
    state.baselineBatchCoverageLoading = false;
    setBaselineBatchStatus("Baseline result filter error: " + error.message, "error");
    updateBaselineBatchSelection();
    return null;
  }
}

function baselineBatchCoverageChanged() {
  state.baselineBatchCoverage = null;
  state.baselineBatchCoverageKey = null;
  state.baselineBatchCoverageRequest += 1;
  if (baselineBatchResultFilterValue() === "missing_valid") {
    loadBaselineBatchCoverage(true);
  } else {
    baselineBatchScopeChanged();
  }
}

function baselineBatchNumber(value) {
  var number = Number(value);
  return Number.isInteger(number) ? number : null;
}

function baselineBatchTotalRange() {
  var records = baselineBatchScopeRecords();
  var startNode = byId("baselineBatchStartIndex");
  var endNode = byId("baselineBatchEndIndex");
  var start = baselineBatchNumber(startNode ? startNode.value : "");
  var endText = endNode ? endNode.value.trim() : "";
  var end = endText === "" ? records.length : baselineBatchNumber(endText);
  return {
    records: records,
    matched: records.length,
    start: start,
    end: end,
    valid: start != null && end != null && start >= 0 && end > start && end <= records.length
  };
}

function baselineBatchSelectionRecords() {
  if (baselineBatchUsesWorkers()) {
    var range = baselineBatchTotalRange();
    if (!range.valid) return [];
    var seen = {};
    var selected = [];
    (state.baselineBatchWorkers || []).forEach(function (worker) {
      var start = baselineBatchNumber(worker.start_index);
      var end = baselineBatchNumber(worker.end_index);
      if (start == null || end == null || start < range.start || end > range.end || end <= start) return;
      for (var index = start; index < end; index += 1) {
        if (seen[index]) continue;
        seen[index] = true;
        selected.push(range.records[index]);
      }
    });
    return selected;
  }
  var records = baselineBatchScopeRecords();
  var start = Number(byId("baselineBatchStartIndex").value || 0);
  var limitText = byId("baselineBatchLimit").value.trim();
  var limit = limitText === "" ? null : Number(limitText);
  if (!Number.isInteger(start) || start < 0) return [];
  if (limit != null && (!Number.isInteger(limit) || limit < 1)) return [];
  return limit == null ? records.slice(start) : records.slice(start, start + limit);
}

function baselineBatchRebalanceWorkers() {
  var range = baselineBatchTotalRange();
  var count = Math.max(1, (state.baselineBatchWorkers || []).length || 1);
  var oldWorkers = state.baselineBatchWorkers || [];
  var workers = [];
  if (range.start != null && range.end != null && range.end > range.start) {
    var total = range.end - range.start;
    var base = Math.floor(total / count);
    var remainder = total % count;
    var cursor = range.start;
    for (var index = 0; index < count; index += 1) {
      var width = base + (index < remainder ? 1 : 0);
      workers.push({
        gpu: oldWorkers[index] && String(oldWorkers[index].gpu || "0") || "0",
        start_index: cursor,
        end_index: cursor + width
      });
      cursor += width;
    }
  } else {
    for (var emptyIndex = 0; emptyIndex < count; emptyIndex += 1) {
      workers.push({
        gpu: oldWorkers[emptyIndex] && String(oldWorkers[emptyIndex].gpu || "0") || "0",
        start_index: 0,
        end_index: 0
      });
    }
  }
  state.baselineBatchWorkers = workers;
  renderBaselineBatchWorkers();
}

function readBaselineBatchWorkers() {
  var workers = [];
  document.querySelectorAll("#baselineBatchWorkers [data-worker-row]").forEach(function (row) {
    var gpu = row.querySelector('[data-worker-field="gpu"]');
    var start = row.querySelector('[data-worker-field="start_index"]');
    var end = row.querySelector('[data-worker-field="end_index"]');
    workers.push({
      gpu: gpu ? gpu.value.trim() : "",
      start_index: start ? start.value : "",
      end_index: end ? end.value : ""
    });
  });
  if (workers.length) state.baselineBatchWorkers = workers;
  return state.baselineBatchWorkers || [];
}

function renderBaselineBatchWorkers() {
  var container = byId("baselineBatchWorkers");
  if (!container) return;
  var workers = state.baselineBatchWorkers || [];
  container.innerHTML = workers.map(function (worker, index) {
    return '<div class="baseline-worker-row rynn-worker-row" data-worker-row data-worker-index="' + index + '">'
      + '<span class="baseline-worker-label rynn-worker-label">Worker ' + (index + 1) + '</span>'
      + '<label><span>GPU ID</span><input data-worker-field="gpu" type="text" inputmode="numeric" value="' + escapeHtml(worker.gpu) + '" aria-label="Worker ' + (index + 1) + ' GPU"></label>'
      + '<label><span>Start (inclusive)</span><input data-worker-field="start_index" type="number" min="0" step="1" value="' + escapeHtml(worker.start_index) + '" aria-label="Worker ' + (index + 1) + ' start index"></label>'
      + '<label><span>End (exclusive)</span><input data-worker-field="end_index" type="number" min="1" step="1" value="' + escapeHtml(worker.end_index) + '" aria-label="Worker ' + (index + 1) + ' end index"></label>'
      + '<button type="button" class="ghost-button baseline-worker-remove rynn-worker-remove" data-remove-worker="' + index + '"' + (workers.length <= 1 ? ' disabled' : '') + '>Remove</button>'
      + '</div>';
  }).join("");
}

function baselineBatchWorkerSummary() {
  var range = baselineBatchTotalRange();
  var workers = readBaselineBatchWorkers();
  var counts = {};
  var gpuCounts = {};
  var rows = [];
  var invalid = !range.valid;
  workers.forEach(function (worker, workerIndex) {
    var start = baselineBatchNumber(worker.start_index);
    var end = baselineBatchNumber(worker.end_index);
    var gpu = String(worker.gpu || "").trim();
    var valid = /^\d+$/.test(gpu) && start != null && end != null
      && start >= (range.start == null ? 0 : range.start)
      && end <= (range.end == null ? 0 : range.end) && end > start;
    if (!valid) invalid = true;
    if (/^\d+$/.test(gpu)) gpuCounts[gpu] = (gpuCounts[gpu] || 0) + 1;
    if (valid) {
      for (var index = start; index < end; index += 1) counts[index] = (counts[index] || 0) + 1;
    }
    rows.push("W" + (workerIndex + 1) + " GPU " + (gpu || "?") + " [" + (start == null ? "?" : start) + "," + (end == null ? "?" : end) + ") = " + (valid ? end - start : "invalid"));
  });
  var overlap = Object.keys(counts).filter(function (index) { return counts[index] > 1; }).map(Number);
  var gaps = [];
  if (range.valid) {
    for (var expected = range.start; expected < range.end; expected += 1) {
      if (!counts[expected]) gaps.push(expected);
    }
  }
  var unique = Object.keys(counts).length;
  var reusedGpu = Object.keys(gpuCounts).filter(function (gpu) { return gpuCounts[gpu] > 1; });
  return {
    range: range,
    invalid: invalid,
    overlap: overlap,
    gaps: gaps,
    unique: unique,
    reusedGpu: reusedGpu,
    text: rows.join(" · ")
  };
}

function updateBaselineBatchSelection() {
  var scope = byId("baselineBatchScope");
  var note = byId("baselineBatchSelection");
  var button = byId("baselineBatchRun");
  if (!scope || !note || !button) return;
  var matched = baselineBatchScopeRecords().length;
  var conditionNode = byId("baselineBatchCondition");
  var condition = conditionNode ? conditionNode.value : (state.instructionCondition || "full_instruction");
  var conditionLabel = instructionConditionLabel(condition);
  var label = window.LF3RDatasetScopes
    && typeof window.LF3RDatasetScopes.scopeLabel === "function"
    ? window.LF3RDatasetScopes.scopeLabel(scope.value)
    : (BASELINE_BATCH_SCOPE_LABELS[scope.value] || scope.value);
  var resultFilter = baselineBatchResultFilterValue();
  var coverage = (
    resultFilter === "missing_valid"
    && state.baselineBatchCoverageKey === baselineBatchCoverageKey()
  ) ? state.baselineBatchCoverage : null;
  var coverageText = "";
  if (resultFilter === "missing_valid") {
    if (state.baselineBatchCoverageLoading) {
      coverageText = "checking existing results; ";
    } else if (!coverage) {
      coverageText = "existing-result coverage unavailable; ";
    } else {
      coverageText = coverage.missing_valid_result_rollouts + " complete annotation(s) without valid result / "
        + coverage.complete_annotation_rollouts + " complete; "
        + coverage.valid_result_rollouts + " existing valid result(s) skipped; "
        + coverage.incomplete_annotation_rollouts + " in-progress/unreviewed skipped; ";
    }
  }
  if (baselineBatchUsesWorkers()) {
    var summary = baselineBatchWorkerSummary();
    var rangeText = summary.range.start == null || summary.range.end == null
      ? "invalid total range" : "total [" + summary.range.start + "," + summary.range.end + ")";
    note.textContent = conditionLabel + " · " + label + ": " + coverageText + matched + " rollout(s) available; " + rangeText
      + "; unique execution " + summary.unique + ". "
      + (summary.overlap.length ? "Overlap " + summary.overlap.length + ". " : "No overlap. ")
      + (summary.gaps.length ? "Gap " + summary.gaps.length + ". " : "No gap. ")
      + (summary.reusedGpu.length ? "GPU reused: " + summary.reusedGpu.join(", ") + "." : "");
    var workerNote = byId("baselineBatchWorkerSummary");
    if (workerNote) {
      workerNote.textContent = summary.text + " · Total " + (summary.range.valid ? summary.range.end - summary.range.start : "invalid")
        + " · Unique " + summary.unique
        + (summary.overlap.length ? " · overlap " + summary.overlap.length : "")
        + (summary.gaps.length ? " · gap " + summary.gaps.length : "")
        + (summary.reusedGpu.length ? " · repeated GPU " + summary.reusedGpu.join(", ") : "");
    }
    button.disabled = summary.invalid || summary.unique === 0 || state.baselineBatchSubmitting;
    return;
  }
  var selected = baselineBatchSelectionRecords().length;
  note.textContent = conditionLabel + " · " + label + ": " + coverageText + matched + " rollout(s) available; " + selected + " selected after index/limit.";
  button.disabled = selected === 0 || state.baselineBatchSubmitting;
}

function updateBaselineBatchAdvancedFields() {
  var method = byId("baselineBatchMethod").value;
  var rynn = method === "rynnvalue";
  document.querySelectorAll("[data-batch-methods]").forEach(function (label) {
    var methods = label.dataset.batchMethods.split(",");
    var visible = methods.indexOf(method) !== -1;
    label.classList.toggle("hidden", !visible);
    label.querySelectorAll("[data-batch-option]").forEach(function (input) { input.disabled = !visible; });
  });
  var parallelPanel = byId("baselineBatchRynnParallel");
  if (parallelPanel) parallelPanel.classList.remove("hidden");
  var endField = byId("baselineBatchEndField");
  var limitField = byId("baselineBatchLimitField");
  if (endField) endField.classList.remove("hidden");
  if (limitField) limitField.classList.add("hidden");
  var gpuNode = byId("baselineBatchGpu");
  if (gpuNode && gpuNode.closest("label")) gpuNode.closest("label").classList.add("hidden");

  var range = baselineBatchTotalRange();
  var endNode = byId("baselineBatchEndIndex");
  if (endNode && !endNode.value.trim() && range.matched) endNode.value = range.matched;
  var workers = state.baselineBatchWorkers || [];
  var hasUsableWorker = workers.some(function (worker) {
    var workerStart = baselineBatchNumber(worker.start_index);
    var workerEnd = baselineBatchNumber(worker.end_index);
    return workerStart != null && workerEnd != null && workerEnd > workerStart;
  });
  if (!workers.length || !hasUsableWorker) {
    state.baselineBatchWorkers = [{
      gpu: workers[0] && String(workers[0].gpu || "0") || "0",
      start_index: range.start == null ? 0 : range.start,
      end_index: range.end == null ? 0 : range.end
    }];
  }
  renderBaselineBatchWorkers();
  var heading = parallelPanel && parallelPanel.querySelector("strong");
  if (heading) heading.textContent = method.toUpperCase() + " rollout workers";
  updateBaselineBatchSelection();
}

function baselineBatchScopeChanged() {
  var records = baselineBatchScopeRecords();
  byId("baselineBatchStartIndex").value = "0";
  byId("baselineBatchLimit").value = "";
  if (byId("baselineBatchEndIndex")) byId("baselineBatchEndIndex").value = String(records.length);
  if (baselineBatchUsesWorkers()) baselineBatchRebalanceWorkers();
  updateBaselineBatchSelection();
}

function baselineBatchRangeChanged() {
  if (baselineBatchUsesWorkers()) baselineBatchRebalanceWorkers();
  else updateBaselineBatchSelection();
}

function addBaselineBatchWorker() {
  readBaselineBatchWorkers();
  state.baselineBatchWorkers.push({ gpu: "0", start_index: 0, end_index: 0 });
  baselineBatchRebalanceWorkers();
  updateBaselineBatchSelection();
}

function removeBaselineBatchWorker(index) {
  readBaselineBatchWorkers();
  if (state.baselineBatchWorkers.length <= 1) return;
  state.baselineBatchWorkers.splice(index, 1);
  baselineBatchRebalanceWorkers();
  updateBaselineBatchSelection();
}

function baselineBatchOptions() {
  var options = {};
  document.querySelectorAll("[data-batch-option]").forEach(function (input) {
    if (input.disabled) return;
    if (input.type === "checkbox") {
      if (input.checked) options[input.dataset.batchOption] = true;
      return;
    }
    if (input.value.trim() === "") return;
    options[input.dataset.batchOption] = input.type === "number" ? Number(input.value) : input.value.trim();
  });
  return options;
}

function setBaselineBatchStatus(message, kind) {
  var status = byId("baselineBatchStatus");
  status.textContent = message;
  status.className = "evaluation-status" + (kind ? " " + kind : "");
}

async function loadBaselineBatchLog(jobId) {
  if (window.LF3RRunsJobs && typeof window.LF3RRunsJobs.refreshSelectedLog === "function") {
    return window.LF3RRunsJobs.refreshSelectedLog("baseline", jobId);
  }
  try {
    var response = await fetch("/api/baseline-jobs/" + encodeURIComponent(jobId) + "/log?tail=200", { cache: "no-store" });
    var payload = await response.json();
    if (response.ok && state.baselineBatchJob && state.baselineBatchJob.job_id === jobId) {
      byId("baselineBatchLog").textContent = payload.log ? payload.log.text : "";
    }
  } catch (_error) {
    // The status endpoint remains useful when a log is not ready yet.
  }
}

function setRejectedRequestLog(message, body) {
  var log = byId("baselineBatchLog");
  if (!log) return;
  log.textContent = [
    "Request rejected before a baseline job was created.",
    "",
    String(message || "Unknown request validation error"),
    "",
    "Submitted request:",
    JSON.stringify(body, null, 2)
  ].join("\n");
}

async function startBaselineBatch(event) {
  if (event) event.preventDefault();
  if (state.baselineBatchSubmitting) return;

  var method = byId("baselineBatchMethod").value;
  var scope = byId("baselineBatchScope").value;
  var condition = byId("baselineBatchCondition").value || "full_instruction";
  var resultFilter = baselineBatchResultFilterValue();
  var memory = Number(byId("baselineBatchMemoryUtilization").value);

  if (resultFilter === "missing_valid") {
    var coverage = await loadBaselineBatchCoverage(true);
    if (!coverage) {
      setBaselineBatchStatus("Could not verify existing baseline result coverage.", "error");
      return;
    }
    if (!coverage.missing_valid_result_rollouts) {
      setBaselineBatchStatus(
        "Every matching rollout already has a valid " + method + " result; nothing to run.",
        "warning"
      );
      return;
    }
  }

  readBaselineBatchWorkers();
  var range = baselineBatchTotalRange();
  var summary = baselineBatchWorkerSummary();
  var workers = (state.baselineBatchWorkers || []).map(function (worker) {
    return {
      gpu: String(worker.gpu || "").trim(),
      start_index: baselineBatchNumber(worker.start_index),
      end_index: baselineBatchNumber(worker.end_index)
    };
  });

  if (!range.valid || summary.invalid) {
    setBaselineBatchStatus("Total range and every worker must use valid ranges inside the scope.", "error");
    return;
  }
  if (workers.some(function (worker) { return !/^\d+$/.test(worker.gpu); })) {
    setBaselineBatchStatus("Every worker GPU must be one numeric CUDA device index.", "error");
    return;
  }
  if (!Number.isFinite(memory) || memory <= 0 || memory > 1) {
    setBaselineBatchStatus("Free-memory fraction must be between 0 and 1.", "error");
    byId("baselineBatchMemoryUtilization").focus();
    return;
  }

  var selected = baselineBatchSelectionRecords();
  if (!selected.length) {
    setBaselineBatchStatus("This scope and worker assignment selects no rollouts.", "warning");
    return;
  }

  var uniqueGpus = [];
  workers.forEach(function (worker) {
    if (uniqueGpus.indexOf(worker.gpu) === -1) uniqueGpus.push(worker.gpu);
  });
  var gpu = uniqueGpus.join(",") || "0";

  var confirmation = "Run " + method + " for " + instructionConditionLabel(condition)
    + " over " + selected.length + " unique rollout(s)? This launches GPU inference."
    + (resultFilter === "missing_valid"
      ? " Rollouts with an existing valid result will be skipped by the server."
      : "")
    + " It will start " + workers.length + " rollout worker(s).";
  if (!window.confirm(confirmation)) return;

  state.baselineBatchSubmitting = true;
  updateBaselineBatchSelection();
  setBaselineBatchStatus("Starting " + method + " batch for " + selected.length + " rollout(s)…", "");
  byId("baselineBatchLog").textContent = "";

  var batchOptions = baselineBatchOptions();
  if (method === "procvlm" && window.LF3RProcvlmMode
      && typeof window.LF3RProcvlmMode.applyOptions === "function") {
    batchOptions = window.LF3RProcvlmMode.applyOptions(batchOptions, "batch");
  }

  var body = {
    baseline: method,
    scope: scope,
    instruction_condition: condition,
    result_filter: resultFilter,
    gpu: gpu,
    memory_utilization: memory,
    start_index: range.start,
    limit: null,
    options: batchOptions
  };

  if (workers.length === 1) {
    body.parallel_workers = 1;
    if (!(range.start === 0 && range.end === range.matched)) {
      body.end_index = range.end;
    }
  } else {
    body.end_index = range.end;
    body.parallel_workers = workers.length;
    body.workers = workers;
  }

  try {
    var response = await fetch("/api/baselines/run-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    var responseText = await response.text();
    var payload = {};
    try {
      payload = responseText ? JSON.parse(responseText) : {};
    } catch (_) {
      payload = { error: responseText || "Server returned a non-JSON response" };
    }

    if (!response.ok) {
      var message = payload.error || ("HTTP " + response.status + " while starting baseline batch");
      setRejectedRequestLog(message, body);
      console.error("LF3R baseline batch request rejected", {
        status: response.status,
        message: message,
        request: body,
        response: payload
      });
      throw new Error(message);
    }

    state.baselineBatchJob = payload.job;
    rememberPersistentJob(payload.job);
    await loadBaselineBatchLog(payload.job.job_id);
    pollBaselineBatchJob(payload.job.job_id);
  } catch (error) {
    setBaselineBatchStatus("Batch baseline error: " + error.message, "error");
    if (!byId("baselineBatchLog").textContent.trim()) {
      setRejectedRequestLog(error.message, body);
    }
  } finally {
    state.baselineBatchSubmitting = false;
    updateBaselineBatchSelection();
  }
}

async function pollBaselineBatchJob(jobId) {
  return pollPersistentJob(jobId);
}
