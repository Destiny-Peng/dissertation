"use strict";

/* Analysis live module. Runtime state is owned by workspace-core.js. */

function workspaceNumber(value) {
  var number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function workspacePercent(value, digits) {
  var number = workspaceNumber(value);
  if (number == null) return "n/a";
  return (number * 100).toFixed(digits == null ? 1 : digits) + "%";
}

function workspaceFormatNumber(value, digits) {
  var number = workspaceNumber(value);
  if (number == null) return "n/a";
  if (Math.abs(number) >= 1000 || (Math.abs(number) > 0 && Math.abs(number) < 0.001)) {
    return number.toExponential(2);
  }
  return number.toFixed(digits == null ? 2 : digits).replace(/0+$/, "").replace(/[.]$/, "");
}

function workspaceMedian(values) {
  if (!values.length) return null;
  var sorted = values.slice().sort(function (left, right) { return left - right; });
  var middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function workspaceQuantile(values, fraction) {
  if (!values.length) return null;
  var sorted = values.slice().sort(function (left, right) { return left - right; });
  if (sorted.length === 1) return sorted[0];
  var position = (sorted.length - 1) * fraction;
  var lower = Math.floor(position);
  var upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function workspaceSummary(values) {
  var clean = values.map(workspaceNumber).filter(function (value) { return value != null; });
  return {
    count: clean.length,
    median: workspaceMedian(clean),
    q25: workspaceQuantile(clean, 0.25),
    q75: workspaceQuantile(clean, 0.75),
    max: clean.length ? Math.max.apply(null, clean) : null
  };
}

function workspaceNormalizeOutcome(value) {
  if (value === "success" || value === "clean_success") return "clean_success";
  if (value === "failure" || value === "terminal_failure") return "terminal_failure";
  if (value === "recovered_success") return "recovered_success";
  return "uncertain";
}

function workspaceLiveEvents(record) {
  var annotation = record.annotation || {};
  var events = Array.isArray(annotation.failure_events) ? annotation.failure_events.slice() : [];
  if (!events.length) {
    var hasLegacyBoundary = [
      annotation.causal_onset_frame,
      annotation.observable_onset_frame,
      annotation.terminal_failure_frame,
      annotation.recovery_frame
    ].some(function (value) { return value != null; });
    if (hasLegacyBoundary) {
      events = [{
        failure_type: annotation.failure_type || "other",
        causal_onset_frame: annotation.causal_onset_frame,
        observable_onset_frame: annotation.observable_onset_frame,
        terminal_failure_frame: annotation.terminal_failure_frame,
        recovery_frame: annotation.recovery_frame
      }];
    }
  }
  return events;
}

function workspacePartitionMatches(record, partition) {
  if (window.LF3RDatasetScopes
      && typeof window.LF3RDatasetScopes.matchesPartition === "function") {
    return window.LF3RDatasetScopes.matchesPartition(record, partition);
  }
  if (partition === "all") return true;
  if (partition === "libero_10" || partition === "libero_spatial") {
    return record.task_suite === partition;
  }
  return record.analysis_partition === partition;
}

function workspaceFilteredLiveRecords() {
  var filters = workspaceState.liveFilters;
  return (state.rollouts || []).filter(function (record) {
    var outcome = workspaceNormalizeOutcome(effectiveOutcome(record));
    return workspacePartitionMatches(record, filters.partition)
      && (filters.suite === "all" || record.task_suite === filters.suite)
      && (filters.task === "all" || String(record.task_id) === String(filters.task))
      && (filters.outcome === "all" || outcome === filters.outcome);
  });
}

function workspaceSetOptions(select, options, selected, allLabel) {
  if (!select) return;
  var html = '<option value="all">' + escapeHtml(allLabel || "All") + "</option>";
  options.forEach(function (option) {
    html += '<option value="' + escapeHtml(option.value) + '">' + escapeHtml(option.label) + "</option>";
  });
  select.innerHTML = html;
  var values = options.map(function (option) { return String(option.value); });
  select.value = selected !== "all" && values.indexOf(String(selected)) !== -1 ? String(selected) : "all";
}

function workspaceRefreshLiveFilterOptions() {
  var suites = [];
  var tasks = [];
  (state.rollouts || []).forEach(function (record) {
    if (record.task_suite && suites.indexOf(record.task_suite) === -1) suites.push(record.task_suite);
    var taskKey = String(record.task_suite || "") + "::" + String(record.task_id);
    if (tasks.every(function (item) { return item.value !== String(record.task_id) || item.suite !== record.task_suite; })) {
      tasks.push({ value: String(record.task_id), suite: record.task_suite, key: taskKey });
    }
  });
  suites.sort();
  tasks.sort(function (left, right) {
    return String(left.suite).localeCompare(String(right.suite)) || Number(left.value) - Number(right.value);
  });
  workspaceSetOptions(
    byId("analysisSuiteFilter"),
    suites.map(function (suite) { return { value: suite, label: labelFor(suite) }; }),
    workspaceState.liveFilters.suite,
    "All suites"
  );
  workspaceSetOptions(
    byId("analysisTaskFilter"),
    tasks.map(function (task) { return { value: task.value, label: labelFor(task.suite) + " · task " + task.value }; }),
    workspaceState.liveFilters.task,
    "All tasks"
  );
  workspaceState.liveFilters.suite = byId("analysisSuiteFilter").value;
  workspaceState.liveFilters.task = byId("analysisTaskFilter").value;
}

function workspaceLiveStats(records) {
  var outcomeCounts = {};
  ANALYSIS_OUTCOMES.forEach(function (definition) { outcomeCounts[definition.value] = 0; });
  var failureTypes = {};
  var eventCount = 0;
  var observableCount = 0;
  var causalToObservable = [];
  var observableFraction = [];
  var observableToRecovery = [];
  records.forEach(function (record) {
    var outcome = workspaceNormalizeOutcome(effectiveOutcome(record));
    outcomeCounts[outcome] = (outcomeCounts[outcome] || 0) + 1;
    workspaceLiveEvents(record).forEach(function (event) {
      eventCount += 1;
      var type = event.failure_type || record.annotation && record.annotation.failure_type || "other";
      failureTypes[type] = (failureTypes[type] || 0) + 1;
      var observable = workspaceNumber(event.observable_onset_frame);
      var causal = workspaceNumber(event.causal_onset_frame);
      var recovery = workspaceNumber(event.recovery_frame);
      var totalFrames = workspaceNumber(record.total_frames);
      if (observable != null) {
        observableCount += 1;
        if (causal != null && causal <= observable) causalToObservable.push(observable - causal);
        if (totalFrames != null && totalFrames > 1) observableFraction.push(observable / (totalFrames - 1));
        if (recovery != null && recovery >= observable) observableToRecovery.push(recovery - observable);
      }
    });
  });
  var resolved = outcomeCounts.clean_success + outcomeCounts.recovered_success + outcomeCounts.terminal_failure;
  var successCount = outcomeCounts.clean_success + outcomeCounts.recovered_success;
  return {
    total: records.length,
    reviewed: records.filter(function (record) { return record.annotation_status === "complete"; }).length,
    resolved: resolved,
    resolvedSuccessRate: resolved ? successCount / resolved : null,
    outcomeCounts: outcomeCounts,
    failureTypes: failureTypes,
    eventCount: eventCount,
    observableCount: observableCount,
    causalToObservable: causalToObservable,
    observableFraction: observableFraction,
    observableToRecovery: observableToRecovery
  };
}

function workspaceRenderKpis(stats) {
  var cards = [
    ["Rollouts", stats.total, "current filtered scope"],
    ["Reviewed", stats.reviewed + " / " + stats.total, "review status = complete"],
    ["Resolved", stats.resolved, "uncertain excluded from rate"],
    ["Resolved success", workspacePercent(stats.resolvedSuccessRate), "clean + recovered / resolved"],
    ["Failure events", stats.eventCount, stats.observableCount + " with observable onset"],
    ["Observable coverage", workspacePercent(stats.eventCount ? stats.observableCount / stats.eventCount : null), "observable onset / failure events"]
  ];
  byId("analysisKpis").innerHTML = cards.map(function (card) {
    return '<div class="analysis-kpi"><div class="analysis-kpi-value">' + escapeHtml(card[1]) + '</div>'
      + '<div class="analysis-kpi-label">' + escapeHtml(card[0]) + '</div>'
      + '<div class="analysis-kpi-note">' + escapeHtml(card[2]) + "</div></div>";
  }).join("");
}

function workspaceSvg(width, height, label, content) {
  return '<svg class="analysis-svg" viewBox="0 0 ' + width + " " + height + '" role="img" aria-label="' + escapeHtml(label) + '"><title>'
    + escapeHtml(label) + "</title>" + content + "</svg>";
}

function workspaceEmpty(message) {
  return '<div class="analysis-empty">' + escapeHtml(message) + "</div>";
}

function workspaceRenderOutcomeChart(stats) {
  var total = stats.total;
  if (!total) return workspaceEmpty("No trajectories in the current scope.");
  var x = 178;
  var width = 620;
  var cursor = x;
  var content = '<text x="16" y="24" class="chart-label">n = ' + total + " trajectories</text>";
  ANALYSIS_OUTCOMES.forEach(function (definition) {
    var count = stats.outcomeCounts[definition.value] || 0;
    var segment = width * count / total;
    if (segment > 0) {
      content += '<rect class="chart-clickable" data-analysis-outcome="' + definition.value + '" x="' + cursor.toFixed(2) + '" y="46" width="' + segment.toFixed(2) + '" height="28" fill="' + definition.color + '"><title>'
        + escapeHtml(definition.label + ": " + count + " (" + workspacePercent(count / total) + ")") + "</title></rect>";
      cursor += segment;
    }
  });
  content += '<line class="chart-axis" x1="' + x + '" y1="75" x2="' + (x + width) + '" y2="75"></line>';
  content += '<text x="' + x + '" y="94">0</text><text x="' + (x + width - 16) + '" y="94">' + total + '</text>';
  return workspaceSvg(830, 112, "Trajectory outcome distribution", content)
    + '<div class="analysis-legend">' + ANALYSIS_OUTCOMES.map(function (definition) {
      return '<span><i style="background:' + definition.color + '"></i>' + escapeHtml(definition.label) + " " + (stats.outcomeCounts[definition.value] || 0) + "</span>";
    }).join("") + "</div>";
}

function workspaceRenderFailureTypeChart(stats) {
  var entries = Object.keys(stats.failureTypes).filter(function (type) { return type !== "none_success"; }).map(function (type) {
    return { type: type, count: stats.failureTypes[type] };
  }).sort(function (left, right) { return right.count - left.count || left.type.localeCompare(right.type); });
  if (!entries.length) return workspaceEmpty("No annotated failure events in the current scope.");
  var max = Math.max.apply(null, entries.map(function (entry) { return entry.count; }));
  var height = Math.max(125, 38 + entries.length * 27);
  var content = "";
  entries.forEach(function (entry, index) {
    var y = 30 + index * 27;
    var barWidth = 470 * entry.count / max;
    content += '<text x="12" y="' + (y + 12) + '" class="chart-label">' + escapeHtml(labelFor(entry.type)) + '</text>';
    content += '<rect x="180" y="' + y + '" width="' + barWidth.toFixed(2) + '" height="16" rx="3" fill="var(--failure)"><title>'
      + escapeHtml(labelFor(entry.type) + ": " + entry.count + " events") + "</title></rect>";
    content += '<text x="' + (194 + barWidth) + '" y="' + (y + 12) + '" class="chart-value">' + entry.count + "</text>";
  });
  return workspaceSvg(720, height, "Failure event type counts", content);
}

function workspaceRenderTimingChart(stats) {
  var definitions = [
    { label: "Causal → observable", values: stats.causalToObservable, suffix: " frames" },
    { label: "Observable position", values: stats.observableFraction, suffix: " of trajectory", fraction: true },
    { label: "Observable → recovery", values: stats.observableToRecovery, suffix: " frames" }
  ];
  var summaries = definitions.map(function (definition) { return workspaceSummary(definition.values); });
  var content = "";
  definitions.forEach(function (definition, index) {
    var summary = summaries[index];
    var y = 30 + index * 48;
    var plotX = 220;
    var plotWidth = 430;
    var axisMax = Math.max(summary.max || 0, 1);
    var plotMax = axisMax * 1.12;
    var scale = function (value) { return plotX + plotWidth * value / plotMax; };
    var axisLabel = workspaceFormatNumber(definition.fraction ? axisMax * 100 : axisMax) + (definition.fraction ? "%" : definition.suffix);
    content += '<text x="12" y="' + (y + 10) + '" class="chart-label">' + escapeHtml(definition.label) + '</text>';
    content += '<line class="chart-grid" x1="' + plotX + '" y1="' + (y + 6) + '" x2="' + (plotX + plotWidth) + '" y2="' + (y + 6) + '"></line>';
    content += '<text x="' + plotX + '" y="' + (y + 21) + '">0</text><text x="' + (plotX + plotWidth - 42) + '" y="' + (y + 21) + '">' + escapeHtml(axisLabel) + '</text>';
    if (summary.count) {
      var q25 = scale(summary.q25);
      var q75 = scale(summary.q75);
      var median = scale(summary.median);
      content += '<line x1="' + q25.toFixed(2) + '" y1="' + (y + 6) + '" x2="' + q75.toFixed(2) + '" y2="' + (y + 6) + '" stroke="var(--recovery)" stroke-width="5" stroke-linecap="round"></line>';
      content += '<circle cx="' + median.toFixed(2) + '" cy="' + (y + 6) + '" r="5" fill="var(--accent)"><title>'
        + escapeHtml("median " + workspaceFormatNumber(definition.fraction ? summary.median * 100 : summary.median) + (definition.fraction ? "%" : definition.suffix) + "; IQR " + workspaceFormatNumber(definition.fraction ? summary.q25 * 100 : summary.q25) + "–" + workspaceFormatNumber(definition.fraction ? summary.q75 * 100 : summary.q75) + (definition.fraction ? "%" : definition.suffix) + "; n=" + summary.count) + '</title></circle>';
      content += '<text x="' + (plotX + plotWidth + 12) + '" y="' + (y + 10) + '" class="chart-value">n=' + summary.count + "</text>";
    } else {
      content += '<text x="' + (plotX + 10) + '" y="' + (y + 10) + '">no paired observations</text>';
    }
  });
  return workspaceSvg(720, 190, "Onset and recovery timing distributions", content)
    + '<div class="analysis-legend"><span><i style="background:var(--recovery)"></i>IQR</span><span><i style="background:var(--accent)"></i>median</span><span>position is normalized by total_frames − 1</span></div>';
}

function workspaceRenderLiveAnalysis() {
  if (!byId("analysisView")) return;
  workspaceRefreshLiveFilterOptions();
  byId("analysisPartitionFilter").value = workspaceState.liveFilters.partition;
  byId("analysisOutcomeFilter").value = workspaceState.liveFilters.outcome;
  var records = workspaceFilteredLiveRecords();
  var stats = workspaceLiveStats(records);
  workspaceRenderKpis(stats);
  byId("analysisOutcomeChart").innerHTML = workspaceRenderOutcomeChart(stats);
  byId("analysisFailureTypeChart").innerHTML = workspaceRenderFailureTypeChart(stats);
  byId("analysisTimingChart").innerHTML = workspaceRenderTimingChart(stats);
  byId("analysisTaskChart").innerHTML = workspaceRenderTaskChart(records);
  var status = byId("analysisStatus");
  status.className = "analysis-status";
  status.textContent = records.length + " rollout(s) in the current live scope. Saved annotations override provisional manifest outcomes; uncertain is excluded from the resolved rate. Controlled data are not mixed into the default natural scope.";
}
