"use strict";

var SETTINGS_DEFAULTS = {
  background_color: "#0b0d10",
  surface_color: "#11151a",
  surface_raised_color: "#171c22",
  control_color: "#0f1318",
  text_color: "#f4f6f7",
  muted_color: "#98a3ad",
  accent_color: "#67d9b5",
  font_scale: 1.0,
  review_font_scale: 1.0,
  analysis_font_scale: 1.0,
  control_font_scale: 1.0,
  density: "comfortable"
};

var SETTINGS_PRESETS = {
  midnight: {
    background_color: "#0b0d10",
    surface_color: "#11151a",
    surface_raised_color: "#171c22",
    control_color: "#0f1318",
    text_color: "#f4f6f7",
    muted_color: "#98a3ad",
    accent_color: "#67d9b5",
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
  partition: "primary_natural",
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
  analysisRunScope: "primary_natural",
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
  analysisTab: "overview",
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
  if (partition === "all") return true;
  if (partition === "primary_natural" || partition === "reference_natural") {
    return record.dataset_role === partition;
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

function workspaceRenderTaskChart(records) {
  if (!records.length) return workspaceEmpty("No task data in the current scope.");
  var groups = {};
  records.forEach(function (record) {
    var key = String(record.task_suite || "unknown") + "::" + String(record.task_id);
    if (!groups[key]) groups[key] = { suite: record.task_suite || "unknown", task: String(record.task_id), counts: { clean_success: 0, recovered_success: 0, terminal_failure: 0, uncertain: 0 } };
    groups[key].counts[workspaceNormalizeOutcome(effectiveOutcome(record))] += 1;
  });
  var entries = Object.keys(groups).map(function (key) {
    var entry = groups[key];
    var resolved = entry.counts.clean_success + entry.counts.recovered_success + entry.counts.terminal_failure;
    entry.resolvedSuccessRate = resolved ? (entry.counts.clean_success + entry.counts.recovered_success) / resolved : null;
    return entry;
  }).sort(function (left, right) {
    return left.suite.localeCompare(right.suite) || Number(left.task) - Number(right.task);
  });
  var width = Math.max(760, entries.length * 66);
  var height = 250;
  var plotX = 42;
  var plotY = 18;
  var plotHeight = 168;
  var plotWidth = width - 70;
  var max = Math.max.apply(null, entries.map(function (entry) {
    return ANALYSIS_OUTCOMES.reduce(function (sum, definition) { return sum + entry.counts[definition.value]; }, 0);
  }).concat([1]));
  var content = '<line class="chart-axis" x1="' + plotX + '" y1="' + (plotY + plotHeight) + '" x2="' + (plotX + plotWidth) + '" y2="' + (plotY + plotHeight) + '"></line>';
  entries.forEach(function (entry, index) {
    var barWidth = Math.max(18, plotWidth / entries.length - 16);
    var x = plotX + index * (plotWidth / entries.length) + 8;
    var cursorY = plotY + plotHeight;
    ANALYSIS_OUTCOMES.forEach(function (definition) {
      var count = entry.counts[definition.value];
      if (!count) return;
      var barHeight = plotHeight * count / max;
      cursorY -= barHeight;
      content += '<rect class="chart-clickable" data-analysis-task="' + escapeHtml(entry.task) + '" data-analysis-suite="' + escapeHtml(entry.suite) + '" x="' + x.toFixed(2) + '" y="' + cursorY.toFixed(2) + '" width="' + barWidth.toFixed(2) + '" height="' + barHeight.toFixed(2) + '" fill="' + definition.color + '"><title>'
        + escapeHtml(labelFor(entry.suite) + " task " + entry.task + " · " + definition.label + ": " + count + "; resolved success rate: " + (entry.resolvedSuccessRate == null ? "n/a" : workspacePercent(entry.resolvedSuccessRate))) + "</title></rect>";
    });
    content += '<text x="' + (x + barWidth / 2).toFixed(2) + '" y="' + (plotY + plotHeight + 18) + '" text-anchor="middle" data-legacy-rotation="rotation(-35 ' + (x + barWidth / 2).toFixed(2) + ' ' + (plotY + plotHeight + 18) + ')">' + escapeHtml(labelFor(entry.suite) + " " + entry.task) + "</text>";
    content += '<text x="' + (x + barWidth / 2).toFixed(2) + '" y="' + (plotY + plotHeight + 51) + '" text-anchor="middle" class="chart-value">rate ' + escapeHtml(entry.resolvedSuccessRate == null ? "n/a" : workspacePercent(entry.resolvedSuccessRate)) + '</text>';
  });
  content += '<text x="10" y="12">count</text>';
  return workspaceSvg(width, height, "Outcome counts by task; click a bar to filter", content)
    + '<div class="analysis-legend">' + ANALYSIS_OUTCOMES.map(function (definition) {
      return '<span><i style="background:' + definition.color + '"></i>' + escapeHtml(definition.label) + "</span>";
    }).join("") + "</div>";
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

function workspaceSnapshotEvents(snapshot) {
  var rows = Array.isArray(snapshot && snapshot.event_metrics) ? snapshot.event_metrics.slice() : [];
  if (rows.length) return rows;
  return (snapshot && snapshot.summary_by_method_signal_outcome || []).map(function (row) {
    return {
      method: row.method,
      signal: row.signal,
      outcome_group: row.outcome_group,
      n_events: row.n_events,
      normalized_response_magnitude: row.normalized_response_magnitude_median,
      post_event_persistence_fraction: row.post_event_persistence_fraction_median,
      recovery_fraction_toward_baseline: row.recovery_fraction_toward_baseline_median,
      recovery_rebound_normalized: row.recovery_rebound_normalized_median
    };
  });
}

function workspaceFilteredSnapshotEvents() {
  var snapshot = workspaceState.analysisSnapshot;
  var filters = workspaceState.snapshotFilters;
  return workspaceSnapshotEvents(snapshot).filter(function (row) {
    return (filters.method === "all" || row.method === filters.method)
      && (filters.outcome === "all" || row.outcome_group === filters.outcome)
      && (filters.task === "all" || String(row.task_id) === String(filters.task));
  });
}

function workspaceRefreshSnapshotTaskOptions(snapshot) {
  var values = {};
  workspaceSnapshotEvents(snapshot).forEach(function (row) {
    if (row.task_id != null) values[String(row.task_id)] = row.task_suite ? labelFor(row.task_suite) + " · task " + row.task_id : "task " + row.task_id;
  });
  var options = Object.keys(values).sort(function (left, right) { return Number(left) - Number(right); }).map(function (value) {
    return { value: value, label: values[value] };
  });
  workspaceSetOptions(byId("analysisSnapshotTask"), options, workspaceState.snapshotFilters.task, "All snapshot tasks");
}

function workspaceAggregateSnapshotRows(rows, field) {
  var groups = {};
  rows.forEach(function (row) {
    var value = workspaceNumber(row[field]);
    if (value == null) return;
    var key = String(row.method) + "::" + String(row.signal) + "::" + String(row.outcome_group);
    if (!groups[key]) groups[key] = { method: row.method, signal: row.signal, outcome: row.outcome_group, values: [] };
    groups[key].values.push(value);
  });
  return Object.keys(groups).map(function (key) {
    var group = groups[key];
    var summary = workspaceSummary(group.values);
    return {
      method: group.method,
      signal: group.signal,
      outcome: group.outcome,
      count: summary.count,
      median: summary.median,
      q25: summary.q25,
      q75: summary.q75
    };
  });
}

function workspaceRenderMetricChart(rows, field, title, label, fraction) {
  var grouped = workspaceAggregateSnapshotRows(rows, field);
  var categories = {};
  grouped.forEach(function (row) { categories[row.method + "::" + row.signal] = { method: row.method, signal: row.signal }; });
  var order = [];
  ANALYSIS_METHODS.forEach(function (method) {
    Object.keys(categories).filter(function (key) { return categories[key].method === method; }).sort().forEach(function (key) { order.push(categories[key]); });
  });
  if (!order.length) return workspaceEmpty("No values for this metric in the selected snapshot scope.");
  var byKey = {};
  grouped.forEach(function (row) { byKey[row.method + "::" + row.signal + "::" + row.outcome] = row; });
  var values = grouped.map(function (row) { return row.q75; }).filter(function (value) { return value != null; });
  var max = Math.max.apply(null, values.concat([fraction ? 1 : 0.1]));
  max *= 1.12;
  var width = 720;
  var height = Math.max(120, 50 + order.length * 39);
  var plotX = 205;
  var plotWidth = 410;
  var content = '<text x="12" y="18" class="chart-label">' + escapeHtml(title) + '</text>';
  order.forEach(function (category, index) {
    var y = 42 + index * 39;
    var key = category.method + "::" + category.signal;
    content += '<text x="12" y="' + (y + 5) + '" class="chart-label">' + escapeHtml(ANALYSIS_METHOD_LABELS[category.method] || category.method) + '</text>';
    content += '<text x="12" y="' + (y + 19) + '">' + escapeHtml(category.signal) + '</text>';
    content += '<line class="chart-grid" x1="' + plotX + '" y1="' + y + '" x2="' + (plotX + plotWidth) + '" y2="' + y + '"></line>';
    ["recovered_success", "terminal_failure"].forEach(function (outcome, outcomeIndex) {
      var row = byKey[key + "::" + outcome];
      if (!row) return;
      var x = plotX + plotWidth * row.median / max;
      var q25 = plotX + plotWidth * row.q25 / max;
      var q75 = plotX + plotWidth * row.q75 / max;
      var color = outcome === "recovered_success" ? "var(--recovery)" : "var(--terminal)";
      content += '<line x1="' + q25.toFixed(2) + '" y1="' + (y + outcomeIndex * 11 - 5) + '" x2="' + q75.toFixed(2) + '" y2="' + (y + outcomeIndex * 11 - 5) + '" stroke="' + color + '" stroke-width="4" stroke-linecap="round"></line>';
      content += '<circle cx="' + x.toFixed(2) + '" cy="' + (y + outcomeIndex * 11 - 5) + '" r="4" fill="' + color + '"><title>'
        + escapeHtml((outcome === "recovered_success" ? "Recovered" : "Terminal") + ": median " + workspaceFormatNumber(fraction ? row.median * 100 : row.median) + (fraction ? "%" : "") + "; IQR " + workspaceFormatNumber(fraction ? row.q25 * 100 : row.q25) + "–" + workspaceFormatNumber(fraction ? row.q75 * 100 : row.q75) + (fraction ? "%" : "") + "; n=" + row.count) + "</title></circle>";
    });
  });
  content += '<line class="chart-axis" x1="' + plotX + '" y1="' + (height - 22) + '" x2="' + (plotX + plotWidth) + '" y2="' + (height - 22) + '"></line>';
  content += '<text x="' + plotX + '" y="' + (height - 7) + '">0</text><text x="' + (plotX + plotWidth - 35) + '" y="' + (height - 7) + '">' + workspaceFormatNumber(fraction ? max * 100 : max) + (fraction ? "%" : "") + '</text>';
  return workspaceSvg(width, height, label, content)
    + '<div class="analysis-legend"><span><i style="background:var(--recovery)"></i>Recovered success</span><span><i style="background:var(--terminal)"></i>Terminal failure</span><span><i style="background:var(--muted)"></i>IQR line · dot = median</span></div>';
}

function workspaceRenderCoverageChart(snapshot) {
  var rows = (snapshot.method_coverage || []).filter(function (row) {
    return workspaceState.snapshotFilters.method === "all" || row.method === workspaceState.snapshotFilters.method;
  });
  if (!rows.length) return workspaceEmpty("Coverage data are unavailable for the selected method.");
  var width = 720;
  var height = 55 + rows.length * 37;
  var content = "";
  rows.forEach(function (row, index) {
    var y = 29 + index * 37;
    var selected = workspaceNumber(row.selected_rollouts) || 0;
    var available = workspaceNumber(row.available_rollouts) || 0;
    var missing = Math.max(0, selected - available);
    var totalWidth = 420;
    var availableWidth = selected ? totalWidth * available / selected : 0;
    var missingWidth = selected ? totalWidth * missing / selected : 0;
    content += '<text x="10" y="' + (y + 10) + '" class="chart-label">' + escapeHtml(ANALYSIS_METHOD_LABELS[row.method] || row.method) + '</text>';
    content += '<rect x="180" y="' + y + '" width="' + availableWidth.toFixed(2) + '" height="18" fill="var(--accent)"><title>' + escapeHtml(available + " available / " + selected + " selected") + '</title></rect>';
    content += '<rect x="' + (180 + availableWidth).toFixed(2) + '" y="' + y + '" width="' + missingWidth.toFixed(2) + '" height="18" fill="var(--line)"><title>' + escapeHtml(missing + " missing") + '</title></rect>';
    content += '<text x="' + (616) + '" y="' + (y + 12) + '" class="chart-value">' + available + '/' + selected + '</text>';
  });
  return workspaceSvg(width, height, "Baseline method coverage", content)
    + '<div class="analysis-legend"><span><i style="background:var(--accent)"></i>available</span><span><i style="background:var(--line)"></i>missing</span></div>';
}

function workspaceRenderThresholdChart(rows, snapshot) {
  var statistics = Array.isArray(snapshot && snapshot.onset_signal_statistics)
    ? snapshot.onset_signal_statistics
    : [];
  if (statistics.length && (!workspaceState.snapshotFilters || workspaceState.snapshotFilters.task === "all")) {
    var filters = workspaceState.snapshotFilters || {};
    rows = statistics.filter(function (row) {
      return row.method && row.signal
        && (!filters.method || filters.method === "all" || row.method === filters.method)
        && (!filters.outcome || filters.outcome === "all" || row.outcome_group === filters.outcome)
        && (!filters.task || filters.task === "all" || row.task_id == null
          || String(row.task_id) === String(filters.task));
    });
  }
  var groups = {};
  rows.forEach(function (row) {
    var key = row.method + "::" + row.signal;
    if (!groups[key]) groups[key] = { method: row.method, signal: row.signal, outcomes: {} };
    var outcome = row.outcome_group;
    if (!groups[key].outcomes[outcome]) groups[key].outcomes[outcome] = { total: 0, exceed: 0, direction: 0, rate: null, directionRate: null };
    var cell = groups[key].outcomes[outcome];
    if (row.n_onset_events != null || row.exceeding_q95_fraction != null
      || row.direction_consistency_fraction_all != null) {
      var summaryTotal = workspaceNumber(row.n_onset_events);
      var summaryExceed = workspaceNumber(row.n_exceeding_q95);
      var summaryDirection = workspaceNumber(row.direction_consistency_count_all);
      cell.total += summaryTotal == null ? 0 : summaryTotal;
      cell.exceed += summaryExceed == null ? 0 : summaryExceed;
      cell.direction += summaryDirection == null ? 0 : summaryDirection;
      cell.rate = workspaceNumber(row.exceeding_q95_fraction);
      cell.directionRate = workspaceNumber(row.direction_consistency_fraction_all);
      return;
    }
    cell.total += 1;
    if (workspaceNumber(row.normalized_response_magnitude) != null
      && workspaceNumber(row.clean_background_response_q95) != null
      && Number(row.normalized_response_magnitude) > Number(row.clean_background_response_q95)) {
      cell.exceed += 1;
    }
    if (workspaceNumber(row.normalized_failure_oriented_response) != null
      && Number(row.normalized_failure_oriented_response) > 0) {
      cell.direction += 1;
    }
  });
  var categories = Object.keys(groups).map(function (key) { return groups[key]; }).sort(function (left, right) {
    return ANALYSIS_METHODS.indexOf(left.method) - ANALYSIS_METHODS.indexOf(right.method)
      || left.signal.localeCompare(right.signal);
  });
  if (!categories.length) return workspaceEmpty("No onset statistics for the selected snapshot scope.");

  var outcomes = [
    { value: "recovered_success", label: "Recovered" },
    { value: "terminal_failure", label: "Terminal" },
    { value: "uncertain", label: "Uncertain" }
  ];
  var html = '<div class="analysis-threshold-table-wrap">'
    + '<table class="analysis-threshold-table" aria-label="Baseline onset threshold and direction statistics">'
    + '<caption>Q95 exceed rate / failure-direction consistency</caption>'
    + '<thead><tr><th scope="col">Method / signal</th>'
    + outcomes.map(function (outcome) {
      return '<th scope="col">' + escapeHtml(outcome.label) + '</th>';
    }).join("")
    + '</tr></thead><tbody>';

  categories.forEach(function (category) {
    html += '<tr><th scope="row">' + escapeHtml(
      (ANALYSIS_METHOD_LABELS[category.method] || category.method) + " / " + category.signal
    ) + '</th>';
    outcomes.forEach(function (outcome) {
      var cell = category.outcomes[outcome.value];
      var rate = cell && cell.rate != null
        ? cell.rate
        : (cell && cell.total ? cell.exceed / cell.total : null);
      var direction = cell && cell.directionRate != null
        ? cell.directionRate
        : (cell && cell.total ? cell.direction / cell.total : null);
      var title = rate == null
        ? "No events"
        : "Q95 exceed " + workspacePercent(rate)
          + "; direction consistent " + workspacePercent(direction)
          + "; n=" + cell.total;
      var opacity = rate == null ? 0.08 : Math.min(0.9, 0.15 + rate * 0.75);
      html += '<td class="threshold-cell threshold-' + outcome.value + '" style="--threshold-opacity:' + opacity.toFixed(2)
        + '" title="' + escapeHtml(title) + '" aria-label="' + escapeHtml(title) + '">'
        + '<span class="threshold-rate">' + (rate == null ? "n/a" : workspacePercent(rate)) + '</span>'
        + (direction == null ? "" : '<span class="threshold-direction">dir ' + workspacePercent(direction) + '</span>')
        + '</td>';
    });
    html += '</tr>';
  });
  return html + '</tbody></table></div>';
}
function workspaceRenderAnomalies(rows) {
  var sorted = rows.slice().filter(function (row) { return workspaceNumber(row.normalized_response_magnitude) != null; }).sort(function (left, right) {
    return Number(right.normalized_response_magnitude) - Number(left.normalized_response_magnitude);
  }).slice(0, 12);
  if (!sorted.length) return workspaceEmpty("No event-level response values in the selected snapshot scope.");
  var html = '<table class="analysis-table"><thead><tr><th>Method / signal</th><th>Outcome</th><th>Failure type</th><th>Rollout</th><th>Frame</th><th>Response</th><th>Clean Q95 percentile</th><th></th></tr></thead><tbody>';
  sorted.forEach(function (row) {
    var rolloutId = row.rollout_id || "unknown";
    html += '<tr><td><strong>' + escapeHtml((ANALYSIS_METHOD_LABELS[row.method] || row.method) + " / " + row.signal) + '</strong></td>'
      + '<td>' + escapeHtml(labelFor(row.outcome_group)) + '</td>'
      + '<td>' + escapeHtml(labelFor(row.failure_type || "unknown")) + '</td>'
      + '<td class="numeric">' + escapeHtml(rolloutId) + '</td>'
      + '<td class="numeric">' + escapeHtml(row.event_frame == null ? "n/a" : row.event_frame) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.normalized_response_magnitude)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspacePercent(row.clean_background_response_percentile)) + '</td>'
      + '<td><button type="button" data-analysis-rollout="' + escapeHtml(rolloutId) + '">Review</button></td></tr>';
  });
  return html + "</tbody></table>";
}


function workspaceLocalizationRows(snapshot) {
  if (!snapshot) return [];
  if (Array.isArray(snapshot.localization_event_metrics)) return snapshot.localization_event_metrics.slice();
  if (snapshot.localization && Array.isArray(snapshot.localization.event_metrics)) {
    return snapshot.localization.event_metrics.slice();
  }
  return [];
}

function workspaceLocalizationSummaryRows(snapshot) {
  if (!snapshot) return [];
  if (Array.isArray(snapshot.localization_summary)) return snapshot.localization_summary.slice();
  if (snapshot.localization && Array.isArray(snapshot.localization.summary)) {
    return snapshot.localization.summary.slice();
  }
  return [];
}

function workspaceLocalizationFailureRows(snapshot) {
  if (!snapshot) return [];
  if (Array.isArray(snapshot.localization_by_failure_type)) return snapshot.localization_by_failure_type.slice();
  if (snapshot.localization && Array.isArray(snapshot.localization.by_failure_type)) {
    return snapshot.localization.by_failure_type.slice();
  }
  return [];
}

function workspaceLocalizationCategory(row) {
  return (ANALYSIS_METHOD_LABELS[row.method] || row.method) + " / " + row.signal;
}


function workspaceRefreshLocalizationOptions(snapshot) {
  var events = workspaceLocalizationRows(snapshot);
  var filters = workspaceState.localizationFilters;
  var signalValues = {};
  var taskValues = {};
  events.forEach(function (row) {
    if (filters.method === "all" || row.method === filters.method) {
      if (row.signal != null) signalValues[String(row.signal)] = String(row.signal);
    }
    if (row.task_id != null) {
      var taskKey = String(row.task_id);
      taskValues[taskKey] = row.task_suite
        ? labelFor(row.task_suite) + " - task " + row.task_id
        : "task " + row.task_id;
    }
  });
  workspaceSetOptions(
    byId("analysisLocalizationSignal"),
    Object.keys(signalValues).sort().map(function (value) {
      return { value: value, label: value };
    }),
    filters.signal,
    "All signals"
  );
  workspaceSetOptions(
    byId("analysisLocalizationTask"),
    Object.keys(taskValues).sort(function (left, right) {
      return Number(left) - Number(right);
    }).map(function (value) {
      return { value: value, label: taskValues[value] };
    }),
    filters.task,
    "All tasks"
  );
  filters.signal = byId("analysisLocalizationSignal").value;
  filters.task = byId("analysisLocalizationTask").value;
}

function workspaceFilteredLocalizationEvents(snapshot) {
  var filters = workspaceState.localizationFilters;
  return workspaceLocalizationRows(snapshot).filter(function (row) {
    return (filters.method === "all" || row.method === filters.method)
      && (filters.signal === "all" || row.signal === filters.signal)
      && (filters.threshold === "all" || row.threshold === filters.threshold)
      && (filters.outcome === "all" || row.outcome_group === filters.outcome)
      && (filters.task === "all" || String(row.task_id) === String(filters.task));
  });
}

function workspaceLocalizationMetricRows(snapshot) {
  var filters = workspaceState.localizationFilters;
  var summary = workspaceLocalizationSummaryRows(snapshot).filter(function (row) {
    return (filters.method === "all" || row.method === filters.method)
      && (filters.signal === "all" || row.signal === filters.signal)
      && (filters.threshold === "all" || row.threshold === filters.threshold)
      && (filters.outcome === "all"
        ? row.outcome_group === "all_events"
        : row.outcome_group === filters.outcome)
      && filters.task === "all";
  });
  if (filters.task === "all") return summary;
  var events = workspaceFilteredLocalizationEvents(snapshot);
  var groups = {};
  events.forEach(function (row) {
    var key = row.method + "::" + row.signal + "::" + row.threshold;
    if (!groups[key]) {
      groups[key] = {
        method: row.method,
        signal: row.signal,
        threshold: row.threshold,
        outcome_group: filters.outcome === "all" ? "all_events" : filters.outcome,
        n_events: 0,
        n_hits: 0
      };
    }
    groups[key].n_events += 1;
    if (row.hit) groups[key].n_hits += 1;
  });
  var allSummary = workspaceLocalizationSummaryRows(snapshot);
  Object.keys(groups).forEach(function (key) {
    var group = groups[key];
    var base = allSummary.find(function (row) {
      return row.method === group.method && row.signal === group.signal
        && row.threshold === group.threshold && row.outcome_group === "all_events";
    });
    var matching = events.filter(function (row) {
      return row.method === group.method && row.signal === group.signal
        && row.threshold === group.threshold;
    });
    var errors = matching.map(function (row) {
      return workspaceNumber(row.absolute_localization_error_frames);
    }).filter(function (value) { return value != null; });
    group.event_hit_rate = group.n_events ? group.n_hits / group.n_events : null;
    group.recall = group.event_hit_rate;
    group.median_absolute_localization_error_frames = workspaceMedian(errors);
    [4, 8, 16, 30].forEach(function (tolerance) {
      var hits = matching.filter(function (row) {
        return row["hit_within_" + tolerance + "_frames"];
      }).length;
      group["hit_rate_within_" + tolerance + "_frames"] = matching.length ? hits / matching.length : null;
    });
    if (base) {
      ["false_alarm_rate", "false_alarm_count", "clean_pseudo_events", "precision", "f1", "auroc", "average_precision"].forEach(function (field) {
        group[field] = workspaceNumber(base[field]);
      });
    }
  });
  return Object.keys(groups).map(function (key) { return groups[key]; });
}


function workspaceRenderLocalizationRecallChart(rows) {
  if (!rows.length) return workspaceEmpty("No localization summary matches the selected filters.");
  var ordered = rows.slice().sort(function (left, right) {
    return ANALYSIS_METHODS.indexOf(left.method) - ANALYSIS_METHODS.indexOf(right.method)
      || String(left.signal).localeCompare(String(right.signal));
  });
  var width = 820;
  var rowHeight = 38;
  var height = 34 + ordered.length * rowHeight;
  var plotX = 260;
  var plotWidth = 410;
  var content = '<line class="chart-axis" x1="' + plotX + '" y1="20" x2="' + (plotX + plotWidth) + '" y2="20"></line>'
    + '<text x="' + plotX + '" y="13">0</text><text x="' + (plotX + plotWidth - 20) + '" y="13">100%</text>';
  ordered.forEach(function (row, index) {
    var y = 30 + index * rowHeight;
    var recall = workspaceNumber(row.recall);
    var falseAlarm = workspaceNumber(row.false_alarm_rate);
    var recallWidth = recall == null ? 0 : plotWidth * Math.max(0, Math.min(1, recall));
    var falseWidth = falseAlarm == null ? 0 : plotWidth * Math.max(0, Math.min(1, falseAlarm));
    var label = workspaceLocalizationCategory(row);
    content += '<text x="10" y="' + (y + 11) + '" class="chart-label">' + escapeHtml(label) + '</text>';
    content += '<rect x="' + plotX + '" y="' + y + '" width="' + recallWidth.toFixed(2) + '" height="10" rx="3" fill="var(--accent)"><title>'
      + escapeHtml("Recall " + workspacePercent(recall) + "; n=" + (row.n_events == null ? "n/a" : row.n_events)) + '</title></rect>';
    content += '<rect x="' + plotX + '" y="' + (y + 14) + '" width="' + falseWidth.toFixed(2) + '" height="10" rx="3" fill="var(--failure)"><title>'
      + escapeHtml("Clean-success false-alarm rate " + workspacePercent(falseAlarm) + "; n=" + (row.clean_pseudo_events == null ? "n/a" : row.clean_pseudo_events)) + '</title></rect>';
    content += '<text x="' + (plotX + plotWidth + 10) + '" y="' + (y + 9) + '" class="chart-value">R ' + escapeHtml(workspacePercent(recall)) + '</text>'
      + '<text x="' + (plotX + plotWidth + 10) + '" y="' + (y + 23) + '">FA ' + escapeHtml(workspacePercent(falseAlarm)) + '</text>';
  });
  return workspaceSvg(width, height, "Failure localization recall and clean-success false-alarm comparison", content)
    + '<div class="analysis-legend"><span><i style="background:var(--accent)"></i>event recall</span><span><i style="background:var(--failure)"></i>clean-success false alarm</span></div>';
}


function workspaceRenderLocalizationErrorChart(rows) {
  if (!rows.length) return workspaceEmpty("No localization error values match the selected filters.");
  var ordered = rows.slice().sort(function (left, right) {
    return workspaceLocalizationCategory(left).localeCompare(workspaceLocalizationCategory(right));
  });
  var width = Math.max(820, 420 + ordered.length * 12);
  var height = 250;
  var plotX = 46;
  var plotY = 18;
  var plotHeight = 168;
  var plotWidth = width - 86;
  var maxError = Math.max.apply(null, ordered.map(function (row) {
    return workspaceNumber(row.median_absolute_localization_error_frames) || 0;
  }).concat([30, 1]));
  var content = '<line class="chart-axis" x1="' + plotX + '" y1="' + (plotY + plotHeight) + '" x2="' + (plotX + plotWidth) + '" y2="' + (plotY + plotHeight) + '"></line>'
    + '<text x="8" y="' + (plotY + 5) + '">frames</text>';
  ordered.forEach(function (row, index) {
    var slot = plotWidth / ordered.length;
    var barWidth = Math.max(16, slot - 8);
    var x = plotX + index * slot + (slot - barWidth) / 2;
    var error = workspaceNumber(row.median_absolute_localization_error_frames);
    var barHeight = error == null ? 0 : plotHeight * Math.min(maxError, error) / maxError;
    content += '<rect x="' + x.toFixed(2) + '" y="' + (plotY + plotHeight - barHeight).toFixed(2) + '" width="' + barWidth.toFixed(2) + '" height="' + barHeight.toFixed(2) + '" rx="3" fill="var(--recovery)"><title>'
      + escapeHtml(workspaceLocalizationCategory(row) + " median absolute error " + workspaceFormatNumber(error) + " frames") + '</title></rect>';
    content += '<text x="' + (x + barWidth / 2).toFixed(2) + '" y="' + (plotY + plotHeight + 18) + '" text-anchor="middle" data-legacy-rotation="rotation(-38 ' + (x + barWidth / 2).toFixed(2) + ' ' + (plotY + plotHeight + 18) + ')">'
      + escapeHtml(workspaceLocalizationCategory(row)) + '</text>';
  });
  var tolerance = [4, 8, 16, 30].map(function (value) {
    var values = rows.map(function (row) {
      return workspaceNumber(row["hit_rate_within_" + value + "_frames"]);
    }).filter(function (number) { return number != null; });
    return value + "f " + workspacePercent(values.length ? workspaceMedian(values) : null);
  }).join(" | ");
  content += '<text x="' + plotX + '" y="' + (height - 10) + '" class="chart-value">Tolerance hit rates: ' + escapeHtml(tolerance) + '</text>';
  return workspaceSvg(width, height, "Median localization error and tolerance hit rates", content)
    + '<div class="analysis-legend"><span><i style="background:var(--recovery)"></i>median absolute error in frames</span><span>tolerance rates are listed below</span></div>';
}


function workspaceRenderLocalizationFailureTypes(snapshot) {
  var filters = workspaceState.localizationFilters;
  var rows = workspaceLocalizationFailureRows(snapshot).filter(function (row) {
    return (filters.method === "all" || row.method === filters.method)
      && (filters.signal === "all" || row.signal === filters.signal)
      && (filters.threshold === "all" || row.threshold === filters.threshold)
      && (filters.outcome === "all" || row.outcome_group === filters.outcome || row.outcome_group == null);
  }).sort(function (left, right) {
    return String(left.failure_type).localeCompare(String(right.failure_type))
      || ANALYSIS_METHODS.indexOf(left.method) - ANALYSIS_METHODS.indexOf(right.method)
      || String(left.signal).localeCompare(String(right.signal));
  });
  if (!rows.length) return workspaceEmpty("No failure-type localization rows match the selected filters.");
  var html = '<table class="analysis-table localization-heat-table"><caption>Recall, false alarms and F1 by failure type; intensity follows recall.</caption><thead><tr>'
    + '<th>Failure type</th><th>Method / signal</th><th>Events</th><th>Recall</th><th>False alarm</th><th>F1</th><th>AUROC</th></tr></thead><tbody>';
  rows.forEach(function (row) {
    var recall = workspaceNumber(row.recall);
    var intensity = recall == null ? 0.05 : Math.max(0.08, Math.min(0.9, recall));
    html += '<tr><td>' + escapeHtml(labelFor(row.failure_type || "other")) + '</td>'
      + '<td><strong>' + escapeHtml(workspaceLocalizationCategory(row)) + '</strong></td>'
      + '<td class="numeric">' + escapeHtml(row.n_events == null ? "n/a" : row.n_events) + '</td>'
      + '<td class="numeric localization-heat-cell" style="--localization-heat:' + intensity.toFixed(2) + '">' + escapeHtml(workspacePercent(recall)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspacePercent(row.false_alarm_rate)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.f1)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.auroc)) + '</td></tr>';
  });
  return html + '</tbody></table>';
}

function workspaceRenderLocalizationEvents(rows) {
  var sorted = rows.slice().sort(function (left, right) {
    return (Number(right.event_score) || 0) - (Number(left.event_score) || 0);
  }).slice(0, 100);
  if (!sorted.length) return workspaceEmpty("No event-level localization rows match the selected filters.");
  var html = '<table class="analysis-table localization-events-table"><thead><tr><th>Method / signal</th><th>Threshold</th><th>Outcome</th><th>Failure type</th><th>Rollout</th><th>Onset</th><th>First native alarm</th><th>Lead / lag</th><th>Abs. error</th><th>Result</th><th></th></tr></thead><tbody>';
  sorted.forEach(function (row) {
    var hit = Boolean(row.hit);
    var rolloutId = row.rollout_id || "unknown";
    html += '<tr><td><strong>' + escapeHtml(workspaceLocalizationCategory(row)) + '</strong></td>'
      + '<td>' + escapeHtml(String(row.threshold || "").toUpperCase()) + '</td>'
      + '<td>' + escapeHtml(labelFor(row.outcome_group || "uncertain")) + '</td>'
      + '<td>' + escapeHtml(labelFor(row.failure_type || "other")) + '</td>'
      + '<td class="numeric">' + escapeHtml(rolloutId) + '</td>'
      + '<td class="numeric">' + escapeHtml(row.observable_onset_frame == null ? "n/a" : row.observable_onset_frame) + '</td>'
      + '<td class="numeric">' + escapeHtml(row.first_alarm_frame == null ? "none" : row.first_alarm_frame) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.signed_lead_lag_frames)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.absolute_localization_error_frames)) + '</td>'
      + '<td><span class="localization-result ' + (hit ? "hit" : "miss") + '">' + (hit ? "hit" : "miss") + '</span></td>'
      + '<td><button type="button" data-analysis-rollout="' + escapeHtml(rolloutId) + '">Review</button></td></tr>';
  });
  return html + '</tbody></table>';
}

function workspaceRenderLocalization(snapshot) {
  var badge = byId("analysisLocalizationBadge");
  var status = byId("analysisLocalizationStatus");
  if (!badge || !status) return;
  var events = workspaceLocalizationRows(snapshot);
  var summary = workspaceLocalizationSummaryRows(snapshot);
  if (!snapshot || !snapshot.available || !snapshot.localization_available || (!events.length && !summary.length)) {
    badge.className = "analysis-badge";
    badge.textContent = "Unavailable";
    status.className = "analysis-status warning";
    status.textContent = "Failure localization is unavailable in this snapshot. The older temporal metrics remain usable.";
    ["analysisLocalizationRecallChart", "analysisLocalizationErrorChart", "analysisLocalizationFailureType", "analysisLocalizationEvents", "analysisLocalizationProvenance"].forEach(function (id) {
      if (byId(id)) byId(id).innerHTML = workspaceEmpty("Localization unavailable");
    });
    return;
  }
  ["method", "threshold", "outcome"].forEach(function (field) {
    var select = byId("analysisLocalization" + field.charAt(0).toUpperCase() + field.slice(1));
    if (select) select.value = workspaceState.localizationFilters[field];
  });
  workspaceRefreshLocalizationOptions(snapshot);
  var rows = workspaceLocalizationMetricRows(snapshot);
  var filteredEvents = workspaceFilteredLocalizationEvents(snapshot);
  badge.className = "analysis-badge ok";
  badge.textContent = "Ready";
  var primary = workspaceState.localizationFilters.threshold === "q95";
  status.className = "analysis-status";
  status.textContent = "Descriptive native-sample diagnostics: " + events.length + " event x method x signal x threshold rows; " + (primary ? "Q95 is the primary view." : "threshold trade-off view.") + " Clean-success pseudo-events provide the false-alarm reference; this is not independently validated detector performance.";
  byId("analysisLocalizationRecallChart").innerHTML = workspaceRenderLocalizationRecallChart(rows);
  byId("analysisLocalizationErrorChart").innerHTML = workspaceRenderLocalizationErrorChart(rows);
  byId("analysisLocalizationFailureType").innerHTML = workspaceRenderLocalizationFailureTypes(snapshot);
  byId("analysisLocalizationEvents").innerHTML = workspaceRenderLocalizationEvents(filteredEvents);
  var coverage = snapshot.method_coverage || [];
  var missing = coverage.reduce(function (sum, row) { return sum + (Number(row.missing_rollouts) || 0); }, 0);
  byId("analysisLocalizationProvenance").innerHTML = "Valid event rows: " + filteredEvents.length + " · method coverage missing outputs: " + missing + " · thresholds use native sample points only. Missing raw outputs are excluded from metric denominators.";
}


function workspaceChangePointSnapshot(snapshot) {
  var changePoint = snapshot && snapshot.change_point;
  return changePoint && changePoint.available ? changePoint : null;
}

function workspaceChangePointRows(changePoint) {
  return changePoint && Array.isArray(changePoint.event_metrics) ? changePoint.event_metrics.slice() : [];
}

function workspaceChangePointSummaryRows(changePoint) {
  return changePoint && Array.isArray(changePoint.summary) ? changePoint.summary.slice() : [];
}

function workspaceChangePointFailureRows(changePoint) {
  return changePoint && Array.isArray(changePoint.by_failure_type) ? changePoint.by_failure_type.slice() : [];
}

function workspaceChangePointBaseFilter(row, filters) {
  return (filters.method === "all" || row.method === filters.method)
    && (filters.signal === "all" || row.signal === filters.signal)
    && (filters.feature === "all" || row.feature === filters.feature)
    && (filters.scale === "all" || String(row.scale_frames) === String(filters.scale))
    && (filters.threshold === "all" || row.threshold === filters.threshold);
}

function workspaceRefreshChangePointOptions(changePoint) {
  if (!changePoint) return;
  var filters = workspaceState.changePointFilters;
  var events = workspaceChangePointRows(changePoint);
  var signals = {};
  var scales = {};
  var tasks = {};
  events.forEach(function (row) {
    if (filters.method === "all" || row.method === filters.method) {
      if (row.signal != null) signals[String(row.signal)] = String(row.signal);
    }
    if (row.scale_frames != null) scales[String(row.scale_frames)] = String(row.scale_frames) + " frames";
    if (row.task_id != null) {
      var taskKey = String(row.task_id);
      tasks[taskKey] = row.task_suite
        ? labelFor(row.task_suite) + " - task " + row.task_id
        : "task " + row.task_id;
    }
  });
  workspaceSetOptions(
    byId("analysisChangePointSignal"),
    Object.keys(signals).sort().map(function (value) { return { value: value, label: value }; }),
    filters.signal,
    "All signals"
  );
  workspaceSetOptions(
    byId("analysisChangePointScale"),
    Object.keys(scales).sort(function (left, right) { return Number(left) - Number(right); }).map(function (value) {
      return { value: value, label: scales[value] };
    }),
    filters.scale,
    "All scales"
  );
  workspaceSetOptions(
    byId("analysisChangePointTask"),
    Object.keys(tasks).sort(function (left, right) { return Number(left) - Number(right); }).map(function (value) {
      return { value: value, label: tasks[value] };
    }),
    filters.task,
    "All tasks"
  );
  filters.signal = byId("analysisChangePointSignal").value;
  filters.scale = byId("analysisChangePointScale").value;
  filters.task = byId("analysisChangePointTask").value;
}

function workspaceChangePointFilteredEvents(changePoint) {
  var filters = workspaceState.changePointFilters;
  return workspaceChangePointRows(changePoint).filter(function (row) {
    return workspaceChangePointBaseFilter(row, filters)
      && (filters.outcome === "all" || row.outcome_group === filters.outcome)
      && (filters.task === "all" || String(row.task_id) === String(filters.task));
  });
}

function workspaceChangePointMetricRows(changePoint) {
  var filters = workspaceState.changePointFilters;
  var summary = workspaceChangePointSummaryRows(changePoint).filter(function (row) {
    return workspaceChangePointBaseFilter(row, filters)
      && (filters.outcome === "all" ? row.outcome_group === "all_events" : row.outcome_group === filters.outcome)
      && filters.task === "all";
  });
  if (filters.task === "all") return summary;
  var events = workspaceChangePointFilteredEvents(changePoint);
  var groups = {};
  events.forEach(function (row) {
    var key = [row.method, row.signal, row.feature, row.scale_frames, row.threshold].join("::");
    if (!groups[key]) {
      groups[key] = {
        method: row.method,
        signal: row.signal,
        feature: row.feature,
        scale_frames: row.scale_frames,
        threshold: row.threshold,
        outcome_group: filters.outcome === "all" ? "all_events" : filters.outcome,
        n_events: 0,
        n_hits: 0,
        peakDistances: [],
        firstErrors: [],
        onsetNear: 0
      };
    }
    var group = groups[key];
    group.n_events += 1;
    if (row.hit) group.n_hits += 1;
    var peakDistance = workspaceNumber(row.peak_distance_frames);
    var firstError = workspaceNumber(row.first_exceedance_absolute_error_frames);
    if (peakDistance != null) group.peakDistances.push(peakDistance);
    if (firstError != null) group.firstErrors.push(firstError);
    if (row.onset_near_unusual) group.onsetNear += 1;
  });
  var allSummary = workspaceChangePointSummaryRows(changePoint);
  return Object.keys(groups).map(function (key) {
    var group = groups[key];
    var base = allSummary.find(function (row) {
      return row.method === group.method && row.signal === group.signal
        && row.feature === group.feature && String(row.scale_frames) === String(group.scale_frames)
        && row.threshold === group.threshold
        && row.outcome_group === (filters.outcome === "all" ? "all_events" : filters.outcome);
    });
    group.recall = group.n_events ? group.n_hits / group.n_events : null;
    group.event_hit_rate = group.recall;
    group.median_peak_distance_frames = workspaceMedian(group.peakDistances);
    group.median_first_exceedance_absolute_error_frames = workspaceMedian(group.firstErrors);
    group.onset_near_unusual_rate = group.n_events ? group.onsetNear / group.n_events : null;
    [4, 8, 16, 30].forEach(function (tolerance) {
      var peakHits = events.filter(function (row) {
        return row.method === group.method && row.signal === group.signal
          && row.feature === group.feature && String(row.scale_frames) === String(group.scale_frames)
          && row.threshold === group.threshold
          && row.peak_distance_frames != null
          && Number(row.peak_distance_frames) <= tolerance
          && row.hit;
      }).length;
      var firstHits = events.filter(function (row) {
        return row.method === group.method && row.signal === group.signal
          && row.feature === group.feature && String(row.scale_frames) === String(group.scale_frames)
          && row.threshold === group.threshold
          && row.first_exceedance_absolute_error_frames != null
          && Number(row.first_exceedance_absolute_error_frames) <= tolerance;
      }).length;
      group["peak_hit_rate_within_" + tolerance + "_frames"] = group.n_events ? peakHits / group.n_events : null;
      group["first_hit_rate_within_" + tolerance + "_frames"] = group.n_events ? firstHits / group.n_events : null;
    });
    if (base) {
      [
        "clean_success_false_alarm_rate",
        "same_rollout_non_onset_false_alarm_rate",
        "false_alarm_rate",
        "false_alarm_count",
        "clean_pseudo_events",
        "precision",
        "f1",
        "auroc",
        "average_precision"
      ].forEach(function (field) {
        group[field] = workspaceNumber(base[field]);
      });
    }
    return group;
  });
}

function workspaceChangePointCategory(row) {
  return (ANALYSIS_METHOD_LABELS[row.method] || row.method)
    + " / " + row.signal + " / " + row.feature + " / " + row.scale_frames + "f";
}


function workspaceRenderChangePointRecallChart(rows) {
  if (!rows.length) return workspaceEmpty("No local change-point summary matches the selected filters.");
  var ordered = rows.slice().sort(function (left, right) {
    return ANALYSIS_METHODS.indexOf(left.method) - ANALYSIS_METHODS.indexOf(right.method)
      || workspaceChangePointCategory(left).localeCompare(workspaceChangePointCategory(right));
  });
  var width = 900;
  var plotX = 300;
  var plotWidth = 430;
  var rowHeight = 34;
  var height = Math.max(70, 32 + ordered.length * rowHeight);
  var content = '<line class="chart-axis" x1="' + plotX + '" y1="20" x2="' + (plotX + plotWidth) + '" y2="20"></line>'
    + '<text x="' + plotX + '" y="13">0</text><text x="' + (plotX + plotWidth - 24) + '" y="13">100%</text>';
  ordered.forEach(function (row, index) {
    var y = 29 + index * rowHeight;
    var recall = workspaceNumber(row.recall);
    var falseAlarm = workspaceNumber(row.false_alarm_rate);
    if (falseAlarm == null) falseAlarm = workspaceNumber(row.clean_success_false_alarm_rate);
    var recallWidth = recall == null ? 0 : plotWidth * Math.max(0, Math.min(1, recall));
    var falseWidth = falseAlarm == null ? 0 : plotWidth * Math.max(0, Math.min(1, falseAlarm));
    var label = workspaceChangePointCategory(row);
    content += '<text x="10" y="' + (y + 10) + '" class="chart-label">' + escapeHtml(label) + '</text>'
      + '<rect x="' + plotX + '" y="' + y + '" width="' + recallWidth.toFixed(2) + '" height="9" rx="2" fill="var(--accent)"><title>'
      + escapeHtml("Recall " + workspacePercent(recall) + "; events=" + (row.n_events == null ? "n/a" : row.n_events)) + '</title></rect>'
      + '<rect x="' + plotX + '" y="' + (y + 13) + '" width="' + falseWidth.toFixed(2) + '" height="9" rx="2" fill="var(--failure)"><title>'
      + escapeHtml("Clean-success trajectory false alarm " + workspacePercent(falseAlarm)) + '</title></rect>'
      + '<text x="' + (plotX + plotWidth + 9) + '" y="' + (y + 9) + '" class="chart-value">R ' + escapeHtml(workspacePercent(recall)) + '</text>'
      + '<text x="' + (plotX + plotWidth + 9) + '" y="' + (y + 22) + '">FA ' + escapeHtml(workspacePercent(falseAlarm)) + '</text>';
  });
  return workspaceSvg(width, height, "Local change-point recall and trajectory false-alarm comparison", content)
    + '<div class="analysis-legend"><span><i style="background:var(--accent)"></i>event recall</span><span><i style="background:var(--failure)"></i>clean-success false alarm</span></div>';
}

function workspaceRenderChangePointErrorChart(rows) {
  if (!rows.length) return workspaceEmpty("No local change-point error values match the selected filters.");
  var ordered = rows.slice().sort(function (left, right) {
    return workspaceChangePointCategory(left).localeCompare(workspaceChangePointCategory(right));
  });
  var width = Math.max(900, 430 + ordered.length * 10);
  var height = 250;
  var plotX = 48;
  var plotY = 18;
  var plotHeight = 170;
  var plotWidth = width - 92;
  var maxError = Math.max.apply(null, ordered.map(function (row) {
    var firstError = workspaceNumber(row.median_first_exceedance_absolute_error_frames);
    return firstError != null
      ? firstError
      : (workspaceNumber(row.median_peak_distance_frames) || 0);
  }).concat([30, 1]));
  var content = '<line class="chart-axis" x1="' + plotX + '" y1="' + (plotY + plotHeight) + '" x2="' + (plotX + plotWidth) + '" y2="' + (plotY + plotHeight) + '"></line>'
    + '<text x="8" y="' + (plotY + 5) + '">frames</text>';
  ordered.forEach(function (row, index) {
    var slot = plotWidth / ordered.length;
    var barWidth = Math.max(12, slot - 6);
    var x = plotX + index * slot + (slot - barWidth) / 2;
    var error = workspaceNumber(row.median_first_exceedance_absolute_error_frames);
    if (error == null) error = workspaceNumber(row.median_peak_distance_frames);
    var peakDistance = workspaceNumber(row.median_peak_distance_frames);
    var barHeight = error == null ? 0 : plotHeight * Math.min(maxError, error) / maxError;
    var label = workspaceChangePointCategory(row);
    content += '<rect x="' + x.toFixed(2) + '" y="' + (plotY + plotHeight - barHeight).toFixed(2) + '" width="' + barWidth.toFixed(2) + '" height="' + barHeight.toFixed(2) + '" rx="2" fill="var(--recovery)"><title>'
      + escapeHtml(label + " median first-crossing absolute error " + workspaceFormatNumber(error) + " frames; peak distance " + workspaceFormatNumber(peakDistance) + "; +/-8 first-crossing hit " + workspacePercent(row.first_hit_rate_within_8_frames)) + '</title></rect>';
    content += '<text x="' + (x + barWidth / 2).toFixed(2) + '" y="' + (plotY + plotHeight + 18) + '" text-anchor="middle" data-legacy-rotation="rotation(-38 ' + (x + barWidth / 2).toFixed(2) + ' ' + (plotY + plotHeight + 18) + ')">'
      + escapeHtml(label) + '</text>';
  });
  var tolerance = [4, 8, 16, 30].map(function (value) {
    var values = rows.map(function (row) {
      var firstRate = workspaceNumber(row["first_hit_rate_within_" + value + "_frames"]);
      return firstRate != null
        ? firstRate
        : workspaceNumber(row["peak_hit_rate_within_" + value + "_frames"]);
    }).filter(function (number) { return number != null; });
    return value + "f " + workspacePercent(values.length ? workspaceMedian(values) : null);
  }).join(" | ");
  content += '<text x="' + plotX + '" y="' + (height - 10) + '" class="chart-value">First-crossing tolerance hit rates: ' + escapeHtml(tolerance) + '</text>';
  return workspaceSvg(width, height, "Local change-point first-crossing localization error and tolerance hits", content)
    + '<div class="analysis-legend"><span><i style="background:var(--recovery)"></i>median first-crossing absolute error (peak distance is fallback)</span><span>first-crossing tolerance rates are listed below</span></div>';
}

function workspaceChangePointFailureMetricRows(changePoint) {
  var filters = workspaceState.changePointFilters;
  var rows = workspaceChangePointFailureRows(changePoint).filter(function (row) {
    return workspaceChangePointBaseFilter(row, filters)
      && (filters.outcome === "all" || row.outcome_group == null || row.outcome_group === filters.outcome);
  });
  if (filters.outcome === "all" && filters.task === "all") return rows;
  var events = workspaceChangePointFilteredEvents(changePoint);
  var groups = {};
  events.forEach(function (row) {
    var key = [row.method, row.signal, row.feature, row.scale_frames, row.threshold, row.failure_type].join("::");
    if (!groups[key]) {
      groups[key] = {
        method: row.method,
        signal: row.signal,
        feature: row.feature,
        scale_frames: row.scale_frames,
        threshold: row.threshold,
        failure_type: row.failure_type,
        n_events: 0,
        n_hits: 0,
        distances: []
      };
    }
    groups[key].n_events += 1;
    if (row.hit) groups[key].n_hits += 1;
    if (row.peak_distance_frames != null) groups[key].distances.push(Number(row.peak_distance_frames));
  });
  var allFailureRows = workspaceChangePointFailureRows(changePoint);
  return Object.keys(groups).map(function (key) {
    var row = groups[key];
    row.recall = row.n_events ? row.n_hits / row.n_events : null;
    row.median_peak_distance_frames = workspaceMedian(row.distances);
    var base = allFailureRows.find(function (candidate) {
      return candidate.method === row.method && candidate.signal === row.signal
        && candidate.feature === row.feature
        && String(candidate.scale_frames) === String(row.scale_frames)
        && candidate.threshold === row.threshold
        && candidate.failure_type === row.failure_type;
    });
    if (base) {
      [
        "false_alarm_rate",
        "false_alarm_count",
        "clean_pseudo_events",
        "precision",
        "f1",
        "auroc",
        "average_precision"
      ].forEach(function (field) {
        row[field] = workspaceNumber(base[field]);
      });
    }
    return row;
  });
}

function workspaceRenderChangePointFailureTypes(changePoint) {
  var rows = workspaceChangePointFailureMetricRows(changePoint);
  if (!rows.length) return workspaceEmpty("No failure-type local change-point rows match the selected filters.");
  rows.sort(function (left, right) {
    return String(left.failure_type).localeCompare(String(right.failure_type))
      || workspaceChangePointCategory(left).localeCompare(workspaceChangePointCategory(right));
  });
  var html = '<table class="analysis-table changepoint-failure-table"><caption>Recall, false alarms, precision/F1 and ranking diagnostics by annotated failure type; each signal, feature, scale, and threshold remains separate.</caption><thead><tr>'
    + '<th>Failure type</th><th>Method / signal / feature / scale</th><th>Threshold</th><th>Events</th><th>Recall</th><th>False alarm</th><th>Precision</th><th>F1</th><th>AUROC</th><th>AP</th><th>Median peak distance</th></tr></thead><tbody>';
  rows.slice(0, 240).forEach(function (row) {
    var recall = workspaceNumber(row.recall);
    var intensity = recall == null ? 0.05 : Math.max(0.08, Math.min(0.9, recall));
    html += '<tr><td>' + escapeHtml(labelFor(row.failure_type || "other")) + '</td>'
      + '<td><strong>' + escapeHtml(workspaceChangePointCategory(row)) + '</strong></td>'
      + '<td>' + escapeHtml(String(row.threshold || "").toUpperCase()) + '</td>'
      + '<td class="numeric">' + escapeHtml(row.n_events == null ? "n/a" : row.n_events) + '</td>'
      + '<td class="numeric localization-heat-cell" style="--localization-heat:' + intensity.toFixed(2) + '">' + escapeHtml(workspacePercent(recall)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspacePercent(row.false_alarm_rate)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.precision)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.f1)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.auroc)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.average_precision)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.median_peak_distance_frames)) + '</td></tr>';
  });
  return html + '</tbody></table>';
}

function workspaceRenderChangePointComparison(changePoint) {
  var rows = (changePoint.comparison || []).filter(function (row) {
    var filters = workspaceState.changePointFilters;
    return (filters.method === "all" || row.method === filters.method)
      && (filters.signal === "all" || row.signal === filters.signal)
      && (filters.feature === "all" || row.feature == null || row.feature === filters.feature)
      && (filters.scale === "all" || row.scale_frames == null || String(row.scale_frames) === String(filters.scale))
      && (filters.threshold === "all" || row.threshold == null || row.threshold === filters.threshold)
      && (filters.outcome === "all" || row.outcome_group == null || row.outcome_group === filters.outcome || row.outcome_group === "all_events");
  });
  if (!rows.length) return workspaceEmpty("No comparison rows match the selected filters.");
  var html = '<table class="analysis-table changepoint-comparison-table"><caption>Current values and legacy values; blank legacy cells mean that full_136 had no equivalent local metric.</caption><thead><tr>'
    + '<th>Family</th><th>Method / signal</th><th>Feature</th><th>Scale</th><th>Outcome</th><th>Metric</th><th>Current</th><th>Legacy full_136</th><th>Delta</th></tr></thead><tbody>';
  rows.slice(0, 240).forEach(function (row) {
    html += '<tr><td>' + escapeHtml(row.comparison_family || "") + '</td>'
      + '<td><strong>' + escapeHtml((ANALYSIS_METHOD_LABELS[row.method] || row.method) + " / " + (row.signal || "")) + '</strong></td>'
      + '<td>' + escapeHtml(row.feature || "global") + '</td>'
      + '<td class="numeric">' + escapeHtml(row.scale_frames == null ? "global" : row.scale_frames + "f") + '</td>'
      + '<td>' + escapeHtml(labelFor(row.outcome_group || "all_events")) + '</td>'
      + '<td>' + escapeHtml(row.metric || "") + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.current_value)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.previous_full_136_value)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.delta_from_previous)) + '</td></tr>';
  });
  return html + '</tbody></table>';
}

function workspaceRenderChangePointEvents(changePoint) {
  var rows = workspaceChangePointFilteredEvents(changePoint).slice().sort(function (left, right) {
    return (Number(right.peak_score) || 0) - (Number(left.peak_score) || 0);
  }).slice(0, 160);
  if (!rows.length) return workspaceEmpty("No event-level local change rows match the selected filters.");
  var html = '<table class="analysis-table changepoint-events-table"><caption>Showing the highest local unusual scores first; event rows use native samples only.</caption><thead><tr>'
    + '<th>Method / signal</th><th>Feature</th><th>Scale</th><th>Threshold</th><th>Outcome</th><th>Failure type</th><th>Rollout</th><th>Onset</th><th>Peak distance</th><th>First error</th><th>Result</th><th></th></tr></thead><tbody>';
  rows.forEach(function (row) {
    var hit = Boolean(row.hit);
    var rolloutId = row.rollout_id || "unknown";
    html += '<tr><td><strong>' + escapeHtml((ANALYSIS_METHOD_LABELS[row.method] || row.method) + " / " + row.signal) + '</strong></td>'
      + '<td>' + escapeHtml(row.feature || "") + '</td>'
      + '<td class="numeric">' + escapeHtml(row.scale_frames == null ? "n/a" : row.scale_frames + "f") + '</td>'
      + '<td>' + escapeHtml(String(row.threshold || "").toUpperCase()) + '</td>'
      + '<td>' + escapeHtml(labelFor(row.outcome_group || "uncertain")) + '</td>'
      + '<td>' + escapeHtml(labelFor(row.failure_type || "other")) + '</td>'
      + '<td class="numeric">' + escapeHtml(rolloutId) + '</td>'
      + '<td class="numeric">' + escapeHtml(row.observable_onset_frame == null ? "n/a" : row.observable_onset_frame) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.peak_distance_frames)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.first_exceedance_absolute_error_frames)) + '</td>'
      + '<td><span class="localization-result ' + (hit ? "hit" : "miss") + '">' + (hit ? "hit" : "miss") + '</span></td>'
      + '<td><button type="button" data-analysis-rollout="' + escapeHtml(rolloutId) + '">Review</button></td></tr>';
  });
  return html + '</tbody></table>';
}

function workspaceRenderChangePoint(snapshot) {
  var badge = byId("analysisChangePointBadge");
  var status = byId("analysisChangePointStatus");
  if (!badge || !status) return;
  var changePoint = workspaceChangePointSnapshot(snapshot);
  if (!changePoint) {
    badge.className = "analysis-badge";
    badge.textContent = "Unavailable";
    status.className = "analysis-status warning";
    status.textContent = snapshot && snapshot.change_point && snapshot.change_point.message
      ? snapshot.change_point.message
      : "Primary local change-point analysis is unavailable; the legacy temporal comparison remains usable.";
    ["analysisChangePointRecallChart", "analysisChangePointErrorChart", "analysisChangePointFailureType", "analysisChangePointComparison", "analysisChangePointEvents", "analysisChangePointProvenance"].forEach(function (id) {
      if (byId(id)) byId(id).innerHTML = workspaceEmpty("Local change-point analysis unavailable");
    });
    return;
  }
  workspaceRefreshChangePointOptions(changePoint);
  var rows = workspaceChangePointMetricRows(changePoint);
  var events = workspaceChangePointFilteredEvents(changePoint);
  var freshness = changePoint.freshness || {};
  var source = changePoint.source || {};
  var counts = changePoint.counts || {};
  badge.className = "analysis-badge" + (freshness.stale ? " warning" : " ok");
  badge.textContent = freshness.stale ? "Stale snapshot" : "Ready";
  status.className = "analysis-status" + (freshness.stale ? " warning" : "");
  status.textContent = "Descriptive local change-point diagnostics: "
    + (counts.rollouts || "?") + " rollouts, " + (counts.observable_events || "?") + " observable events, "
    + (changePoint.local_scales_frames || []).join("/") + " frame scales. Direction is inspected only after an unusual change; this is not independently validated detector performance.";
  var coverage = (changePoint.method_coverage || []).map(function (row) {
    return (ANALYSIS_METHOD_LABELS[row.method] || row.method) + " " + (row.available_rollouts || 0) + "/" + (row.selected_rollouts || 0);
  }).join("; ");
  byId("analysisChangePointProvenance").innerHTML = '<strong>Source:</strong> ' + escapeHtml(source.directory || "unknown")
    + ' · <strong>generated:</strong> ' + escapeHtml(source.generated_at || "unknown")
    + ' · <strong>coverage:</strong> ' + escapeHtml(coverage)
    + ' · <strong>valid event rows:</strong> ' + escapeHtml(events.length)
    + ' · native samples only; reference regions are clean-success and same-rollout non-onset regions.';
  byId("analysisChangePointRecallChart").innerHTML = workspaceRenderChangePointRecallChart(rows);
  byId("analysisChangePointErrorChart").innerHTML = workspaceRenderChangePointErrorChart(rows);
  byId("analysisChangePointFailureType").innerHTML = workspaceRenderChangePointFailureTypes(changePoint);
  byId("analysisChangePointComparison").innerHTML = workspaceRenderChangePointComparison(changePoint);
  byId("analysisChangePointEvents").innerHTML = workspaceRenderChangePointEvents(changePoint);
}

function workspaceEventTriggeredSnapshot(snapshot) {
  var value = snapshot && snapshot.event_triggered;
  return value && value.available ? value : null;
}

function workspaceEventTriggeredRowMatches(row, filters, includeScale) {
  var group = row.event_group || row.comparison_group;
  if (filters.method !== "all" && row.method !== filters.method) return false;
  if (filters.signal !== "all" && row.signal !== filters.signal) return false;
  if (filters.event_group !== "all" && group !== filters.event_group) return false;
  if (includeScale && filters.scale !== "all"
      && row.scale_frames != null
      && String(row.scale_frames) !== String(filters.scale)) return false;
  return true;
}

function workspaceEventTriggeredGroups(rows) {
  var groups = {};
  rows.forEach(function (row) {
    var group = row.event_group || row.comparison_group;
    if (ANALYSIS_EVENT_GROUPS.indexOf(group) !== -1) groups[group] = true;
  });
  return ANALYSIS_EVENT_GROUPS.filter(function (group) { return groups[group]; });
}

function workspaceRefreshEventTriggeredOptions(eventTriggered) {
  if (!eventTriggered) return;
  var filters = workspaceState.eventTriggeredFilters;
  var methods = (eventTriggered.methods || ANALYSIS_METHODS).filter(function (method) {
    return ANALYSIS_METHODS.indexOf(method) !== -1;
  });
  workspaceSetOptions(
    byId("analysisEventTriggeredMethod"),
    methods.map(function (method) {
      return { value: method, label: ANALYSIS_METHOD_LABELS[method] || method };
    }),
    filters.method,
    "All methods"
  );
  filters.method = byId("analysisEventTriggeredMethod").value;

  var signalValues = {};
  var curveRows = eventTriggered.curves || [];
  var changeRows = eventTriggered.change_scores || [];
  curveRows.concat(changeRows).forEach(function (row) {
    if (filters.method === "all" || row.method === filters.method) {
      if (row.signal != null) signalValues[String(row.signal)] = String(row.signal);
    }
  });
  if (filters.method !== "all" && eventTriggered.signals_by_method
      && Array.isArray(eventTriggered.signals_by_method[filters.method])) {
    eventTriggered.signals_by_method[filters.method].forEach(function (signal) {
      signalValues[String(signal)] = String(signal);
    });
  }
  workspaceSetOptions(
    byId("analysisEventTriggeredSignal"),
    Object.keys(signalValues).sort().map(function (signal) {
      return { value: signal, label: signal };
    }),
    filters.signal,
    "All signals"
  );
  filters.signal = byId("analysisEventTriggeredSignal").value;

  var scaleValues = {};
  changeRows.forEach(function (row) {
    if ((filters.method === "all" || row.method === filters.method)
        && (filters.signal === "all" || row.signal === filters.signal)
        && row.scale_frames != null) {
      scaleValues[String(row.scale_frames)] = String(row.scale_frames) + " frames";
    }
  });
  workspaceSetOptions(
    byId("analysisEventTriggeredScale"),
    Object.keys(scaleValues).sort(function (left, right) {
      return Number(left) - Number(right);
    }).map(function (scale) {
      return { value: scale, label: scaleValues[scale] };
    }),
    filters.scale,
    "All scales"
  );
  filters.scale = byId("analysisEventTriggeredScale").value;
  var groupSelect = byId("analysisEventTriggeredGroup");
  if (groupSelect) {
    var groupValues = ANALYSIS_EVENT_GROUPS.filter(function (group) {
      return workspaceEventTriggeredGroups(curveRows).indexOf(group) !== -1;
    });
    Array.prototype.forEach.call(groupSelect.options, function (option) {
      option.hidden = option.value !== "all" && groupValues.indexOf(option.value) === -1;
    });
    groupSelect.value = filters.event_group;
    if (groupSelect.value !== filters.event_group) filters.event_group = "all";
  }
}

function workspaceEventTriggeredDomain(rows, lowField, highField, includeZero) {
  var values = [];
  rows.forEach(function (row) {
    var low = workspaceNumber(row[lowField]);
    var high = workspaceNumber(row[highField]);
    if (low != null) values.push(low);
    if (high != null) values.push(high);
  });
  if (includeZero) values.push(0);
  if (!values.length) return [-1, 1];
  var lowValue = Math.min.apply(null, values);
  var highValue = Math.max.apply(null, values);
  if (lowValue === highValue) {
    var padding = Math.max(1, Math.abs(lowValue) * 0.2);
    return [lowValue - padding, highValue + padding];
  }
  var margin = (highValue - lowValue) * 0.08;
  return [lowValue - margin, highValue + margin];
}

function workspaceEventTriggeredX(relative, left, width, pre, post) {
  var value = workspaceNumber(relative);
  if (value == null) return null;
  var span = Math.max(1, post + pre);
  return left + Math.max(0, Math.min(1, (value + pre) / span)) * width;
}

function workspaceEventTriggeredY(value, top, height, domain) {
  var number = workspaceNumber(value);
  if (number == null) return null;
  var span = Math.max(1e-12, domain[1] - domain[0]);
  return top + height - (number - domain[0]) / span * height;
}

function workspaceEventTriggeredValidPoints(rows, field, xFn, yFn) {
  return rows.map(function (row) {
    var x = xFn(row.relative_frame);
    var y = yFn(row[field]);
    return x == null || y == null ? null : { x: x, y: y, row: row };
  }).filter(function (point) { return point != null; });
}

function workspaceEventTriggeredSeries(rows, field, xFn, yFn, color, dashed, label, fillLow, fillHigh, fillOpacity) {
  var points = workspaceEventTriggeredValidPoints(rows, field, xFn, yFn);
  if (!points.length) return "";
  var content = '<g><title>' + escapeHtml(label) + '</title>';
  if (fillLow && fillHigh) {
    var lower = workspaceEventTriggeredValidPoints(rows, fillLow, xFn, yFn);
    var upper = workspaceEventTriggeredValidPoints(rows, fillHigh, xFn, yFn);
    if (lower.length && upper.length) {
      var polygon = upper.concat(lower.slice().reverse()).map(function (point) {
        return point.x.toFixed(2) + "," + point.y.toFixed(2);
      }).join(" ");
      content += '<polygon points="' + polygon + '" fill="' + color + '" opacity="' + fillOpacity + '"></polygon>';
    }
  }
  content += '<polyline fill="none" stroke="' + color + '" stroke-width="' + (dashed ? "1.4" : "2") + '"'
    + (dashed ? ' stroke-dasharray="5 4"' : "") + ' points="'
    + points.map(function (point) {
      return point.x.toFixed(2) + "," + point.y.toFixed(2);
    }).join(" ") + '"></polyline>';
  content += points.map(function (point) {
    return '<circle cx="' + point.x.toFixed(2) + '" cy="' + point.y.toFixed(2) + '" r="1.7" fill="' + color + '"></circle>';
  }).join("");
  return content + "</g>";
}

function workspaceEventTriggeredAxis(left, top, width, height, pre, post, domain, zero) {
  var zeroLine = "";
  if (zero && domain[0] <= 0 && domain[1] >= 0) {
    var y = workspaceEventTriggeredY(0, top, height, domain);
    zeroLine = '<line class="chart-grid" x1="' + left + '" y1="' + y.toFixed(2)
      + '" x2="' + (left + width) + '" y2="' + y.toFixed(2) + '"></line>';
  }
  return '<line class="chart-axis" x1="' + left + '" y1="' + (top + height)
    + '" x2="' + (left + width) + '" y2="' + (top + height) + '"></line>'
    + '<line class="chart-grid" x1="' + left + '" y1="' + top
    + '" x2="' + left + '" y2="' + (top + height) + '"></line>'
    + zeroLine
    + '<text x="' + (left - 8) + '" y="' + (top + 7) + '" text-anchor="end">'
    + escapeHtml(workspaceFormatNumber(domain[1])) + '</text>'
    + '<text x="' + (left - 8) + '" y="' + (top + height) + '" text-anchor="end">'
    + escapeHtml(workspaceFormatNumber(domain[0])) + '</text>'
    + '<text x="' + left + '" y="' + (top + height + 17) + '" text-anchor="middle">-'
    + escapeHtml(String(pre)) + '</text>'
    + '<text x="' + (left + width / 2) + '" y="' + (top + height + 17) + '" text-anchor="middle">0</text>'
    + '<text x="' + (left + width) + '" y="' + (top + height + 17) + '" text-anchor="middle">+'
    + escapeHtml(String(post)) + '</text>';
}

function workspaceRenderEventTriggeredSignalChart(eventTriggered) {
  var filters = workspaceState.eventTriggeredFilters;
  var rows = (eventTriggered.curves || []).filter(function (row) {
    return workspaceEventTriggeredRowMatches(row, filters, false);
  });
  var groups = workspaceEventTriggeredGroups(rows);
  if (!rows.length || !groups.length) return workspaceEmpty("No onset-aligned signal curves match the selected filters.");
  var parameters = eventTriggered.parameters || {};
  var pre = Number(parameters.pre_window_frames) || 60;
  var post = Number(parameters.post_window_frames) || 60;
  var width = 860;
  var height = 438;
  var left = 68;
  var plotWidth = 760;
  var panelHeight = 154;
  var domains = {
    raw: workspaceEventTriggeredDomain(rows, "raw_q25", "raw_q75", false),
    normalized: workspaceEventTriggeredDomain(rows, "normalized_q25", "normalized_q75", true)
  };
  var content = "";
  [
    ["raw", "Raw native signal", "raw_median", "raw_q25", "raw_q75", false],
    ["normalized", "Onset-normalized signal", "normalized_median", "normalized_q25", "normalized_q75", true]
  ].forEach(function (panel, index) {
    var top = 30 + index * 197;
    var xFn = function (relative) {
      return workspaceEventTriggeredX(relative, left, plotWidth, pre, post);
    };
    var yFn = function (value) {
      return workspaceEventTriggeredY(value, top, panelHeight, domains[panel[0]]);
    };
    content += '<text x="' + left + '" y="' + (top - 10) + '" class="chart-label">'
      + escapeHtml(panel[1]) + '</text>';
    content += workspaceEventTriggeredAxis(
      left, top, plotWidth, panelHeight, pre, post, domains[panel[0]], panel[5]
    );
    groups.forEach(function (eventGroup) {
      var eventRows = rows.filter(function (row) {
        return row.comparison_group === eventGroup && row.group === eventGroup;
      }).sort(function (a, b) { return Number(a.relative_frame) - Number(b.relative_frame); });
      var cleanRows = rows.filter(function (row) {
        return row.comparison_group === eventGroup && row.group === "matched_clean_success";
      }).sort(function (a, b) { return Number(a.relative_frame) - Number(b.relative_frame); });
      var color = ANALYSIS_EVENT_COLORS[eventGroup];
      content += workspaceEventTriggeredSeries(
        eventRows, panel[2], xFn, yFn, color, false,
        (ANALYSIS_METHOD_LABELS[filters.method] || "Selected method") + " / "
          + (filters.signal === "all" ? "all signals" : filters.signal) + " / " + eventGroup,
        panel[3], panel[4], "0.18"
      );
      content += workspaceEventTriggeredSeries(
        cleanRows, panel[2], xFn, yFn, color, true,
        (ANALYSIS_METHOD_LABELS[filters.method] || "Selected method") + " / "
          + (filters.signal === "all" ? "all signals" : filters.signal) + " / "
          + eventGroup + " matched clean control",
        panel[3], panel[4], "0.06"
      );
    });
    content += '<text x="' + (left + plotWidth / 2) + '" y="' + (top + panelHeight + 35)
      + '" text-anchor="middle">relative frame; observable onset = 0</text>';
  });
  return workspaceSvg(
    width, height, "Onset-aligned raw and normalized native signal curves", content
  ) + '<div class="analysis-legend">'
    + groups.map(function (group) {
      return '<span><i style="background:' + ANALYSIS_EVENT_COLORS[group] + '"></i>'
        + escapeHtml(group.replace(/_/g, " ")) + '</span>';
    }).join("")
    + '<span>solid = event · dashed = matched clean control</span></div>';
}

function workspaceRenderEventTriggeredChangeChart(eventTriggered) {
  var filters = workspaceState.eventTriggeredFilters;
  var allRows = eventTriggered.change_scores || [];
  var rows = allRows.filter(function (row) {
    return workspaceEventTriggeredRowMatches(row, filters, true);
  });
  if (!rows.length) return workspaceEmpty("No local change-score curves match the selected filters.");
  var scales = {};
  rows.forEach(function (row) {
    if (row.scale_frames != null) scales[String(row.scale_frames)] = Number(row.scale_frames);
  });
  var scaleList = Object.keys(scales).map(function (key) { return scales[key]; }).sort(function (a, b) { return a - b; });
  if (!scaleList.length) return workspaceEmpty("No local change-score scales are available.");
  var parameters = eventTriggered.parameters || {};
  var pre = Number(parameters.pre_window_frames) || 60;
  var post = Number(parameters.post_window_frames) || 60;
  var columns = Math.min(2, scaleList.length);
  var panelWidth = 430;
  var panelHeight = 220;
  var rowsCount = Math.ceil(scaleList.length / columns);
  var width = panelWidth * columns;
  var height = panelHeight * rowsCount + 12;
  var content = "";
  scaleList.forEach(function (scale, index) {
    var originX = (index % columns) * panelWidth;
    var originY = Math.floor(index / columns) * panelHeight;
    var plotLeft = originX + 55;
    var plotTop = originY + 30;
    var plotWidth = panelWidth - 78;
    var plotHeight = 148;
    var scaleRows = rows.filter(function (row) {
      return Number(row.scale_frames) === scale;
    });
    var domain = workspaceEventTriggeredDomain(
      scaleRows, "score_q25", "score_q75", true
    );
    var xFn = function (relative) {
      return workspaceEventTriggeredX(relative, plotLeft, plotWidth, pre, post);
    };
    var yFn = function (value) {
      return workspaceEventTriggeredY(value, plotTop, plotHeight, domain);
    };
    content += '<text x="' + plotLeft + '" y="' + (originY + 18) + '" class="chart-label">scale = '
      + escapeHtml(String(scale)) + ' frames</text>';
    content += workspaceEventTriggeredAxis(
      plotLeft, plotTop, plotWidth, plotHeight, pre, post, domain, true
    );
    workspaceEventTriggeredGroups(scaleRows).forEach(function (eventGroup) {
      var eventRows = scaleRows.filter(function (row) {
        return row.comparison_group === eventGroup && row.group === eventGroup;
      }).sort(function (a, b) { return Number(a.relative_frame) - Number(b.relative_frame); });
      var cleanRows = scaleRows.filter(function (row) {
        return row.comparison_group === eventGroup && row.group === "matched_clean_success";
      }).sort(function (a, b) { return Number(a.relative_frame) - Number(b.relative_frame); });
      var color = ANALYSIS_EVENT_COLORS[eventGroup];
      content += workspaceEventTriggeredSeries(
        eventRows, "score_median", xFn, yFn, color, false,
        eventGroup + " local change score (" + scale + " frames)",
        "score_q25", "score_q75", "0.18"
      );
      content += workspaceEventTriggeredSeries(
        cleanRows, "score_median", xFn, yFn, color, true,
        eventGroup + " matched clean control (" + scale + " frames)",
        "score_q25", "score_q75", "0.06"
      );
    });
    content += '<text x="' + (plotLeft + plotWidth / 2) + '" y="' + (plotTop + plotHeight + 37)
      + '" text-anchor="middle">relative frame</text>';
  });
  return workspaceSvg(
    width, height, "Native local change-score curves around observable onset", content
  ) + '<div class="analysis-legend">'
    + ANALYSIS_EVENT_GROUPS.map(function (group) {
      return '<span><i style="background:' + ANALYSIS_EVENT_COLORS[group] + '"></i>'
        + escapeHtml(group.replace(/_/g, " ")) + '</span>';
    }).join("")
    + '<span>solid = event · dashed = matched clean control</span></div>';
}

function workspaceEventTriggeredSummaryRows(eventTriggered) {
  var filters = workspaceState.eventTriggeredFilters;
  return (eventTriggered.summary || []).filter(function (row) {
    if (!workspaceEventTriggeredRowMatches(row, filters, false)) return false;
    if (filters.scale !== "all" && row.representation === "local_change_score"
        && String(row.scale_frames) !== String(filters.scale)) return false;
    return true;
  }).sort(function (left, right) {
    return ANALYSIS_METHODS.indexOf(left.method) - ANALYSIS_METHODS.indexOf(right.method)
      || String(left.signal).localeCompare(String(right.signal))
      || String(left.event_group).localeCompare(String(right.event_group))
      || String(left.representation).localeCompare(String(right.representation))
      || (Number(left.scale_frames) || 0) - (Number(right.scale_frames) || 0);
  });
}

function workspaceRenderEventTriggeredSummary(eventTriggered) {
  var rows = workspaceEventTriggeredSummaryRows(eventTriggered);
  if (!rows.length) return workspaceEmpty("No case/control separation summary matches the selected filters.");
  var html = '<table class="analysis-table event-triggered-summary-table"><caption>Strongest aggregate case-versus-control separation; peak selection is descriptive and signal-specific.</caption><thead><tr>'
    + '<th>Method / signal</th><th>Event group</th><th>Representation</th><th>Scale</th><th>Strongest frame</th><th>Phase</th><th>Signed separation</th><th>Separation z</th><th>Typical lag</th><th>Before</th><th>At onset</th><th>After</th><th>Events</th><th>Controls</th></tr></thead><tbody>';
  rows.slice(0, 240).forEach(function (row) {
    var scale = row.scale_frames == null ? "signal" : row.scale_frames + "f";
    html += '<tr><td><strong>' + escapeHtml((ANALYSIS_METHOD_LABELS[row.method] || row.method) + " / " + row.signal) + '</strong></td>'
      + '<td>' + escapeHtml(String(row.event_group || "").replace(/_/g, " ")) + '</td>'
      + '<td>' + escapeHtml(String(row.representation || "").replace(/_/g, " ")) + '</td>'
      + '<td class="numeric">' + escapeHtml(scale) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.strongest_relative_frame, 0)) + '</td>'
      + '<td>' + escapeHtml(row.strongest_phase || "n/a") + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.strongest_signed_separation)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.strongest_separation_z)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.typical_lag_frames)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspacePercent(row.before_fraction)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspacePercent(row.at_onset_fraction)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspacePercent(row.after_fraction)) + '</td>'
      + '<td class="numeric">' + escapeHtml(row.n_events == null ? "n/a" : row.n_events) + '</td>'
      + '<td class="numeric">' + escapeHtml(row.n_controls == null ? "n/a" : row.n_controls) + '</td></tr>';
  });
  return html + '</tbody></table>';
}

function workspaceRenderEventTriggeredPeaks(eventTriggered) {
  var filters = workspaceState.eventTriggeredFilters;
  var rows = (eventTriggered.peak_events || []).filter(function (row) {
    if (!workspaceEventTriggeredRowMatches(row, filters, false)) return false;
    if (filters.scale !== "all" && row.representation === "local_change_score"
        && String(row.scale_frames) !== String(filters.scale)) return false;
    return true;
  }).sort(function (left, right) {
    return (Number(right.peak_separation_z) || 0) - (Number(left.peak_separation_z) || 0);
  }).slice(0, 160);
  if (!rows.length) return workspaceEmpty("No event-level peak rows match the selected filters.");
  var html = '<table class="analysis-table event-triggered-peaks-table"><caption>Top event-level peaks against the matched clean curve; click Review to inspect the rollout.</caption><thead><tr>'
    + '<th>Method / signal</th><th>Event group</th><th>Failure type</th><th>Task</th><th>Rollout</th><th>Peak frame</th><th>Phase</th><th>Peak separation z</th><th>Control</th><th></th></tr></thead><tbody>';
  rows.forEach(function (row) {
    var rolloutId = row.rollout_id || "unknown";
    html += '<tr><td><strong>' + escapeHtml((ANALYSIS_METHOD_LABELS[row.method] || row.method) + " / " + row.signal) + '</strong></td>'
      + '<td>' + escapeHtml(String(row.event_group || "").replace(/_/g, " ")) + '</td>'
      + '<td>' + escapeHtml(labelFor(row.failure_type || "other")) + '</td>'
      + '<td class="numeric">' + escapeHtml(row.task_id == null ? "n/a" : row.task_id) + '</td>'
      + '<td class="numeric">' + escapeHtml(rolloutId) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.peak_relative_frame, 0)) + '</td>'
      + '<td>' + escapeHtml(row.peak_phase || "n/a") + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.peak_separation_z)) + '</td>'
      + '<td class="numeric">' + escapeHtml(row.control_rollout_id || "n/a") + '</td>'
      + '<td><button type="button" data-analysis-rollout="' + escapeHtml(rolloutId) + '">Review</button></td></tr>';
  });
  return html + '</tbody></table>';
}

function workspaceRenderEventTriggered(snapshot) {
  var badge = byId("analysisEventTriggeredBadge");
  var status = byId("analysisEventTriggeredStatus");
  if (!badge || !status) return;
  var eventTriggered = workspaceEventTriggeredSnapshot(snapshot);
  if (!eventTriggered) {
    badge.className = "analysis-badge";
    badge.textContent = "Unavailable";
    status.className = "analysis-status warning";
    status.textContent = snapshot && snapshot.event_triggered && snapshot.event_triggered.message
      ? snapshot.event_triggered.message
      : "No complete event-triggered signal snapshot is available.";
    [
      "analysisEventTriggeredSignalChart",
      "analysisEventTriggeredChangeChart",
      "analysisEventTriggeredSummary",
      "analysisEventTriggeredPeaks",
      "analysisEventTriggeredProvenance"
    ].forEach(function (id) {
      if (byId(id)) byId(id).innerHTML = workspaceEmpty("Event-triggered analysis unavailable");
    });
    return;
  }
  workspaceRefreshEventTriggeredOptions(eventTriggered);
  var filters = workspaceState.eventTriggeredFilters;
  ["method", "signal", "event_group", "scale"].forEach(function (field) {
    var suffix = field === "event_group" ? "Group" : field.charAt(0).toUpperCase() + field.slice(1);
    var select = byId("analysisEventTriggered" + suffix);
    if (select) select.value = filters[field];
  });
  var freshness = eventTriggered.freshness || {};
  var source = eventTriggered.source || {};
  var counts = eventTriggered.counts || {};
  badge.className = "analysis-badge" + (freshness.stale ? " warning" : " ok");
  badge.textContent = freshness.stale ? "Stale snapshot" : "Ready";
  status.className = "analysis-status" + (freshness.stale ? " warning" : "");
  var eventCounts = eventTriggered.event_group_counts || {};
  status.textContent = "Descriptive onset-aligned curves: "
    + (counts.rollouts || "?") + " rollouts, "
    + (counts.observable_events || "?") + " observable events (terminal "
    + (eventCounts.terminal_failure || 0) + ", recovered "
    + (eventCounts.recovered_success || 0) + "). Native sampling is preserved; no interpolation is used.";
  var coverage = (eventTriggered.method_coverage || []).filter(function (row) {
    return (filters.method === "all" || row.method === filters.method)
      && (filters.signal === "all" || row.signal === filters.signal);
  });
  var missingIds = [];
  coverage.forEach(function (row) {
    var rawIds = row.missing_method_rollout_ids;
    if (typeof rawIds === "string") {
      try {
        var ids = JSON.parse(rawIds);
        if (Array.isArray(ids)) missingIds = missingIds.concat(ids);
      } catch (error) {
        if (rawIds) missingIds.push(rawIds);
      }
    }
  });
  var coverageText = coverage.map(function (row) {
    return (ANALYSIS_METHOD_LABELS[row.method] || row.method) + " / " + row.signal + " "
      + (row.available_method_rollouts || 0) + "/" + (row.selected_rollouts || 0);
  }).join("; ");
  byId("analysisEventTriggeredProvenance").innerHTML = '<strong>Source:</strong> '
    + escapeHtml(source.directory || "unknown") + ' · <strong>generated:</strong> '
    + escapeHtml(source.generated_at || "unknown") + ' · <strong>coverage:</strong> '
    + escapeHtml(coverageText || "n/a") + ' · <strong>summary rows:</strong> '
    + escapeHtml(String((eventTriggered.summary || []).length)) + ' · <strong>missing raw rollout IDs:</strong> '
    + escapeHtml(missingIds.length ? missingIds.join(", ") : "none")
    + ' · matched clean controls are recorded per event; metrics are descriptive only.';
  byId("analysisEventTriggeredSignalChart").innerHTML = workspaceRenderEventTriggeredSignalChart(eventTriggered);
  byId("analysisEventTriggeredChangeChart").innerHTML = workspaceRenderEventTriggeredChangeChart(eventTriggered);
  byId("analysisEventTriggeredSummary").innerHTML = workspaceRenderEventTriggeredSummary(eventTriggered);
  byId("analysisEventTriggeredPeaks").innerHTML = workspaceRenderEventTriggeredPeaks(eventTriggered);
}

function workspaceRenderSnapshot() {
  var snapshot = workspaceState.analysisSnapshot;
  if (!snapshot) {
    byId("analysisSnapshotBadge").className = "analysis-badge";
    byId("analysisSnapshotBadge").textContent = workspaceState.analysisLoading ? "Loading" : "Unavailable";
    byId("analysisSnapshotStatus").textContent = workspaceState.analysisLoading ? "Reading the latest complete baseline analysis snapshot…" : "No baseline analysis snapshot is available.";
    workspaceRenderChangePoint(null);
    workspaceRenderEventTriggered(null);
    return;
  }
  if (!snapshot.available) {
    byId("analysisSnapshotBadge").className = "analysis-badge missing";
    byId("analysisSnapshotBadge").textContent = "Missing";
    byId("analysisSnapshotStatus").className = "analysis-status warning";
    byId("analysisSnapshotStatus").textContent = snapshot.message || "No baseline snapshot available.";
    ["analysisCoverageChart", "analysisThresholdChart", "analysisResponseChart", "analysisPersistenceChart", "analysisRecoveryChart", "analysisAnomalies", "analysisProvenance", "analysisLocalizationRecallChart", "analysisLocalizationErrorChart", "analysisLocalizationFailureType", "analysisLocalizationEvents", "analysisLocalizationProvenance"].forEach(function (id) {
      byId(id).innerHTML = workspaceEmpty("Unavailable");
    });
    workspaceRenderChangePoint(snapshot);
    workspaceRenderEventTriggered(snapshot);
    return;
  }
  workspaceRefreshSnapshotTaskOptions(snapshot);
  var freshness = snapshot.freshness || {};
  var source = snapshot.source || {};
  var badgeClass = freshness.stale ? "analysis-badge warning" : "analysis-badge ok";
  byId("analysisSnapshotBadge").className = badgeClass;
  byId("analysisSnapshotBadge").textContent = freshness.stale ? "Stale snapshot" : "Ready";
  byId("analysisSnapshotStatus").className = "analysis-status" + (freshness.stale ? " warning" : "");
  byId("analysisSnapshotStatus").textContent = "Snapshot generated " + (source.generated_at || "unknown time") + " · " + (freshness.snapshot_rollout_count || "?") + " selected rollouts · filters below apply to snapshot event metrics.";
  var rows = workspaceFilteredSnapshotEvents();
  byId("analysisCoverageChart").innerHTML = workspaceRenderCoverageChart(snapshot);
  byId("analysisThresholdChart").innerHTML = workspaceRenderThresholdChart(rows, snapshot);
  byId("analysisResponseChart").innerHTML = workspaceRenderMetricChart(rows, "normalized_response_magnitude", "Median ± IQR", "Normalized response magnitude", false);
  byId("analysisPersistenceChart").innerHTML = workspaceRenderMetricChart(rows, "post_event_persistence_fraction", "Median ± IQR", "Post-event persistence fraction", true);
  byId("analysisRecoveryChart").innerHTML = workspaceRenderMetricChart(rows, "recovery_fraction_toward_baseline", "Recovered events only", "Recovery fraction toward baseline", true);
  byId("analysisAnomalies").innerHTML = workspaceRenderAnomalies(rows);
  workspaceRenderLocalization(snapshot);
  workspaceRenderChangePoint(snapshot);
  workspaceRenderEventTriggered(snapshot);
  var parameters = snapshot.parameters || {};
  var coverage = freshness.snapshot_rollout_count ? "snapshot rollouts: " + freshness.snapshot_rollout_count + "; current manifest: " + freshness.current_rollout_count : "";
  byId("analysisProvenance").innerHTML = '<strong>Source:</strong> ' + escapeHtml(source.directory || "unknown") + ' · <strong>manifest match:</strong> ' + escapeHtml(String(freshness.manifest_matches)) + ' · '
    + escapeHtml(coverage) + ' · <strong>window:</strong> −' + escapeHtml(parameters.pre_window_frames == null ? "?" : parameters.pre_window_frames) + ' / +' + escapeHtml(parameters.post_window_frames == null ? "?" : parameters.post_window_frames) + ' frames · native samples are not interpolated. Results are descriptive and are not detector-performance estimates.';
}

var ANALYSIS_RUN_SELECT_IDS = {
  safe: "analysisRunSafe",
  procvlm: "analysisRunProcvlm",
  rynnvalue: "analysisRunRynnvalue",
  robo_dopamine: "analysisRunRoboDopamine"
};
var ANALYSIS_RUN_SCOPE_LABELS = {
  all: "All manifest rollouts",
  natural_observation: "All natural observations",
  primary_natural: "Primary natural",
  reference_natural: "Reference natural",
  controlled_analysis: "Controlled analysis"
};

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

function workspaceJobChanged(job) {
  if (!job || job.job_type !== "analysis") return;
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
  (jobs || []).filter(function (job) {
    return job.job_type === "analysis";
  }).forEach(function (job) {
    workspaceJobChanged(job);
  });
  var analysisJobs = (jobs || []).filter(function (job) {
    return job.job_type === "analysis";
  });
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
  if (workspaceState.baselineRunsLoading && !force) return;
  if (!force && workspaceState.baselineRunsScope === scope && workspaceState.baselineRunsLoaded) {
    workspaceRenderAnalysisRunPanel();
    return;
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
    workspaceState.settings = payload.settings || Object.assign({}, SETTINGS_DEFAULTS);
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

function workspaceParseRoute() {
  var hash = window.location.hash || "#/review";
  var raw = hash.replace(/^#\/?/, "");
  var parts = raw.split("/");
  var view = ["review", "analysis", "settings"].indexOf(parts[0]) === -1 ? "review" : parts[0];
  var id = parts.length > 1 && parts[1] ? decodeURIComponent(parts.slice(1).join("/")) : null;
  return { view: view, id: id, hash: hash };
}

function workspaceHasUnsavedChanges() {
  return Boolean(state.dirty || workspaceState.settingsDirty);
}

function workspaceShowView(view) {
  ["reviewWorkspace", "analysisView", "settingsView"].forEach(function (id) {
    var node = byId(id);
    if (node) node.classList.toggle("hidden", node.dataset.view !== view);
  });
  document.querySelectorAll("[data-route]").forEach(function (link) {
    var active = link.dataset.route === view;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.documentElement.dataset.activeView = view;
  document.body.dataset.view = view;
  byId("pageTitle").textContent = view === "analysis" ? "Analysis" : view === "settings" ? "Settings" : "Failure Review";
  document.title = "LF3R " + (view === "review" ? "Failure Review" : labelFor(view));
}

function workspaceRenderRoute() {
  var route = workspaceParseRoute();
  if (workspaceState.lastHash && workspaceState.lastHash !== route.hash && workspaceHasUnsavedChanges()) {
    if (!window.confirm("Discard unsaved changes and leave this page?")) {
      window.history.replaceState(null, "", workspaceState.lastHash);
      return;
    }
    state.dirty = false;
    workspaceState.settingsDirty = false;
  }
  workspaceState.lastHash = route.hash;
  workspaceState.view = route.view;
  state.view = route.view;
  workspaceShowView(route.view);
  if (route.view === "review") {
    if (route.id && (state.rollouts || []).some(function (record) { return record.id === route.id; })) {
      selectRollout(route.id);
    } else if (!selectedRollout() && state.filtered && state.filtered.length) {
      selectRollout(state.filtered[0].id);
    }
  } else if (route.view === "analysis") {
    workspaceRenderLiveAnalysis();
    workspaceRenderAnalysisRunPanel();
    workspaceLoadAnalysisEnvironment();
    workspaceLoadAnalysis(false);
    workspaceLoadBaselineRuns(workspaceState.analysisRunScope || "natural_observation");
  } else if (route.view === "settings") {
    workspaceLoadSettings();
  }
}

function workspaceDataChanged() {
  if (workspaceState.view === "review") {
    var route = workspaceParseRoute();
    if (route.id && (state.rollouts || []).some(function (record) { return record.id === route.id; }) && state.selectedId !== route.id) {
      selectRollout(route.id);
    }
  }
  if (workspaceState.view === "analysis") {
    workspaceRenderLiveAnalysis();
    workspaceRenderAnalysisRunPanel();
    workspaceRenderAnalysisJobs();
  }
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

/* -------------------------------------------------------------------------
 * Readable Analysis dashboard
 *
 * The legacy renderers above remain available for old callers, but the live
 * page uses these dashboard renderers. Categorical data is HTML so labels can
 * wrap naturally; SVG is reserved for the one-dimensional signal curves.
 * ------------------------------------------------------------------------- */

function workspaceDashboardShortSignal(signal) {
  var labels = {
    avg_token_entropy: "avg entropy",
    avg_token_prob: "avg prob",
    max_token_entropy: "max entropy",
    max_token_prob: "max prob",
    progress: "progress",
    value: "value",
    hop: "hop"
  };
  return labels[String(signal)] || String(signal || "signal").replace(/_/g, " ");
}

function workspaceDashboardMethodSignal(row) {
  return (ANALYSIS_METHOD_LABELS[row.method] || row.method || "method")
    + " / " + workspaceDashboardShortSignal(row.signal);
}

function workspaceDashboardNumber(value) {
  var number = workspaceNumber(value);
  return number == null ? null : number;
}

function workspaceDashboardRowMatches(row, filters, includeOutcome) {
  if (filters.method !== "all" && row.method !== filters.method) return false;
  if (filters.signal !== "all" && row.signal !== filters.signal) return false;
  if (filters.feature !== "all" && row.feature !== filters.feature) return false;
  if (filters.scale !== "all" && String(row.scale_frames) !== String(filters.scale)) return false;
  if (filters.threshold !== "all" && row.threshold !== filters.threshold) return false;
  if (filters.task !== "all" && String(row.task_id) !== String(filters.task)) return false;
  if (includeOutcome && filters.outcome !== "all" && row.outcome_group !== filters.outcome) return false;
  return true;
}

function workspaceDashboardChangeRows(snapshot) {
  var changePoint = snapshot && snapshot.change_point;
  var rows = workspaceState.analysisDashboardRows && workspaceState.analysisDashboardRows.summary;
  if (!Array.isArray(rows)) rows = changePoint && changePoint.summary;
  if (!Array.isArray(rows)) return [];
  var filters = workspaceState.changePointFilters;
  var base = rows.filter(function (row) {
    return workspaceDashboardRowMatches(row, filters, filters.outcome !== "all");
  });
  if (filters.outcome === "all") {
    var allEvents = base.filter(function (row) { return row.outcome_group === "all_events"; });
    if (allEvents.length) return allEvents;
  }
  return base;
}

function workspaceDashboardFailureRows(snapshot) {
  var changePoint = snapshot && snapshot.change_point;
  var rows = workspaceState.analysisDashboardRows && workspaceState.analysisDashboardRows.failure;
  if (!Array.isArray(rows)) {
    rows = changePoint && (
      changePoint.localization_by_failure_type || changePoint.by_failure_type
    );
  }
  if (!Array.isArray(rows)) return [];
  var filters = workspaceState.localizationFilters;
  return rows.filter(function (row) {
    return workspaceDashboardRowMatches(row, {
      method: filters.method,
      signal: filters.signal,
      feature: workspaceState.changePointFilters.feature,
      scale: workspaceState.changePointFilters.scale,
      threshold: filters.threshold,
      task: filters.task,
      outcome: filters.outcome
    }, filters.outcome !== "all");
  });
}

function workspaceDashboardQuery(kind, filters) {
  var params = { kind: kind, page: "1", page_size: "100" };
  Object.keys(filters || {}).forEach(function (key) {
    if (filters[key] != null && filters[key] !== "" && filters[key] !== "all") {
      params[key] = String(filters[key]);
    }
  });
  return Object.keys(params).map(function (key) {
    return encodeURIComponent(key) + "=" + encodeURIComponent(params[key]);
  }).join("&");
}

function workspaceDashboardFetchRows(kind, filters, done, page, rows, token) {
  page = page || 1;
  rows = rows || [];
  var query = Object.assign({}, filters || {}, {
    kind: kind,
    page: String(page),
    page_size: "100"
  });
  var requestToken = token == null ? null : token;
  fetch("/api/analysis/details?" + workspaceDashboardQuery(kind, query), { cache: "no-store" })
    .then(function (response) {
      return response.json().then(function (payload) {
        if (!response.ok) throw new Error(payload.error || "Could not load Analysis details");
        return payload;
      });
    })
    .then(function (payload) {
      var combined = rows.concat(payload.items || []);
      if (requestToken != null
          && requestToken !== (workspaceState.analysisDetails.request || 0)
          && kind.indexOf("event_triggered") !== 0) return;
      if (payload.page_count && page < payload.page_count && page < 20) {
        workspaceDashboardFetchRows(kind, filters, done, page + 1, combined, requestToken);
      } else {
        done(combined, payload);
      }
    })
    .catch(function (error) {
      done([], { error: error.message });
    });
}

function workspaceDashboardLoadComparisonRows(snapshot) {
  if (workspaceState.analysisDashboardRows && workspaceState.analysisDashboardRows.summary) return;
  workspaceState.analysisDashboardRows = workspaceState.analysisDashboardRows || {};
  workspaceState.analysisDashboardRows.summary = null;
  var filters = workspaceState.changePointFilters;
  var request = {};
  ["method", "signal", "feature", "scale", "threshold", "task"].forEach(function (key) {
    if (filters[key] !== "all") request[key] = filters[key];
  });
  workspaceDashboardFetchRows("changepoint_summary", request, function (rows, meta) {
    workspaceState.analysisDashboardRows.summary = rows;
    if (meta && meta.error) {
      var host = byId("analysisChangePointRecallChart");
      if (host) host.innerHTML = workspaceEmpty("Could not load comparison details: " + meta.error);
    }
    workspaceDashboardRenderComparisonCharts(workspaceDashboardChangeRows(snapshot));
  });
}

function workspaceDashboardLoadFailureRows(snapshot) {
  if (workspaceState.analysisDashboardRows && workspaceState.analysisDashboardRows.failure) return;
  workspaceState.analysisDashboardRows = workspaceState.analysisDashboardRows || {};
  workspaceState.analysisDashboardRows.failure = null;
  var cpFilters = workspaceState.changePointFilters;
  var filters = workspaceState.localizationFilters;
  var request = {
    feature: cpFilters.feature,
    scale: cpFilters.scale,
    threshold: filters.threshold
  };
  if (filters.outcome !== "all") request.outcome = filters.outcome;
  ["method", "signal", "task"].forEach(function (key) {
    if (filters[key] !== "all") request[key] = filters[key];
  });
  workspaceDashboardFetchRows("localization_failure_types", request, function (rows, meta) {
    workspaceState.analysisDashboardRows.failure = rows;
    if (meta && meta.error) {
      var host = byId("analysisChangePointFailureType");
      if (host) host.innerHTML = workspaceEmpty("Could not load failure-type details: " + meta.error);
    }
    workspaceDashboardRenderFailureTypes(snapshot);
  });
}

function workspaceDashboardMetricBar(label, value, fraction, title) {
  var number = workspaceDashboardNumber(value);
  var display = number == null ? "n/a" : fraction ? workspacePercent(number) : workspaceFormatNumber(number);
  var width = number == null ? 0 : Math.max(0, Math.min(100, fraction ? number * 100 : number));
  return '<div class="analysis-bar-metric" title="' + escapeHtml(title || (label + ": " + display)) + '">'
    + '<span class="analysis-bar-label">' + escapeHtml(label) + '</span>'
    + '<span class="analysis-bar-track"><span class="analysis-bar-fill" style="width:' + width.toFixed(1) + '%"></span></span>'
    + '<span class="analysis-bar-value">' + escapeHtml(display) + '</span></div>';
}

function workspaceDashboardAggregateSummary(rows) {
  var groups = {};
  rows.forEach(function (row) {
    var key = String(row.method) + "::" + String(row.signal);
    if (!groups[key]) groups[key] = { method: row.method, signal: row.signal, rows: [] };
    groups[key].rows.push(row);
  });
  return Object.keys(groups).map(function (key) {
    var group = groups[key];
    var first = group.rows[0];
    var output = Object.assign({}, first);
    [
      "recall", "event_hit_rate", "clean_success_false_alarm_rate",
      "false_alarm_rate", "f1", "auroc",
      "median_first_exceedance_absolute_error_frames",
      "median_peak_distance_frames", "peak_hit_rate_within_4_frames",
      "peak_hit_rate_within_8_frames", "peak_hit_rate_within_16_frames",
      "peak_hit_rate_within_30_frames", "first_hit_rate_within_4_frames",
      "first_hit_rate_within_8_frames", "first_hit_rate_within_16_frames",
      "first_hit_rate_within_30_frames"
    ].forEach(function (field) {
      var values = group.rows.map(function (row) {
        return workspaceDashboardNumber(row[field]);
      }).filter(function (value) { return value != null; });
      if (values.length) output[field] = workspaceMedian(values);
    });
    output.n_events = group.rows.reduce(function (sum, row) {
      return sum + (workspaceDashboardNumber(row.n_events) || 0);
    }, 0);
    return output;
  }).sort(function (left, right) {
    return ANALYSIS_METHODS.indexOf(left.method) - ANALYSIS_METHODS.indexOf(right.method)
      || String(left.signal).localeCompare(String(right.signal));
  });
}

function workspaceDashboardRowTitle(row) {
  return workspaceDashboardMethodSignal(row)
    + " | feature=" + (row.feature_label || row.feature || "n/a")
    + " | scale=" + (row.scale_frames == null ? "n/a" : row.scale_frames + " frames")
    + " | threshold=" + (row.threshold || "n/a")
    + " | outcome=" + (row.outcome_group || "all events")
    + " | n=" + (row.n_events == null ? "n/a" : row.n_events);
}

function workspaceDashboardRenderComparisonCharts(rows) {
  var recallHost = byId("analysisChangePointRecallChart");
  var errorHost = byId("analysisChangePointErrorChart");
  var tableHost = byId("analysisChangePointComparison");
  if (!recallHost || !errorHost) return;
  var grouped = workspaceDashboardAggregateSummary(rows);
  if (!grouped.length) {
    recallHost.innerHTML = workspaceEmpty("No rows match the selected method, signal, feature, scale, threshold, and outcome.");
    errorHost.innerHTML = workspaceEmpty("No localization values match the selected filters.");
    if (tableHost) tableHost.innerHTML = "";
    return;
  }
  var recall = '<div class="analysis-horizontal-list" role="list" aria-label="Recall and false alarm by method and signal">';
  grouped.forEach(function (row) {
    var title = workspaceDashboardRowTitle(row);
    recall += '<div class="analysis-summary-row" role="listitem" title="' + escapeHtml(title) + '">'
      + '<div class="analysis-summary-row-heading"><strong>' + escapeHtml(workspaceDashboardMethodSignal(row)) + '</strong><span>n=' + escapeHtml(String(row.n_events == null ? "n/a" : row.n_events)) + '</span></div>'
      + '<div class="analysis-summary-row-meta">' + escapeHtml((row.feature_label || row.feature || "feature") + " / " + (row.scale_frames == null ? "all scales" : row.scale_frames + "f") + " / " + (row.threshold || "threshold")) + '</div>'
      + workspaceDashboardMetricBar("Recall", row.recall == null ? row.event_hit_rate : row.recall, true, title)
      + workspaceDashboardMetricBar("False alarm", row.clean_success_false_alarm_rate == null ? row.false_alarm_rate : row.clean_success_false_alarm_rate, true, title)
      + '</div>';
  });
  recallHost.innerHTML = recall + '</div><p class="analysis-chart-caption">Percentages use the selected event group. Hover or focus a row for complete metadata.</p>';

  var errors = '<div class="analysis-horizontal-list" role="list" aria-label="Localization error and tolerance hits by method and signal">';
  grouped.forEach(function (row) {
    var title = workspaceDashboardRowTitle(row);
    var error = row.median_first_exceedance_absolute_error_frames;
    if (error == null) error = row.median_peak_distance_frames;
    errors += '<div class="analysis-summary-row" role="listitem" title="' + escapeHtml(title) + '">'
      + '<div class="analysis-summary-row-heading"><strong>' + escapeHtml(workspaceDashboardMethodSignal(row)) + '</strong><span>' + escapeHtml(error == null ? "error n/a" : workspaceFormatNumber(error, 1) + " frames") + '</span></div>'
      + '<div class="analysis-summary-row-meta">median absolute localization error; tolerance hit rates</div>'
      + workspaceDashboardMetricBar("+4f hit", row.first_hit_rate_within_4_frames == null ? row.peak_hit_rate_within_4_frames : row.first_hit_rate_within_4_frames, true, title)
      + workspaceDashboardMetricBar("+8f hit", row.first_hit_rate_within_8_frames == null ? row.peak_hit_rate_within_8_frames : row.first_hit_rate_within_8_frames, true, title)
      + workspaceDashboardMetricBar("+16f hit", row.first_hit_rate_within_16_frames == null ? row.peak_hit_rate_within_16_frames : row.first_hit_rate_within_16_frames, true, title)
      + workspaceDashboardMetricBar("+30f hit", row.first_hit_rate_within_30_frames == null ? row.peak_hit_rate_within_30_frames : row.first_hit_rate_within_30_frames, true, title)
      + '</div>';
  });
  errorHost.innerHTML = errors + '</div><p class="analysis-chart-caption">Error is shown as a frame value; hit bars are percentages. No categorical axis is used.</p>';

  if (tableHost) {
    var table = '<table class="analysis-table analysis-summary-table" aria-label="Selected method comparison summary"><thead><tr><th>Method / signal</th><th>Recall</th><th>False alarm</th><th>Median error</th><th>F1</th><th>AUROC</th><th>n</th></tr></thead><tbody>';
    grouped.forEach(function (row) {
      var error = row.median_first_exceedance_absolute_error_frames;
      if (error == null) error = row.median_peak_distance_frames;
      table += '<tr title="' + escapeHtml(workspaceDashboardRowTitle(row)) + '"><th scope="row">' + escapeHtml(workspaceDashboardMethodSignal(row)) + '</th>'
        + '<td class="numeric">' + escapeHtml(workspacePercent(row.recall == null ? row.event_hit_rate : row.recall)) + '</td>'
        + '<td class="numeric">' + escapeHtml(workspacePercent(row.clean_success_false_alarm_rate == null ? row.false_alarm_rate : row.clean_success_false_alarm_rate)) + '</td>'
        + '<td class="numeric">' + escapeHtml(error == null ? "n/a" : workspaceFormatNumber(error, 1) + "f") + '</td>'
        + '<td class="numeric">' + escapeHtml(workspacePercent(row.f1)) + '</td>'
        + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.auroc)) + '</td>'
        + '<td class="numeric">' + escapeHtml(String(row.n_events == null ? "n/a" : row.n_events)) + '</td></tr>';
    });
    tableHost.innerHTML = table + '</tbody></table>';
  }
}

function workspaceDashboardHeatMetric(row, metric) {
  if (metric === "false_alarm") return row.clean_success_false_alarm_rate == null ? row.false_alarm_rate : row.clean_success_false_alarm_rate;
  if (metric === "localization_error") {
    return row.median_absolute_localization_error_frames == null
      ? (row.median_first_exceedance_absolute_error_frames == null ? row.median_peak_distance_frames : row.median_first_exceedance_absolute_error_frames)
      : row.median_absolute_localization_error_frames;
  }
  return row[metric];
}

function workspaceDashboardRenderFailureHeatmap(rows) {
  var host = byId("analysisChangePointFailureType");
  if (!host) return;
  var metric = workspaceState.analysisFailureMetric || "recall";
  var wantedTypes = ["collision", "control_error", "dropped_object", "grasp_failure", "placement_failure", "timeout"];
  var types = wantedTypes.slice();
  rows.forEach(function (row) {
    if (row.failure_type && types.indexOf(row.failure_type) === -1) types.push(row.failure_type);
  });
  var columns = [];
  rows.forEach(function (row) {
    var key = String(row.method) + "::" + String(row.signal);
    if (!columns.some(function (item) { return item.key === key; })) {
      columns.push({ key: key, method: row.method, signal: row.signal });
    }
  });
  columns.sort(function (left, right) {
    return ANALYSIS_METHODS.indexOf(left.method) - ANALYSIS_METHODS.indexOf(right.method)
      || String(left.signal).localeCompare(String(right.signal));
  });
  if (!columns.length) {
    host.innerHTML = workspaceEmpty("No failure-type rows match the selected filters.");
    return;
  }
  var index = {};
  rows.forEach(function (row) {
    index[String(row.method) + "::" + String(row.signal) + "::" + String(row.failure_type)] = row;
  });
  var html = '<div class="analysis-heatmap-scroll"><table class="analysis-heatmap-table" aria-label="Failure type heatmap">'
    + '<caption>' + escapeHtml(metric.replace(/_/g, " ")) + '; each cell shows value and event count</caption><thead><tr><th scope="col">Failure type</th>';
  columns.forEach(function (column) {
    html += '<th scope="col" title="' + escapeHtml((ANALYSIS_METHOD_LABELS[column.method] || column.method) + " / " + column.signal) + '">'
      + escapeHtml((ANALYSIS_METHOD_LABELS[column.method] || column.method) + " / " + workspaceDashboardShortSignal(column.signal)) + '</th>';
  });
  html += '</tr></thead><tbody>';
  types.forEach(function (failureType) {
    html += '<tr><th scope="row">' + escapeHtml(labelFor(failureType)) + '</th>';
    columns.forEach(function (column) {
      var row = index[column.key + "::" + failureType];
      var value = row ? workspaceDashboardHeatMetric(row, metric) : null;
      var number = workspaceDashboardNumber(value);
      var isError = metric === "localization_error";
      var normalized = number == null ? 0 : isError ? Math.min(1, number / 60) : Math.max(0, Math.min(1, number));
      var label = number == null ? "n/a" : isError ? workspaceFormatNumber(number, 1) + "f" : workspacePercent(number);
      var n = row && row.n_events != null ? row.n_events : 0;
      var title = row ? workspaceDashboardRowTitle(row) + " | metric=" + metric + " | value=" + label : "No event for this failure type";
      html += '<td class="analysis-heat-cell" style="--heat:' + normalized.toFixed(3) + '" title="' + escapeHtml(title) + '" aria-label="' + escapeHtml(title) + '"><strong>' + escapeHtml(label) + '</strong><small>n=' + escapeHtml(String(n)) + '</small></td>';
    });
    html += '</tr>';
  });
  host.innerHTML = html + '</tbody></table></div><p class="analysis-chart-caption">Metric: ' + escapeHtml(metric.replace(/_/g, " ")) + '. Missing cells have no events in the selected snapshot.</p>';
}

function workspaceDashboardPopulateChangeOptions(snapshot) {
  var cp = snapshot && snapshot.change_point;
  if (!cp) return;
  var rows = cp.summary || [];
  var signals = {};
  rows.forEach(function (row) {
    if (row.signal != null) signals[String(row.signal)] = String(row.signal);
  });
  var methods = (cp.methods || ANALYSIS_METHODS).filter(function (method) {
    return ANALYSIS_METHODS.indexOf(method) !== -1;
  });
  workspaceSetOptions(byId("analysisChangePointMethod"), methods.map(function (value) {
    return { value: value, label: ANALYSIS_METHOD_LABELS[value] || value };
  }), workspaceState.changePointFilters.method, "All methods");
  workspaceSetOptions(byId("analysisChangePointSignal"), Object.keys(signals).sort().map(function (value) {
    return { value: value, label: workspaceDashboardShortSignal(value) };
  }), workspaceState.changePointFilters.signal, "All native signals");
  var scales = (cp.local_scales_frames || []).map(function (value) {
    return { value: String(value), label: String(value) + " frames" };
  });
  workspaceSetOptions(byId("analysisChangePointScale"), scales, workspaceState.changePointFilters.scale, "All scales");
  workspaceState.changePointFilters.method = byId("analysisChangePointMethod").value;
  workspaceState.changePointFilters.signal = byId("analysisChangePointSignal").value;
  workspaceState.changePointFilters.scale = byId("analysisChangePointScale").value;
  if (byId("analysisChangePointFeature")) byId("analysisChangePointFeature").value = workspaceState.changePointFilters.feature;
  if (byId("analysisChangePointThreshold")) byId("analysisChangePointThreshold").value = workspaceState.changePointFilters.threshold;
  if (byId("analysisChangePointOutcome")) byId("analysisChangePointOutcome").value = workspaceState.changePointFilters.outcome;
}

function workspaceDashboardRenderComparison(snapshot) {
  var cp = snapshot && snapshot.change_point;
  var badge = byId("analysisChangePointBadge");
  var status = byId("analysisChangePointStatus");
  if (!cp || !cp.available) {
    if (badge) {
      badge.className = "analysis-badge missing";
      badge.textContent = "Unavailable";
    }
    if (status) status.textContent = cp && cp.message ? cp.message : "No complete change-point snapshot is available.";
    ["analysisChangePointRecallChart", "analysisChangePointErrorChart", "analysisChangePointComparison"].forEach(function (id) {
      if (byId(id)) byId(id).innerHTML = workspaceEmpty("Change-point analysis unavailable");
    });
    return;
  }
  if (badge) {
    badge.className = "analysis-badge ok";
    badge.textContent = "Ready";
  }
  workspaceDashboardPopulateChangeOptions(snapshot);
  var filters = workspaceState.changePointFilters;
  if (status) status.textContent = "Q95 / level / 16f is the default. Current selection: "
    + (filters.threshold || "all") + " / " + (filters.feature || "all") + " / "
    + (filters.scale === "all" ? "all scales" : filters.scale + " frames")
    + " / " + (filters.outcome === "all" ? "all events" : filters.outcome.replace(/_/g, " ")) + ".";
  var source = cp.source || {};
  var freshness = cp.freshness || {};
  var provenance = byId("analysisChangePointProvenance");
  if (provenance) provenance.innerHTML = '<strong>Snapshot:</strong> ' + escapeHtml(source.directory || "unknown")
    + ' / <strong>generated:</strong> ' + escapeHtml(source.generated_at || "unknown")
    + ' / <strong>selected:</strong> ' + escapeHtml(String(source.selection_count == null ? "n/a" : source.selection_count))
    + ' / <strong>freshness:</strong> ' + escapeHtml(freshness.stale ? "stale" : "current")
    + ' / native samples only; descriptive diagnostic.';
  if (!workspaceState.analysisDashboardRows || !workspaceState.analysisDashboardRows.summary) {
    byId("analysisChangePointRecallChart").innerHTML = workspaceEmpty("Loading selected comparison rows...");
    byId("analysisChangePointErrorChart").innerHTML = workspaceEmpty("Loading selected comparison rows...");
    workspaceDashboardLoadComparisonRows(snapshot);
    return;
  }
  workspaceDashboardRenderComparisonCharts(workspaceDashboardChangeRows(snapshot));
}

function workspaceDashboardPopulateFailureOptions(snapshot) {
  var cp = snapshot && snapshot.change_point || {};
  var rows = cp.localization_by_failure_type || cp.by_failure_type || [];
  var signals = {};
  var tasks = {};
  rows.forEach(function (row) {
    if (row.signal != null) signals[String(row.signal)] = String(row.signal);
    if (row.task_id != null) tasks[String(row.task_id)] = labelFor(row.task_suite || "task") + " / task " + row.task_id;
  });
  workspaceSetOptions(byId("analysisLocalizationSignal"), Object.keys(signals).sort().map(function (value) {
    return { value: value, label: workspaceDashboardShortSignal(value) };
  }), workspaceState.localizationFilters.signal, "All native signals");
  workspaceSetOptions(byId("analysisLocalizationTask"), Object.keys(tasks).sort(function (a, b) {
    return Number(a) - Number(b);
  }).map(function (value) {
    return { value: value, label: tasks[value] };
  }), workspaceState.localizationFilters.task, "All tasks");
  workspaceState.localizationFilters.signal = byId("analysisLocalizationSignal").value;
  workspaceState.localizationFilters.task = byId("analysisLocalizationTask").value;
}

function workspaceDashboardRenderFailureTypes(snapshot) {
  var cp = snapshot && snapshot.change_point;
  var badge = byId("analysisLocalizationBadge");
  var status = byId("analysisLocalizationStatus");
  if (!cp || !cp.available) {
    if (badge) {
      badge.className = "analysis-badge missing";
      badge.textContent = "Unavailable";
    }
    if (status) status.textContent = "No change-point failure-type snapshot is available.";
    if (byId("analysisChangePointFailureType")) byId("analysisChangePointFailureType").innerHTML = workspaceEmpty("Failure-type analysis unavailable");
    return;
  }
  if (badge) {
    badge.className = "analysis-badge ok";
    badge.textContent = "Ready";
  }
  if (status) status.textContent = "Failure-type heatmap for " + (workspaceState.analysisFailureMetric || "recall").replace(/_/g, " ") + ". Values are descriptive and retain native signal identity.";
  workspaceDashboardPopulateFailureOptions(snapshot);
  var provenance = byId("analysisLocalizationProvenance");
  if (provenance) provenance.innerHTML = '<strong>Source:</strong> ' + escapeHtml((cp.source || {}).directory || "unknown")
    + ' / <strong>default:</strong> Q95 / level / 16f / <strong>coverage:</strong> '
    + escapeHtml(String((cp.counts || {}).rollouts || (cp.source || {}).selection_count || "n/a")) + " selected rollouts";
  if (!workspaceState.analysisDashboardRows || !workspaceState.analysisDashboardRows.failure) {
    byId("analysisChangePointFailureType").innerHTML = workspaceEmpty("Loading failure-type summary...");
    workspaceDashboardLoadFailureRows(snapshot);
    return;
  }
  workspaceDashboardRenderFailureHeatmap(workspaceDashboardFailureRows(snapshot));
}

function workspaceDashboardRenderConclusions(snapshot) {
  var host = byId("analysisTopConclusions");
  if (!host) return;
  if (!snapshot || !snapshot.available) {
    host.innerHTML = workspaceEmpty((snapshot && snapshot.message) || "No analysis snapshot is available.");
    return;
  }
  var live = snapshot.live || {};
  var cp = snapshot.change_point || {};
  var coverage = cp.method_coverage || snapshot.method_coverage || [];
  var available = coverage.map(function (row) {
    return (ANALYSIS_METHOD_LABELS[row.method] || row.method) + " " + (row.available_rollouts == null ? "n/a" : row.available_rollouts) + "/" + (row.selected_rollouts == null ? "n/a" : row.selected_rollouts);
  }).join(", ");
  var source = cp.source || snapshot.source || {};
  host.innerHTML = '<div class="analysis-conclusion-grid">'
    + '<div><strong>Scope</strong><span>Primary natural by default; ' + escapeHtml(String(live.rollouts == null ? source.selection_count || "n/a" : live.rollouts)) + ' live rollout(s).</span></div>'
    + '<div><strong>Resolved success</strong><span>' + escapeHtml(workspacePercent(live.resolved_success_rate)) + ' across ' + escapeHtml(String(live.resolved == null ? "n/a" : live.resolved)) + ' resolved rollout(s).</span></div>'
    + '<div><strong>Baseline coverage</strong><span>' + escapeHtml(available || "No coverage rows") + '.</span></div>'
    + '<div><strong>Interpretation</strong><span>Q95 / local level / 16f is the readable starting point. Signals remain independent; these are descriptive diagnostics, not held-out detector scores.</span></div>'
    + '</div>';
}

function workspaceDashboardRenderArchive(snapshot) {
  var linksHost = byId("analysisArchiveLinks");
  if (linksHost) {
    var links = snapshot && snapshot.artifact_links || [];
    linksHost.innerHTML = links.length
      ? '<div class="analysis-download-grid">' + links.map(function (link) {
        var artifactUrl = link.url || ("/api/analysis/artifacts/" + encodeURIComponent(link.name));
        return '<a class="analysis-download-link" href="' + escapeHtml(artifactUrl) + '" download title="' + escapeHtml(link.path || link.name) + '"><strong>' + escapeHtml(link.name) + '</strong><small>' + escapeHtml(link.source || "snapshot") + '</small></a>';
      }).join("") + '</div>'
      : workspaceEmpty("No downloadable artifacts are available.");
  }
  var badge = byId("analysisSnapshotBadge");
  var status = byId("analysisSnapshotStatus");
  var source = snapshot && (snapshot.source || {});
  if (badge) {
    badge.className = snapshot && snapshot.available ? "analysis-badge ok" : "analysis-badge missing";
    badge.textContent = snapshot && snapshot.available ? "Dashboard" : "Unavailable";
  }
  if (status) status.textContent = snapshot && snapshot.available
    ? "Legacy data are collapsed. Snapshot generated " + (source.generated_at || "unknown") + "; use Event explorer for paginated rows."
    : (snapshot && snapshot.message) || "No legacy snapshot is available.";
  if (byId("analysisProvenance")) byId("analysisProvenance").innerHTML = snapshot && snapshot.compatibility
    ? '<strong>Compatibility:</strong> ' + escapeHtml(snapshot.compatibility.full_endpoint || "/api/analysis?view=full") + ' / full event arrays are intentionally not loaded here.'
    : "";
  if (snapshot && snapshot.available) {
    var legacyRows = workspaceSnapshotEvents(snapshot);
    if (byId("analysisAnomalies")) {
      byId("analysisAnomalies").innerHTML = legacyRows.length
        ? workspaceRenderAnomalies(legacyRows)
        : workspaceEmpty("Open Event explorer to load legacy anomalies on demand.");
    }
    if (byId("analysisResponseChart")) byId("analysisResponseChart").innerHTML = workspaceRenderMetricChart(legacyRows, "normalized_response_magnitude", "Median response", "Legacy response", false);
    if (byId("analysisPersistenceChart")) byId("analysisPersistenceChart").innerHTML = workspaceRenderMetricChart(legacyRows, "post_event_persistence_fraction", "Persistence", "Legacy persistence", true);
    if (byId("analysisRecoveryChart")) byId("analysisRecoveryChart").innerHTML = workspaceRenderMetricChart(legacyRows, "recovery_fraction_toward_baseline", "Recovery", "Legacy recovery", true);
    if (byId("analysisThresholdChart")) byId("analysisThresholdChart").innerHTML = workspaceRenderThresholdChart(legacyRows, snapshot);
  }
}

function workspaceDashboardSelectSignalOptions(snapshot) {
  var eventTriggered = snapshot && snapshot.event_triggered;
  if (!eventTriggered || !eventTriggered.available) return;
  var filters = workspaceState.eventTriggeredFilters;
  var methods = (eventTriggered.methods || ANALYSIS_METHODS).filter(function (method) {
    return ANALYSIS_METHODS.indexOf(method) !== -1;
  });
  if (filters.method === "all" && methods.length) filters.method = methods[0];
  workspaceSetOptions(byId("analysisEventTriggeredMethod"), methods.map(function (method) {
    return { value: method, label: ANALYSIS_METHOD_LABELS[method] || method };
  }), filters.method, "All methods");
  filters.method = byId("analysisEventTriggeredMethod").value;
  var signals = (eventTriggered.signals_by_method || {})[filters.method] || [];
  if (!signals.length && eventTriggered.summary) {
    signals = eventTriggered.summary.filter(function (row) { return row.method === filters.method; }).map(function (row) { return row.signal; }).filter(Boolean);
  }
  signals = signals.filter(function (value, index, array) { return array.indexOf(value) === index; });
  if (filters.signal === "all" && signals.length) filters.signal = signals[0];
  workspaceSetOptions(byId("analysisEventTriggeredSignal"), signals.map(function (signal) {
    return { value: signal, label: workspaceDashboardShortSignal(signal) };
  }), filters.signal, "All native signals");
  filters.signal = byId("analysisEventTriggeredSignal").value;
  if (filters.event_group === "all") filters.event_group = "terminal_failure";
  if (byId("analysisEventTriggeredGroup")) byId("analysisEventTriggeredGroup").value = filters.event_group;
}

function workspaceDashboardCurveSvg(rows, fieldPrefix, title, method, signal, group) {
  var all = Array.isArray(rows) ? rows : [];
  var caseRows = all.filter(function (row) {
    return row.comparison_group === group && row.group === group
      && Number(row.relative_frame) >= -60 && Number(row.relative_frame) <= 60;
  }).sort(function (a, b) { return Number(a.relative_frame) - Number(b.relative_frame); });
  var controlRows = all.filter(function (row) {
    return row.comparison_group === group && row.group === "matched_clean_success"
      && Number(row.relative_frame) >= -60 && Number(row.relative_frame) <= 60;
  }).sort(function (a, b) { return Number(a.relative_frame) - Number(b.relative_frame); });
  if (!caseRows.length && !controlRows.length) return workspaceEmpty("No native curve points for this selection.");
  var valueField = fieldPrefix === "score" ? "score_median" : "normalized_median";
  var lowField = fieldPrefix === "score" ? "score_q25" : "normalized_q25";
  var highField = fieldPrefix === "score" ? "score_q75" : "normalized_q75";
  var allRows = caseRows.concat(controlRows);
  var values = [];
  allRows.forEach(function (row) {
    [row[lowField], row[valueField], row[highField]].forEach(function (value) {
      var number = workspaceDashboardNumber(value);
      if (number != null) values.push(number);
    });
  });
  if (!values.length) return workspaceEmpty("Curve has no numeric values.");
  var min = Math.min.apply(null, values);
  var max = Math.max.apply(null, values);
  if (min === max) { min -= 1; max += 1; }
  var left = 56, top = 22, width = 620, height = 170;
  var x = function (value) {
    var bounded = Math.max(-60, Math.min(60, Number(value)));
    return left + (bounded + 60) / 120 * width;
  };
  var y = function (value) { return top + (max - Number(value)) / (max - min) * height; };
  var validRows = function (source, low, high) {
    return source.filter(function (row) {
      return workspaceDashboardNumber(row.relative_frame) != null
        && workspaceDashboardNumber(row[valueField]) != null
        && (!low || workspaceDashboardNumber(row[low]) != null)
        && (!high || workspaceDashboardNumber(row[high]) != null);
    });
  };
  var line = function (source) {
    return validRows(source, null, null).map(function (row, index) {
      return (index ? "L" : "M") + x(row.relative_frame).toFixed(2) + "," + y(row[valueField]).toFixed(2);
    }).join(" ");
  };
  var clipId = ("analysis-curve-" + method + "-" + signal + "-" + group + "-" + fieldPrefix)
    .replace(/[^A-Za-z0-9_-]/g, "-");
  var band = function (source, color, label) {
    var valid = validRows(source, lowField, highField);
    if (valid.length < 2) return "";
    var upper = valid.map(function (row, index) {
      return (index ? "L" : "M") + x(row.relative_frame).toFixed(2) + "," + y(row[highField]).toFixed(2);
    }).join(" ");
    var lower = valid.slice().reverse().map(function (row) {
      return "L" + x(row.relative_frame).toFixed(2) + "," + y(row[lowField]).toFixed(2);
    }).join(" ");
    return '<path d="' + upper + " " + lower + ' Z" fill="' + color + '" opacity=".16" clip-path="url(#' + clipId + ')" aria-label="' + escapeHtml(label + " IQR") + '"><title>' + escapeHtml(label + " IQR") + '</title></path>';
  };
  var content = '<defs><clipPath id="' + clipId + '"><rect x="' + left + '" y="' + top + '" width="' + width + '" height="' + height + '"></rect></clipPath></defs>'
    + '<line class="chart-axis" x1="' + left + '" y1="' + (top + height) + '" x2="' + (left + width) + '" y2="' + (top + height) + '"></line>'
    + '<line class="chart-axis" x1="' + left + '" y1="' + top + '" x2="' + left + '" y2="' + (top + height) + '"></line>';
  [-60, -30, 0, 30, 60].forEach(function (relative) {
    var px = x(relative);
    content += '<line class="chart-grid" x1="' + px.toFixed(2) + '" y1="' + top + '" x2="' + px.toFixed(2) + '" y2="' + (top + height) + '"></line>'
      + '<text x="' + px.toFixed(2) + '" y="' + (top + height + 18) + '" text-anchor="middle">' + relative + '</text>';
  });
  content += '<text x="' + (left + width / 2) + '" y="' + (top + height + 37) + '" text-anchor="middle">relative frame</text>'
    + '<text x="13" y="' + (top + height / 2) + '" text-anchor="middle">value</text>';
  content += band(caseRows, "var(--terminal)", "selected event median");
  content += band(controlRows, "var(--muted)", "matched clean control");
  var casePath = line(caseRows);
  var controlPath = line(controlRows);
  if (casePath) content += '<path d="' + casePath + '" fill="none" stroke="var(--terminal)" stroke-width="2" clip-path="url(#' + clipId + ')" aria-label="event median"><title>selected event median</title></path>';
  if (controlPath) content += '<path d="' + controlPath + '" fill="none" stroke="var(--muted)" stroke-width="2" stroke-dasharray="5 4" clip-path="url(#' + clipId + ')" aria-label="clean control median"><title>matched clean control median</title></path>';
  [caseRows, controlRows].forEach(function (source, sourceIndex) {
    validRows(source, null, null).forEach(function (row) {
      var number = workspaceDashboardNumber(row[valueField]);
      content += '<circle cx="' + x(row.relative_frame).toFixed(2) + '" cy="' + y(number).toFixed(2) + '" r="1.8" fill="' + (sourceIndex ? "var(--muted)" : "var(--terminal)") + '" clip-path="url(#' + clipId + ')"></circle>';
    });
  });
  content += '<line x1="' + x(0).toFixed(2) + '" y1="' + top + '" x2="' + x(0).toFixed(2) + '" y2="' + (top + height) + '" stroke="var(--observable)" stroke-width="2" stroke-dasharray="4 3"></line>';
  return workspaceSvg(720, 255, title, content) + '<div class="analysis-legend"><span><i style="background:var(--terminal)"></i>selected event median + IQR</span><span><i style="background:var(--muted)"></i>matched clean control + IQR</span><span><i style="background:var(--observable)"></i>onset = 0</span></div>';
}
function workspaceDashboardRenderSignalSummary(snapshot, method, signal, group) {
  var host = byId("analysisEventTriggeredSummary");
  if (!host) return;
  var rows = (snapshot.event_triggered && snapshot.event_triggered.summary || []).filter(function (row) {
    return row.method === method && row.signal === signal
      && (row.event_group === group || row.event_group === "all");
  }).slice(0, 6);
  if (!rows.length) {
    host.innerHTML = workspaceEmpty("No aggregate separation summary for this selection.");
    return;
  }
  var html = '<table class="analysis-table analysis-summary-table" aria-label="Selected signal separation summary"><thead><tr><th>Representation</th><th>Scale</th><th>Strongest frame</th><th>Separation z</th><th>Events</th><th>Controls</th></tr></thead><tbody>';
  rows.forEach(function (row) {
    html += '<tr title="' + escapeHtml(workspaceDashboardMethodSignal(row) + " / " + row.event_group) + '"><th scope="row">' + escapeHtml(String(row.representation || "").replace(/_/g, " ")) + '</th>'
      + '<td class="numeric">' + escapeHtml(row.scale_frames == null ? "signal" : row.scale_frames + "f") + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.strongest_relative_frame, 0)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspaceFormatNumber(row.strongest_separation_z)) + '</td>'
      + '<td class="numeric">' + escapeHtml(String(row.n_events == null ? "n/a" : row.n_events)) + '</td>'
      + '<td class="numeric">' + escapeHtml(String(row.n_controls == null ? "n/a" : row.n_controls)) + '</td></tr>';
  });
  host.innerHTML = html + '</tbody></table>';
}

function workspaceDashboardRenderSignalShape(snapshot) {
  var eventTriggered = snapshot && snapshot.event_triggered;
  var badge = byId("analysisEventTriggeredBadge");
  var status = byId("analysisEventTriggeredStatus");
  if (!eventTriggered || !eventTriggered.available) {
    if (badge) {
      badge.className = "analysis-badge missing";
      badge.textContent = "Unavailable";
    }
    if (status) status.textContent = eventTriggered && eventTriggered.message
      ? eventTriggered.message
      : "No complete event-triggered snapshot is available.";
    ["analysisEventTriggeredSignalChart", "analysisEventTriggeredChangeChart", "analysisEventTriggeredSummary", "analysisEventTriggeredPeaks"].forEach(function (id) {
      if (byId(id)) byId(id).innerHTML = workspaceEmpty("Signal-shape analysis unavailable");
    });
    return;
  }
  if (badge) {
    badge.className = "analysis-badge ok";
    badge.textContent = "Ready";
  }
  workspaceDashboardSelectSignalOptions(snapshot);
  var filters = workspaceState.eventTriggeredFilters;
  var method = filters.method;
  var signal = filters.signal;
  var group = filters.event_group;
  if (!method || method === "all" || !signal || signal === "all") {
    if (status) status.textContent = "Choose one method and native signal to render a single onset-aligned curve.";
    return;
  }
  if (status) status.textContent = ANALYSIS_METHOD_LABELS[method] + " / " + signal + " / " + group + " / x axis is fixed at -60 to +60 relative frames.";
  var source = eventTriggered.source || {};
  if (byId("analysisEventTriggeredProvenance")) byId("analysisEventTriggeredProvenance").innerHTML = '<strong>Snapshot:</strong> ' + escapeHtml(source.directory || "unknown") + ' / <strong>generated:</strong> ' + escapeHtml(source.generated_at || "unknown") + ' / native points only.';
  var cache = workspaceState.analysisSignalRows;
  var cacheKey = method + "::" + signal;
  if (!cache || cache.key !== cacheKey) {
    var requestId = ++workspaceState.analysisSignalRequest;
    workspaceState.analysisSignalRows = { key: cacheKey, loading: true, curves: [], changes: [] };
    byId("analysisEventTriggeredSignalChart").innerHTML = workspaceEmpty("Loading native signal curve...");
    byId("analysisEventTriggeredChangeChart").innerHTML = workspaceEmpty("Loading native change-score curve...");
    var pending = 2;
    var finish = function (field, rows) {
      if (requestId !== workspaceState.analysisSignalRequest) return;
      workspaceState.analysisSignalRows[field] = rows;
      pending -= 1;
      if (!pending) {
        workspaceState.analysisSignalRows.loading = false;
        workspaceDashboardRenderSignalShape(snapshot);
      }
    };
    workspaceDashboardFetchRows("event_triggered_curves", { method: method, signal: signal }, function (rows) {
      finish("curves", rows);
    }, 1, [], requestId);
    workspaceDashboardFetchRows("event_triggered_change_scores", { method: method, signal: signal }, function (rows) {
      finish("changes", rows);
    }, 1, [], requestId);
    return;
  }
  var curveRows = cache.curves || [];
  var changeRows = cache.changes || [];
  byId("analysisEventTriggeredSignalChart").innerHTML = workspaceDashboardCurveSvg(
    curveRows, "signal", "Onset-aligned " + method + " " + signal + " signal", method, signal, group
  );
  byId("analysisEventTriggeredChangeChart").innerHTML = workspaceDashboardCurveSvg(
    changeRows, "score", "Onset-aligned " + method + " " + signal + " local change score", method, signal, group
  );
  workspaceDashboardRenderSignalSummary(snapshot, method, signal, group);
  var peaks = (eventTriggered.peak_events || []).filter(function (row) {
    return row.method === method && row.signal === signal && row.event_group === group;
  }).sort(function (left, right) {
    return (Number(right.peak_separation_z) || 0) - (Number(left.peak_separation_z) || 0);
  }).slice(0, 8);
  if (byId("analysisEventTriggeredPeaks")) {
    if (!peaks.length) {
      byId("analysisEventTriggeredPeaks").innerHTML = workspaceEmpty("Open Event explorer for paginated event-triggered peaks.");
    } else {
      var peakHtml = '<table class="analysis-table analysis-summary-table"><thead><tr><th>Failure type</th><th>Task</th><th>Rollout</th><th>Peak offset</th><th>Separation z</th><th></th></tr></thead><tbody>';
      peaks.forEach(function (row) {
        peakHtml += '<tr><td>' + escapeHtml(labelFor(row.failure_type || "other")) + '</td><td class="numeric">' + escapeHtml(row.task_id == null ? "n/a" : String(row.task_id)) + '</td><td class="numeric">' + escapeHtml(row.rollout_id || "unknown") + '</td><td class="numeric">' + escapeHtml(workspaceFormatNumber(row.peak_relative_frame, 0)) + '</td><td class="numeric">' + escapeHtml(workspaceFormatNumber(row.peak_separation_z)) + '</td><td><button type="button" data-analysis-rollout="' + escapeHtml(row.rollout_id || "") + '">Review</button></td></tr>';
      });
      byId("analysisEventTriggeredPeaks").innerHTML = peakHtml + '</tbody></table>';
    }
  }
}

function workspaceDashboardDetailsFilters() {
  var details = workspaceState.analysisDetails;
  var filters = Object.assign({}, details.filters);
  filters.method = byId("analysisDetailsMethod").value;
  filters.signal = byId("analysisDetailsSignal").value;
  filters.feature = byId("analysisDetailsFeature").value;
  filters.scale = byId("analysisDetailsScale").value;
  filters.threshold = byId("analysisDetailsThreshold").value;
  filters.outcome = byId("analysisDetailsOutcome").value;
  filters.failure_type = byId("analysisDetailsFailureType").value;
  filters.task = byId("analysisDetailsTask").value;
  details.filters = filters;
  return filters;
}

function workspaceDashboardPopulateDetailsFilters(snapshot) {
  var details = workspaceState.analysisDetails;
  var cp = snapshot && snapshot.change_point || {};
  var rows = [].concat(cp.summary || [], cp.by_failure_type || []);
  var methodOptions = ANALYSIS_METHODS.map(function (method) {
    return { value: method, label: ANALYSIS_METHOD_LABELS[method] || method };
  });
  workspaceSetOptions(byId("analysisDetailsMethod"), methodOptions, details.filters.method, "All methods");
  var signalSet = {};
  var featureSet = {};
  var scaleSet = {};
  var failureSet = {};
  rows.forEach(function (row) {
    if (row.signal != null) signalSet[String(row.signal)] = String(row.signal);
    if (row.feature != null) featureSet[String(row.feature)] = String(row.feature);
    if (row.scale_frames != null) scaleSet[String(row.scale_frames)] = String(row.scale_frames) + " frames";
    if (row.failure_type != null) failureSet[String(row.failure_type)] = labelFor(row.failure_type);
  });
  workspaceSetOptions(byId("analysisDetailsSignal"), Object.keys(signalSet).sort().map(function (value) {
    return { value: value, label: workspaceDashboardShortSignal(value) };
  }), details.filters.signal, "All native signals");
  workspaceSetOptions(byId("analysisDetailsFeature"), Object.keys(featureSet).sort().map(function (value) {
    return { value: value, label: value };
  }), details.filters.feature, "All features");
  workspaceSetOptions(byId("analysisDetailsScale"), Object.keys(scaleSet).sort(function (a, b) {
    return Number(a) - Number(b);
  }).map(function (value) { return { value: value, label: scaleSet[value] }; }), details.filters.scale, "All scales");
  workspaceSetOptions(byId("analysisDetailsFailureType"), Object.keys(failureSet).sort().map(function (value) {
    return { value: value, label: failureSet[value] };
  }), details.filters.failure_type, "All failure types");
  var taskOptions = {};
  (state.rollouts || []).forEach(function (record) {
    if (record.task_id != null) taskOptions[String(record.task_id)] = labelFor(record.task_suite || "task") + " / task " + record.task_id;
  });
  workspaceSetOptions(byId("analysisDetailsTask"), Object.keys(taskOptions).sort(function (a, b) {
    return Number(a) - Number(b);
  }).map(function (value) { return { value: value, label: taskOptions[value] }; }), details.filters.task, "All tasks");
}

function workspaceDashboardRenderDetails(payload) {
  var host = byId("analysisDetailsTable");
  if (!host) return;
  var details = workspaceState.analysisDetails;
  var count = byId("analysisDetailsCount");
  var status = byId("analysisDetailsStatus");
  if (!payload || payload.error) {
    host.innerHTML = workspaceEmpty((payload && payload.error) || "Analysis details unavailable.");
    if (status) status.textContent = payload && payload.error || "Analysis details unavailable.";
    return;
  }
  if (count) count.textContent = String(payload.total || 0) + " rows";
  if (status) status.textContent = payload.total
    ? "Showing " + ((payload.page - 1) * payload.page_size + 1) + "-" + Math.min(payload.page * payload.page_size, payload.total) + " of " + payload.total + " rows."
    : "No rows match the current filters.";
  if (!payload.items || !payload.items.length) {
    host.innerHTML = workspaceEmpty("No rows match the current filters.");
  } else {
    var html = '<table class="analysis-table analysis-details-table" aria-label="Paginated Analysis event details"><thead><tr><th>Method / signal</th><th>Feature / scale</th><th>Outcome</th><th>Failure type</th><th>Rollout</th><th>Frame</th><th>Score / error</th><th>Hit</th><th></th></tr></thead><tbody>';
    payload.items.forEach(function (row) {
      var score = row.event_score;
      if (score == null) score = row.onset_score;
      if (score == null) score = row.peak_score;
      if (score == null) score = row.absolute_localization_error_frames;
      var rollout = row.rollout_id || "unknown";
      var full = JSON.stringify(row);
      html += '<tr title="' + escapeHtml(full) + '"><th scope="row">' + escapeHtml(workspaceDashboardMethodSignal(row)) + '</th>'
        + '<td>' + escapeHtml((row.feature_label || row.feature || "n/a") + " / " + (row.scale_frames == null ? "n/a" : row.scale_frames + "f")) + '</td>'
        + '<td>' + escapeHtml(String(row.outcome_group || row.event_group || "n/a").replace(/_/g, " ")) + '</td>'
        + '<td>' + escapeHtml(labelFor(row.failure_type || "other")) + '</td>'
        + '<td class="numeric">' + escapeHtml(rollout) + '</td>'
        + '<td class="numeric">' + escapeHtml(row.event_frame == null ? (row.observable_onset_frame == null ? "n/a" : String(row.observable_onset_frame)) : String(row.event_frame)) + '</td>'
        + '<td class="numeric">' + escapeHtml(score == null ? "n/a" : workspaceFormatNumber(score)) + '</td>'
        + '<td>' + escapeHtml(row.hit == null ? (row.miss ? "miss" : "n/a") : row.hit ? "hit" : "miss") + '</td>'
        + '<td><button type="button" data-analysis-rollout="' + escapeHtml(rollout) + '">Review</button></td></tr>';
    });
    host.innerHTML = html + '</tbody></table>';
  }
  byId("analysisDetailsPage").textContent = "Page " + (payload.page || 1) + " / " + (payload.page_count || 1);
  byId("analysisDetailsPrevious").disabled = !payload.page || payload.page <= 1;
  byId("analysisDetailsNext").disabled = !payload.page_count || payload.page >= payload.page_count;
}

function workspaceLoadAnalysisDetails() {
  var details = workspaceState.analysisDetails;
  if (details.loading) return;
  details.loading = true;
  var requestId = ++details.request;
  var filters = workspaceDashboardDetailsFilters();
  var params = Object.assign({}, filters, {
    kind: details.kind,
    page: details.page,
    page_size: details.pageSize,
    sort: details.sort
  });
  var query = Object.keys(params).map(function (key) {
    return encodeURIComponent(key) + "=" + encodeURIComponent(params[key]);
  }).join("&");
  byId("analysisDetailsStatus").textContent = "Loading rows...";
  fetch("/api/analysis/details?" + query, { cache: "no-store" })
    .then(function (response) {
      return response.json().then(function (payload) {
        if (!response.ok) throw new Error(payload.error || "Could not load Analysis details");
        return payload;
      });
    })
    .then(function (payload) {
      if (requestId !== details.request) return;
      details.payload = payload;
      workspaceDashboardRenderDetails(payload);
    })
    .catch(function (error) {
      if (requestId === details.request) workspaceDashboardRenderDetails({ error: error.message });
    })
    .finally(function () {
      if (requestId === details.request) details.loading = false;
    });
}

function workspaceRenderSnapshot() {
  return workspaceDashboardRenderSnapshot();
}

function workspaceDashboardRenderSnapshot() {
  var snapshot = workspaceState.analysisSnapshot;
  var status = byId("analysisStatus");
  if (!snapshot || !snapshot.available) {
    if (status) {
      status.className = "analysis-status warning";
      status.textContent = snapshot && snapshot.message || (workspaceState.analysisLoading ? "Loading analysis..." : "No complete analysis snapshot is available.");
    }
    workspaceDashboardRenderConclusions(snapshot);
    workspaceDashboardRenderComparison(snapshot);
    workspaceDashboardRenderFailureTypes(snapshot);
    workspaceDashboardRenderSignalShape(snapshot);
    workspaceDashboardRenderArchive(snapshot);
    return;
  }
  if (status) {
    status.className = "analysis-status";
    status.textContent = "Dashboard loaded for primary natural by default. Live annotations update independently; snapshot data are read-only.";
  }
  workspaceDashboardRenderConclusions(snapshot);
  var cp = snapshot.change_point || {};
  if (byId("analysisCoverageChart")) byId("analysisCoverageChart").innerHTML = workspaceRenderCoverageChart(snapshot);
  if (byId("analysisOverviewProvenance")) {
    var source = cp.source || snapshot.source || {};
    byId("analysisOverviewProvenance").innerHTML = '<strong>Snapshot:</strong> ' + escapeHtml(source.directory || "unknown") + ' / generated ' + escapeHtml(source.generated_at || "unknown") + ' / switch to Method comparison or Event explorer for details.';
  }
  var route = workspaceParseRoute();
  workspaceDashboardRenderArchive(snapshot);
  if (route.analysisTab === "comparison") workspaceDashboardRenderComparison(snapshot);
  if (route.analysisTab === "failures") workspaceDashboardRenderFailureTypes(snapshot);
  if (route.analysisTab === "signals") workspaceDashboardRenderSignalShape(snapshot);
  if (route.analysisTab === "events") {
    workspaceDashboardPopulateDetailsFilters(snapshot);
    workspaceLoadAnalysisDetails();
  }
}

function workspaceRenderTaskChart(records) {
  var groups = {};
  (records || []).forEach(function (record) {
    var key = String(record.task_suite || "unknown") + "::" + String(record.task_id);
    if (!groups[key]) groups[key] = { suite: record.task_suite || "unknown", task: String(record.task_id), counts: { clean_success: 0, recovered_success: 0, terminal_failure: 0, uncertain: 0 } };
    groups[key].counts[workspaceNormalizeOutcome(effectiveOutcome(record))] += 1;
  });
  var entries = Object.keys(groups).map(function (key) {
    var entry = groups[key];
    var resolved = entry.counts.clean_success + entry.counts.recovered_success + entry.counts.terminal_failure;
    entry.rate = resolved ? (entry.counts.clean_success + entry.counts.recovered_success) / resolved : null;
    entry.total = Object.keys(entry.counts).reduce(function (sum, field) { return sum + entry.counts[field]; }, 0);
    return entry;
  }).sort(function (a, b) { return a.suite.localeCompare(b.suite) || Number(a.task) - Number(b.task); });
  if (!entries.length) return workspaceEmpty("No task rows in the selected live scope.");
  var html = '<div class="analysis-task-list" role="list" aria-label="Outcome by task">';
  entries.forEach(function (entry) {
    var label = labelFor(entry.suite) + " / task " + entry.task;
    html += '<button type="button" class="analysis-task-row" data-analysis-task="' + escapeHtml(entry.task) + '" data-analysis-suite="' + escapeHtml(entry.suite) + '" title="' + escapeHtml(label + "; click to filter") + '"><span class="analysis-task-label"><strong>' + escapeHtml(label) + '</strong><small>resolved success ' + escapeHtml(workspacePercent(entry.rate)) + ' / n=' + entry.total + '</small></span><span class="analysis-task-counts">';
    ANALYSIS_OUTCOMES.forEach(function (definition) {
      html += '<span class="analysis-task-count" style="--task-color:' + definition.color + '"><i></i>' + escapeHtml(definition.label) + ': ' + entry.counts[definition.value] + '</span>';
    });
    html += '</span></button>';
  });
  return html + '</div>';
}

function workspaceParseRoute() {
  var hash = window.location.hash || "#/review";
  var raw = hash.replace(/^#\/?/, "");
  var parts = raw.split("/");
  var view = ["review", "analysis", "settings"].indexOf(parts[0]) === -1 ? "review" : parts[0];
  var analysisTabs = ["overview", "comparison", "failures", "events", "signals", "archive"];
  var analysisTab = view === "analysis" && analysisTabs.indexOf(parts[1]) !== -1 ? parts[1] : "overview";
  var id = view === "review" && parts.length > 1 && parts[1]
    ? decodeURIComponent(parts.slice(1).join("/")) : null;
  return { view: view, id: id, analysisTab: analysisTab, hash: hash };
}

function workspaceRenderAnalysisTabs(tab) {
  var allowed = ["overview", "comparison", "failures", "events", "signals", "archive"];
  if (allowed.indexOf(tab) === -1) tab = "overview";
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
    if (!window.confirm("Discard unsaved changes and leave this page?")) {
      window.history.replaceState(null, "", workspaceState.lastHash);
      return;
    }
    state.dirty = false;
    workspaceState.settingsDirty = false;
  }
  workspaceState.lastHash = route.hash;
  workspaceState.view = route.view;
  state.view = route.view;
  workspaceShowView(route.view);
  if (route.view === "review") {
    if (route.id && (state.rollouts || []).some(function (record) { return record.id === route.id; })) {
      selectRollout(route.id);
    } else if (!selectedRollout() && state.filtered && state.filtered.length) {
      selectRollout(state.filtered[0].id);
    }
  } else if (route.view === "analysis") {
    workspaceRenderAnalysisTabs(route.analysisTab);
    workspaceRenderLiveAnalysis();
    workspaceRenderAnalysisRunPanel();
    workspaceLoadAnalysisEnvironment();
    workspaceLoadAnalysis(false);
    workspaceLoadBaselineRuns(workspaceState.analysisRunScope || "primary_natural");
  } else if (route.view === "settings") {
    workspaceLoadSettings();
  }
}

function workspaceDataChanged() {
  if (workspaceState.view === "review") {
    var route = workspaceParseRoute();
    if (route.id && (state.rollouts || []).some(function (record) { return record.id === route.id; }) && state.selectedId !== route.id) {
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
