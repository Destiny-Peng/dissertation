"use strict";

/* Analysis snapshot module. Runtime state is owned by workspace-core.js. */

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
