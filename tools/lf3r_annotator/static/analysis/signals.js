"use strict";

/* Analysis dashboard signal-curve rendering. */

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
