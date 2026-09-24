"use strict";

/* Analysis localization module. Runtime state is owned by workspace-core.js. */

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
