"use strict";

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

function workspaceDashboardRenderRolloutOutcome(snapshot) {
  var host = byId("analysisRolloutOutcomeTable");
  var badge = byId("analysisRolloutOutcomeBadge");
  var provenance = byId("analysisRolloutOutcomeProvenance");
  if (!host) return;
  var thresholdSelect = byId("analysisOutcomeThreshold");
  var threshold = thresholdSelect ? thresholdSelect.value : "q95";
  var rows = Array.isArray(snapshot && snapshot.rollout_outcome_summary)
    ? snapshot.rollout_outcome_summary.filter(function (row) {
        return row.method !== "safe" && String(row.threshold || "") === threshold;
      })
    : [];
  if (!rows.length) {
    host.innerHTML = workspaceEmpty(
      "No final rollout-outcome statistics are available in the current analysis snapshot."
    );
    if (badge) {
      badge.className = "analysis-badge";
      badge.textContent = "Unavailable";
    }
    if (provenance) provenance.textContent = "";
    return;
  }

  rows.sort(function (left, right) {
    return ANALYSIS_METHODS.indexOf(left.method) - ANALYSIS_METHODS.indexOf(right.method);
  });

  function successPositiveMetrics(row) {
    var alreadySuccessPositive = String(row.positive_class || "").indexOf("success") !== -1;
    var tp = Number(row.tp), fn = Number(row.fn), fp = Number(row.fp), tn = Number(row.tn);
    if (!alreadySuccessPositive && [tp, fn, fp, tn].every(Number.isFinite)) {
      var oldTp = tp, oldFn = fn, oldFp = fp, oldTn = tn;
      tp = oldTn;
      fn = oldFp;
      fp = oldFn;
      tn = oldTp;
    }
    function rate(a, b) {
      return Number.isFinite(a) && Number.isFinite(b) && b > 0 ? a / b : null;
    }
    var successRecall = row.success_recall;
    var failureRecall = row.failure_recall;
    var precision = row.precision;
    var f1 = row.f1;
    if (!alreadySuccessPositive) {
      successRecall = rate(tp, tp + fn);
      failureRecall = rate(tn, tn + fp);
      precision = rate(tp, tp + fp);
      f1 = precision != null && successRecall != null && precision + successRecall > 0
        ? 2 * precision * successRecall / (precision + successRecall) : null;
    }
    return {
      tp: tp, fn: fn, fp: fp, tn: tn,
      accuracy: row.accuracy,
      successRecall: successRecall == null ? row.recall : successRecall,
      failureRecall: failureRecall == null ? row.specificity : failureRecall,
      precision: precision,
      f1: f1
    };
  }

  var coverageRows = (
    snapshot
    && snapshot.rollout_outcome_snapshot
    && Array.isArray(snapshot.rollout_outcome_snapshot.method_coverage)
  ) ? snapshot.rollout_outcome_snapshot.method_coverage : [];
  var coverageByMethod = {};
  coverageRows.forEach(function (row) { coverageByMethod[row.method] = row; });
  var html = '<table class="analysis-table analysis-summary-table" aria-label="Final rollout outcome classification">'
    + '<thead><tr><th>Method / signal</th><th>N</th><th>Coverage</th><th>Accuracy</th><th>Success recall</th>'
    + '<th>Failure recall</th><th>Precision</th><th>F1</th></tr></thead><tbody>';
  rows.forEach(function (row) {
    var metrics = successPositiveMetrics(row);
    var method = (ANALYSIS_METHOD_LABELS[row.method] || row.method || "method")
      + " / " + workspaceDashboardShortSignal(row.signal);
    var rule = row.success_rule || (row.failure_rule ? "success = NOT(" + row.failure_rule + ")" : threshold.toUpperCase());
    var rowTitle = (row.signal_note || "")
      + " | TP/FN=" + metrics.tp + "/" + metrics.fn
      + " | FP/TN=" + metrics.fp + "/" + metrics.tn
      + " | " + rule;
    html += '<tr title="' + escapeHtml(rowTitle) + '"><th scope="row">' + escapeHtml(method) + '</th>'
      + '<td class="numeric">' + escapeHtml(String(row.n_resolved == null ? "n/a" : row.n_resolved)) + '</td>'
      + '<td class="numeric">' + (function () {
          var coverage = coverageByMethod[row.method] || {};
          var available = Number(coverage.available_rollouts);
          var population = Number(coverage.evaluation_population);
          if (!Number.isFinite(available) || !Number.isFinite(population) || population <= 0) return "n/a";
          return escapeHtml(String(available) + "/" + String(population));
        })() + '</td>'
      + '<td class="numeric">' + escapeHtml(workspacePercent(metrics.accuracy)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspacePercent(metrics.successRecall)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspacePercent(metrics.failureRecall)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspacePercent(metrics.precision)) + '</td>'
      + '<td class="numeric">' + escapeHtml(workspacePercent(metrics.f1)) + '</td></tr>';
  });
  host.innerHTML = html + '</tbody></table>';
  if (badge) {
    badge.className = "analysis-badge ok";
    badge.textContent = threshold.toUpperCase() + " · " + rows.length + " methods";
  }
  if (provenance) {
    var info = snapshot.rollout_outcome || {};
    var outcomeSnapshot = snapshot.rollout_outcome_snapshot || {};
    var outcomeSource = outcomeSnapshot.source || {};
    provenance.innerHTML = '<strong>Positive:</strong> clean + recovered success'
      + ' / <strong>negative:</strong> terminal failure'
      + (outcomeSource.directory ? ' / <strong>snapshot:</strong> ' + escapeHtml(outcomeSource.directory) : '')
      + ' / <strong>threshold:</strong> ' + escapeHtml(threshold.toUpperCase()) + ' calibrated from same-cohort final-success terminal scores'
      + ' / <strong>uncertain:</strong> excluded from metrics'
      + ' / <strong>methods:</strong> ProcVLM, RynnValue, Robo-Dopamine'
      + ' / <strong>interpretation:</strong> descriptive same-cohort calibration, not held-out accuracy'
      + (info.prediction_rows == null ? "" : ' / <strong>prediction rows:</strong> ' + escapeHtml(String(info.prediction_rows)));
  }
}


function workspaceDashboardOutcomeSweepThreshold(row) {
  var value = row.raw_progress_threshold;
  if (value == null) value = row.raw_terminal_threshold;
  return Number(value);
}

function workspaceDashboardOutcomeSweepChart(rows, title) {
  var width = 720;
  var height = 260;
  var margin = { top: 18, right: 18, bottom: 38, left: 48 };
  var plotWidth = width - margin.left - margin.right;
  var plotHeight = height - margin.top - margin.bottom;
  var metrics = [
    { key: "accuracy", label: "Accuracy", className: "accuracy" },
    { key: "success_recall", label: "Success recall", className: "success-recall" },
    { key: "failure_recall", label: "Failure recall", className: "failure-recall" },
    { key: "f1", label: "F1", className: "f1" }
  ];
  var ordered = rows.slice().sort(function (left, right) {
    return workspaceDashboardOutcomeSweepThreshold(left)
      - workspaceDashboardOutcomeSweepThreshold(right);
  });
  var thresholds = ordered.map(workspaceDashboardOutcomeSweepThreshold)
    .filter(Number.isFinite);
  if (!thresholds.length) {
    return '<div class="analysis-sweep-card">' + workspaceEmpty("No usable threshold points.") + '</div>';
  }
  var xMin = Math.min.apply(null, thresholds);
  var xMax = Math.max.apply(null, thresholds);

  function x(value) {
    if (xMax === xMin) return margin.left + plotWidth / 2;
    return margin.left + ((value - xMin) / (xMax - xMin)) * plotWidth;
  }

  function y(value) {
    return margin.top + (1 - Math.max(0, Math.min(1, value))) * plotHeight;
  }

  function metricPath(metric) {
    var points = ordered.map(function (row) {
      var threshold = workspaceDashboardOutcomeSweepThreshold(row);
      var value = Number(row[metric]);
      if (!Number.isFinite(threshold) || !Number.isFinite(value)) return null;
      return [x(threshold), y(value)];
    }).filter(Boolean);
    return points.map(function (point, index) {
      return (index ? "L" : "M") + point[0].toFixed(2) + "," + point[1].toFixed(2);
    }).join(" ");
  }

  var n = ordered.length && ordered[0].n_resolved != null
    ? String(ordered[0].n_resolved)
    : "n/a";
  var html = '<section class="analysis-sweep-card">'
    + '<div class="analysis-sweep-heading"><strong>' + escapeHtml(title) + '</strong>'
    + '<span>N=' + escapeHtml(n) + '</span></div>'
    + '<div class="analysis-chart">'
    + '<svg class="analysis-svg analysis-sweep-svg" viewBox="0 0 ' + width + ' ' + height
    + '" role="img" aria-label="' + escapeHtml(title + " threshold sweep") + '">';

  [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].forEach(function (tick, index) {
    var py = y(tick);
    var major = index % 2 === 0;
    html += '<line class="analysis-sweep-gridline analysis-sweep-gridline-horizontal'
      + (major ? ' major' : ' minor') + '" x1="' + margin.left + '" y1="' + py.toFixed(2)
      + '" x2="' + (margin.left + plotWidth) + '" y2="' + py.toFixed(2) + '"></line>';
    if (major) {
      html += '<text class="analysis-sweep-axis-label" x="' + (margin.left - 8) + '" y="' + (py + 4).toFixed(2)
        + '" text-anchor="end">' + Math.round(tick * 100) + '%</text>';
    }
  });

  thresholds.forEach(function (threshold) {
    var px = x(threshold);
    html += '<line class="analysis-sweep-gridline analysis-sweep-gridline-vertical" x1="'
      + px.toFixed(2) + '" y1="' + margin.top + '" x2="' + px.toFixed(2) + '" y2="'
      + (margin.top + plotHeight) + '"></line>'
      + '<line class="analysis-sweep-tick" x1="' + px.toFixed(2) + '" y1="'
      + (margin.top + plotHeight) + '" x2="' + px.toFixed(2) + '" y2="'
      + (margin.top + plotHeight + 4) + '"></line>'
      + '<text class="analysis-sweep-axis-label" x="' + px.toFixed(2) + '" y="'
      + (margin.top + plotHeight + 20) + '" text-anchor="middle">'
      + escapeHtml(threshold.toFixed(2)) + '</text>';
  });

  html += '<line class="analysis-sweep-axis" x1="' + margin.left + '" y1="' + margin.top
    + '" x2="' + margin.left + '" y2="' + (margin.top + plotHeight) + '"></line>'
    + '<line class="analysis-sweep-axis" x1="' + margin.left + '" y1="' + (margin.top + plotHeight)
    + '" x2="' + (margin.left + plotWidth) + '" y2="' + (margin.top + plotHeight) + '"></line>';

  metrics.forEach(function (metric) {
    var path = metricPath(metric.key);
    if (!path) return;
    html += '<path class="analysis-sweep-line ' + metric.className + '" d="' + path + '"></path>';
    ordered.forEach(function (row) {
      var threshold = workspaceDashboardOutcomeSweepThreshold(row);
      var value = Number(row[metric.key]);
      if (!Number.isFinite(threshold) || !Number.isFinite(value)) return;
      html += '<circle class="analysis-sweep-point ' + metric.className + '" cx="'
        + x(threshold).toFixed(2) + '" cy="' + y(value).toFixed(2) + '" r="4" tabindex="0"'
        + ' data-sweep-threshold="' + escapeHtml(threshold.toFixed(4)) + '"'
        + ' data-sweep-value="' + escapeHtml(value.toFixed(8)) + '"'
        + ' data-sweep-metric="' + escapeHtml(metric.label) + '"'
        + ' aria-label="' + escapeHtml(
          metric.label + ", threshold " + threshold.toFixed(2) + ", value " + workspacePercent(value)
        ) + '">'
        + '<title>' + escapeHtml(
          metric.label + " · threshold " + threshold.toFixed(2) + " · " + workspacePercent(value)
        ) + '</title></circle>';
    });
  });
  html += '</svg></div><div class="analysis-sweep-legend">';
  metrics.forEach(function (metric) {
    html += '<span><i class="analysis-sweep-swatch ' + metric.className + '"></i>'
      + escapeHtml(metric.label) + '</span>';
  });
  return html
    + '</div>'
    + '<div class="analysis-sweep-tooltip" role="status" aria-live="polite" hidden>'
    + '<strong data-sweep-tooltip-metric>Metric</strong>'
    + '<span>x · threshold <b data-sweep-tooltip-threshold>n/a</b></span>'
    + '<span>y · value <b data-sweep-tooltip-value>n/a</b></span>'
    + '</div></section>';
}

function workspaceDashboardInstallOutcomeSweepTooltips(host) {
  if (!host) return;
  host.querySelectorAll(".analysis-sweep-card").forEach(function (card) {
    var tooltip = card.querySelector(".analysis-sweep-tooltip");
    if (!tooltip) return;

    function hide(point) {
      if (point) point.classList.remove("is-active");
      tooltip.hidden = true;
    }

    function show(point, clientX, clientY) {
      card.querySelectorAll(".analysis-sweep-point.is-active").forEach(function (active) {
        if (active !== point) active.classList.remove("is-active");
      });
      point.classList.add("is-active");

      var threshold = Number(point.dataset.sweepThreshold);
      var value = Number(point.dataset.sweepValue);
      var metric = point.dataset.sweepMetric || "Metric";
      var thresholdNode = tooltip.querySelector("[data-sweep-tooltip-threshold]");
      var valueNode = tooltip.querySelector("[data-sweep-tooltip-value]");
      var metricNode = tooltip.querySelector("[data-sweep-tooltip-metric]");
      if (thresholdNode) {
        thresholdNode.textContent = Number.isFinite(threshold)
          ? threshold.toFixed(2)
          : "n/a";
      }
      if (valueNode) {
        valueNode.textContent = Number.isFinite(value)
          ? workspacePercent(value)
          : "n/a";
      }
      if (metricNode) metricNode.textContent = metric;

      tooltip.hidden = false;
      var cardRect = card.getBoundingClientRect();
      var pointRect = point.getBoundingClientRect();
      var left = Number.isFinite(clientX)
        ? clientX - cardRect.left + 12
        : pointRect.left - cardRect.left + pointRect.width / 2 + 10;
      var top = Number.isFinite(clientY)
        ? clientY - cardRect.top + 12
        : pointRect.top - cardRect.top - 10;

      tooltip.style.left = left + "px";
      tooltip.style.top = top + "px";

      var tooltipRect = tooltip.getBoundingClientRect();
      var padding = 8;
      var maxLeft = Math.max(padding, card.clientWidth - tooltipRect.width - padding);
      var maxTop = Math.max(padding, card.clientHeight - tooltipRect.height - padding);
      tooltip.style.left = Math.max(padding, Math.min(left, maxLeft)) + "px";
      tooltip.style.top = Math.max(padding, Math.min(top, maxTop)) + "px";
    }

    card.querySelectorAll(".analysis-sweep-point").forEach(function (point) {
      point.addEventListener("mouseenter", function (event) {
        show(point, event.clientX, event.clientY);
      });
      point.addEventListener("mousemove", function (event) {
        show(point, event.clientX, event.clientY);
      });
      point.addEventListener("mouseleave", function () {
        hide(point);
      });
      point.addEventListener("focus", function () {
        show(point, NaN, NaN);
      });
      point.addEventListener("blur", function () {
        hide(point);
      });
    });
  });
}

function workspaceDashboardRenderOutcomeThresholdSweep(snapshot) {
  var host = byId("analysisOutcomeThresholdSweep");
  if (!host) return;
  var outcomeSnapshot = snapshot && snapshot.rollout_outcome_snapshot;
  var rows = outcomeSnapshot && Array.isArray(outcomeSnapshot.threshold_sweep)
    ? outcomeSnapshot.threshold_sweep
    : [];
  rows = rows.filter(function (row) {
    return row.method === "procvlm" || row.method === "robo_dopamine";
  });
  if (!rows.length) {
    host.innerHTML = workspaceEmpty(
      "No progress-threshold sweep is available in this snapshot. Re-run Outcome Evaluation to generate it."
    );
    return;
  }

  var groups = {};
  rows.forEach(function (row) {
    var aggregation = String(row.score_aggregation || "final");
    var key = String(row.method) + "::" + aggregation;
    if (!groups[key]) groups[key] = [];
    groups[key].push(row);
  });

  var order = [
    ["procvlm", "final"],
    ["procvlm", "maximum"],
    ["robo_dopamine", "final"],
    ["robo_dopamine", "maximum"]
  ];
  var html = '<div class="analysis-sweep-grid">';
  order.forEach(function (pair) {
    var key = pair[0] + "::" + pair[1];
    if (!groups[key] || !groups[key].length) return;
    var method = ANALYSIS_METHOD_LABELS[pair[0]] || pair[0];
    var aggregation = pair[1] === "maximum" ? "Maximum progress" : "Final progress";
    html += workspaceDashboardOutcomeSweepChart(
      groups[key],
      method + " · " + aggregation
    );
  });
  html += '</div>';
  if (!groups["procvlm::maximum"] && !groups["robo_dopamine::maximum"]) {
    html += '<p class="analysis-chart-caption">This is an older snapshot with final-progress rows only. '
      + 'Re-run Outcome Evaluation to generate maximum-progress curves.</p>';
  } else {
    html += '<p class="analysis-chart-caption">Each panel applies the same raw progress thresholds (0.50–0.95). '
      + 'Maximum progress uses the highest native progress value observed anywhere in the rollout; final progress uses the last native sample.</p>';
  }
  host.innerHTML = html;
  workspaceDashboardInstallOutcomeSweepTooltips(host);
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
    + '<div><strong>Scope</strong><span>LIBERO-10 by default; ' + escapeHtml(String(live.rollouts == null ? source.selection_count || "n/a" : live.rollouts)) + ' live rollout(s).</span></div>'
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

function workspaceRenderSnapshot() {
  return workspaceDashboardRenderSnapshot();
}

function workspaceDashboardRenderSnapshot() {
  var snapshot = workspaceState.analysisSnapshot;
  var status = byId("analysisStatus");
  var route = workspaceParseRoute();
  if (!snapshot || !snapshot.available) {
    if (status) {
      status.className = "analysis-status warning";
      status.textContent = snapshot && snapshot.message
        || (workspaceState.analysisLoading ? "Loading analysis..." : "No complete analysis snapshot is available.");
    }
    if (route.analysisTab === "outcome") {
      workspaceDashboardRenderRolloutOutcome(snapshot);
      workspaceDashboardRenderOutcomeThresholdSweep(snapshot);
    } else if (route.analysisTab === "localization") {
      workspaceDashboardRenderComparison(snapshot);
      workspaceDashboardRenderFailureTypes(snapshot);
    }
    return;
  }
  if (status) {
    status.className = "analysis-status";
    status.textContent = "Analysis results loaded. Outcome evaluation and failure localization are kept as separate research questions.";
  }
  if (route.analysisTab === "outcome") {
    workspaceDashboardRenderRolloutOutcome(snapshot);
    workspaceDashboardRenderOutcomeThresholdSweep(snapshot);
    return;
  }
  if (route.analysisTab === "localization") {
    workspaceDashboardRenderComparison(snapshot);
    workspaceDashboardRenderFailureTypes(snapshot);
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
