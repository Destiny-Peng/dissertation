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
    status.textContent = "Dashboard loaded for LIBERO-10 by default. Live annotations update independently; snapshot data are read-only.";
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
