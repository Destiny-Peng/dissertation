"use strict";

/* Analysis dashboard paginated event details. */

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
