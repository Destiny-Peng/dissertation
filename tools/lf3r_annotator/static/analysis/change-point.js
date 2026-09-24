"use strict";

/* Analysis change-point module. Runtime state is owned by workspace-core.js. */

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
