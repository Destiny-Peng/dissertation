"use strict";

/* Analysis event-triggered module. Runtime state is owned by workspace-core.js. */

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
