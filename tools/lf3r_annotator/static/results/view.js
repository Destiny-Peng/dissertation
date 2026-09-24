"use strict";

/* Results cards, localization summaries, and current-frame rendering. */

function renderPosthocLocalizationControls(method, result) {
  if (method !== "robo_dopamine" || !result || !result.available) return "";
  var history = Array.isArray(result.posthoc_localizations)
    ? result.posthoc_localizations : [];
  var active = result.localization_prediction || null;
  var selectedCheckpoint = state.roboPosthocCheckpoint
    || (active && active.checkpoint) || "";
  var historyHtml = "";
  if (history.length) {
    historyHtml = '<label><span>Saved prediction</span><select data-posthoc-localization-select>'
      + history.map(function (row, index) {
        var frame = Number(row.predicted_frame);
        var checkpoint = String(row.checkpoint || "");
        var label = "frame " + (Number.isFinite(frame) ? Math.round(frame) : "?")
          + " · " + (row.checkpoint_config_id || checkpoint.split("/").slice(-2).join("/"));
        var activeMatch = active
          && String(active.checkpoint_sha256 || active.checkpoint || "")
            === String(row.checkpoint_sha256 || row.checkpoint || "");
        return '<option value="' + index + '"' + (activeMatch ? " selected" : "")
          + '>' + escapeHtml(label) + '</option>';
      }).join("")
      + '</select></label>';
  }
  return '<div class="evaluation-posthoc-localization">'
    + '<div class="evaluation-posthoc-heading"><strong>Post-hoc localization</strong>'
    + '<span>No Robo-Dopamine rerun; uses saved fused progress + hop.</span></div>'
    + '<div class="evaluation-posthoc-controls">'
    + '<label class="evaluation-posthoc-checkpoint"><span>Localization checkpoint</span>'
    + '<input type="text" data-posthoc-localization-ckpt value="' + escapeHtml(selectedCheckpoint)
    + '" placeholder="outputs/robo_localization/.../repeat_XX.pt"></label>'
    + historyHtml
    + '<button type="button" class="ghost-button" data-run-posthoc-localization="current">Current rollout</button>'
    + '<button type="button" class="ghost-button" data-run-posthoc-localization="all">All in this run</button>'
    + '</div><div class="evaluation-meta" data-posthoc-localization-status></div></div>';
}

function baselineCardCollapsed(method) {
  return Boolean(state.baselineCollapsed && state.baselineCollapsed[method]);
}

function persistBaselineCollapsed() {
  try {
    sessionStorage.setItem(
      "lf3r.results.baselineCollapsed",
      JSON.stringify(state.baselineCollapsed || {})
    );
  } catch (_error) {}
}

function renderEvaluationHistory(result) {
  var samples = result.samples || [];
  if (!samples.length) return '<div class="evaluation-empty">No per-frame output history.</div>';
  var visible = samples.slice(0, 60);
  var rows = visible.map(function (sample) {
    return '<div class="evaluation-history-row"><strong>f' + escapeHtml(sample.frame) + '</strong> '
      + escapeHtml(compactSampleOutput(sample)) + "</div>";
  }).join("");
  var suffix = samples.length > visible.length ? '<div class="evaluation-empty">Showing first ' + visible.length + " of " + samples.length + " samples.</div>" : "";
  return '<details class="evaluation-history"><summary>Output history (' + samples.length + " samples)</summary>"
    + '<div class="evaluation-history-list">' + rows + suffix + "</div></details>";
}

function renderEvaluationCard(method, result, record) {
  var available = Boolean(result.available);
  var viewingCondition = state.instructionCondition || "full_instruction";
  var validation = result.validation || {};
  var status = validation.status || (available ? "ok" : "missing");
  var variant = currentInstructionVariant(record);
  var canRunCondition = viewingCondition === "full_instruction" || Boolean(variant.available);
  var action = canRunCondition
    ? '<button class="ghost-button baseline-run-button" type="button" data-run-baseline="' + escapeHtml(method)
      + '" title="Configure parameters and run this baseline on the current rollout">Configure &amp; run</button>'
    : '<span class="evaluation-meta">Condition view only</span>';
  var body = renderBaselineRunControls(method, result, record);
  if (available) {
    body += renderPosthocLocalizationControls(method, result)
      + renderLocalizationPredictionSummary(method, result)
      + LF3RResultsCharts.renderLocalizationPredictionCurve(method, result, record)
      + LF3RResultsCharts.renderSignalChart(method, result, record)
      + '<div class="evaluation-current">'
      + '<div class="evaluation-current-body">'
      + '<pre class="evaluation-output" data-current-output>No output at this frame.</pre></div></div>'
      + renderEvaluationHistory(result);
  } else {
    body += '<div class="evaluation-empty">' + escapeHtml(validation.message || "No usable output for this rollout.") + "</div>";
  }
  if (validation.message && available && status !== "ok") {
    body += '<div class="evaluation-meta">' + escapeHtml(validation.message) + "</div>";
  }
  var collapsed = baselineCardCollapsed(method);
  return '<article class="evaluation-card ' + (available ? "available" : "unavailable")
    + (collapsed ? ' is-collapsed' : '') + '" data-evaluation-method="' + escapeHtml(method) + '">'
    + '<div class="evaluation-card-header"><div class="evaluation-card-title">' + escapeHtml(result.label || method)
    + ' <span class="evaluation-badge ' + escapeHtml(status) + '">' + escapeHtml(status) + "</span></div>"
    + '<div class="evaluation-card-actions">' + action
    + '<button type="button" class="ghost-button evaluation-card-toggle" data-toggle-baseline-card="'
    + escapeHtml(method) + '" aria-expanded="' + String(!collapsed)
    + '" title="' + (collapsed ? "Expand" : "Collapse") + ' baseline result">'
    + (collapsed ? "Expand" : "Collapse") + '</button></div></div>'
    + '<div class="evaluation-card-body"' + (collapsed ? ' hidden' : '') + '>'
    + body + "</div></article>";
}

function renderEvaluationPanel(payload) {
  state.evaluation = payload;
  byId("evaluationSharedLegend").innerHTML = TIMELINE_MARKER_DEFINITIONS.map(function (definition) {
    return '<span class="evaluation-marker-key"><i class="marker ' + definition.cssClass + '"></i>' + escapeHtml(definition.label) + '</span>';
  }).join('');
  var methods = payload && payload.methods ? payload.methods : {};
  var methodOrder = payload && Array.isArray(payload.method_order) && payload.method_order.length
    ? payload.method_order
    : ["safe", "procvlm", "rynnvalue", "robo_dopamine", "densereward"];
  var available = payload && payload.available_methods ? payload.available_methods.length : 0;
  var record = selectedRollout();
  var conditionLabel = payload && payload.condition_label
    ? payload.condition_label
    : instructionConditionLabel(state.instructionCondition);
  if (payload && payload.variant_available === false) {
    byId("evaluationStatus").textContent = conditionLabel + " is not available for this rollout.";
  } else if ((payload && payload.condition) !== "full_instruction") {
    byId("evaluationStatus").textContent = conditionLabel + ": " + available + " / " + methodOrder.length
      + " variant baseline outputs available. Full-instruction results are not reused.";
  } else {
    byId("evaluationStatus").textContent = available + " / " + methodOrder.length + " baseline outputs available for this rollout.";
  }
  if (state.baselineRunNotice) {
    byId("evaluationStatus").textContent += " · " + state.baselineRunNotice;
    state.baselineRunNotice = "";
  }
  byId("evaluationMethods").innerHTML = methodOrder.map(function (method) {
    return renderEvaluationCard(method, methods[method] || {
      method: method,
      label: method,
      available: false,
      validation: { status: "missing", message: "No response for this method." }
    }, record);
  }).join("");
  updateEvaluationCurrent();
  if (typeof window.lf3rResultsLayoutRefresh === "function") {
    window.lf3rResultsLayoutRefresh();
  }
}


var RESULTS_NUMERIC_ONLY_MESSAGE = "Numeric signals only; this baseline has no text output.";

function resultsSignalKeys(method, signals) {
  var available = Object.keys(signals || {});
  var preferred = {
    procvlm: ["progress"],
    robo_dopamine: ["progress", "hop"],
    rynnvalue: ["value", "relative_value"],
    densereward: ["reward"]
  }[method] || [];
  var selected = preferred.filter(function (name) {
    return available.indexOf(name) !== -1;
  });
  return selected.length ? selected : available.slice(0, 6);
}

function resultsCurrentNumericText(method, sample, videoFrame) {
  if (!sample || !sample.signals) return "";
  var keys = resultsSignalKeys(method, sample.signals);
  if (!keys.length) return "";
  var sampledFrame = Number(sample.frame);
  var currentFrameValue = Number(videoFrame);
  var header = Number.isFinite(sampledFrame) && sampledFrame === currentFrameValue
    ? "Current frame: " + sampledFrame
    : (Number.isFinite(sampledFrame)
      ? "Video frame: " + currentFrameValue + " · nearest sampled frame: " + sampledFrame
      : "Video frame: " + currentFrameValue);
  return header + "\n" + keys.map(function (name) {
    return evaluationSignalLabel(name) + " = " + formatEvaluationNumber(sample.signals[name]);
  }).join("\n");
}

function resultsReleaseStableHeight(output) {
  output.classList.remove("is-stable-height");
  output.classList.add("has-current-numeric-values");
  output.style.removeProperty("--lf3r-output-height");
  output.style.setProperty("height", "auto", "important");
  output.style.setProperty("min-height", "0", "important");
  output.style.setProperty("max-height", "none", "important");
  output.style.setProperty("overflow", "visible", "important");
}

function updateEvaluationCurrent() {
  updateSignalPlayheads();
  if (!state.evaluation || !state.evaluation.methods) return;
  document.querySelectorAll("[data-evaluation-method]").forEach(function (card) {
    var method = String(card.dataset.evaluationMethod || "");
    var result = state.evaluation.methods[method];
    if (!result || !result.available) return;
    var sample = nearestEvaluationSample(result.samples || [], state.currentFrame);
    var output = card.querySelector("[data-current-output]");
    if (!output) return;
    var text = sampleOutputText(sample);
    if (sample && text === RESULTS_NUMERIC_ONLY_MESSAGE) {
      var numericText = resultsCurrentNumericText(method, sample, state.currentFrame);
      if (numericText) {
        text = numericText + "\n\n" + RESULTS_NUMERIC_ONLY_MESSAGE;
        resultsReleaseStableHeight(output);
      }
    }
    output.textContent = text;
    output.hidden = !String(text || "").trim();
  });
}
