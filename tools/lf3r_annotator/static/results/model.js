"use strict";

/* Results sample formatting and signal state. */

function formatEvaluationNumber(value) {
  var number = Number(value);
  if (!Number.isFinite(number)) return "-";
  if (Math.abs(number) >= 1000 || (Math.abs(number) > 0 && Math.abs(number) < 0.001)) {
    return number.toExponential(3);
  }
  return number.toFixed(4).replace(/0+$/, "").replace(/[.]$/, "");
}

function nearestEvaluationSample(samples, frame) {
  if (!samples || !samples.length) return null;
  return samples.reduce(function (nearest, sample) {
    return Math.abs(Number(sample.frame) - frame) < Math.abs(Number(nearest.frame) - frame)
      ? sample
      : nearest;
  });
}

function rynnAnalysisIsDuplicate(sample) {
  if (!sample || !sample.analysis_text || sample.parsed_analysis == null) return false;
  var analysis = String(sample.analysis_text);
  var parsed = typeof sample.parsed_analysis === "string"
    ? sample.parsed_analysis
    : JSON.stringify(sample.parsed_analysis);
  return analysis.indexOf("Video Description:") !== -1
    && analysis.indexOf("Match:") !== -1
    && analysis.indexOf("Success:") !== -1
    && parsed.indexOf("description") !== -1
    && parsed.indexOf("match") !== -1
    && parsed.indexOf("success") !== -1;
}

function sampleOutputText(sample) {
  if (!sample) return "No output at this frame.";
  var parts = [];
  var textEntries = [];
  // ProcVLM reasoning is derived from model_output by removing the final
  // progress sentence. Showing both therefore duplicates the same text.
  if (sample.model_output != null && sample.model_output !== "") {
    textEntries.push(["Model output", sample.model_output]);
  } else if (sample.reasoning != null && sample.reasoning !== "") {
    textEntries.push(["Reasoning", sample.reasoning]);
  }
  [
    ["Analysis", sample.analysis_text],
    ["Parsed analysis", rynnAnalysisIsDuplicate(sample) ? null : sample.parsed_analysis],
    ["Prediction", sample.pred]
  ].forEach(function (entry) {
    textEntries.push(entry);
  });
  textEntries.forEach(function (entry) {
    if (entry[1] == null || entry[1] === "") return;
    var value = typeof entry[1] === "string" ? entry[1] : JSON.stringify(entry[1], null, 2);
    parts.push(entry[0] + ":\n" + value);
  });
  return parts.length ? parts.join("\n\n") : "Numeric signals only; this baseline has no text output.";
}

var EVALUATION_SIGNAL_LABELS = {
  value: "absolute remaining time",
  relative_value: "relative temporal displacement",
  progress: "progress",
  hop: "hop",
  reward: "reward"
};

function evaluationSignalLabel(name) {
  return EVALUATION_SIGNAL_LABELS[name] || name;
}

function evaluationSignalsText(sample) {
  if (!sample || !sample.signals) return "No numeric signals.";
  var entries = Object.keys(sample.signals).map(function (name) {
    return '<span class="evaluation-signal" title="' + escapeHtml(name) + '"><b>' + escapeHtml(evaluationSignalLabel(name)) + '</b> ' + escapeHtml(formatEvaluationNumber(sample.signals[name])) + "</span>";
  });
  return entries.length ? entries.join("") : "No numeric signals.";
}

function evaluationSignalVisibilityKey(method, record) {
  return String(record && record.id || "") + "::" + String(method || "");
}

function defaultEvaluationSignalVisibility(names) {
  var hasPrimarySignals = names.indexOf("progress") !== -1 || names.indexOf("hop") !== -1;
  var visibility = {};
  names.forEach(function (name) {
    visibility[name] = !hasPrimarySignals || name === "progress" || name === "hop";
  });
  return visibility;
}

function evaluationSignalVisibility(method, record, names) {
  var key = evaluationSignalVisibilityKey(method, record);
  if (!state.evaluationSignalVisibility[key]) {
    state.evaluationSignalVisibility[key] = defaultEvaluationSignalVisibility(names);
  }
  names.forEach(function (name) {
    if (state.evaluationSignalVisibility[key][name] == null) {
      state.evaluationSignalVisibility[key][name] = names.indexOf("progress") === -1 && names.indexOf("hop") === -1;
    }
  });
  return state.evaluationSignalVisibility[key];
}

function toggleEvaluationSignal(button) {
  var card = button.closest("[data-evaluation-method]");
  if (!card) return;
  var method = card.dataset.evaluationMethod;
  var record = selectedRollout();
  var name = button.dataset.signalName;
  if (!name) return;
  var key = evaluationSignalVisibilityKey(method, record);
  if (!state.evaluationSignalVisibility[key]) state.evaluationSignalVisibility[key] = {};
  var visible = button.getAttribute("aria-pressed") !== "true";
  state.evaluationSignalVisibility[key][name] = visible;
  button.setAttribute("aria-pressed", String(visible));
  button.classList.toggle("is-hidden", !visible);
  button.title = (visible ? "Hide " : "Show ") + evaluationSignalLabel(name);
  card.querySelectorAll("[data-signal-row]").forEach(function (row) {
    if (row.dataset.signalName === name) row.hidden = !visible;
  });
  card.querySelectorAll("[data-evaluation-signal-path]").forEach(function (path) {
    if (path.dataset.signalName !== name) return;
    path.style.display = visible ? "" : "none";
    path.setAttribute("aria-hidden", String(!visible));
  });
}

function renderLocalizationPredictionMarker(method, result, domain) {
  if (method !== "robo_dopamine" || !result || !result.localization_prediction) return "";
  var prediction = result.localization_prediction;
  var frame = Number(prediction.predicted_frame);
  if (!Number.isFinite(frame)) return "";
  var clamped = Math.max(0, Math.min(domain, frame));
  var left = clamped / Math.max(1, domain) * 100;
  var checkpoint = String(prediction.checkpoint || "");
  var configId = prediction.checkpoint_config_id == null
    ? ""
    : String(prediction.checkpoint_config_id);
  var repeat = prediction.checkpoint_repeat == null
    ? ""
    : String(prediction.checkpoint_repeat);
  var title = "Localization checkpoint prediction: frame " + Math.round(frame);
  if (configId) title += " · " + configId;
  if (repeat) title += " · repeat " + repeat;
  if (checkpoint) title += " · " + checkpoint;
  return '<span class="evaluation-localization-pin" style="left:' + left.toFixed(4)
    + '%" title="' + escapeHtml(title) + '"><span>f'
    + escapeHtml(Math.round(frame)) + '</span></span>';
}

function renderLocalizationPredictionSummary(method, result) {
  if (method !== "robo_dopamine" || !result || !result.localization_prediction) return "";
  var prediction = result.localization_prediction;
  var frame = Number(prediction.predicted_frame);
  if (!Number.isFinite(frame)) return "";
  var details = [];
  if (prediction.checkpoint_config_id != null && prediction.checkpoint_config_id !== "") {
    details.push(String(prediction.checkpoint_config_id));
  }
  if (prediction.checkpoint_repeat != null && prediction.checkpoint_repeat !== "") {
    details.push("repeat " + String(prediction.checkpoint_repeat));
  }
  var checkpoint = String(prediction.checkpoint || "");
  var shortCheckpoint = checkpoint ? checkpoint.split("/").slice(-3).join("/") : "";
  var peakScore = Number(prediction.predicted_sigmoid);
  return '<div class="evaluation-localization-summary">'
    + '<span class="evaluation-localization-summary-label">Localization peak</span>'
    + '<strong>Frame ' + escapeHtml(Math.round(frame))
    + (Number.isFinite(peakScore) ? ' · score ' + escapeHtml(formatEvaluationNumber(peakScore)) : '')
    + '</strong>'
    + (details.length ? '<span>' + escapeHtml(details.join(" · ")) + '</span>' : "")
    + (shortCheckpoint ? '<small title="' + escapeHtml(checkpoint) + '">'
      + escapeHtml(shortCheckpoint) + '</small>' : "")
    + '</div>';
}

function updateSignalPlayheads() {
  document.querySelectorAll('[data-signal-seek]').forEach(function (plot) {
    var frame = Math.max(0, Math.min(Number(plot.dataset.frameMax), currentFrame()));
    plot.setAttribute('aria-valuenow', String(frame));
    plot.querySelector('[data-signal-playhead]').style.left = (frame / Math.max(1, Number(plot.dataset.frameMax)) * 100) + '%';
  });
}

function compactSampleOutput(sample) {
  var text = sampleOutputText(sample).replace(/\s+/g, " ").trim();
  var hasText = [sample.model_output, sample.reasoning, sample.analysis_text, sample.parsed_analysis, sample.pred].some(function (value) { return value != null && value !== ""; });
  if (!hasText && sample.signals) {
    text = Object.keys(sample.signals).map(function (name) { return evaluationSignalLabel(name) + "=" + formatEvaluationNumber(sample.signals[name]); }).join(", ");
  }
  return text.length > 220 ? text.slice(0, 217) + "..." : text;
}

var BASELINE_AUTO_RUN = "__automatic__";
