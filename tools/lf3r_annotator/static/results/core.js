"use strict";

/*
 * Results page core.
 *
 * Owns baseline-result rendering, signal/localization visualization, run
 * selection, post-hoc localization, result loading, and progressive refresh.
 * Keep Runs/batch execution logic in app.js until the Runs refactor.
 */

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

function renderLocalizationPredictionCurve(method, result, record) {
  if (method !== "robo_dopamine" || !result || !result.localization_prediction) return "";
  var prediction = result.localization_prediction;
  var frames = Array.isArray(prediction.frames) ? prediction.frames : [];
  var scores = Array.isArray(prediction.sigmoid_scores) ? prediction.sigmoid_scores : [];
  var count = Math.min(frames.length, scores.length);
  if (!count) return "";

  var rows = [];
  for (var index = 0; index < count; index += 1) {
    var frame = Number(frames[index]);
    var score = Number(scores[index]);
    if (!Number.isFinite(frame) || !Number.isFinite(score)) continue;
    rows.push({ frame: frame, score: Math.max(0, Math.min(1, score)) });
  }
  if (!rows.length) return "";

  var maxFrame = Math.max.apply(null, rows.map(function (row) { return row.frame; }));
  var domain = timelineDomainMax(record, maxFrame);
  var points = rows.map(function (row) {
    return (Math.max(0, Math.min(domain, row.frame)) / Math.max(1, domain) * 100).toFixed(4)
      + ',' + (87 - row.score * 79).toFixed(4);
  }).join(' ');

  return '<section class="signal-row evaluation-localization-curve" data-localization-curve>'
    + '<div class="signal-row-title">Localization score</div><div class="signal-axis-layout">'
    + '<div class="signal-y-axis" aria-label="Localization score axis"><span>1</span><span>0.5</span><span>0</span></div>'
    + '<div class="evaluation-chart"><div class="timeline-track evaluation-chart-plot" data-signal-seek data-frame-max="'
    + domain + '" tabindex="0" role="slider" aria-label="Localization score: click to seek video; arrow keys step frames"'
    + ' aria-valuemin="0" aria-valuemax="' + domain + '" aria-valuenow="' + currentFrame() + '">'
    + '<svg viewBox="0 0 100 105" preserveAspectRatio="none" role="img" aria-label="Localization checkpoint score versus video frame">'
    + '<title>Localization checkpoint sigmoid score; raw logits are preserved in the saved localization artifact</title>'
    + '<path d="M0 8H100 M0 47.5H100 M0 87H100" stroke="var(--line)" stroke-width="1" vector-effect="non-scaling-stroke"/>'
    + '<polyline fill="none" stroke="#d6a5f5" stroke-width="1.8" vector-effect="non-scaling-stroke" points="'
    + points + '"/></svg>'
    + '<div class="evaluation-chart-markers" data-evaluation-onset-markers data-frame-max="' + domain + '">'
    + renderEvaluationOnsetMarkers(record, domain)
    + renderLocalizationPredictionMarker(method, result, domain)
    + '</div><div class="signal-playhead" data-signal-playhead style="left:'
    + (Math.max(0, Math.min(domain, currentFrame())) / Math.max(1, domain) * 100)
    + '%"></div></div></div></div>'
    + '<div class="signal-frame-axis"><span>0</span><span>video frame</span><span>' + domain + '</span></div>'
    + '<div class="evaluation-meta">Full checkpoint output curve · sigmoid score shown; raw logits are saved alongside it.</div>'
    + '</section>';
}

function signalDomain(method, name, values) {
    // ProcVLM progress is emitted as a percentage on the 0–100 scale.
    if (method === "procvlm" && name === "progress") {
      return { low: 0, high: 100, fixed: true };
    }

    // Robo-Dopamine progress and its perspective components share the same
    // normalized [0, 1] semantics. Hop is a temporal difference and may be
    // signed, so keep hop/component-hop on the original data-driven scale.
    if (method === "robo_dopamine" && (name === "progress" || /_progress$/.test(name))) {
      return { low: 0, high: 1, fixed: true };
    }

    var low = Math.min.apply(null, values);
    var high = Math.max.apply(null, values);
    if (low === high) {
      var pad = Math.max(Math.abs(low) * 0.05, 0.01);
      low -= pad;
      high += pad;
    }
    return { low: low, high: high, fixed: false };
  }

function renderSignalChart(method, result, record) {
    var samples = (result.samples || []).filter(function (sample) {
      return sample && Number.isFinite(Number(sample.frame)) && sample.signals;
    });
    var names = [];
    samples.forEach(function (sample) {
      Object.keys(sample.signals).forEach(function (name) {
        if (names.indexOf(name) === -1) names.push(name);
      });
    });
    if (!samples.length || !names.length) {
      return '<div class="evaluation-empty">No numeric signal samples to plot.</div>';
    }

    var domain = timelineDomainMax(
      record,
      Math.max.apply(null, samples.map(function (sample) { return Number(sample.frame); }))
    );
    var colors = ["#67d9b5", "#77bdfb", "#e7c15c", "#ef8d53", "#d6a5f5", "#f07869"];
    var visibility = evaluationSignalVisibility(method, record, names);
    var legend = names.map(function (name, index) {
      var visible = visibility[name] !== false;
      return '<button type="button" class="evaluation-signal-toggle' + (visible ? '' : ' is-hidden')
        + '" data-evaluation-signal-toggle data-signal-name="' + escapeHtml(name)
        + '" aria-pressed="' + visible + '" title="Show/hide '
        + escapeHtml(evaluationSignalLabel(name)) + '"><i style="background:'
        + colors[index % colors.length] + '"></i>'
        + escapeHtml(evaluationSignalLabel(name)) + '</button>';
    }).join('');

    var plots = names.map(function (name, index) {
      var rows = samples.filter(function (sample) {
        return sample.signals[name] != null && Number.isFinite(Number(sample.signals[name]));
      });
      if (!rows.length) return '';

      var values = rows.map(function (sample) { return Number(sample.signals[name]); });
      var axis = signalDomain(method, name, values);
      var low = axis.low;
      var high = axis.high;
      var span = high - low;
      var points = rows.map(function (sample) {
        var value = Number(sample.signals[name]);
        // Fixed semantic ranges should not be distorted by a malformed sample;
        // clamp only for drawing while the original numeric value remains in
        // the output/history data.
        var plottedValue = axis.fixed ? Math.max(low, Math.min(high, value)) : value;
        return (Math.max(0, Math.min(domain, Number(sample.frame))) / domain * 100).toFixed(4)
          + ',' + (87 - (plottedValue - low) / span * 79).toFixed(4);
      }).join(' ');

      var scaleTitle = axis.fixed
        ? 'Fixed semantic value scale ' + low + '–' + high + '; video frame 0–' + domain
        : 'Original signal values; video frame 0–' + domain;

      return '<section class="signal-row" data-signal-row data-signal-name="' + escapeHtml(name) + '"'
        + (visibility[name] === false ? ' hidden' : '')
        + '><div class="signal-row-title">'
        + (names.length > 1 ? escapeHtml(evaluationSignalLabel(name)) : '')
        + '</div><div class="signal-axis-layout">'
        + '<div class="signal-y-axis" aria-label="Vertical axis values"><span>'
        + formatEvaluationNumber(high) + '</span><span>'
        + formatEvaluationNumber((high + low) / 2) + '</span><span>'
        + formatEvaluationNumber(low) + '</span></div>'
        + '<div class="evaluation-chart"><div class="timeline-track evaluation-chart-plot" data-signal-seek data-frame-max="'
        + domain + '" tabindex="0" role="slider" aria-label="'
        + escapeHtml(evaluationSignalLabel(name))
        + ': click to seek video; arrow keys step frames" aria-valuemin="0" aria-valuemax="'
        + domain + '" aria-valuenow="' + currentFrame() + '">'
        + '<svg viewBox="0 0 100 105" preserveAspectRatio="none" role="img" aria-label="'
        + escapeHtml(evaluationSignalLabel(name)) + ' versus video frame">'
        + '<title>' + escapeHtml(scaleTitle) + '</title>'
        + '<path d="M0 8H100 M0 47.5H100 M0 87H100" stroke="var(--line)" stroke-width="1" vector-effect="non-scaling-stroke"/>'
        + '<polyline fill="none" stroke="' + colors[index % colors.length]
        + '" stroke-width="1.5" vector-effect="non-scaling-stroke" points="'
        + points + '"/></svg>'
        + '<div class="evaluation-chart-markers" data-evaluation-onset-markers data-frame-max="'
        + domain + '">' + renderEvaluationOnsetMarkers(record, domain)
        + (typeof renderLocalizationPredictionMarker === "function"
          ? renderLocalizationPredictionMarker(method, result, domain)
          : "")
        + '</div><div class="signal-playhead" data-signal-playhead style="left:'
        + (Math.max(0, Math.min(domain, currentFrame())) / domain * 100)
        + '%"></div></div></div></div>'
        + '<div class="signal-frame-axis"><span>0</span><span>video frame</span><span>'
        + domain + '</span></div></section>';
    }).join('');

    return '<div class="evaluation-chart-legend">' + legend + '</div>' + plots;
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

function baselineRunPreferenceKey(method, record, condition) {
  return String(condition || "full_instruction") + "::" + String(record && record.id || "") + "::" + String(method || "");
}

function baselineRunAllKey(method, condition) {
  return String(condition || "full_instruction") + "::" + String(method || "");
}

function hasOwn(object, key) {
  return Object.prototype.hasOwnProperty.call(object, key);
}

function baselineRunOverride(method, record, condition) {
  var allKey = baselineRunAllKey(method, condition);
  if (hasOwn(state.baselineRunAll, allKey) && state.baselineRunAll[allKey] !== BASELINE_AUTO_RUN) {
    return state.baselineRunAll[allKey];
  }
  var localKey = baselineRunPreferenceKey(method, record, condition);
  if (hasOwn(state.baselineRunSelections, localKey)
      && state.baselineRunSelections[localKey] !== BASELINE_AUTO_RUN) {
    return state.baselineRunSelections[localKey];
  }
  return "";
}

function baselineRunSelectionValue(method, record, condition) {
  var allKey = baselineRunAllKey(method, condition);
  if (hasOwn(state.baselineRunAll, allKey)) return state.baselineRunAll[allKey];
  var localKey = baselineRunPreferenceKey(method, record, condition);
  if (hasOwn(state.baselineRunSelections, localKey)) return state.baselineRunSelections[localKey];
  return BASELINE_AUTO_RUN;
}

function baselineRunDisplay(run) {
  var root = String(run && run.run_root || "unknown");
  var parts = root.split("/");
  var shortRoot = parts.slice(Math.max(0, parts.length - 2)).join("/");
  var completed = run && run.completed_jobs != null ? run.completed_jobs : 0;
  var selected = run && run.selected_rollouts != null ? run.selected_rollouts : (run && run.run_rollout_count || 0);
  var when = run && (run.completed_at || run.created_at);
  var date = when ? formatDate(when) : "unknown time";
  var status = run && run.status ? " · " + run.status : "";
  var procedureMode = run && run.procedure_mode
    ? " · " + String(run.procedure_mode)
    : "";
  return date + " · " + completed + "/" + selected + status + procedureMode + " · " + shortRoot;
}

function baselineRunOptions(method, record, condition) {
  if (!Array.isArray(state.baselineRuns) || !record) return [];
  return state.baselineRuns.filter(function (run) {
    if ((run.method || run.baseline) !== method) return false;
    var ids = Array.isArray(run.run_source_rollout_ids)
      ? run.run_source_rollout_ids
      : (Array.isArray(run.run_rollout_ids) ? run.run_rollout_ids : []);
    if (ids.indexOf(record.id) === -1) return false;
    var runCondition = run.instruction_condition || "unknown";
    if (condition === "full_instruction") return runCondition === "full_instruction" || runCondition === "unknown";
    return runCondition === condition;
  });
}

function renderBaselineRunControls(method, result, record) {
  var condition = state.instructionCondition || "full_instruction";
  var selected = baselineRunSelectionValue(method, record, condition);
  var options = baselineRunOptions(method, record, condition);
  var optionHtml = '<option value="' + BASELINE_AUTO_RUN + '"' + (selected === BASELINE_AUTO_RUN ? " selected" : "") + '>Automatic · newest available</option>';
  options.forEach(function (run) {
    var root = String(run.run_root || "");
    optionHtml += '<option value="' + escapeHtml(root) + '" title="' + escapeHtml(root) + '"'
      + (selected === root ? " selected" : "") + '>'
      + escapeHtml(baselineRunDisplay(run)) + '</option>';
  });
  var allKey = baselineRunAllKey(method, condition);
  var applied = hasOwn(state.baselineRunAll, allKey) && state.baselineRunAll[allKey] !== BASELINE_AUTO_RUN;
  var note = applied ? "Applied to all rollouts" : "This rollout";
  var disabled = options.length === 0 ? " disabled" : "";
  return '<div class="evaluation-run-controls">'
    + '<label><span>Result run</span><select data-evaluation-run-select data-evaluation-method="' + escapeHtml(method)
    + '" aria-label="' + escapeHtml((result.label || method) + " result run") + '"' + disabled + '>'
    + optionHtml + '</select></label>'
    + '<button type="button" class="ghost-button" data-apply-baseline-run data-evaluation-method="' + escapeHtml(method)
    + '"' + disabled + '>Apply to all</button>'
    + (options.length ? "" : '<small>No completed run available</small>')
    + '</div>';
}

async 
var resultsCatalogSerial = 0;
var resultsRefreshTimer = null;

function loadBaselineRunCatalog(condition, force) {
  condition = condition || state.instructionCondition || "full_instruction";
  if (!force && Array.isArray(state.baselineRuns) && state.baselineRunsCondition === condition) {
    return Promise.resolve(state.baselineRuns);
  }
  if (!force && state.baselineRunsLoading && state.baselineRunsCondition === condition) {
    return state.baselineRunsLoading;
  }

  var serial = ++resultsCatalogSerial;
  state.baselineRunsCondition = condition;
  var request = fetch(
    "/api/baselines/runs?scope=all&condition=" + encodeURIComponent(condition),
    { cache: "no-store" }
  )
    .then(function (response) {
      if (!response.ok) throw new Error("Could not load baseline run catalog");
      return response.json();
    })
    .then(function (payload) {
      var runs = payload.runs || [];
      if (serial === resultsCatalogSerial) {
        state.baselineRuns = runs;
        state.baselineRunsCondition = payload.condition || condition;
        state.baselineRunNotice = "";
      }
      return runs;
    })
    .catch(function (error) {
      if (serial === resultsCatalogSerial) {
        state.baselineRuns = [];
        state.baselineRunsCondition = condition;
        state.baselineRunNotice = error.message;
      }
      return [];
    })
    .finally(function () {
      if (serial === resultsCatalogSerial) state.baselineRunsLoading = null;
    });

  state.baselineRunsLoading = request;
  return request;
}

function resultsCompletedCount(job) {
  var value = Number(job && job.completed_jobs);
  return Number.isFinite(value) ? value : 0;
}

function resultsMatchingProgressiveRun(job, record, condition) {
  if (!job || !record) return null;
  var options = baselineRunOptions(job.baseline, record, condition) || [];
  var jobRoot = String(job.run_root || "");
  for (var index = 0; index < options.length; index += 1) {
    var run = options[index];
    if (jobRoot && String(run.run_root || "") !== jobRoot) continue;
    if (!jobRoot && run.status !== "running" && run.status !== "complete"
        && run.status !== "complete_with_errors") continue;
    return run;
  }
  return null;
}

function resultsCurrentDisplayedRunRoot(method) {
  var result = state.evaluation && state.evaluation.methods
    ? state.evaluation.methods[method]
    : null;
  return String(result && result.run && result.run.run_root || "");
}

function resultsOnPersistentJobChanged(previous, job) {
  if (!job || job.job_type !== "baseline"
      || resultsCompletedCount(job) <= resultsCompletedCount(previous)) return;
  if (resultsRefreshTimer) window.clearTimeout(resultsRefreshTimer);
  resultsRefreshTimer = window.setTimeout(function () {
    resultsRefreshTimer = null;
    var rolloutId = state.selectedId;
    var condition = state.instructionCondition || "full_instruction";
    var record = (state.rollouts || []).find(function (item) {
      return item.id === rolloutId;
    }) || null;
    loadBaselineRunCatalog(condition, true).then(function () {
      if (!rolloutId || state.selectedId !== rolloutId
          || state.instructionCondition !== condition) return;
      var run = resultsMatchingProgressiveRun(job, record, condition);
      if (!run) return;
      if (resultsCurrentDisplayedRunRoot(job.baseline) === String(run.run_root || "")) return;
      loadEvaluation(rolloutId);
    });
  }, 180);
}

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
      + renderLocalizationPredictionCurve(method, result, record)
      + renderSignalChart(method, result, record)
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

async function runPosthocLocalization(scope, button) {
  var record = selectedRollout();
  if (!record || !state.evaluation || !state.evaluation.methods) return;
  var result = state.evaluation.methods.robo_dopamine;
  if (!result || !result.available || !result.run || !result.run.run_root) return;
  var card = button.closest("[data-evaluation-method]");
  var input = card && card.querySelector("[data-posthoc-localization-ckpt]");
  var status = card && card.querySelector("[data-posthoc-localization-status]");
  var checkpoint = input ? input.value.trim() : "";
  if (!checkpoint) {
    if (status) status.textContent = "Enter a localization checkpoint first.";
    return;
  }
  state.roboPosthocCheckpoint = checkpoint;
  try {
    sessionStorage.setItem("lf3r.results.roboPosthocCheckpoint", checkpoint);
  } catch (_error) {}
  var original = button.textContent;
  button.disabled = true;
  button.textContent = scope === "all" ? "Applying…" : "Running…";
  if (status) {
    status.textContent = scope === "all"
      ? "Applying checkpoint to every saved fused rollout in this Robo-Dopamine run…"
      : "Running localization head on this saved fused result…";
  }
  try {
    var response = await fetch(
      "/api/baselines/posthoc-localization/" + encodeURIComponent(record.id),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_root: result.run.run_root,
          checkpoint: checkpoint,
          scope: scope,
          source_worker_result: (result.raw_files || []).find(function (path) {
            return /\/worker_result[.]json$/.test(String(path));
          }) || ""
        })
      }
    );
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Post-hoc localization failed");
    var data = payload.posthoc_localization || {};
    var completed = Array.isArray(data.results) ? data.results.length : 0;
    var failed = Array.isArray(data.errors) ? data.errors.length : 0;
    state.baselineRunNotice = "Post-hoc localization: " + completed
      + " result(s)" + (failed ? ", " + failed + " skipped/failed" : "");
    await loadEvaluation(record.id);
  } catch (error) {
    if (status) status.textContent = "Post-hoc localization error: " + error.message;
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function selectPosthocLocalization(select) {
  if (!state.evaluation || !state.evaluation.methods) return;
  var result = state.evaluation.methods.robo_dopamine;
  if (!result || !Array.isArray(result.posthoc_localizations)) return;
  var index = Number(select.value);
  var chosen = result.posthoc_localizations[index];
  if (!chosen) return;
  result.localization_prediction = chosen;
  renderEvaluationPanel(state.evaluation);
}

async function loadEvaluation(rolloutId) {
  var requestId = ++state.evaluationRequest;
  var condition = state.instructionCondition || "full_instruction";
  var record = state.rollouts.find(function (item) { return item.id === rolloutId; }) || null;
  state.evaluation = null;
  byId("evaluationStatus").textContent = "Loading baseline outputs...";
  byId("evaluationMethods").innerHTML = '<div class="evaluation-empty">Reading completed baseline runs...</div>';
  try {
    await loadBaselineRunCatalog(condition);
    var query = ["condition=" + encodeURIComponent(condition)];
    ["safe", "procvlm", "rynnvalue", "robo_dopamine", "densereward"].forEach(function (method) {
      var override = baselineRunOverride(method, record, condition);
      if (override) query.push("run_" + method + "=" + encodeURIComponent(override));
    });
    var response = await fetch(
      "/api/baselines/" + encodeURIComponent(rolloutId) + "?" + query.join("&"),
      { cache: "no-store" }
    );
    var payload = await response.json();
    if (requestId !== state.evaluationRequest
      || state.selectedId !== rolloutId
      || state.instructionCondition !== condition) return;
    if (!response.ok) throw new Error(payload.error || "Could not load baseline outputs");
    renderEvaluationPanel(payload.evaluation);
  } catch (error) {
    if (requestId !== state.evaluationRequest
      || state.selectedId !== rolloutId
      || state.instructionCondition !== condition) return;
    byId("evaluationStatus").textContent = "Baseline output error: " + error.message;
    byId("evaluationMethods").innerHTML = '<div class="evaluation-empty">Select Refresh to try again.</div>';
  }
}


function bindResultsEvents() {
  var reload = byId("reloadEvaluation");
  if (reload) {
    reload.addEventListener("click", function () {
      if (state.selectedId) loadEvaluation(state.selectedId);
    });
  }

  var methods = byId("evaluationMethods");
  if (!methods) return;

  methods.addEventListener("change", function (event) {
    var posthocSelect = event.target.closest("[data-posthoc-localization-select]");
    if (posthocSelect) {
      selectPosthocLocalization(posthocSelect);
      return;
    }
    var select = event.target.closest("[data-evaluation-run-select]");
    if (!select) return;
    var record = selectedRollout();
    if (!record) return;
    var method = select.dataset.evaluationMethod;
    var key = baselineRunPreferenceKey(method, record, state.instructionCondition);
    state.baselineRunSelections[key] = select.value || BASELINE_AUTO_RUN;
    loadEvaluation(record.id);
  });

  methods.addEventListener("click", function (event) {
    var collapseButton = event.target.closest("[data-toggle-baseline-card]");
    if (collapseButton) {
      var methodName = collapseButton.dataset.toggleBaselineCard;
      state.baselineCollapsed[methodName] = !baselineCardCollapsed(methodName);
      persistBaselineCollapsed();
      var card = collapseButton.closest("[data-evaluation-method]");
      var body = card && card.querySelector(".evaluation-card-body");
      var collapsed = baselineCardCollapsed(methodName);
      if (card) card.classList.toggle("is-collapsed", collapsed);
      if (body) body.hidden = collapsed;
      collapseButton.textContent = collapsed ? "Expand" : "Collapse";
      collapseButton.setAttribute("aria-expanded", String(!collapsed));
      collapseButton.title = (collapsed ? "Expand" : "Collapse") + " baseline result";
      return;
    }

    var posthocButton = event.target.closest("[data-run-posthoc-localization]");
    if (posthocButton) {
      runPosthocLocalization(posthocButton.dataset.runPosthocLocalization, posthocButton);
      return;
    }

    var applyButton = event.target.closest("[data-apply-baseline-run]");
    if (applyButton) {
      var record = selectedRollout();
      if (!record) return;
      var method = applyButton.dataset.evaluationMethod;
      var card = applyButton.closest("[data-evaluation-method]");
      var select = card && card.querySelector("[data-evaluation-run-select]");
      var value = select ? (select.value || BASELINE_AUTO_RUN) : BASELINE_AUTO_RUN;
      var key = baselineRunAllKey(method, state.instructionCondition);
      if (value === BASELINE_AUTO_RUN) delete state.baselineRunAll[key];
      else state.baselineRunAll[key] = value;
      state.baselineRunNotice = value === BASELINE_AUTO_RUN
        ? (method + " reverted to automatic run selection for all rollouts")
        : (method + " run applied to all rollouts when that run contains the rollout");
      loadEvaluation(record.id);
      return;
    }

    var signalButton = event.target.closest("[data-evaluation-signal-toggle]");
    if (signalButton) {
      toggleEvaluationSignal(signalButton);
      return;
    }

    var runButton = event.target.closest("[data-run-baseline]");
    if (runButton) {
      var opener = window.lf3rOpenSingleBaselineConfig;
      if (typeof opener === "function") {
        opener(String(runButton.dataset.runBaseline || ""));
      } else {
        var status = byId("evaluationStatus");
        if (status) status.textContent = "Baseline configurator is still loading. Try again.";
      }
    }
  });
}
