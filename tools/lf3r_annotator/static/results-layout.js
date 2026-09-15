"use strict";

(function installReviewLayoutEnhancements() {
  /* Load the final visual layer after all legacy / refresh styles. */
  if (!document.getElementById("lf3rPolishStyles")) {
    var polish = document.createElement("link");
    polish.id = "lf3rPolishStyles";
    polish.rel = "stylesheet";
    polish.href = "/static/styles-polish.css?v=20260915-polish";
    document.head.appendChild(polish);
  }

  /* Migrate only the untouched legacy default palette. Custom user themes are
     left alone. This also keeps Restore defaults aligned with the new theme. */
  var LEGACY_PALETTE = {
    background_color: "#0b0d10",
    surface_color: "#11151a",
    surface_raised_color: "#171c22",
    control_color: "#0f1318",
    text_color: "#f4f6f7",
    muted_color: "#98a3ad",
    accent_color: "#67d9b5"
  };
  var POLISHED_PALETTE = {
    background_color: "#0d1117",
    surface_color: "#121820",
    surface_raised_color: "#18202a",
    control_color: "#0f151c",
    text_color: "#e7edf3",
    muted_color: "#929eaa",
    accent_color: "#79aa9e"
  };

  function usesLegacyDefaultPalette(settings) {
    if (!settings) return false;
    return Object.keys(LEGACY_PALETTE).every(function (key) {
      return String(settings[key] || "").toLowerCase() === LEGACY_PALETTE[key];
    });
  }

  function migratedPalette(settings) {
    if (!usesLegacyDefaultPalette(settings)) return settings;
    return Object.assign({}, settings, POLISHED_PALETTE);
  }

  if (window.SETTINGS_DEFAULTS) Object.assign(window.SETTINGS_DEFAULTS, POLISHED_PALETTE);
  if (window.SETTINGS_PRESETS && window.SETTINGS_PRESETS.midnight) {
    Object.assign(window.SETTINGS_PRESETS.midnight, POLISHED_PALETTE);
  }

  var originalApplySettings = window.workspaceApplySettings;
  if (typeof originalApplySettings === "function") {
    window.workspaceApplySettings = function (settings) {
      return originalApplySettings(migratedPalette(settings));
    };
  }

  var originalSettingsToForm = window.workspaceSettingsToForm;
  if (typeof originalSettingsToForm === "function") {
    window.workspaceSettingsToForm = function (settings) {
      return originalSettingsToForm(migratedPalette(settings));
    };
  }

  function refreshLoadedDefaultPalette() {
    if (!window.workspaceState || !workspaceState.settings || !usesLegacyDefaultPalette(workspaceState.settings)) return;
    workspaceState.settings = migratedPalette(workspaceState.settings);
    if (typeof window.workspaceApplySettings === "function") window.workspaceApplySettings(workspaceState.settings);
    if (typeof window.workspaceSettingsToForm === "function") window.workspaceSettingsToForm(workspaceState.settings);
  }
  window.setTimeout(refreshLoadedDefaultPalette, 0);
  window.setTimeout(refreshLoadedDefaultPalette, 250);

  /* RynnValue's human-readable Analysis and Parsed analysis encode the same
     description/match/success information. Keep only the readable version. */
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

  var originalSampleOutputText = window.sampleOutputText;
  if (typeof originalSampleOutputText === "function") {
    window.sampleOutputText = function (sample) {
      if (!rynnAnalysisIsDuplicate(sample)) return originalSampleOutputText(sample);
      var parts = [];
      if (sample.model_output != null && sample.model_output !== "") {
        parts.push("Model output:\n" + String(sample.model_output));
      } else if (sample.reasoning != null && sample.reasoning !== "") {
        parts.push("Reasoning:\n" + String(sample.reasoning));
      }
      if (sample.analysis_text != null && sample.analysis_text !== "") {
        parts.push("Analysis:\n" + String(sample.analysis_text));
      }
      if (sample.pred != null && sample.pred !== "") {
        var prediction = typeof sample.pred === "string" ? sample.pred : JSON.stringify(sample.pred, null, 2);
        parts.push("Prediction:\n" + prediction);
      }
      return parts.length ? parts.join("\n\n") : "Numeric signals only; this baseline has no text output.";
    };
  }

  var workspace = document.getElementById("reviewWorkspace");
  var queue = workspace && workspace.querySelector(".queue-panel");
  var methods = document.getElementById("evaluationMethods");
  if (!workspace || !queue || !methods) return;

  var queueCollapsed = false;
  try {
    var stored = localStorage.getItem("lf3r.review.queueCollapsed");
    if (stored == null) stored = localStorage.getItem("lf3r.results.queueCollapsed");
    queueCollapsed = stored === "true";
  } catch (_) {}

  var queueToggle = document.createElement("button");
  queueToggle.id = "reviewQueueToggle";
  queueToggle.type = "button";
  queueToggle.className = "review-queue-toggle";
  queueToggle.hidden = true;
  queue.appendChild(queueToggle);

  var sizingScheduled = false;
  var lastMethodsWidth = 0;

  function isSharedReviewView() {
    return document.body.dataset.view === "results" || document.body.dataset.view === "annotate";
  }

  function isResultsView() {
    return document.body.dataset.view === "results";
  }

  function scheduleOutputSizing() {
    if (sizingScheduled) return;
    sizingScheduled = true;
    window.requestAnimationFrame(function () {
      window.requestAnimationFrame(sizeEvaluationOutputs);
    });
  }

  function applyQueueState() {
    var active = isSharedReviewView() && queueCollapsed;
    workspace.classList.toggle("review-queue-collapsed", active);
    queueToggle.hidden = !isSharedReviewView();
    queueToggle.textContent = active ? "›" : "‹";
    queueToggle.setAttribute("aria-expanded", String(!active));
    queueToggle.setAttribute("aria-label", active ? "Expand rollout queue" : "Collapse rollout queue");
    queueToggle.title = active ? "Expand rollout queue" : "Collapse rollout queue";
    scheduleOutputSizing();
  }

  queueToggle.addEventListener("click", function () {
    queueCollapsed = !queueCollapsed;
    try {
      localStorage.setItem("lf3r.review.queueCollapsed", String(queueCollapsed));
      localStorage.setItem("lf3r.results.queueCollapsed", String(queueCollapsed));
    } catch (_) {}
    applyQueueState();
  });

  function uniqueOutputTexts(result) {
    var seen = Object.create(null);
    var texts = [];
    (result && result.samples || []).forEach(function (sample) {
      var text = typeof window.sampleOutputText === "function"
        ? window.sampleOutputText(sample)
        : "";
      text = String(text || "");
      if (!text.trim() || seen[text]) return;
      seen[text] = true;
      texts.push(text);
    });
    return texts.length ? texts : ["No output at this frame."];
  }

  function sizeEvaluationOutput(card, result) {
    var output = card.querySelector("[data-current-output]");
    if (!output) return;

    output.classList.remove("is-stable-height");
    output.style.removeProperty("--lf3r-output-height");

    var width = output.getBoundingClientRect().width;
    if (!Number.isFinite(width) || width < 24) return;

    var measurer = output.cloneNode(false);
    measurer.removeAttribute("data-current-output");
    measurer.removeAttribute("hidden");
    measurer.classList.add("evaluation-output-measurer");
    measurer.style.width = width + "px";
    output.parentNode.appendChild(measurer);

    var maxHeight = 0;
    uniqueOutputTexts(result).forEach(function (text) {
      measurer.textContent = text;
      maxHeight = Math.max(maxHeight, measurer.scrollHeight);
    });
    measurer.remove();

    if (maxHeight > 0) {
      output.style.setProperty("--lf3r-output-height", Math.ceil(maxHeight + 2) + "px");
      output.classList.add("is-stable-height");
    }
  }

  function sizeEvaluationOutputs() {
    sizingScheduled = false;
    if (!isResultsView() || !window.state || !state.evaluation || !state.evaluation.methods) return;
    document.querySelectorAll("[data-evaluation-method]").forEach(function (card) {
      var result = state.evaluation.methods[card.dataset.evaluationMethod];
      if (!result || !result.available) return;
      sizeEvaluationOutput(card, result);
    });
  }

  var originalUpdateEvaluationCurrent = window.updateEvaluationCurrent;
  if (typeof originalUpdateEvaluationCurrent === "function") {
    window.updateEvaluationCurrent = function () {
      var result = originalUpdateEvaluationCurrent.apply(this, arguments);
      /* The displayed strings may have changed length after seeking, but the
         reserved box height remains the maximum for the whole rollout. */
      return result;
    };
  }

  var bodyObserver = new MutationObserver(function (mutations) {
    if (mutations.some(function (mutation) { return mutation.attributeName === "data-view"; })) {
      applyQueueState();
    }
  });
  bodyObserver.observe(document.body, { attributes: true, attributeFilter: ["data-view"] });

  var methodsObserver = new MutationObserver(function () {
    scheduleOutputSizing();
  });
  methodsObserver.observe(methods, { childList: true });

  if (typeof ResizeObserver === "function") {
    var resizeObserver = new ResizeObserver(function (entries) {
      var width = entries.length ? entries[0].contentRect.width : 0;
      if (Math.abs(width - lastMethodsWidth) < 1) return;
      lastMethodsWidth = width;
      scheduleOutputSizing();
    });
    resizeObserver.observe(methods);
  }

  window.addEventListener("resize", scheduleOutputSizing);
  window.lf3rResultsLayoutRefresh = function () {
    applyQueueState();
    scheduleOutputSizing();
  };

  applyQueueState();
  scheduleOutputSizing();
})();