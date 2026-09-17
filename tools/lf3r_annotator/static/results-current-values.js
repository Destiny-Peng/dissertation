"use strict";

(function installResultsCurrentValues() {
  if (typeof window.updateEvaluationCurrent !== "function") return;
  if (window.updateEvaluationCurrent.__lf3rCurrentNumericValues) return;

  var originalUpdateEvaluationCurrent = window.updateEvaluationCurrent;
  var NUMERIC_ONLY_MESSAGE = "Numeric signals only; this baseline has no text output.";

  function signalKeys(method, signals) {
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
    if (selected.length) return selected;

    // SAFE and any future numeric-only method can still expose a compact
    // current-value summary without turning the output panel into a long dump.
    return available.slice(0, 6);
  }

  function currentNumericText(method, sample, videoFrame) {
    if (!sample || !sample.signals) return "";
    var keys = signalKeys(method, sample.signals);
    if (!keys.length) return "";

    var sampledFrame = Number(sample.frame);
    var currentFrameValue = Number(videoFrame);
    var header;
    if (Number.isFinite(sampledFrame) && sampledFrame === currentFrameValue) {
      header = "Current frame: " + sampledFrame;
    } else if (Number.isFinite(sampledFrame)) {
      header = "Video frame: " + currentFrameValue + " · nearest sampled frame: " + sampledFrame;
    } else {
      header = "Video frame: " + currentFrameValue;
    }

    var lines = keys.map(function (name) {
      return evaluationSignalLabel(name) + " = " + formatEvaluationNumber(sample.signals[name]);
    });
    return header + "\n" + lines.join("\n");
  }

  function releaseStableHeight(output) {
    // results-layout.js normally fixes the output height to the largest native
    // sampleOutputText() string so text-heavy methods do not jump while the
    // video plays. Numeric-only methods used to measure only the short fallback
    // sentence, though, and the current-value readout added here is taller.
    // Let these compact numeric cards size to their actual content instead.
    output.classList.remove("is-stable-height");
    output.classList.add("has-current-numeric-values");
    output.style.removeProperty("--lf3r-output-height");
    output.style.setProperty("height", "auto", "important");
    output.style.setProperty("min-height", "0", "important");
    output.style.setProperty("max-height", "none", "important");
    output.style.setProperty("overflow", "visible", "important");
  }

  function updateEvaluationCurrentWithValues() {
    originalUpdateEvaluationCurrent();
    if (!state.evaluation || !state.evaluation.methods) return;

    document.querySelectorAll("[data-evaluation-method]").forEach(function (card) {
      var method = String(card.dataset.evaluationMethod || "");
      var result = state.evaluation.methods[method];
      if (!result || !result.available) return;

      var sample = nearestEvaluationSample(result.samples || [], state.currentFrame);
      var output = card.querySelector("[data-current-output]");
      if (!output || !sample) return;

      var originalText = sampleOutputText(sample);
      // Keep text-producing methods unchanged. The requested readout fills the
      // otherwise-unhelpful numeric-only message used by Robo-Dopamine and
      // other scalar baselines.
      if (originalText !== NUMERIC_ONLY_MESSAGE) return;

      var numericText = currentNumericText(method, sample, state.currentFrame);
      output.textContent = numericText
        ? numericText + "\n\n" + NUMERIC_ONLY_MESSAGE
        : NUMERIC_ONLY_MESSAGE;
      output.hidden = false;
      if (numericText) releaseStableHeight(output);
    });
  }

  updateEvaluationCurrentWithValues.__lf3rCurrentNumericValues = true;
  window.updateEvaluationCurrent = updateEvaluationCurrentWithValues;
})();
