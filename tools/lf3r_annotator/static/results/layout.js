"use strict";

(function installResultsLayout() {
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

  var queueToggle = document.getElementById("reviewQueueToggle");
  if (!queueToggle) {
    queueToggle = document.createElement("button");
    queueToggle.id = "reviewQueueToggle";
    queueToggle.type = "button";
    queueToggle.className = "review-queue-toggle";
    queueToggle.hidden = true;
    queue.appendChild(queueToggle);
  }

  var sizingScheduled = false;
  var lastMethodsWidth = 0;
  var OUTPUT_MEASURE_BATCH_SIZE = 24;

  function isSharedReviewView() {
    return document.body.dataset.view === "results" || document.body.dataset.view === "annotate";
  }

  function isResultsView() {
    return document.body.dataset.view === "results";
  }

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

  function scheduleOutputMeasure(callback) {
    if (typeof window.requestIdleCallback === "function") {
      window.requestIdleCallback(callback, { timeout: 80 });
      return;
    }
    window.setTimeout(callback, 0);
  }

  function sizeEvaluationOutput(card, result) {
    var output = card.querySelector("[data-current-output]");
    if (!output || output.classList.contains("has-current-numeric-values")) return;

    output.classList.remove("is-stable-height");
    output.style.removeProperty("--lf3r-output-height");

    var width = output.getBoundingClientRect().width;
    if (!Number.isFinite(width) || width < 24) return;

    var texts = uniqueOutputTexts(result);
    var token = String((Number(output.dataset.outputMeasureToken) || 0) + 1);
    output.dataset.outputMeasureToken = token;
    var index = 0;
    var maxHeight = 0;

    function measureNextBatch() {
      if (!output.isConnected || output.dataset.outputMeasureToken !== token) return;
      if (output.classList.contains("has-current-numeric-values")) return;

      var end = Math.min(texts.length, index + OUTPUT_MEASURE_BATCH_SIZE);
      var fragment = document.createDocumentFragment();
      var measurers = texts.slice(index, end).map(function (text) {
        var measurer = output.cloneNode(false);
        measurer.removeAttribute("id");
        measurer.removeAttribute("data-current-output");
        measurer.removeAttribute("data-output-measure-token");
        measurer.removeAttribute("hidden");
        measurer.classList.add("evaluation-output-measurer");
        measurer.style.width = width + "px";
        measurer.textContent = text;
        fragment.appendChild(measurer);
        return measurer;
      });
      output.parentNode.appendChild(fragment);

      // One layout calculation covers the whole bounded batch instead of
      // materializing every sampled output in the DOM at the same time.
      measurers.forEach(function (measurer) {
        maxHeight = Math.max(maxHeight, measurer.scrollHeight);
      });
      measurers.forEach(function (measurer) {
        measurer.remove();
      });
      index = end;

      if (index < texts.length) {
        scheduleOutputMeasure(measureNextBatch);
        return;
      }
      if (maxHeight > 0 && output.dataset.outputMeasureToken === token) {
        output.style.setProperty("--lf3r-output-height", Math.ceil(maxHeight + 2) + "px");
        output.classList.add("is-stable-height");
      }
    }

    scheduleOutputMeasure(measureNextBatch);
  }

  function sizeEvaluationOutputs() {
    sizingScheduled = false;
    if (!isResultsView() || !window.state || !state.evaluation || !state.evaluation.methods) return;
    methods.querySelectorAll("[data-evaluation-method]").forEach(function (card) {
      if (card.classList.contains("is-collapsed")) return;
      var result = state.evaluation.methods[card.dataset.evaluationMethod];
      if (!result || !result.available) return;
      sizeEvaluationOutput(card, result);
    });
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
  }

  function refresh() {
    applyQueueState();
    scheduleOutputSizing();
  }

  queueToggle.addEventListener("click", function () {
    queueCollapsed = !queueCollapsed;
    try {
      localStorage.setItem("lf3r.review.queueCollapsed", String(queueCollapsed));
      localStorage.setItem("lf3r.results.queueCollapsed", String(queueCollapsed));
    } catch (_) {}
    refresh();
  });

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
  window.lf3rResultsLayoutRefresh = refresh;
  refresh();
})();
