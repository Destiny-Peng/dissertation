"use strict";

(function installReviewLayoutEnhancements() {
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