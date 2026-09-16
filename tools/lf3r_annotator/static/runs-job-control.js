"use strict";

(function installRunsJobControls() {
  var view = document.getElementById("runsView");
  var jobs = document.getElementById("baselineBatchJobs");
  if (!view || !jobs || view.dataset.runsJobControls === "true") return;
  view.dataset.runsJobControls = "true";

  var cancelPending = Object.create(null);

  function cardStatus(card) {
    var badge = card && card.querySelector(".persistent-job-heading .analysis-badge");
    if (!badge) return "";
    return String(badge.textContent || "").trim().toLowerCase();
  }

  function setStatusMessage(message, kind) {
    var target = document.getElementById("baselineBatchStatus");
    if (!target) return;
    if (target.textContent !== message) target.textContent = message;
    target.classList.toggle("error", kind === "error");
  }

  function decorateCards() {
    jobs.querySelectorAll(".persistent-job-card").forEach(function (card) {
      var jobId = String(card.dataset.persistentJob || "");
      if (!jobId) return;
      var status = cardStatus(card);
      var active = status === "running" || status === "queued";
      if (!active) {
        delete cancelPending[jobId];
        var stale = card.querySelector("[data-baseline-job-cancel]");
        if (stale) stale.remove();
        return;
      }

      var actions = card.querySelector(".persistent-job-actions");
      if (!actions) return;
      var button = actions.querySelector("[data-baseline-job-cancel]");
      if (!button) {
        button = document.createElement("button");
        button.type = "button";
        button.className = "ghost-button job-cancel-button";
        button.dataset.baselineJobCancel = jobId;
        var idNode = actions.querySelector(".persistent-job-id");
        actions.insertBefore(button, idNode || null);
      }
      var pending = Boolean(cancelPending[jobId]);
      var label = pending ? "Cancelling…" : "Cancel job";
      var ariaLabel = pending ? "Cancellation requested" : "Cancel baseline job " + jobId;
      if (button.disabled !== pending) button.disabled = pending;
      if (button.textContent !== label) button.textContent = label;
      if (button.getAttribute("aria-label") !== ariaLabel) {
        button.setAttribute("aria-label", ariaLabel);
      }
    });
  }

  async function cancelJob(button) {
    var jobId = String(button.dataset.baselineJobCancel || "");
    if (!jobId || cancelPending[jobId]) return;
    var ok = window.confirm(
      "Cancel this baseline job? Completed rollout outputs will be kept, but remaining work will stop."
    );
    if (!ok) return;

    cancelPending[jobId] = true;
    decorateCards();
    setStatusMessage("Cancellation requested for " + jobId + ". Waiting for the tmux job to stop…", "");
    try {
      var response = await fetch(
        "/api/baseline-jobs/" + encodeURIComponent(jobId) + "/cancel",
        { method: "POST", cache: "no-store" }
      );
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not cancel baseline job");
      setStatusMessage("Cancelling " + jobId + ". Completed outputs are preserved.", "");
    } catch (error) {
      delete cancelPending[jobId];
      decorateCards();
      setStatusMessage("Cancel failed: " + error.message, "error");
    }
  }

  view.addEventListener("click", function (event) {
    var button = event.target.closest("[data-baseline-job-cancel]");
    if (!button || !view.contains(button)) return;
    event.preventDefault();
    event.stopPropagation();
    cancelJob(button);
  });

  // Persistent job rendering replaces the children of baselineBatchJobs as a
  // unit. Watching descendants is unnecessary and lets our own button/text
  // mutations retrigger the observer. Observe only direct list replacement.
  new MutationObserver(decorateCards).observe(jobs, { childList: true });
  decorateCards();
})();
