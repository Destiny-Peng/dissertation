"use strict";

(function installRunsLogUi() {
  var view = document.getElementById("runsView");
  if (!view || view.dataset.runsLogUi === "true") return;
  view.dataset.runsLogUi = "true";

  var channels = {
    baseline: {
      jobs: document.getElementById("baselineBatchJobs"),
      log: document.getElementById("baselineBatchLog"),
      endpoint: function (jobId) {
        return "/api/baseline-jobs/" + encodeURIComponent(jobId) + "/log?tail=240";
      },
      requestSerial: 0
    },
    rollout_generation: {
      jobs: document.getElementById("rolloutGenerationJobs"),
      log: document.getElementById("rolloutGenerationLog"),
      endpoint: function (jobId) {
        return "/api/rollout-jobs/" + encodeURIComponent(jobId) + "/log?tail=240";
      },
      requestSerial: 0
    }
  };

  function activityFor(channel) {
    return channel && channel.log ? channel.log.closest(".runs-activity") : null;
  }

  function visibleJobId(channel) {
    if (!channel || !channel.log || channel.log.hidden) return "";
    return String(channel.log.dataset.visibleJobId || "");
  }

  function invalidateLogRequests(channel) {
    if (!channel) return 0;
    channel.requestSerial = (channel.requestSerial || 0) + 1;
    return channel.requestSerial;
  }

  function setLogOpen(channel, jobId, open) {
    if (!channel || !channel.log) return;
    channel.log.hidden = !open;
    channel.log.dataset.visibleJobId = open ? String(jobId || "") : "";
    var activity = activityFor(channel);
    if (activity) activity.classList.toggle("runs-log-open", Boolean(open));
    if (open) {
      channel.log.dataset.followTail = "true";
      window.requestAnimationFrame(function () {
        channel.log.scrollTop = channel.log.scrollHeight;
      });
    }
    syncButtons();
  }

  function closeAllInitialLogs() {
    Object.keys(channels).forEach(function (key) {
      var channel = channels[key];
      if (!channel.log) return;
      invalidateLogRequests(channel);
      setLogOpen(channel, "", false);
    });
  }

  function syncButtons() {
    Object.keys(channels).forEach(function (jobType) {
      var channel = channels[jobType];
      if (!channel.jobs || !channel.log) return;

      var currentJobId = visibleJobId(channel);
      var matchingVisibleButton = false;
      channel.jobs.querySelectorAll("[data-persistent-job-log]").forEach(function (button) {
        var jobId = String(button.dataset.persistentJobLog || "");
        var open = Boolean(currentJobId && jobId === currentJobId);
        if (open) matchingVisibleButton = true;
        if (button.textContent !== (open ? "Hide log" : "View log")) {
          button.textContent = open ? "Hide log" : "View log";
        }
        if (button.getAttribute("aria-expanded") !== String(open)) {
          button.setAttribute("aria-expanded", String(open));
        }
      });

      if (currentJobId && !matchingVisibleButton) {
        invalidateLogRequests(channel);
        channel.log.hidden = true;
        channel.log.dataset.visibleJobId = "";
        var activity = activityFor(channel);
        if (activity) activity.classList.remove("runs-log-open");
      }
    });

    view.querySelectorAll(".persistent-job-card .job-card-log").forEach(function (output) {
      if (!output.hidden) output.hidden = true;
    });
  }

  async function fetchLogText(channel, jobId) {
    var response = await fetch(channel.endpoint(jobId), { cache: "no-store" });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not read job log");
    return payload.log ? payload.log.text : "";
  }

  async function showLog(jobType, jobId, button) {
    var channel = channels[jobType];
    if (!channel || !channel.log) return;

    if (visibleJobId(channel) === jobId) {
      invalidateLogRequests(channel);
      setLogOpen(channel, jobId, false);
      return;
    }

    var requestSerial = invalidateLogRequests(channel);
    if (button) {
      button.disabled = true;
      button.textContent = "Loading…";
    }
    try {
      var text = await fetchLogText(channel, jobId);
      // A user can switch logs while this request is in flight. Never let a
      // late response from the old selection steal the shared log panel.
      if (requestSerial !== channel.requestSerial) return;
      channel.log.textContent = text;
      setLogOpen(channel, jobId, true);
    } catch (error) {
      if (requestSerial !== channel.requestSerial) return;
      channel.log.textContent = "Job log error: " + error.message;
      setLogOpen(channel, jobId, true);
    } finally {
      if (button) button.disabled = false;
      syncButtons();
    }
  }

  async function refreshSelectedLog(jobType, jobId) {
    var channel = channels[jobType];
    jobId = String(jobId || "");
    if (!channel || !channel.log || visibleJobId(channel) !== jobId) return;

    var requestSerial = invalidateLogRequests(channel);
    try {
      var text = await fetchLogText(channel, jobId);
      // Re-check both the request generation and selected job after await.
      // Multiple active jobs are polled concurrently, and their responses can
      // arrive in any order.
      if (requestSerial !== channel.requestSerial || visibleJobId(channel) !== jobId) return;
      channel.log.textContent = text;
    } catch (_error) {
      // Keep the last visible log text on a transient refresh failure. The job
      // status poll remains responsible for surfacing persistent errors.
    }
  }

  function installLegacyPollingBridge() {
    // app.js polls every active persistent job. Its legacy refresh functions
    // write the shared Runs <pre> for whichever job is globally "latest".
    // When several jobs run together that races with the user's View log
    // selection. Replace those writers so polling refreshes only the log that
    // is actually visible.
    var baselineRefresh = function (jobId) {
      return refreshSelectedLog("baseline", jobId);
    };
    var rolloutRefresh = function (jobId) {
      return refreshSelectedLog("rollout_generation", jobId);
    };

    window.loadBaselineBatchLog = baselineRefresh;
    window.loadRolloutGenerationLog = rolloutRefresh;
    try { loadBaselineBatchLog = baselineRefresh; } catch (_) {}
    try { loadRolloutGenerationLog = rolloutRefresh; } catch (_) {}
  }

  view.addEventListener("click", function (event) {
    var button = event.target.closest("[data-persistent-job-log]");
    if (!button || !view.contains(button)) return;
    var jobType = String(button.dataset.persistentJobType || "");
    if (!channels[jobType]) return;

    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    showLog(jobType, String(button.dataset.persistentJobLog || ""), button);
  }, true);

  Object.keys(channels).forEach(function (key) {
    var channel = channels[key];
    if (channel.log) {
      channel.log.addEventListener("scroll", function () {
        var distance = channel.log.scrollHeight - channel.log.scrollTop - channel.log.clientHeight;
        channel.log.dataset.followTail = distance < 48 ? "true" : "false";
      });

      // Log text can change inside the <pre>; this observer only updates
      // scrollTop, so it cannot create another DOM mutation.
      new MutationObserver(function () {
        if (channel.log.hidden || channel.log.dataset.followTail !== "true") return;
        window.requestAnimationFrame(function () {
          channel.log.scrollTop = channel.log.scrollHeight;
        });
      }).observe(channel.log, { childList: true, characterData: true, subtree: true });
    }

    // Job rendering replaces direct cards in each job-list container. Do not
    // observe the whole Runs subtree: syncButtons() itself edits button labels.
    if (channel.jobs) {
      new MutationObserver(syncButtons).observe(channel.jobs, { childList: true });
    }
  });

  installLegacyPollingBridge();
  closeAllInitialLogs();
  syncButtons();
})();
