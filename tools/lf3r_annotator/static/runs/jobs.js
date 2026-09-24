"use strict";

window.LF3RRunsJobs = (function createRunsJobsController() {
  var view = document.getElementById("runsView");
  var baselineContainer = document.getElementById("baselineBatchJobs");
  if (!view || !baselineContainer) {
    return {
      renderBaselineJobs: function () { return false; },
      afterRender: function () {},
      handleClick: function () { return false; },
      refreshSelectedLog: function () { return Promise.resolve(); }
    };
  }

  var ACTIVE_STATUSES = ["queued", "running", "cancelling"];
  var FAILURE_STATUSES = ["failed", "memory_blocked", "cancelled", "complete_with_errors"];
  var VALID_FILTERS = ["active", "all", "running", "queued", "complete", "problem"];
  var filter = "active";
  var cancelPending = Object.create(null);

  try {
    filter = localStorage.getItem("lf3r.runs.baselineJobFilter") || "active";
  } catch (_) {}
  if (VALID_FILTERS.indexOf(filter) === -1) filter = "active";

  var channels = {
    baseline: {
      jobs: baselineContainer,
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

  function timestamp(job) {
    return Date.parse(job.submitted_at || job.tmux_created_at || job.started_at || job.finished_at || "") || 0;
  }

  function sortNewest(jobs) {
    return (jobs || []).slice().sort(function (left, right) {
      var delta = timestamp(right) - timestamp(left);
      return delta || String(right.job_id || "").localeCompare(String(left.job_id || ""));
    });
  }

  function filterMatches(job) {
    var status = String(job.status || "unknown");
    if (filter === "all") return true;
    if (filter === "active") return ACTIVE_STATUSES.indexOf(status) !== -1;
    if (filter === "complete") return status === "complete";
    if (filter === "problem") return FAILURE_STATUSES.indexOf(status) !== -1;
    return status === filter;
  }

  function emptyMessage() {
    if (filter === "all") return "No baseline jobs recorded.";
    if (filter === "active") return "No active baseline jobs. Choose All to inspect history.";
    if (filter === "problem") return "No failed, cancelled, blocked, or partial baseline jobs.";
    return "No baseline jobs match this status filter.";
  }

  function renderBaselineJobs(jobs) {
    if (typeof window.lf3rRenderJobCards !== "function") return false;
    var all = sortNewest(jobs || []);
    var visible = all.filter(filterMatches);
    window.lf3rRenderJobCards("baselineBatchJobs", visible, emptyMessage());
    baselineContainer.classList.add("runs-filtered-job-list");

    var count = document.getElementById("baselineJobFilterCount");
    if (count) count.textContent = visible.length + " / " + all.length;
    return true;
  }

  function installFilterToolbar() {
    var activity = baselineContainer.closest(".runs-activity");
    var heading = activity && activity.querySelector(".runs-activity-heading");
    if (!heading || document.getElementById("baselineJobFilter")) return;

    var toolbar = document.createElement("div");
    toolbar.className = "runs-job-filter";
    toolbar.innerHTML = [
      '<label for="baselineJobFilter"><span>Show</span>',
      '<select id="baselineJobFilter">',
      '<option value="active">Active</option>',
      '<option value="all">All</option>',
      '<option value="running">Running</option>',
      '<option value="queued">Queued</option>',
      '<option value="complete">Complete</option>',
      '<option value="problem">Failed / cancelled</option>',
      '</select></label>',
      '<span id="baselineJobFilterCount" class="runs-job-filter-count">0 / 0</span>'
    ].join("");
    heading.appendChild(toolbar);

    var select = document.getElementById("baselineJobFilter");
    select.value = filter;
    select.addEventListener("change", function () {
      filter = select.value;
      try { localStorage.setItem("lf3r.runs.baselineJobFilter", filter); } catch (_) {}
      if (typeof window.renderPersistentJobLists === "function") {
        window.renderPersistentJobLists();
      } else if (typeof renderPersistentJobLists === "function") {
        renderPersistentJobLists();
      }
    });
  }

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

  function followTail(channel) {
    if (!channel || !channel.log || channel.log.hidden) return;
    if (channel.log.dataset.followTail !== "true") return;
    window.requestAnimationFrame(function () {
      channel.log.scrollTop = channel.log.scrollHeight;
    });
  }

  function setLogOpen(channel, jobId, open) {
    if (!channel || !channel.log) return;
    channel.log.hidden = !open;
    channel.log.dataset.visibleJobId = open ? String(jobId || "") : "";
    var activity = activityFor(channel);
    if (activity) activity.classList.toggle("runs-log-open", Boolean(open));
    if (open) {
      channel.log.dataset.followTail = "true";
      followTail(channel);
    }
    syncLogButtons();
  }

  function closeInitialLogs() {
    Object.keys(channels).forEach(function (key) {
      var channel = channels[key];
      if (!channel.log) return;
      invalidateLogRequests(channel);
      setLogOpen(channel, "", false);
      channel.log.addEventListener("scroll", function () {
        var distance = channel.log.scrollHeight - channel.log.scrollTop - channel.log.clientHeight;
        channel.log.dataset.followTail = distance < 48 ? "true" : "false";
      });
    });
  }

  function syncLogButtons() {
    Object.keys(channels).forEach(function (jobType) {
      var channel = channels[jobType];
      if (!channel.jobs || !channel.log) return;

      var currentJobId = visibleJobId(channel);
      var matchingVisibleButton = false;
      channel.jobs.querySelectorAll("[data-persistent-job-log]").forEach(function (button) {
        var jobId = String(button.dataset.persistentJobLog || "");
        var open = Boolean(currentJobId && jobId === currentJobId);
        if (open) matchingVisibleButton = true;
        button.textContent = open ? "Hide log" : "View log";
        button.setAttribute("aria-expanded", String(open));
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
      output.hidden = true;
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
      var logText = await fetchLogText(channel, jobId);
      if (requestSerial !== channel.requestSerial) return;
      channel.log.textContent = logText;
      setLogOpen(channel, jobId, true);
    } catch (error) {
      if (requestSerial !== channel.requestSerial) return;
      channel.log.textContent = "Job log error: " + error.message;
      setLogOpen(channel, jobId, true);
    } finally {
      if (button) button.disabled = false;
      syncLogButtons();
    }
  }

  async function refreshSelectedLog(jobType, jobId) {
    var channel = channels[jobType];
    jobId = String(jobId || "");
    if (!channel || !channel.log || visibleJobId(channel) !== jobId) return;

    var requestSerial = invalidateLogRequests(channel);
    try {
      var logText = await fetchLogText(channel, jobId);
      if (requestSerial !== channel.requestSerial || visibleJobId(channel) !== jobId) return;
      channel.log.textContent = logText;
      followTail(channel);
    } catch (_error) {
      // Keep the last visible text on a transient polling failure.
    }
  }

  function jobById(jobId) {
    return (state.persistentJobs || []).find(function (job) {
      return String(job.job_id || "") === String(jobId || "");
    }) || null;
  }

  function decorateCancelButtons() {
    baselineContainer.querySelectorAll(".persistent-job-card").forEach(function (card) {
      var jobId = String(card.dataset.persistentJob || "");
      if (!jobId) return;
      var job = jobById(jobId);
      var status = String(job && job.status || "").toLowerCase();
      var active = ACTIVE_STATUSES.indexOf(status) !== -1;
      var actions = card.querySelector(".persistent-job-actions");
      if (!actions) return;

      var button = actions.querySelector("[data-baseline-job-cancel]");
      if (!active) {
        delete cancelPending[jobId];
        if (button) button.remove();
        return;
      }

      if (!button) {
        button = document.createElement("button");
        button.type = "button";
        button.className = "ghost-button job-cancel-button";
        button.dataset.baselineJobCancel = jobId;
        var idNode = actions.querySelector(".persistent-job-id");
        actions.insertBefore(button, idNode || null);
      }

      var pending = Boolean(cancelPending[jobId]) || status === "cancelling";
      button.disabled = pending;
      button.textContent = pending ? "Cancelling…" : "Cancel job";
      button.setAttribute(
        "aria-label",
        pending ? "Cancellation requested" : "Cancel baseline job " + jobId
      );
    });
  }

  function setStatusMessage(message, kind) {
    var target = document.getElementById("baselineBatchStatus");
    if (!target) return;
    if (target.textContent !== message) target.textContent = message;
    target.classList.toggle("error", kind === "error");
  }

  async function cancelJob(button) {
    var jobId = String(button.dataset.baselineJobCancel || "");
    if (!jobId || cancelPending[jobId]) return;
    if (!window.confirm(
      "Cancel this baseline job? Completed rollout outputs will be kept, but remaining work will stop."
    )) return;

    cancelPending[jobId] = true;
    decorateCancelButtons();
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
      decorateCancelButtons();
      setStatusMessage("Cancel failed: " + error.message, "error");
    }
  }

  function handleClick(event) {
    var target = event.target;
    var cancelButton = target && target.closest
      ? target.closest("[data-baseline-job-cancel]")
      : null;
    if (cancelButton && view.contains(cancelButton)) {
      event.preventDefault();
      cancelJob(cancelButton);
      return true;
    }

    var logButton = target && target.closest
      ? target.closest("[data-persistent-job-log]")
      : null;
    if (!logButton || !view.contains(logButton)) return false;
    var jobType = String(logButton.dataset.persistentJobType || "");
    if (!channels[jobType]) return false;

    event.preventDefault();
    showLog(jobType, String(logButton.dataset.persistentJobLog || ""), logButton);
    return true;
  }

  function afterRender() {
    decorateCancelButtons();
    syncLogButtons();
  }

  installFilterToolbar();
  closeInitialLogs();

  window.setTimeout(function () {
    if (typeof window.renderPersistentJobLists === "function") {
      window.renderPersistentJobLists();
    } else if (typeof renderPersistentJobLists === "function") {
      renderPersistentJobLists();
    }
  }, 0);

  return {
    renderBaselineJobs: renderBaselineJobs,
    afterRender: afterRender,
    handleClick: handleClick,
    refreshSelectedLog: refreshSelectedLog
  };
})();
