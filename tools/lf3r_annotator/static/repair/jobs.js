"use strict";

window.LF3RRepairJobs = (function createRepairJobsController() {
  var ACTIVE_STATUSES = ["queued", "running", "cancelling"];
  var PROBLEM_STATUSES = ["failed", "memory_blocked", "cancelled", "complete_with_errors"];
  var VALID_FILTERS = ["active", "all", "running", "queued", "complete", "problem"];
  var jobs = [];
  var filter = "active";
  var visibleLogId = "";
  var logRequest = 0;
  var refreshTimer = null;
  var installed = false;

  try {
    filter = localStorage.getItem("lf3r.repair.jobFilter") || "active";
  } catch (_) {}
  if (VALID_FILTERS.indexOf(filter) === -1) filter = "active";

  function node(id) { return document.getElementById(id); }

  function timestamp(job) {
    return Date.parse(
      job.submitted_at || job.tmux_created_at || job.started_at || job.finished_at || ""
    ) || 0;
  }

  function sortNewest(rows) {
    return (rows || []).slice().sort(function (left, right) {
      var delta = timestamp(right) - timestamp(left);
      return delta || String(right.job_id || "").localeCompare(String(left.job_id || ""));
    });
  }

  function matches(job) {
    var status = String(job.status || "unknown");
    if (filter === "all") return true;
    if (filter === "active") return ACTIVE_STATUSES.indexOf(status) !== -1;
    if (filter === "complete") return status === "complete";
    if (filter === "problem") return PROBLEM_STATUSES.indexOf(status) !== -1;
    return status === filter;
  }

  function emptyMessage() {
    if (filter === "all") return "No Repair jobs recorded.";
    if (filter === "active") return "No active Repair jobs. Choose All or Complete to inspect history.";
    if (filter === "complete") return "No completed Repair jobs.";
    if (filter === "problem") return "No failed, cancelled, blocked, or partial Repair jobs.";
    return "No Repair jobs match this status filter.";
  }

  function activeJob() {
    return jobs.find(function (job) {
      return ACTIVE_STATUSES.indexOf(String(job.status || "")) !== -1;
    }) || null;
  }

  function renderStatus() {
    var target = node("repairJobStatus");
    if (!target) return;
    var job = activeJob();
    if (!job) {
      target.textContent = jobs.length
        ? "No active Repair job. Historical jobs are available below."
        : "No Repair jobs recorded.";
      return;
    }
    var progress = Math.round(Math.max(0, Math.min(1, Number(job.progress || 0))) * 100);
    target.textContent = String(job.status || "queued")
      + " · " + String(job.phase || "queued")
      + " · " + progress + "% · " + String(job.run_id || job.job_id || "");
  }

  function syncLogButtons() {
    var list = node("repairJobList");
    if (!list) return;
    var matching = false;
    list.querySelectorAll("[data-persistent-job-log]").forEach(function (button) {
      var jobId = String(button.dataset.persistentJobLog || "");
      var open = Boolean(visibleLogId && jobId === visibleLogId);
      if (open) matching = true;
      button.textContent = open ? "Hide log" : "View log";
      button.setAttribute("aria-expanded", String(open));
    });
    if (visibleLogId && !matching) closeLog();
    list.querySelectorAll("[data-persistent-job-log-output]").forEach(function (output) {
      output.hidden = true;
    });
  }

  function render(rows) {
    install();
    jobs = sortNewest((rows || []).filter(function (job) {
      return String(job.job_type || "") === "repair_synthetic_suffix";
    }));
    var visible = jobs.filter(matches);
    if (typeof window.lf3rRenderJobCards === "function") {
      window.lf3rRenderJobCards("repairJobList", visible, emptyMessage());
    } else {
      var list = node("repairJobList");
      if (list) list.textContent = visible.length
        ? visible.length + " Repair job(s)"
        : emptyMessage();
    }
    var listNode = node("repairJobList");
    if (listNode) listNode.classList.add("runs-filtered-job-list");
    var count = node("repairJobFilterCount");
    if (count) count.textContent = visible.length + " / " + jobs.length;
    var select = node("repairJobFilter");
    if (select) select.value = filter;
    renderStatus();
    syncLogButtons();
  }

  function closeLog() {
    logRequest += 1;
    visibleLogId = "";
    var output = node("repairJobLog");
    if (output) {
      output.hidden = true;
      output.dataset.visibleJobId = "";
    }
    var activity = output && output.closest(".repair-activity");
    if (activity) activity.classList.remove("runs-log-open");
    syncLogButtons();
  }

  function followTail() {
    var output = node("repairJobLog");
    if (!output || output.hidden || output.dataset.followTail !== "true") return;
    window.requestAnimationFrame(function () {
      output.scrollTop = output.scrollHeight;
    });
  }

  async function showLog(jobId, button) {
    var output = node("repairJobLog");
    if (!output || !jobId) return;
    if (visibleLogId === jobId && !output.hidden) {
      closeLog();
      return;
    }
    var serial = ++logRequest;
    if (button) {
      button.disabled = true;
      button.textContent = "Loading…";
    }
    try {
      var response = await fetch(
        "/api/repair/synthetic-suffix/jobs/" + encodeURIComponent(jobId) + "/log?tail=240",
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not read Repair job log");
      if (serial !== logRequest) return;
      visibleLogId = jobId;
      output.textContent = payload.log ? payload.log.text || "" : "";
      output.hidden = false;
      output.dataset.visibleJobId = jobId;
      output.dataset.followTail = "true";
      var activity = output.closest(".repair-activity");
      if (activity) activity.classList.add("runs-log-open");
      followTail();
    } catch (error) {
      if (serial !== logRequest) return;
      visibleLogId = jobId;
      output.textContent = "Job log error: " + String(error.message || error);
      output.hidden = false;
      output.dataset.visibleJobId = jobId;
    } finally {
      if (button) button.disabled = false;
      syncLogButtons();
    }
  }

  async function refreshVisibleLog() {
    if (!visibleLogId) return;
    var jobId = visibleLogId;
    try {
      var response = await fetch(
        "/api/repair/synthetic-suffix/jobs/" + encodeURIComponent(jobId) + "/log?tail=240",
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (!response.ok || visibleLogId !== jobId) return;
      var output = node("repairJobLog");
      if (!output || output.hidden) return;
      output.textContent = payload.log ? payload.log.text || "" : "";
      followTail();
    } catch (_) {}
  }

  async function refresh() {
    if (typeof window.lf3rRefreshPersistentJobs === "function") {
      await window.lf3rRefreshPersistentJobs();
      if (typeof window.lf3rGetPersistentJobs === "function") {
        render(window.lf3rGetPersistentJobs("repair_synthetic_suffix"));
      }
      await refreshVisibleLog();
      return;
    }
    try {
      var response = await fetch(
        "/api/jobs?job_type=repair_synthetic_suffix",
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not load Repair jobs");
      render(payload.jobs || []);
      await refreshVisibleLog();
    } catch (error) {
      var status = node("repairJobStatus");
      if (status) status.textContent = "Repair job list unavailable: " + String(error.message || error);
    }
  }

  function scheduleRefresh() {
    if (refreshTimer) window.clearTimeout(refreshTimer);
    refreshTimer = window.setTimeout(async function tick() {
      refreshTimer = null;
      if (document.body.dataset.view !== "repair") return;
      await refresh();
      scheduleRefresh();
    }, 1800);
  }

  function install() {
    if (installed) return true;
    var select = node("repairJobFilter");
    var list = node("repairJobList");
    var output = node("repairJobLog");
    if (!select || !list || !output) return false;
    installed = true;
    select.value = filter;
    select.addEventListener("change", function () {
      filter = VALID_FILTERS.indexOf(select.value) >= 0 ? select.value : "active";
      try { localStorage.setItem("lf3r.repair.jobFilter", filter); } catch (_) {}
      render(jobs);
    });
    list.addEventListener("click", function (event) {
      var button = event.target && event.target.closest
        ? event.target.closest("[data-persistent-job-log]")
        : null;
      if (!button || !list.contains(button)) return;
      event.preventDefault();
      event.stopPropagation();
      showLog(String(button.dataset.persistentJobLog || ""), button);
    }, true);
    output.addEventListener("scroll", function () {
      var distance = output.scrollHeight - output.scrollTop - output.clientHeight;
      output.dataset.followTail = distance < 48 ? "true" : "false";
    });
    var refreshButton = node("repairRefreshRuns");
    if (refreshButton) refreshButton.addEventListener("click", refresh);
    return true;
  }

  window.addEventListener("lf3r:viewchange", function (event) {
    if (!event.detail || event.detail.view !== "repair") {
      if (refreshTimer) {
        window.clearTimeout(refreshTimer);
        refreshTimer = null;
      }
      return;
    }
    install();
    refresh();
    scheduleRefresh();
  });

  window.setTimeout(function () {
    if (!install()) return;
    if (typeof window.lf3rGetPersistentJobs === "function") {
      render(window.lf3rGetPersistentJobs("repair_synthetic_suffix"));
    }
    if (document.body.dataset.view === "repair") {
      refresh();
      scheduleRefresh();
    }
  }, 0);

  return {
    render: render,
    refresh: refresh,
    closeLog: closeLog
  };
})();