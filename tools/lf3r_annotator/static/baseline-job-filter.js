"use strict";

(function installBaselineJobFilter() {
  if (!window.state || typeof window.lf3rRenderJobCards !== "function") return;

  var container = document.getElementById("baselineBatchJobs");
  if (!container) return;

  var originalRenderLists = window.renderPersistentJobLists;
  if (typeof originalRenderLists !== "function") return;

  var ACTIVE_STATUSES = ["queued", "running", "cancelling"];
  var FAILURE_STATUSES = ["failed", "memory_blocked", "cancelled", "complete_with_errors"];
  var filter = "active";
  try {
    filter = localStorage.getItem("lf3r.runs.baselineJobFilter") || "active";
  } catch (_) {}

  function timestamp(job) {
    return Date.parse(job.submitted_at || job.tmux_created_at || job.started_at || job.finished_at || "") || 0;
  }

  function sortNewest(jobs) {
    return jobs.slice().sort(function (left, right) {
      var delta = timestamp(right) - timestamp(left);
      return delta || String(right.job_id || "").localeCompare(String(left.job_id || ""));
    });
  }

  function matches(job) {
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

  function renderBaselineJobs() {
    var all = sortNewest((state.persistentJobs || []).filter(function (job) {
      return job.job_type === "baseline";
    }));
    var visible = all.filter(matches);
    window.lf3rRenderJobCards("baselineBatchJobs", visible, emptyMessage());
    container.classList.add("runs-filtered-job-list");

    var count = document.getElementById("baselineJobFilterCount");
    if (count) count.textContent = visible.length + " / " + all.length;
  }

  function renderListsWithBaselineFilter() {
    originalRenderLists();
    renderBaselineJobs();
  }

  window.renderPersistentJobLists = renderListsWithBaselineFilter;
  try { renderPersistentJobLists = renderListsWithBaselineFilter; } catch (_) {}

  var activity = container.closest(".runs-activity");
  var heading = activity && activity.querySelector(".runs-activity-heading");
  if (heading && !document.getElementById("baselineJobFilter")) {
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
    var valid = ["active", "all", "running", "queued", "complete", "problem"];
    if (valid.indexOf(filter) === -1) filter = "active";
    select.value = filter;
    select.addEventListener("change", function () {
      filter = select.value;
      try { localStorage.setItem("lf3r.runs.baselineJobFilter", filter); } catch (_) {}
      renderBaselineJobs();
    });
  }

  renderBaselineJobs();
})();
