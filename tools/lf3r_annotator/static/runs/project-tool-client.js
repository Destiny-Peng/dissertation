"use strict";

window.LF3RProjectToolClient = (function createProjectToolClient() {
  var pollTimer = null;
  var activeJobId = null;
  var protocol = null;
  var lastRegistryRefreshAt = 0;
  var callbacks = {
    onJobUpdate: null,
    onJobsUpdated: null
  };

  function activityNodes() {
    return [
      {
        status: document.getElementById("projectToolStatus"),
        jobs: document.getElementById("projectToolJobs"),
        log: document.getElementById("projectToolLog")
      },
      {
        status: document.getElementById("projectToolStatusTools"),
        jobs: document.getElementById("projectToolJobsTools"),
        log: document.getElementById("projectToolLogTools")
      }
    ];
  }

  function setStatus(message) {
    activityNodes().forEach(function (nodes) {
      if (nodes.status) nodes.status.textContent = String(message || "");
    });
  }

  function renderJobs(jobs) {
    activityNodes().forEach(function (nodes) {
      if (!nodes.jobs) return;
      nodes.jobs.innerHTML = jobs.length ? jobs.slice(0, 5).map(function (job) {
        return '<button type="button" class="runs-tool-job'
          + (job.job_id === activeJobId ? ' active' : '')
          + '" data-tool-job="' + job.job_id + '"><strong>'
          + (job.tool_label || job.action || 'Project tool')
          + '</strong><span>' + (job.status || 'unknown') + '</span></button>';
      }).join("") : '<div class="runs-tool-job-empty">No project-tool jobs yet.</div>';
    });

    Array.prototype.slice.call(document.querySelectorAll("[data-tool-job]")).forEach(function (button) {
      button.addEventListener("click", function () {
        activeJobId = button.dataset.toolJob;
        refresh();
        poll();
      });
    });
  }

  async function notifyJobsUpdated(jobs) {
    if (typeof callbacks.onJobsUpdated !== "function") return;
    try {
      await callbacks.onJobsUpdated(jobs || []);
    } catch (error) {
      console.error("LF3R project-tool jobs callback failed", error);
    }
  }

  async function notifyJobUpdate(job, logText) {
    if (typeof callbacks.onJobUpdate !== "function") return;
    try {
      await callbacks.onJobUpdate(job || {}, String(logText || ""));
    } catch (error) {
      console.error("LF3R project-tool update callback failed", error);
    }
  }

  async function refresh() {
    try {
      var response = await fetch("/api/tool-jobs", { cache: "no-store" });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not read project-tool jobs");
      protocol = payload.project_tools_protocol || null;
      var jobs = payload.jobs || [];
      if (!activeJobId && jobs.length) activeJobId = jobs[0].job_id;
      renderJobs(jobs);
      lastRegistryRefreshAt = Date.now();
      await notifyJobsUpdated(jobs);
      return jobs;
    } catch (error) {
      return [];
    }
  }

  async function poll() {
    if (pollTimer) clearTimeout(pollTimer);
    if (!activeJobId) return;
    try {
      var responses = await Promise.all([
        fetch("/api/tool-jobs/" + encodeURIComponent(activeJobId), { cache: "no-store" }),
        fetch("/api/tool-jobs/" + encodeURIComponent(activeJobId) + "/log?tail=300", { cache: "no-store" })
      ]);
      var jobPayload = await responses[0].json();
      var logPayload = await responses[1].json();
      if (!responses[0].ok) throw new Error(jobPayload.error || "Could not read job");
      var job = jobPayload.job || {};
      var logText = logPayload.log && logPayload.log.text ? logPayload.log.text : "";

      activityNodes().forEach(function (nodes) {
        if (nodes.status) {
          nodes.status.textContent = (job.tool_label || job.action || "Project tool")
            + " · " + (job.status || "unknown")
            + (job.error ? " · " + job.error : "");
        }
        if (nodes.log) nodes.log.textContent = logText;
      });

      var active = job.status === "queued" || job.status === "running";
      Array.prototype.slice.call(document.querySelectorAll("[data-tool-job]")).forEach(function (button) {
        if (String(button.dataset.toolJob || "") !== String(job.job_id || "")) return;
        var status = button.querySelector("span");
        if (status) status.textContent = job.status || "unknown";
      });

      if (!active || Date.now() - lastRegistryRefreshAt >= 5000) {
        await refresh();
      }
      await notifyJobUpdate(job, logText);
      if (active) {
        pollTimer = setTimeout(poll, 1000);
      }
    } catch (error) {
      setStatus(String(error.message || error));
    }
  }

  function waitMs(milliseconds) {
    return new Promise(function (resolve) {
      window.setTimeout(resolve, milliseconds);
    });
  }

  function makeClientRequestId(action) {
    return "web:" + action + ":" + Date.now() + ":" + Math.random().toString(36).slice(2, 10);
  }

  async function recoverSubmission(action, clientRequestId) {
    for (var attempt = 0; attempt < 5; attempt += 1) {
      try {
        var response = await fetch("/api/tool-jobs", { cache: "no-store" });
        var payload = await response.json();
        if (response.ok) {
          protocol = payload.project_tools_protocol || null;
          var jobs = payload.jobs || [];
          var match = jobs.find(function (job) {
            return job
              && job.action === action
              && job.client_request_id === clientRequestId;
          });
          if (match) return match;
        }
      } catch (_) {}
      if (attempt < 4) await waitMs(250 * (attempt + 1));
    }
    return null;
  }

  async function submit(action, options) {
    setStatus("Submitting " + action + "…");

    var clientRequestId = makeClientRequestId(action);
    var controller = typeof AbortController === "function" ? new AbortController() : null;
    var timeoutId = window.setTimeout(function () {
      if (controller) controller.abort();
    }, 8000);
    var job = null;

    try {
      var response = await fetch("/api/tools/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: action,
          options: options || {},
          client_request_id: clientRequestId
        }),
        signal: controller ? controller.signal : undefined
      });
      var payload = await response.json();
      if (!response.ok) {
        var message = payload.error || "Project-tool request failed";
        setStatus(message);
        throw new Error(message);
      }
      job = payload.job;
    } catch (error) {
      job = await recoverSubmission(action, clientRequestId);
      if (!job) {
        var staleBackend = protocol !== "immediate-registry-v1";
        var message = staleBackend
          ? "Project-tool backend is stale or incompatible. Restart the LF3R WebUI server after git pull."
          : (error && error.name === "AbortError"
            ? "Project-tool submission timed out before a job could be confirmed."
            : "Project-tool submission failed: " + String(error.message || error));
        setStatus(message);
        throw new Error(message);
      }
      setStatus("Recovered submitted " + action + " job.");
    } finally {
      window.clearTimeout(timeoutId);
    }

    activeJobId = job.job_id;
    await refresh();
    poll();
    return job;
  }

  function install(options) {
    options = options || {};
    callbacks.onJobUpdate = typeof options.onJobUpdate === "function" ? options.onJobUpdate : null;
    callbacks.onJobsUpdated = typeof options.onJobsUpdated === "function" ? options.onJobsUpdated : null;
    return refresh().then(function () {
      if (activeJobId) poll();
    });
  }

  return {
    install: install,
    submit: submit,
    refresh: refresh,
    poll: poll,
    activeJobId: function () { return activeJobId; }
  };
})();
