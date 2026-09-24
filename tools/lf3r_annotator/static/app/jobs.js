"use strict";

/* Shared persistent-job orchestration. */

function persistentJobEndpoint(jobType, jobId) {
  var encoded = encodeURIComponent(jobId);
  if (jobType === "baseline") return "/api/baseline-jobs/" + encoded;
  if (jobType === "analysis") return "/api/analysis-jobs/" + encoded;
  return "/api/rollout-jobs/" + encoded;
}

function persistentJobLogEndpoint(jobType, jobId) {
  return persistentJobEndpoint(jobType, jobId) + "/log?tail=240";
}

function persistentJobProgress(job) {
  if (job.job_type === "baseline") {
    var baselineTotal = job.unique_selected_rollouts == null
      ? (job.selected_rollouts || 0) : job.unique_selected_rollouts;
    return (job.completed_jobs || 0) + "/" + baselineTotal
      + " complete / " + (job.failed_jobs || 0) + " failed";
  }
  if (job.job_type === "analysis") {
    return (job.selected_rollouts || 0) + " rollout(s) selected";
  }
  return (job.completed_rollouts || 0) + "/" + (job.expected_rollouts || job.requested_rollouts || 0)
    + " rollout(s) complete";
}

function persistentJobMethod(job) {
  if (job.job_type === "baseline") return job.baseline || "baseline";
  if (job.job_type === "analysis") return "Temporal analysis";
  return "OpenVLA rollout generation";
}

function persistentWorkerSummary(job) {
  if (job.job_type !== "baseline" || !job.parallel_workers || !job.worker_progress || !job.worker_progress.length) {
    return "";
  }
  return '<div class="persistent-job-workers" aria-label="Baseline worker progress">'
    + job.worker_progress.map(function (worker) {
      var total = worker.unique_count == null ? worker.requested_count : worker.unique_count;
      return '<span>W' + escapeHtml(worker.worker_index + 1)
        + ' · GPU ' + escapeHtml(worker.gpu)
        + ' · [' + escapeHtml(worker.start_index) + ',' + escapeHtml(worker.end_index) + ')'
        + ' · ' + escapeHtml((worker.completed_jobs || 0) + '/' + (total || 0))
        + ' · ' + escapeHtml(worker.status || "queued") + '</span>';
    }).join("") + '</div>';
}

function persistentJobScope(job) {
  if (job.job_type === "rollout_generation") {
    var suite = job.task_suite === "libero_spatial" ? "LIBERO-Spatial" : "LIBERO-10";
    return suite + " - " + (job.run_note || "");
  }
  if (job.job_type === "baseline" && job.scope
      && window.LF3RDatasetScopes
      && typeof window.LF3RDatasetScopes.scopeLabel === "function") {
    return window.LF3RDatasetScopes.scopeLabel(job.scope);
  }
  return job.scope || (job.rollout_id ? "single rollout: " + job.rollout_id : "unspecified scope");
}

function persistentJobClass(status) {
  return String(status || "unknown").replace(/[^A-Za-z0-9_-]/g, "-");
}

function latestPersistentJob(jobs) {
  return (jobs || []).slice().sort(function (left, right) {
    var leftTime = Date.parse(left.submitted_at || left.tmux_created_at || left.started_at || "") || 0;
    var rightTime = Date.parse(right.submitted_at || right.tmux_created_at || right.started_at || "") || 0;
    if (leftTime !== rightTime) return rightTime - leftTime;
    return String(right.job_id || "").localeCompare(String(left.job_id || ""));
  })[0] || null;
}

function renderPersistentJobCards(containerId, jobs, emptyMessage) {
  var container = byId(containerId);
  if (!container) return;
  if (!jobs || !jobs.length) {
    container.innerHTML = '<div class="job-list-empty">' + escapeHtml(emptyMessage || "No persistent jobs recorded.") + "</div>";
    return;
  }
  container.innerHTML = jobs.map(function (job) {
    var status = String(job.status || "unknown");
    var tmuxState = job.tmux_state || "unknown";
    var gpu = job.gpu ? "GPU " + job.gpu : "CPU";
    var interpreter = job.interpreter ? " / " + job.interpreter : "";
    var runRoot = job.run_root || job.output_dir || "";
    return '<article class="persistent-job-card" data-persistent-job="' + escapeHtml(job.job_id) + '">'
      + '<div class="persistent-job-heading"><strong>' + escapeHtml(persistentJobMethod(job)) + "</strong>"
      + '<span class="analysis-badge job-status-' + escapeHtml(persistentJobClass(status)) + '">' + escapeHtml(status) + "</span></div>"
      + '<div class="persistent-job-meta"><span>' + escapeHtml(persistentJobScope(job)) + "</span>"
      + "<span>" + escapeHtml(gpu) + "</span><span>" + escapeHtml(persistentJobProgress(job)) + "</span></div>"
      + persistentWorkerSummary(job)
      + '<div class="persistent-job-meta persistent-job-runtime"><span>tmux: ' + escapeHtml(job.tmux_session || "unavailable")
      + " / " + escapeHtml(tmuxState) + "</span><span>" + escapeHtml(runRoot) + escapeHtml(interpreter) + "</span></div>"
      + '<div class="persistent-job-actions"><button type="button" class="ghost-button" data-persistent-job-log="' + escapeHtml(job.job_id)
      + '" data-persistent-job-type="' + escapeHtml(job.job_type) + '" aria-expanded="false">View log</button>'
      + '<span class="persistent-job-id">' + escapeHtml(job.job_id) + "</span></div>"
      + '<pre class="job-card-log" data-persistent-job-log-output="' + escapeHtml(job.job_id) + '" hidden></pre></article>';
  }).join("");
  container.querySelectorAll("[data-persistent-job-log-output]").forEach(function (output) {
    var jobId = output.dataset.persistentJobLogOutput;
    if (!state.persistentJobLogOpen[jobId]) return;
    output.textContent = state.persistentJobLogText[jobId] || "";
    output.hidden = false;
    var card = output.closest(".persistent-job-card");
    var button = card && card.querySelector("[data-persistent-job-log]");
    if (button) {
      button.textContent = "Hide log";
      button.setAttribute("aria-expanded", "true");
    }
  });
}

function renderPersistentJobLists() {
  var jobs = state.persistentJobs || [];
  var baselineJobs = jobs.filter(function (job) { return job.job_type === "baseline"; });
  var runsJobs = window.LF3RRunsJobs;
  if (!runsJobs || typeof runsJobs.renderBaselineJobs !== "function"
      || runsJobs.renderBaselineJobs(baselineJobs) !== true) {
    var latestBaseline = latestPersistentJob(baselineJobs);
    renderPersistentJobCards(
      "baselineBatchJobs",
      latestBaseline ? [latestBaseline] : [],
      "No baseline jobs recorded. Older logs remain available on disk and through the job API."
    );
  }

  var generationJobs = jobs.filter(function (job) { return job.job_type === "rollout_generation"; });
  var latestGeneration = latestPersistentJob(generationJobs);
  renderPersistentJobCards(
    "rolloutGenerationJobs",
    latestGeneration ? [latestGeneration] : [],
    "No rollout-generation jobs recorded."
  );

  if (runsJobs && typeof runsJobs.afterRender === "function") {
    runsJobs.afterRender();
  }
  if (typeof window.lf3rWorkspaceJobsChanged === "function") {
    window.lf3rWorkspaceJobsChanged(jobs);
  }
}

function rememberPersistentJob(job) {
  if (!job || !job.job_id) return;
  var previous = (state.persistentJobs || []).find(function (item) {
    return item.job_id === job.job_id;
  }) || null;
  var replaced = false;
  state.persistentJobs = (state.persistentJobs || []).map(function (item) {
    if (item.job_id !== job.job_id) return item;
    replaced = true;
    return job;
  });
  if (!replaced) state.persistentJobs.push(job);
  if (job.job_type === "baseline") {
    state.baselineBatchJobs[job.job_id] = job;
    if (job.baseline_mode === "rollout" && state.baselineJob && state.baselineJob.job_id === job.job_id) {
      state.baselineJob = job;
    }
  } else if (job.job_type === "rollout_generation") {
    state.rolloutGenerationJobs[job.job_id] = job;
    if (state.rolloutGenerationJob && state.rolloutGenerationJob.job_id === job.job_id) {
      state.rolloutGenerationJob = job;
    }
  }
  renderPersistentJobLists();
  if (typeof window.lf3rWorkspaceJobChanged === "function") {
    window.lf3rWorkspaceJobChanged(job);
  }
  if (typeof resultsOnPersistentJobChanged === "function") {
    resultsOnPersistentJobChanged(previous, job);
  }
}

function activePersistentJob(job) {
  return job && (job.status === "queued" || job.status === "running");
}

async function loadPersistentJobLog(jobId, jobType, button) {
  try {
    var response = await fetch(persistentJobLogEndpoint(jobType, jobId), { cache: "no-store" });
    var payload = await response.json();
    var outputs = document.querySelectorAll("[data-persistent-job-log-output]");
    var output = null;
    outputs.forEach(function (node) {
      if (node.dataset.persistentJobLogOutput === jobId) output = node;
    });
    if (!response.ok) throw new Error(payload.error || "Could not read job log");
    if (output) {
      state.persistentJobLogText[jobId] = payload.log ? payload.log.text : "";
      state.persistentJobLogOpen[jobId] = true;
      output.textContent = state.persistentJobLogText[jobId];
      output.hidden = false;
    }
    if (button) {
      button.textContent = "Hide log";
      button.setAttribute("aria-expanded", "true");
    }
  } catch (error) {
    if (button) button.textContent = "Log error";
    var target = byId("baselineBatchStatus");
    if (jobType === "analysis") target = byId("analysisRunSelection") || target;
    if (jobType === "rollout_generation") target = byId("rolloutGenerationStatus") || target;
    if (target) target.textContent = "Job log error: " + error.message;
  }
}

async function pollPersistentJob(jobId) {
  if (state.persistentJobPollTimers[jobId]) return;
  var known = (state.persistentJobs || []).find(function (item) { return item.job_id === jobId; });
  if (!known || !activePersistentJob(known)) return;
  state.persistentJobPollTimers[jobId] = true;
  try {
    var response = await fetch(persistentJobEndpoint(known.job_type, jobId), { cache: "no-store" });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not read persistent job");
    var job = payload.job;
    rememberPersistentJob(job);
    if (job.job_type === "baseline" && job.baseline_mode === "batch") {
      await loadBaselineBatchLog(jobId);
      if (state.baselineBatchJob && state.baselineBatchJob.job_id === jobId) {
        if (activePersistentJob(job)) {
          setBaselineBatchStatus("Running " + job.baseline + " / " + persistentJobProgress(job) + " / " + job.status + "...", "");
        } else if (job.status === "complete") {
          setBaselineBatchStatus("Batch " + job.baseline + " complete: " + persistentJobProgress(job) + ".", "");
        } else {
          setBaselineBatchStatus("Batch " + job.baseline + " ended as " + job.status + "; inspect its log.", "error");
        }
      }
    } else if (job.job_type === "rollout_generation") {
      await loadRolloutGenerationLog(jobId);
      if (state.rolloutGenerationJob && state.rolloutGenerationJob.job_id === jobId) {
        setRolloutGenerationStatus(rolloutGenerationJobMessage(job), job.status === "failed" ? "error" : job.status === "memory_blocked" ? "warning" : "");
      }
    } else if (job.job_type === "analysis" && typeof window.lf3rWorkspaceJobChanged === "function") {
      window.lf3rWorkspaceJobChanged(job);
    }
    if (activePersistentJob(job)) {
      state.persistentJobPollTimers[jobId] = window.setTimeout(function () {
        delete state.persistentJobPollTimers[jobId];
        pollPersistentJob(jobId);
      }, 1500);
    } else {
      delete state.persistentJobPollTimers[jobId];
      if (job.status === "complete" && job.job_type === "rollout_generation") {
        loadRollouts(state.selectedId).catch(function () {});
      }
      if (job.status === "complete" && job.job_type === "baseline"
          && job.baseline_mode === "rollout" && job.rollout_id) {
        loadEvaluation(job.rollout_id);
      }
    }
  } catch (error) {
    delete state.persistentJobPollTimers[jobId];
    var status = byId("baselineBatchStatus");
    if (known.job_type === "analysis") status = byId("analysisRunSelection") || status;
    if (known.job_type === "rollout_generation") status = byId("rolloutGenerationStatus") || status;
    if (status) status.textContent = "Persistent job error: " + error.message;
    if (activePersistentJob(known)) {
      state.persistentJobPollTimers[jobId] = window.setTimeout(function () {
        delete state.persistentJobPollTimers[jobId];
        pollPersistentJob(jobId);
      }, 3000);
    }
  }
}

async function loadPersistentJobs() {
  try {
    var response = await fetch("/api/jobs", { cache: "no-store" });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not load persistent jobs");
    state.persistentJobs = payload.jobs || [];
    state.baselineBatchJobs = {};
    state.rolloutGenerationJobs = {};
    var baselineJobs = state.persistentJobs.filter(function (job) { return job.job_type === "baseline"; });
    var generationJobs = state.persistentJobs.filter(function (job) { return job.job_type === "rollout_generation"; });
    baselineJobs.forEach(function (job) { state.baselineBatchJobs[job.job_id] = job; });
    generationJobs.forEach(function (job) { state.rolloutGenerationJobs[job.job_id] = job; });
    state.baselineBatchJob = latestPersistentJob(baselineJobs);
    state.rolloutGenerationJob = latestPersistentJob(generationJobs);
    renderPersistentJobLists();
    state.persistentJobs.filter(activePersistentJob).forEach(function (job) {
      pollPersistentJob(job.job_id);
    });
  } catch (error) {
    var status = byId("baselineBatchStatus");
    if (status) status.textContent = "Persistent job list unavailable: " + error.message;
  }
}

async function pollBaselineJob(jobId, rolloutId) {
  return pollPersistentJob(jobId);
}
