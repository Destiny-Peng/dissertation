"use strict";

var state = {
  rollouts: [],
  manifests: [],
  manifestFilter: "all",
  filtered: [],
  selectedId: null,
  currentFrame: 0,
  failureEvents: [],
  activeFailureEvent: null,
  dirty: false,
  evaluation: null,
  instructionCondition: "full_instruction",
  evaluationRequest: 0,
  evaluationSignalVisibility: {},
  baselineCollapsed: {},
  roboPosthocCheckpoint: "",
  baselineRuns: null,
  baselineRunsCondition: null,
  baselineRunsLoading: null,
  baselineRunSelections: {},
  baselineRunAll: {},
  baselineRunNotice: "",
  baselineBatchCoverage: null,
  baselineBatchCoverageKey: null,
  baselineBatchCoverageRequest: 0,
  baselineBatchCoverageLoading: false,
  baselineJob: null,
  baselineBatchJob: null,
  baselineBatchJobs: {},
  baselineBatchWorkers: [{ gpu: "0", start_index: 0, end_index: 0 }],
  rolloutGenerationJob: null,
  rolloutGenerationJobs: {},
  persistentJobs: [],
  persistentJobLogOpen: {},
  persistentJobLogText: {},
  persistentJobPollTimers: {},
  baselineBatchSubmitting: false,
  rolloutGenerationSubmitting: false
};

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}


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

var cliHelpState = {
  entries: null,
  popover: null,
  target: null,
  hideTimer: null
};

function cliHelpEntry(key) {
  if (!cliHelpState.entries) return null;
  var parts = String(key || "").split(".");
  var value = cliHelpState.entries;
  parts.forEach(function (part) {
    if (value) value = value[part];
  });
  return value && typeof value === "object" ? value : null;
}

function loadCliHelpMetadata() {
  fetch("/static/parameter_help.json", { cache: "no-store" })
    .then(function (response) {
      if (!response.ok) throw new Error("parameter help unavailable");
      return response.json();
    })
    .then(function (payload) {
      cliHelpState.entries = payload || {};
      if (cliHelpState.target) showCliHelp(cliHelpState.target);
    })
    .catch(function () {
      cliHelpState.entries = {};
    });
}

function ensureCliHelpPopover() {
  if (cliHelpState.popover) return cliHelpState.popover;
  var popover = document.createElement("div");
  popover.id = "cliHelpPopover";
  popover.className = "cli-help-popover";
  popover.setAttribute("role", "tooltip");
  popover.hidden = true;
  document.body.appendChild(popover);
  cliHelpState.popover = popover;
  return popover;
}

function positionCliHelp(target) {
  var popover = cliHelpState.popover;
  if (!popover || !target || popover.hidden) return;
  var rect = target.getBoundingClientRect();
  var margin = 8;
  var left = Math.max(margin, Math.min(rect.left, window.innerWidth - popover.offsetWidth - margin));
  var top = rect.bottom + margin;
  if (top + popover.offsetHeight > window.innerHeight - margin) {
    top = rect.top - popover.offsetHeight - margin;
  }
  popover.style.left = Math.round(Math.max(margin, left)) + "px";
  popover.style.top = Math.round(Math.max(margin, top)) + "px";
}

function showCliHelp(target) {
  if (!target || !target.dataset.cliHelp) return;
  if (cliHelpState.hideTimer) {
    window.clearTimeout(cliHelpState.hideTimer);
    cliHelpState.hideTimer = null;
  }
  var popover = ensureCliHelpPopover();
  var key = target.dataset.cliHelp;
  var entry = cliHelpEntry(key);
  var lines = entry ? [
    "CLI: " + (entry.cli || "shared setting"),
    "Default: " + (entry.default == null ? "unset" : entry.default),
    "Applies to: " + (entry.applies_to || "web UI"),
    entry.forwarded_as ? "Forwarded as: " + entry.forwarded_as : "",
    entry.aliases && entry.aliases.length ? "Aliases: " + entry.aliases.join(", ") : "",
    entry.description || ""
  ] : ["Parameter help: " + key, "Loading CLI metadata..."];
  popover.textContent = lines.filter(Boolean).join("\n");
  popover.hidden = false;
  cliHelpState.target = target;
  target.classList.add("cli-help-target");
  target.setAttribute("aria-describedby", "cliHelpPopover");
  positionCliHelp(target);
}

function hideCliHelp() {
  if (cliHelpState.hideTimer) {
    window.clearTimeout(cliHelpState.hideTimer);
    cliHelpState.hideTimer = null;
  }
  if (cliHelpState.target) {
    cliHelpState.target.classList.remove("cli-help-target");
    if (cliHelpState.target.getAttribute("aria-describedby") === "cliHelpPopover") {
      cliHelpState.target.removeAttribute("aria-describedby");
    }
  }
  if (cliHelpState.popover) cliHelpState.popover.hidden = true;
  cliHelpState.target = null;
}

function scheduleHideCliHelp() {
  if (cliHelpState.hideTimer) window.clearTimeout(cliHelpState.hideTimer);
  cliHelpState.hideTimer = window.setTimeout(hideCliHelp, 80);
}

function cliHelpTarget(event) {
  var node = event && event.target;
  return node && node.closest ? node.closest("[data-cli-help]") : null;
}

function selectedRollout() {
  return state.rollouts.find(function (record) {
    return record.id === state.selectedId;
  }) || null;
}
function effectiveOutcome(record) {
  return record.annotation && record.annotation.outcome_label
    ? record.annotation.outcome_label
    : record.ground_truth_outcome;
}


function isControlled(record) {
  return record.source_kind === "controlled_injected";
}

function isRealRobotManifest(record) {
  return record.source_kind === "real_robot"
    || record.analysis_partition === "real_robot_analysis";
}

function isExternalManifest(record) {
  return record.manifest_primary === false || isRealRobotManifest(record);
}

function provenanceClass(record) {
  if (isControlled(record)) return "controlled";
  if (isExternalManifest(record)) return "external";
  return "natural";
}

function provenanceLabel(record) {
  if (isControlled(record)) return "controlled";
  if (isRealRobotManifest(record)) return "real robot";
  if (isExternalManifest(record)) return "secondary";
  return "natural";
}

function populateManifestFilter(manifests) {
  var select = byId("manifestFilter");
  if (!select) return;
  var current = state.manifestFilter || select.value || "all";
  var options = ['<option value="all">All manifests</option>'];
  (manifests || []).forEach(function (manifest) {
    var path = String(manifest.path || "");
    if (!path) return;
    var label = manifest.label && manifest.label !== path
      ? String(manifest.label) + " · " + path
      : path;
    if (manifest.rollouts != null) label += " (" + manifest.rollouts + ")";
    options.push('<option value="' + escapeHtml(path) + '">' + escapeHtml(label) + "</option>");
  });
  select.innerHTML = options.join("");
  var valid = current === "all" || (manifests || []).some(function (manifest) {
    return manifest.path === current;
  });
  select.value = valid ? current : "all";
  state.manifestFilter = select.value;
}

function labelFor(value) {
  return String(value || "").replace(/_/g, " ");
}

function badge(value, cssClass) {
  return '<span class="badge ' + escapeHtml(cssClass || "") + '">' + escapeHtml(labelFor(value).toUpperCase()) + "</span>";
}

async function loadRollouts(preferredId) {
  var response = await fetch("/api/rollouts", { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Could not load rollout manifest");
  }
  var payload = await response.json();
  state.rollouts = payload.rollouts || [];
  state.manifests = payload.manifests || [];
  populateManifestFilter(state.manifests);
  byId("datasetStatus").textContent = "Dataset online · " + state.rollouts.length
    + " rollouts · " + (state.manifests.length || 1) + " manifest"
    + ((state.manifests.length || 1) === 1 ? "" : "s");
  updateProgress();
  applyFilters();
  if (typeof updateBaselineBatchAdvancedFields === "function") updateBaselineBatchAdvancedFields();
  var target = preferredId || state.selectedId;
  if (target && state.rollouts.some(function (record) { return record.id === target; })) {
    selectRollout(target);
  } else if (state.filtered.length) {
    selectRollout(state.filtered[0].id);
  }
  if (typeof window.lf3rWorkspaceDataChanged === "function") window.lf3rWorkspaceDataChanged();
}

function updateProgress() {
  var complete = state.rollouts.filter(function (record) {
    return record.annotation_status === "complete";
  }).length;
  byId("progressSummary").textContent = complete + " / " + state.rollouts.length + " reviewed";
}

function applyFilters() {
  var query = byId("searchInput").value.trim().toLowerCase();
  var origin = byId("originFilter").value;
  var manifest = byId("manifestFilter").value;
  state.manifestFilter = manifest;
  var outcome = byId("outcomeFilter").value;
  var review = byId("reviewFilter").value;
  state.filtered = state.rollouts.filter(function (record) {
    var haystack = [record.id, record.task_description, record.task_suite, record.task_id, record.manifest_source, record.manifest_label].join(" ").toLowerCase();
    return (!query || haystack.indexOf(query) !== -1)
      && (origin === "all" || record.source_kind === origin)
      && (manifest === "all" || record.manifest_source === manifest)
      && (outcome === "all" || effectiveOutcome(record) === outcome)
      && (review === "all" || record.annotation_status === review);
  });
  byId("visibleCount").textContent = state.filtered.length;
  renderRolloutList();
}

function renderRolloutList() {
  var container = byId("rolloutList");
  if (!state.filtered.length) {
    container.innerHTML = '<div class="empty-state"><p>No rollouts match these filters.</p></div>';
    return;
  }
  container.innerHTML = state.filtered.map(function (record) {
    var originClass = provenanceClass(record);
    var selectedClass = record.id === state.selectedId ? " active" : "";
    var title = record.task_description || (record.task_suite + " task " + record.task_id);
    var sourceLabel = record.manifest_label || record.manifest_source || "manifest";
    return '<button class="rollout-card ' + originClass + selectedClass + '" data-rollout-id="' + escapeHtml(record.id) + '" type="button">'
      + '<div class="badge-row">'
      + badge(provenanceLabel(record), originClass)
      + badge(effectiveOutcome(record), effectiveOutcome(record))
      + badge(record.annotation_status, record.annotation_status)
      + "</div>"
      + '<div class="card-title">' + escapeHtml(title) + "</div>"
      + '<div class="card-footer"><span>' + escapeHtml(record.task_suite) + " · task " + escapeHtml(record.task_id) + " · " + escapeHtml(sourceLabel) + '</span><span>' + escapeHtml(record.total_frames) + "f</span></div>"
      + "</button>";
  }).join("");
  container.querySelectorAll("[data-rollout-id]").forEach(function (button) {
    button.addEventListener("click", function () {
      maybeSelectRollout(button.dataset.rolloutId);
    });
  });
}

function maybeSelectRollout(id) {
  if (state.dirty && !window.confirm("Discard unsaved annotation changes?")) {
    return;
  }
  selectRollout(id);
}

var INSTRUCTION_CONDITION_ORDER = ["full_instruction", "subtask_a", "subtask_b"];

function instructionConditionLabel(condition, option) {
  if (option && option.label) return option.label;
  if (condition === "full_instruction") return "Full instruction";
  if (condition === "subtask_a") return "A";
  if (condition === "subtask_b") return "B";
  return condition;
}

function currentInstructionVariant(record) {
  var options = record && record.instruction_variants ? record.instruction_variants : {};
  var condition = state.instructionCondition;
  if (!options[condition]) condition = "full_instruction";
  state.instructionCondition = condition;
  return options[condition] || {
    condition: "full_instruction",
    label: "Full instruction",
    instruction: record ? record.task_description : "",
    available: true,
    counterfactual: false
  };
}

function updateInstructionVariantHeader(record) {
  var variant = currentInstructionVariant(record);
  var condition = variant.condition || state.instructionCondition;
  var instruction = variant.instruction || record.task_description || (record.task_suite + " task " + record.task_id);
  byId("taskTitle").textContent = instruction;
  byId("recordMeta").textContent = record.id
    + " - task " + record.task_id
    + " - episode " + record.episode_index
    + " - " + record.fps + " fps"
    + " - " + instructionConditionLabel(condition, variant);
  var note;
  if (condition === "full_instruction") {
    note = "Original full instruction. Existing completed baseline outputs are classified here.";
  } else if (variant.available) {
    note = "Counterfactual " + instructionConditionLabel(condition, variant)
      + " label on the same video and annotation. No baseline output has been run for this condition yet.";
  } else {
    note = "This rollout has no validated " + instructionConditionLabel(condition, variant) + " instruction variant.";
  }
  byId("instructionVariantNote").textContent = note;
}

function renderInstructionVariantControl(record) {
  var select = byId("instructionCondition");
  var options = record && record.instruction_variants ? record.instruction_variants : {};
  var conditions = INSTRUCTION_CONDITION_ORDER.filter(function (condition) {
    return Boolean(options[condition]);
  });
  if (!conditions.length) conditions = ["full_instruction"];
  var current = conditions.indexOf(state.instructionCondition) >= 0
    ? state.instructionCondition
    : "full_instruction";
  state.instructionCondition = current;
  select.innerHTML = conditions.map(function (condition) {
    var option = options[condition] || { condition: condition };
    var label = instructionConditionLabel(condition, option);
    var suffix = condition === "full_instruction" ? "" : " - counterfactual";
    return '<option value="' + escapeHtml(condition) + '">'
      + escapeHtml(label + suffix) + "</option>";
  }).join("");
  select.value = current;
  updateInstructionVariantHeader(record);
}

function selectRollout(id) {
  var record = state.rollouts.find(function (item) { return item.id === id; });
  if (!record) return;
  state.selectedId = id;
  state.currentFrame = 0;
  state.dirty = false;
  byId("emptyState").classList.add("hidden");
  byId("reviewContent").classList.remove("hidden");
  renderRolloutList();

  var originClass = provenanceClass(record);
  byId("recordBadges").innerHTML = badge(
    isControlled(record) ? "controlled injection" : (isRealRobotManifest(record) ? "real robot" : (isExternalManifest(record) ? "secondary manifest" : "natural policy")),
    originClass
  )
    + badge(record.analysis_partition, originClass)
    + badge(effectiveOutcome(record), effectiveOutcome(record));
  renderInstructionVariantControl(record);
  byId("frameSlider").max = Math.max(0, Number(record.total_frames) - 1);

  var video = byId("rolloutVideo");
  video.pause();
  // Bust browser caches that may contain the pre-transcode MPEG-4 Part 2
  // response from before the server started serving its H.264 copy.
  video.src = "/api/videos/" + encodeURIComponent(record.id) + "?v=video-h264-20260916";
  video.load();
  video.playbackRate = Number(byId("speedSelect").value);
  byId("playButton").textContent = "Play";
  byId("playOverlay").classList.remove("hidden");

  var injection = record.injection || null;
  var known = byId("knownEvent");
  known.classList.toggle("controlled", isControlled(record));
  if (injection) {
    byId("knownEventText").textContent = labelFor(injection.type)
      + " · causal frame " + injection.causal_onset_frame
      + (injection.end_frame == null ? " onward" : "–" + injection.end_frame)
      + (injection.environment_timestep == null ? "" : " · environment timestep " + injection.environment_timestep);
  } else {
    byId("knownEventText").textContent = "No injected event. Causal onset may remain blank; annotate only evidence in the natural rollout.";
  }

  loadForm(record);
  seekFrame(0);
  updateNavigationButtons();
  loadEvaluation(record.id);
}

function loadForm(record) {
  var annotation = record.annotation || {};
  var knownCausal = record.injection ? record.injection.causal_onset_frame : null;
  var outcome = annotation.outcome_label || record.ground_truth_outcome || "uncertain";
  var defaultType = annotation.failure_type || (outcome === "success" ? "none_success" : "other");
  byId("annotatorInput").value = annotation.annotator || sessionStorage.getItem("lf3r_annotator") || "";
  byId("outcomeLabel").value = outcome;
  byId("reviewStatus").value = annotation.review_status || "in_progress";
  byId("failureType").value = defaultType;
  byId("confidence").value = annotation.confidence == null ? "" : String(annotation.confidence);
  state.failureEvents = normalizeFailureEvents(annotation, defaultType, knownCausal);
  if (!state.failureEvents.length && (outcome === "failure" || outcome === "recovered_success" || knownCausal != null)) {
    state.failureEvents.push(emptyFailureEvent(defaultType, knownCausal));
  }
  state.activeFailureEvent = state.failureEvents.length ? 0 : null;
  renderFailureEvents();
  byId("notes").value = annotation.notes || "";
  byId("formError").textContent = "";
  byId("saveState").textContent = annotation.updated_at ? "Saved " + formatDate(annotation.updated_at) : "Not saved";
  byId("saveState").classList.toggle("saved", Boolean(annotation.updated_at));
  state.dirty = false;
  renderTimelineMarkers();
}

function emptyFailureEvent(defaultType, knownCausal) {
  return {
    failure_type: defaultType || "other",
    causal_onset_frame: knownCausal == null ? null : Number(knownCausal),
    observable_onset_frame: null,
    terminal_failure_frame: null,
    recovery_frame: null,
    notes: ""
  };
}

function normalizeFailureEvents(annotation, defaultType, knownCausal) {
  var source;
  if (Array.isArray(annotation.failure_events)) {
    source = annotation.failure_events;
  } else {
    var legacyCausal = annotation.causal_onset_frame != null ? annotation.causal_onset_frame : knownCausal;
    var hasLegacy = legacyCausal != null
      || annotation.observable_onset_frame != null
      || annotation.terminal_failure_frame != null
      || annotation.recovery_frame != null;
    source = hasLegacy ? [{
      failure_type: defaultType,
      causal_onset_frame: legacyCausal,
      observable_onset_frame: annotation.observable_onset_frame,
      terminal_failure_frame: annotation.terminal_failure_frame,
      recovery_frame: annotation.recovery_frame,
      notes: ""
    }] : [];
  }
  return source.map(function (event) {
    return {
      failure_type: event.failure_type || defaultType || "other",
      causal_onset_frame: event.causal_onset_frame == null ? null : Number(event.causal_onset_frame),
      observable_onset_frame: event.observable_onset_frame == null ? null : Number(event.observable_onset_frame),
      terminal_failure_frame: event.terminal_failure_frame == null ? null : Number(event.terminal_failure_frame),
      recovery_frame: event.recovery_frame == null ? null : Number(event.recovery_frame),
      notes: event.notes || ""
    };
  });
}

var failureTypeChoices = [
  ["none_success", "No failure / success"],
  ["grasp_failure", "Grasp failure"],
  ["placement_failure", "Placement failure"],
  ["dropped_object", "Dropped object"],
  ["wrong_object", "Wrong object"],
  ["collision", "Collision"],
  ["timeout_no_progress", "Timeout / no progress"],
  ["control_error", "Control error"],
  ["observation_error", "Observation error"],
  ["other", "Other"]
];

function failureTypeOptions(selected) {
  return failureTypeChoices.map(function (choice) {
    return '<option value="' + choice[0] + '"' + (choice[0] === selected ? " selected" : "") + ">"
      + escapeHtml(choice[1]) + "</option>";
  }).join("");
}

function eventFrameControl(index, field, title, help, cssClass) {
  var event = state.failureEvents[index];
  var value = event[field] == null ? "" : String(event[field]);
  var record = selectedRollout();
  var maxFrame = record ? Math.max(0, Number(record.total_frames) - 1) : 0;
  return '<div class="onset-field ' + cssClass + '-field">'
    + '<div><span>' + escapeHtml(title) + '</span><small>' + escapeHtml(help) + '</small></div>'
    + '<div class="frame-entry">'
    + '<input type="number" min="0" max="' + maxFrame + '" placeholder="—" value="' + escapeHtml(value) + '" data-event-field="' + field + '">'
    + '<button data-event-set="' + field + '" type="button">Use current</button>'
    + '<button data-event-clear="' + field + '" class="clear-button" type="button">×</button>'
    + "</div></div></div>";
}

function renderFailureEvents() {
  var container = byId("failureEvents");
  byId("failureEventCount").textContent = String(state.failureEvents.length);
  if (!state.failureEvents.length) {
    container.innerHTML = '<div class="no-failure-events">No failure events. Add one for each failed attempt or visible deviation.</div>';
    renderTimelineMarkers();
    return;
  }
  container.innerHTML = state.failureEvents.map(function (event, index) {
    var activeClass = index === state.activeFailureEvent ? " active" : "";
    return '<article class="failure-event-card' + activeClass + '" data-event-index="' + index + '" tabindex="0">'
      + '<div class="failure-event-title"><strong>Failure event #' + (index + 1) + '</strong>'
      + '<button type="button" data-remove-event="' + index + '" class="remove-event-button">Remove</button></div>'
      + '<label class="field event-type-field"><span>Failure type</span><select data-event-field="failure_type">'
      + failureTypeOptions(event.failure_type) + "</select></label>"
      + '<div class="onset-group">'
      + eventFrameControl(index, "causal_onset_frame", "Causal / injected onset", "Action or event that starts this failure", "causal")
      + eventFrameControl(index, "observable_onset_frame", "Observable onset", "First visible deviation for this failure", "observable")
      + eventFrameControl(index, "terminal_failure_frame", "Terminal failure", "Use only if recovery is no longer plausible", "terminal")
      + eventFrameControl(index, "recovery_frame", "Recovery", "Frame where this failed attempt has recovered", "recovery")
      + "</div>"
      + '<label class="field event-notes-field"><span>Event notes</span>'
      + '<textarea rows="2" maxlength="1000" data-event-field="notes" placeholder="Evidence or recovery behavior…">'
      + escapeHtml(event.notes) + "</textarea></label></article>";
  }).join("");
  renderTimelineMarkers();
}

function setActiveFailureEvent(index) {
  if (index < 0 || index >= state.failureEvents.length) return;
  state.activeFailureEvent = index;
  byId("failureEvents").querySelectorAll(".failure-event-card").forEach(function (card) {
    card.classList.toggle("active", Number(card.dataset.eventIndex) === index);
  });
}

function addFailureEvent(seed) {
  var defaultType = byId("failureType").value || "other";
  state.failureEvents.push(seed || emptyFailureEvent(defaultType, null));
  state.activeFailureEvent = state.failureEvents.length - 1;
  renderFailureEvents();
  markDirty();
}

function removeFailureEvent(index) {
  state.failureEvents.splice(index, 1);
  state.activeFailureEvent = state.failureEvents.length
    ? Math.min(index, state.failureEvents.length - 1)
    : null;
  renderFailureEvents();
  markDirty();
}

function syncFailureEventInput(target) {
  var card = target.closest(".failure-event-card");
  if (!card) return;
  var index = Number(card.dataset.eventIndex);
  var field = target.dataset.eventField;
  if (!field || !state.failureEvents[index]) return;
  state.failureEvents[index][field] = target.type === "number"
    ? (target.value === "" ? null : Number(target.value))
    : target.value;
  if (field === "terminal_failure_frame" && target.value !== "") {
    state.failureEvents[index].recovery_frame = null;
    var recoveryInput = card.querySelector('[data-event-field="recovery_frame"]');
    if (recoveryInput) recoveryInput.value = "";
  } else if (field === "recovery_frame" && target.value !== "") {
    state.failureEvents[index].terminal_failure_frame = null;
    var terminalInput = card.querySelector('[data-event-field="terminal_failure_frame"]');
    if (terminalInput) terminalInput.value = "";
  }
  setActiveFailureEvent(index);
}

function setActiveEventFrame(field, value) {
  if (state.activeFailureEvent == null) {
    addFailureEvent();
  }
  var event = state.failureEvents[state.activeFailureEvent];
  event[field] = value == null ? null : Number(value);
  if (field === "terminal_failure_frame" && value != null) event.recovery_frame = null;
  if (field === "recovery_frame" && value != null) event.terminal_failure_frame = null;
  renderFailureEvents();
  markDirty();
}

function currentFrame() {
  return state.currentFrame;
}

function seekFrame(frame) {
  var record = selectedRollout();
  if (!record) return;
  var bounded = Math.max(0, Math.min(Number(record.total_frames) - 1, Number(frame) || 0));
  state.currentFrame = Math.round(bounded);
  byId("frameSlider").value = state.currentFrame;
  byId("rolloutVideo").currentTime = state.currentFrame / Number(record.fps);
  updateReadout();
}

function stepFrames(delta) {
  seekFrame(currentFrame() + delta);
}

function updateReadout() {
  var record = selectedRollout();
  if (!record) return;
  var seconds = state.currentFrame / Number(record.fps);
  var environmentTimestep = record.first_environment_timestep == null
    ? ""
    : " · env t=" + (Number(record.first_environment_timestep) + state.currentFrame);
  byId("timeReadout").textContent = formatTime(seconds);
  byId("frameReadout").textContent = "Frame " + state.currentFrame + " / " + (Number(record.total_frames) - 1) + environmentTimestep;
  updateEvaluationCurrent();
}

function formatTime(seconds) {
  var minutes = Math.floor(seconds / 60);
  var remainder = seconds - minutes * 60;
  return String(minutes).padStart(2, "0") + ":" + remainder.toFixed(3).padStart(6, "0");
}

function formatDate(value) {
  try {
    return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch (_error) {
    return "recently";
  }
}

function togglePlayback() {
  var video = byId("rolloutVideo");
  if (video.paused) {
    video.play();
  } else {
    video.pause();
  }
}

var TIMELINE_MARKER_DEFINITIONS = [
  { field: "causal_onset_frame", cssClass: "causal", label: "Causal / injected onset" },
  { field: "observable_onset_frame", cssClass: "observable", label: "Observable onset" },
  { field: "terminal_failure_frame", cssClass: "terminal", label: "Terminal failure" },
  { field: "recovery_frame", cssClass: "recovery", label: "Recovery" }
];

function timelineDomainMax(record, fallback) {
  var totalFrames = record ? Number(record.total_frames) : NaN;
  if (Number.isFinite(totalFrames) && totalFrames > 1) return totalFrames - 1;
  var fallbackNumber = Number(fallback);
  return Number.isFinite(fallbackNumber) && fallbackNumber > 0 ? fallbackNumber : 1;
}

function collectTimelineMarkers(record, domainMax) {
  var total = timelineDomainMax(record, domainMax);
  var markers = [];
  state.failureEvents.forEach(function (event, eventIndex) {
    TIMELINE_MARKER_DEFINITIONS.forEach(function (definition) {
      var rawFrame = event[definition.field];
      if (rawFrame == null || rawFrame === "") return;
      var frame = Number(rawFrame);
      if (!Number.isFinite(frame)) return;
      markers.push({
        eventIndex: eventIndex,
        cssClass: definition.cssClass,
        label: definition.label,
        frame: Math.max(0, Math.min(total, frame))
      });
    });
  });
  return { total: total, markers: markers };
}

function timelineMarkerTitle(marker) {
  return "Failure event #" + (marker.eventIndex + 1) + " · " + marker.label
    + " · frame " + Math.round(marker.frame);
}

function timelinePercent(frame, total) {
  var denominator = Number(total);
  var value = Number(frame);
  if (!Number.isFinite(denominator) || denominator <= 0 || !Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value / denominator * 100));
}

function renderTimelineMarkers() {
  var record = selectedRollout();
  var container = byId("timelineMarkers");
  if (!record || !container) return;
  var timeline = collectTimelineMarkers(record);
  container.innerHTML = timeline.markers.map(function (marker) {
    var position = timelinePercent(marker.frame, timeline.total);
    return '<i class="timeline-pin ' + marker.cssClass + '" title="' + escapeHtml(timelineMarkerTitle(marker))
      + '" style="left:' + position.toFixed(4) + '%"></i>';
  }).join("");
  updateEvaluationChartMarkers();
}

function renderEvaluationOnsetMarkers(record, domainMax) {
  if (!record) return "";
  var timeline = collectTimelineMarkers(record, domainMax);
  return timeline.markers.map(function (marker) {
    var position = timelinePercent(marker.frame, timeline.total);
    var title = timelineMarkerTitle(marker);
    return '<span class="evaluation-onset-pin ' + marker.cssClass + '" style="left:' + position.toFixed(4)
      + '%" title="' + escapeHtml(title) + '" aria-label="' + escapeHtml(title) + '"></span>';
  }).join("");
}

function updateEvaluationChartMarkers() {
  var record = selectedRollout();
  if (!record) return;
  document.querySelectorAll("[data-evaluation-onset-markers]").forEach(function (container) {
    var domainMax = Number(container.dataset.frameMax);
    container.innerHTML = renderEvaluationOnsetMarkers(record, domainMax);
  });
}

function markDirty() {
  state.dirty = true;
  byId("saveState").textContent = "Unsaved changes";
  byId("saveState").classList.remove("saved");
  renderTimelineMarkers();
}

function formPayload() {
  var first = state.failureEvents.length ? state.failureEvents[0] : {};
  var events = state.failureEvents.map(function (event) {
    return {
      failure_type: event.failure_type,
      causal_onset_frame: event.causal_onset_frame,
      observable_onset_frame: event.observable_onset_frame,
      terminal_failure_frame: event.terminal_failure_frame,
      recovery_frame: event.recovery_frame,
      notes: event.notes
    };
  });
  return {
    annotator: byId("annotatorInput").value.trim(),
    outcome_label: byId("outcomeLabel").value,
    review_status: byId("reviewStatus").value,
    failure_type: byId("failureType").value,
    confidence: byId("confidence").value === "" ? null : Number(byId("confidence").value),
    failure_events: events,
    causal_onset_frame: first.causal_onset_frame == null ? null : first.causal_onset_frame,
    observable_onset_frame: first.observable_onset_frame == null ? null : first.observable_onset_frame,
    terminal_failure_frame: first.terminal_failure_frame == null ? null : first.terminal_failure_frame,
    recovery_frame: first.recovery_frame == null ? null : first.recovery_frame,
    notes: byId("notes").value
  };
}

async function saveAnnotation(event) {
  if (event) event.preventDefault();
  var record = selectedRollout();
  if (!record) return;
  var payload = formPayload();
  payload.review_status = "complete";
  if (!payload.annotator) {
    byId("formError").textContent = "Annotator is required.";
    byId("annotatorInput").focus();
    return;
  }
  byId("formError").textContent = "";
  byId("saveButton").disabled = true;
  byId("saveState").textContent = "Saving…";
  try {
    var response = await fetch("/api/annotations/" + encodeURIComponent(record.id), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    var result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Save failed");
    }
    sessionStorage.setItem("lf3r_annotator", payload.annotator);
    record.annotation = result.annotation;
    record.annotation_status = result.annotation.review_status;
    byId("reviewStatus").value = result.annotation.review_status;
    state.dirty = false;
    byId("saveState").textContent = "Saved " + formatDate(result.annotation.updated_at);
    byId("saveState").classList.add("saved");
    updateProgress();
    applyFilters();
    if (typeof window.lf3rWorkspaceDataChanged === "function") window.lf3rWorkspaceDataChanged();
  } catch (error) {
    byId("formError").textContent = error.message;
    byId("saveState").textContent = "Save failed";
  } finally {
    byId("saveButton").disabled = false;
  }
}

function updateNavigationButtons() {
  var index = state.filtered.findIndex(function (record) { return record.id === state.selectedId; });
  byId("previousButton").disabled = index <= 0;
  byId("nextButton").disabled = index < 0 || index >= state.filtered.length - 1;
}

function navigate(delta) {
  var index = state.filtered.findIndex(function (record) { return record.id === state.selectedId; });
  var target = state.filtered[index + delta];
  if (target) maybeSelectRollout(target.id);
}

function isTypingTarget(target) {
  return target && ["INPUT", "TEXTAREA", "SELECT"].indexOf(target.tagName) !== -1;
}



var BASELINE_BATCH_SCOPE_LABELS = {
  all: "All loaded rollouts",
  libero_10: "LIBERO-10",
  libero_spatial: "LIBERO-Spatial",
  controlled_analysis: "Controlled"
};

function baselineBatchMatchesScope(record, scope) {
  if (scope === "all") return true;
  if (scope === "libero_10" || scope === "libero_spatial") return record.task_suite === scope;
  return record.analysis_partition === scope;
}

function baselineBatchUsesWorkers() {
  // All baseline methods now share the rollout-level worker allocator.
  return true;
}

function baselineBatchIsRynnValue() {
  var method = byId("baselineBatchMethod");
  return Boolean(method && method.value === "rynnvalue");
}

function baselineBatchUnfilteredScopeRecords() {
  var scope = byId("baselineBatchScope");
  if (!scope) return [];
  var conditionNode = byId("baselineBatchCondition");
  var condition = conditionNode ? conditionNode.value : (state.instructionCondition || "full_instruction");
  return (state.rollouts || []).filter(function (record) {
    if (!baselineBatchMatchesScope(record, scope.value)) return false;
    if (condition === "full_instruction") return true;
    return Boolean(record.instruction_variants && record.instruction_variants[condition]);
  });
}

function baselineBatchResultFilterValue() {
  var node = byId("baselineBatchResultFilter");
  return node ? (node.value || "all") : "all";
}

function baselineBatchCoverageKey() {
  var method = byId("baselineBatchMethod");
  var scope = byId("baselineBatchScope");
  var condition = byId("baselineBatchCondition");
  return [
    method ? method.value : "",
    scope ? scope.value : "",
    condition ? condition.value : "full_instruction"
  ].join("::");
}

function baselineBatchScopeRecords() {
  var records = baselineBatchUnfilteredScopeRecords();
  if (baselineBatchResultFilterValue() !== "missing_valid") return records;
  var key = baselineBatchCoverageKey();
  var coverage = state.baselineBatchCoverage;
  if (!coverage || state.baselineBatchCoverageKey !== key) return [];
  var missing = {};
  (coverage.missing_source_rollout_ids || []).forEach(function (rolloutId) {
    missing[String(rolloutId)] = true;
  });
  return records.filter(function (record) {
    return Boolean(missing[String(record.id)]);
  });
}

async function loadBaselineBatchCoverage(force) {
  if (baselineBatchResultFilterValue() !== "missing_valid") {
    state.baselineBatchCoverage = null;
    state.baselineBatchCoverageKey = null;
    state.baselineBatchCoverageLoading = false;
    return null;
  }
  var key = baselineBatchCoverageKey();
  if (!force && state.baselineBatchCoverage && state.baselineBatchCoverageKey === key) {
    return state.baselineBatchCoverage;
  }
  var requestId = ++state.baselineBatchCoverageRequest;
  var method = byId("baselineBatchMethod").value;
  var scope = byId("baselineBatchScope").value;
  var condition = byId("baselineBatchCondition").value || "full_instruction";
  state.baselineBatchCoverageLoading = true;
  var note = byId("baselineBatchSelection");
  if (note) note.textContent = "Checking existing " + method + " results…";
  try {
    var response = await fetch(
      "/api/baselines/result-coverage?baseline=" + encodeURIComponent(method)
      + "&scope=" + encodeURIComponent(scope)
      + "&condition=" + encodeURIComponent(condition),
      { cache: "no-store" }
    );
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not inspect baseline result coverage");
    if (requestId !== state.baselineBatchCoverageRequest || key !== baselineBatchCoverageKey()) return null;
    var previousIds = state.baselineBatchCoverage && state.baselineBatchCoverageKey === key
      ? JSON.stringify(state.baselineBatchCoverage.missing_source_rollout_ids || [])
      : null;
    var nextIds = JSON.stringify(payload.missing_source_rollout_ids || []);
    state.baselineBatchCoverage = payload;
    state.baselineBatchCoverageKey = key;
    state.baselineBatchCoverageLoading = false;
    if (previousIds !== nextIds) baselineBatchScopeChanged();
    else updateBaselineBatchSelection();
    return payload;
  } catch (error) {
    if (requestId !== state.baselineBatchCoverageRequest) return null;
    state.baselineBatchCoverage = null;
    state.baselineBatchCoverageKey = key;
    state.baselineBatchCoverageLoading = false;
    setBaselineBatchStatus("Baseline result filter error: " + error.message, "error");
    updateBaselineBatchSelection();
    return null;
  }
}

function baselineBatchCoverageChanged() {
  state.baselineBatchCoverage = null;
  state.baselineBatchCoverageKey = null;
  state.baselineBatchCoverageRequest += 1;
  if (baselineBatchResultFilterValue() === "missing_valid") {
    loadBaselineBatchCoverage(true);
  } else {
    baselineBatchScopeChanged();
  }
}

function baselineBatchNumber(value) {
  var number = Number(value);
  return Number.isInteger(number) ? number : null;
}

function baselineBatchTotalRange() {
  var records = baselineBatchScopeRecords();
  var startNode = byId("baselineBatchStartIndex");
  var endNode = byId("baselineBatchEndIndex");
  var start = baselineBatchNumber(startNode ? startNode.value : "");
  var endText = endNode ? endNode.value.trim() : "";
  var end = endText === "" ? records.length : baselineBatchNumber(endText);
  return {
    records: records,
    matched: records.length,
    start: start,
    end: end,
    valid: start != null && end != null && start >= 0 && end > start && end <= records.length
  };
}

function baselineBatchSelectionRecords() {
  if (baselineBatchUsesWorkers()) {
    var range = baselineBatchTotalRange();
    if (!range.valid) return [];
    var seen = {};
    var selected = [];
    (state.baselineBatchWorkers || []).forEach(function (worker) {
      var start = baselineBatchNumber(worker.start_index);
      var end = baselineBatchNumber(worker.end_index);
      if (start == null || end == null || start < range.start || end > range.end || end <= start) return;
      for (var index = start; index < end; index += 1) {
        if (seen[index]) continue;
        seen[index] = true;
        selected.push(range.records[index]);
      }
    });
    return selected;
  }
  var records = baselineBatchScopeRecords();
  var start = Number(byId("baselineBatchStartIndex").value || 0);
  var limitText = byId("baselineBatchLimit").value.trim();
  var limit = limitText === "" ? null : Number(limitText);
  if (!Number.isInteger(start) || start < 0) return [];
  if (limit != null && (!Number.isInteger(limit) || limit < 1)) return [];
  return limit == null ? records.slice(start) : records.slice(start, start + limit);
}

function baselineBatchRebalanceWorkers() {
  var range = baselineBatchTotalRange();
  var count = Math.max(1, (state.baselineBatchWorkers || []).length || 1);
  var oldWorkers = state.baselineBatchWorkers || [];
  var workers = [];
  if (range.start != null && range.end != null && range.end > range.start) {
    var total = range.end - range.start;
    var base = Math.floor(total / count);
    var remainder = total % count;
    var cursor = range.start;
    for (var index = 0; index < count; index += 1) {
      var width = base + (index < remainder ? 1 : 0);
      workers.push({
        gpu: oldWorkers[index] && String(oldWorkers[index].gpu || "0") || "0",
        start_index: cursor,
        end_index: cursor + width
      });
      cursor += width;
    }
  } else {
    for (var emptyIndex = 0; emptyIndex < count; emptyIndex += 1) {
      workers.push({
        gpu: oldWorkers[emptyIndex] && String(oldWorkers[emptyIndex].gpu || "0") || "0",
        start_index: 0,
        end_index: 0
      });
    }
  }
  state.baselineBatchWorkers = workers;
  renderBaselineBatchWorkers();
}

function readBaselineBatchWorkers() {
  var workers = [];
  document.querySelectorAll("#baselineBatchWorkers [data-worker-row]").forEach(function (row) {
    var gpu = row.querySelector('[data-worker-field="gpu"]');
    var start = row.querySelector('[data-worker-field="start_index"]');
    var end = row.querySelector('[data-worker-field="end_index"]');
    workers.push({
      gpu: gpu ? gpu.value.trim() : "",
      start_index: start ? start.value : "",
      end_index: end ? end.value : ""
    });
  });
  if (workers.length) state.baselineBatchWorkers = workers;
  return state.baselineBatchWorkers || [];
}

function renderBaselineBatchWorkers() {
  var container = byId("baselineBatchWorkers");
  if (!container) return;
  var workers = state.baselineBatchWorkers || [];
  container.innerHTML = workers.map(function (worker, index) {
    return '<div class="baseline-worker-row rynn-worker-row" data-worker-row data-worker-index="' + index + '">'
      + '<span class="baseline-worker-label rynn-worker-label">Worker ' + (index + 1) + '</span>'
      + '<label><span>GPU ID</span><input data-worker-field="gpu" type="text" inputmode="numeric" value="' + escapeHtml(worker.gpu) + '" aria-label="Worker ' + (index + 1) + ' GPU"></label>'
      + '<label><span>Start (inclusive)</span><input data-worker-field="start_index" type="number" min="0" step="1" value="' + escapeHtml(worker.start_index) + '" aria-label="Worker ' + (index + 1) + ' start index"></label>'
      + '<label><span>End (exclusive)</span><input data-worker-field="end_index" type="number" min="1" step="1" value="' + escapeHtml(worker.end_index) + '" aria-label="Worker ' + (index + 1) + ' end index"></label>'
      + '<button type="button" class="ghost-button baseline-worker-remove rynn-worker-remove" data-remove-worker="' + index + '"' + (workers.length <= 1 ? ' disabled' : '') + '>Remove</button>'
      + '</div>';
  }).join("");
}

function baselineBatchWorkerSummary() {
  var range = baselineBatchTotalRange();
  var workers = readBaselineBatchWorkers();
  var counts = {};
  var gpuCounts = {};
  var rows = [];
  var invalid = !range.valid;
  workers.forEach(function (worker, workerIndex) {
    var start = baselineBatchNumber(worker.start_index);
    var end = baselineBatchNumber(worker.end_index);
    var gpu = String(worker.gpu || "").trim();
    var valid = /^\d+$/.test(gpu) && start != null && end != null
      && start >= (range.start == null ? 0 : range.start)
      && end <= (range.end == null ? 0 : range.end) && end > start;
    if (!valid) invalid = true;
    if (/^\d+$/.test(gpu)) gpuCounts[gpu] = (gpuCounts[gpu] || 0) + 1;
    if (valid) {
      for (var index = start; index < end; index += 1) counts[index] = (counts[index] || 0) + 1;
    }
    rows.push("W" + (workerIndex + 1) + " GPU " + (gpu || "?") + " [" + (start == null ? "?" : start) + "," + (end == null ? "?" : end) + ") = " + (valid ? end - start : "invalid"));
  });
  var overlap = Object.keys(counts).filter(function (index) { return counts[index] > 1; }).map(Number);
  var gaps = [];
  if (range.valid) {
    for (var expected = range.start; expected < range.end; expected += 1) {
      if (!counts[expected]) gaps.push(expected);
    }
  }
  var unique = Object.keys(counts).length;
  var reusedGpu = Object.keys(gpuCounts).filter(function (gpu) { return gpuCounts[gpu] > 1; });
  return {
    range: range,
    invalid: invalid,
    overlap: overlap,
    gaps: gaps,
    unique: unique,
    reusedGpu: reusedGpu,
    text: rows.join(" · ")
  };
}

function updateBaselineBatchSelection() {
  var scope = byId("baselineBatchScope");
  var note = byId("baselineBatchSelection");
  var button = byId("baselineBatchRun");
  if (!scope || !note || !button) return;
  var matched = baselineBatchScopeRecords().length;
  var conditionNode = byId("baselineBatchCondition");
  var condition = conditionNode ? conditionNode.value : (state.instructionCondition || "full_instruction");
  var conditionLabel = instructionConditionLabel(condition);
  var label = BASELINE_BATCH_SCOPE_LABELS[scope.value] || scope.value;
  var resultFilter = baselineBatchResultFilterValue();
  var coverage = (
    resultFilter === "missing_valid"
    && state.baselineBatchCoverageKey === baselineBatchCoverageKey()
  ) ? state.baselineBatchCoverage : null;
  var coverageText = "";
  if (resultFilter === "missing_valid") {
    if (state.baselineBatchCoverageLoading) {
      coverageText = "checking existing results; ";
    } else if (!coverage) {
      coverageText = "existing-result coverage unavailable; ";
    } else {
      coverageText = coverage.missing_valid_result_rollouts + " complete annotation(s) without valid result / "
        + coverage.complete_annotation_rollouts + " complete; "
        + coverage.valid_result_rollouts + " existing valid result(s) skipped; "
        + coverage.incomplete_annotation_rollouts + " in-progress/unreviewed skipped; ";
    }
  }
  if (baselineBatchUsesWorkers()) {
    var summary = baselineBatchWorkerSummary();
    var rangeText = summary.range.start == null || summary.range.end == null
      ? "invalid total range" : "total [" + summary.range.start + "," + summary.range.end + ")";
    note.textContent = conditionLabel + " · " + label + ": " + coverageText + matched + " rollout(s) available; " + rangeText
      + "; unique execution " + summary.unique + ". "
      + (summary.overlap.length ? "Overlap " + summary.overlap.length + ". " : "No overlap. ")
      + (summary.gaps.length ? "Gap " + summary.gaps.length + ". " : "No gap. ")
      + (summary.reusedGpu.length ? "GPU reused: " + summary.reusedGpu.join(", ") + "." : "");
    var workerNote = byId("baselineBatchWorkerSummary");
    if (workerNote) {
      workerNote.textContent = summary.text + " · Total " + (summary.range.valid ? summary.range.end - summary.range.start : "invalid")
        + " · Unique " + summary.unique
        + (summary.overlap.length ? " · overlap " + summary.overlap.length : "")
        + (summary.gaps.length ? " · gap " + summary.gaps.length : "")
        + (summary.reusedGpu.length ? " · repeated GPU " + summary.reusedGpu.join(", ") : "");
    }
    button.disabled = summary.invalid || summary.unique === 0 || state.baselineBatchSubmitting;
    return;
  }
  var selected = baselineBatchSelectionRecords().length;
  note.textContent = conditionLabel + " · " + label + ": " + coverageText + matched + " rollout(s) available; " + selected + " selected after index/limit.";
  button.disabled = selected === 0 || state.baselineBatchSubmitting;
}

function updateBaselineBatchAdvancedFields() {
  var method = byId("baselineBatchMethod").value;
  var rynn = method === "rynnvalue";
  document.querySelectorAll("[data-batch-methods]").forEach(function (label) {
    var methods = label.dataset.batchMethods.split(",");
    var visible = methods.indexOf(method) !== -1;
    label.classList.toggle("hidden", !visible);
    label.querySelectorAll("[data-batch-option]").forEach(function (input) { input.disabled = !visible; });
  });
  var parallelPanel = byId("baselineBatchRynnParallel");
  if (parallelPanel) parallelPanel.classList.remove("hidden");
  var endField = byId("baselineBatchEndField");
  var limitField = byId("baselineBatchLimitField");
  if (endField) endField.classList.remove("hidden");
  if (limitField) limitField.classList.add("hidden");
  var gpuNode = byId("baselineBatchGpu");
  if (gpuNode && gpuNode.closest("label")) gpuNode.closest("label").classList.add("hidden");

  var range = baselineBatchTotalRange();
  var endNode = byId("baselineBatchEndIndex");
  if (endNode && !endNode.value.trim() && range.matched) endNode.value = range.matched;
  var workers = state.baselineBatchWorkers || [];
  var hasUsableWorker = workers.some(function (worker) {
    var workerStart = baselineBatchNumber(worker.start_index);
    var workerEnd = baselineBatchNumber(worker.end_index);
    return workerStart != null && workerEnd != null && workerEnd > workerStart;
  });
  if (!workers.length || !hasUsableWorker) {
    state.baselineBatchWorkers = [{
      gpu: workers[0] && String(workers[0].gpu || "0") || "0",
      start_index: range.start == null ? 0 : range.start,
      end_index: range.end == null ? 0 : range.end
    }];
  }
  renderBaselineBatchWorkers();
  var heading = parallelPanel && parallelPanel.querySelector("strong");
  if (heading) heading.textContent = method.toUpperCase() + " rollout workers";
  updateBaselineBatchSelection();
}

function baselineBatchScopeChanged() {
  var records = baselineBatchScopeRecords();
  byId("baselineBatchStartIndex").value = "0";
  byId("baselineBatchLimit").value = "";
  if (byId("baselineBatchEndIndex")) byId("baselineBatchEndIndex").value = String(records.length);
  if (baselineBatchUsesWorkers()) baselineBatchRebalanceWorkers();
  updateBaselineBatchSelection();
}

function baselineBatchRangeChanged() {
  if (baselineBatchUsesWorkers()) baselineBatchRebalanceWorkers();
  else updateBaselineBatchSelection();
}

function addBaselineBatchWorker() {
  readBaselineBatchWorkers();
  state.baselineBatchWorkers.push({ gpu: "0", start_index: 0, end_index: 0 });
  baselineBatchRebalanceWorkers();
  updateBaselineBatchSelection();
}

function removeBaselineBatchWorker(index) {
  readBaselineBatchWorkers();
  if (state.baselineBatchWorkers.length <= 1) return;
  state.baselineBatchWorkers.splice(index, 1);
  baselineBatchRebalanceWorkers();
  updateBaselineBatchSelection();
}

function baselineBatchOptions() {
  var options = {};
  document.querySelectorAll("[data-batch-option]").forEach(function (input) {
    if (input.disabled) return;
    if (input.type === "checkbox") {
      if (input.checked) options[input.dataset.batchOption] = true;
      return;
    }
    if (input.value.trim() === "") return;
    options[input.dataset.batchOption] = input.type === "number" ? Number(input.value) : input.value.trim();
  });
  return options;
}

function setBaselineBatchStatus(message, kind) {
  var status = byId("baselineBatchStatus");
  status.textContent = message;
  status.className = "evaluation-status" + (kind ? " " + kind : "");
}

async function loadBaselineBatchLog(jobId) {
  if (window.LF3RRunsJobs && typeof window.LF3RRunsJobs.refreshSelectedLog === "function") {
    return window.LF3RRunsJobs.refreshSelectedLog("baseline", jobId);
  }
  try {
    var response = await fetch("/api/baseline-jobs/" + encodeURIComponent(jobId) + "/log?tail=200", { cache: "no-store" });
    var payload = await response.json();
    if (response.ok && state.baselineBatchJob && state.baselineBatchJob.job_id === jobId) {
      byId("baselineBatchLog").textContent = payload.log ? payload.log.text : "";
    }
  } catch (_error) {
    // The status endpoint remains useful when a log is not ready yet.
  }
}

function setRejectedRequestLog(message, body) {
  var log = byId("baselineBatchLog");
  if (!log) return;
  log.textContent = [
    "Request rejected before a baseline job was created.",
    "",
    String(message || "Unknown request validation error"),
    "",
    "Submitted request:",
    JSON.stringify(body, null, 2)
  ].join("\n");
}

async function startBaselineBatch(event) {
  if (event) event.preventDefault();
  if (state.baselineBatchSubmitting) return;

  var method = byId("baselineBatchMethod").value;
  var scope = byId("baselineBatchScope").value;
  var condition = byId("baselineBatchCondition").value || "full_instruction";
  var resultFilter = baselineBatchResultFilterValue();
  var memory = Number(byId("baselineBatchMemoryUtilization").value);

  if (resultFilter === "missing_valid") {
    var coverage = await loadBaselineBatchCoverage(true);
    if (!coverage) {
      setBaselineBatchStatus("Could not verify existing baseline result coverage.", "error");
      return;
    }
    if (!coverage.missing_valid_result_rollouts) {
      setBaselineBatchStatus(
        "Every matching rollout already has a valid " + method + " result; nothing to run.",
        "warning"
      );
      return;
    }
  }

  readBaselineBatchWorkers();
  var range = baselineBatchTotalRange();
  var summary = baselineBatchWorkerSummary();
  var workers = (state.baselineBatchWorkers || []).map(function (worker) {
    return {
      gpu: String(worker.gpu || "").trim(),
      start_index: baselineBatchNumber(worker.start_index),
      end_index: baselineBatchNumber(worker.end_index)
    };
  });

  if (!range.valid || summary.invalid) {
    setBaselineBatchStatus("Total range and every worker must use valid ranges inside the scope.", "error");
    return;
  }
  if (workers.some(function (worker) { return !/^\d+$/.test(worker.gpu); })) {
    setBaselineBatchStatus("Every worker GPU must be one numeric CUDA device index.", "error");
    return;
  }
  if (!Number.isFinite(memory) || memory <= 0 || memory > 1) {
    setBaselineBatchStatus("Free-memory fraction must be between 0 and 1.", "error");
    byId("baselineBatchMemoryUtilization").focus();
    return;
  }

  var selected = baselineBatchSelectionRecords();
  if (!selected.length) {
    setBaselineBatchStatus("This scope and worker assignment selects no rollouts.", "warning");
    return;
  }

  var uniqueGpus = [];
  workers.forEach(function (worker) {
    if (uniqueGpus.indexOf(worker.gpu) === -1) uniqueGpus.push(worker.gpu);
  });
  var gpu = uniqueGpus.join(",") || "0";

  var confirmation = "Run " + method + " for " + instructionConditionLabel(condition)
    + " over " + selected.length + " unique rollout(s)? This launches GPU inference."
    + (resultFilter === "missing_valid"
      ? " Rollouts with an existing valid result will be skipped by the server."
      : "")
    + " It will start " + workers.length + " rollout worker(s).";
  if (!window.confirm(confirmation)) return;

  state.baselineBatchSubmitting = true;
  updateBaselineBatchSelection();
  setBaselineBatchStatus("Starting " + method + " batch for " + selected.length + " rollout(s)…", "");
  byId("baselineBatchLog").textContent = "";

  var body = {
    baseline: method,
    scope: scope,
    instruction_condition: condition,
    result_filter: resultFilter,
    gpu: gpu,
    memory_utilization: memory,
    start_index: range.start,
    limit: null,
    options: baselineBatchOptions()
  };

  if (workers.length === 1) {
    body.parallel_workers = 1;
    if (!(range.start === 0 && range.end === range.matched)) {
      body.end_index = range.end;
    }
  } else {
    body.end_index = range.end;
    body.parallel_workers = workers.length;
    body.workers = workers;
  }

  try {
    var response = await fetch("/api/baselines/run-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    var responseText = await response.text();
    var payload = {};
    try {
      payload = responseText ? JSON.parse(responseText) : {};
    } catch (_) {
      payload = { error: responseText || "Server returned a non-JSON response" };
    }

    if (!response.ok) {
      var message = payload.error || ("HTTP " + response.status + " while starting baseline batch");
      setRejectedRequestLog(message, body);
      console.error("LF3R baseline batch request rejected", {
        status: response.status,
        message: message,
        request: body,
        response: payload
      });
      throw new Error(message);
    }

    state.baselineBatchJob = payload.job;
    rememberPersistentJob(payload.job);
    await loadBaselineBatchLog(payload.job.job_id);
    pollBaselineBatchJob(payload.job.job_id);
  } catch (error) {
    setBaselineBatchStatus("Batch baseline error: " + error.message, "error");
    if (!byId("baselineBatchLog").textContent.trim()) {
      setRejectedRequestLog(error.message, body);
    }
  } finally {
    state.baselineBatchSubmitting = false;
    updateBaselineBatchSelection();
  }
}

async function pollBaselineBatchJob(jobId) {
  return pollPersistentJob(jobId);
}


function setRolloutGenerationStatus(message, kind) {
  var status = byId("rolloutGenerationStatus");
  var badgeNode = byId("rolloutGenerationBadge");
  if (status) {
    status.textContent = message;
    status.className = "evaluation-status" + (kind ? " " + kind : "");
  }
  if (badgeNode) {
    var job = state.rolloutGenerationJob;
    var label = job && job.status ? job.status : "idle";
    badgeNode.textContent = label;
    badgeNode.className = "analysis-badge" + (
      label === "complete" ? " ok" :
      label === "failed" || label === "memory_blocked" ? " missing" :
      label === "running" || label === "queued" ? " warning" : ""
    );
  }
}

function updateRolloutGenerationSelection() {
  var suiteNode = byId("rolloutGenerationSuite");
  var suite = suiteNode ? suiteNode.value : "libero_10";
  var isSpatial = suite === "libero_spatial";
  var start = Number(byId("rolloutGenerationTaskStart").value);
  var end = Number(byId("rolloutGenerationTaskEnd").value);
  var trials = Number(byId("rolloutGenerationTrials").value);
  var renderResolution = Number(byId("rolloutGenerationRenderResolution").value);
  var recordResolution = Number(byId("rolloutGenerationRecordResolution").value);
  var videoViewMode = byId("rolloutGenerationVideoViewMode").value;
  var saveLatent = byId("rolloutGenerationLogSafeFeatures").checked;
  var note = byId("rolloutGenerationSelection");
  var button = byId("rolloutGenerationRun");
  var title = byId("rolloutGenerationTitle");
  var description = byId("rolloutGenerationDescription");
  if (title) {
    title.textContent = isSpatial
      ? "Generate natural LIBERO-Spatial rollouts"
      : "Generate natural LIBERO-10 rollouts";
  }
  if (description) {
    var viewText = videoViewMode === "libero_three_view"
      ? " Camera videos are generated afterward as physical dataset facts: one cam_high MP4 plus one cam_wrist MP4. Robo-Dopamine later maps cam_wrist to both left/right input slots in its adapter. Canonical policy video remains unchanged."
      : " Only the canonical single-view replay video is generated.";
    description.textContent = (isSpatial
      ? "Uses the existing OpenVLA LIBERO-Spatial checkpoint. Render is " + renderResolution + "x" + renderResolution + ", record is " + recordResolution + "x" + recordResolution + ", and policy preprocessing remains 224x224."
      : "Uses the existing OpenVLA LIBERO-10 natural generator and output root. Render is " + renderResolution + "x" + renderResolution + ", record is " + recordResolution + "x" + recordResolution + ", and policy preprocessing remains 224x224.")
      + viewText + " GPU selection is user-managed; the WebUI does not block launch based on utilization or free memory.";
  }
  var valid = (suite === "libero_10" || suite === "libero_spatial")
    && (videoViewMode === "single_view" || videoViewMode === "libero_three_view")
    && Number.isInteger(start) && Number.isInteger(end) && Number.isInteger(trials)
    && Number.isInteger(renderResolution) && Number.isInteger(recordResolution)
    && renderResolution >= 64 && renderResolution <= 2048 && renderResolution % 2 === 0
    && recordResolution >= 64 && recordResolution <= 2048 && recordResolution % 2 === 0
    && start >= 0 && end >= start && end <= 9 && trials >= 1 && trials <= 50;
  if (!valid) {
    note.textContent = "Task range must be 0-9, trials 1-50, and both resolutions must be even values from 64 to 2048.";
    button.disabled = true;
    return;
  }
  var expected = (end - start + 1) * trials;
  note.textContent = (isSpatial ? "LIBERO-Spatial" : "LIBERO-10")
    + " output: " + expected + " rollout(s), render " + renderResolution + "x" + renderResolution
    + ", record " + recordResolution + "x" + recordResolution
    + ", camera videos " + (videoViewMode === "libero_three_view" ? "Robo-Dopamine 3-view" : "canonical only")
    + "; run note is generated automatically. "
    + (saveLatent ? "Latent saving enabled." : "Latent saving disabled.");
  button.disabled = state.rolloutGenerationSubmitting;
}

async function loadRolloutGenerationLog(jobId) {
  if (window.LF3RRunsJobs && typeof window.LF3RRunsJobs.refreshSelectedLog === "function") {
    return window.LF3RRunsJobs.refreshSelectedLog("rollout_generation", jobId);
  }
  try {
    var response = await fetch("/api/rollout-jobs/" + encodeURIComponent(jobId) + "/log?tail=200", { cache: "no-store" });
    var payload = await response.json();
    if (response.ok && state.rolloutGenerationJob && state.rolloutGenerationJob.job_id === jobId) {
      byId("rolloutGenerationLog").textContent = payload.log ? payload.log.text : "";
    }
  } catch (_error) {
    // The status endpoint remains useful while the log is being created.
  }
}

function rolloutGenerationJobMessage(job) {
  var progress = (job.completed_rollouts || 0) + "/" + (job.expected_rollouts || 0);
  var suite = job.task_suite === "libero_spatial" ? "LIBERO-Spatial" : "LIBERO-10";
  var resolution = "render " + (job.render_resolution || "?") + "x" + (job.render_resolution || "?")
    + ", record " + (job.record_resolution || "?") + "x" + (job.record_resolution || "?")
    + ", camera videos " + (job.video_view_mode === "libero_three_view" ? "Robo-Dopamine 3-view" : "canonical only");
  if (job.status === "queued") return suite + " generation queued (" + resolution + ") - " + progress + " rollout(s) complete...";
  if (job.status === "running") return "Generating " + suite + " rollouts (" + resolution + ") - " + progress + " complete...";
  if (job.status === "complete") {
    return suite + " generation complete (" + resolution + "): " + progress + " rollout(s); manifest "
      + (job.manifest_rebuilt ? "rebuilt." : "was not rebuilt.");
  }
  if (job.status === "memory_blocked") {
    return suite + " generation blocked by the memory-only GPU gate; utilization is informational. Inspect the log below.";
  }
  return suite + " generation failed after " + progress + " rollout(s); inspect the log below.";
}

async function startRolloutGeneration(event) {
  if (event) event.preventDefault();
  if (state.rolloutGenerationSubmitting) return;
  var suiteNode = byId("rolloutGenerationSuite");
  var taskSuite = suiteNode ? suiteNode.value : "libero_10";
  var suiteLabel = taskSuite === "libero_spatial" ? "LIBERO-Spatial" : "LIBERO-10";
  if (taskSuite !== "libero_10" && taskSuite !== "libero_spatial") {
    setRolloutGenerationStatus("Choose a supported task suite.", "error");
    if (suiteNode) suiteNode.focus();
    return;
  }
  var gpu = byId("rolloutGenerationGpu").value.trim();
  var taskStart = Number(byId("rolloutGenerationTaskStart").value);
  var taskEnd = Number(byId("rolloutGenerationTaskEnd").value);
  var trials = Number(byId("rolloutGenerationTrials").value);
  var seed = Number(byId("rolloutGenerationSeed").value);
  var renderResolution = Number(byId("rolloutGenerationRenderResolution").value);
  var recordResolution = Number(byId("rolloutGenerationRecordResolution").value);
  var videoViewMode = byId("rolloutGenerationVideoViewMode").value;
  var saveLatent = byId("rolloutGenerationLogSafeFeatures").checked;
  var label = byId("rolloutGenerationLabel").value.trim();
  if (!/^\d+$/.test(gpu)) {
    setRolloutGenerationStatus("GPU must be one numeric CUDA device index.", "error");
    byId("rolloutGenerationGpu").focus();
    return;
  }
  if (!Number.isInteger(taskStart) || !Number.isInteger(taskEnd)
      || taskStart < 0 || taskEnd < taskStart || taskEnd > 9) {
    setRolloutGenerationStatus("Task range must satisfy 0 <= start <= end <= 9.", "error");
    return;
  }
  if (!Number.isInteger(trials) || trials < 1 || trials > 50) {
    setRolloutGenerationStatus("Trials must be an integer from 1 to 50.", "error");
    byId("rolloutGenerationTrials").focus();
    return;
  }
  if (!Number.isInteger(seed) || seed < 0) {
    setRolloutGenerationStatus("Seed must be a non-negative integer.", "error");
    byId("rolloutGenerationSeed").focus();
    return;
  }
  if (!Number.isInteger(renderResolution) || renderResolution < 64 || renderResolution > 2048 || renderResolution % 2 !== 0) {
    setRolloutGenerationStatus("Render resolution must be an even integer from 64 to 2048.", "error");
    byId("rolloutGenerationRenderResolution").focus();
    return;
  }
  if (!Number.isInteger(recordResolution) || recordResolution < 64 || recordResolution > 2048 || recordResolution % 2 !== 0) {
    setRolloutGenerationStatus("Record resolution must be an even integer from 64 to 2048.", "error");
    byId("rolloutGenerationRecordResolution").focus();
    return;
  }
  if (label && !/^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$/.test(label)) {
    setRolloutGenerationStatus("Run-note label may use only letters, numbers, dot, underscore, or hyphen.", "error");
    byId("rolloutGenerationLabel").focus();
    return;
  }
  if (videoViewMode !== "single_view" && videoViewMode !== "libero_three_view") {
    setRolloutGenerationStatus("Choose a supported review-video mode.", "error");
    byId("rolloutGenerationVideoViewMode").focus();
    return;
  }
  var expected = (taskEnd - taskStart + 1) * trials;
  var multiviewNote = videoViewMode === "libero_three_view"
    ? " A no-model LIBERO replay will then record agent + side + wrist views."
    : "";
  if (!window.confirm("Generate " + expected + " OpenVLA " + suiteLabel + " rollout(s)? This launches GPU inference." + multiviewNote)) return;
  var button = byId("rolloutGenerationRun");
  state.rolloutGenerationSubmitting = true;
  updateRolloutGenerationSelection();
  byId("rolloutGenerationLog").textContent = "";
  setRolloutGenerationStatus("Starting natural " + suiteLabel + " generation...", "");
  try {
    var response = await fetch("/api/rollouts/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task_suite: taskSuite,
        gpu: gpu,
        task_start: taskStart,
        task_end: taskEnd,
        trials: trials,
        seed: seed,
        run_label: label,
        log_safe_features: saveLatent,
        render_resolution: renderResolution,
        record_resolution: recordResolution,
        video_view_mode: videoViewMode
      })
    });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not start rollout generation");
    state.rolloutGenerationJob = payload.job;
    rememberPersistentJob(payload.job);
    setRolloutGenerationStatus(rolloutGenerationJobMessage(payload.job), "");
    await loadRolloutGenerationLog(payload.job.job_id);
    pollRolloutGenerationJob(payload.job.job_id);
  } catch (error) {
    setRolloutGenerationStatus("Rollout generation error: " + error.message, "error");
  } finally {
    state.rolloutGenerationSubmitting = false;
    updateRolloutGenerationSelection();
  }
}
async function pollRolloutGenerationJob(jobId) {
  return pollPersistentJob(jobId);
}


async function startBaselineRun(method) {
  var record = selectedRollout();
  if (!record) return;
  var condition = state.instructionCondition || "full_instruction";
  var conditionLabel = instructionConditionLabel(condition);
  // Single-rollout reruns use the same stable defaults as the former compact controls.
  var gpu = "0";
  var memory = 0.80;
  if (!window.confirm("Run " + method + " for this one " + conditionLabel + " rollout? This launches GPU inference.")) return;
  byId("evaluationStatus").textContent = "Starting " + method + " rollout validation...";
  try {
    var response = await fetch("/api/baselines/run/" + encodeURIComponent(record.id), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        baseline: method,
        gpu: gpu,
        memory_utilization: memory,
        instruction_condition: condition,
        options: method === "procvlm" ? { procvlm_enable_value_head: byId("procvlmEnableValueHead").checked } : {}
      })
    });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not start baseline");
    state.baselineJob = payload.job;
    rememberPersistentJob(payload.job);
    byId("evaluationStatus").textContent = "Running " + method + "  -  job " + payload.job.job_id + "...";
    pollBaselineJob(payload.job.job_id, record.id);
  } catch (error) {
    byId("evaluationStatus").textContent = "Baseline run error: " + error.message;
  }
}

async function pollBaselineJob(jobId, rolloutId) {
  return pollPersistentJob(jobId);
}


function installEvents() {
  try {
    var collapsed = JSON.parse(
      sessionStorage.getItem("lf3r.results.baselineCollapsed") || "{}"
    );
    if (collapsed && typeof collapsed === "object") {
      state.baselineCollapsed = collapsed;
    }
    state.roboPosthocCheckpoint =
      sessionStorage.getItem("lf3r.results.roboPosthocCheckpoint") || "";
  } catch (_error) {}
  ["searchInput", "originFilter", "manifestFilter", "outcomeFilter", "reviewFilter"].forEach(function (id) {
    byId(id).addEventListener(id === "searchInput" ? "input" : "change", applyFilters);
  });
  byId("frameSlider").addEventListener("input", function (event) {
    seekFrame(Number(event.target.value));
  });
  document.querySelectorAll("[data-step]").forEach(function (button) {
    button.addEventListener("click", function () { stepFrames(Number(button.dataset.step)); });
  });
  byId("addFailureEvent").addEventListener("click", function () {
    addFailureEvent();
  });
  byId("failureEvents").addEventListener("click", function (event) {
    var removeButton = event.target.closest("[data-remove-event]");
    if (removeButton) {
      removeFailureEvent(Number(removeButton.dataset.removeEvent));
      return;
    }
    var setButton = event.target.closest("[data-event-set]");
    if (setButton) {
      var setCard = setButton.closest(".failure-event-card");
      setActiveFailureEvent(Number(setCard.dataset.eventIndex));
      setActiveEventFrame(setButton.dataset.eventSet, currentFrame());
      return;
    }
    var clearButton = event.target.closest("[data-event-clear]");
    if (clearButton) {
      var clearCard = clearButton.closest(".failure-event-card");
      setActiveFailureEvent(Number(clearCard.dataset.eventIndex));
      setActiveEventFrame(clearButton.dataset.eventClear, null);
      return;
    }
    var card = event.target.closest(".failure-event-card");
    if (card) setActiveFailureEvent(Number(card.dataset.eventIndex));
  });
  byId("failureEvents").addEventListener("input", function (event) {
    if (!event.target.dataset.eventField) return;
    syncFailureEventInput(event.target);
    markDirty();
  });
  byId("failureEvents").addEventListener("focusin", function (event) {
    var card = event.target.closest(".failure-event-card");
    if (card) setActiveFailureEvent(Number(card.dataset.eventIndex));
  });
  byId("outcomeLabel").addEventListener("change", function () {
    if ((this.value === "failure" || this.value === "recovered_success") && !state.failureEvents.length) {
      addFailureEvent();
    }
  });
  byId("annotationForm").addEventListener("submit", saveAnnotation);
  byId("annotationForm").addEventListener("input", markDirty);
  byId("playButton").addEventListener("click", togglePlayback);
  byId("playOverlay").addEventListener("click", togglePlayback);
  byId("rolloutVideo").addEventListener("click", togglePlayback);
  byId("rolloutVideo").addEventListener("play", function () {
    byId("playButton").textContent = "Pause";
    byId("playOverlay").classList.add("hidden");
  });
  byId("rolloutVideo").addEventListener("pause", function () {
    byId("playButton").textContent = "Play";
    byId("playOverlay").classList.remove("hidden");
  });
  byId("rolloutVideo").addEventListener("timeupdate", function () {
    var record = selectedRollout();
    if (!record || byId("rolloutVideo").seeking) return;
    state.currentFrame = Math.min(
      Number(record.total_frames) - 1,
      Math.floor(byId("rolloutVideo").currentTime * Number(record.fps) + 0.0001)
    );
    byId("frameSlider").value = state.currentFrame;
    updateReadout();
  });
  byId("speedSelect").addEventListener("change", function () {
    byId("rolloutVideo").playbackRate = Number(byId("speedSelect").value);
  });
  byId("previousButton").addEventListener("click", function () { navigate(-1); });
  byId("nextButton").addEventListener("click", function () { navigate(1); });
  byId("instructionCondition").addEventListener("change", function () {
    state.instructionCondition = this.value || "full_instruction";
    state.baselineRuns = null;
    state.baselineRunsCondition = null;
    state.baselineRunsLoading = null;
    var batchCondition = byId("baselineBatchCondition");
    if (batchCondition) {
      batchCondition.value = state.instructionCondition;
      baselineBatchScopeChanged();
    }
    var record = selectedRollout();
    if (!record) return;
    renderInstructionVariantControl(record);
    loadEvaluation(record.id);
  });
  bindResultsEvents();
  byId("baselineBatchMethod").addEventListener("change", function () {
    updateBaselineBatchAdvancedFields();
    if (baselineBatchResultFilterValue() === "missing_valid") {
      baselineBatchCoverageChanged();
    } else {
      updateBaselineBatchSelection();
    }
  });
  byId("baselineBatchScope").addEventListener("change", function () {
    if (baselineBatchResultFilterValue() === "missing_valid") {
      baselineBatchCoverageChanged();
    } else {
      baselineBatchScopeChanged();
    }
  });
  byId("baselineBatchCondition").addEventListener("change", function () {
    if (baselineBatchResultFilterValue() === "missing_valid") {
      baselineBatchCoverageChanged();
    } else {
      baselineBatchScopeChanged();
    }
  });
  byId("baselineBatchResultFilter").addEventListener("change", baselineBatchCoverageChanged);
  byId("baselineBatchStartIndex").addEventListener("input", baselineBatchRangeChanged);
  byId("baselineBatchLimit").addEventListener("input", updateBaselineBatchSelection);
  byId("baselineBatchEndIndex").addEventListener("input", baselineBatchRangeChanged);
  byId("baselineBatchAddWorker").addEventListener("click", addBaselineBatchWorker);
  byId("baselineBatchWorkers").addEventListener("input", function () {
    readBaselineBatchWorkers();
    updateBaselineBatchSelection();
  });
  byId("baselineBatchWorkers").addEventListener("click", function (event) {
    var button = event.target.closest("[data-remove-worker]");
    if (!button) return;
    removeBaselineBatchWorker(Number(button.dataset.removeWorker));
  });
  byId("baselineBatchForm").addEventListener("submit", startBaselineBatch);
  ["rolloutGenerationTaskStart", "rolloutGenerationTaskEnd", "rolloutGenerationTrials", "rolloutGenerationRenderResolution", "rolloutGenerationRecordResolution"].forEach(function (id) {
    byId(id).addEventListener("input", updateRolloutGenerationSelection);
  });
  byId("rolloutGenerationVideoViewMode").addEventListener("change", updateRolloutGenerationSelection);
  byId("rolloutGenerationLogSafeFeatures").addEventListener("change", updateRolloutGenerationSelection);
  byId("rolloutGenerationSuite").addEventListener("change", function () {
    var isSpatial = byId("rolloutGenerationSuite").value === "libero_spatial";
    byId("rolloutGenerationRenderResolution").value = "256";
    byId("rolloutGenerationRecordResolution").value = isSpatial ? "256" : "224";
    updateRolloutGenerationSelection();
  });
  byId("rolloutGenerationForm").addEventListener("submit", startRolloutGeneration);
  document.addEventListener("click", function (event) {
    if (window.LF3RRunsJobs && typeof window.LF3RRunsJobs.handleClick === "function"
        && window.LF3RRunsJobs.handleClick(event)) {
      return;
    }
    var button = event.target.closest("[data-persistent-job-log]");
    if (!button) return;
    var output = null;
    document.querySelectorAll("[data-persistent-job-log-output]").forEach(function (node) {
      if (node.dataset.persistentJobLogOutput === button.dataset.persistentJobLog) output = node;
    });
    if (output && !output.hidden) {
      state.persistentJobLogOpen[button.dataset.persistentJobLog] = false;
      output.hidden = true;
      button.textContent = "View log";
      button.setAttribute("aria-expanded", "false");
      return;
    }
    loadPersistentJobLog(
      button.dataset.persistentJobLog,
      button.dataset.persistentJobType,
      button
    );
  });
  document.addEventListener("pointerover", function (event) {
    var target = cliHelpTarget(event);
    if (target) {
      if (cliHelpState.target !== target) showCliHelp(target);
    }
  });
  document.addEventListener("pointerout", function (event) {
    var target = cliHelpTarget(event);
    if (target && (!event.relatedTarget || !target.contains(event.relatedTarget))) scheduleHideCliHelp();
  });
  document.addEventListener("focusin", function (event) {
    var target = cliHelpTarget(event);
    if (target) showCliHelp(target);
  });
  document.addEventListener("focusout", function (event) {
    var target = cliHelpTarget(event);
    if (target && (!event.relatedTarget || !target.contains(event.relatedTarget))) scheduleHideCliHelp();
  });
  window.addEventListener("resize", function () {
    if (cliHelpState.target) positionCliHelp(cliHelpState.target);
  });
  loadCliHelpMetadata();
  updateBaselineBatchAdvancedFields();
  updateRolloutGenerationSelection();
  window.lf3rRenderJobCards = renderPersistentJobCards;
  window.lf3rGetPersistentJobs = function (jobType) {
    return (state.persistentJobs || []).filter(function (job) {
      return !jobType || job.job_type === jobType;
    });
  };
  window.lf3rRefreshPersistentJobs = loadPersistentJobs;
  window.setTimeout(loadPersistentJobs, 0);
  window.addEventListener("beforeunload", function (event) {
    if (state.dirty) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  document.addEventListener("keydown", function (event) {
    if (["annotate", "results"].indexOf(document.body.dataset.view) === -1) return;
    if (event.key === "/" && !isTypingTarget(event.target)) {
      event.preventDefault();
      byId("searchInput").focus();
      return;
    }
    if (isTypingTarget(event.target)) return;
    if (document.body.dataset.view === "results" && ["1", "2", "3", "s", "S"].indexOf(event.key) !== -1) return;
    if (event.code === "Space") {
      event.preventDefault();
      togglePlayback();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      stepFrames(event.shiftKey ? -10 : -1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      stepFrames(event.shiftKey ? 10 : 1);
    } else if (event.key === "1") {
      setActiveEventFrame("causal_onset_frame", currentFrame());
    } else if (event.key === "2") {
      setActiveEventFrame("observable_onset_frame", currentFrame());
    } else if (event.key === "3") {
      setActiveEventFrame("terminal_failure_frame", currentFrame());
    } else if (event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveAnnotation();
    } else if (event.key.toLowerCase() === "n") {
      navigate(1);
    } else if (event.key.toLowerCase() === "p") {
      navigate(-1);
    }
  });
}

installEvents();
loadRollouts().catch(function (error) {
  byId("datasetStatus").textContent = "Dataset error";
  byId("rolloutList").innerHTML = '<div class="form-error">' + escapeHtml(error.message) + "</div>";
});

// Chart interaction is delegated because cards are replaced when changing runs.
document.addEventListener('click', function (event) {
  var plot = event.target.closest('[data-signal-seek]');
  if (!plot) return;
  var box = plot.getBoundingClientRect();
  if (box.width > 0) seekFrame((event.clientX - box.left) / box.width * Number(plot.dataset.frameMax));
});
document.addEventListener('keydown', function (event) {
  var plot = event.target.closest('[data-signal-seek]');
  if (!plot || ['ArrowLeft', 'ArrowRight', 'Home', 'End'].indexOf(event.key) === -1) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  seekFrame(event.key === 'Home' ? 0 : event.key === 'End' ? Number(plot.dataset.frameMax) : currentFrame() + (event.key === 'ArrowLeft' ? -1 : 1));
}, true);

byId('rolloutVideo').addEventListener('seeked', function () {
  var record = selectedRollout();
  if (!record) return;
  state.currentFrame = Math.max(0, Math.min(Number(record.total_frames) - 1, Math.floor(this.currentTime * Number(record.fps) + 0.0001)));
  byId('frameSlider').value = state.currentFrame;
  updateReadout();
});

(function installResultsVideoPin() {
  var toggle = byId('resultsPinVideo');
  var dock = byId('resultsVideoDock');
  try { toggle.checked = localStorage.getItem('lf3r.results.pinVideo') === 'true'; } catch (_) {}
  function applyPin() {
    dock.classList.toggle('is-pinned', toggle.checked);
    try { localStorage.setItem('lf3r.results.pinVideo', String(toggle.checked)); } catch (_) {}
  }
  toggle.addEventListener('change', applyPin);
  applyPin();
})();
