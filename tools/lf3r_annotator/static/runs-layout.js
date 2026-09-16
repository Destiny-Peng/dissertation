"use strict";

(function installRunsLayoutEnhancements() {
  var view = document.getElementById("runsView");
  if (!view || view.dataset.runsEnhanced === "true") return;

  var batchPanel = view.querySelector(".batch-panel");
  var rolloutPanel = view.querySelector(".rollout-generation-panel");
  if (!batchPanel || !rolloutPanel) return;
  view.dataset.runsEnhanced = "true";

  /* Keep the additional Runs styling isolated from the legacy stylesheet. */
  if (!document.querySelector('link[data-runs-tools-style="true"]')) {
    var styleLink = document.createElement("link");
    styleLink.rel = "stylesheet";
    styleLink.href = "/static/styles-tools.css?v=non-analysis-tools-20260916";
    styleLink.dataset.runsToolsStyle = "true";
    document.head.appendChild(styleLink);
  }

  var legacyTitle = view.querySelector(":scope > h2");
  if (legacyTitle) legacyTitle.classList.add("runs-legacy-title");

  var header = document.createElement("header");
  header.className = "runs-shell-header";
  header.innerHTML = [
    '<div class="runs-shell-copy">',
      '<p class="eyebrow">EXPERIMENT CONTROL</p>',
      '<h2>Runs</h2>',
      '<p>Inference, rollout generation, SAFE training, and project utilities. Analysis tooling stays separate.</p>',
    '</div>',
    '<div class="runs-mode-switch" role="tablist" aria-label="Run type">',
      '<button id="runsBaselineTab" type="button" role="tab" data-runs-mode="baseline">Baseline inference</button>',
      '<button id="runsRolloutTab" type="button" role="tab" data-runs-mode="rollout">Rollout generation</button>',
      '<button id="runsSafeTab" type="button" role="tab" data-runs-mode="safe">SAFE training</button>',
      '<button id="runsToolsTab" type="button" role="tab" data-runs-mode="tools">Utilities</button>',
    '</div>'
  ].join("");

  var gpuStrip = document.createElement("section");
  gpuStrip.className = "runs-gpu-strip";
  gpuStrip.innerHTML = [
    '<div class="runs-gpu-heading">',
      '<div><p class="eyebrow">GPU STATUS</p><strong>Current device state</strong><span id="runsGpuTimestamp">Not refreshed yet</span></div>',
      '<button id="runsGpuRefresh" class="ghost-button" type="button">Refresh GPU</button>',
    '</div>',
    '<div id="runsGpuCards" class="runs-gpu-cards"><div class="runs-gpu-empty">Press Refresh GPU to query nvidia-smi.</div></div>'
  ].join("");

  var workspace = document.createElement("div");
  workspace.className = "runs-console";
  batchPanel.parentNode.insertBefore(header, batchPanel);
  batchPanel.parentNode.insertBefore(gpuStrip, batchPanel);
  batchPanel.parentNode.insertBefore(workspace, batchPanel);
  workspace.appendChild(batchPanel);
  workspace.appendChild(rolloutPanel);

  batchPanel.dataset.runsPanel = "baseline";
  rolloutPanel.dataset.runsPanel = "rollout";
  batchPanel.id = batchPanel.id || "runsBaselinePanel";
  rolloutPanel.id = rolloutPanel.id || "runsRolloutPanel";
  batchPanel.setAttribute("role", "tabpanel");
  rolloutPanel.setAttribute("role", "tabpanel");
  batchPanel.setAttribute("aria-labelledby", "runsBaselineTab");
  rolloutPanel.setAttribute("aria-labelledby", "runsRolloutTab");

  function toolActivityMarkup() {
    return [
      '<aside class="runs-tool-activity">',
        '<div class="runs-activity-heading"><p class="eyebrow">ACTIVITY</p><h3>Project tool job</h3></div>',
        '<div id="projectToolStatus" class="runs-activity-status">No project-tool job selected.</div>',
        '<div id="projectToolJobs" class="runs-tool-job-list"></div>',
        '<pre id="projectToolLog" class="job-log runs-tool-log" aria-label="Project tool log"></pre>',
      '</aside>'
    ].join("");
  }

  var safePanel = document.createElement("section");
  safePanel.className = "evaluation-panel runs-custom-panel";
  safePanel.id = "runsSafePanel";
  safePanel.dataset.runsPanel = "safe";
  safePanel.setAttribute("role", "tabpanel");
  safePanel.setAttribute("aria-labelledby", "runsSafeTab");
  safePanel.innerHTML = [
    '<div class="evaluation-heading runs-custom-heading">',
      '<div><p class="eyebrow">SAFE WORKFLOW</p><h2>Prepare, train, and validate SAFE detectors</h2></div>',
    '</div>',
    '<div class="runs-custom-layout">',
      '<div class="runs-tool-stack">',
        '<section class="runs-tool-card">',
          '<div class="runs-tool-card-heading"><span class="runs-step">1</span><div><h3>Prepare dataset</h3><p>Stage official SAFE-format CSV / pickle files from the LF3R manifest.</p></div></div>',
          '<div class="runs-tool-grid">',
            '<label><span>Dataset role</span><select id="safePrepareRole"><option value="primary_natural">Primary natural</option><option value="reference_natural">Reference natural</option><option value="controlled_analysis">Controlled analysis</option><option value="all">All</option></select></label>',
            '<label><span>Partition</span><select id="safePreparePartition"><option value="natural_observation">Natural observation</option><option value="controlled_analysis">Controlled analysis</option><option value="all">All</option></select></label>',
            '<label class="runs-tool-wide"><span>Source run name (optional)</span><input id="safePrepareRunName" type="text" placeholder="Use when task/episode basenames are duplicated"></label>',
            '<label class="runs-tool-wide"><span>Output dataset directory</span><input id="safePrepareOutput" type="text"></label>',
          '</div>',
          '<div class="runs-tool-action"><span>Source rollout files are not modified.</span><button id="safePrepareRun" class="save-button" type="button">Prepare dataset</button></div>',
        '</section>',
        '<section class="runs-tool-card">',
          '<div class="runs-tool-card-heading"><span class="runs-step">2</span><div><h3>Train detector</h3><p>Run the official SAFE Hydra trainer through the LF3R wrapper.</p></div></div>',
          '<div class="runs-tool-grid">',
            '<label class="runs-tool-wide"><span>Dataset directory</span><input id="safeTrainDataset" type="text"></label>',
            '<label><span>Model</span><select id="safeTrainModel"><option value="mlp">SAFE-MLP</option><option value="lstm">SAFE-LSTM</option></select></label>',
            '<label><span>GPU</span><input id="safeTrainGpu" type="text" value="0"></label>',
            '<label><span>Epochs</span><input id="safeTrainEpochs" type="number" min="1" value="1000"></label>',
            '<label><span>Batch size</span><input id="safeTrainBatch" type="number" min="1" value="512"></label>',
            '<label><span>Hidden dim</span><input id="safeTrainHidden" type="number" min="1" value="256"></label>',
            '<label><span>Seed</span><input id="safeTrainSeed" type="text" value="0"></label>',
            '<label class="runs-tool-wide"><span>Logs root</span><input id="safeTrainLogs" type="text" value="outputs/safe_training/logs/web"></label>',
            '<label class="runs-check"><input id="safeTrainNormalize" type="checkbox"> Normalize hidden states</label>',
          '</div>',
          '<div class="runs-tool-action"><span>GPU-only; there is no CPU fallback.</span><button id="safeTrainRun" class="save-button" type="button">Train SAFE</button></div>',
        '</section>',
        '<section class="runs-tool-card">',
          '<div class="runs-tool-card-heading"><span class="runs-step">3</span><div><h3>Validate checkpoint</h3><p>Reload one trained checkpoint and write finite validation scores.</p></div></div>',
          '<div class="runs-tool-grid">',
            '<label class="runs-tool-wide"><span>Dataset directory</span><input id="safeValidateDataset" type="text"></label>',
            '<label class="runs-tool-wide"><span>Checkpoint</span><input id="safeValidateCheckpoint" type="text" placeholder="outputs/safe_training/logs/.../model_final.ckpt"></label>',
            '<label><span>Model</span><select id="safeValidateModel"><option value="mlp">SAFE-MLP</option><option value="lstm">SAFE-LSTM</option></select></label>',
            '<label><span>GPU</span><input id="safeValidateGpu" type="text" value="0"></label>',
            '<label class="runs-tool-wide"><span>Output JSON</span><input id="safeValidateOutput" type="text" value="outputs/safe_training/validation/web_scores.json"></label>',
          '</div>',
          '<div class="runs-tool-action"><span>This validates loading and finite scores; it does not retrain.</span><button id="safeValidateRun" class="ghost-button" type="button">Validate checkpoint</button></div>',
        '</section>',
      '</div>',
      toolActivityMarkup(),
    '</div>'
  ].join("");

  var toolsPanel = document.createElement("section");
  toolsPanel.className = "evaluation-panel runs-custom-panel";
  toolsPanel.id = "runsToolsPanel";
  toolsPanel.dataset.runsPanel = "tools";
  toolsPanel.setAttribute("role", "tabpanel");
  toolsPanel.setAttribute("aria-labelledby", "runsToolsTab");
  toolsPanel.innerHTML = [
    '<div class="evaluation-heading runs-custom-heading">',
      '<div><p class="eyebrow">PROJECT UTILITIES</p><h2>Data export, diagnostics, and maintenance</h2></div>',
    '</div>',
    '<div class="runs-custom-layout">',
      '<div class="runs-tool-stack">',
        '<section class="runs-tool-card">',
          '<div class="runs-tool-card-heading"><div><h3>Export rollout package</h3><p>Create a self-contained share package with videos, sidecars, annotations, and manifest.</p></div></div>',
          '<div class="runs-outcome-row" id="exportOutcomes">',
            '<label><input type="checkbox" value="failure" checked> Failure</label>',
            '<label><input type="checkbox" value="recovered_success"> Recovered success</label>',
            '<label><input type="checkbox" value="success"> Success</label>',
            '<label><input type="checkbox" value="uncertain"> Uncertain</label>',
          '</div>',
          '<div class="runs-tool-grid">',
            '<label><span>Dataset role</span><select id="exportDatasetRole"><option value="primary_natural">Primary natural</option><option value="reference_natural">Reference natural</option><option value="controlled_analysis">Controlled analysis</option><option value="all">All</option></select></label>',
            '<label><span>Review status</span><select id="exportReviewStatus"><option value="complete">Complete</option><option value="in_progress">In progress</option><option value="unreviewed">Unreviewed</option><option value="all">All</option></select></label>',
            '<label class="runs-tool-wide"><span>Output directory (optional)</span><input id="exportOutputDir" type="text" placeholder="Blank = timestamped outputs/shares package"></label>',
          '</div>',
          '<div class="runs-tool-action"><button id="exportDryRun" class="ghost-button" type="button">Preview selection</button><button id="exportRun" class="save-button" type="button">Create package</button></div>',
        '</section>',
        '<section class="runs-tool-card">',
          '<div class="runs-tool-card-heading"><div><h3>Robo-Dopamine interval sweep</h3><p>Run one checkpoint once and compare native sampling intervals without mixing outputs.</p></div></div>',
          '<div class="runs-tool-grid">',
            '<label class="runs-tool-wide"><span>Rollout</span><select id="roboSweepRollout"><option value="">Loading rollouts…</option></select></label>',
            '<label><span>Intervals</span><input id="roboSweepIntervals" type="text" value="2,5,10"></label>',
            '<label><span>GPU</span><input id="roboSweepGpu" type="text" value="0"></label>',
            '<label><span>Free-memory fraction</span><input id="roboSweepMemory" type="number" min="0.05" max="1" step="0.05" value="0.60"></label>',
            '<label class="runs-tool-wide"><span>Output parent</span><input id="roboSweepOutput" type="text" value="outputs/baselines/robo_interval_sweeps"></label>',
          '</div>',
          '<div class="runs-tool-action"><span>Diagnostic sweep only; it does not claim detector performance.</span><button id="roboSweepRun" class="save-button" type="button">Run interval sweep</button></div>',
        '</section>',
        '<section class="runs-tool-card runs-maintenance-card">',
          '<div class="runs-tool-card-heading"><div><h3>Validation & maintenance</h3><p>Bounded checks around existing project artifacts. No Analysis jobs are launched here.</p></div></div>',
          '<div class="runs-maintenance-actions">',
            '<button id="validateBaselinesRun" class="ghost-button" type="button">Validate baseline pipelines</button>',
            '<button id="validateVariantsRun" class="ghost-button" type="button">Validate instruction variants</button>',
            '<button id="rebuildManifestRun" class="ghost-button" type="button">Rescan rollout manifest</button>',
          '</div>',
        '</section>',
      '</div>',
      toolActivityMarkup().replace('id="projectToolStatus"', 'id="projectToolStatusTools"').replace('id="projectToolJobs"', 'id="projectToolJobsTools"').replace('id="projectToolLog"', 'id="projectToolLogTools"'),
    '</div>'
  ].join("");

  workspace.appendChild(safePanel);
  workspace.appendChild(toolsPanel);

  function wrapSetupBlock(node, title, subtitle) {
    if (!node || node.parentElement.classList.contains("runs-config-block")) return;
    var wrapper = document.createElement("section");
    wrapper.className = "runs-config-block";
    var heading = document.createElement("div");
    heading.className = "runs-config-heading";
    heading.innerHTML = '<h3>' + title + '</h3>' + (subtitle ? '<p>' + subtitle + '</p>' : '');
    node.parentNode.insertBefore(wrapper, node);
    wrapper.appendChild(heading);
    wrapper.appendChild(node);
  }

  wrapSetupBlock(batchPanel.querySelector(".batch-control-grid"), "Run setup", "Choose the method, scope, GPU allocation, and rollout range.");
  wrapSetupBlock(rolloutPanel.querySelector(".rollout-generation-grid"), "Generation setup", "Choose the dataset scope, rendering settings, and generation range.");

  function makeActivity(panel, jobsId, logId, title, statusId) {
    var jobs = document.getElementById(jobsId);
    var log = document.getElementById(logId);
    var status = statusId ? document.getElementById(statusId) : null;
    if (!jobs || !log || jobs.parentElement.classList.contains("runs-activity")) return;
    var activity = document.createElement("aside");
    activity.className = "runs-activity";
    activity.innerHTML = '<div class="runs-activity-heading"><p class="eyebrow">ACTIVITY</p><h3>' + title + '</h3></div>';
    jobs.parentNode.insertBefore(activity, jobs);
    if (status) {
      status.classList.add("runs-activity-status");
      activity.appendChild(status);
    }
    activity.appendChild(jobs);
    activity.appendChild(log);
  }

  makeActivity(batchPanel, "baselineBatchJobs", "baselineBatchLog", "Baseline job");
  makeActivity(rolloutPanel, "rolloutGenerationJobs", "rolloutGenerationLog", "Generation job", "rolloutGenerationStatus");

  var advanced = document.getElementById("baselineBatchAdvanced");
  if (advanced && !advanced.dataset.runsDisclosure) {
    advanced.dataset.runsDisclosure = "true";
    advanced.classList.add("runs-advanced", "is-collapsed");
    var originalHeading = advanced.querySelector(":scope > h3");
    if (originalHeading) originalHeading.hidden = true;
    var disclosure = document.createElement("button");
    disclosure.type = "button";
    disclosure.className = "runs-advanced-toggle";
    disclosure.setAttribute("aria-expanded", "false");
    disclosure.innerHTML = '<span><strong>Advanced runner parameters</strong><small>Model-specific overrides, validation, video rendering, and dry-run controls</small></span><i aria-hidden="true">⌄</i>';
    advanced.insertBefore(disclosure, advanced.firstChild);
    disclosure.addEventListener("click", function () {
      var open = advanced.classList.toggle("is-open");
      advanced.classList.toggle("is-collapsed", !open);
      disclosure.setAttribute("aria-expanded", String(open));
      disclosure.querySelector("i").textContent = open ? "⌃" : "⌄";
    });
  }

  function decorateAction(button, note) {
    if (!button || button.parentElement.classList.contains("runs-action-row")) return;
    var row = document.createElement("div");
    row.className = "runs-action-row";
    var helper = document.createElement("span");
    helper.textContent = note;
    button.parentNode.insertBefore(row, button);
    row.appendChild(helper);
    row.appendChild(button);
  }

  decorateAction(document.getElementById("baselineBatchRun"), "Starts a persistent background job using the configuration above.");
  decorateAction(document.getElementById("rolloutGenerationRun"), "Starts natural rollout generation using the configuration above.");

  var buttons = Array.prototype.slice.call(header.querySelectorAll("[data-runs-mode]"));
  var panels = Array.prototype.slice.call(workspace.querySelectorAll("[data-runs-panel]"));
  var validModes = ["baseline", "rollout", "safe", "tools"];
  var currentMode = "baseline";
  try {
    var storedMode = localStorage.getItem("lf3r.runs.mode");
    if (validModes.indexOf(storedMode) >= 0) currentMode = storedMode;
  } catch (_) {}

  buttons.forEach(function (button) {
    var panel = document.querySelector('[data-runs-panel="' + button.dataset.runsMode + '"]');
    if (panel) button.setAttribute("aria-controls", panel.id);
  });

  function setMode(mode) {
    currentMode = validModes.indexOf(mode) >= 0 ? mode : "baseline";
    buttons.forEach(function (button) {
      var active = button.dataset.runsMode === currentMode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    panels.forEach(function (panel) {
      panel.hidden = panel.dataset.runsPanel !== currentMode;
    });
    try { localStorage.setItem("lf3r.runs.mode", currentMode); } catch (_) {}
  }

  buttons.forEach(function (button, index) {
    button.addEventListener("click", function () { setMode(button.dataset.runsMode); });
    button.addEventListener("keydown", function (event) {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      var next = event.key === "ArrowRight" ? (index + 1) % buttons.length : (index - 1 + buttons.length) % buttons.length;
      buttons[next].focus();
      setMode(buttons[next].dataset.runsMode);
    });
  });

  function text(id) {
    var node = document.getElementById(id);
    return node ? String(node.value || "").trim() : "";
  }

  function numberValue(id, fallback) {
    var value = Number(text(id));
    return Number.isFinite(value) ? value : fallback;
  }

  function formatGiB(mib) {
    if (!Number.isFinite(Number(mib))) return "—";
    return (Number(mib) / 1024).toFixed(1) + " GiB";
  }

  async function refreshGpuStatus() {
    var button = document.getElementById("runsGpuRefresh");
    var cards = document.getElementById("runsGpuCards");
    var timestamp = document.getElementById("runsGpuTimestamp");
    button.disabled = true;
    button.textContent = "Refreshing…";
    try {
      var response = await fetch("/api/gpu-status", { cache: "no-store" });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not read GPU status");
      var status = payload.gpu_status || {};
      if (!status.available) throw new Error(status.error || "GPU status unavailable");
      var gpus = status.gpus || [];
      cards.innerHTML = gpus.length ? gpus.map(function (gpu) {
        var freeFraction = Number(gpu.memory_free_fraction);
        var freePercent = Number.isFinite(freeFraction) ? Math.round(freeFraction * 100) : 0;
        return [
          '<article class="runs-gpu-card">',
            '<div class="runs-gpu-card-top"><strong>GPU ' + gpu.index + '</strong><span>' + (gpu.gpu_utilization_percent == null ? '—' : gpu.gpu_utilization_percent + '% util') + '</span></div>',
            '<div class="runs-gpu-name">' + String(gpu.name || "NVIDIA GPU") + '</div>',
            '<div class="runs-gpu-memory"><span><b>' + formatGiB(gpu.memory_free_mib) + '</b> free / ' + formatGiB(gpu.memory_total_mib) + '</span><span>' + freePercent + '% free</span></div>',
            '<div class="runs-gpu-meter"><i style="width:' + Math.max(0, Math.min(100, freePercent)) + '%"></i></div>',
            '<div class="runs-gpu-meta"><span>' + (gpu.temperature_c == null ? '—' : gpu.temperature_c + '°C') + '</span><span>' + (gpu.power_draw_w == null ? '—' : gpu.power_draw_w.toFixed(0) + ' W') + '</span></div>',
          '</article>'
        ].join("");
      }).join("") : '<div class="runs-gpu-empty">No NVIDIA GPUs reported.</div>';
      timestamp.textContent = "Refreshed " + new Date(status.queried_at || Date.now()).toLocaleTimeString();
    } catch (error) {
      cards.innerHTML = '<div class="runs-gpu-empty error">' + String(error.message || error) + '</div>';
      timestamp.textContent = "Refresh failed";
    } finally {
      button.disabled = false;
      button.textContent = "Refresh GPU";
    }
  }

  document.getElementById("runsGpuRefresh").addEventListener("click", refreshGpuStatus);

  var toolPollTimer = null;
  var activeToolJobId = null;

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

  function renderToolJobs(jobs) {
    activityNodes().forEach(function (nodes) {
      if (!nodes.jobs) return;
      nodes.jobs.innerHTML = jobs.length ? jobs.slice(0, 5).map(function (job) {
        return '<button type="button" class="runs-tool-job' + (job.job_id === activeToolJobId ? ' active' : '') + '" data-tool-job="' + job.job_id + '"><strong>' + (job.tool_label || job.action || 'Project tool') + '</strong><span>' + (job.status || 'unknown') + '</span></button>';
      }).join("") : '<div class="runs-tool-job-empty">No project-tool jobs yet.</div>';
    });
    Array.prototype.slice.call(document.querySelectorAll("[data-tool-job]")).forEach(function (button) {
      button.addEventListener("click", function () {
        activeToolJobId = button.dataset.toolJob;
        refreshToolJobs();
        pollToolJob();
      });
    });
  }

  async function refreshToolJobs() {
    try {
      var response = await fetch("/api/tool-jobs", { cache: "no-store" });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not read project-tool jobs");
      var jobs = payload.jobs || [];
      if (!activeToolJobId && jobs.length) activeToolJobId = jobs[0].job_id;
      renderToolJobs(jobs);
    } catch (_) {}
  }

  async function pollToolJob() {
    if (toolPollTimer) clearTimeout(toolPollTimer);
    if (!activeToolJobId) return;
    try {
      var responses = await Promise.all([
        fetch("/api/tool-jobs/" + encodeURIComponent(activeToolJobId), { cache: "no-store" }),
        fetch("/api/tool-jobs/" + encodeURIComponent(activeToolJobId) + "/log?tail=300", { cache: "no-store" })
      ]);
      var jobPayload = await responses[0].json();
      var logPayload = await responses[1].json();
      if (!responses[0].ok) throw new Error(jobPayload.error || "Could not read job");
      var job = jobPayload.job || {};
      var logText = logPayload.log && logPayload.log.text ? logPayload.log.text : "";
      activityNodes().forEach(function (nodes) {
        if (nodes.status) nodes.status.textContent = (job.tool_label || job.action || "Project tool") + " · " + (job.status || "unknown") + (job.error ? " · " + job.error : "");
        if (nodes.log) nodes.log.textContent = logText;
      });
      await refreshToolJobs();
      if (job.status === "queued" || job.status === "running") {
        toolPollTimer = setTimeout(pollToolJob, 1000);
      }
    } catch (error) {
      activityNodes().forEach(function (nodes) {
        if (nodes.status) nodes.status.textContent = String(error.message || error);
      });
    }
  }

  async function submitTool(action, options) {
    activityNodes().forEach(function (nodes) {
      if (nodes.status) nodes.status.textContent = "Submitting " + action + "…";
    });
    var response = await fetch("/api/tools/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: action, options: options || {} })
    });
    var payload = await response.json();
    if (!response.ok) {
      var message = payload.error || "Project-tool request failed";
      activityNodes().forEach(function (nodes) { if (nodes.status) nodes.status.textContent = message; });
      throw new Error(message);
    }
    activeToolJobId = payload.job.job_id;
    await refreshToolJobs();
    pollToolJob();
    return payload.job;
  }

  var prepareStamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..*/, "").replace("T", "_");
  document.getElementById("safePrepareOutput").value = "outputs/safe_training/datasets/web_" + prepareStamp;
  document.getElementById("safeTrainDataset").value = document.getElementById("safePrepareOutput").value;
  document.getElementById("safeValidateDataset").value = document.getElementById("safePrepareOutput").value;

  document.getElementById("safePrepareRun").addEventListener("click", function () {
    var output = text("safePrepareOutput");
    document.getElementById("safeTrainDataset").value = output;
    document.getElementById("safeValidateDataset").value = output;
    submitTool("safe_prepare", {
      dataset_role: text("safePrepareRole"),
      partition: text("safePreparePartition"),
      run_name: text("safePrepareRunName"),
      output: output
    }).catch(function () {});
  });

  document.getElementById("safeTrainRun").addEventListener("click", function () {
    submitTool("safe_train", {
      dataset_dir: text("safeTrainDataset"),
      model: text("safeTrainModel"),
      gpu: text("safeTrainGpu"),
      epochs: numberValue("safeTrainEpochs", 1000),
      batch_size: numberValue("safeTrainBatch", 512),
      hidden_dim: numberValue("safeTrainHidden", 256),
      seed: text("safeTrainSeed"),
      logs_root: text("safeTrainLogs"),
      normalize: document.getElementById("safeTrainNormalize").checked
    }).catch(function () {});
  });

  document.getElementById("safeValidateRun").addEventListener("click", function () {
    submitTool("safe_validate", {
      dataset_dir: text("safeValidateDataset"),
      checkpoint: text("safeValidateCheckpoint"),
      model: text("safeValidateModel"),
      gpu: text("safeValidateGpu"),
      output: text("safeValidateOutput"),
      hidden_dim: numberValue("safeTrainHidden", 256)
    }).catch(function () {});
  });

  function exportOptions(dryRun) {
    var outcomes = Array.prototype.slice.call(document.querySelectorAll("#exportOutcomes input:checked")).map(function (node) { return node.value; });
    return {
      outcomes: outcomes,
      dataset_role: text("exportDatasetRole"),
      review_status: text("exportReviewStatus"),
      output_dir: text("exportOutputDir"),
      dry_run: !!dryRun
    };
  }
  document.getElementById("exportDryRun").addEventListener("click", function () { submitTool("export_cases", exportOptions(true)).catch(function () {}); });
  document.getElementById("exportRun").addEventListener("click", function () { submitTool("export_cases", exportOptions(false)).catch(function () {}); });

  document.getElementById("roboSweepRun").addEventListener("click", function () {
    var intervals = text("roboSweepIntervals").split(",").map(function (item) { return Number(item.trim()); }).filter(function (item) { return Number.isInteger(item) && item > 0; });
    submitTool("robo_interval_sweep", {
      rollout_ids: [text("roboSweepRollout")],
      intervals: intervals,
      gpu: text("roboSweepGpu"),
      memory_utilization: numberValue("roboSweepMemory", 0.6),
      output_dir: text("roboSweepOutput")
    }).catch(function () {});
  });

  document.getElementById("validateBaselinesRun").addEventListener("click", function () { submitTool("validate_baselines", { check_environments: true }).catch(function () {}); });
  document.getElementById("validateVariantsRun").addEventListener("click", function () { submitTool("validate_variants", {}).catch(function () {}); });
  document.getElementById("rebuildManifestRun").addEventListener("click", function () { submitTool("rebuild_manifest", {}).catch(function () {}); });

  async function loadRolloutOptions() {
    var select = document.getElementById("roboSweepRollout");
    try {
      var response = await fetch("/api/rollouts", { cache: "no-store" });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not load rollouts");
      var rows = payload.rollouts || [];
      select.innerHTML = rows.map(function (row) {
        var label = (row.task_suite || "") + " · task " + row.task_id + " · ep " + row.episode_index + " · " + row.id;
        return '<option value="' + row.id + '">' + label + '</option>';
      }).join("");
    } catch (error) {
      select.innerHTML = '<option value="">' + String(error.message || error) + '</option>';
    }
  }

  setMode(currentMode);
  refreshToolJobs().then(function () { if (activeToolJobId) pollToolJob(); });
  loadRolloutOptions();
})();
