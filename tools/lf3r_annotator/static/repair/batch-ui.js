"use strict";

(function createRepairBatchUi() {
  var installed = false;
  var validatedKey = "";
  var pollTimer = null;
  var refreshTimer = null;

  function node(id) { return document.getElementById(id); }
  function state() {
    return window.LF3RRepairSyntheticSuffix
      ? window.LF3RRepairSyntheticSuffix.state
      : null;
  }
  function esc(value) {
    return typeof escapeHtml === "function"
      ? escapeHtml(value)
      : String(value == null ? "" : value);
  }

  function unique(rows, key) {
    return Array.from(new Set((rows || []).map(function (row) {
      return String(row[key] == null ? "" : row[key]);
    }).filter(Boolean))).sort();
  }

  function isBatch() {
    var select = node("repairRunMode");
    return Boolean(select && select.value === "batch");
  }

  function selectedModel() {
    return String((node("repairWorldModel") || {}).value || "a2world");
  }

  function modelLabel() {
    return selectedModel() === "wan2_2" ? "Wan2.2-I2V-A14B" : (selectedModel() === "ctrl_world" ? "Ctrl-World" : "A2World");
  }

  function checkpointDefault() {
    var type = String((node("repairCheckpointType") || {}).value || "libero_adapted");
    if (type === "generic_pretrained") return "checkpoints/a2world-pretrained.pt";
    if (type === "libero_adapted") return "checkpoints/a2world-libero.pt";
    return "";
  }

  function optionalText(id) {
    var target = node(id);
    var value = target ? String(target.value || "").trim() : "";
    return value || null;
  }

  function commonPayload() {
    var model = selectedModel();
    var worldModel;
    if (model === "wan2_2") {
      worldModel = {name: "wan2_2", python: optionalText("repairWanPython"),
                    checkpoint: optionalText("repairWanCheckpoint"), source_root: optionalText("repairWanSourceRoot")};
    } else if (model === "ctrl_world") {
      worldModel = {
        name: "ctrl_world",
        checkpoint_type: "droid_pretrained",
        checkpoint: optionalText("repairCtrlCheckpoint"),
        source_root: optionalText("repairCtrlSourceRoot"),
        python: optionalText("repairCtrlPython"),
        svd_model_path: optionalText("repairCtrlSvd"),
        clip_model_path: optionalText("repairCtrlClip"),
        data_stat_path: optionalText("repairCtrlDataStat"),
        target_fps: Math.max(0.1, Number(node("repairCtrlTargetFps").value || 5)),
        num_inference_steps: Math.max(
          1,
          Math.round(Number(node("repairCtrlInferenceSteps").value || 50))
        ),
        guidance_scale: Math.max(0, Number(node("repairCtrlGuidance").value || 1)),
        seed: Math.round(Number(node("repairCtrlSeed").value || 0)),
        text_conditioning: Boolean(node("repairCtrlTextConditioning").checked),
        camera_mapping: {
          exterior_1: "cam_high",
          exterior_2: "cam_high",
          wrist: "cam_wrist"
        }
      };
      Object.keys(worldModel).forEach(function (key) {
        if (worldModel[key] == null) delete worldModel[key];
      });
    } else {
      worldModel = {
        name: "a2world",
        checkpoint_type: node("repairCheckpointType").value,
        checkpoint: String(node("repairCheckpoint").value || "").trim() || checkpointDefault(),
        base_checkpoints: String(node("repairBaseCheckpoints").value || "").trim() || "checkpoints",
        duplicate_missing_views: Boolean(node("repairDuplicateViews").checked),
        num_sampling_steps: Math.max(
          1,
          Math.round(Number(node("repairSamplingSteps").value || 35))
        ),
        guidance: Math.max(0, Number(node("repairGuidance").value || 0)),
        seed: Math.round(Number(node("repairSeed").value || 0)),
        history: Boolean(node("repairHistory").checked),
        camera_mapping: {
          agentview: "cam_high",
          eye_in_hand: "cam_wrist"
        }
      };
    }
    var gpu = String(node("repairGpu").value || "").trim();
    if (!/^\d+$/.test(gpu)) throw new Error("GPU must be one numeric CUDA device index.");

    var cutType = String(node("repairCutType").value || "progress");
    var payload = {
      gpu_index: Number(gpu),
      cut_type: cutType,
      alignment_min_psnr: Number(node("repairAlignmentPsnr").value || 20),
      generated_includes_condition: model === "wan2_2",
      world_model: worldModel
    };
    if (cutType === "frame") {
      payload.cut_frame = Math.max(0, Math.round(Number(node("repairCutFrame").value || 0)));
    } else {
      payload.cut_progress = Math.max(
        0.01,
        Math.min(0.99, Number(node("repairCutProgress").value || 50) / 100)
      );
    }
    return payload;
  }

  function optionLabelForManifest(value, rows) {
    var match = (rows || []).find(function (row) {
      return String(row.manifest_source || "") === value;
    });
    return match && match.manifest_label ? String(match.manifest_label) : value;
  }

  function refillSelect(select, values, rows, kind) {
    if (!select) return;
    var previous = select.value;
    select.innerHTML = values.map(function (value) {
      var label = kind === "manifest" ? optionLabelForManifest(value, rows) : value;
      return '<option value="' + esc(value) + '">' + esc(label) + "</option>";
    }).join("");
    if (values.indexOf(previous) !== -1) select.value = previous;
  }

  function refreshOptions() {
    var current = state();
    if (!current || !current.rollouts || !current.rollouts.length) return;
    var rows = current.rollouts.filter(function (row) {
      var eligibility = (row.model_eligibility || {})[selectedModel()];
      return Boolean(eligibility ? eligibility.eligible : row.repair_eligible);
    });
    var manifests = unique(rows, "manifest_source");
    var suites = unique(rows, "task_suite");
    refillSelect(node("repairBatchManifest"), manifests, rows, "manifest");
    refillSelect(node("repairBatchSuite"), suites, rows, "suite");

    var selected = rows.find(function (row) {
      return row.id === current.selectedRolloutId;
    });
    if (selected) {
      if (node("repairBatchManifest") && !node("repairBatchManifest").dataset.initialized) {
        node("repairBatchManifest").value = String(selected.manifest_source || manifests[0] || "");
        node("repairBatchManifest").dataset.initialized = "true";
      }
      if (node("repairBatchSuite") && !node("repairBatchSuite").dataset.initialized) {
        node("repairBatchSuite").value = String(selected.task_suite || suites[0] || "");
        node("repairBatchSuite").dataset.initialized = "true";
      }
    }
    updateTaskBounds();
    renderSelection();
  }

  function scopedRows() {
    var current = state();
    if (!current) return [];
    var manifest = String((node("repairBatchManifest") || {}).value || "");
    var suite = String((node("repairBatchSuite") || {}).value || "");
    return (current.rollouts || []).filter(function (row) {
      var eligibility = (row.model_eligibility || {})[selectedModel()];
      return Boolean(eligibility ? eligibility.eligible : row.repair_eligible)
        && String(row.manifest_source || "") === manifest
        && String(row.task_suite || "") === suite;
    });
  }

  function numericTasks(rows) {
    return Array.from(new Set((rows || []).map(function (row) {
      var value = Number(row.task_id);
      return Number.isInteger(value) ? value : null;
    }).filter(function (value) { return value != null; }))).sort(function (a, b) {
      return a - b;
    });
  }

  function updateTaskBounds() {
    var tasks = numericTasks(scopedRows());
    if (!tasks.length) return;
    var minTask = tasks[0];
    var maxTask = tasks[tasks.length - 1];
    ["repairBatchTaskStart", "repairBatchTaskEnd"].forEach(function (id) {
      var input = node(id);
      if (!input) return;
      input.min = String(minTask);
      input.max = String(maxTask);
    });
    if (!node("repairBatchTaskStart").dataset.initialized) {
      node("repairBatchTaskStart").value = String(minTask);
      node("repairBatchTaskStart").dataset.initialized = "true";
    }
    if (!node("repairBatchTaskEnd").dataset.initialized) {
      node("repairBatchTaskEnd").value = String(maxTask);
      node("repairBatchTaskEnd").dataset.initialized = "true";
    }
  }

  function selection() {
    var rows = scopedRows();
    var start = Number(node("repairBatchTaskStart").value);
    var end = Number(node("repairBatchTaskEnd").value);
    var perTask = Number(node("repairBatchDemosPerTask").value);
    if (!Number.isInteger(start) || !Number.isInteger(end) || start > end) {
      throw new Error("Batch task range is invalid.");
    }
    if (!Number.isInteger(perTask) || perTask < 1 || perTask > 50) {
      throw new Error("Demos per task must be an integer from 1 to 50.");
    }
    var selected = [];
    for (var task = start; task <= end; task += 1) {
      var taskRows = rows.filter(function (row) {
        return Number(row.task_id) === task;
      }).sort(function (left, right) {
        var a = Number(left.episode_index);
        var b = Number(right.episode_index);
        if (Number.isFinite(a) && Number.isFinite(b) && a !== b) return a - b;
        return String(left.id || "").localeCompare(String(right.id || ""));
      });
      selected = selected.concat(taskRows.slice(0, perTask));
    }
    if (selected.length > 500) {
      throw new Error("One Repair batch is limited to 500 rollouts.");
    }
    return selected;
  }

  function renderSelection() {
    var target = node("repairBatchSelection");
    if (!target) return;
    try {
      var rows = selection();
      var start = Number(node("repairBatchTaskStart").value);
      var end = Number(node("repairBatchTaskEnd").value);
      var perTask = Number(node("repairBatchDemosPerTask").value);
      var expected = Math.max(0, end - start + 1) * perTask;
      var taskCounts = {};
      rows.forEach(function (row) {
        var key = String(row.task_id);
        taskCounts[key] = (taskCounts[key] || 0) + 1;
      });
      var missing = [];
      for (var task = start; task <= end; task += 1) {
        if (!taskCounts[String(task)]) missing.push(task);
      }
      target.innerHTML = '<small>Batch selection</small><strong>'
        + esc(rows.length + " rollout(s) selected · requested up to " + expected)
        + (missing.length ? esc(" · no eligible demos for task " + missing.join(", ")) : "")
        + "</strong>";
    } catch (error) {
      target.innerHTML = '<small>Batch selection</small><strong class="repair-error">'
        + esc(error.message) + "</strong>";
    }
  }

  function batchPayload() {
    var rows = selection();
    if (!rows.length) throw new Error("No Repair-eligible rollouts match the batch range.");
    var payload = commonPayload();
    payload.rollout_id = rows[0].id;
    payload.rollout_ids = rows.map(function (row) { return row.id; });
    payload.batch_selection = {
      manifest_source: String(node("repairBatchManifest").value || ""),
      task_suite: String(node("repairBatchSuite").value || ""),
      task_start: Number(node("repairBatchTaskStart").value),
      task_end: Number(node("repairBatchTaskEnd").value),
      demos_per_task: Number(node("repairBatchDemosPerTask").value),
      selected_rollouts: rows.length
    };
    return payload;
  }

  function validationKey(payload) {
    return JSON.stringify({
      rollout_ids: payload.rollout_ids,
      gpu_index: payload.gpu_index,
      cut_type: payload.cut_type,
      cut_progress: payload.cut_progress,
      cut_frame: payload.cut_frame,
      alignment_min_psnr: payload.alignment_min_psnr,
      world_model: payload.world_model
    });
  }

  function invalidate() {
    validatedKey = "";
    if (!isBatch()) return;
    var button = node("repairRunButton");
    if (button) button.disabled = true;
    var validation = node("repairValidation");
    if (validation) {
      validation.className = "repair-validation";
      validation.textContent = "Validate the batch configuration before launch. All selected rollouts are preflighted again on the server before the batch job is created.";
    }
  }

  function syncMode() {
    var batch = isBatch();
    document.querySelectorAll("[data-repair-batch-field]").forEach(function (field) {
      field.classList.toggle("hidden", !batch);
    });
    var validationButton = node("repairValidateButton");
    if (validationButton) validationButton.textContent = batch ? "Validate batch" : "Validate inputs";
    var runButton = node("repairRunButton");
    if (runButton) runButton.textContent = (batch ? "Run batch " : "Run ") + modelLabel();
    if (batch) {
      refreshOptions();
      invalidate();
    } else if (node("repairWorldModel")) {
      // Let the existing single-run controller restore its validation state.
      node("repairWorldModel").dispatchEvent(new Event("change", { bubbles: false }));
    }
  }

  async function validateBatch() {
    var target = node("repairValidation");
    var button = node("repairValidateButton");
    if (button) button.disabled = true;
    try {
      var payload = batchPayload();
      if (target) {
        target.className = "repair-validation";
        target.textContent = "Validating representative rollout and shared model/runtime inputs…";
      }
      var response = await fetch("/api/repair/synthetic-suffix/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      var data = await response.json();
      if (!response.ok) throw new Error(data.error || "Batch validation failed");
      var validation = data.validation || {};
      var blockers = validation.blockers || [];
      if (!validation.ready) {
        if (target) {
          target.className = "repair-validation repair-error";
          target.textContent = "Batch blocked.\n- " + blockers.join("\n- ");
        }
        validatedKey = "";
        node("repairRunButton").disabled = true;
        return false;
      }
      validatedKey = validationKey(payload);
      if (target) {
        target.className = "repair-validation repair-ok";
        target.textContent = [
          "Batch representative validation passed.",
          "Selected rollouts: " + payload.rollout_ids.length,
          "model = " + String(validation.model_name || selectedModel()),
          selectedModel() === "wan2_2" ? [
            "condition RGB frame: cam_high[" + validation.alignment.condition_frame + "]",
            "instruction: " + String((validation.world_model || {}).instruction || "unavailable"),
            "Wan Python: " + String((validation.world_model || {}).python || "unset"),
            "checkpoint: " + String((validation.world_model || {}).checkpoint || "unset"),
            "Actions / states / LIBERO runtime: not required"
          ].join("\n") : "",
          "GPU " + String(payload.gpu_index),
          "cut = " + (payload.cut_type === "frame"
            ? ("frame " + payload.cut_frame)
            : (Math.round(Number(payload.cut_progress) * 100) + "% progress")),
          "All selected rollouts will be preflighted before the persistent batch job is launched.",
          "Execution is sequential on the selected GPU."
        ].join("\n");
      }
      node("repairRunButton").disabled = false;
      return true;
    } catch (error) {
      validatedKey = "";
      if (target) {
        target.className = "repair-validation repair-error";
        target.textContent = "Batch validation failed: " + String(error.message || error);
      }
      if (node("repairRunButton")) node("repairRunButton").disabled = true;
      return false;
    } finally {
      if (button) button.disabled = false;
    }
  }

  function jobMessage(job) {
    var done = Number(job.completed_runs || 0);
    var failed = Number(job.failed_runs || 0);
    var total = Number(job.expected_runs || job.selected_rollouts || 0);
    return String(job.status || "queued") + " · " + String(job.phase || "queued")
      + " · " + done + "/" + total + " complete"
      + (failed ? " · " + failed + " failed" : "")
      + (job.current_rollout ? " · " + String(job.current_rollout) : "");
  }

  async function pollBatch(jobId) {
    if (pollTimer) window.clearTimeout(pollTimer);
    try {
      var response = await fetch(
        "/api/repair/synthetic-suffix/jobs/" + encodeURIComponent(jobId),
        { cache: "no-store" }
      );
      var data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not read Repair batch job");
      var job = data.job || {};
      var current = state();
      if (current) current.job = job;
      if (node("repairJobStatus")) node("repairJobStatus").textContent = jobMessage(job);
      if (typeof window.rememberPersistentJob === "function") {
        window.rememberPersistentJob(job);
      }
      if (window.LF3RRepairJobs && typeof window.LF3RRepairJobs.render === "function"
          && typeof window.lf3rGetPersistentJobs === "function") {
        window.LF3RRepairJobs.render(window.lf3rGetPersistentJobs("repair_synthetic_suffix"));
      }
      if (["complete", "complete_with_errors", "failed"].indexOf(job.status) !== -1) {
        if (node("repairRunButton") && isBatch()) {
          var payload = batchPayload();
          node("repairRunButton").disabled = validatedKey !== validationKey(payload);
        }
        if (window.LF3RRepairJobs && typeof window.LF3RRepairJobs.refresh === "function") {
          window.LF3RRepairJobs.refresh();
        }
        if (window.LF3RRepairSyntheticSuffix
            && typeof window.LF3RRepairSyntheticSuffix.refresh === "function") {
          window.LF3RRepairSyntheticSuffix.refresh();
        }
        return;
      }
    } catch (error) {
      if (node("repairJobStatus")) {
        node("repairJobStatus").textContent = "Repair batch status error: " + String(error.message || error);
      }
    }
    pollTimer = window.setTimeout(function () { pollBatch(jobId); }, 1200);
  }

  async function startBatch() {
    var payload;
    try {
      payload = batchPayload();
    } catch (error) {
      node("repairValidation").className = "repair-validation repair-error";
      node("repairValidation").textContent = String(error.message || error);
      return;
    }
    if (validatedKey !== validationKey(payload)) {
      var ready = await validateBatch();
      if (!ready) return;
      payload = batchPayload();
    }
    var count = payload.rollout_ids.length;
    var model = modelLabel();
    if (!window.confirm(
      "Run " + model + " on " + count + " Repair rollout(s)? "
      + "They will execute sequentially on GPU " + payload.gpu_index + "."
    )) return;

    var runButton = node("repairRunButton");
    if (runButton) runButton.disabled = true;
    if (node("repairJobStatus")) {
      node("repairJobStatus").textContent = "Submitting Repair batch…";
    }
    try {
      var response = await fetch("/api/repair/synthetic-suffix/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      var data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not start Repair batch");
      var job = data.job || {};
      var current = state();
      if (current) current.job = job;
      if (typeof window.rememberPersistentJob === "function") {
        window.rememberPersistentJob(job);
      }
      if (node("repairJobStatus")) node("repairJobStatus").textContent = jobMessage(job);
      if (window.LF3RRepairJobs && typeof window.LF3RRepairJobs.refresh === "function") {
        window.LF3RRepairJobs.refresh();
      }
      pollBatch(job.job_id);
    } catch (error) {
      if (runButton) runButton.disabled = false;
      node("repairValidation").className = "repair-validation repair-error";
      node("repairValidation").textContent = "Batch submission failed: " + String(error.message || error);
    }
  }

  function install() {
    if (installed) return true;
    var panel = node("repairGeneratePanel");
    if (!panel || !node("repairGpu") || !window.LF3RRepairSyntheticSuffix) return false;
    installed = true;

    var controls = document.createElement("div");
    controls.id = "repairBatchControls";
    controls.className = "repair-config-grid repair-config-section";
    controls.innerHTML = [
      '<label><span>Run mode</span><select id="repairRunMode">',
      '<option value="single">Single rollout</option>',
      '<option value="batch">Batch</option>',
      '</select></label>',
      '<label data-repair-batch-field class="hidden"><span>Manifest</span><select id="repairBatchManifest"></select></label>',
      '<label data-repair-batch-field class="hidden"><span>Suite</span><select id="repairBatchSuite"></select></label>',
      '<label data-repair-batch-field class="hidden"><span>Task start</span><input id="repairBatchTaskStart" type="number" min="0" step="1" value="0"></label>',
      '<label data-repair-batch-field class="hidden"><span>Task end</span><input id="repairBatchTaskEnd" type="number" min="0" step="1" value="9"></label>',
      '<label data-repair-batch-field class="hidden"><span>Demos / task</span><input id="repairBatchDemosPerTask" type="number" min="1" max="50" step="1" value="1"></label>',
      '<div id="repairBatchSelection" data-repair-batch-field class="repair-kv repair-config-span-2 hidden"><small>Batch selection</small><strong>—</strong></div>'
    ].join("");
    var firstConfig = panel.querySelector(".repair-config-grid");
    if (firstConfig && firstConfig.nextSibling) {
      firstConfig.parentNode.insertBefore(controls, firstConfig.nextSibling);
    } else {
      panel.appendChild(controls);
    }

    var savedMode = "single";
    try { savedMode = localStorage.getItem("lf3r.repair.runMode") || "single"; } catch (_) {}
    node("repairRunMode").value = savedMode === "batch" ? "batch" : "single";
    node("repairRunMode").addEventListener("change", function () {
      try { localStorage.setItem("lf3r.repair.runMode", this.value); } catch (_) {}
      syncMode();
    });

    [
      "repairBatchManifest",
      "repairBatchSuite",
      "repairBatchTaskStart",
      "repairBatchTaskEnd",
      "repairBatchDemosPerTask"
    ].forEach(function (id) {
      node(id).addEventListener("input", function () {
        if (id === "repairBatchManifest" || id === "repairBatchSuite") updateTaskBounds();
        renderSelection();
        invalidate();
      });
      node(id).addEventListener("change", function () {
        if (id === "repairBatchManifest" || id === "repairBatchSuite") updateTaskBounds();
        renderSelection();
        invalidate();
      });
    });

    node("repairValidateButton").addEventListener("click", function (event) {
      if (!isBatch()) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      validateBatch();
    }, true);
    node("repairRunButton").addEventListener("click", function (event) {
      if (!isBatch()) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      startBatch();
    }, true);

    document.addEventListener("input", function (event) {
      if (!isBatch()) return;
      var id = event.target && event.target.id ? String(event.target.id) : "";
      if (id.indexOf("repairWan") === 0 || id.indexOf("repairCtrl") === 0 || id.indexOf("repairCheckpoint") === 0
          || id.indexOf("repairBase") === 0 || id.indexOf("repairSampling") === 0
          || id.indexOf("repairGuidance") === 0 || id.indexOf("repairSeed") === 0
          || id === "repairHistory" || id === "repairDuplicateViews"
          || id === "repairGpu" || id === "repairAlignmentPsnr"
          || id.indexOf("repairCut") === 0) {
        invalidate();
      }
    }, true);
    document.addEventListener("change", function (event) {
      if (!isBatch()) return;
      var id = event.target && event.target.id ? String(event.target.id) : "";
      if (id === "repairWorldModel") {
        window.setTimeout(function () {
          syncMode();
          invalidate();
        }, 0);
      }
    }, true);

    syncMode();
    refreshOptions();
    return true;
  }

  function scheduleRefresh() {
    if (refreshTimer) window.clearTimeout(refreshTimer);
    refreshTimer = window.setTimeout(function tick() {
      refreshTimer = null;
      if (document.body.dataset.view !== "repair") return;
      refreshOptions();
      scheduleRefresh();
    }, 1200);
  }

  window.addEventListener("lf3r:viewchange", function (event) {
    if (!event.detail || event.detail.view !== "repair") {
      if (refreshTimer) {
        window.clearTimeout(refreshTimer);
        refreshTimer = null;
      }
      return;
    }
    if (install()) {
      refreshOptions();
      scheduleRefresh();
    }
  });

  window.setTimeout(function () {
    if (install() && document.body.dataset.view === "repair") scheduleRefresh();
  }, 0);

  window.LF3RRepairBatch = {
    install: install,
    selection: selection,
    refresh: refreshOptions,
    isBatch: isBatch
  };
})();
