"use strict";

(function installSingleBaselineConfigurator() {
  var methodsHost = document.getElementById("evaluationMethods");
  if (!methodsHost || document.getElementById("singleBaselineDrawer")) return;

  var METHOD_LABELS = {
    safe: "SAFE",
    procvlm: "ProcVLM",
    rynnvalue: "RynnValue",
    robo_dopamine: "Robo-Dopamine",
    densereward: "DenseReward"
  };

  var COMMON_EXECUTION = [
    { key: "validate_environment", label: "Validate environment first", type: "checkbox", defaultValue: false },
    { key: "dry_run", label: "Plan only (dry run)", type: "checkbox", defaultValue: false }
  ];

  var METHOD_FIELDS = {
    safe: {
      core: [],
      advanced: [],
      execution: [
        { key: "render_video", label: "Render baseline video", type: "checkbox", defaultValue: false }
      ].concat(COMMON_EXECUTION)
    },
    procvlm: {
      core: [
        { key: "procvlm_window_size", label: "Window size", type: "number", min: 1, step: 1, defaultValue: 4 },
        { key: "procvlm_max_sampled_frames", label: "Max sampled frames", type: "number", min: 1, step: 1, placeholder: "Runner default" },
        { key: "procvlm_max_new_tokens", label: "Max new tokens", type: "number", min: 1, step: 1, defaultValue: 4096 },
        { key: "procvlm_enable_value_head", label: "Enable value head", type: "checkbox", defaultValue: false, sourceId: "procvlmEnableValueHead" }
      ],
      advanced: [
        { key: "model_path", label: "Model path", type: "text", placeholder: "Use configured checkpoint" },
        { key: "dtype", label: "Dtype", type: "text", defaultValue: "bf16" },
        { key: "tensor_parallel_size", label: "Tensor parallel size", type: "number", min: 1, step: 1, defaultValue: 1 }
      ],
      execution: [
        { key: "render_video", label: "Render baseline video", type: "checkbox", defaultValue: false }
      ].concat(COMMON_EXECUTION)
    },
    rynnvalue: {
      core: [
        { key: "rynn_num_frames", label: "Frames per prefix", type: "number", min: 1, step: 1, defaultValue: 16 },
        { key: "rynn_evaluation_interval", label: "Target frame interval", type: "number", min: 1, step: 1, defaultValue: 8 },
        { key: "rynn_batch_size", label: "Batch size", type: "number", min: 1, step: 1, defaultValue: 1 },
        { key: "rynn_max_image_side", label: "Max image side", type: "number", min: 1, step: 1, defaultValue: 448 },
        { key: "rynn_max_new_tokens", label: "Max new tokens", type: "number", min: 1, step: 1, defaultValue: 128 }
      ],
      advanced: [
        { key: "model_path", label: "Model path", type: "text", placeholder: "Use configured checkpoint" },
        { key: "robot_description", label: "Robot description", type: "text", defaultValue: "A Franka Panda 7-DoF robot arm with a parallel-jaw gripper." },
        { key: "camera_description", label: "Camera description", type: "text", defaultValue: "A fixed third-person agent-view RGB camera observing the robot workspace." }
      ],
      execution: [
        { key: "render_video", label: "Render baseline video", type: "checkbox", defaultValue: false }
      ].concat(COMMON_EXECUTION)
    },
    robo_dopamine: {
      core: [
        { key: "robo_frame_interval", label: "Frame interval", type: "number", min: 1, step: 1, defaultValue: 4 },
        { key: "robo_batch_size", label: "Batch size", type: "number", min: 1, step: 1, defaultValue: 1 },
        { key: "robo_eval_mode", label: "Evaluation mode", type: "select", defaultValue: "fused", options: [
          ["fused", "Fused (all three perspectives)"],
          ["forward", "Forward"],
          ["incremental", "Incremental"],
          ["backward", "Backward"]
        ] }
      ],
      advanced: [
        { key: "model_path", label: "Model path", type: "text", placeholder: "Use configured checkpoint" },
        { key: "dtype", label: "Dtype", type: "text", defaultValue: "bf16" },
        { key: "tensor_parallel_size", label: "Tensor parallel size", type: "number", min: 1, step: 1, defaultValue: 1 },
        { key: "goal_image", label: "Goal image", type: "text", placeholder: "Project-relative image path" }
      ],
      execution: [
        { key: "render_video", label: "Render baseline video", type: "checkbox", defaultValue: false }
      ].concat(COMMON_EXECUTION)
    },
    densereward: {
      core: [
        { key: "densereward_frame_interval", label: "Frame interval", type: "number", min: 1, step: 1, defaultValue: 1 },
        { key: "densereward_max_new_tokens", label: "Max new tokens", type: "number", min: 1, step: 1, defaultValue: 32 }
      ],
      advanced: [
        { key: "model_path", label: "Model path", type: "text", placeholder: "Use configured checkpoint" }
      ],
      execution: COMMON_EXECUTION.slice()
    }
  };

  var backdrop = document.createElement("div");
  backdrop.className = "single-baseline-backdrop";
  backdrop.hidden = true;

  var drawer = document.createElement("aside");
  drawer.id = "singleBaselineDrawer";
  drawer.className = "single-baseline-drawer";
  drawer.hidden = true;
  drawer.setAttribute("aria-modal", "true");
  drawer.setAttribute("role", "dialog");
  drawer.setAttribute("aria-labelledby", "singleBaselineTitle");
  drawer.innerHTML = [
    '<div class="single-baseline-header">',
      '<div><p class="eyebrow">SINGLE ROLLOUT INFERENCE</p><h2 id="singleBaselineTitle">Configure baseline</h2></div>',
      '<button type="button" class="single-baseline-close" aria-label="Close baseline configuration">×</button>',
    '</div>',
    '<div class="single-baseline-context">',
      '<div><span>Method</span><strong data-single-method>—</strong></div>',
      '<div><span>Instruction</span><strong data-single-condition>—</strong></div>',
      '<div class="single-baseline-rollout"><span>Rollout</span><strong data-single-rollout>—</strong></div>',
    '</div>',
    '<form class="single-baseline-form">',
      '<section class="single-baseline-section">',
        '<div class="single-baseline-section-heading"><h3>Runtime</h3><p>Resources for this one rollout only.</p></div>',
        '<div class="single-baseline-grid">',
          '<label><span>GPU IDs</span><input name="gpu" type="text" value="0" maxlength="31"></label>',
          '<label><span>Free-memory fraction</span><input name="memory_utilization" type="number" min="0.05" max="1" step="0.05" value="0.80"></label>',
        '</div>',
      '</section>',
      '<section class="single-baseline-section" data-single-core-section>',
        '<div class="single-baseline-section-heading"><h3>Method settings</h3><p>Parameters that usually matter when validating this baseline.</p></div>',
        '<div class="single-baseline-grid" data-single-core></div>',
      '</section>',
      '<details class="single-baseline-advanced" data-single-advanced-section>',
        '<summary><span><strong>Advanced overrides</strong><small>Checkpoint and model-specific settings</small></span><i>⌄</i></summary>',
        '<div class="single-baseline-grid" data-single-advanced></div>',
      '</details>',
      '<section class="single-baseline-section single-baseline-execution">',
        '<div class="single-baseline-section-heading"><h3>Execution</h3><p>Optional runner behavior.</p></div>',
        '<div class="single-baseline-checks" data-single-execution></div>',
      '</section>',
      '<div class="single-baseline-status" data-single-status>Configure the run, then launch inference for the current rollout.</div>',
      '<div class="single-baseline-actions">',
        '<button type="button" class="ghost-button" data-single-reset>Reset from Runs</button>',
        '<span></span>',
        '<button type="button" class="ghost-button" data-single-cancel>Cancel</button>',
        '<button type="submit" class="save-button" data-single-submit>Run baseline</button>',
      '</div>',
    '</form>'
  ].join("");

  document.body.appendChild(backdrop);
  document.body.appendChild(drawer);

  var form = drawer.querySelector(".single-baseline-form");
  var coreHost = drawer.querySelector("[data-single-core]");
  var advancedHost = drawer.querySelector("[data-single-advanced]");
  var executionHost = drawer.querySelector("[data-single-execution]");
  var coreSection = drawer.querySelector("[data-single-core-section]");
  var advancedSection = drawer.querySelector("[data-single-advanced-section]");
  var statusNode = drawer.querySelector("[data-single-status]");
  var submitButton = drawer.querySelector("[data-single-submit]");
  var activeMethod = null;

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function storageKey(method) {
    return "lf3r.singleBaseline.config." + method;
  }

  function readStored(method) {
    try {
      var raw = sessionStorage.getItem(storageKey(method));
      return raw ? JSON.parse(raw) : null;
    } catch (_) {
      return null;
    }
  }

  function writeStored(method, value) {
    try { sessionStorage.setItem(storageKey(method), JSON.stringify(value)); } catch (_) {}
  }

  function sourceNode(field) {
    if (field.sourceId) return document.getElementById(field.sourceId);
    return document.querySelector('[data-batch-option="' + field.key + '"]');
  }

  function sourceValue(field) {
    var node = sourceNode(field);
    if (!node) return field.defaultValue == null ? "" : field.defaultValue;
    if (field.type === "checkbox") return Boolean(node.checked);
    return node.value == null || node.value === "" ? (field.defaultValue == null ? "" : field.defaultValue) : node.value;
  }

  function fieldMarkup(field, value) {
    if (field.type === "checkbox") {
      return '<label class="single-baseline-check"><input data-single-option="' + escapeHtml(field.key) + '" type="checkbox"'
        + (value ? ' checked' : '') + '><span>' + escapeHtml(field.label) + '</span></label>';
    }
    var attrs = ' data-single-option="' + escapeHtml(field.key) + '"';
    if (field.min != null) attrs += ' min="' + escapeHtml(field.min) + '"';
    if (field.max != null) attrs += ' max="' + escapeHtml(field.max) + '"';
    if (field.step != null) attrs += ' step="' + escapeHtml(field.step) + '"';
    if (field.placeholder) attrs += ' placeholder="' + escapeHtml(field.placeholder) + '"';
    if (field.type === "select") {
      return '<label><span>' + escapeHtml(field.label) + '</span><select' + attrs + '>'
        + (field.options || []).map(function (option) {
          return '<option value="' + escapeHtml(option[0]) + '"' + (String(value) === String(option[0]) ? ' selected' : '') + '>'
            + escapeHtml(option[1]) + '</option>';
        }).join('') + '</select></label>';
    }
    return '<label><span>' + escapeHtml(field.label) + '</span><input' + attrs + ' type="' + escapeHtml(field.type || "text")
      + '" value="' + escapeHtml(value == null ? "" : value) + '"></label>';
  }

  function valuesFromRuns(method) {
    var definition = METHOD_FIELDS[method] || METHOD_FIELDS.safe;
    var values = {};
    definition.core.concat(definition.advanced, definition.execution).forEach(function (field) {
      values[field.key] = sourceValue(field);
    });
    var gpu = document.getElementById("baselineBatchGpu");
    var memory = document.getElementById("baselineBatchMemoryUtilization");
    values.gpu = gpu && gpu.value ? gpu.value : "0";
    values.memory_utilization = memory && memory.value ? memory.value : "0.80";
    return values;
  }

  function renderFields(method, preferStored) {
    var definition = METHOD_FIELDS[method] || METHOD_FIELDS.safe;
    var values = preferStored ? readStored(method) : null;
    if (!values) values = valuesFromRuns(method);

    coreHost.innerHTML = definition.core.map(function (field) {
      return fieldMarkup(field, values[field.key] == null ? sourceValue(field) : values[field.key]);
    }).join("");
    advancedHost.innerHTML = definition.advanced.map(function (field) {
      return fieldMarkup(field, values[field.key] == null ? sourceValue(field) : values[field.key]);
    }).join("");
    executionHost.innerHTML = definition.execution.map(function (field) {
      return fieldMarkup(field, values[field.key] == null ? sourceValue(field) : values[field.key]);
    }).join("");
    coreSection.hidden = definition.core.length === 0;
    advancedSection.hidden = definition.advanced.length === 0;
    advancedSection.open = false;

    form.elements.gpu.value = values.gpu || "0";
    form.elements.memory_utilization.value = values.memory_utilization || "0.80";
  }

  function selectedRecord() {
    if (typeof window.selectedRollout === "function") return window.selectedRollout();
    if (window.state && state.selectedId && Array.isArray(state.rollouts)) {
      return state.rollouts.find(function (item) { return item.id === state.selectedId; }) || null;
    }
    return null;
  }

  function conditionLabel(condition) {
    if (typeof window.instructionConditionLabel === "function") return window.instructionConditionLabel(condition);
    return String(condition || "full_instruction").replace(/_/g, " ");
  }

  function openDrawer(method) {
    var record = selectedRecord();
    if (!record) return;
    activeMethod = method;
    renderFields(method, true);
    drawer.querySelector("[data-single-method]").textContent = METHOD_LABELS[method] || method;
    var condition = window.state && state.instructionCondition || "full_instruction";
    drawer.querySelector("[data-single-condition]").textContent = conditionLabel(condition);
    drawer.querySelector("[data-single-rollout]").textContent = record.task_description || record.task || record.id;
    statusNode.textContent = "This runs only the currently selected rollout. Batch configuration in Runs is unchanged.";
    statusNode.className = "single-baseline-status";
    submitButton.disabled = false;
    submitButton.textContent = "Run " + (METHOD_LABELS[method] || method);
    backdrop.hidden = false;
    drawer.hidden = false;
    requestAnimationFrame(function () {
      backdrop.classList.add("is-open");
      drawer.classList.add("is-open");
      var first = drawer.querySelector("input, select, button");
      if (first) first.focus({ preventScroll: true });
    });
  }

  function closeDrawer() {
    backdrop.classList.remove("is-open");
    drawer.classList.remove("is-open");
    window.setTimeout(function () {
      backdrop.hidden = true;
      drawer.hidden = true;
    }, 160);
  }

  function collectConfig() {
    var definition = METHOD_FIELDS[activeMethod] || METHOD_FIELDS.safe;
    var config = {
      gpu: String(form.elements.gpu.value || "0").trim(),
      memory_utilization: String(form.elements.memory_utilization.value || "0.80").trim()
    };
    var options = {};
    definition.core.concat(definition.advanced, definition.execution).forEach(function (field) {
      var node = drawer.querySelector('[data-single-option="' + field.key + '"]');
      if (!node) return;
      if (field.type === "checkbox") {
        options[field.key] = Boolean(node.checked);
        config[field.key] = Boolean(node.checked);
        return;
      }
      var raw = String(node.value == null ? "" : node.value).trim();
      config[field.key] = raw;
      if (raw === "") return;
      if (field.type === "number") {
        var numeric = Number(raw);
        if (!Number.isFinite(numeric)) throw new Error(field.label + " must be numeric.");
        options[field.key] = numeric;
      } else {
        options[field.key] = raw;
      }
    });
    return { config: config, options: options };
  }

  async function submitRun(event) {
    event.preventDefault();
    if (!activeMethod) return;
    var record = selectedRecord();
    if (!record) return;
    var collected;
    try {
      collected = collectConfig();
    } catch (error) {
      statusNode.textContent = error.message;
      statusNode.className = "single-baseline-status error";
      return;
    }

    var memory = Number(collected.config.memory_utilization);
    if (!Number.isFinite(memory) || memory < 0.05 || memory > 1) {
      statusNode.textContent = "Free-memory fraction must be between 0.05 and 1.";
      statusNode.className = "single-baseline-status error";
      return;
    }

    writeStored(activeMethod, collected.config);
    submitButton.disabled = true;
    statusNode.textContent = "Starting " + (METHOD_LABELS[activeMethod] || activeMethod) + " for this rollout…";
    statusNode.className = "single-baseline-status running";
    var condition = window.state && state.instructionCondition || "full_instruction";

    try {
      var response = await fetch("/api/baselines/run/" + encodeURIComponent(record.id), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          baseline: activeMethod,
          gpu: collected.config.gpu || "0",
          memory_utilization: memory,
          instruction_condition: condition,
          options: collected.options
        })
      });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not start baseline");
      if (window.state) state.baselineJob = payload.job;
      if (typeof window.rememberPersistentJob === "function") window.rememberPersistentJob(payload.job);
      var evaluationStatus = document.getElementById("evaluationStatus");
      if (evaluationStatus) {
        evaluationStatus.textContent = "Running " + (METHOD_LABELS[activeMethod] || activeMethod)
          + " · job " + payload.job.job_id + "…";
      }
      if (typeof window.pollBaselineJob === "function") window.pollBaselineJob(payload.job.job_id, record.id);
      statusNode.textContent = "Job started: " + payload.job.job_id;
      statusNode.className = "single-baseline-status success";
      submitButton.textContent = "Started";
      window.setTimeout(closeDrawer, 650);
    } catch (error) {
      submitButton.disabled = false;
      statusNode.textContent = "Baseline run error: " + error.message;
      statusNode.className = "single-baseline-status error";
    }
  }

  function updateRunButtons() {
    methodsHost.querySelectorAll("[data-run-baseline]").forEach(function (button) {
      button.textContent = "Configure & run";
      button.title = "Configure parameters and run this baseline on the current rollout";
    });
  }

  methodsHost.addEventListener("click", function (event) {
    var button = event.target.closest("[data-run-baseline]");
    if (!button) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    openDrawer(button.dataset.runBaseline);
  }, true);

  new MutationObserver(updateRunButtons).observe(methodsHost, { childList: true, subtree: true });
  updateRunButtons();

  drawer.querySelector(".single-baseline-close").addEventListener("click", closeDrawer);
  drawer.querySelector("[data-single-cancel]").addEventListener("click", closeDrawer);
  backdrop.addEventListener("click", closeDrawer);
  drawer.querySelector("[data-single-reset]").addEventListener("click", function () {
    if (!activeMethod) return;
    try { sessionStorage.removeItem(storageKey(activeMethod)); } catch (_) {}
    renderFields(activeMethod, false);
    statusNode.textContent = "Reset to the current values from Runs.";
    statusNode.className = "single-baseline-status";
  });
  form.addEventListener("submit", submitRun);

  advancedSection.addEventListener("toggle", function () {
    var icon = advancedSection.querySelector("summary i");
    if (icon) icon.textContent = advancedSection.open ? "⌃" : "⌄";
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && !drawer.hidden) closeDrawer();
  });
})();
