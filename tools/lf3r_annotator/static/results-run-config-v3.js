"use strict";

(function installSingleBaselineConfiguratorV3() {
  if (window.__lf3rSingleBaselineConfiguratorV3Installed) return;
  window.__lf3rSingleBaselineConfiguratorV3Installed = true;

  var methodsHost = document.getElementById("evaluationMethods");
  if (!methodsHost) return;

  var existingDrawer = document.getElementById("singleBaselineDrawer");
  if (existingDrawer) existingDrawer.remove();
  document.querySelectorAll(".single-baseline-backdrop").forEach(function (node) {
    node.remove();
  });

  var LABELS = {
    safe: "SAFE",
    procvlm: "ProcVLM",
    rynnvalue: "RynnValue",
    robo_dopamine: "Robo-Dopamine",
    densereward: "DenseReward"
  };

  var COMMON = [
    { key: "validate_environment", label: "Validate environment first", type: "checkbox", defaultValue: false },
    { key: "dry_run", label: "Plan only (dry run)", type: "checkbox", defaultValue: false }
  ];

  var FIELDS = {
    safe: {
      core: [],
      advanced: [],
      execution: [
        { key: "render_video", label: "Render baseline video", type: "checkbox", defaultValue: false }
      ].concat(COMMON)
    },
    procvlm: {
      core: [
        { key: "procvlm_window_size", label: "Window size", type: "number", defaultValue: 4, min: 1 },
        { key: "procvlm_max_sampled_frames", label: "Max sampled frames", type: "number", placeholder: "Runner default", min: 1 },
        { key: "procvlm_max_new_tokens", label: "Max new tokens", type: "number", defaultValue: 4096, min: 1 },
        { key: "procvlm_enable_value_head", label: "Enable value head", type: "checkbox", defaultValue: false, sourceId: "procvlmEnableValueHead" }
      ],
      advanced: [
        { key: "model_path", label: "Model path", type: "text", placeholder: "Use configured checkpoint" },
        { key: "dtype", label: "Dtype", type: "text", defaultValue: "bf16" },
        { key: "tensor_parallel_size", label: "Tensor parallel size", type: "number", defaultValue: 1, min: 1 }
      ],
      execution: [
        { key: "render_video", label: "Render baseline video", type: "checkbox", defaultValue: false }
      ].concat(COMMON)
    },
    rynnvalue: {
      core: [
        { key: "rynn_num_frames", label: "Frames per prefix", type: "number", defaultValue: 16, min: 1 },
        { key: "rynn_evaluation_interval", label: "Target frame interval", type: "number", defaultValue: 8, min: 1 },
        { key: "rynn_batch_size", label: "Batch size", type: "number", defaultValue: 1, min: 1 },
        { key: "rynn_max_image_side", label: "Max image side", type: "number", defaultValue: 448, min: 1 },
        { key: "rynn_max_new_tokens", label: "Max new tokens", type: "number", defaultValue: 128, min: 1 }
      ],
      advanced: [
        { key: "model_path", label: "Model path", type: "text", placeholder: "Use configured checkpoint" },
        { key: "robot_description", label: "Robot description", type: "text", defaultValue: "A Franka Panda 7-DoF robot arm with a parallel-jaw gripper." },
        { key: "camera_description", label: "Camera description", type: "text", defaultValue: "A fixed third-person agent-view RGB camera observing the robot workspace." }
      ],
      execution: [
        { key: "render_video", label: "Render baseline video", type: "checkbox", defaultValue: false }
      ].concat(COMMON)
    },
    robo_dopamine: {
      core: [
        { key: "robo_frame_interval", label: "Frame interval", type: "number", defaultValue: 4, min: 1 },
        { key: "robo_batch_size", label: "Batch size", type: "number", defaultValue: 1, min: 1 },
        { key: "robo_eval_mode", label: "Evaluation mode", type: "select", defaultValue: "fused", options: [["fused", "Fused"], ["forward", "Forward"], ["incremental", "Incremental"], ["backward", "Backward"]] }
      ],
      advanced: [
        { key: "model_path", label: "Model path", type: "text", placeholder: "Use configured checkpoint" },
        { key: "dtype", label: "Dtype", type: "text", defaultValue: "bf16" },
        { key: "tensor_parallel_size", label: "Tensor parallel size", type: "number", defaultValue: 1, min: 1 },
        { key: "goal_image", label: "Goal image", type: "text", placeholder: "Project-relative image path" }
      ],
      execution: [
        { key: "render_video", label: "Render baseline video", type: "checkbox", defaultValue: false }
      ].concat(COMMON)
    },
    densereward: {
      core: [
        { key: "densereward_frame_interval", label: "Frame interval", type: "number", defaultValue: 1, min: 1 },
        { key: "densereward_max_new_tokens", label: "Max new tokens", type: "number", defaultValue: 32, min: 1 }
      ],
      advanced: [
        { key: "model_path", label: "Model path", type: "text", placeholder: "Use configured checkpoint" }
      ],
      execution: COMMON.slice()
    }
  };

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function sourceValue(field) {
    var node = field.sourceId
      ? document.getElementById(field.sourceId)
      : document.querySelector('[data-batch-option="' + field.key + '"]');
    if (!node) return field.defaultValue == null ? "" : field.defaultValue;
    if (field.type === "checkbox") return Boolean(node.checked);
    return node.value === ""
      ? (field.defaultValue == null ? "" : field.defaultValue)
      : node.value;
  }

  function storageKey(method) {
    return "lf3r.singleBaseline.config." + method;
  }

  function readStored(method) {
    try {
      var value = JSON.parse(sessionStorage.getItem(storageKey(method)) || "null");
      return value && typeof value === "object" && !Array.isArray(value) ? value : null;
    } catch (_) {
      return null;
    }
  }

  function writeStored(method, value) {
    try {
      sessionStorage.setItem(storageKey(method), JSON.stringify(value));
    } catch (_) {}
  }

  var backdrop = document.createElement("div");
  backdrop.className = "single-baseline-backdrop";
  backdrop.hidden = true;

  var drawer = document.createElement("aside");
  drawer.id = "singleBaselineDrawer";
  drawer.className = "single-baseline-drawer";
  drawer.hidden = true;
  drawer.setAttribute("role", "dialog");
  drawer.setAttribute("aria-modal", "true");
  drawer.innerHTML = [
    '<div class="single-baseline-header"><div><p class="eyebrow">SINGLE ROLLOUT INFERENCE</p><h2>Configure baseline</h2></div><button type="button" class="single-baseline-close" aria-label="Close">×</button></div>',
    '<div class="single-baseline-context"><div><span>Method</span><strong data-single-method>—</strong></div><div><span>Instruction</span><strong data-single-condition>—</strong></div><div class="single-baseline-rollout"><span>Rollout</span><strong data-single-rollout>—</strong></div></div>',
    '<form class="single-baseline-form">',
      '<section class="single-baseline-section"><div class="single-baseline-section-heading"><h3>Runtime</h3><p>Resources for this rollout.</p></div><div class="single-baseline-grid"><label><span>GPU IDs</span><input name="gpu" value="0"></label><label><span>vLLM free-memory target</span><input name="memory_utilization" type="number" min="0.05" max="1" step="0.05" value="0.80"></label></div></section>',
      '<section class="single-baseline-section" data-core-section><div class="single-baseline-section-heading"><h3>Method settings</h3><p>Frequently changed inference parameters.</p></div><div class="single-baseline-grid" data-core></div></section>',
      '<details class="single-baseline-advanced" data-advanced-section><summary><span><strong>Advanced overrides</strong><small>Checkpoint and model-specific settings</small></span><i>⌄</i></summary><div class="single-baseline-grid" data-advanced></div></details>',
      '<section class="single-baseline-section"><div class="single-baseline-section-heading"><h3>Execution</h3><p>Optional runner behavior.</p></div><div class="single-baseline-checks" data-execution></div></section>',
      '<div class="single-baseline-status" data-status>Configure the run, then launch inference.</div>',
      '<div class="single-baseline-actions"><button type="button" class="ghost-button" data-reset>Reset from Runs</button><span></span><button type="button" class="ghost-button" data-cancel>Cancel</button><button type="submit" class="save-button" data-submit>Run baseline</button></div>',
    '</form>'
  ].join("");
  document.body.appendChild(backdrop);
  document.body.appendChild(drawer);

  var form = drawer.querySelector("form");
  var coreHost = drawer.querySelector("[data-core]");
  var advancedHost = drawer.querySelector("[data-advanced]");
  var executionHost = drawer.querySelector("[data-execution]");
  var coreSection = drawer.querySelector("[data-core-section]");
  var advancedSection = drawer.querySelector("[data-advanced-section]");
  var statusNode = drawer.querySelector("[data-status]");
  var submit = drawer.querySelector("[data-submit]");
  var activeMethod = null;
  var closeTimer = null;

  function fieldHtml(field, value) {
    if (field.type === "checkbox") {
      return '<label class="single-baseline-check"><input data-option="' + esc(field.key)
        + '" type="checkbox"' + (value ? " checked" : "") + '><span>'
        + esc(field.label) + "</span></label>";
    }
    if (field.type === "select") {
      return '<label><span>' + esc(field.label) + '</span><select data-option="'
        + esc(field.key) + '">' + (field.options || []).map(function (option) {
          return '<option value="' + esc(option[0]) + '"'
            + (String(value) === String(option[0]) ? " selected" : "") + ">"
            + esc(option[1]) + "</option>";
        }).join("") + "</select></label>";
    }
    return '<label><span>' + esc(field.label) + '</span><input data-option="'
      + esc(field.key) + '" type="' + esc(field.type || "text") + '"'
      + (field.min != null ? ' min="' + field.min + '"' : "")
      + (field.placeholder ? ' placeholder="' + esc(field.placeholder) + '"' : "")
      + ' value="' + esc(value == null ? "" : value) + '"></label>';
  }

  function defaults(method) {
    var definition = FIELDS[method] || FIELDS.safe;
    var out = {};
    definition.core.concat(definition.advanced, definition.execution).forEach(function (field) {
      out[field.key] = sourceValue(field);
    });
    var gpu = document.getElementById("baselineBatchGpu");
    var memory = document.getElementById("baselineBatchMemoryUtilization");
    out.gpu = gpu && gpu.value ? gpu.value : "0";
    out.memory_utilization = memory && memory.value ? memory.value : "0.80";
    return out;
  }

  function render(method, preferStored) {
    var definition = FIELDS[method] || FIELDS.safe;
    var values = preferStored ? readStored(method) : null;
    if (!values) values = defaults(method);

    coreHost.innerHTML = definition.core.map(function (field) {
      return fieldHtml(field, values[field.key]);
    }).join("");
    advancedHost.innerHTML = definition.advanced.map(function (field) {
      return fieldHtml(field, values[field.key]);
    }).join("");
    executionHost.innerHTML = definition.execution.map(function (field) {
      return fieldHtml(field, values[field.key]);
    }).join("");

    coreSection.hidden = definition.core.length === 0;
    advancedSection.hidden = definition.advanced.length === 0;
    advancedSection.open = false;

    var gpuInput = form.elements.namedItem("gpu");
    var memoryInput = form.elements.namedItem("memory_utilization");
    if (gpuInput) gpuInput.value = values.gpu || "0";
    if (memoryInput) memoryInput.value = values.memory_utilization || "0.80";
  }

  function selectedRecord() {
    var record = null;
    try {
      if (typeof window.selectedRollout === "function") record = window.selectedRollout();
    } catch (error) {
      console.error("selectedRollout() failed while opening baseline config", error);
    }
    if (record) return record;
    if (!window.state || !Array.isArray(state.rollouts)) return null;

    var wanted = state.selectedId;
    if (!wanted && state.evaluation) {
      wanted = state.evaluation.rollout_id || state.evaluation.source_rollout_id;
    }
    if (!wanted) return null;
    return state.rollouts.find(function (item) {
      return String(item.id) === String(wanted);
    }) || null;
  }

  function reportOpenError(message, error) {
    console.error(message, error || "");
    var evaluationStatus = document.getElementById("evaluationStatus");
    if (evaluationStatus) evaluationStatus.textContent = message;
  }

  function open(method) {
    if (!Object.prototype.hasOwnProperty.call(FIELDS, method)) {
      reportOpenError("Cannot configure baseline: unknown method " + method + ".");
      return false;
    }

    var record = selectedRecord();
    if (!record) {
      reportOpenError("Cannot configure baseline: no rollout is currently selected.");
      return false;
    }

    if (closeTimer) {
      window.clearTimeout(closeTimer);
      closeTimer = null;
    }

    try {
      activeMethod = method;
      render(method, true);
      drawer.querySelector("[data-single-method]").textContent = LABELS[method] || method;
      drawer.querySelector("[data-single-condition]").textContent =
        typeof window.instructionConditionLabel === "function"
          ? window.instructionConditionLabel(state.instructionCondition || "full_instruction")
          : (state.instructionCondition || "full_instruction");
      drawer.querySelector("[data-single-rollout]").textContent = record.task_description || record.id;
      statusNode.textContent = "This runs only the currently selected rollout.";
      statusNode.className = "single-baseline-status";
      submit.disabled = false;
      submit.textContent = "Run " + (LABELS[method] || method);

      backdrop.hidden = false;
      drawer.hidden = false;
      document.body.classList.add("single-baseline-open");
      backdrop.classList.add("is-open");
      drawer.classList.add("is-open");
      return true;
    } catch (error) {
      activeMethod = null;
      backdrop.hidden = true;
      drawer.hidden = true;
      backdrop.classList.remove("is-open");
      drawer.classList.remove("is-open");
      document.body.classList.remove("single-baseline-open");
      reportOpenError("Baseline configuration UI error: " + (error && error.message ? error.message : String(error)), error);
      return false;
    }
  }

  function close() {
    backdrop.classList.remove("is-open");
    drawer.classList.remove("is-open");
    document.body.classList.remove("single-baseline-open");
    if (closeTimer) window.clearTimeout(closeTimer);
    closeTimer = window.setTimeout(function () {
      backdrop.hidden = true;
      drawer.hidden = true;
      closeTimer = null;
    }, 170);
  }

  function collect() {
    var definition = FIELDS[activeMethod] || FIELDS.safe;
    var gpuInput = form.elements.namedItem("gpu");
    var memoryInput = form.elements.namedItem("memory_utilization");
    var config = {
      gpu: String(gpuInput && gpuInput.value || "0").trim(),
      memory_utilization: String(memoryInput && memoryInput.value || "0.80").trim()
    };
    var options = {};

    definition.core.concat(definition.advanced, definition.execution).forEach(function (field) {
      var node = drawer.querySelector('[data-option="' + field.key + '"]');
      if (!node) return;
      var value = field.type === "checkbox"
        ? Boolean(node.checked)
        : String(node.value || "").trim();
      config[field.key] = value;
      if (field.type === "checkbox") options[field.key] = value;
      else if (value !== "") options[field.key] = field.type === "number" ? Number(value) : value;
    });
    return { config: config, options: options };
  }

  async function submitRun(event) {
    event.preventDefault();
    var record = selectedRecord();
    if (!record || !activeMethod) {
      statusNode.textContent = "The selected rollout or baseline method is no longer available.";
      statusNode.className = "single-baseline-status error";
      return;
    }

    var collected = collect();
    var memory = Number(collected.config.memory_utilization);
    if (!Number.isFinite(memory) || memory < 0.05 || memory > 1) {
      statusNode.textContent = "vLLM free-memory target must be between 0.05 and 1.";
      statusNode.className = "single-baseline-status error";
      return;
    }

    writeStored(activeMethod, collected.config);
    submit.disabled = true;
    statusNode.textContent = "Starting " + (LABELS[activeMethod] || activeMethod) + "…";
    statusNode.className = "single-baseline-status running";

    try {
      var response = await fetch("/api/baselines/run/" + encodeURIComponent(record.id), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          baseline: activeMethod,
          gpu: collected.config.gpu || "0",
          memory_utilization: memory,
          instruction_condition: state.instructionCondition || "full_instruction",
          options: collected.options
        })
      });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not start baseline");

      state.baselineJob = payload.job;
      if (typeof window.rememberPersistentJob === "function") {
        window.rememberPersistentJob(payload.job);
      }
      if (typeof window.pollBaselineJob === "function") {
        window.pollBaselineJob(payload.job.job_id, record.id);
      } else if (typeof window.pollPersistentJob === "function") {
        window.pollPersistentJob(payload.job.job_id);
      }

      var evaluationStatus = document.getElementById("evaluationStatus");
      if (evaluationStatus) {
        evaluationStatus.textContent = "Running " + (LABELS[activeMethod] || activeMethod)
          + " · job " + payload.job.job_id + "…";
      }
      statusNode.textContent = "Job started: " + payload.job.job_id;
      statusNode.className = "single-baseline-status success";
      window.setTimeout(close, 650);
    } catch (error) {
      submit.disabled = false;
      statusNode.textContent = "Baseline run error: " + error.message;
      statusNode.className = "single-baseline-status error";
    }
  }

  function enhanceButtons() {
    methodsHost.querySelectorAll("[data-run-baseline]").forEach(function (button) {
      if (button.dataset.singleRunConfigured === "v3") return;
      button.dataset.singleRunConfigured = "v3";
      button.textContent = "Configure & run";
      button.title = "Configure parameters and run this baseline on the current rollout";
    });
  }

  document.addEventListener("click", function (event) {
    var target = event.target;
    var button = target && target.closest ? target.closest("[data-run-baseline]") : null;
    if (!button || !methodsHost.contains(button)) return;

    event.preventDefault();
    event.stopImmediatePropagation();
    open(String(button.dataset.runBaseline || ""));
  }, true);

  // app.js replaces the direct children of evaluationMethods when Results are
  // rendered. Descendant observation is unnecessary and can feed our own
  // button-label writes back into the observer.
  var observer = new MutationObserver(function (mutations) {
    var needsEnhance = mutations.some(function (mutation) {
      return mutation.addedNodes && mutation.addedNodes.length;
    });
    if (needsEnhance) enhanceButtons();
  });
  observer.observe(methodsHost, { childList: true });
  enhanceButtons();

  drawer.querySelector(".single-baseline-close").addEventListener("click", close);
  drawer.querySelector("[data-cancel]").addEventListener("click", close);
  backdrop.addEventListener("click", close);
  drawer.querySelector("[data-reset]").addEventListener("click", function () {
    if (!activeMethod) return;
    try {
      sessionStorage.removeItem(storageKey(activeMethod));
    } catch (_) {}
    render(activeMethod, false);
  });
  advancedSection.addEventListener("toggle", function () {
    var icon = advancedSection.querySelector("summary i");
    if (icon) icon.textContent = advancedSection.open ? "⌃" : "⌄";
  });
  form.addEventListener("submit", submitRun);
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && !drawer.hidden) close();
  });

  window.lf3rOpenSingleBaselineConfig = open;
})();
