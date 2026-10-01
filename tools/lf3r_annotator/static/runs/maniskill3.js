"use strict";

/* Additive ManiSkill3 rollout generation. Existing LIBERO controls and submit
 * handling remain untouched whenever the generator selector is set to LIBERO.
 */

(function installManiSkill3RolloutGeneration() {
  var form = document.getElementById("rolloutGenerationForm");
  var suite = document.getElementById("rolloutGenerationSuite");
  if (!form || !suite) return;

  var grid = suite.closest(".rollout-generation-grid");
  var suiteLabel = suite.closest("label");
  if (!grid || !suiteLabel) return;

  var originalSelection = window.updateRolloutGenerationSelection;
  var originalJobMessage = window.rolloutGenerationJobMessage;
  var originalTrialsCaption = null;
  var trialsInput = document.getElementById("rolloutGenerationTrials");
  var trialsLabel = trialsInput && trialsInput.closest("label");
  var trialsCaption = trialsLabel && trialsLabel.querySelector("span");
  if (trialsCaption) originalTrialsCaption = trialsCaption.textContent;

  function fieldLabel(id, caption, input) {
    var label = document.createElement("label");
    label.dataset.maniskillField = "true";
    label.classList.add("hidden");
    var span = document.createElement("span");
    span.textContent = caption;
    input.id = id;
    label.appendChild(span);
    label.appendChild(input);
    return label;
  }

  var backendSelect = document.createElement("select");
  backendSelect.id = "rolloutGenerationBackend";
  backendSelect.innerHTML = [
    '<option value="libero" selected>LIBERO / OpenVLA (existing)</option>',
    '<option value="maniskill3">ManiSkill3 / motion planning (success only)</option>'
  ].join("");
  var backendLabel = document.createElement("label");
  var backendCaption = document.createElement("span");
  backendCaption.textContent = "Generator";
  backendLabel.appendChild(backendCaption);
  backendLabel.appendChild(backendSelect);
  grid.insertBefore(backendLabel, suiteLabel);

  var envSelect = document.createElement("select");
  [
    "PickCube-v1",
    "StackCube-v1",
    "PegInsertionSide-v1",
    "PlugCharger-v1",
    "PlaceSphere-v1",
    "PushCube-v1",
    "PullCubeTool-v1",
    "LiftPegUpright-v1",
    "PullCube-v1",
    "DrawTriangle-v1",
    "DrawSVG-v1",
    "StackPyramid-v1"
  ].forEach(function (envId) {
    var option = document.createElement("option");
    option.value = envId;
    option.textContent = envId;
    envSelect.appendChild(option);
  });
  var envLabel = fieldLabel("rolloutGenerationManiSkillEnv", "ManiSkill3 task", envSelect);

  function resolutionInput(value) {
    var input = document.createElement("input");
    input.type = "number";
    input.min = "64";
    input.max = "2048";
    input.step = "2";
    input.value = String(value);
    return input;
  }
  var renderWidthLabel = fieldLabel(
    "rolloutGenerationRenderWidth",
    "Render width",
    resolutionInput(512)
  );
  var renderHeightLabel = fieldLabel(
    "rolloutGenerationRenderHeight",
    "Render height",
    resolutionInput(512)
  );

  suiteLabel.insertAdjacentElement("beforebegin", envLabel);
  suiteLabel.insertAdjacentElement("beforebegin", renderWidthLabel);
  suiteLabel.insertAdjacentElement("beforebegin", renderHeightLabel);

  var liberoInputIds = [
    "rolloutGenerationSuite",
    "rolloutGenerationVideoViewMode",
    "rolloutGenerationRenderResolution",
    "rolloutGenerationRecordResolution",
    "rolloutGenerationTaskStart",
    "rolloutGenerationTaskEnd"
  ];

  function maniskillSelected() {
    return backendSelect.value === "maniskill3";
  }

  function toggleModeFields(isManiSkill) {
    document.querySelectorAll("[data-maniskill-field]").forEach(function (node) {
      node.classList.toggle("hidden", !isManiSkill);
    });
    liberoInputIds.forEach(function (id) {
      var node = document.getElementById(id);
      var label = node && node.closest("label");
      if (label) label.classList.toggle("hidden", isManiSkill);
    });
    document.querySelectorAll("[data-multiview-size-field]").forEach(function (node) {
      node.classList.toggle("hidden", isManiSkill || document.getElementById("rolloutGenerationVideoViewMode").value !== "libero_three_view");
    });
    var options = document.querySelector(".rollout-generation-options");
    if (options) options.classList.toggle("hidden", isManiSkill);
    if (trialsCaption) {
      trialsCaption.textContent = isManiSkill ? "Successful rollouts" : originalTrialsCaption;
    }
  }

  function integerInRange(value, minimum, maximum) {
    return Number.isInteger(value) && value >= minimum && value <= maximum;
  }

  function evenResolution(value) {
    return integerInRange(value, 64, 2048) && value % 2 === 0;
  }

  function updateManiSkillSelection() {
    var isManiSkill = maniskillSelected();
    toggleModeFields(isManiSkill);
    if (!isManiSkill) {
      if (typeof originalSelection === "function") originalSelection();
      return;
    }

    var title = document.getElementById("rolloutGenerationTitle");
    var description = document.getElementById("rolloutGenerationDescription");
    var note = document.getElementById("rolloutGenerationSelection");
    var button = document.getElementById("rolloutGenerationRun");
    var envId = envSelect.value;
    var gpu = document.getElementById("rolloutGenerationGpu").value.trim();
    var trials = Number(document.getElementById("rolloutGenerationTrials").value);
    var seed = Number(document.getElementById("rolloutGenerationSeed").value);
    var width = Number(document.getElementById("rolloutGenerationRenderWidth").value);
    var height = Number(document.getElementById("rolloutGenerationRenderHeight").value);
    var label = document.getElementById("rolloutGenerationLabel").value.trim();

    if (title) title.textContent = "Generate successful ManiSkill3 rollouts";
    if (description) {
      description.textContent = "Uses the configured project-local ManiSkill3 environment and official Panda motion-planning solutions. Only successful trajectories/videos are kept. Render width and height are configurable here; the existing LIBERO/OpenVLA generator is unchanged.";
    }

    var valid = /^\d+$/.test(gpu)
      && integerInRange(trials, 1, 50)
      && Number.isInteger(seed) && seed >= 0
      && evenResolution(width) && evenResolution(height)
      && (!label || /^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$/.test(label));

    if (!valid) {
      if (note) {
        note.textContent = "GPU must be numeric; successful rollouts must be 1-50; seed must be non-negative; render width/height must be even values from 64 to 2048.";
      }
      if (button) button.disabled = true;
      return;
    }

    if (note) {
      note.textContent = "ManiSkill3 " + envId + ": " + trials
        + " successful rollout(s), render " + width + "x" + height
        + ". Output is stored under outputs/maniskill3; no LIBERO manifest is rebuilt.";
    }
    if (button) button.disabled = Boolean(state.rolloutGenerationSubmitting);
  }

  window.rolloutGenerationJobMessage = function rolloutGenerationJobMessageWithManiSkill(job) {
    if (!job || job.generator !== "maniskill3") {
      return typeof originalJobMessage === "function"
        ? originalJobMessage(job)
        : "Rollout generation status unavailable.";
    }
    var progress = (job.completed_rollouts || 0) + "/" + (job.expected_rollouts || 0);
    var label = "ManiSkill3 " + (job.maniskill_env_id || "task");
    var resolution = (job.render_width || "?") + "x" + (job.render_height || "?");
    if (job.status === "queued") {
      return label + " generation queued at " + resolution + " - " + progress + " successful rollout(s).";
    }
    if (job.status === "running") {
      return "Generating " + label + " at " + resolution + " - " + progress + " successful rollout(s).";
    }
    if (job.status === "complete") {
      return label + " generation complete at " + resolution + ": " + progress
        + " successful rollout(s), saved under " + (job.run_root || "outputs/maniskill3") + ".";
    }
    return label + " generation failed after " + progress + " successful rollout(s); inspect the log below.";
  };

  async function startManiSkillGeneration(event) {
    event.preventDefault();
    event.stopImmediatePropagation();
    if (state.rolloutGenerationSubmitting) return;

    updateManiSkillSelection();
    var button = document.getElementById("rolloutGenerationRun");
    if (button && button.disabled) {
      setRolloutGenerationStatus("Fix the ManiSkill3 generation settings before starting.", "error");
      return;
    }

    var gpu = document.getElementById("rolloutGenerationGpu").value.trim();
    var envId = envSelect.value;
    var trials = Number(document.getElementById("rolloutGenerationTrials").value);
    var seed = Number(document.getElementById("rolloutGenerationSeed").value);
    var width = Number(document.getElementById("rolloutGenerationRenderWidth").value);
    var height = Number(document.getElementById("rolloutGenerationRenderHeight").value);
    var label = document.getElementById("rolloutGenerationLabel").value.trim();

    if (!window.confirm(
      "Generate " + trials + " successful ManiSkill3 " + envId
      + " rollout(s) at " + width + "x" + height
      + "? The existing LIBERO generator will not be used."
    )) return;

    state.rolloutGenerationSubmitting = true;
    updateManiSkillSelection();
    document.getElementById("rolloutGenerationLog").textContent = "";
    setRolloutGenerationStatus("Starting ManiSkill3 " + envId + " generation...", "");
    try {
      var response = await fetch("/api/rollouts/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          generator: "maniskill3",
          gpu: gpu,
          maniskill_env_id: envId,
          trials: trials,
          seed: seed,
          run_label: label,
          render_width: width,
          render_height: height
        })
      });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not start ManiSkill3 generation");
      state.rolloutGenerationJob = payload.job;
      rememberPersistentJob(payload.job);
      setRolloutGenerationStatus(window.rolloutGenerationJobMessage(payload.job), "");
      await loadRolloutGenerationLog(payload.job.job_id);
      pollRolloutGenerationJob(payload.job.job_id);
    } catch (error) {
      setRolloutGenerationStatus("ManiSkill3 generation error: " + error.message, "error");
    } finally {
      state.rolloutGenerationSubmitting = false;
      updateManiSkillSelection();
    }
  }

  form.addEventListener("submit", function (event) {
    if (!maniskillSelected()) return;
    startManiSkillGeneration(event);
  }, true);

  backendSelect.addEventListener("change", updateManiSkillSelection);
  [envSelect,
   document.getElementById("rolloutGenerationRenderWidth"),
   document.getElementById("rolloutGenerationRenderHeight")].forEach(function (node) {
    node.addEventListener(node.tagName === "SELECT" ? "change" : "input", updateManiSkillSelection);
  });
  [
    "rolloutGenerationGpu",
    "rolloutGenerationTrials",
    "rolloutGenerationSeed",
    "rolloutGenerationLabel"
  ].forEach(function (id) {
    var node = document.getElementById(id);
    if (node) node.addEventListener("input", updateManiSkillSelection);
  });

  updateManiSkillSelection();
})();
