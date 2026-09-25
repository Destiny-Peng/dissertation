"use strict";

/* Rollout catalog, filtering, instruction variants, and selection. */

var INSTRUCTION_CONDITION_ORDER = ["full_instruction", "subtask_a", "subtask_b"];

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
    if (!path || manifest.valid === false) return;
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
  var payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Could not load rollout manifest");
  }
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
}

function updateRolloutListSelection(id) {
  var container = byId("rolloutList");
  if (!container) return;
  container.querySelectorAll("[data-rollout-id]").forEach(function (button) {
    button.classList.toggle("active", button.dataset.rolloutId === id);
  });
}

function maybeSelectRollout(id) {
  if (state.dirty && !window.confirm("Discard unsaved annotation changes?")) {
    return;
  }
  selectRollout(id);
}

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
  updateRolloutListSelection(id);

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
  if (window.LF3RReviewVideoViews
      && typeof window.LF3RReviewVideoViews.applyRecord === "function") {
    window.LF3RReviewVideoViews.applyRecord(record);
  } else {
    // Fallback for partial/static loads where the shared view controller is unavailable.
    video.src = "/api/videos/" + encodeURIComponent(record.id) + "?v=video-h264-20260916";
    video.load();
    video.playbackRate = Number(byId("speedSelect").value);
  }
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
