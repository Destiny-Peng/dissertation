"use strict";

/* Annotate/review data, form, failure-event, and timeline logic. */

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
