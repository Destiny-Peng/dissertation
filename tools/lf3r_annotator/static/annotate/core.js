"use strict";

/* Annotation form payload, save, and rollout navigation. */

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
