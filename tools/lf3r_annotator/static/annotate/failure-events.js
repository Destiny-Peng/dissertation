"use strict";

/* Failure-event editing and frame assignment. */

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

var failureTypeHotkeys = {
  "1": "grasp_failure",
  "2": "placement_failure",
  "3": "dropped_object",
  "4": "wrong_object",
  "5": "collision",
  "6": "timeout_no_progress",
  "7": "control_error",
  "8": "observation_error",
  "9": "other"
};

var failurePointHotkeys = {
  causal_onset_frame: "Q",
  observable_onset_frame: "W",
  terminal_failure_frame: "E",
  recovery_frame: "R"
};

function failureTypeHotkey(value) {
  var key = Object.keys(failureTypeHotkeys).find(function (candidate) {
    return failureTypeHotkeys[candidate] === value;
  });
  return key || "";
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

function failureTypeOptions(selected) {
  return failureTypeChoices.map(function (choice) {
    var hotkey = failureTypeHotkey(choice[0]);
    var label = hotkey ? hotkey + " · " + choice[1] : choice[1];
    return '<option value="' + choice[0] + '"' + (choice[0] === selected ? " selected" : "") + ">"
      + escapeHtml(label) + "</option>";
  }).join("");
}

function eventFrameControl(index, field, title, help, cssClass) {
  var event = state.failureEvents[index];
  var value = event[field] == null ? "" : String(event[field]);
  var record = selectedRollout();
  var maxFrame = record ? Math.max(0, Number(record.total_frames) - 1) : 0;
  var hotkey = failurePointHotkeys[field] || "";
  return '<div class="onset-field ' + cssClass + '-field">'
    + '<div><span>' + escapeHtml(title) + (hotkey ? ' <kbd>' + hotkey + '</kbd>' : '') + '</span><small>' + escapeHtml(help) + '</small></div>'
    + '<div class="frame-entry">'
    + '<input type="number" min="0" max="' + maxFrame + '" placeholder="—" value="' + escapeHtml(value) + '" data-event-field="' + field + '">'
    + '<button data-event-set="' + field + '" type="button">Use current' + (hotkey ? ' · ' + hotkey : '') + '</button>'
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

function cycleActiveFailureEvent(delta) {
  if (!state.failureEvents.length) return;
  var current = state.activeFailureEvent == null ? 0 : Number(state.activeFailureEvent);
  var next = (current + Number(delta) + state.failureEvents.length) % state.failureEvents.length;
  setActiveFailureEvent(next);
  var card = byId("failureEvents").querySelector('[data-event-index="' + next + '"]');
  if (card && typeof card.scrollIntoView === "function") {
    card.scrollIntoView({ block: "nearest" });
  }
}

function setActiveFailureType(value) {
  if (!failureTypeChoices.some(function (choice) { return choice[0] === value; })) return;
  if (state.activeFailureEvent == null || !state.failureEvents[state.activeFailureEvent]) {
    addFailureEvent(emptyFailureEvent(value, null));
  } else {
    state.failureEvents[state.activeFailureEvent].failure_type = value;
    var card = byId("failureEvents").querySelector(
      '[data-event-index="' + state.activeFailureEvent + '"]'
    );
    var select = card ? card.querySelector('[data-event-field="failure_type"]') : null;
    if (select) select.value = value;
    markDirty();
  }
  if (state.activeFailureEvent === 0) {
    byId("failureType").value = value;
  }
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
