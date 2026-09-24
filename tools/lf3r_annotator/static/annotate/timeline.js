"use strict";

/* Video playback, frame navigation, and timeline markers. */

var TIMELINE_MARKER_DEFINITIONS = [
  { field: "causal_onset_frame", cssClass: "causal", label: "Causal / injected onset" },
  { field: "observable_onset_frame", cssClass: "observable", label: "Observable onset" },
  { field: "terminal_failure_frame", cssClass: "terminal", label: "Terminal failure" },
  { field: "recovery_frame", cssClass: "recovery", label: "Recovery" }
];

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
