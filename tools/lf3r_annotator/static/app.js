"use strict";

var state = {
  rollouts: [],
  manifests: [],
  manifestFilter: "all",
  filtered: [],
  selectedId: null,
  currentFrame: 0,
  failureEvents: [],
  activeFailureEvent: null,
  dirty: false,
  evaluation: null,
  instructionCondition: "full_instruction",
  evaluationRequest: 0,
  evaluationSignalVisibility: {},
  baselineCollapsed: {},
  roboPosthocCheckpoint: "",
  baselineRuns: null,
  baselineRunsCondition: null,
  baselineRunsLoading: null,
  baselineRunSelections: {},
  baselineRunAll: {},
  baselineRunNotice: "",
  baselineBatchCoverage: null,
  baselineBatchCoverageKey: null,
  baselineBatchCoverageRequest: 0,
  baselineBatchCoverageLoading: false,
  baselineJob: null,
  baselineBatchJob: null,
  baselineBatchJobs: {},
  baselineBatchWorkers: [{ gpu: "0", start_index: 0, end_index: 0 }],
  rolloutGenerationJob: null,
  rolloutGenerationJobs: {},
  persistentJobs: [],
  persistentJobLogOpen: {},
  persistentJobLogText: {},
  persistentJobPollTimers: {},
  baselineBatchSubmitting: false,
  rolloutGenerationSubmitting: false
};

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}


installEvents();
loadRollouts().catch(function (error) {
  byId("datasetStatus").textContent = "Dataset error";
  byId("rolloutList").innerHTML = '<div class="form-error">' + escapeHtml(error.message) + "</div>";
});

// Chart interaction is delegated because cards are replaced when changing runs.
document.addEventListener('click', function (event) {
  var plot = event.target.closest('[data-signal-seek]');
  if (!plot) return;
  var box = plot.getBoundingClientRect();
  if (box.width > 0) seekFrame((event.clientX - box.left) / box.width * Number(plot.dataset.frameMax));
});
document.addEventListener('keydown', function (event) {
  var plot = event.target.closest('[data-signal-seek]');
  if (!plot || ['ArrowLeft', 'ArrowRight', 'Home', 'End'].indexOf(event.key) === -1) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  seekFrame(event.key === 'Home' ? 0 : event.key === 'End' ? Number(plot.dataset.frameMax) : currentFrame() + (event.key === 'ArrowLeft' ? -1 : 1));
}, true);

byId('rolloutVideo').addEventListener('seeked', function () {
  var record = selectedRollout();
  if (!record) return;
  state.currentFrame = Math.max(0, Math.min(Number(record.total_frames) - 1, Math.floor(this.currentTime * Number(record.fps) + 0.0001)));
  byId('frameSlider').value = state.currentFrame;
  updateReadout();
});

(function installResultsVideoPin() {
  var toggle = byId('resultsPinVideo');
  var dock = byId('resultsVideoDock');
  try { toggle.checked = localStorage.getItem('lf3r.results.pinVideo') === 'true'; } catch (_) {}
  function applyPin() {
    dock.classList.toggle('is-pinned', toggle.checked);
    try { localStorage.setItem('lf3r.results.pinVideo', String(toggle.checked)); } catch (_) {}
  }
  toggle.addEventListener('change', applyPin);
  applyPin();
})();
