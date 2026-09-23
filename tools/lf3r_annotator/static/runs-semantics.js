"use strict";

(function alignRunsSemantics() {
  if (!window.state) return;

  var SCOPE_LABELS = {
    libero_10: "LIBERO-10",
    libero_spatial: "LIBERO-Spatial",
    controlled_analysis: "Controlled",
    all: "All loaded rollouts"
  };

  var scope = document.getElementById("baselineBatchScope");
  if (scope) {
    var previous = scope.value;
    scope.innerHTML = [
      '<option value="libero_10">LIBERO-10</option>',
      '<option value="libero_spatial">LIBERO-Spatial</option>',
      '<option value="controlled_analysis">Controlled</option>',
      '<option value="all">All loaded rollouts</option>'
    ].join("");
    scope.value = Object.prototype.hasOwnProperty.call(SCOPE_LABELS, previous)
      ? previous
      : "libero_10";
  }

  if (window.BASELINE_BATCH_SCOPE_LABELS) {
    Object.keys(SCOPE_LABELS).forEach(function (key) {
      window.BASELINE_BATCH_SCOPE_LABELS[key] = SCOPE_LABELS[key];
    });
  }
  try {
    Object.keys(SCOPE_LABELS).forEach(function (key) {
      BASELINE_BATCH_SCOPE_LABELS[key] = SCOPE_LABELS[key];
    });
  } catch (_) {}

  var title = document.getElementById("baselineBatchTitle");
  if (title) title.textContent = "Run one method over a dataset group";

  var memoryLabel = document.getElementById("baselineBatchMemoryUtilization");
  if (memoryLabel && memoryLabel.closest("label")) {
    var labelText = memoryLabel.closest("label").querySelector("span");
    if (labelText) labelText.textContent = "vLLM free-memory target";
  }

  var resourceNote = document.querySelector("#baselineBatchForm .batch-resource-warning");
  if (resourceNote) {
    resourceNote.textContent = "GPU choice is user-managed. The status panel is informational; the WebUI does not block launch based on utilization or free-memory thresholds.";
  }

  var suite = document.getElementById("rolloutGenerationSuite");
  if (suite) {
    var libero10 = suite.querySelector('option[value="libero_10"]');
    var spatial = suite.querySelector('option[value="libero_spatial"]');
    if (libero10) libero10.textContent = "LIBERO-10";
    if (spatial) spatial.textContent = "LIBERO-Spatial";
  }

  function updateGenerationDescription() {
    var description = document.getElementById("rolloutGenerationDescription");
    if (!description) return;
    var suiteNode = document.getElementById("rolloutGenerationSuite");
    var renderNode = document.getElementById("rolloutGenerationRenderResolution");
    var recordNode = document.getElementById("rolloutGenerationRecordResolution");
    var viewNode = document.getElementById("rolloutGenerationVideoViewMode");
    var suiteName = suiteNode && suiteNode.value === "libero_spatial" ? "LIBERO-Spatial" : "LIBERO-10";
    var render = renderNode ? renderNode.value : "?";
    var record = recordNode ? recordNode.value : "?";
    var multiview = viewNode && viewNode.value === "libero_three_view";
    description.textContent = "Uses the existing OpenVLA " + suiteName
      + " checkpoint. Render is " + render + "x" + render
      + ", record is " + record + "x" + record
      + ", and policy preprocessing remains 224x224. "
      + (multiview
        ? "After inference, LF3R replays the recorded actions without the model and writes the three Robo-Dopamine input slots: cam_high from agentview, cam_left_wrist from robot0_eye_in_hand, and cam_right_wrist from sideview. "
        : "Only the canonical single-view replay video is written. ")
      + "GPU selection is manual; no GPU memory/utilization admission gate is applied.";
  }

  ["rolloutGenerationSuite", "rolloutGenerationRenderResolution", "rolloutGenerationRecordResolution", "rolloutGenerationVideoViewMode"].forEach(function (id) {
    var node = document.getElementById(id);
    if (!node) return;
    var eventName = (id === "rolloutGenerationSuite" || id === "rolloutGenerationVideoViewMode") ? "change" : "input";
    node.addEventListener(eventName, updateGenerationDescription);
  });
  updateGenerationDescription();

  var baselineJobs = document.getElementById("baselineBatchJobs");
  if (baselineJobs) baselineJobs.setAttribute("aria-label", "Persistent baseline jobs");
  var baselineLog = document.getElementById("baselineBatchLog");
  if (baselineLog) baselineLog.setAttribute("aria-label", "Selected baseline job log");
  var activity = baselineJobs && baselineJobs.closest(".runs-activity");
  var activityTitle = activity && activity.querySelector(".runs-activity-heading h3");
  if (activityTitle) activityTitle.textContent = "Baseline jobs";

  var originalPersistentJobScope = window.persistentJobScope;
  if (typeof originalPersistentJobScope === "function") {
    var semanticPersistentJobScope = function (job) {
      if (job && job.job_type === "baseline" && SCOPE_LABELS[job.scope]) {
        return SCOPE_LABELS[job.scope];
      }
      return originalPersistentJobScope(job);
    };
    window.persistentJobScope = semanticPersistentJobScope;
    try { persistentJobScope = semanticPersistentJobScope; } catch (_) {}
  }

  var originalJobMessage = window.rolloutGenerationJobMessage;
  if (typeof originalJobMessage === "function") {
    var semanticJobMessage = function (job) {
      if (job && job.status === "memory_blocked") {
        return "This older rollout-generation job was stopped by a legacy resource check; inspect its log.";
      }
      return originalJobMessage(job);
    };
    window.rolloutGenerationJobMessage = semanticJobMessage;
    try { rolloutGenerationJobMessage = semanticJobMessage; } catch (_) {}
  }

  var originalCliHelpEntry = window.cliHelpEntry;
  if (typeof originalCliHelpEntry === "function") {
    var semanticCliHelpEntry = function (key) {
      var entry = originalCliHelpEntry(key);
      if (!entry) return entry;
      if (key === "baseline.scope") {
        return Object.assign({}, entry, {
          default: "LIBERO-10",
          description: "Groups the current dataset into LIBERO-10, LIBERO-Spatial, Controlled, or all loaded rollouts. LIBERO groups are selected by task_suite."
        });
      }
      return entry;
    };
    window.cliHelpEntry = semanticCliHelpEntry;
    try { cliHelpEntry = semanticCliHelpEntry; } catch (_) {}
  }

  if (typeof window.baselineBatchScopeChanged === "function") {
    window.baselineBatchScopeChanged();
  } else {
    try {
      if (typeof baselineBatchScopeChanged === "function") baselineBatchScopeChanged();
    } catch (_) {}
  }
})();
