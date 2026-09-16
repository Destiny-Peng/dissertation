"use strict";

(function installRunsLayoutEnhancements() {
  var view = document.getElementById("runsView");
  if (!view || view.dataset.runsEnhanced === "true") return;

  var batchPanel = view.querySelector(".batch-panel");
  var rolloutPanel = view.querySelector(".rollout-generation-panel");
  if (!batchPanel || !rolloutPanel) return;
  view.dataset.runsEnhanced = "true";

  var legacyTitle = view.querySelector(":scope > h2");
  if (legacyTitle) legacyTitle.classList.add("runs-legacy-title");

  var header = document.createElement("header");
  header.className = "runs-shell-header";
  header.innerHTML = [
    '<div class="runs-shell-copy">',
      '<p class="eyebrow">EXPERIMENT CONTROL</p>',
      '<h2>Runs</h2>',
      '<p>Launch inference and rollout jobs without mixing configuration, status, and logs into one long form.</p>',
    '</div>',
    '<div class="runs-mode-switch" role="tablist" aria-label="Run type">',
      '<button id="runsBaselineTab" type="button" role="tab" data-runs-mode="baseline">Baseline inference</button>',
      '<button id="runsRolloutTab" type="button" role="tab" data-runs-mode="rollout">Rollout generation</button>',
    '</div>'
  ].join("");

  var workspace = document.createElement("div");
  workspace.className = "runs-console";
  batchPanel.parentNode.insertBefore(header, batchPanel);
  batchPanel.parentNode.insertBefore(workspace, batchPanel);
  workspace.appendChild(batchPanel);
  workspace.appendChild(rolloutPanel);

  batchPanel.dataset.runsPanel = "baseline";
  rolloutPanel.dataset.runsPanel = "rollout";
  batchPanel.id = batchPanel.id || "runsBaselinePanel";
  rolloutPanel.id = rolloutPanel.id || "runsRolloutPanel";
  batchPanel.setAttribute("role", "tabpanel");
  rolloutPanel.setAttribute("role", "tabpanel");
  batchPanel.setAttribute("aria-labelledby", "runsBaselineTab");
  rolloutPanel.setAttribute("aria-labelledby", "runsRolloutTab");

  function wrapSetupBlock(node, title, subtitle) {
    if (!node || node.parentElement.classList.contains("runs-config-block")) return;
    var wrapper = document.createElement("section");
    wrapper.className = "runs-config-block";
    var heading = document.createElement("div");
    heading.className = "runs-config-heading";
    heading.innerHTML = '<h3>' + title + '</h3>' + (subtitle ? '<p>' + subtitle + '</p>' : '');
    node.parentNode.insertBefore(wrapper, node);
    wrapper.appendChild(heading);
    wrapper.appendChild(node);
  }

  wrapSetupBlock(
    batchPanel.querySelector(".batch-control-grid"),
    "Run setup",
    "Choose the method, scope, GPU allocation, and rollout range."
  );
  wrapSetupBlock(
    rolloutPanel.querySelector(".rollout-generation-grid"),
    "Generation setup",
    "Choose the dataset scope, rendering settings, and generation range."
  );

  function makeActivity(panel, jobsId, logId, title, statusId) {
    var jobs = document.getElementById(jobsId);
    var log = document.getElementById(logId);
    var status = statusId ? document.getElementById(statusId) : null;
    if (!jobs || !log || jobs.parentElement.classList.contains("runs-activity")) return;
    var activity = document.createElement("aside");
    activity.className = "runs-activity";
    activity.innerHTML = '<div class="runs-activity-heading"><p class="eyebrow">ACTIVITY</p><h3>' + title + '</h3></div>';
    jobs.parentNode.insertBefore(activity, jobs);
    if (status) {
      status.classList.add("runs-activity-status");
      activity.appendChild(status);
    }
    activity.appendChild(jobs);
    activity.appendChild(log);
  }

  makeActivity(batchPanel, "baselineBatchJobs", "baselineBatchLog", "Baseline job");
  makeActivity(rolloutPanel, "rolloutGenerationJobs", "rolloutGenerationLog", "Generation job", "rolloutGenerationStatus");

  var advanced = document.getElementById("baselineBatchAdvanced");
  if (advanced && !advanced.dataset.runsDisclosure) {
    advanced.dataset.runsDisclosure = "true";
    advanced.classList.add("runs-advanced", "is-collapsed");
    var originalHeading = advanced.querySelector(":scope > h3");
    if (originalHeading) originalHeading.hidden = true;
    var disclosure = document.createElement("button");
    disclosure.type = "button";
    disclosure.className = "runs-advanced-toggle";
    disclosure.setAttribute("aria-expanded", "false");
    disclosure.innerHTML = '<span><strong>Advanced runner parameters</strong><small>Model-specific overrides, validation, video rendering, and dry-run controls</small></span><i aria-hidden="true">⌄</i>';
    advanced.insertBefore(disclosure, advanced.firstChild);
    disclosure.addEventListener("click", function () {
      var open = advanced.classList.toggle("is-open");
      advanced.classList.toggle("is-collapsed", !open);
      disclosure.setAttribute("aria-expanded", String(open));
      disclosure.querySelector("i").textContent = open ? "⌃" : "⌄";
    });
  }

  function decorateAction(button, note) {
    if (!button || button.parentElement.classList.contains("runs-action-row")) return;
    var row = document.createElement("div");
    row.className = "runs-action-row";
    var helper = document.createElement("span");
    helper.textContent = note;
    button.parentNode.insertBefore(row, button);
    row.appendChild(helper);
    row.appendChild(button);
  }

  decorateAction(document.getElementById("baselineBatchRun"), "Starts a persistent background job using the configuration above.");
  decorateAction(document.getElementById("rolloutGenerationRun"), "Starts natural rollout generation using the configuration above.");

  var buttons = Array.prototype.slice.call(header.querySelectorAll("[data-runs-mode]"));
  var panels = Array.prototype.slice.call(workspace.querySelectorAll("[data-runs-panel]"));
  var currentMode = "baseline";
  try {
    var storedMode = localStorage.getItem("lf3r.runs.mode");
    if (storedMode === "baseline" || storedMode === "rollout") currentMode = storedMode;
  } catch (_) {}

  buttons.forEach(function (button) {
    var panel = button.dataset.runsMode === "baseline" ? batchPanel : rolloutPanel;
    button.setAttribute("aria-controls", panel.id);
  });

  function setMode(mode) {
    currentMode = mode === "rollout" ? "rollout" : "baseline";
    buttons.forEach(function (button) {
      var active = button.dataset.runsMode === currentMode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    panels.forEach(function (panel) {
      panel.hidden = panel.dataset.runsPanel !== currentMode;
    });
    try { localStorage.setItem("lf3r.runs.mode", currentMode); } catch (_) {}
  }

  buttons.forEach(function (button, index) {
    button.addEventListener("click", function () { setMode(button.dataset.runsMode); });
    button.addEventListener("keydown", function (event) {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      var next = event.key === "ArrowRight" ? (index + 1) % buttons.length : (index - 1 + buttons.length) % buttons.length;
      buttons[next].focus();
      setMode(buttons[next].dataset.runsMode);
    });
  });

  setMode(currentMode);
})();
