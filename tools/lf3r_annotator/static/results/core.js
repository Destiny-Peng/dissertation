"use strict";

/*
 * Results controller.
 *
 * Loading and event binding live here; model/catalog/view/action details are
 * owned by neighboring semantic modules.
 */

async function loadEvaluation(rolloutId) {
  var requestId = ++state.evaluationRequest;
  var condition = state.instructionCondition || "full_instruction";
  var record = state.rollouts.find(function (item) { return item.id === rolloutId; }) || null;
  state.evaluation = null;
  byId("evaluationStatus").textContent = "Loading baseline outputs...";
  byId("evaluationMethods").innerHTML = '<div class="evaluation-empty">Reading baseline outputs for this rollout...</div>';
  try {
    var query = ["condition=" + encodeURIComponent(condition)];
    ["safe", "procvlm", "rynnvalue", "robo_dopamine", "densereward"].forEach(function (method) {
      var override = baselineRunOverride(method, record, condition);
      if (override) query.push("run_" + method + "=" + encodeURIComponent(override));
    });
    var response = await fetch(
      "/api/baselines/" + encodeURIComponent(rolloutId) + "?" + query.join("&"),
      { cache: "no-store" }
    );
    var payload = await response.json();
    if (requestId !== state.evaluationRequest
      || state.selectedId !== rolloutId
      || state.instructionCondition !== condition) return;
    if (!response.ok) throw new Error(payload.error || "Could not load baseline outputs");
    renderEvaluationPanel(payload.evaluation);
    loadBaselineRunCatalog(condition).then(function () {
      if (requestId !== state.evaluationRequest
          || state.selectedId !== rolloutId
          || state.instructionCondition !== condition) return;
      refreshBaselineRunControls(rolloutId, condition);
    });
  } catch (error) {
    if (requestId !== state.evaluationRequest
      || state.selectedId !== rolloutId
      || state.instructionCondition !== condition) return;
    byId("evaluationStatus").textContent = "Baseline output error: " + error.message;
    byId("evaluationMethods").innerHTML = '<div class="evaluation-empty">Select Refresh to try again.</div>';
  }
}


function hydrateEvaluationCardBody(card, methodName) {
  if (!card || !methodName) return;
  var body = card.querySelector("[data-evaluation-card-body]");
  if (!body || body.dataset.bodyRendered === "true") return;
  var result = state.evaluation && state.evaluation.methods
    ? state.evaluation.methods[methodName] : null;
  var record = selectedRollout();
  if (!result || !record) return;
  body.innerHTML = renderEvaluationCardBody(methodName, result, record);
  body.dataset.bodyRendered = "true";
  updateEvaluationCurrent();
  scheduleEvaluationChartHydration(card);
  if (typeof window.lf3rResultsLayoutRefresh === "function") {
    window.lf3rResultsLayoutRefresh();
  }
}

function bindResultsEvents() {
  var reload = byId("reloadEvaluation");
  if (reload) {
    reload.addEventListener("click", function () {
      if (state.selectedId) loadEvaluation(state.selectedId);
    });
  }

  var methods = byId("evaluationMethods");
  if (!methods) return;

  methods.addEventListener("change", function (event) {
    var posthocSelect = event.target.closest("[data-posthoc-localization-select]");
    if (posthocSelect) {
      selectPosthocLocalization(posthocSelect);
      return;
    }
    var select = event.target.closest("[data-evaluation-run-select]");
    if (!select) return;
    var record = selectedRollout();
    if (!record) return;
    var method = select.dataset.evaluationMethod;
    var key = baselineRunPreferenceKey(method, record, state.instructionCondition);
    state.baselineRunSelections[key] = select.value || BASELINE_AUTO_RUN;
    loadEvaluation(record.id);
  });

  methods.addEventListener("toggle", function (event) {
    var history = event.target;
    if (!history || !history.matches || !history.matches("[data-evaluation-history]")) return;
    if (history.open) hydrateEvaluationHistory(history);
  }, true);

  methods.addEventListener("click", function (event) {
    var collapseButton = event.target.closest("[data-toggle-baseline-card]");
    if (collapseButton) {
      var methodName = collapseButton.dataset.toggleBaselineCard;
      state.baselineCollapsed[methodName] = !baselineCardCollapsed(methodName);
      persistBaselineCollapsed();
      var card = collapseButton.closest("[data-evaluation-method]");
      var body = card && card.querySelector("[data-evaluation-card-body]");
      var collapsed = baselineCardCollapsed(methodName);
      if (card) card.classList.toggle("is-collapsed", collapsed);
      if (!collapsed) hydrateEvaluationCardBody(card, methodName);
      if (body) body.hidden = collapsed;
      collapseButton.textContent = collapsed ? "Expand" : "Collapse";
      collapseButton.setAttribute("aria-expanded", String(!collapsed));
      collapseButton.title = (collapsed ? "Expand" : "Collapse") + " baseline result";
      return;
    }

    var posthocButton = event.target.closest("[data-run-posthoc-localization]");
    if (posthocButton) {
      runPosthocLocalization(posthocButton.dataset.runPosthocLocalization, posthocButton);
      return;
    }

    var applyButton = event.target.closest("[data-apply-baseline-run]");
    if (applyButton) {
      var record = selectedRollout();
      if (!record) return;
      var method = applyButton.dataset.evaluationMethod;
      var card = applyButton.closest("[data-evaluation-method]");
      var select = card && card.querySelector("[data-evaluation-run-select]");
      var value = select ? (select.value || BASELINE_AUTO_RUN) : BASELINE_AUTO_RUN;
      var key = baselineRunAllKey(method, state.instructionCondition);
      if (value === BASELINE_AUTO_RUN) delete state.baselineRunAll[key];
      else state.baselineRunAll[key] = value;
      state.baselineRunNotice = value === BASELINE_AUTO_RUN
        ? (method + " reverted to automatic run selection for all rollouts")
        : (method + " run applied to all rollouts when that run contains the rollout");
      loadEvaluation(record.id);
      return;
    }

    var signalButton = event.target.closest("[data-evaluation-signal-toggle]");
    if (signalButton) {
      toggleEvaluationSignal(signalButton);
      return;
    }

    var runButton = event.target.closest("[data-run-baseline]");
    if (runButton) {
      var opener = window.lf3rOpenSingleBaselineConfig;
      if (typeof opener === "function") {
        opener(String(runButton.dataset.runBaseline || ""));
      } else {
        var status = byId("evaluationStatus");
        if (status) status.textContent = "Baseline configurator is still loading. Try again.";
      }
    }
  });
}
