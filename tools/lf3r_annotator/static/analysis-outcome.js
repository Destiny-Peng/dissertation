"use strict";

(function installOutcomeEvaluationUi() {
  var initialized = false;
  var methods = ["procvlm", "rynnvalue", "robo_dopamine"];
  var labels = {
    procvlm: "ProcVLM",
    rynnvalue: "RynnValue",
    robo_dopamine: "Robo-Dopamine"
  };
  var state = {
    scope: "libero_10",
    coverage: [],
    evaluationPopulation: 0,
    loading: false,
    job: null,
    polling: false
  };

  function node(id) { return document.getElementById(id); }

  function escapeText(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function environmentReady() {
    return Boolean(
      window.workspaceState
      && workspaceState.analysisEnvironment
      && workspaceState.analysisEnvironment.ready
    );
  }

  function activeJob() {
    return state.job && ["queued", "running"].indexOf(state.job.status) !== -1;
  }

  function setStatus(message, kind) {
    var target = node("analysisOutcomeRunStatus");
    if (!target) return;
    target.textContent = message;
    target.className = "analysis-run-selection" + (kind ? " " + kind : "");
  }

  function renderCoverage() {
    var host = node("analysisOutcomeCoverage");
    if (!host) return;
    if (!state.coverage.length) {
      host.innerHTML = '<div class="analysis-empty">No coverage information available.</div>';
      return;
    }
    var html = '<table class="analysis-table analysis-summary-table" aria-label="Outcome evaluation coverage">'
      + '<thead><tr><th>Method</th><th>Valid outputs</th><th>Evaluation population</th><th>Coverage</th></tr></thead><tbody>';
    state.coverage.forEach(function (row) {
      var available = Number(row.available_rollouts || 0);
      var population = Number(row.evaluation_population || state.evaluationPopulation || 0);
      var fraction = population > 0 ? available / population : 0;
      html += '<tr><th scope="row">' + escapeText(labels[row.baseline] || row.baseline) + '</th>'
        + '<td class="numeric">' + escapeText(available) + '</td>'
        + '<td class="numeric">' + escapeText(population) + '</td>'
        + '<td class="numeric">' + escapeText((100 * fraction).toFixed(1) + "%") + '</td></tr>';
    });
    host.innerHTML = html + '</tbody></table>';
  }

  function updateButton() {
    var button = node("analysisOutcomeRunButton");
    if (!button) return;
    var hasAnyCoverage = state.coverage.some(function (row) {
      return Number(row.available_rollouts || 0) > 0;
    });
    button.disabled = state.loading || activeJob() || !environmentReady()
      || state.evaluationPopulation <= 0 || !hasAnyCoverage;
  }

  async function loadCoverage() {
    var scope = node("analysisOutcomeRunScope");
    state.scope = scope ? scope.value : state.scope;
    state.loading = true;
    updateButton();
    setStatus("Checking completed annotations and saved-output coverage for " + state.scope + "…", "");
    try {
      var response = await fetch(
        "/api/analysis/outcome-coverage?scope=" + encodeURIComponent(state.scope),
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not read outcome coverage");
      state.coverage = Array.isArray(payload.coverage) ? payload.coverage : [];
      state.evaluationPopulation = Number(payload.evaluation_population || 0);
      renderCoverage();
      var availableText = state.coverage.map(function (row) {
        return (labels[row.method] || row.method) + " "
          + Number(row.available_rollouts || 0) + "/" + state.evaluationPopulation;
      }).join(" · ");
      if (!state.evaluationPopulation) {
        setStatus("No review_status=complete rollouts are available in this scope.", "warning");
      } else if (!state.coverage.some(function (row) { return Number(row.available_rollouts || 0) > 0; })) {
        setStatus("No parseable saved baseline outputs overlap the completed annotations.", "warning");
      } else {
        setStatus(
          "Evaluation population: " + state.evaluationPopulation
            + " completed annotation(s). Each method uses its own newest-output union: "
            + availableText + ".",
          ""
        );
      }
    } catch (error) {
      state.coverage = [];
      state.evaluationPopulation = 0;
      renderCoverage();
      setStatus("Coverage check failed: " + error.message, "error");
    } finally {
      state.loading = false;
      updateButton();
    }
  }

  async function loadLog(jobId) {
    try {
      var response = await fetch(
        "/api/analysis-jobs/" + encodeURIComponent(jobId) + "/log?tail=200",
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (response.ok && state.job && state.job.job_id === jobId) {
        node("analysisOutcomeRunLog").textContent = payload.log ? payload.log.text : "";
      }
    } catch (_error) {}
  }

  async function poll(jobId) {
    if (state.polling) return;
    state.polling = true;
    try {
      var response = await fetch(
        "/api/analysis-jobs/" + encodeURIComponent(jobId),
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not read outcome-evaluation job");
      if (!payload.job || payload.job.analysis_kind !== "rollout_outcome_evaluation") return;
      state.job = payload.job;
      await loadLog(jobId);
      if (state.job.status === "queued" || state.job.status === "running") {
        setStatus(
          "Outcome evaluation " + state.job.status + " · "
            + state.job.selected_rollouts + " completed annotation(s) · CPU post-processing only.",
          ""
        );
        state.polling = false;
        updateButton();
        window.setTimeout(function () { poll(jobId); }, 1200);
        return;
      }
      if (state.job.status === "complete") {
        setStatus("Outcome evaluation complete · " + (state.job.output_dir || "snapshot ready"), "");
        if (typeof window.workspaceLoadAnalysis === "function") {
          await window.workspaceLoadAnalysis(true);
        }
        await loadCoverage();
      } else {
        setStatus("Outcome evaluation failed; inspect the log.", "error");
      }
    } catch (error) {
      setStatus("Outcome evaluation job error: " + error.message, "error");
    } finally {
      state.polling = false;
      updateButton();
    }
  }

  async function start(event) {
    event.preventDefault();
    var label = node("analysisOutcomeOutputLabel").value.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(label)) {
      setStatus("Output label may contain only letters, numbers, dot, underscore, or hyphen.", "error");
      return;
    }
    node("analysisOutcomeRunLog").textContent = "";
    setStatus("Resolving newest saved output per rollout and starting outcome evaluation…", "");
    try {
      var response = await fetch("/api/analysis/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          analysis_kind: "rollout_outcome_evaluation",
          scope: state.scope,
          output_label: label
        })
      });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not start outcome evaluation");
      state.job = payload.job;
      updateButton();
      await loadLog(state.job.job_id);
      poll(state.job.job_id);
    } catch (error) {
      state.job = null;
      setStatus("Outcome evaluation error: " + error.message, "error");
      updateButton();
    }
  }

  async function recover() {
    try {
      var response = await fetch("/api/jobs?job_type=analysis", { cache: "no-store" });
      var payload = await response.json();
      if (!response.ok) return;
      var jobs = (payload.jobs || []).filter(function (job) {
        return job.analysis_kind === "rollout_outcome_evaluation";
      });
      if (!jobs.length) return;
      jobs.sort(function (left, right) {
        return String(right.submitted_at || "").localeCompare(String(left.submitted_at || ""));
      });
      state.job = jobs[0];
      if (activeJob()) poll(state.job.job_id);
      else loadLog(state.job.job_id);
    } catch (_error) {}
  }

  function init() {
    if (initialized || !node("analysisOutcomeRunForm")) return;
    initialized = true;
    node("analysisOutcomeRunForm").addEventListener("submit", start);
    node("analysisOutcomeRunScope").addEventListener("change", loadCoverage);
    loadCoverage();
    recover();
    updateButton();
  }

  window.lf3rOutcomeEnvironmentChanged = updateButton;
  window.addEventListener("lf3r:viewchange", function (event) {
    if (event.detail && event.detail.view === "analysis") init();
  });
  init();
})();
