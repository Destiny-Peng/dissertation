"use strict";

(function installOutcomeEvaluationUi() {
  var initialized = false;
  var state = {
    scope: "libero_10",
    runs: [],
    job: null,
    loading: false,
    polling: false
  };

  function node(id) { return document.getElementById(id); }

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

  function selectedValues(select) {
    if (!select) return [];
    return Array.prototype.filter.call(select.options, function (option) {
      return option.selected && option.value;
    }).map(function (option) { return option.value; });
  }

  function runLabel(run) {
    var coverage = Number(run.selected_scope_rollouts || 0);
    var available = Number(run.scope_rollout_count || run.run_rollout_count || run.selected_rollouts || 0);
    var date = run.completed_at || run.created_at || "";
    return (run.run_root || "run") + " · " + available + "/" + coverage + (date ? " · " + date : "");
  }

  function methodSelect(method) {
    var ids = {
      safe: "analysisOutcomeRunSafe",
      procvlm: "analysisOutcomeRunProcvlm",
      rynnvalue: "analysisOutcomeRunRynnvalue",
      robo_dopamine: "analysisOutcomeRunRoboDopamine"
    };
    return node(ids[method]);
  }

  function populate() {
    ["safe", "procvlm", "rynnvalue", "robo_dopamine"].forEach(function (method) {
      var select = methodSelect(method);
      if (!select) return;
      var choices = state.runs.filter(function (run) {
        if (run.baseline !== method) return false;
        return method === "rynnvalue"
          ? Boolean(run.compatible || run.partial_compatible)
          : Boolean(run.compatible);
      });
      if (!choices.length) {
        select.innerHTML = '<option value="">No compatible completed run</option>';
        select.disabled = true;
        return;
      }
      select.disabled = false;
      select.innerHTML = choices.map(function (run) {
        var label = runLabel(run).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        var value = String(run.run_root || "").replace(/&/g, "&amp;").replace(/"/g, "&quot;");
        return '<option value="' + value + '">' + label + '</option>';
      }).join("");
      if (method === "rynnvalue") {
        var full = choices.find(function (run) { return run.compatible; });
        Array.prototype.forEach.call(select.options, function (option) {
          if (full) option.selected = option.value === full.run_root;
          else {
            var run = choices.find(function (item) { return item.run_root === option.value; });
            option.selected = Boolean(run && run.partial_compatible);
          }
        });
      } else {
        select.selectedIndex = 0;
      }
    });
    updateButton();
  }

  function allRunsSelected() {
    return ["safe", "procvlm", "rynnvalue", "robo_dopamine"].every(function (method) {
      return selectedValues(methodSelect(method)).length > 0;
    });
  }

  function updateButton() {
    var button = node("analysisOutcomeRunButton");
    if (!button) return;
    button.disabled = state.loading || activeJob() || !allRunsSelected() || !environmentReady();
  }

  async function loadRuns() {
    var scope = node("analysisOutcomeRunScope");
    state.scope = scope ? scope.value : state.scope;
    state.loading = true;
    updateButton();
    setStatus("Loading completed baseline runs for " + state.scope + "…", "");
    try {
      var response = await fetch(
        "/api/baselines/runs?scope=" + encodeURIComponent(state.scope),
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not discover baseline runs");
      state.runs = Array.isArray(payload.runs) ? payload.runs : [];
      populate();
      if (allRunsSelected()) {
        setStatus("Compatible saved outputs are ready. This job only reads existing baseline results.", "");
      } else {
        setStatus("A compatible completed run is required for all four methods.", "warning");
      }
    } catch (error) {
      state.runs = [];
      populate();
      setStatus("Baseline run discovery failed: " + error.message, "error");
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
            + state.job.selected_rollouts + " rollout(s) · CPU post-processing only.",
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
    var runs = {};
    ["safe", "procvlm", "robo_dopamine"].forEach(function (method) {
      runs[method] = selectedValues(methodSelect(method))[0] || "";
    });
    runs.rynnvalue = selectedValues(methodSelect("rynnvalue"));
    if (!allRunsSelected()) {
      setStatus("Select compatible completed runs for all four methods.", "warning");
      return;
    }
    var label = node("analysisOutcomeOutputLabel").value.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(label)) {
      setStatus("Output label may contain only letters, numbers, dot, underscore, or hyphen.", "error");
      return;
    }
    node("analysisOutcomeRunLog").textContent = "";
    setStatus("Starting outcome evaluation…", "");
    try {
      var response = await fetch("/api/analysis/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          analysis_kind: "rollout_outcome_evaluation",
          scope: state.scope,
          runs: runs,
          output_label: label,
          allow_partial_coverage: true
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
      state.job = jobs[0];
      if (activeJob()) poll(state.job.job_id);
      else loadLog(state.job.job_id);
    } catch (_error) {}
  }

  function init() {
    if (initialized || !node("analysisOutcomeRunForm")) return;
    initialized = true;
    node("analysisOutcomeRunForm").addEventListener("submit", start);
    node("analysisOutcomeRunScope").addEventListener("change", loadRuns);
    ["analysisOutcomeRunSafe", "analysisOutcomeRunProcvlm", "analysisOutcomeRunRynnvalue", "analysisOutcomeRunRoboDopamine"]
      .forEach(function (id) { node(id).addEventListener("change", updateButton); });
    loadRuns();
    recover();
    updateButton();
  }

  window.lf3rOutcomeEnvironmentChanged = updateButton;
  window.addEventListener("lf3r:viewchange", function (event) {
    if (event.detail && event.detail.view === "analysis") init();
  });
  init();
})();
