"use strict";

(function installLocalizationLab() {
  var state = {
    presets: [],
    currentJob: null,
    polling: false,
    activeTab: "builder"
  };

  var PARAMS = [
    ["data.success_ratio", "Success ratio"],
    ["data.population", "Training population"],
    ["target", "Whole target config"],
    ["target.kind", "Target kind"],
    ["target.sigma_pre", "Gaussian σ pre"],
    ["target.sigma_post", "Gaussian σ post"],
    ["target.tau_event", "Event decay τ"],
    ["model.hidden", "Hidden size"],
    ["loss.name", "Loss"],
    ["loss.distance_weight", "Distance weight"],
    ["loss.ranking_weight", "Ranking weight"],
    ["loss.ranking_margin", "Ranking margin"],
    ["training.batch_size", "Batch size"],
    ["training.learning_rate", "Learning rate"],
    ["training.weight_decay", "Weight decay"],
    ["training.grad_clip", "Gradient clip"]
  ];

  function node(id) { return document.getElementById(id); }
  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function n(id) {
    var value = Number(node(id).value);
    if (!Number.isFinite(value)) throw new Error(id + " is not numeric");
    return value;
  }
  function clone(value) { return JSON.parse(JSON.stringify(value)); }

  function baseFromForm() {
    var population = node("localizationPopulation").value;
    return {
      data: {
        population: population,
        success_ratio: population === "failure_only" ? 0 : n("localizationSuccessRatio")
      },
      target: {
        kind: node("localizationTargetKind").value,
        sigma_pre: n("localizationSigmaPre"),
        sigma_post: n("localizationSigmaPost"),
        tau_event: n("localizationTauEvent")
      },
      model: { hidden: Math.round(n("localizationHidden")) },
      loss: {
        name: node("localizationLoss").value,
        distance_weight: n("localizationDistanceWeight"),
        ranking_weight: n("localizationRankingWeight"),
        ranking_margin: n("localizationRankingMargin")
      },
      training: {
        device: node("localizationDevice").value,
        batch_size: Math.round(n("localizationBatchSize")),
        epochs: Math.round(n("localizationEpochs")),
        patience: Math.round(n("localizationPatience")),
        learning_rate: n("localizationLearningRate"),
        weight_decay: n("localizationWeightDecay"),
        grad_clip: n("localizationGradClip"),
        seed: Math.round(n("localizationSeed")),
        train_fraction: n("localizationTrainFraction"),
        val_fraction: n("localizationValFraction")
      }
    };
  }

  function setBase(base) {
    var data = base.data || {}, target = base.target || {}, model = base.model || {};
    var loss = base.loss || {}, training = base.training || {};
    node("localizationPopulation").value = data.population || "failure_only";
    node("localizationSuccessRatio").value = data.success_ratio == null ? 0 : data.success_ratio;
    node("localizationTargetKind").value = target.kind || "hard";
    node("localizationSigmaPre").value = target.sigma_pre == null ? 3 : target.sigma_pre;
    node("localizationSigmaPost").value = target.sigma_post == null ? 3 : target.sigma_post;
    node("localizationTauEvent").value = target.tau_event == null ? 20 : target.tau_event;
    node("localizationHidden").value = model.hidden == null ? 16 : model.hidden;
    node("localizationLoss").value = loss.name || "bce";
    node("localizationDistanceWeight").value = loss.distance_weight == null ? 1 : loss.distance_weight;
    node("localizationRankingWeight").value = loss.ranking_weight == null ? 1 : loss.ranking_weight;
    node("localizationRankingMargin").value = loss.ranking_margin == null ? 1 : loss.ranking_margin;
    node("localizationDevice").value = training.device || "auto";
    node("localizationBatchSize").value = training.batch_size == null ? 32 : training.batch_size;
    node("localizationEpochs").value = training.epochs == null ? 300 : training.epochs;
    node("localizationPatience").value = training.patience == null ? 35 : training.patience;
    node("localizationLearningRate").value = training.learning_rate == null ? 0.003 : training.learning_rate;
    node("localizationWeightDecay").value = training.weight_decay == null ? 0.0001 : training.weight_decay;
    node("localizationGradClip").value = training.grad_clip == null ? 5 : training.grad_clip;
    node("localizationSeed").value = training.seed == null ? 17 : training.seed;
    node("localizationTrainFraction").value = training.train_fraction == null ? 0.70 : training.train_fraction;
    node("localizationValFraction").value = training.val_fraction == null ? 0.15 : training.val_fraction;
    updateConditionalFields();
  }

  function parseValues(raw) {
    raw = String(raw || "").trim();
    if (!raw) throw new Error("Sweep values cannot be empty");
    if (raw[0] === "[") {
      var parsed = JSON.parse(raw);
      if (!Array.isArray(parsed) || !parsed.length) throw new Error("Sweep JSON must be a non-empty array");
      return parsed;
    }
    return raw.split(",").map(function (part) {
      part = part.trim();
      if (!part) return null;
      if (part === "true") return true;
      if (part === "false") return false;
      var number = Number(part);
      return Number.isFinite(number) && part !== "" ? number : part;
    }).filter(function (value) { return value !== null; });
  }

  function formatValues(values) {
    return JSON.stringify(values || []);
  }

  function paramOptions(selected) {
    return PARAMS.map(function (item) {
      return '<option value="' + esc(item[0]) + '"' + (item[0] === selected ? " selected" : "") + ">"
        + esc(item[1]) + "</option>";
    }).join("");
  }

  function addSweepRow(host, definition) {
    definition = definition || { path: "target.kind", values: ["hard", "gaussian"] };
    var row = document.createElement("div");
    row.className = "localization-sweep-row";
    row.innerHTML =
      '<select data-sweep-path>' + paramOptions(definition.path) + '</select>'
      + '<input data-sweep-values type="text" value="' + esc(formatValues(definition.values)) + '">'
      + '<button type="button" class="ghost-button" data-remove-sweep>Remove</button>';
    row.querySelector("[data-remove-sweep]").addEventListener("click", function () {
      row.remove();
      refreshPreview();
    });
    row.querySelectorAll("input,select").forEach(function (input) {
      input.addEventListener("input", refreshPreview);
      input.addEventListener("change", refreshPreview);
    });
    host.appendChild(row);
    refreshPreview();
  }

  function readSweep(host) {
    return Array.prototype.map.call(host.querySelectorAll(".localization-sweep-row"), function (row) {
      return {
        path: row.querySelector("[data-sweep-path]").value,
        values: parseValues(row.querySelector("[data-sweep-values]").value)
      };
    });
  }

  function defaultSelector() {
    return {
      metric: "in_interval_rate_mean",
      mode: "max",
      tie_breakers: [
        { metric: "mae_samples_mean", mode: "min" },
        { metric: "mse_samples_mean", mode: "min" }
      ]
    };
  }

  function addStage(definition) {
    definition = definition || { name: "stage_" + (node("localizationStages").children.length + 1), sweep: [] };
    var card = document.createElement("section");
    card.className = "localization-stage-card";
    var selector = definition.select || defaultSelector();
    card.innerHTML =
      '<div class="localization-stage-header">'
      + '<input data-stage-name type="text" value="' + esc(definition.name || "stage") + '" maxlength="64">'
      + '<button type="button" class="ghost-button" data-remove-stage>Remove stage</button>'
      + '</div>'
      + '<div class="analysis-run-grid localization-stage-selector">'
      + '<label><span>Select best by</span><select data-stage-metric>'
      + '<option value="in_interval_rate_mean">In-interval rate</option>'
      + '<option value="first_event_in_interval_rate_mean">First-event in-interval</option>'
      + '<option value="mae_samples_mean">MAE</option>'
      + '<option value="mse_samples_mean">MSE</option>'
      + '<option value="before_interval_rate_mean">Before-interval rate</option>'
      + '<option value="after_interval_rate_mean">After-interval rate</option>'
      + '</select></label>'
      + '<label><span>Direction</span><select data-stage-mode>'
      + '<option value="max">Maximize</option><option value="min">Minimize</option>'
      + '</select></label>'
      + '</div>'
      + '<div class="localization-stage-sweeps"></div>'
      + '<button type="button" class="ghost-button" data-add-stage-sweep>Add sweep dimension</button>';
    card.querySelector("[data-stage-metric]").value = selector.metric || "in_interval_rate_mean";
    card.querySelector("[data-stage-mode]").value = selector.mode || "max";
    node("localizationStages").appendChild(card);
    var sweepHost = card.querySelector(".localization-stage-sweeps");
    (definition.sweep || []).forEach(function (item) { addSweepRow(sweepHost, item); });
    if (!(definition.sweep || []).length) addSweepRow(sweepHost);
    card.querySelector("[data-add-stage-sweep]").addEventListener("click", function () {
      addSweepRow(sweepHost);
    });
    card.querySelector("[data-remove-stage]").addEventListener("click", function () {
      card.remove(); refreshPreview();
    });
    card.querySelector("[data-stage-name]").addEventListener("input", refreshPreview);
    card.querySelector("[data-stage-metric]").addEventListener("change", refreshPreview);
    card.querySelector("[data-stage-mode]").addEventListener("change", refreshPreview);
    refreshPreview();
  }

  function readStages() {
    return Array.prototype.map.call(node("localizationStages").querySelectorAll(".localization-stage-card"), function (card) {
      var selector = defaultSelector();
      selector.metric = card.querySelector("[data-stage-metric]").value;
      selector.mode = card.querySelector("[data-stage-mode]").value;
      return {
        name: card.querySelector("[data-stage-name]").value.trim(),
        sweep: readSweep(card.querySelector(".localization-stage-sweeps")),
        select: selector
      };
    });
  }

  function currentSpec() {
    var useStages = node("localizationUseStages").checked;
    return {
      schema_version: 1,
      name: node("localizationExperimentName").value.trim(),
      base: baseFromForm(),
      sweep: useStages ? [] : readSweep(node("localizationSweepRows")),
      stages: useStages ? readStages() : [],
      repeats: Math.round(n("localizationRepeats"))
    };
  }

  function productCount(sweep) {
    if (!sweep.length) return 1;
    return sweep.reduce(function (total, item) { return total * Math.max(1, (item.values || []).length); }, 1);
  }

  function estimate(spec) {
    var configurations = spec.stages && spec.stages.length
      ? spec.stages.reduce(function (total, stage) { return total + productCount(stage.sweep || []); }, 0)
      : productCount(spec.sweep || []);
    return { configurations: configurations, trainingRuns: configurations * Number(spec.repeats || 1) };
  }

  function updateConditionalFields() {
    var failureOnly = node("localizationPopulation").value === "failure_only";
    node("localizationSuccessRatio").disabled = failureOnly;
    if (failureOnly) node("localizationSuccessRatio").value = 0;
    var gaussian = node("localizationTargetKind").value === "gaussian";
    node("localizationSigmaPre").disabled = !gaussian;
    node("localizationSigmaPost").disabled = !gaussian;
  }

  function refreshPreview() {
    try {
      var spec = currentSpec();
      var work = estimate(spec);
      node("localizationSpecPreview").textContent = JSON.stringify(spec, null, 2);
      node("localizationEstimate").textContent =
        work.configurations + " configuration(s) × " + spec.repeats + " repeat(s) = " + work.trainingRuns + " training run(s)";
      var warning = "";
      if (work.trainingRuns > 500) warning = "Large experiment: more than 500 training runs.";
      else if (work.trainingRuns > 100) warning = "Large experiment: more than 100 training runs.";
      node("localizationEstimateWarning").textContent = warning;
      node("localizationEstimateWarning").className = "analysis-run-field-help" + (warning ? " warning" : "");
    } catch (error) {
      node("localizationSpecPreview").textContent = "Invalid spec: " + error.message;
      node("localizationEstimate").textContent = "Invalid experiment";
      node("localizationEstimateWarning").textContent = error.message;
    }
  }

  function applyPreset(spec) {
    if (!spec) return;
    node("localizationExperimentName").value = spec.name || "localization_experiment";
    node("localizationRepeats").value = spec.repeats == null ? 5 : spec.repeats;
    setBase(spec.base || {});
    node("localizationSweepRows").innerHTML = "";
    (spec.sweep || []).forEach(function (item) { addSweepRow(node("localizationSweepRows"), item); });
    node("localizationStages").innerHTML = "";
    var useStages = Array.isArray(spec.stages) && spec.stages.length > 0;
    node("localizationUseStages").checked = useStages;
    node("localizationStages").classList.toggle("hidden", !useStages);
    node("localizationAddStage").classList.toggle("hidden", !useStages);
    node("localizationSweepSection").classList.toggle("hidden", useStages);
    if (useStages) spec.stages.forEach(addStage);
    refreshPreview();
  }

  async function fetchJson(url, options) {
    var response = await fetch(url, options || { cache: "no-store" });
    var text = await response.text();
    var payload;
    try { payload = text ? JSON.parse(text) : {}; }
    catch (_error) { throw new Error("HTTP " + response.status + " returned non-JSON"); }
    if (!response.ok) throw new Error(payload.error || ("HTTP " + response.status));
    return payload;
  }

  async function loadPresets() {
    var payload = await fetchJson("/api/analysis/localization/presets", { cache: "no-store" });
    state.presets = payload.presets || [];
    var select = node("localizationPresetSelect");
    select.innerHTML = state.presets.map(function (preset) {
      return '<option value="' + esc(preset.name) + '">' + esc(preset.name)
        + (preset.builtin ? " · built-in" : "") + '</option>';
    }).join("");
    renderPresetList();
    if (state.presets.length && !select.value) select.value = state.presets[0].name;
  }

  function selectedPreset() {
    var name = node("localizationPresetSelect").value;
    return state.presets.find(function (preset) { return preset.name === name; }) || null;
  }

  function renderPresetList() {
    var host = node("localizationPresetList");
    if (!state.presets.length) {
      host.innerHTML = '<p class="analysis-empty">No presets.</p>';
      return;
    }
    host.innerHTML = '<table class="analysis-table"><thead><tr><th>Name</th><th>Type</th><th>Repeats</th><th>Actions</th></tr></thead><tbody>'
      + state.presets.map(function (preset) {
        return '<tr><td><strong>' + esc(preset.name) + '</strong></td>'
          + '<td>' + (preset.builtin ? "Built-in" : "Project") + '</td>'
          + '<td class="numeric">' + esc(preset.repeats || 1) + '</td>'
          + '<td><button class="ghost-button" type="button" data-load-preset="' + esc(preset.name) + '">Load</button> '
          + (preset.builtin ? "" : '<button class="ghost-button" type="button" data-delete-preset="' + esc(preset.name) + '">Delete</button>')
          + '</td></tr>';
      }).join("") + '</tbody></table>';
  }

  async function savePreset() {
    var spec = currentSpec();
    var requested = node("localizationPresetName").value.trim() || spec.name;
    if (!requested) throw new Error("Preset name is required");
    spec.name = requested;
    var existing = state.presets.find(function (preset) { return preset.name === requested; });
    var overwrite = Boolean(existing && !existing.builtin);
    if (existing && existing.builtin) throw new Error("Built-in presets cannot be overwritten");
    if (overwrite && !window.confirm("Overwrite preset " + requested + "?")) return;
    await fetchJson("/api/analysis/localization/presets/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec: spec, overwrite: overwrite })
    });
    await loadPresets();
    node("localizationPresetSelect").value = requested;
    node("localizationBuilderStatus").textContent = "Preset saved: " + requested;
  }

  async function deletePreset(name) {
    if (!window.confirm("Delete preset " + name + "?")) return;
    await fetchJson("/api/analysis/localization/presets/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name })
    });
    await loadPresets();
  }

  function setLabTab(tab) {
    state.activeTab = tab;
    document.querySelectorAll("[data-localization-tab]").forEach(function (button) {
      button.classList.toggle("active", button.dataset.localizationTab === tab);
    });
    document.querySelectorAll("[data-localization-panel]").forEach(function (panel) {
      panel.classList.toggle("hidden", panel.dataset.localizationPanel !== tab);
    });
  }

  async function runExperiment() {
    var spec = currentSpec();
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(spec.name)) {
      throw new Error("Experiment name is invalid");
    }
    var work = estimate(spec);
    if (work.trainingRuns > 500 && !window.confirm("This experiment expands to " + work.trainingRuns + " training runs. Start it?")) return;
    node("localizationBuilderStatus").textContent = "Submitting experiment…";
    var payload = await fetchJson("/api/analysis/localization/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec: spec })
    });
    state.currentJob = payload.job;
    node("localizationJobBadge").textContent = state.currentJob.status || "queued";
    setLabTab("runs");
    pollJob(state.currentJob.job_id);
  }

  async function loadJobLog(jobId) {
    var payload = await fetchJson("/api/analysis-jobs/" + encodeURIComponent(jobId) + "/log?tail=300", { cache: "no-store" });
    node("localizationJobLog").textContent = payload.log ? payload.log.text : "";
  }

  async function pollJob(jobId) {
    if (state.polling) return;
    state.polling = true;
    try {
      var payload = await fetchJson("/api/analysis-jobs/" + encodeURIComponent(jobId), { cache: "no-store" });
      var job = payload.job;
      if (!job || job.analysis_kind !== "robo_localization_experiment") return;
      state.currentJob = job;
      node("localizationJobBadge").textContent = job.status || "unknown";
      await loadJobLog(jobId);
      if (job.status === "queued" || job.status === "running") {
        window.setTimeout(function () { state.polling = false; pollJob(jobId); }, 1500);
        return;
      }
      await loadRuns();
    } catch (error) {
      node("localizationJobBadge").textContent = "failed";
      node("localizationJobLog").textContent += "\n" + error.message;
    } finally {
      if (!state.currentJob || ["queued", "running"].indexOf(state.currentJob.status) === -1) state.polling = false;
    }
  }

  async function loadRuns() {
    var payload = await fetchJson("/api/analysis/localization", { cache: "no-store" });
    var runs = payload.localization && payload.localization.runs || [];
    var host = node("localizationRunsList");
    if (!runs.length) {
      host.innerHTML = '<p class="analysis-empty">No completed Localization Lab experiments yet.</p>';
      return;
    }
    host.innerHTML = '<table class="analysis-table"><thead><tr><th>Experiment</th><th>Generated</th><th>Configs</th><th>Train runs</th><th>Artifacts</th></tr></thead><tbody>'
      + runs.map(function (run) {
        return '<tr><td><strong>' + esc(run.name || run.directory) + '</strong><br><small>' + esc(run.directory) + '</small></td>'
          + '<td>' + esc(run.generated_at || "") + '</td>'
          + '<td class="numeric">' + esc(run.configuration_count) + '</td>'
          + '<td class="numeric">' + esc(run.training_run_count) + '</td>'
          + '<td><a href="' + esc(run.summary_url) + '" download>summary.csv</a> · '
          + '<a href="' + esc(run.manifest_url) + '" download>manifest</a></td></tr>';
      }).join("") + '</tbody></table>';
  }

  async function recoverJob() {
    try {
      var payload = await fetchJson("/api/jobs?job_type=analysis", { cache: "no-store" });
      var jobs = (payload.jobs || []).filter(function (job) {
        return job.analysis_kind === "robo_localization_experiment";
      });
      if (!jobs.length) return;
      state.currentJob = jobs[0];
      node("localizationJobBadge").textContent = state.currentJob.status || "unknown";
      if (state.currentJob.status === "queued" || state.currentJob.status === "running") pollJob(state.currentJob.job_id);
      else loadJobLog(state.currentJob.job_id);
    } catch (_error) {}
  }

  async function refresh() {
    if (!node("analysisTabLocalization")) return;
    try {
      await Promise.all([loadPresets(), loadRuns()]);
      refreshPreview();
    } catch (error) {
      node("localizationBuilderStatus").textContent = "Localization Lab error: " + error.message;
      node("localizationBuilderStatus").className = "analysis-status error";
    }
  }

  function init() {
    if (!node("analysisTabLocalization")) return;
    document.querySelectorAll("[data-localization-tab]").forEach(function (button) {
      button.addEventListener("click", function () { setLabTab(button.dataset.localizationTab); });
    });
    node("localizationPresetSelect").addEventListener("change", function () {
      var preset = selectedPreset();
      if (preset) applyPreset(clone(preset));
    });
    node("localizationAddSweep").addEventListener("click", function () {
      addSweepRow(node("localizationSweepRows"));
    });
    node("localizationUseStages").addEventListener("change", function () {
      var enabled = node("localizationUseStages").checked;
      node("localizationStages").classList.toggle("hidden", !enabled);
      node("localizationAddStage").classList.toggle("hidden", !enabled);
      node("localizationSweepSection").classList.toggle("hidden", enabled);
      if (enabled && !node("localizationStages").children.length) addStage();
      refreshPreview();
    });
    node("localizationAddStage").addEventListener("click", function () { addStage(); });
    node("localizationRunExperiment").addEventListener("click", function () {
      runExperiment().catch(function (error) {
        node("localizationBuilderStatus").textContent = error.message;
        node("localizationBuilderStatus").className = "analysis-status error";
      });
    });
    node("localizationSavePreset").addEventListener("click", function () {
      savePreset().catch(function (error) {
        node("localizationBuilderStatus").textContent = error.message;
      });
    });
    node("localizationSavePresetFromBuilder").addEventListener("click", function () {
      node("localizationPresetName").value = node("localizationExperimentName").value;
      setLabTab("presets");
    });
    node("localizationRefreshRuns").addEventListener("click", function () {
      loadRuns().catch(function (error) { node("localizationJobLog").textContent = error.message; });
    });
    node("localizationPresetList").addEventListener("click", function (event) {
      var load = event.target.closest("[data-load-preset]");
      var del = event.target.closest("[data-delete-preset]");
      if (load) {
        var preset = state.presets.find(function (item) { return item.name === load.dataset.loadPreset; });
        if (preset) {
          applyPreset(clone(preset));
          node("localizationPresetSelect").value = preset.name;
          setLabTab("builder");
        }
      }
      if (del) deletePreset(del.dataset.deletePreset).catch(function (error) {
        node("localizationBuilderStatus").textContent = error.message;
      });
    });
    document.querySelectorAll("#analysisTabLocalization input, #analysisTabLocalization select").forEach(function (input) {
      input.addEventListener("input", function () { updateConditionalFields(); refreshPreview(); });
      input.addEventListener("change", function () { updateConditionalFields(); refreshPreview(); });
    });
    addSweepRow(node("localizationSweepRows"), { path: "target.kind", values: ["hard", "gaussian"] });
    window.localizationLabRefresh = refresh;
    refresh();
    recoverJob();
  }

  init();
})();