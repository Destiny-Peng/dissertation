"use strict";

(function installLocalizationLab() {
  var state = {
    presets: [],
    challengeSets: [],
    runs: [],
    challengeData: null,
    activeRunResult: null,
    selectedChallengeRollouts: new Set(),
    currentJob: null,
    pollingToken: 0,
    activeTab: "builder"
  };

  var PARAMS = [
    {
      path: "data.success_ratio", label: "Success ratio", type: "number", min: 0,
      defaults: [0, 0.5, 1, 2],
      help: "Numbers >= 0. Used only with population=failure_success. Example: [0,0.5,1,2]"
    },
    {
      path: "data.population", label: "Training population", type: "enum",
      allowed: ["failure_only", "failure_success"],
      defaults: ["failure_only", "failure_success"],
      help: "Allowed: failure_only, failure_success"
    },
    {
      path: "target.kind", label: "Target kind", type: "enum",
      allowed: ["hard", "gaussian"], defaults: ["hard", "gaussian"],
      help: "Allowed: hard, gaussian"
    },
    {
      path: "target.sigma_pre", label: "Gaussian σ pre", type: "number", minExclusive: 0,
      defaults: [1, 2, 3, 5], help: "Positive numbers only. Example: [1,2,3,5]"
    },
    {
      path: "target.sigma_post", label: "Gaussian σ post", type: "number", minExclusive: 0,
      defaults: [1, 2, 3, 5], help: "Positive numbers only. Example: [1,2,3,5]"
    },
    {
      path: "target.tau_event", label: "Event decay τ", type: "number", minExclusive: 0,
      defaults: [10, 20, 40], help: "Positive native-sample decay constants. Example: [10,20,40]"
    },
    {
      path: "model.hidden", label: "Hidden size", type: "integer", min: 1, max: 512,
      defaults: [16, 32], help: "Integer 1-512. Example: [16,32]"
    },
    {
      path: "loss.name", label: "Loss", type: "enum",
      allowed: [
        "bce",
        "temporal_softmax_ce",
        "temporal_softmax_ce_distance",
        "temporal_softmax_ce_squared_distance",
        "temporal_softmax_ce_ranking",
        "temporal_softmax_ce_distance_ranking"
      ],
      defaults: [
        "bce",
        "temporal_softmax_ce",
        "temporal_softmax_ce_distance",
        "temporal_softmax_ce_squared_distance",
        "temporal_softmax_ce_ranking",
        "temporal_softmax_ce_distance_ranking"
      ],
      help: "Allowed: bce, temporal_softmax_ce, temporal_softmax_ce_distance, temporal_softmax_ce_squared_distance, temporal_softmax_ce_ranking, temporal_softmax_ce_distance_ranking"
    },
    {
      path: "loss.distance_weight", label: "Distance weight", type: "number", min: 0,
      defaults: [0.5, 1, 2], help: "Numbers >= 0. Example: [0.5,1,2]"
    },
    {
      path: "loss.ranking_weight", label: "Ranking weight", type: "number", min: 0,
      defaults: [0.5, 1, 2], help: "Numbers >= 0. Example: [0.5,1,2]"
    },
    {
      path: "loss.ranking_margin", label: "Ranking margin", type: "number", min: 0,
      defaults: [0.5, 1, 2], help: "Numbers >= 0. Example: [0.5,1,2]"
    },
    {
      path: "training.batch_size", label: "Batch size", type: "integer", min: 1, max: 128,
      defaults: [16, 32, 64], help: "Integer 1-128. Example: [16,32,64]"
    },
    {
      path: "training.learning_rate", label: "Learning rate", type: "number", minExclusive: 0,
      defaults: [0.001, 0.003, 0.01], help: "Positive numbers. Example: [0.001,0.003,0.01]"
    },
    {
      path: "training.weight_decay", label: "Weight decay", type: "number", min: 0,
      defaults: [0, 0.0001, 0.001], help: "Numbers >= 0. Example: [0,0.0001,0.001]"
    },
    {
      path: "training.grad_clip", label: "Gradient clip", type: "number", minExclusive: 0,
      defaults: [1, 5, 10], help: "Positive numbers. Example: [1,5,10]"
    }
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
    var challengeName = node("localizationChallengeTrainSet").value;
    var challengeSet = state.challengeSets.find(function (item) { return item.name === challengeName; }) || null;
    var forceChallenge = node("localizationForceChallengeTrain").checked && challengeSet;
    return {
      data: {
        population: population,
        success_ratio: population === "failure_only" ? 0 : n("localizationSuccessRatio"),
        challenge_set_name: forceChallenge ? challengeName : "",
        force_train_rollout_ids: forceChallenge ? clone(challengeSet.rollout_ids || []) : []
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
        parallel_workers: Math.round(n("localizationParallelWorkers")),
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
    node("localizationChallengeTrainSet").value = data.challenge_set_name || "";
    node("localizationForceChallengeTrain").checked = Array.isArray(data.force_train_rollout_ids) && data.force_train_rollout_ids.length > 0;
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
    node("localizationParallelWorkers").value = training.parallel_workers == null ? 4 : training.parallel_workers;
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

  function paramDefinition(path) {
    return PARAMS.find(function (item) { return item.path === path; }) || null;
  }

  function paramOptions(selected) {
    return PARAMS.map(function (item) {
      return '<option value="' + esc(item.path) + '"' + (item.path === selected ? " selected" : "") + ">"
        + esc(item.label) + "</option>";
    }).join("");
  }

  function validateSweepValues(path, values) {
    var definition = paramDefinition(path);
    if (!definition) throw new Error("Unsupported sweep parameter: " + path);
    if (!Array.isArray(values) || !values.length) {
      throw new Error(definition.label + ": at least one value is required");
    }
    values.forEach(function (value) {
      if (definition.type === "enum") {
        if (definition.allowed.indexOf(value) === -1) {
          throw new Error(
            definition.label + ': invalid value "' + value + '". Allowed: '
            + definition.allowed.join(", ")
          );
        }
        return;
      }
      var number = Number(value);
      if (!Number.isFinite(number)) {
        throw new Error(definition.label + ": values must be numeric");
      }
      if (definition.type === "integer" && !Number.isInteger(number)) {
        throw new Error(definition.label + ": values must be integers");
      }
      if (definition.min != null && number < definition.min) {
        throw new Error(definition.label + ": values must be >= " + definition.min);
      }
      if (definition.minExclusive != null && number <= definition.minExclusive) {
        throw new Error(definition.label + ": values must be > " + definition.minExclusive);
      }
      if (definition.max != null && number > definition.max) {
        throw new Error(definition.label + ": values must be <= " + definition.max);
      }
    });
    return values;
  }

  function updateSweepRowHelp(row, resetValues) {
    var pathSelect = row.querySelector("[data-sweep-path]");
    var valuesInput = row.querySelector("[data-sweep-values]");
    var help = row.querySelector("[data-sweep-help]");
    var definition = paramDefinition(pathSelect.value);
    if (!definition) return;
    help.textContent = definition.help;
    valuesInput.placeholder = formatValues(definition.defaults);
    if (resetValues) valuesInput.value = formatValues(definition.defaults);
  }

  function addSweepRow(host, definition) {
    definition = definition || { path: "target.kind", values: ["hard", "gaussian"] };
    var row = document.createElement("div");
    row.className = "localization-sweep-row";
    row.innerHTML =
      '<label class="localization-control"><span>Parameter</span>'
      + '<select data-sweep-path>' + paramOptions(definition.path) + '</select></label>'
      + '<label class="localization-control"><span>Values</span>'
      + '<input data-sweep-values type="text" value="' + esc(formatValues(definition.values)) + '">'
      + '<small class="localization-sweep-help" data-sweep-help></small></label>'
      + '<button type="button" class="ghost-button localization-sweep-remove" data-remove-sweep>Remove</button>';
    updateSweepRowHelp(row, false);
    row.querySelector("[data-remove-sweep]").addEventListener("click", function () {
      row.remove();
      refreshPreview();
    });
    row.querySelector("[data-sweep-path]").addEventListener("change", function () {
      updateSweepRowHelp(row, true);
      refreshPreview();
    });
    row.querySelector("[data-sweep-values]").addEventListener("input", refreshPreview);
    row.querySelector("[data-sweep-values]").addEventListener("change", refreshPreview);
    host.appendChild(row);
    refreshPreview();
  }

  function readSweep(host) {
    return Array.prototype.map.call(host.querySelectorAll(".localization-sweep-row"), function (row) {
      var path = row.querySelector("[data-sweep-path]").value;
      var values = parseValues(row.querySelector("[data-sweep-values]").value);
      validateSweepValues(path, values);
      return { path: path, values: values };
    });
  }

  function updateTargetVariantState(row) {
    var kind = row.querySelector("[data-variant-kind]").value;
    var disabled = kind === "hard";
    row.querySelector("[data-variant-sigma-pre]").disabled = disabled;
    row.querySelector("[data-variant-sigma-post]").disabled = disabled;
  }

  function addTargetVariant(host, definition) {
    definition = definition || {
      name: "gaussian_sigma_3",
      set: {
        "target.kind": "gaussian",
        "target.sigma_pre": 3,
        "target.sigma_post": 3,
        "target.tau_event": 20
      }
    };
    var assignments = definition.set || {};
    var kind = assignments["target.kind"] || "gaussian";
    var row = document.createElement("div");
    row.className = "localization-target-variant-row";
    row.innerHTML =
      '<label class="localization-control"><span>Name</span>'
      + '<input data-variant-name type="text" maxlength="64" value="' + esc(definition.name || "target_variant") + '"></label>'
      + '<label class="localization-control"><span>Target</span>'
      + '<select data-variant-kind><option value="hard">hard</option><option value="gaussian">gaussian</option></select></label>'
      + '<label class="localization-control"><span>σ pre</span>'
      + '<input data-variant-sigma-pre type="number" min="0.01" step="0.5" value="' + esc(assignments["target.sigma_pre"] == null ? 3 : assignments["target.sigma_pre"]) + '"></label>'
      + '<label class="localization-control"><span>σ post</span>'
      + '<input data-variant-sigma-post type="number" min="0.01" step="0.5" value="' + esc(assignments["target.sigma_post"] == null ? 3 : assignments["target.sigma_post"]) + '"></label>'
      + '<label class="localization-control"><span>Event τ</span>'
      + '<input data-variant-tau type="number" min="0.01" step="1" value="' + esc(assignments["target.tau_event"] == null ? 20 : assignments["target.tau_event"]) + '"></label>'
      + '<button type="button" class="ghost-button localization-variant-remove" data-remove-variant>Remove</button>';
    row.querySelector("[data-variant-kind]").value = kind;
    row.querySelector("[data-remove-variant]").addEventListener("click", function () {
      row.remove();
      refreshPreview();
    });
    row.querySelector("[data-variant-kind]").addEventListener("change", function () {
      updateTargetVariantState(row);
      refreshPreview();
    });
    row.querySelectorAll("input").forEach(function (input) {
      input.addEventListener("input", refreshPreview);
      input.addEventListener("change", refreshPreview);
    });
    host.appendChild(row);
    updateTargetVariantState(row);
    refreshPreview();
  }

  function readTargetVariants(host) {
    return Array.prototype.map.call(host.querySelectorAll(".localization-target-variant-row"), function (row) {
      var name = row.querySelector("[data-variant-name]").value.trim();
      if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name)) {
        throw new Error("Target variant name is invalid: " + name);
      }
      var kind = row.querySelector("[data-variant-kind]").value;
      var sigmaPre = Number(row.querySelector("[data-variant-sigma-pre]").value);
      var sigmaPost = Number(row.querySelector("[data-variant-sigma-post]").value);
      var tau = Number(row.querySelector("[data-variant-tau]").value);
      if (!(tau > 0)) throw new Error(name + ": Event τ must be > 0");
      if (kind === "gaussian" && (!(sigmaPre > 0) || !(sigmaPost > 0))) {
        throw new Error(name + ": Gaussian σ pre/post must be > 0");
      }
      return {
        name: name,
        set: {
          "target.kind": kind,
          "target.sigma_pre": Number.isFinite(sigmaPre) ? sigmaPre : 3,
          "target.sigma_post": Number.isFinite(sigmaPost) ? sigmaPost : 3,
          "target.tau_event": tau
        }
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
      + '<div class="localization-stage-subsection">'
      + '<h5 class="localization-subsection-title">Independent sweep dimensions</h5>'
      + '<div class="localization-stage-sweeps"></div>'
      + '<button type="button" class="ghost-button" data-add-stage-sweep>Add sweep dimension</button>'
      + '</div>'
      + '<div class="localization-stage-subsection">'
      + '<h5 class="localization-subsection-title">Coupled target variants</h5>'
      + '<div class="localization-stage-variants localization-target-variant-list"></div>'
      + '<button type="button" class="ghost-button" data-add-stage-variant>Add target variant</button>'
      + '</div>';
    card.querySelector("[data-stage-metric]").value = selector.metric || "in_interval_rate_mean";
    card.querySelector("[data-stage-mode]").value = selector.mode || "max";
    node("localizationStages").appendChild(card);
    var sweepHost = card.querySelector(".localization-stage-sweeps");
    var variantHost = card.querySelector(".localization-stage-variants");
    (definition.sweep || []).forEach(function (item) { addSweepRow(sweepHost, item); });
    (definition.variants || []).forEach(function (item) { addTargetVariant(variantHost, item); });
    if (!(definition.sweep || []).length && !(definition.variants || []).length) addSweepRow(sweepHost);
    card.querySelector("[data-add-stage-sweep]").addEventListener("click", function () {
      addSweepRow(sweepHost);
    });
    card.querySelector("[data-add-stage-variant]").addEventListener("click", function () {
      addTargetVariant(variantHost);
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
        variants: readTargetVariants(card.querySelector(".localization-stage-variants")),
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
      variants: useStages ? [] : readTargetVariants(node("localizationTargetVariants")),
      stages: useStages ? readStages() : [],
      repeats: Math.round(n("localizationRepeats"))
    };
  }

  function productCount(sweep) {
    if (!sweep.length) return 1;
    return sweep.reduce(function (total, item) { return total * Math.max(1, (item.values || []).length); }, 1);
  }

  function configurationCount(sweep, variants) {
    return productCount(sweep || []) * Math.max(1, (variants || []).length);
  }

  function estimate(spec) {
    var configurations = spec.stages && spec.stages.length
      ? spec.stages.reduce(function (total, stage) {
          return total + configurationCount(stage.sweep || [], stage.variants || []);
        }, 0)
      : configurationCount(spec.sweep || [], spec.variants || []);
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
      var workers = Math.max(1, Number(spec.base.training.parallel_workers || 1));
      node("localizationEstimate").textContent =
        work.configurations + " configuration(s) × " + spec.repeats + " repeat(s) = "
        + work.trainingRuns + " training run(s) · up to " + workers + " config worker(s)";
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
    node("localizationTargetVariants").innerHTML = "";
    (spec.sweep || []).forEach(function (item) { addSweepRow(node("localizationSweepRows"), item); });
    (spec.variants || []).forEach(function (item) { addTargetVariant(node("localizationTargetVariants"), item); });
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

  async function loadChallengeSets() {
    var payload = await fetchJson("/api/analysis/localization/challenge-sets", { cache: "no-store" });
    state.challengeSets = payload.challenge_sets || [];
    var select = node("localizationChallengeTrainSet");
    var previous = select.value;
    select.innerHTML = '<option value="">None</option>' + state.challengeSets.map(function (item) {
      return '<option value="' + esc(item.name) + '">' + esc(item.name)
        + ' · ' + esc(item.size || 0) + ' rollout(s)</option>';
    }).join("");
    if (state.challengeSets.some(function (item) { return item.name === previous; })) {
      select.value = previous;
    }
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
    if (tab === "challenge" && !state.challengeData && node("localizationChallengeRun").value) {
      loadChallengeRun().catch(function (error) {
        node("localizationChallengeStatus").textContent = error.message;
        node("localizationChallengeStatus").className = "analysis-status error";
      });
    }
  }

  function selectedChallengeConfigIds() {
    return Array.prototype.map.call(
      node("localizationChallengeConfigs").querySelectorAll("input[data-challenge-config]:checked"),
      function (input) { return input.value; }
    );
  }

  function challengeRowsForSelection() {
    if (!state.challengeData || !state.challengeData.available) return [];
    var configIds = selectedChallengeConfigIds();
    if (!configIds.length) return [];
    var criterion = node("localizationChallengeCriterion").value;
    var byRollout = {};
    state.challengeData.rows.forEach(function (row) {
      if (configIds.indexOf(row.config_id) === -1) return;
      if (!byRollout[row.rollout_id]) byRollout[row.rollout_id] = {};
      byRollout[row.rollout_id][row.config_id] = row;
    });
    var result = [];
    Object.keys(byRollout).forEach(function (rolloutId) {
      var configRows = configIds.map(function (configId) { return byRollout[rolloutId][configId]; });
      if (configRows.some(function (row) { return !row; })) return;
      var first = configRows[0];
      function avg(field) {
        return configRows.reduce(function (total, row) { return total + Number(row[field] || 0); }, 0) / configRows.length;
      }
      var persistent = configRows.every(function (row) {
        return criterion === "outside_3"
          ? row.all_repeats_failed_within_3
          : row.all_repeats_failed_interval;
      });
      result.push({
        rollout_id: rolloutId,
        config_rows: configRows,
        persistent: persistent,
        repeat_count: Math.min.apply(null, configRows.map(function (row) { return Number(row.repeat_count || 0); })),
        in_interval_success_rate: avg("in_interval_success_rate"),
        within_3_success_rate: avg("within_3_success_rate"),
        median_absolute_interval_error: avg("median_absolute_interval_error"),
        mean_absolute_interval_error: avg("mean_absolute_interval_error"),
        worst_absolute_interval_error: Math.max.apply(null, configRows.map(function (row) {
          return Number(row.worst_absolute_interval_error || 0);
        })),
        train_exposure_rate: avg("train_exposure_rate"),
        forced_train_rate: avg("forced_train_rate"),
        train_in_interval_success_rate: (function () {
          var values = configRows.map(function (row) { return row.train_in_interval_success_rate; })
            .filter(function (value) { return value != null; }).map(Number);
          return values.length ? values.reduce(function (a, b) { return a + b; }, 0) / values.length : null;
        })(),
        task_id: first.task_id,
        task_key: first.task_key,
        primary_failure_type: first.primary_failure_type,
        failure_types: first.failure_types || [],
        multi_event: Boolean(first.multi_event),
        recovery_like: Boolean(first.recovery_like)
      });
    });
    var taskFilter = node("localizationChallengeTask").value;
    var failureFilter = node("localizationChallengeFailureType").value;
    var flagFilter = node("localizationChallengeCaseFlag").value;
    result = result.filter(function (row) {
      if (taskFilter && String(row.task_id) !== taskFilter) return false;
      if (failureFilter && String(row.primary_failure_type || "unknown") !== failureFilter) return false;
      if (flagFilter === "multi_event" && !row.multi_event) return false;
      if (flagFilter === "recovery_like" && !row.recovery_like) return false;
      return true;
    });
    result.sort(function (a, b) {
      return a.within_3_success_rate - b.within_3_success_rate
        || a.in_interval_success_rate - b.in_interval_success_rate
        || b.median_absolute_interval_error - a.median_absolute_interval_error
        || b.mean_absolute_interval_error - a.mean_absolute_interval_error
        || b.worst_absolute_interval_error - a.worst_absolute_interval_error
        || a.rollout_id.localeCompare(b.rollout_id);
    });
    return result;
  }

  function formatRate(value) {
    if (value == null || !Number.isFinite(Number(value))) return "—";
    return (100 * Number(value)).toFixed(0) + "%";
  }

  function renderChallengeComposition(rows) {
    var selected = rows.filter(function (row) {
      return state.selectedChallengeRollouts.has(row.rollout_id);
    });
    var host = node("localizationChallengeComposition");
    if (!selected.length) {
      host.innerHTML = '<p class="analysis-card-note">No challenge rollouts selected.</p>';
      return;
    }
    var tasks = {}, failures = {}, multi = 0, recovery = 0, persistent = 0;
    selected.forEach(function (row) {
      var task = String(row.task_id);
      tasks[task] = (tasks[task] || 0) + 1;
      var failure = row.primary_failure_type || "unknown";
      failures[failure] = (failures[failure] || 0) + 1;
      if (row.multi_event) multi += 1;
      if (row.recovery_like) recovery += 1;
      if (row.persistent) persistent += 1;
    });
    function pairs(value) {
      return Object.keys(value).sort().map(function (key) {
        return esc(key) + ": " + esc(value[key]);
      }).join(" · ");
    }
    host.innerHTML =
      '<div class="localization-challenge-summary">'
      + '<strong>' + selected.length + ' selected</strong>'
      + '<span>Persistent: ' + persistent + '</span>'
      + '<span>Multi-event: ' + multi + '</span>'
      + '<span>Recovery-like: ' + recovery + '</span>'
      + '</div>'
      + '<p><strong>Tasks</strong> ' + pairs(tasks) + '</p>'
      + '<p><strong>Failure types</strong> ' + pairs(failures) + '</p>';
  }

  function renderChallengeTable() {
    var rows = challengeRowsForSelection();
    var persistentOnly = node("localizationChallengePersistentOnly").checked;
    var visible = persistentOnly ? rows.filter(function (row) { return row.persistent; }) : rows;
    node("localizationChallengeStatus").className = "analysis-status";
    node("localizationChallengeStatus").textContent =
      visible.length + " rollout(s) shown · " + rows.filter(function (row) { return row.persistent; }).length
      + " common persistent failure(s) · full-repeat inference";
    renderChallengeComposition(rows);
    var host = node("localizationChallengeTable");
    if (!visible.length) {
      host.innerHTML = '<p class="analysis-empty">No rollouts match the current config/criterion selection.</p>';
      return;
    }
    host.innerHTML = '<table class="analysis-table localization-challenge-table"><thead><tr>'
      + '<th>Use</th><th>Rank</th><th>Rollout</th><th>Repeats</th><th>In interval</th><th>±3</th>'
      + '<th>Median |err|</th><th>Mean |err|</th><th>Worst |err|</th><th>Train exposure</th>'
      + '<th>Forced train</th><th>Train in-interval</th><th>Task</th><th>Failure</th><th>Flags</th>'
      + '</tr></thead><tbody>'
      + visible.map(function (row, index) {
        var flags = [];
        if (row.persistent) flags.push("persistent");
        if (row.multi_event) flags.push("multi-event");
        if (row.recovery_like) flags.push("recovery");
        return '<tr>'
          + '<td><input type="checkbox" data-challenge-rollout="' + esc(row.rollout_id) + '"'
          + (state.selectedChallengeRollouts.has(row.rollout_id) ? " checked" : "") + '></td>'
          + '<td class="numeric">' + (index + 1) + '</td>'
          + '<td><a href="#/review/' + encodeURIComponent(row.rollout_id) + '">' + esc(row.rollout_id) + '</a></td>'
          + '<td class="numeric">' + esc(row.repeat_count) + '</td>'
          + '<td class="numeric">' + formatRate(row.in_interval_success_rate) + '</td>'
          + '<td class="numeric">' + formatRate(row.within_3_success_rate) + '</td>'
          + '<td class="numeric">' + Number(row.median_absolute_interval_error).toFixed(1) + '</td>'
          + '<td class="numeric">' + Number(row.mean_absolute_interval_error).toFixed(1) + '</td>'
          + '<td class="numeric">' + Number(row.worst_absolute_interval_error).toFixed(0) + '</td>'
          + '<td class="numeric">' + formatRate(row.train_exposure_rate) + '</td>'
          + '<td class="numeric">' + formatRate(row.forced_train_rate) + '</td>'
          + '<td class="numeric">' + formatRate(row.train_in_interval_success_rate) + '</td>'
          + '<td>' + esc(row.task_id) + '</td>'
          + '<td>' + esc(row.primary_failure_type || "unknown") + '</td>'
          + '<td>' + esc(flags.join(", ")) + '</td>'
          + '</tr>';
      }).join("") + '</tbody></table>';
  }

  function proposeChallengeSet() {
    var rows = challengeRowsForSelection();
    var requested = Math.max(1, Math.round(n("localizationChallengeSize")));
    var persistent = rows.filter(function (row) { return row.persistent; });
    var seedN = Math.min(persistent.length, Math.max(1, Math.floor(requested * 0.75)));
    var selected = persistent.slice(0, seedN);
    var selectedIds = new Set(selected.map(function (row) { return row.rollout_id; }));
    var pool = rows.filter(function (row) { return !selectedIds.has(row.rollout_id); });
    while (selected.length < requested && pool.length) {
      var seenTasks = new Set(selected.map(function (row) { return String(row.task_id); }));
      var seenFailures = new Set(selected.map(function (row) { return row.primary_failure_type || "unknown"; }));
      var hasMulti = selected.some(function (row) { return row.multi_event; });
      var hasRecovery = selected.some(function (row) { return row.recovery_like; });
      var bestIndex = 0, bestScore = -1;
      pool.forEach(function (row, index) {
        var diversity = 0;
        if (!seenTasks.has(String(row.task_id))) diversity += 4;
        if (!seenFailures.has(row.primary_failure_type || "unknown")) diversity += 4;
        if (row.multi_event && !hasMulti) diversity += 2;
        if (row.recovery_like && !hasRecovery) diversity += 2;
        if (row.persistent) diversity += 3;
        var score = diversity * 1000 - index;
        if (score > bestScore) { bestScore = score; bestIndex = index; }
      });
      var picked = pool.splice(bestIndex, 1)[0];
      selected.push(picked);
      selectedIds.add(picked.rollout_id);
    }
    state.selectedChallengeRollouts = selectedIds;
    if (selected.some(function (row) { return !row.persistent; })) {
      node("localizationChallengePersistentOnly").checked = false;
    }
    renderChallengeTable();
  }

  function renderChallengeFilters() {
    var data = state.challengeData;
    var taskSelect = node("localizationChallengeTask");
    var failureSelect = node("localizationChallengeFailureType");
    if (!data || !data.available) {
      taskSelect.innerHTML = '<option value="">All tasks</option>';
      failureSelect.innerHTML = '<option value="">All failure types</option>';
      return;
    }
    var tasks = Array.from(new Set(data.rows.map(function (row) { return String(row.task_id); }))).sort();
    var failures = Array.from(new Set(data.rows.map(function (row) {
      return String(row.primary_failure_type || "unknown");
    }))).sort();
    taskSelect.innerHTML = '<option value="">All tasks</option>' + tasks.map(function (value) {
      return '<option value="' + esc(value) + '">' + esc(value) + '</option>';
    }).join("");
    failureSelect.innerHTML = '<option value="">All failure types</option>' + failures.map(function (value) {
      return '<option value="' + esc(value) + '">' + esc(value) + '</option>';
    }).join("");
    node("localizationChallengeCaseFlag").value = "";
  }

  function renderChallengeConfigs() {
    var host = node("localizationChallengeConfigs");
    var data = state.challengeData;
    if (!data || !data.available) {
      host.innerHTML = '<p class="analysis-empty">' + esc(data && data.reason || "Challenge data unavailable.") + '</p>';
      return;
    }
    var defaultId = data.default_config_id || (data.configs[0] && data.configs[0].config_id) || "";
    host.innerHTML = data.configs.map(function (config) {
      var checked = config.config_id === defaultId ? " checked" : "";
      return '<label class="localization-challenge-config">'
        + '<input type="checkbox" data-challenge-config value="' + esc(config.config_id) + '"' + checked + '>'
        + '<span>' + esc(config.label) + (config.best ? " · best" : "") + '</span></label>';
    }).join("");
  }

  async function loadChallengeRun() {
    var runId = node("localizationChallengeRun").value;
    if (!runId) {
      state.challengeData = null;
      node("localizationChallengeConfigs").innerHTML = "";
      node("localizationChallengeTable").innerHTML = '<p class="analysis-empty">No challenge-capable run available.</p>';
      return;
    }
    node("localizationChallengeStatus").textContent = "Loading challenge results…";
    var payload = await fetchJson(
      "/api/analysis/localization/challenge/" + encodeURIComponent(runId),
      { cache: "no-store" }
    );
    state.challengeData = payload.challenge || null;
    state.selectedChallengeRollouts = new Set();
    renderChallengeFilters();
    renderChallengeConfigs();
    renderChallengeTable();
  }

  async function saveChallengeSet() {
    if (!state.challengeData || !state.challengeData.available) throw new Error("Challenge data is unavailable");
    var name = node("localizationChallengeName").value.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name)) throw new Error("Challenge-set name is invalid");
    var configIds = selectedChallengeConfigIds();
    if (!configIds.length) throw new Error("Select at least one training result");
    var rolloutIds = Array.from(state.selectedChallengeRollouts);
    if (!rolloutIds.length) throw new Error("Select or propose at least one rollout");
    var existing = state.challengeSets.find(function (item) { return item.name === name; });
    var overwrite = Boolean(existing);
    if (overwrite && !window.confirm("Overwrite challenge set " + name + "?")) return;
    await fetchJson("/api/analysis/localization/challenge/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: name,
        source_run: state.challengeData.run_id,
        config_ids: configIds,
        criterion: node("localizationChallengeCriterion").value,
        rollout_ids: rolloutIds,
        overwrite: overwrite
      })
    });
    await loadChallengeSets();
    node("localizationChallengeTrainSet").value = name;
    node("localizationChallengeStatus").textContent =
      "Saved " + name + " with " + rolloutIds.length + " rollout(s). It is now available for forced training.";
  }


  function formatMetric(value, percent) {
    if (value == null || !Number.isFinite(Number(value))) return "—";
    var number = Number(value);
    return percent ? (100 * number).toFixed(1) + "%" : number.toFixed(3).replace(/\.000$/, "");
  }

  function formatMetricCell(row, meanField, varianceField, percent) {
    var main = formatMetric(row[meanField], percent);
    var variance = row[varianceField];
    if (variance == null || !Number.isFinite(Number(variance))) return main;
    return '<span class="localization-result-metric">' + main
      + '<small>var ' + formatMetric(variance, false) + '</small></span>';
  }

  function renderBestRepeatCell(best) {
    if (!best || best.repeat == null) return "—";
    var metrics = [];
    if (best.in_interval_rate != null && Number.isFinite(Number(best.in_interval_rate))) {
      metrics.push("In " + formatMetric(best.in_interval_rate, true));
    }
    if (best.mae_samples != null && Number.isFinite(Number(best.mae_samples))) {
      metrics.push("MAE " + formatMetric(best.mae_samples, false));
    }
    if (best.mse_samples != null && Number.isFinite(Number(best.mse_samples))) {
      metrics.push("MSE " + formatMetric(best.mse_samples, false));
    }
    var checkpoint = String(best.checkpoint || "");
    var checkpointLabel = checkpoint
      ? checkpoint.split("/").slice(-3).join("/")
      : "";
    var title = "Descriptive repeat ranking: maximize test in-interval rate, then minimize test MAE and MSE.";
    if (checkpoint) title += " Checkpoint: " + checkpoint;
    return '<div class="localization-best-repeat" title="' + esc(title) + '">'
      + '<strong>repeat ' + esc(best.repeat) + '</strong>'
      + (metrics.length ? '<small>' + esc(metrics.join(" · ")) + '</small>' : "")
      + (checkpointLabel
          ? '<code title="' + esc(checkpoint) + '">' + esc(checkpointLabel) + '</code>'
          : "")
      + '</div>';
  }

  function renderRunResult(result) {
    state.activeRunResult = result;
    var shell = node("localizationRunResult");
    var host = node("localizationRunResultContent");
    if (!result) {
      shell.classList.add("hidden");
      host.innerHTML = "";
      return;
    }
    shell.classList.remove("hidden");
    var header =
      '<div class="analysis-card-heading localization-run-result-heading">'
      + '<div><p class="eyebrow">RUN RESULT</p><h4>' + esc(result.name || result.run_id) + '</h4>'
      + '<small>' + esc(result.generated_at || "") + ' · '
      + esc(result.configuration_count || 0) + ' config(s) · '
      + esc(result.training_run_count || 0) + ' training run(s)'
      + (result.checkpoint_count != null ? ' · ' + esc(result.checkpoint_count) + ' checkpoint(s)' : "")
      + '</small></div>'
      + '<div class="localization-run-result-actions">'
      + (result.challenge_available
          ? '<button type="button" class="ghost-button" data-open-challenge-run="' + esc(result.run_id) + '">Open Challenge Set</button>'
          : '')
      + '</div></div>';

    var stages = (result.stages || []).map(function (stage) {
      var rows = stage.rows || [];
      var selector = stage.selector || {};
      var selectorText = selector.metric
        ? ('Select by ' + selector.metric + ' ' + (selector.mode || 'max'))
        : '';
      return '<section class="localization-run-stage">'
        + '<div class="localization-run-stage-heading"><div><h5>' + esc(stage.stage || "main") + '</h5>'
        + '<small>' + esc(rows.length) + ' config(s)'
        + (stage.parallel_workers != null ? ' · workers ' + esc(stage.parallel_workers) : '')
        + (selectorText ? ' · ' + esc(selectorText) : '')
        + '</small></div>'
        + (stage.best_config_id ? '<span class="analysis-badge">Best ' + esc(stage.best_config_id) + '</span>' : '')
        + '</div>'
        + '<div class="analysis-table-wrap"><table class="analysis-table localization-run-result-table"><thead><tr>'
        + '<th>Config</th><th>Repeats</th><th title="Descriptive ranking within this config: max test in-interval, then min test MAE/MSE">Best repeat</th>'
        + '<th>In interval</th><th>First event</th><th>±3</th>'
        + '<th>Median |err|</th><th>MAE</th><th>MSE</th><th>Before</th><th>After</th>'
        + '<th>Batch</th><th>Best config</th>'
        + '</tr></thead><tbody>'
        + rows.map(function (row) {
          return '<tr class="' + (row.best ? 'localization-best-row' : '') + '">'
            + '<td><strong>' + esc(row.label || row.config_id) + '</strong></td>'
            + '<td class="numeric">' + esc(row.repeat_n == null ? "—" : row.repeat_n) + '</td>'
            + '<td>' + renderBestRepeatCell(row.best_repeat) + '</td>'
            + '<td class="numeric">' + formatMetricCell(row, "in_interval_rate_mean", "in_interval_rate_variance", true) + '</td>'
            + '<td class="numeric">' + formatMetricCell(row, "first_event_in_interval_rate_mean", "first_event_in_interval_rate_variance", true) + '</td>'
            + '<td class="numeric">' + formatMetricCell(row, "within_3_mean", "within_3_variance", true) + '</td>'
            + '<td class="numeric">' + formatMetricCell(row, "median_absolute_interval_error_samples_mean", "median_absolute_interval_error_samples_variance", false) + '</td>'
            + '<td class="numeric">' + formatMetricCell(row, "mae_samples_mean", "mae_samples_variance", false) + '</td>'
            + '<td class="numeric">' + formatMetricCell(row, "mse_samples_mean", "mse_samples_variance", false) + '</td>'
            + '<td class="numeric">' + formatMetricCell(row, "before_interval_rate_mean", "before_interval_rate_variance", true) + '</td>'
            + '<td class="numeric">' + formatMetricCell(row, "after_interval_rate_mean", "after_interval_rate_variance", true) + '</td>'
            + '<td class="numeric">' + esc(row["training.batch_size"] == null ? "—" : row["training.batch_size"]) + '</td>'
            + '<td>' + (row.best ? '<strong>Selected</strong>' : '') + '</td>'
            + '</tr>';
        }).join("")
        + '</tbody></table></div></section>';
    }).join("");

    host.innerHTML = header + stages;
  }

  async function loadRunResult(runId) {
    if (!runId) return;
    node("localizationRunResult").classList.remove("hidden");
    node("localizationRunResultContent").innerHTML =
      '<p class="analysis-card-note">Loading run result…</p>';
    var payload = await fetchJson(
      "/api/analysis/localization/result/" + encodeURIComponent(runId),
      { cache: "no-store" }
    );
    renderRunResult(payload.result || null);
  }

  async function runExperiment() {
    var spec = currentSpec();
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(spec.name)) {
      throw new Error("Experiment name is invalid");
    }
    var work = estimate(spec);
    if (work.trainingRuns > 500 && !window.confirm("This experiment expands to " + work.trainingRuns + " training runs. Start it?")) return;
    node("localizationBuilderStatus").textContent = "Submitting experiment…";
    renderRunResult(null);
    var payload = await fetchJson("/api/analysis/localization/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec: spec })
    });
    state.currentJob = payload.job;
    node("localizationJobBadge").textContent = state.currentJob.status || "queued";
    setLabTab("runs");
    beginJobPolling(state.currentJob.job_id);
  }

  async function loadJobLog(jobId) {
    var payload = await fetchJson("/api/analysis-jobs/" + encodeURIComponent(jobId) + "/log?tail=300", { cache: "no-store" });
    node("localizationJobLog").textContent = payload.log ? payload.log.text : "";
  }

  function beginJobPolling(jobId) {
    state.pollingToken += 1;
    var token = state.pollingToken;
    pollJob(jobId, token);
  }

  async function pollJob(jobId, token) {
    if (token !== state.pollingToken) return;
    try {
      var payload = await fetchJson(
        "/api/analysis-jobs/" + encodeURIComponent(jobId),
        { cache: "no-store" }
      );
      if (token !== state.pollingToken) return;
      var job = payload.job;
      if (!job || job.analysis_kind !== "robo_localization_experiment") return;
      state.currentJob = job;
      node("localizationJobBadge").textContent = job.status || "unknown";
      await loadJobLog(jobId);
      if (token !== state.pollingToken) return;
      if (job.status === "queued" || job.status === "running") {
        window.setTimeout(function () {
          pollJob(jobId, token);
        }, 1500);
        return;
      }
      await loadRuns();
      if (token !== state.pollingToken) return;
      if (job.status === "complete") {
        var runId = String(job.output_dir || "").split("/").filter(Boolean).pop();
        if (runId) {
          try { await loadRunResult(runId); } catch (_error) {}
        }
      }
    } catch (error) {
      if (token !== state.pollingToken) return;
      node("localizationJobBadge").textContent = "failed";
      node("localizationJobLog").textContent += "\n" + error.message;
    }
  }

  async function loadRuns() {
    var payload = await fetchJson("/api/analysis/localization", { cache: "no-store" });
    var runs = payload.localization && payload.localization.runs || [];
    state.runs = runs;
    var challengeSelect = node("localizationChallengeRun");
    var previousRun = challengeSelect.value;
    var challengeRuns = runs.filter(function (run) { return run.challenge_available; });
    challengeSelect.innerHTML = challengeRuns.map(function (run) {
      return '<option value="' + esc(run.run_id) + '">' + esc(run.name || run.run_id)
        + ' · ' + esc(run.generated_at || "") + '</option>';
    }).join("");
    if (challengeRuns.some(function (run) { return run.run_id === previousRun; })) {
      challengeSelect.value = previousRun;
    }
    var host = node("localizationRunsList");
    if (!runs.length) {
      host.innerHTML = '<p class="analysis-empty">No completed Localization Lab experiments yet.</p>';
      return;
    }
    host.innerHTML = '<table class="analysis-table"><thead><tr><th>Experiment</th><th>Generated</th><th>Configs</th><th>Train runs</th><th>Results</th><th>Artifacts</th></tr></thead><tbody>'
      + runs.map(function (run) {
        return '<tr><td><strong>' + esc(run.name || run.directory) + '</strong><br><small>' + esc(run.directory) + '</small></td>'
          + '<td>' + esc(run.generated_at || "") + '</td>'
          + '<td class="numeric">' + esc(run.configuration_count) + '</td>'
          + '<td class="numeric">' + esc(run.training_run_count) + '</td>'
          + '<td><button type="button" class="ghost-button" data-view-localization-result="' + esc(run.run_id) + '">View results</button></td>'
          + '<td><a href="' + esc(run.summary_url) + '" download>summary.csv</a> · '
          + '<a href="' + esc(run.manifest_url) + '" download>manifest</a>'
          + (run.all_failure_url ? ' · <a href="' + esc(run.all_failure_url) + '" download>all failures</a>' : "")
          + '</td></tr>';
      }).join("") + '</tbody></table>';
  }

  async function recoverJob() {
    var tokenAtStart = state.pollingToken;
    var jobAtStart = state.currentJob && state.currentJob.job_id;
    try {
      var payload = await fetchJson("/api/jobs?job_type=analysis", { cache: "no-store" });

      // A new foreground job may have been submitted while this recovery
      // request was in flight. Never let stale startup recovery steal polling
      // back from that newly submitted job.
      if (
        state.pollingToken !== tokenAtStart
        || (
          state.currentJob
          && state.currentJob.job_id
          && state.currentJob.job_id !== jobAtStart
        )
      ) {
        return;
      }

      var jobs = (payload.jobs || []).filter(function (job) {
        return job.analysis_kind === "robo_localization_experiment";
      });
      if (!jobs.length) return;
      state.currentJob = jobs[0];
      node("localizationJobBadge").textContent = state.currentJob.status || "unknown";
      if (state.currentJob.status === "queued" || state.currentJob.status === "running") {
        beginJobPolling(state.currentJob.job_id);
      } else {
        loadJobLog(state.currentJob.job_id);
      }
    } catch (_error) {}
  }

  async function refresh() {
    if (!node("analysisTabLocalization")) return;
    try {
      await Promise.all([loadPresets(), loadRuns(), loadChallengeSets()]);
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
    node("localizationAddTargetVariant").addEventListener("click", function () {
      addTargetVariant(node("localizationTargetVariants"));
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
    node("localizationRunsList").addEventListener("click", function (event) {
      var button = event.target.closest("[data-view-localization-result]");
      if (!button) return;
      loadRunResult(button.dataset.viewLocalizationResult).catch(function (error) {
        node("localizationRunResult").classList.remove("hidden");
        node("localizationRunResultContent").innerHTML =
          '<div class="analysis-status error">' + esc(error.message) + '</div>';
      });
    });
    node("localizationRunResult").addEventListener("click", function (event) {
      var button = event.target.closest("[data-open-challenge-run]");
      if (!button) return;
      var runId = button.dataset.openChallengeRun;
      node("localizationChallengeRun").value = runId;
      setLabTab("challenge");
      loadChallengeRun().catch(function (error) {
        node("localizationChallengeStatus").textContent = error.message;
        node("localizationChallengeStatus").className = "analysis-status error";
      });
    });
    node("localizationChallengeRefresh").addEventListener("click", function () {
      Promise.all([loadRuns(), loadChallengeSets()]).then(loadChallengeRun).catch(function (error) {
        node("localizationChallengeStatus").textContent = error.message;
      });
    });
    node("localizationChallengeRun").addEventListener("change", function () {
      loadChallengeRun().catch(function (error) {
        node("localizationChallengeStatus").textContent = error.message;
        node("localizationChallengeStatus").className = "analysis-status error";
      });
    });
    node("localizationChallengeCriterion").addEventListener("change", function () {
      state.selectedChallengeRollouts = new Set();
      renderChallengeTable();
    });
    ["localizationChallengeTask", "localizationChallengeFailureType", "localizationChallengeCaseFlag"].forEach(function (id) {
      node(id).addEventListener("change", function () {
        state.selectedChallengeRollouts = new Set();
        renderChallengeTable();
      });
    });
    node("localizationChallengePersistentOnly").addEventListener("change", renderChallengeTable);
    node("localizationChallengePropose").addEventListener("click", proposeChallengeSet);
    node("localizationChallengeSave").addEventListener("click", function () {
      saveChallengeSet().catch(function (error) {
        node("localizationChallengeStatus").textContent = error.message;
        node("localizationChallengeStatus").className = "analysis-status error";
      });
    });
    node("localizationChallengeConfigs").addEventListener("change", function (event) {
      if (event.target.matches("[data-challenge-config]")) {
        state.selectedChallengeRollouts = new Set();
        renderChallengeTable();
      }
    });
    node("localizationChallengeTable").addEventListener("change", function (event) {
      var input = event.target.closest("[data-challenge-rollout]");
      if (!input) return;
      if (input.checked) state.selectedChallengeRollouts.add(input.dataset.challengeRollout);
      else state.selectedChallengeRollouts.delete(input.dataset.challengeRollout);
      renderChallengeComposition(challengeRowsForSelection());
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