"use strict";

(function installRoboHopAnalysisUi() {
  var hopState = {
    scope: "libero_10",
    runs: [],
    job: null,
    loadingRuns: false,
    polling: false,
    intervalRanking: {
      population: "first_eligible_event_per_failed_rollout",
      selector: "offline_max_change_score",
      leftWindow: "all",
      rightWindow: "all",
      sortKey: "mse_samples",
      direction: "asc",
      limit: "25"
    }
  };

  function node(id) {
    return document.getElementById(id);
  }

  function esc(value) {
    if (typeof window.escapeHtml === "function") return window.escapeHtml(String(value == null ? "" : value));
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function percent(value, digits) {
    var number = Number(value);
    if (!Number.isFinite(number)) return "n/a";
    return (number * 100).toFixed(digits == null ? 1 : digits) + "%";
  }

  function number(value, digits) {
    var n = Number(value);
    if (!Number.isFinite(n)) return "n/a";
    return n.toFixed(digits == null ? 2 : digits).replace(/0+$/, "").replace(/[.]$/, "");
  }

  function badge(status) {
    var target = node("analysisHopBadge");
    if (!target) return;
    var labels = {
      idle: "Idle",
      queued: "Queued",
      running: "Running",
      complete: "Complete",
      failed: "Failed"
    };
    target.textContent = labels[status] || String(status || "Idle");
    target.className = "analysis-badge"
      + (status === "complete" ? " ok" : "")
      + (status === "failed" ? " error" : "");
  }

  function status(message, kind) {
    var target = node("analysisHopSelection");
    if (!target) return;
    target.textContent = message;
    target.className = "analysis-run-selection" + (kind ? " " + kind : "");
  }

  function environmentReady() {
    return Boolean(
      window.workspaceState
      && workspaceState.analysisEnvironment
      && workspaceState.analysisEnvironment.ready
    );
  }

  function activeJob() {
    return hopState.job && ["queued", "running"].indexOf(hopState.job.status) !== -1;
  }

  function updateButton() {
    var button = node("analysisHopRunButton");
    var select = node("analysisHopRun");
    if (!button || !select) return;
    button.disabled = hopState.loadingRuns || activeJob() || !select.value || !environmentReady();
  }

  function runLabel(run) {
    if (typeof window.workspaceRunLabel === "function") return window.workspaceRunLabel(run);
    return (run.run_root || "unknown run") + " · "
      + (run.run_rollout_count || run.selected_rollouts || 0) + " rollout(s)";
  }

  function populateRuns() {
    var select = node("analysisHopRun");
    if (!select) return;
    var roboRuns = (hopState.runs || []).filter(function (run) {
      return run.baseline === "robo_dopamine"
        && (run.status === "complete" || run.status === "complete_with_errors");
    });
    var choices = roboRuns.filter(function (run) {
      return Number(run.fused_scope_rollout_count || 0) > 0;
    });
    var previous = select.value;
    if (!choices.length) {
      select.innerHTML = '<option value="">No compatible completed Robo-Dopamine run</option>';
      select.disabled = true;
      status(
        roboRuns.length
          ? roboRuns.length + " completed Robo-Dopamine run(s) were found, but none contain saved fused hop for this scope."
          : "No completed Robo-Dopamine run was found for this scope.",
        "warning"
      );
    } else {
      select.innerHTML = choices.map(function (run) {
        var available = Number(run.fused_scope_rollout_count || 0);
        var requested = Number(run.selected_scope_rollouts || 0);
        var coverage = requested ? Math.round(1000 * available / requested) / 10 : 0;
        return '<option value="' + esc(run.run_root) + '">'
          + esc(runLabel(run) + " · fused " + available + "/" + requested + " (" + coverage + "%)")
          + '</option>';
      }).join("");
      select.disabled = false;
      if (choices.some(function (run) { return run.run_root === previous; })) {
        select.value = previous;
      }
      status(
        choices.length + " completed Robo-Dopamine run(s) contain saved fused hop in this scope. Only fused hop is analyzed.",
        ""
      );
    }
    updateButton();
  }

  async function loadRuns() {
    var scopeSelect = node("analysisHopScope");
    hopState.scope = scopeSelect ? scopeSelect.value : hopState.scope;
    hopState.loadingRuns = true;
    updateButton();
    status("Loading completed Robo-Dopamine runs for " + hopState.scope + "…", "");
    try {
      if (typeof window.workspaceLoadBaselineRuns !== "function") {
        throw new Error("Shared baseline run catalog is unavailable");
      }
      await window.workspaceLoadBaselineRuns(hopState.scope, false);
      hopState.runs = (
        window.workspaceState
        && Array.isArray(window.workspaceState.baselineRuns)
      ) ? window.workspaceState.baselineRuns : [];
      populateRuns();
    } catch (error) {
      hopState.runs = [];
      var select = node("analysisHopRun");
      if (select) {
        select.innerHTML = '<option value="">Run discovery failed</option>';
        select.disabled = true;
      }
      status("Robo-Dopamine run discovery failed: " + error.message, "error");
    } finally {
      hopState.loadingRuns = false;
      updateButton();
    }
  }

  function configParameters(row) {
    var fields = [
      ["epsilon", "ε"],
      ["n", "n"],
      ["m", "m"],
      ["k", "k"],
      ["theta", "θ"],
      ["A", "A"],
      ["delta", "δ"]
    ];
    var parts = [];
    fields.forEach(function (pair) {
      if (row[pair[0]] != null && row[pair[0]] !== "") {
        parts.push(pair[1] + "=" + number(row[pair[0]], 3));
      }
    });
    return parts.join(", ") || row.parameters_json || "n/a";
  }

  function familyLabel(value) {
    var raw = String(value || "");
    var labels = {
      consecutive: "Consecutive regression",
      k_of_m: "k-of-m regression",
      window_mean: "Window mean regression",
      cumulative_regression: "Cumulative regression",
      regression_window_min: "Regression window-min",
      stagnation_consecutive: "Stagnation consecutive",
      stagnation_k_of_m: "Stagnation k-of-m",
      phenotype_or: "Stagnation OR regression"
    };
    return labels[raw] || raw.replace(/_/g, " ");
  }

  function ensembleParameters(row, prefix) {
    var mapped = {
      epsilon: row[prefix + "_epsilon"],
      n: row[prefix + "_n"],
      m: row[prefix + "_m"],
      k: row[prefix + "_k"],
      theta: row[prefix + "_theta"],
      A: row[prefix + "_A"],
      delta: row[prefix + "_delta"],
      parameters_json: row[prefix + "_parameters_json"]
    };
    return configParameters(mapped);
  }

  function parametersFromJson(raw) {
    if (!raw) return "n/a";
    try {
      var parsed = JSON.parse(raw);
      return configParameters(parsed);
    } catch (_error) {
      return String(raw);
    }
  }

  function localizationFamilyLabel(row) {
    if (String(row.detector_family || "") !== "phenotype_or") {
      return familyLabel(row.detector_family);
    }
    var left = familyLabel(row.a_detector_family || "stagnation");
    var right = familyLabel(row.b_detector_family || "regression");
    return left + " OR " + right;
  }

  function localizationConfigLabel(row) {
    if (String(row.detector_family || "") !== "phenotype_or") {
      return String(row.config_id || "");
    }
    return "OR combination";
  }

  function localizationConfigParameters(row) {
    if (String(row.detector_family || "") !== "phenotype_or") {
      return configParameters(row);
    }
    var left = parametersFromJson(row.a_parameters_json);
    var right = parametersFromJson(row.b_parameters_json);
    return "stagnation: " + left + " · regression: " + right;
  }


  function rankingSortValue(row, key) {
    var value = row[key];
    if (value == null || value === "") return null;
    var numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : String(value);
  }

  function sortedRankingRows(rows, state) {
    var filtered = (rows || []).filter(function (row) {
      if (String(row.population) !== String(state.population)) return false;
      if (state.selector && state.selector !== "all"
          && String(row.selector) !== String(state.selector)) return false;
      if (state.leftWindow && state.leftWindow !== "all"
          && String(row.L) !== String(state.leftWindow)) return false;
      if (state.rightWindow && state.rightWindow !== "all"
          && String(row.R) !== String(state.rightWindow)) return false;
      return true;
    });
    filtered.sort(function (left, right) {
      var a = rankingSortValue(left, state.sortKey);
      var b = rankingSortValue(right, state.sortKey);
      if (a == null && b == null) return String(left.config_id).localeCompare(String(right.config_id));
      if (a == null) return 1;
      if (b == null) return -1;
      var order = 0;
      if (typeof a === "number" && typeof b === "number") {
        order = a - b;
      } else {
        order = String(a).localeCompare(String(b));
      }
      if (state.direction === "desc") order = -order;
      return order || String(left.config_id).localeCompare(String(right.config_id));
    });
    if (String(state.limit) !== "all") {
      filtered = filtered.slice(0, Math.max(1, Number(state.limit) || 25));
    }
    return filtered;
  }

  function rankingControlsHtml(prefix, state, populations, sortOptions) {
    var populationOptions = populations.map(function (item) {
      return '<option value="' + esc(item.value) + '"'
        + (String(state.population) === String(item.value) ? ' selected' : '')
        + '>' + esc(item.label) + '</option>';
    }).join("");
    var sortHtml = sortOptions.map(function (item) {
      return '<option value="' + esc(item.value) + '"'
        + (String(state.sortKey) === String(item.value) ? ' selected' : '')
        + '>' + esc(item.label) + '</option>';
    }).join("");
    var limits = ["10", "25", "50", "all"].map(function (value) {
      return '<option value="' + value + '"'
        + (String(state.limit) === value ? ' selected' : '')
        + '>' + (value === "all" ? "All" : value) + '</option>';
    }).join("");
    var selectorOptions = [
      ["offline_max_change_score", "Offline max change score"],
      ["first_trigger", "Original first trigger"],
      ["global_fused_progress_argmax", "Global fused-progress argmax"],
      ["all", "All selectors"]
    ].map(function (item) {
      return '<option value="' + item[0] + '"'
        + (String(state.selector) === item[0] ? ' selected' : '')
        + '>' + item[1] + '</option>';
    }).join("");
    var windowOptions = ["all", "2", "3", "5"].map(function (value) {
      return '<option value="' + value + '">'
        + (value === "all" ? "All" : value)
        + '</option>';
    }).join("");
    function selectedWindows(raw, selected) {
      return raw.replace('value="' + selected + '"', 'value="' + selected + '" selected');
    }
    return '<div class="analysis-run-grid">'
      + '<label><span>Population</span><select id="' + prefix + 'Population">' + populationOptions + '</select></label>'
      + '<label><span>Selector</span><select id="' + prefix + 'Selector">' + selectorOptions + '</select></label>'
      + '<label><span>L</span><select id="' + prefix + 'Left">' + selectedWindows(windowOptions, String(state.leftWindow)) + '</select></label>'
      + '<label><span>R</span><select id="' + prefix + 'Right">' + selectedWindows(windowOptions, String(state.rightWindow)) + '</select></label>'
      + '<label><span>Sort by</span><select id="' + prefix + 'Sort">' + sortHtml + '</select></label>'
      + '<label><span>Direction</span><select id="' + prefix + 'Direction">'
      + '<option value="asc"' + (state.direction === "asc" ? ' selected' : '') + '>Ascending</option>'
      + '<option value="desc"' + (state.direction === "desc" ? ' selected' : '') + '>Descending</option>'
      + '</select></label>'
      + '<label><span>Rows</span><select id="' + prefix + 'Limit">' + limits + '</select></label>'
      + '</div>';
  }

  function bindRankingControls(prefix, state) {
    var population = node(prefix + "Population");
    var selector = node(prefix + "Selector");
    var left = node(prefix + "Left");
    var right = node(prefix + "Right");
    var sort = node(prefix + "Sort");
    var direction = node(prefix + "Direction");
    var limit = node(prefix + "Limit");
    if (population) population.addEventListener("change", function () {
      state.population = population.value;
      renderSnapshot();
    });
    if (selector) selector.addEventListener("change", function () {
      state.selector = selector.value;
      renderSnapshot();
    });
    if (left) left.addEventListener("change", function () {
      state.leftWindow = left.value;
      renderSnapshot();
    });
    if (right) right.addEventListener("change", function () {
      state.rightWindow = right.value;
      renderSnapshot();
    });
    if (sort) sort.addEventListener("change", function () {
      state.sortKey = sort.value;
      renderSnapshot();
    });
    if (direction) direction.addEventListener("change", function () {
      state.direction = direction.value;
      renderSnapshot();
    });
    if (limit) limit.addEventListener("change", function () {
      state.limit = limit.value;
      renderSnapshot();
    });
  }



  function resultCell(label, value, note) {
    return '<div class="analysis-result-cell">'
      + '<span>' + esc(label) + '</span>'
      + '<strong>' + esc(value) + '</strong>'
      + (note ? '<small>' + esc(note) + '</small>' : '')
      + '</div>';
  }

  function renderSweepSummary(hop) {
    var selected = (hop.selected_configs || []).filter(function (row) {
      return String(row.selection_status || "selected") === "selected";
    });
    var ensemble = (hop.ensemble_selected || []).filter(function (row) {
      return String(row.selection_status || "") === "selected";
    });
    var sweep = hop.sweep_summary || [];
    var breakdown = hop.breakdown_summary || [];
    var html = '<section class="analysis-subsection analysis-sweep-summary">'
      + '<div class="analysis-subsection-heading"><div><h4>Sweep results</h4>'
      + '<p class="analysis-card-note">Compact view of the completed detector sweep. Full CSV artifacts remain downloadable below.</p></div>'
      + '<span class="analysis-badge ok">Complete</span></div>'
      + '<div class="analysis-result-grid">'
      + resultCell("Configs evaluated", String(sweep.length || hop.detector_config_n || 0), "single-detector sweep rows")
      + resultCell("Selected configs", String(selected.length), "after clean-FPR constraints")
      + resultCell("Selected ensembles", String(ensemble.length), "OR combinations")
      + resultCell("Signal modes", (hop.signal_modes || []).join(", ") || "fused", "")
      + '</div>';

    if (selected.length) {
      html += '<details class="analysis-result-details"><summary>Selected detector configs (' + selected.length + ')</summary>'
        + '<div class="analysis-table-wrap"><table class="analysis-table"><thead><tr>'
        + '<th>Family</th><th>Constraint</th><th>Parameters</th><th>Grasp R@10</th><th>Grasp eventual</th>'
        + '<th>Failure coverage</th><th>Clean FPR</th><th>Median delay</th>'
        + '</tr></thead><tbody>';
      selected.forEach(function (row) {
        html += '<tr><td>' + esc(familyLabel(row.detector_family)) + '</td>'
          + '<td class="numeric">≤ ' + esc(percent(row.clean_fpr_constraint)) + '</td>'
          + '<td><small>' + esc(configParameters(row)) + '</small></td>'
          + '<td class="numeric">' + esc(percent(row.grasp_recall_at_10)) + '</td>'
          + '<td class="numeric"><strong>' + esc(percent(row.grasp_recall_eventual)) + '</strong></td>'
          + '<td class="numeric">' + esc(percent(row.overall_failed_rollout_coverage)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.clean_rollout_fpr)) + '</td>'
          + '<td class="numeric">' + esc(number(row.grasp_median_delay_samples)) + '</td></tr>';
      });
      html += '</tbody></table></div></details>';
    }

    if (ensemble.length) {
      html += '<details class="analysis-result-details"><summary>Selected OR ensembles (' + ensemble.length + ')</summary>'
        + '<div class="analysis-table-wrap"><table class="analysis-table"><thead><tr>'
        + '<th>Target</th><th>Constraint</th><th>Detector A</th><th>Detector B</th>'
        + '<th>Grasp R@10</th><th>Grasp eventual</th><th>Failure coverage</th><th>Clean FPR</th>'
        + '</tr></thead><tbody>';
      ensemble.forEach(function (row) {
        html += '<tr><td>' + esc(String(row.selection_target || "").replace(/_/g, " ")) + '</td>'
          + '<td class="numeric">≤ ' + esc(percent(row.clean_fpr_constraint)) + '</td>'
          + '<td>' + esc(familyLabel(row.detector_a_family)) + '<br><small>' + esc(ensembleParameters(row, "a")) + '</small></td>'
          + '<td>' + esc(familyLabel(row.detector_b_family)) + '<br><small>' + esc(ensembleParameters(row, "b")) + '</small></td>'
          + '<td class="numeric">' + esc(percent(row.grasp_recall_at_10)) + '</td>'
          + '<td class="numeric"><strong>' + esc(percent(row.grasp_recall_eventual)) + '</strong></td>'
          + '<td class="numeric">' + esc(percent(row.overall_failed_rollout_coverage)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.clean_rollout_fpr)) + '</td></tr>';
      });
      html += '</tbody></table></div></details>';
    }

    if (breakdown.length) {
      html += '<details class="analysis-result-details"><summary>Failure-type breakdown (' + breakdown.length + ' rows)</summary>'
        + '<div class="analysis-table-wrap"><table class="analysis-table"><thead><tr>'
        + '<th>Family</th><th>Failure type</th><th>N</th><th>Recall@3</th><th>Recall@10</th><th>Eventual</th><th>Median delay</th>'
        + '</tr></thead><tbody>';
      breakdown.slice(0, 60).forEach(function (row) {
        html += '<tr><td>' + esc(familyLabel(row.detector_family)) + '</td>'
          + '<td>' + esc(String(row.failure_type || "").replace(/_/g, " ")) + '</td>'
          + '<td class="numeric">' + esc(row.event_n == null ? "n/a" : row.event_n) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_at_3)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_at_10)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_eventual)) + '</td>'
          + '<td class="numeric">' + esc(number(row.median_delay_samples)) + '</td></tr>';
      });
      html += '</tbody></table></div></details>';
    }

    return html + '</section>';
  }

  function renderSnapshot() {
    var host = node("analysisHopResults");
    var artifacts = node("analysisHopArtifacts");
    if (!host || !artifacts) return;
    var snapshot = window.workspaceState && workspaceState.analysisSnapshot;
    var hop = snapshot && (snapshot.robo_hop || snapshot.robo_incremental_hop);
    if (!hop || !hop.available) {
      host.innerHTML = '<p class="analysis-empty">' + esc(
        (hop && hop.message) || "No completed Robo-Dopamine fused-hop failure analysis snapshot yet."
      ) + '</p>';
      artifacts.innerHTML = "";
      if (!activeJob()) badge("idle");
      return;
    }

    var html = renderSweepSummary(hop);

    var intervalRows = hop.interval_localization_rows || [];
    if (intervalRows.length) {
      var intervalSortOptions = [
        { value: "mse_samples", label: "MSE" },
        { value: "mae_samples", label: "MAE" },
        { value: "median_absolute_interval_error_samples", label: "Median |error|" },
        { value: "median_signed_interval_error_samples", label: "Median signed interval error" },
        { value: "in_interval_rate", label: "In-interval rate" },
        { value: "within_1", label: "Within 1 sample" },
        { value: "within_3", label: "Within 3 samples" },
        { value: "within_5", label: "Within 5 samples" },
        { value: "before_interval_rate", label: "Before-interval rate" },
        { value: "after_interval_rate", label: "After-interval rate" },
        { value: "trigger_coverage", label: "Trigger coverage" }
      ];
      var intervalPopulations = [
        { value: "all_eligible_failure_events", label: "All eligible failure events" },
        { value: "first_eligible_event_per_failed_rollout", label: "First eligible event per failed rollout" },
        { value: "grasp_failure", label: "Grasp failure" }
      ];
      var intervalViewRows = sortedRankingRows(intervalRows, hopState.intervalRanking);
      html += '<section class="analysis-subsection">'
        + '<div class="analysis-subsection-heading"><div><h4>Failed-rollout interval localization · [causal, observable]</h4></div></div>'
        + '<p class="analysis-card-note">Offline selector keeps every positive episode start, scores s by median(hop[s-L:s]) − median(hop[s:s+R]), and chooses the maximum. Compare it directly with the original first trigger and earliest global fused-progress argmax. Predictions inside [causal onset, observable onset] have zero error.</p>'
        + rankingControlsHtml("analysisHopIntervalRank", hopState.intervalRanking, intervalPopulations, intervalSortOptions)
        + '<table class="analysis-table"><caption>'
        + esc(intervalViewRows.length) + ' row(s) shown · sorted by ' + esc(hopState.intervalRanking.sortKey)
        + ' ' + esc(hopState.intervalRanking.direction)
        + '</caption>'
        + '<thead><tr><th>#</th><th>N</th><th>Selector</th><th>L / R</th><th>Family</th><th>Config</th><th>Parameters</th>'
        + '<th>Trigger coverage</th><th>In interval</th><th>Within 1</th><th>Within 3</th><th>Within 5</th>'
        + '<th>MSE</th><th>MAE</th><th>Median |error|</th><th>Median signed</th>'
        + '<th>Before / After</th></tr></thead><tbody>';
      intervalViewRows.forEach(function (row, index) {
        html += '<tr>'
          + '<td class="numeric"><strong>' + esc(index + 1) + '</strong></td>'
          + '<td class="numeric">' + esc(row.eligible_event_n) + '</td>'
          + '<td>' + esc(String(row.selector || "").replace(/_/g, " ")) + '</td>'
          + '<td class="numeric">' + esc(row.L == null ? "—" : row.L) + ' / ' + esc(row.R == null ? "—" : row.R) + '</td>'
          + '<td>' + esc(localizationFamilyLabel(row)) + '</td>'
          + '<td><code>' + esc(localizationConfigLabel(row)) + '</code></td>'
          + '<td><small>' + esc(localizationConfigParameters(row)) + '</small></td>'
          + '<td class="numeric">' + esc(percent(row.trigger_coverage)) + '</td>'
          + '<td class="numeric"><strong>' + esc(percent(row.in_interval_rate)) + '</strong></td>'
          + '<td class="numeric">' + esc(percent(row.within_1)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.within_3)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.within_5)) + '</td>'
          + '<td class="numeric"><strong>' + esc(number(row.mse_samples)) + '</strong></td>'
          + '<td class="numeric">' + esc(number(row.mae_samples)) + '</td>'
          + '<td class="numeric">' + esc(number(row.median_absolute_interval_error_samples)) + '</td>'
          + '<td class="numeric">' + esc(number(row.median_signed_interval_error_samples)) + '</td>'
          + '<td class="numeric">' + esc(row.before_interval_n) + ' / ' + esc(row.after_interval_n) + '</td>'
          + '</tr>';
      });
      html += '</tbody></table></section>';
    }

    host.innerHTML = html;
    bindRankingControls("analysisHopIntervalRank", hopState.intervalRanking);

    var source = hop.source || {};
    var freshness = hop.freshness || {};
    var signal = hop.signal || {};
    var searchCache = hop.search_cache || {};
    var rolloutCount = signal.common_rollout_n;
    var cacheText = "";
    if (searchCache.enabled) {
      if (searchCache.reuse_source === "cache") {
        cacheText = " · search cache HIT";
      } else if (searchCache.reuse_source === "legacy_analysis") {
        cacheText = " · reused prior search";
      } else {
        cacheText = " · search computed";
      }
    }
    var sourceText = "Latest snapshot: " + (source.directory || "unknown")
      + " · fused-hop rollouts " + (rolloutCount == null ? "n/a" : rolloutCount)
      + cacheText
      + (freshness.stale ? " · STALE against current annotations/manifest" : " · current");
    var links = (hop.artifacts || []).map(function (item) {
      return '<a class="analysis-download-link" href="' + esc(item.url) + '" download><strong>'
        + esc(item.name) + '</strong><small>fused-hop failure analysis</small></a>';
    }).join("");
    artifacts.innerHTML = '<p>' + esc(sourceText) + '</p>'
      + (links ? '<div class="analysis-download-grid">' + links + '</div>' : "");
    if (!activeJob()) badge("complete");
  }

  async function loadLog(jobId) {
    try {
      var response = await fetch(
        "/api/analysis-jobs/" + encodeURIComponent(jobId) + "/log?tail=240",
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (response.ok && hopState.job && hopState.job.job_id === jobId && node("analysisHopLog")) {
        node("analysisHopLog").textContent = payload.log ? payload.log.text : "";
      }
    } catch (_error) {}
  }

  async function pollJob(jobId) {
    if (hopState.polling) return;
    hopState.polling = true;
    try {
      var response = await fetch(
        "/api/analysis-jobs/" + encodeURIComponent(jobId),
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not read Robo-Dopamine hop-comparison job");
      var job = payload.job;
      if (!job || ["robo_hop_comparison", "robo_incremental_hop"].indexOf(job.analysis_kind) === -1) return;
      hopState.job = job;
      badge(job.status);
      await loadLog(jobId);
      if (job.status === "queued" || job.status === "running") {
        status(
          "Fused-hop failure analysis " + job.status + " · "
            + job.selected_rollouts + " rollout(s) · CPU post-processing only.",
          ""
        );
        window.setTimeout(function () {
          hopState.polling = false;
          pollJob(jobId);
        }, 1500);
        updateButton();
        return;
      }
      if (job.status === "complete") {
        status("Fused-hop failure analysis complete · " + (job.output_dir || "snapshot ready"), "");
        if (typeof window.workspaceLoadAnalysis === "function") {
          await window.workspaceLoadAnalysis(true);
        }
        renderSnapshot();
      } else {
        status("Fused-hop failure analysis failed; inspect the log.", "error");
      }
    } catch (error) {
      badge("failed");
      status("Robo-Dopamine hop-comparison job error: " + error.message, "error");
    } finally {
      if (!activeJob()) hopState.polling = false;
      updateButton();
    }
  }

  async function start(event) {
    if (event) event.preventDefault();
    var scope = node("analysisHopScope").value;
    var run = node("analysisHopRun").value;
    var label = node("analysisHopOutputLabel").value.trim();
    var taskCv = Boolean(node("analysisHopTaskCv").checked);
    var cpuLimit = Number(node("analysisHopCpuLimit").value);
    if (!run) {
      status("Select a compatible completed Robo-Dopamine run.", "warning");
      return;
    }
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(label)) {
      status("Output label may contain only letters, numbers, dot, underscore, or hyphen.", "error");
      node("analysisHopOutputLabel").focus();
      return;
    }
    if (!Number.isInteger(cpuLimit) || cpuLimit < 1 || cpuLimit > 16) {
      status("CPU limit must be an integer from 1 to 16.", "error");
      node("analysisHopCpuLimit").focus();
      return;
    }
    if (!environmentReady()) {
      status("The dedicated Analysis environment is unavailable.", "error");
      return;
    }

    badge("queued");
    status("Starting fused-hop Robo-Dopamine failure analysis…", "");
    node("analysisHopLog").textContent = "";
    node("analysisHopRunButton").disabled = true;
    try {
      var response = await fetch("/api/analysis/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          analysis_kind: "robo_hop_comparison",
          scope: scope,
          runs: { robo_dopamine: run },
          task_cv: taskCv,
          cpu_limit: cpuLimit,
          output_label: label
        })
      });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not start fused-hop failure analysis");
      hopState.job = payload.job;
      await loadLog(payload.job.job_id);
      pollJob(payload.job.job_id);
    } catch (error) {
      hopState.job = null;
      badge("failed");
      status("Fused-hop failure analysis error: " + error.message, "error");
      updateButton();
    }
  }

  async function recoverLatestJob() {
    try {
      var response = await fetch("/api/jobs?job_type=analysis", { cache: "no-store" });
      var payload = await response.json();
      if (!response.ok) return;
      var jobs = (payload.jobs || []).filter(function (job) {
        return ["robo_hop_comparison", "robo_incremental_hop"].indexOf(job.analysis_kind) !== -1;
      });
      if (!jobs.length) return;
      hopState.job = jobs[0];
      badge(hopState.job.status);
      if (hopState.job.status === "queued" || hopState.job.status === "running") {
        pollJob(hopState.job.job_id);
      } else {
        await loadLog(hopState.job.job_id);
      }
    } catch (_error) {}
  }

  function patchSnapshotRenderer() {
    if (typeof window.workspaceRenderSnapshot !== "function") return;
    var previous = window.workspaceRenderSnapshot;
    window.workspaceRenderSnapshot = function () {
      var result = previous.apply(this, arguments);
      renderSnapshot();
      return result;
    };
  }

  function init() {
    var form = node("analysisHopForm");
    var scope = node("analysisHopScope");
    var run = node("analysisHopRun");
    if (!form || !scope || !run) return;
    form.addEventListener("submit", start);
    scope.addEventListener("change", loadRuns);
    run.addEventListener("change", updateButton);
    patchSnapshotRenderer();
    if (typeof window.workspaceRenderAnalysisRunPanel === "function") {
      var previousRunPanel = window.workspaceRenderAnalysisRunPanel;
      window.workspaceRenderAnalysisRunPanel = function () {
        var result = previousRunPanel.apply(this, arguments);
        updateButton();
        return result;
      };
    }
    loadRuns();
    recoverLatestJob();
    renderSnapshot();
    updateButton();
  }

  init();
})();
