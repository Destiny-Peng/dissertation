"use strict";

(function installRoboHopAnalysisUi() {
  var hopState = {
    scope: "libero_10",
    runs: [],
    job: null,
    loadingRuns: false,
    polling: false
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
    if (String(value || "") === "regression_window_min") {
      return "regression · window min";
    }
    return String(value || "")
      .replace(/^stagnation_/, "stagnation · ")
      .replace(/_/g, " ");
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

  function renderSnapshot() {
    var host = node("analysisHopResults");
    var artifacts = node("analysisHopArtifacts");
    if (!host || !artifacts) return;
    var snapshot = window.workspaceState && workspaceState.analysisSnapshot;
    var hop = snapshot && (snapshot.robo_hop || snapshot.robo_incremental_hop);
    if (!hop || !hop.available) {
      host.innerHTML = '<p class="analysis-empty">No completed Robo-Dopamine fused-hop failure analysis snapshot yet.</p>';
      artifacts.innerHTML = "";
      if (!activeJob()) badge("idle");
      return;
    }

    var rows = (hop.best_configs || []).filter(function (row) {
      return row.selection_status === "selected"
        && (!row.signal_mode || String(row.signal_mode) === "fused");
    }).sort(function (left, right) {
      return String(left.detector_family).localeCompare(String(right.detector_family))
        || Number(left.clean_fpr_constraint) - Number(right.clean_fpr_constraint);
    });

    var html = '<section class="analysis-subsection">'
      + '<h4>Fused hop · single detectors</h4>'
      + '<p class="analysis-card-note">Event failures use observable-onset localization; terminal failures without events use rollout-start time-to-alarm; clean successes remain the FPR control.</p>';

    if (!rows.length) {
      html += '<p class="analysis-empty">No selected single-detector configuration satisfies the clean-FPR constraints.</p>';
    } else {
      html += '<table class="analysis-table"><caption>Existing single-detector representatives; no new threshold tuning.</caption>'
        + '<thead><tr><th>Family</th><th>FPR cap</th><th>Parameters</th>'
        + '<th>Grasp R@10</th><th>Grasp eventual</th>'
        + '<th>Event R@1</th><th>R@3</th><th>R@5</th><th>R@10</th><th>R@20</th><th>Eventual</th>'
        + '<th>No-event R@1</th><th>R@3</th><th>R@5</th><th>R@10</th><th>R@20</th><th>Eventual</th>'
        + '<th>Overall failed-rollout coverage</th><th>Clean FPR</th></tr></thead><tbody>';
      rows.forEach(function (row) {
        html += '<tr>'
          + '<td><strong>' + esc(familyLabel(row.detector_family)) + '</strong></td>'
          + '<td class="numeric">≤ ' + esc(percent(row.clean_fpr_constraint, 0)) + '</td>'
          + '<td>' + esc(configParameters(row)) + '</td>'
          + '<td class="numeric"><strong>' + esc(percent(row.grasp_recall_at_10)) + '</strong></td>'
          + '<td class="numeric"><strong>' + esc(percent(row.grasp_recall_eventual)) + '</strong></td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_at_1)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_at_3)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_at_5)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_at_10)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_at_20)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_eventual)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.no_event_recall_at_1)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.no_event_recall_at_3)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.no_event_recall_at_5)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.no_event_recall_at_10)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.no_event_recall_at_20)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.no_event_recall_eventual)) + '</td>'
          + '<td class="numeric"><strong>' + esc(percent(row.overall_failed_rollout_coverage)) + '</strong></td>'
          + '<td class="numeric">' + esc(percent(row.clean_rollout_fpr)) + '</td>'
          + '</tr>';
      });
      html += '</tbody></table>';
    }
    html += '</section>';

    var ensembles = (hop.ensemble_selected || []).filter(function (row) {
      return row.selection_status === "selected"
        && (String(row.selection_target) === "grasp_recall_eventual"
          || String(row.selection_target) === "grasp_recall_at_10");
    }).sort(function (left, right) {
      var targetOrder = {
        grasp_recall_eventual: 0,
        grasp_recall_at_10: 1
      };
      return Number(left.clean_fpr_constraint) - Number(right.clean_fpr_constraint)
        || (targetOrder[String(left.selection_target)] || 0)
          - (targetOrder[String(right.selection_target)] || 0)
        || Number(left.pair_priority || 99) - Number(right.pair_priority || 99);
    });

    html += '<section class="analysis-subsection">'
      + '<h4>Fused hop · stagnation OR regression joint search</h4>'
      + '<p class="analysis-card-note">Stagnation uses near-zero consecutive/k-of-m evidence; regression uses a short rolling-window minimum with empirical θr candidates from the saved fused-hop data. Under each total clean-FPR cap, configurations are selected separately for grasp eventual recall and grasp Recall@10; overall failed-rollout coverage remains a reported secondary metric.</p>';

    if (!ensembles.length) {
      html += '<p class="analysis-empty">No joint OR ensemble result is available in this snapshot.</p>';
    } else {
      html += '<table class="analysis-table"><thead><tr><th>FPR cap</th><th>Selection target</th><th>Stagnation</th><th>Regression</th>'
        + '<th>Grasp R@10</th><th>Gain@10</th><th>Grasp eventual</th><th>Gain eventual</th>'
        + '<th>Event R@1</th><th>R@3</th><th>R@5</th><th>R@10</th><th>R@20</th><th>Eventual</th>'
        + '<th>No-event R@1</th><th>R@3</th><th>R@5</th><th>R@10</th><th>R@20</th><th>Eventual</th>'
        + '<th>Overall coverage</th><th>Event delay</th><th>No-event delay</th>'
        + '<th>FP overlap</th><th>OR FPR</th></tr></thead><tbody>';
      ensembles.forEach(function (row) {
        html += '<tr>'
          + '<td class="numeric">≤ ' + esc(percent(row.clean_fpr_constraint, 0)) + '</td>'
          + '<td>' + esc(String(row.selection_target) === "grasp_recall_eventual" ? "grasp eventual" : "grasp R@10") + '</td>'
          + '<td><strong>' + esc(familyLabel(row.detector_a_family)) + '</strong><small>'
          + esc(ensembleParameters(row, "a")) + '</small></td>'
          + '<td><strong>' + esc(familyLabel(row.detector_b_family)) + '</strong><small>'
          + esc(ensembleParameters(row, "b")) + '</small></td>'
          + '<td class="numeric"><strong>' + esc(percent(row.grasp_recall_at_10)) + '</strong></td>'
          + '<td class="numeric">' + esc(percent(row.grasp_gain_vs_best_at_10)) + '</td>'
          + '<td class="numeric"><strong>' + esc(percent(row.grasp_recall_eventual)) + '</strong></td>'
          + '<td class="numeric">' + esc(percent(row.grasp_gain_vs_best_eventual)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_at_1)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_at_3)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_at_5)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_at_10)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_at_20)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.event_recall_eventual)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.no_event_recall_at_1)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.no_event_recall_at_3)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.no_event_recall_at_5)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.no_event_recall_at_10)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.no_event_recall_at_20)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.no_event_recall_eventual)) + '</td>'
          + '<td class="numeric"><strong>' + esc(percent(row.overall_failed_rollout_coverage)) + '</strong></td>'
          + '<td class="numeric">' + esc(number(row.event_median_delay_samples)) + ' samples</td>'
          + '<td class="numeric">' + esc(number(row.no_event_median_delay_samples)) + ' samples</td>'
          + '<td class="numeric">' + esc(row.fp_overlap_n == null ? "n/a" : row.fp_overlap_n) + '</td>'
          + '<td class="numeric"><strong>' + esc(percent(row.clean_rollout_fpr)) + '</strong></td>'
          + '</tr>';
      });
      html += '</tbody></table>';
    }

    var oracleGrasp = (hop.oracle_summary || []).filter(function (row) {
      return String(row.population) === "grasp_failure";
    }).sort(function (left, right) {
      var order = { stagnation: 0, regression: 1, combined: 2 };
      return (order[String(left.signal_family)] || 99)
        - (order[String(right.signal_family)] || 99);
    });

    if (oracleGrasp.length) {
      html += '<section class="analysis-subsection">'
        + '<h4>FPR-unconstrained oracle detectability · grasp failure</h4>'
        + '<p class="analysis-card-note">Oracle rows ask whether any searched parameter state can localize each event. A positive episode must start at/after observable onset, with at most one native sample of early tolerance; an alarm that was already continuously positive long before onset does not count.</p>'
        + '<table class="analysis-table"><thead><tr><th>Phenotype</th>'
        + '<th>Oracle R@1</th><th>R@3</th><th>R@5</th><th>R@10</th><th>R@20</th><th>Eventual</th>'
        + '<th>Strict eventual</th><th>Median best onset offset</th><th>P25–P75 offset</th></tr></thead><tbody>';
      oracleGrasp.forEach(function (row) {
        html += '<tr>'
          + '<td><strong>' + esc(String(row.signal_family)) + '</strong></td>'
          + '<td class="numeric">' + esc(percent(row.oracle_recall_at_1)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.oracle_recall_at_3)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.oracle_recall_at_5)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.oracle_recall_at_10)) + '</td>'
          + '<td class="numeric">' + esc(percent(row.oracle_recall_at_20)) + '</td>'
          + '<td class="numeric"><strong>' + esc(percent(row.oracle_recall_eventual)) + '</strong></td>'
          + '<td class="numeric">' + esc(percent(row.strict_recall_eventual)) + '</td>'
          + '<td class="numeric">' + esc(number(row.median_best_start_offset_samples)) + ' samples</td>'
          + '<td class="numeric">' + esc(number(row.p25_best_start_offset_samples)) + '–'
          + esc(number(row.p75_best_start_offset_samples)) + '</td>'
          + '</tr>';
      });
      html += '</tbody></table>';

      var oracleCombined = oracleGrasp.find(function (row) {
        return String(row.signal_family) === "combined";
      });
      var constrained = (hop.ensemble_selected || []).filter(function (row) {
        return row.selection_status === "selected"
          && String(row.selection_target) === "grasp_recall_eventual";
      }).sort(function (left, right) {
        return Number(right.clean_fpr_constraint) - Number(left.clean_fpr_constraint);
      });
      if (oracleCombined) {
        html += '<table class="analysis-table"><caption>Oracle capacity versus deployable clean-FPR budgets</caption>'
          + '<thead><tr><th>Setting</th><th>Grasp R@3</th><th>R@10</th><th>Eventual</th>'
          + '<th>Median delay</th><th>Observed clean FPR</th></tr></thead><tbody>'
          + '<tr><td><strong>Oracle · ignore FPR</strong></td>'
          + '<td class="numeric">' + esc(percent(oracleCombined.oracle_recall_at_3)) + '</td>'
          + '<td class="numeric">' + esc(percent(oracleCombined.oracle_recall_at_10)) + '</td>'
          + '<td class="numeric"><strong>' + esc(percent(oracleCombined.oracle_recall_eventual)) + '</strong></td>'
          + '<td class="numeric">offset ' + esc(number(oracleCombined.median_best_start_offset_samples)) + '</td>'
          + '<td class="numeric">ignored</td></tr>';
        constrained.forEach(function (row) {
          html += '<tr><td>≤ ' + esc(percent(row.clean_fpr_constraint, 0)) + ' clean FPR</td>'
            + '<td class="numeric">' + esc(percent(row.grasp_recall_at_3)) + '</td>'
            + '<td class="numeric">' + esc(percent(row.grasp_recall_at_10)) + '</td>'
            + '<td class="numeric"><strong>' + esc(percent(row.grasp_recall_eventual)) + '</strong></td>'
            + '<td class="numeric">' + esc(number(row.grasp_median_delay_samples)) + ' samples (1-based)</td>'
            + '<td class="numeric">' + esc(percent(row.clean_rollout_fpr)) + '</td></tr>';
        });
        html += '</tbody></table>';
      }
      html += '</section>';
    }

    var typed = (hop.ensemble_by_failure_type || []).filter(function (row) {
      return Math.abs(Number(row.clean_fpr_constraint) - 0.20) < 1e-9
        && String(row.horizon) === "eventual"
        && String(row.selected_for || "").indexOf("grasp_recall_eventual") !== -1;
    });
    if (typed.length) {
      html += '<details class="analysis-subsection"><summary><strong>Failure-type complementarity · ≤20% · eventual</strong></summary>'
        + '<table class="analysis-table"><thead><tr><th>Failure type</th><th>A</th><th>B</th><th>Events</th>'
        + '<th>Overlap</th><th>A-only</th><th>B-only</th><th>OR recall</th><th>TP Jaccard</th></tr></thead><tbody>';
      typed.forEach(function (row) {
        html += '<tr>'
          + '<td><strong>' + esc(String(row.failure_type).replace(/_/g, " ")) + '</strong></td>'
          + '<td>' + esc(familyLabel(row.detector_a_family)) + '</td>'
          + '<td>' + esc(familyLabel(row.detector_b_family)) + '</td>'
          + '<td class="numeric">' + esc(row.event_n) + '</td>'
          + '<td class="numeric">' + esc(row.overlap_n) + '</td>'
          + '<td class="numeric">' + esc(row.a_only_n) + '</td>'
          + '<td class="numeric">' + esc(row.b_only_n) + '</td>'
          + '<td class="numeric"><strong>' + esc(percent(row.or_recall)) + '</strong></td>'
          + '<td class="numeric">' + esc(percent(row.tp_jaccard)) + '</td>'
          + '</tr>';
      });
      html += '</tbody></table></details>';
    }

    host.innerHTML = html;

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
