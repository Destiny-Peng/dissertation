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
      return Number(run.four_signal_scope_rollout_count || 0) > 0;
    });
    var previous = select.value;
    if (!choices.length) {
      select.innerHTML = '<option value="">No compatible completed Robo-Dopamine run</option>';
      select.disabled = true;
      status(
        roboRuns.length
          ? roboRuns.length + " completed Robo-Dopamine run(s) were found, but none contain a common rollout set with all four hop signals: incremental, forward, backward, and fused."
          : "No completed Robo-Dopamine run was found for this scope.",
        "warning"
      );
    } else {
      select.innerHTML = choices.map(function (run) {
        var available = Number(run.four_signal_scope_rollout_count || 0);
        var requested = Number(run.selected_scope_rollouts || 0);
        var coverage = requested ? Math.round(1000 * available / requested) / 10 : 0;
        return '<option value="' + esc(run.run_root) + '">'
          + esc(runLabel(run) + " · four-signal " + available + "/" + requested + " (" + coverage + "%)")
          + '</option>';
      }).join("");
      select.disabled = false;
      if (choices.some(function (run) { return run.run_root === previous; })) {
        select.value = previous;
      }
      status(
        choices.length + " completed Robo-Dopamine run(s) contain a common four-signal rollout set in this scope. The analysis compares incremental, forward, backward, and fused hop on exactly the same rollouts.",
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
    status("Discovering completed Robo-Dopamine runs for " + hopState.scope + "…", "");
    try {
      var response = await fetch(
        "/api/baselines/runs?scope=" + encodeURIComponent(hopState.scope),
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not discover Robo-Dopamine runs");
      hopState.runs = payload.runs || [];
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
    return String(value || "")
      .replace(/^stagnation_/, "stagnation · ")
      .replace(/_/g, " ");
  }

  function renderSnapshot() {
    var host = node("analysisHopResults");
    var artifacts = node("analysisHopArtifacts");
    if (!host || !artifacts) return;
    var snapshot = window.workspaceState && workspaceState.analysisSnapshot;
    var hop = snapshot && (snapshot.robo_hop || snapshot.robo_incremental_hop);
    if (!hop || !hop.available) {
      host.innerHTML = '<p class="analysis-empty">No completed four-signal Robo-Dopamine hop analysis snapshot yet.</p>';
      artifacts.innerHTML = "";
      if (!activeJob()) badge("idle");
      return;
    }

    var modeOrder = ["incremental", "forward", "backward", "fused"];
    var rows = (hop.best_configs || []).filter(function (row) {
      return row.selection_status === "selected";
    }).sort(function (left, right) {
      return modeOrder.indexOf(String(left.signal_mode)) - modeOrder.indexOf(String(right.signal_mode))
        || String(left.detector_family).localeCompare(String(right.detector_family))
        || Number(left.clean_fpr_constraint) - Number(right.clean_fpr_constraint);
    });

    if (!rows.length) {
      host.innerHTML = '<p class="analysis-empty">The latest snapshot has no parameter configuration satisfying the requested clean-FPR constraints.</p>';
    } else {
      var sweep = Array.isArray(hop.sweep_summary) ? hop.sweep_summary : [];
      var sections = modeOrder.map(function (mode) {
        var modeRows = rows.filter(function (row) {
          return String(row.signal_mode) === mode;
        });
        if (!modeRows.length) return "";

        var tenPercent = modeRows.filter(function (row) {
          return Math.abs(Number(row.clean_fpr_constraint) - 0.10) < 1e-9;
        });
        var modeSweep = sweep.filter(function (row) {
          return String(row.signal_mode) === mode;
        });
        var feasible5 = modeSweep.filter(function (row) {
          return Number(row.clean_rollout_fpr) <= 0.05 + 1e-12;
        }).length;
        var feasible10 = modeSweep.filter(function (row) {
          return Number(row.clean_rollout_fpr) <= 0.10 + 1e-12;
        }).length;
        var feasible20 = modeSweep.filter(function (row) {
          return Number(row.clean_rollout_fpr) <= 0.20 + 1e-12;
        }).length;

        var headline = tenPercent.length
          ? '<div class="analysis-kpis">'
            + tenPercent.map(function (row) {
              return '<article><span>' + esc(familyLabel(row.detector_family)) + ' · ≤10% clean FPR</span>'
                + '<strong>' + esc(percent(row.event_recall_at_3)) + '</strong>'
                + '<small>recall@3 · median ' + esc(number(row.median_delay_samples))
                + ' samples · observed FPR ' + esc(percent(row.clean_rollout_fpr)) + '</small></article>';
            }).join('')
            + '</div>'
          : '';

        var html = '<section class="analysis-subsection">'
          + '<h4>' + esc(mode.charAt(0).toUpperCase() + mode.slice(1)) + ' hop</h4>'
          + '<p class="analysis-card-note">Full sweep: ' + esc(modeSweep.length)
          + ' configurations · feasible under clean-FPR caps: '
          + esc(feasible5) + ' @5%, ' + esc(feasible10) + ' @10%, ' + esc(feasible20) + ' @20%.</p>'
          + headline
          + '<table class="analysis-table"><caption>Best ' + esc(mode)
          + ' configurations under the 5%, 10%, and 20% clean-rollout FPR constraints.</caption>'
          + '<thead><tr><th>Family</th><th>Clean-FPR cap</th><th>Parameters</th><th>Events</th><th>Recall@3</th><th>Median delay</th><th>Clean FPR</th></tr></thead><tbody>';

        modeRows.forEach(function (row) {
          html += '<tr>'
            + '<td><strong>' + esc(familyLabel(row.detector_family)) + '</strong></td>'
            + '<td class="numeric">≤ ' + esc(percent(row.clean_fpr_constraint, 0)) + '</td>'
            + '<td>' + esc(configParameters(row)) + '</td>'
            + '<td class="numeric">' + esc(row.event_n == null ? "n/a" : row.event_n) + '</td>'
            + '<td class="numeric">' + esc(percent(row.event_recall_at_3)) + '</td>'
            + '<td class="numeric">' + esc(number(row.median_delay_samples)) + ' samples / '
            + esc(number(row.median_delay_frames)) + ' frames</td>'
            + '<td class="numeric">' + esc(percent(row.clean_rollout_fpr)) + '</td>'
            + '</tr>';
        });
        return html + '</tbody></table></section>';
      }).join("");
      host.innerHTML = sections;
    }

    var source = hop.source || {};
    var freshness = hop.freshness || {};
    var signal = hop.signal || {};
    var commonCount = signal.common_rollout_n;
    var sourceText = "Latest snapshot: " + (source.directory || "unknown")
      + " · common four-signal rollouts " + (commonCount == null ? "n/a" : commonCount)
      + (freshness.stale ? " · STALE against current annotations/manifest" : " · current");
    var links = (hop.artifacts || []).map(function (item) {
      return '<a class="analysis-download-link" href="' + esc(item.url) + '" download><strong>'
        + esc(item.name) + '</strong><small>four-signal hop comparison</small></a>';
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
          "Four-signal hop analysis " + job.status + " · "
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
        status("Four-signal hop analysis complete · " + (job.output_dir || "snapshot ready"), "");
        if (typeof window.workspaceLoadAnalysis === "function") {
          await window.workspaceLoadAnalysis(true);
        }
        renderSnapshot();
      } else {
        status("Four-signal hop analysis failed; inspect the log.", "error");
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
    if (!run) {
      status("Select a compatible completed Robo-Dopamine run.", "warning");
      return;
    }
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(label)) {
      status("Output label may contain only letters, numbers, dot, underscore, or hyphen.", "error");
      node("analysisHopOutputLabel").focus();
      return;
    }
    if (!environmentReady()) {
      status("The dedicated Analysis environment is unavailable.", "error");
      return;
    }

    badge("queued");
    status("Starting four-signal Robo-Dopamine hop analysis…", "");
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
          output_label: label
        })
      });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not start four-signal hop analysis");
      hopState.job = payload.job;
      await loadLog(payload.job.job_id);
      pollJob(payload.job.job_id);
    } catch (error) {
      hopState.job = null;
      badge("failed");
      status("Four-signal hop analysis error: " + error.message, "error");
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
