"use strict";

(function installBiLstmAblationUi() {
  var state = {
    runs: [],
    job: null,
    snapshot: null,
    loadingRuns: false,
    snapshotLoading: false,
    polling: false
  };

  function node(id) { return document.getElementById(id); }
  function esc(value) {
    if (typeof window.escapeHtml === "function") {
      return window.escapeHtml(String(value == null ? "" : value));
    }
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function num(value, digits) {
    var n = Number(value);
    if (!Number.isFinite(n)) return "n/a";
    return n.toFixed(digits == null ? 3 : digits).replace(/0+$/, "").replace(/[.]$/, "");
  }
  function pct(value) {
    var n = Number(value);
    return Number.isFinite(n) ? (100 * n).toFixed(1) + "%" : "n/a";
  }
  function active() {
    return state.job && ["queued", "running"].indexOf(state.job.status) !== -1;
  }
  function badge(status) {
    var target = node("analysisBiLstmBadge");
    if (!target) return;
    target.textContent = status || "Idle";
    target.className = "analysis-badge"
      + (status === "complete" ? " ok" : "")
      + (status === "failed" ? " error" : "");
  }
  function status(message, kind) {
    var target = node("analysisBiLstmSelection");
    if (!target) return;
    target.textContent = message;
    target.className = "analysis-run-selection" + (kind ? " " + kind : "");
  }
  function updateButton() {
    var button = node("analysisBiLstmRunButton");
    var select = node("analysisBiLstmRun");
    if (!button || !select) return;
    button.disabled = state.loadingRuns || active() || !select.value;
  }
  function runLabel(run) {
    if (typeof window.workspaceRunLabel === "function") return window.workspaceRunLabel(run);
    return (run.run_root || "unknown run") + " · "
      + (run.run_rollout_count || run.selected_rollouts || 0) + " rollout(s)";
  }

  function populateRuns() {
    var select = node("analysisBiLstmRun");
    if (!select) return;
    var choices = (state.runs || []).filter(function (run) {
      return run.baseline === "robo_dopamine"
        && ["complete", "complete_with_errors"].indexOf(run.status) !== -1
        && Number(run.fused_scope_rollout_count || run.fused_rollout_count || 0) > 0;
    });
    var previous = select.value;
    if (!choices.length) {
      select.innerHTML = '<option value="">No compatible completed Robo-Dopamine run</option>';
      select.disabled = true;
      status("No completed Robo-Dopamine run with saved fused-hop outputs was found.", "warning");
    } else {
      select.innerHTML = choices.map(function (run) {
        return '<option value="' + esc(run.run_root) + '">' + esc(runLabel(run)) + '</option>';
      }).join("");
      select.disabled = false;
      if (choices.some(function (run) { return run.run_root === previous; })) {
        select.value = previous;
      }
      status(
        choices.length + " compatible Robo-Dopamine run(s). Training reuses saved fused progress/hop only.",
        ""
      );
    }
    updateButton();
  }

  async function loadRuns() {
    state.loadingRuns = true;
    updateButton();
    try {
      var response = await fetch(
        "/api/baselines/runs?scope=all",
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Could not discover Robo-Dopamine runs");
      }
      state.runs = Array.isArray(payload.runs) ? payload.runs : [];
      populateRuns();
    } catch (error) {
      state.runs = [];
      status("Run discovery failed: " + error.message, "error");
    } finally {
      state.loadingRuns = false;
      updateButton();
    }
  }

  function renderSnapshot() {
    var host = node("analysisBiLstmResults");
    var artifacts = node("analysisBiLstmArtifacts");
    if (!host || !artifacts) return;
    var snapshot = state.snapshot;
    if (!snapshot || !snapshot.available) {
      host.innerHTML = '<p class="analysis-empty">'
        + esc((snapshot && snapshot.message) || "No completed BiLSTM success-ratio snapshot yet.")
        + '</p>';
      artifacts.innerHTML = "";
      if (!active()) badge("Idle");
      return;
    }

    var comparison = (snapshot.comparison || []).slice().sort(function (a, b) {
      var modelOrder = String(a.model).localeCompare(String(b.model));
      if (modelOrder) return modelOrder;
      return Number(a.success_ratio || 0) - Number(b.success_ratio || 0);
    });
    var delta = snapshot.delta || [];
    var html = "";

    ["tiny_bilstm_h16", "tiny_bilstm_h32"].forEach(function (model) {
      var rows = comparison.filter(function (row) { return row.model === model; });
      if (!rows.length) return;
      html += '<section class="analysis-result-section"><h4><code>' + esc(model) + '</code> success-ratio trend</h4>'
        + '<div class="analysis-result-grid">';
      rows.forEach(function (row) {
        html += '<article class="analysis-result-cell"><small>' + esc(num(row.success_ratio, 1)) + '× success</small>'
          + '<strong>' + esc(pct(row.before_interval_rate_mean)) + '</strong>'
          + '<span>before interval</span>'
          + '<span>in interval ' + esc(pct(row.in_interval_rate_mean)) + '</span>'
          + '<span>success n≈' + esc(num(row.success_train_n_mean, 1))
          + ' / failure n≈' + esc(num(row.failure_train_n_mean, 1)) + '</span></article>';
      });
      html += '</div></section>';
    });

    if (comparison.length) {
      html += '<div class="analysis-table-wrap"><table class="analysis-table"><thead><tr>'
        + '<th>Model</th><th>Success ratio</th><th>Failure train</th><th>Success train</th>'
        + '<th>In interval</th><th>±1</th><th>±3</th><th>±5</th>'
        + '<th>Before</th><th>After</th><th>Median |err|</th><th>MAE</th><th>MSE</th>'
        + '</tr></thead><tbody>';
      comparison.forEach(function (row) {
        html += '<tr><td><code>' + esc(row.model) + '</code></td>'
          + '<td class="numeric"><strong>' + esc(num(row.success_ratio, 1)) + '×</strong></td>'
          + '<td class="numeric">' + esc(num(row.failure_train_n_mean, 1)) + '</td>'
          + '<td class="numeric">' + esc(num(row.success_train_n_mean, 1)) + '</td>'
          + '<td class="numeric">' + esc(pct(row.in_interval_rate_mean)) + '</td>'
          + '<td class="numeric">' + esc(pct(row.within_1_mean)) + '</td>'
          + '<td class="numeric">' + esc(pct(row.within_3_mean)) + '</td>'
          + '<td class="numeric">' + esc(pct(row.within_5_mean)) + '</td>'
          + '<td class="numeric"><strong>' + esc(pct(row.before_interval_rate_mean)) + '</strong></td>'
          + '<td class="numeric">' + esc(pct(row.after_interval_rate_mean)) + '</td>'
          + '<td class="numeric">' + esc(num(row.median_absolute_interval_error_samples_mean)) + '</td>'
          + '<td class="numeric">' + esc(num(row.mae_samples_mean)) + '</td>'
          + '<td class="numeric">' + esc(num(row.mse_samples_mean)) + '</td></tr>';
      });
      html += '</tbody></table></div>';
    }

    if (delta.length) {
      html += '<details class="analysis-result-details"><summary>Each non-zero ratio − 0× baseline</summary>'
        + '<div class="analysis-table-wrap"><table class="analysis-table"><thead><tr>'
        + '<th>Model</th><th>Ratio</th><th>Δ in interval</th><th>Δ before</th><th>Δ after</th><th>Δ MAE</th><th>Δ MSE</th>'
        + '</tr></thead><tbody>';
      delta.forEach(function (row) {
        html += '<tr><td><code>' + esc(row.model) + '</code></td>'
          + '<td class="numeric">' + esc(num(row.success_ratio, 1)) + '×</td>'
          + '<td class="numeric">' + esc(pct(row.in_interval_rate_delta_vs_ratio0)) + '</td>'
          + '<td class="numeric"><strong>' + esc(pct(row.before_interval_rate_delta_vs_ratio0)) + '</strong></td>'
          + '<td class="numeric">' + esc(pct(row.after_interval_rate_delta_vs_ratio0)) + '</td>'
          + '<td class="numeric">' + esc(num(row.mae_samples_delta_vs_ratio0)) + '</td>'
          + '<td class="numeric">' + esc(num(row.mse_samples_delta_vs_ratio0)) + '</td></tr>';
      });
      html += '</tbody></table></div></details>';
    }

    host.innerHTML = html;
    var source = snapshot.source || {};
    var links = (snapshot.artifacts || []).map(function (item) {
      return '<a class="analysis-download-link" href="' + esc(item.url) + '" download>'
        + '<strong>' + esc(item.name) + '</strong><small>BiLSTM success-ratio ablation</small></a>';
    }).join("");
    artifacts.innerHTML = '<p>Latest snapshot: ' + esc(source.directory || "unknown")
      + ' · ' + esc(source.generated_at || "unknown time") + '</p>'
      + (links ? '<div class="analysis-download-grid">' + links + '</div>' : "");
    if (!active()) badge("complete");
  }

  async function loadSnapshot() {
    if (state.snapshotLoading) return;
    state.snapshotLoading = true;
    try {
      var response = await fetch("/api/analysis/robo-localization-head", { cache: "no-store" });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not load BiLSTM snapshot");
      state.snapshot = payload.localization_head || {
        available: false,
        message: "Empty BiLSTM analysis response"
      };
    } catch (error) {
      state.snapshot = { available: false, message: "BiLSTM snapshot error: " + error.message };
    } finally {
      state.snapshotLoading = false;
      renderSnapshot();
    }
  }

  async function loadLog(jobId) {
    try {
      var response = await fetch(
        "/api/analysis-jobs/" + encodeURIComponent(jobId) + "/log?tail=240",
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (response.ok && state.job && state.job.job_id === jobId) {
        var log = node("analysisBiLstmLog");
        var nextText = payload.log ? payload.log.text : "";
        if (log && log.textContent !== nextText) log.textContent = nextText;
      }
    } catch (_error) {}
  }

  async function pollJob(jobId) {
    if (state.polling) return;
    state.polling = true;
    try {
      var response = await fetch(
        "/api/analysis-jobs/" + encodeURIComponent(jobId),
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not read BiLSTM job");
      var job = payload.job;
      if (!job || job.analysis_kind !== "robo_bilstm_success_ablation") return;
      state.job = job;
      badge(job.status);
      await loadLog(jobId);
      if (["queued", "running"].indexOf(job.status) !== -1) {
        status(
          "BiLSTM training " + job.status + " · " + (job.parameters.device || "auto")
          + " · repeats " + (job.parameters.repeats || "?"),
          ""
        );
        window.setTimeout(function () {
          state.polling = false;
          pollJob(jobId);
        }, 1500);
        updateButton();
        return;
      }
      if (job.status === "complete") {
        status("BiLSTM success-ratio ablation complete · " + (job.output_dir || ""), "");
        await loadSnapshot();
      } else {
        status("BiLSTM training failed; inspect the log.", "error");
      }
    } catch (error) {
      badge("failed");
      status("BiLSTM training job error: " + error.message, "error");
    } finally {
      if (!active()) state.polling = false;
      updateButton();
    }
  }

  async function start(event) {
    if (event) event.preventDefault();
    var run = node("analysisBiLstmRun").value;
    var label = node("analysisBiLstmOutputLabel").value.trim();
    if (!run) {
      status("Select a completed Robo-Dopamine run.", "warning");
      return;
    }
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(label)) {
      status("Invalid output label.", "error");
      return;
    }
    var payload = {
      analysis_kind: "robo_bilstm_success_ablation",
      runs: { robo_dopamine: run },
      output_label: label,
      device: node("analysisBiLstmDevice").value,
      repeats: Number(node("analysisBiLstmRepeats").value),
      epochs: Number(node("analysisBiLstmEpochs").value),
      patience: Number(node("analysisBiLstmPatience").value),
      learning_rate: Number(node("analysisBiLstmLearningRate").value),
      weight_decay: Number(node("analysisBiLstmWeightDecay").value),
      grad_clip: Number(node("analysisBiLstmGradClip").value)
    };

    badge("queued");
    status("Starting PyTorch BiLSTM success-ratio ablation…", "");
    var log = node("analysisBiLstmLog");
    if (log) log.textContent = "";
    updateButton();
    try {
      var response = await fetch("/api/analysis/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      var body = await response.json();
      if (!response.ok) throw new Error(body.error || "Could not start BiLSTM training");
      state.job = body.job;
      badge(state.job.status);
      await loadLog(state.job.job_id);
      pollJob(state.job.job_id);
    } catch (error) {
      state.job = null;
      badge("failed");
      status("BiLSTM training error: " + error.message, "error");
      updateButton();
    }
  }

  async function recoverLatestJob() {
    try {
      var response = await fetch("/api/jobs?job_type=analysis", { cache: "no-store" });
      var payload = await response.json();
      if (!response.ok) return;
      var jobs = (payload.jobs || []).filter(function (job) {
        return job.analysis_kind === "robo_bilstm_success_ablation";
      });
      if (!jobs.length) return;
      state.job = jobs[0];
      badge(state.job.status);
      if (active()) pollJob(state.job.job_id);
      else await loadLog(state.job.job_id);
    } catch (_error) {}
  }

  function init() {
    var form = node("analysisBiLstmForm");
    if (!form) return;
    form.addEventListener("submit", start);
    node("analysisBiLstmRun").addEventListener("change", updateButton);
    loadRuns();
    recoverLatestJob();
    loadSnapshot();
    updateButton();
  }

  init();
})();
