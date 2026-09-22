"use strict";

(function installLabelLossAblationUi() {
  var state = {
    job: null,
    snapshot: null,
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
  function meanVar(row, metric, asPercent) {
    var mean = row[metric + "_mean"];
    var variance = row[metric + "_variance"];
    var meanText = asPercent ? pct(mean) : num(mean);
    var varText = num(variance, 4);
    return meanText + " / var " + varText;
  }
  async function fetchJson(url, options) {
    var response = await fetch(url, options || { cache: "no-store" });
    var text = await response.text();
    var payload = null;
    try {
      payload = text ? JSON.parse(text) : {};
    } catch (_error) {
      var compact = text.replace(/\s+/g, " ").trim().slice(0, 180);
      throw new Error(
        "HTTP " + response.status + " from " + url
        + " returned non-JSON"
        + (compact ? ": " + compact : "")
      );
    }
    if (!response.ok) {
      throw new Error(
        (payload && payload.error)
        || ("HTTP " + response.status + " from " + url)
      );
    }
    return payload;
  }

  async function readAnalysisJob(jobId) {
    var directError = null;
    try {
      var direct = await fetchJson(
        "/api/analysis-jobs/" + encodeURIComponent(jobId),
        { cache: "no-store" }
      );
      if (direct && direct.job) return direct.job;
      directError = new Error("Direct analysis-job response did not contain job");
    } catch (error) {
      directError = error;
    }

    try {
      var listing = await fetchJson(
        "/api/jobs?job_type=analysis",
        { cache: "no-store" }
      );
      var jobs = (listing.jobs || []).filter(function (job) {
        return String(job.job_id || "") === String(jobId);
      });
      if (jobs.length) return jobs[0];
    } catch (fallbackError) {
      throw new Error(
        "Direct job lookup failed (" + directError.message
        + "); fallback job list also failed (" + fallbackError.message + ")"
      );
    }
    throw new Error(
      "Analysis job " + jobId + " was not found. Direct lookup: "
      + directError.message
    );
  }

  function active() {
    return state.job && ["queued", "running"].indexOf(state.job.status) !== -1;
  }
  function badge(status) {
    var target = node("analysisLabelLossBadge");
    if (!target) return;
    target.textContent = status || "Idle";
    target.className = "analysis-badge"
      + (status === "complete" ? " ok" : "")
      + (status === "failed" ? " error" : "");
  }
  function status(message, kind) {
    var target = node("analysisLabelLossSelection");
    if (!target) return;
    target.textContent = message;
    target.className = "analysis-run-selection" + (kind ? " " + kind : "");
  }
  function updateButton() {
    var button = node("analysisLabelLossRunButton");
    if (button) button.disabled = !!active();
  }

  function table(rows, columns) {
    if (!rows || !rows.length) return '<p class="analysis-empty">No rows.</p>';
    var html = '<div class="analysis-table-wrap"><table class="analysis-table"><thead><tr>';
    columns.forEach(function (column) {
      html += '<th>' + esc(column.label) + '</th>';
    });
    html += '</tr></thead><tbody>';
    rows.forEach(function (row) {
      html += '<tr>';
      columns.forEach(function (column) {
        var value = row[column.key];
        var rendered = column.render ? esc(column.render(value, row)) : esc(value);
        html += '<td' + (column.numeric ? ' class="numeric"' : '') + '>' + rendered + '</td>';
      });
      html += '</tr>';
    });
    return html + '</tbody></table></div>';
  }

  function renderSnapshot() {
    var host = node("analysisLabelLossResults");
    var artifacts = node("analysisLabelLossArtifacts");
    if (!host || !artifacts) return;

    var snapshot = state.snapshot;
    if (!snapshot || !snapshot.available) {
      host.innerHTML = '<p class="analysis-empty">'
        + esc((snapshot && snapshot.message) || "No completed label/loss ablation snapshot yet.")
        + '</p>';
      artifacts.innerHTML = "";
      if (!active()) badge("Idle");
      return;
    }

    var dataset = snapshot.dataset_summary || {};
    var best = snapshot.best_configuration || {};
    var bestLabel = best.best_label || {};
    var bestLoss = best.best_loss || {};
    var labelRows = snapshot.label_ablation || [];
    var lossRows = snapshot.loss_ablation || [];

    var html = '<section class="analysis-result-section"><h4>Dataset / target construction</h4>'
      + '<div class="analysis-result-grid">'
      + '<article class="analysis-result-cell"><small>Failure rollouts</small><strong>'
      + esc(dataset.failure_rollout_n == null ? "n/a" : dataset.failure_rollout_n)
      + '</strong><span>failure-only training</span></article>'
      + '<article class="analysis-result-cell"><small>Annotated events</small><strong>'
      + esc(dataset.annotated_failure_event_n == null ? "n/a" : dataset.annotated_failure_event_n)
      + '</strong><span>all valid events retained</span></article>'
      + '<article class="analysis-result-cell"><small>Multi-event rollouts</small><strong>'
      + esc(dataset.multi_event_rollout_n == null ? "n/a" : dataset.multi_event_rollout_n)
      + '</strong><span>later events exponentially downweighted</span></article>'
      + '<article class="analysis-result-cell"><small>Pseudo frame-0</small><strong>'
      + esc(dataset.pseudo_no_event_frame0_rollout_n == null ? "n/a" : dataset.pseudo_no_event_frame0_rollout_n)
      + '</strong><span>eventless terminal failures</span></article>'
      + '<article class="analysis-result-cell"><small>Event decay τ</small><strong>'
      + esc(num(dataset.tau_event_native_samples, 1))
      + '</strong><span>native samples</span></article>'
      + '<article class="analysis-result-cell"><small>Events / rollout</small><strong>'
      + esc(JSON.stringify(dataset.events_per_rollout || {}))
      + '</strong><span>count distribution after pseudo-event construction</span></article>'
      + '</div></section>';

    html += '<section class="analysis-result-section"><h4>Best configuration</h4>'
      + '<div class="analysis-result-grid">'
      + '<article class="analysis-result-cell"><small>Best label</small><strong>'
      + esc(bestLabel.label_config || "n/a")
      + '</strong><span>in interval ' + esc(pct(bestLabel.in_interval_rate_mean))
      + ' · MAE ' + esc(num(bestLabel.mae_samples_mean)) + '</span></article>'
      + '<article class="analysis-result-cell"><small>Best loss</small><strong>'
      + esc(bestLoss.loss || "n/a")
      + '</strong><span>in interval ' + esc(pct(bestLoss.in_interval_rate_mean))
      + ' · MAE ' + esc(num(bestLoss.mae_samples_mean)) + '</span></article>'
      + '<article class="analysis-result-cell"><small>Asymmetric follow-up</small><strong>'
      + esc(best.asymmetric_followup_ran ? "ran" : "skipped")
      + '</strong><span>soft-label improvement gate</span></article>'
      + '</div></section>';

    html += '<section class="analysis-result-section"><h4>Label ablation</h4>'
      + table(labelRows, [
        { key: "label_config", label: "Label" },
        { key: "repeat_n", label: "Repeats", numeric: true },
        { key: "in_interval_rate_mean", label: "In interval (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "in_interval_rate", true); } },
        { key: "first_event_in_interval_rate_mean", label: "First event (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "first_event_in_interval_rate", true); } },
        { key: "within_1_mean", label: "±1 (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "within_1", true); } },
        { key: "within_3_mean", label: "±3 (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "within_3", true); } },
        { key: "within_5_mean", label: "±5 (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "within_5", true); } },
        { key: "before_interval_rate_mean", label: "Before (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "before_interval_rate", true); } },
        { key: "after_interval_rate_mean", label: "After (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "after_interval_rate", true); } },
        { key: "median_absolute_interval_error_samples_mean", label: "Median |err| (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "median_absolute_interval_error_samples", false); } },
        { key: "mae_samples_mean", label: "MAE (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "mae_samples", false); } },
        { key: "mse_samples_mean", label: "MSE (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "mse_samples", false); } }
      ]) + '</section>';

    html += '<section class="analysis-result-section"><h4>Loss ablation</h4>'
      + table(lossRows, [
        { key: "loss", label: "Loss" },
        { key: "label_config", label: "Label" },
        { key: "repeat_n", label: "Repeats", numeric: true },
        { key: "in_interval_rate_mean", label: "In interval (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "in_interval_rate", true); } },
        { key: "first_event_in_interval_rate_mean", label: "First event (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "first_event_in_interval_rate", true); } },
        { key: "within_1_mean", label: "±1 (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "within_1", true); } },
        { key: "within_3_mean", label: "±3 (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "within_3", true); } },
        { key: "within_5_mean", label: "±5 (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "within_5", true); } },
        { key: "before_interval_rate_mean", label: "Before (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "before_interval_rate", true); } },
        { key: "after_interval_rate_mean", label: "After (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "after_interval_rate", true); } },
        { key: "median_absolute_interval_error_samples_mean", label: "Median |err| (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "median_absolute_interval_error_samples", false); } },
        { key: "mae_samples_mean", label: "MAE (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "mae_samples", false); } },
        { key: "mse_samples_mean", label: "MSE (mean / var)", numeric: true, render: function (_v, row) { return meanVar(row, "mse_samples", false); } }
      ]) + '</section>';

    host.innerHTML = html;

    var source = snapshot.source || {};
    var links = (snapshot.artifacts || []).map(function (item) {
      return '<a class="analysis-download-link" href="' + esc(item.url) + '" download>'
        + '<strong>' + esc(item.name) + '</strong>'
        + '<small>BiLSTM label / loss ablation</small></a>';
    }).join("");
    artifacts.innerHTML = '<p>Latest snapshot: ' + esc(source.directory || "unknown")
      + ' · ' + esc(source.generated_at || "unknown time")
      + ' · ' + esc(source.selection_mode || "unknown selection") + '</p>'
      + (links ? '<div class="analysis-download-grid">' + links + '</div>' : "");

    if (!active()) badge("complete");
  }

  async function loadSnapshot() {
    if (state.snapshotLoading) return;
    state.snapshotLoading = true;
    try {
      var payload = await fetchJson(
        "/api/analysis/robo-label-loss",
        { cache: "no-store" }
      );
      state.snapshot = payload.label_loss || {
        available: false,
        message: "Empty label/loss analysis response"
      };
    } catch (error) {
      state.snapshot = {
        available: false,
        message: "Label/loss snapshot error: " + error.message
      };
    } finally {
      state.snapshotLoading = false;
      renderSnapshot();
    }
  }

  async function loadLog(jobId) {
    try {
      var payload = await fetchJson(
        "/api/analysis-jobs/" + encodeURIComponent(jobId) + "/log?tail=260",
        { cache: "no-store" }
      );
      if (state.job && state.job.job_id === jobId) {
        var log = node("analysisLabelLossLog");
        var nextText = payload.log ? payload.log.text : "";
        if (log && log.textContent !== nextText) log.textContent = nextText;
      }
    } catch (_error) {}
  }

  async function pollJob(jobId) {
    if (state.polling) return;
    state.polling = true;
    try {
      var job = await readAnalysisJob(jobId);
      if (!job || job.analysis_kind !== "robo_bilstm_label_loss_ablation") return;
      state.job = job;
      badge(job.status);
      await loadLog(jobId);
      if (["queued", "running"].indexOf(job.status) !== -1) {
        var parameters = job.parameters || {};
        status(
          "Label/loss ablation " + job.status
          + " · h16 failure-only · latest fused per rollout"
          + " · τ=" + (parameters.tau_event == null ? "?" : parameters.tau_event)
          + " · repeats " + (parameters.repeats || "?")
          + " · batch " + (parameters.batch_size || "?")
          + " · " + (parameters.device || "auto"),
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
        status("Label/loss ablation complete · " + (job.output_dir || ""), "");
        await loadSnapshot();
      } else {
        status("Label/loss ablation failed; inspect the log.", "error");
      }
    } catch (error) {
      badge("failed");
      status("Label/loss job error: " + error.message, "error");
    } finally {
      if (!active()) state.polling = false;
      updateButton();
    }
  }

  function finiteNumber(id, label, minimum, strictlyPositive) {
    var value = Number(node(id).value);
    if (!Number.isFinite(value)
      || (strictlyPositive ? value <= minimum : value < minimum)) {
      throw new Error(label + " is invalid.");
    }
    return value;
  }

  async function start(event) {
    if (event) event.preventDefault();
    var label = node("analysisLabelLossOutputLabel").value.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(label)) {
      status("Invalid output label.", "error");
      return;
    }

    var payload;
    try {
      payload = {
        analysis_kind: "robo_bilstm_label_loss_ablation",
        output_label: label,
        device: node("analysisLabelLossDevice").value,
        tau_event: finiteNumber("analysisLabelLossTauEvent", "Event decay τ", 0, true),
        repeats: Math.round(finiteNumber("analysisLabelLossRepeats", "Repeats", 0, true)),
        epochs: Math.round(finiteNumber("analysisLabelLossEpochs", "Epochs", 0, true)),
        patience: Math.round(finiteNumber("analysisLabelLossPatience", "Patience", 0, true)),
        batch_size: Math.round(finiteNumber("analysisLabelLossBatchSize", "Batch size", 0, true)),
        learning_rate: finiteNumber("analysisLabelLossLearningRate", "Learning rate", 0, true),
        weight_decay: finiteNumber("analysisLabelLossWeightDecay", "Weight decay", 0, false),
        grad_clip: finiteNumber("analysisLabelLossGradClip", "Gradient clip", 0, true),
        distance_weight: finiteNumber("analysisLabelLossDistanceWeight", "Distance weight", 0, false),
        ranking_weight: finiteNumber("analysisLabelLossRankingWeight", "Ranking weight", 0, false),
        ranking_margin: finiteNumber("analysisLabelLossRankingMargin", "Ranking margin", 0, false),
        run_asymmetric_if_soft_improves: !!node("analysisLabelLossAsymmetric").checked
      };
    } catch (error) {
      status(error.message, "error");
      return;
    }

    badge("queued");
    status("Starting BiLSTM label/loss ablation…", "");
    var log = node("analysisLabelLossLog");
    if (log) log.textContent = "";
    updateButton();

    try {
      var body = await fetchJson("/api/analysis/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      state.job = body.job;
      badge(state.job.status);
      await loadLog(state.job.job_id);
      pollJob(state.job.job_id);
    } catch (error) {
      state.job = null;
      badge("failed");
      status("Label/loss start error: " + error.message, "error");
      updateButton();
    }
  }

  async function recoverLatestJob() {
    try {
      var payload = await fetchJson(
        "/api/jobs?job_type=analysis",
        { cache: "no-store" }
      );
      var jobs = (payload.jobs || []).filter(function (job) {
        return job.analysis_kind === "robo_bilstm_label_loss_ablation";
      });
      if (!jobs.length) return;
      state.job = jobs[0];
      badge(state.job.status);
      if (active()) pollJob(state.job.job_id);
      else await loadLog(state.job.job_id);
    } catch (_error) {}
  }

  function init() {
    var form = node("analysisLabelLossForm");
    if (!form) return;
    form.addEventListener("submit", start);
    recoverLatestJob();
    loadSnapshot();
    updateButton();
  }

  init();
})();
