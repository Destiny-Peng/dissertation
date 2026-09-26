"use strict";

(function repairSyntheticSuffixModule() {
  var repairState = {
    loaded: false,
    loading: false,
    rollouts: [],
    runs: [],
    selectedRolloutId: "",
    selectedRunId: "",
    validation: null,
    job: null,
    jobTimer: null,
    detail: null,
    syncing: false
  };

  function esc(value) {
    return typeof escapeHtml === "function"
      ? escapeHtml(value)
      : String(value == null ? "" : value);
  }

  async function fetchJson(url, options) {
    var response = await fetch(url, options || { cache: "no-store" });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || ("Request failed: " + response.status));
    return payload;
  }

  function node(id) { return document.getElementById(id); }

  function uniqueValues(rows, key) {
    return Array.from(new Set(rows.map(function (row) {
      return String(row[key] == null ? "" : row[key]);
    }).filter(Boolean))).sort();
  }

  function fillSelect(select, values, allLabel) {
    if (!select) return;
    var current = select.value;
    select.innerHTML = '<option value="all">' + esc(allLabel) + "</option>"
      + values.map(function (value) {
        return '<option value="' + esc(value) + '">' + esc(value) + "</option>";
      }).join("");
    if (values.indexOf(current) !== -1 || current === "all") select.value = current || "all";
  }

  function filteredRollouts() {
    var manifest = node("repairManifestFilter").value;
    var suite = node("repairSuiteFilter").value;
    var task = node("repairTaskFilter").value;
    return repairState.rollouts.filter(function (row) {
      if (manifest !== "all" && String(row.manifest_source || "") !== manifest) return false;
      if (suite !== "all" && String(row.task_suite || "") !== suite) return false;
      if (task !== "all" && String(row.task_id) !== task) return false;
      return true;
    });
  }

  function selectedRollout() {
    return repairState.rollouts.find(function (row) {
      return row.id === repairState.selectedRolloutId;
    }) || null;
  }

  function renderFilters() {
    fillSelect(
      node("repairManifestFilter"),
      uniqueValues(repairState.rollouts, "manifest_source"),
      "All manifests"
    );
    fillSelect(
      node("repairSuiteFilter"),
      uniqueValues(repairState.rollouts, "task_suite"),
      "All suites"
    );
    var scoped = repairState.rollouts.filter(function (row) {
      var manifest = node("repairManifestFilter").value;
      var suite = node("repairSuiteFilter").value;
      return (manifest === "all" || String(row.manifest_source || "") === manifest)
        && (suite === "all" || String(row.task_suite || "") === suite);
    });
    fillSelect(
      node("repairTaskFilter"),
      uniqueValues(scoped, "task_id"),
      "All tasks"
    );
  }

  function renderRolloutSelect() {
    var rows = filteredRollouts();
    var select = node("repairRolloutSelect");
    if (!rows.length) {
      select.innerHTML = '<option value="">No success rollouts match the filters</option>';
      repairState.selectedRolloutId = "";
      renderRolloutSummary();
      return;
    }
    var eligible = rows.filter(function (row) { return Boolean(row.repair_eligible); });
    select.innerHTML = rows.map(function (row) {
      var readyBits = [
        row.actions_available ? "actions✓" : "actions—",
        row.sim_state_available ? "state✓" : "state—",
        (row.views || []).join("+") || "no-view"
      ].join(" · ");
      var suffix = row.repair_eligible
        ? ""
        : " · unavailable: " + ((row.eligibility_reasons || []).join(", ") || "missing required input");
      return '<option value="' + esc(row.id) + '"'
        + (row.repair_eligible ? "" : " disabled")
        + ">" + esc(row.id)
        + " · " + esc(row.task_suite) + "/task" + esc(row.task_id)
        + " · " + esc(readyBits + suffix) + "</option>";
    }).join("");
    if (!eligible.some(function (row) { return row.id === repairState.selectedRolloutId; })) {
      repairState.selectedRolloutId = eligible.length ? eligible[0].id : "";
    }
    select.value = repairState.selectedRolloutId;
    renderRolloutSummary();
  }

  function kv(label, value, className) {
    return '<div class="repair-kv"><small>' + esc(label) + '</small><strong'
      + (className ? ' class="' + className + '"' : "") + ">"
      + esc(value) + "</strong></div>";
  }

  function renderRolloutSummary() {
    var host = node("repairRolloutSummary");
    var row = selectedRollout();
    if (!row) {
      host.innerHTML = kv("Selection", "No rollout selected", "repair-warn");
      return;
    }
    host.innerHTML = [
      kv("Manifest", row.manifest_label || row.manifest_source || "—"),
      kv("Source", row.official_demo ? "Official LIBERO demonstration" : (row.source_kind || "—")),
      kv("Suite / task", String(row.task_suite || "—") + " / " + String(row.task_id)),
      kv("Frames", String(row.frames || "—")),
      kv("Outcome", row.outcome || "—", "repair-ok"),
      kv("Views", (row.views || []).join(", ") || "—"),
      kv("GT actions", row.actions_available ? "Available" : "Unavailable", row.actions_available ? "repair-ok" : "repair-error"),
      kv("Sim state", row.sim_state_available ? "Available" : "Unavailable", row.sim_state_available ? "repair-ok" : "repair-error"),
      kv(
        "Existing WM runs",
        (row.wm_runs || []).length
          ? row.wm_runs.map(function (run) {
              return String(run.world_model || "WM") + ": " + String(run.status || "unknown");
            }).join(", ")
          : "—"
      ),
      kv("Task", row.task_description || "—")
    ].join("");
    updateCutControls();
    renderAdapterPreview();
  }

  function currentCut() {
    var type = node("repairCutType").value;
    var row = selectedRollout();
    var total = row ? Number(row.frames || 0) : 0;
    if (type === "frame") {
      var frame = Math.max(0, Math.min(Math.max(0, total - 2), Number(node("repairCutFrame").value || 0)));
      return { type: type, frame: frame, progress: total > 1 ? frame / (total - 1) : 0 };
    }
    var progress = Math.max(1, Math.min(99, Number(node("repairCutProgress").value || 50))) / 100;
    var computed = Math.floor(Math.max(0, total - 1) * progress);
    computed = Math.min(Math.max(0, total - 2), computed);
    return { type: type, frame: computed, progress: progress };
  }

  function updateCutControls() {
    var row = selectedRollout();
    var total = row ? Number(row.frames || 0) : 0;
    var type = node("repairCutType").value;
    node("repairCutProgressWrap").classList.toggle("hidden", type !== "progress");
    node("repairCutFrameWrap").classList.toggle("hidden", type !== "frame");
    node("repairCutFrame").max = String(Math.max(0, total - 2));
    if (type === "frame" && Number(node("repairCutFrame").value) > total - 2) {
      node("repairCutFrame").value = String(Math.max(0, total - 2));
    }
    var cut = currentCut();
    node("repairCutReadout").textContent = row
      ? "RGB frame " + cut.frame + " · branch state " + (cut.frame + 1)
        + " · GT actions start " + (cut.frame + 1)
      : "Select a rollout";
    var percent = total > 1 ? Math.max(0, Math.min(100, cut.frame / (total - 1) * 100)) : 50;
    node("repairCutPrefix").style.width = percent + "%";
    node("repairCutSuffix").style.width = (100 - percent) + "%";
    repairState.validation = null;
    renderValidation(null);
  }

  function checkpointDefault() {
    var type = node("repairCheckpointType").value;
    if (type === "generic_pretrained") return "checkpoints/a2world-pretrained.pt";
    if (type === "libero_adapted") return "checkpoints/a2world-libero.pt";
    return "";
  }

  function renderAdapterPreview() {
    var row = selectedRollout();
    var views = row ? (row.views || []) : [];
    node("repairCameraMapping").textContent =
      "agentview ← cam_high\neye_in_hand ← cam_wrist"
      + (views.indexOf("cam_wrist") === -1
        ? "\ncam_wrist missing: explicit duplication is required to proceed."
        : "");
    var type = node("repairCheckpointType").value;
    node("repairActionAdapter").textContent =
      "LIBERO 7D → A2World LIBERO servo (checkpoint-independent)";
  }

  function syncCheckpointPlaceholder(force) {
    var input = node("repairCheckpoint");
    var defaultValue = checkpointDefault();
    input.placeholder = defaultValue || "checkpoints/custom-a2world.pt";
    if (force || !input.value.trim()) input.value = defaultValue;
  }

  function buildPayload() {
    var row = selectedRollout();
    if (!row) throw new Error("Select a success rollout first");
    var cut = currentCut();
    var payload = {
      rollout_id: row.id,
      cut_type: cut.type,
      cut_progress: cut.progress,
      cut_frame: cut.frame,
      alignment_min_psnr: Number(node("repairAlignmentPsnr").value || 20),
      generated_includes_condition: false,
      world_model: {
        name: "a2world",
        checkpoint_type: node("repairCheckpointType").value,
        checkpoint: node("repairCheckpoint").value.trim() || checkpointDefault(),
        base_checkpoints: node("repairBaseCheckpoints").value.trim() || "checkpoints",
        duplicate_missing_views: node("repairDuplicateViews").checked,
        num_sampling_steps: Math.max(
          1,
          Math.round(Number(node("repairSamplingSteps").value || 35))
        ),
        guidance: Math.max(0, Number(node("repairGuidance").value || 0)),
        seed: Math.round(Number(node("repairSeed").value || 0)),
        history: node("repairHistory").checked,
        camera_mapping: {
          agentview: "cam_high",
          eye_in_hand: "cam_wrist"
        }
      }
    };
    if (cut.type !== "progress") delete payload.cut_progress;
    if (cut.type !== "frame") delete payload.cut_frame;
    return payload;
  }

  function renderValidation(validation) {
    var host = node("repairValidation");
    if (!validation) {
      host.className = "repair-validation";
      host.textContent = "Run validation to check manifest inputs, exact LIBERO indices, and A2World availability. The simulator smoke test always runs again inside the job before generation.";
      node("repairRunButton").disabled = true;
      return;
    }
    var blockers = validation.blockers || [];
    host.className = "repair-validation " + (validation.ready ? "repair-ok" : "repair-error");
    var smoke = ((validation.validation || {}).alignment_smoke_test || {});
    var comparisons = smoke.comparisons || {};
    var smokeLines = Object.keys(comparisons).map(function (view) {
      var item = comparisons[view] || {};
      function fmt(value) {
        return value == null ? "—" : Number(value).toFixed(2);
      }
      return view + ": restore PSNR " + fmt(item.restore_psnr)
        + " · step PSNR " + fmt(item.step_psnr)
        + " · orientation " + String(item.orientation_transform || "—")
        + " · " + (item.passed ? "pass" : "fail");
    });
    host.textContent = [
      validation.ready ? "Validation passed." : "Blocked.",
      "condition_rgb = rgb[" + validation.alignment.condition_frame + "]",
      "branch_state = states[" + validation.alignment.branch_state_index + "]",
      "future_actions = actions[" + validation.alignment.gt_action_start + ":]",
      "Alignment smoke test: " + (smoke.passed ? "passed" : "not passed"),
      smokeLines.length ? smokeLines.join("\n") : (smoke.error || ""),
      "A2World: " + (validation.world_model.available ? "available" : "unavailable"),
      "A2World config: steps " + String(validation.world_model.num_sampling_steps)
        + " · guidance " + String(validation.world_model.guidance)
        + " · seed " + String(validation.world_model.seed)
        + " · history " + (validation.world_model.history ? "on" : "off"),
      "RGB adapter: derived after alignment (manifest ↔ A2World LIBERO convention)",
      validation.gpu && validation.gpu.selected
        ? ("GPU " + validation.gpu.selected.index + ": "
          + Number(validation.gpu.selected.gpu_utilization_percent).toFixed(1)
          + "% utilization")
        : "GPU: no eligible device below 50%",
      validation.a2world_action_horizon
        ? (
          "A2World horizon: " + validation.a2world_action_horizon.gt_future_action_count
          + " GT future actions · tail padding "
          + validation.a2world_action_horizon.tail_padding_count
          + " · exported padded frames 0"
        )
        : "",
      blockers.length ? ("Blockers:\n- " + blockers.join("\n- ")) : "Ready to generate."
    ].filter(Boolean).join("\n");
    node("repairRunButton").disabled = !validation.ready;
  }

  async function validateCurrent() {
    node("repairValidateButton").disabled = true;
    node("repairValidation").textContent = "Validating metadata and adapter inputs…";
    try {
      var payload = await fetchJson("/api/repair/synthetic-suffix/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload())
      });
      repairState.validation = payload.validation;
      renderValidation(payload.validation);
    } catch (error) {
      repairState.validation = null;
      renderValidation({
        ready: false,
        alignment: { condition_frame: "?", branch_state_index: "?", gt_action_start: "?" },
        world_model: { available: false },
        blockers: [error.message]
      });
    } finally {
      node("repairValidateButton").disabled = false;
    }
  }

  async function startRun() {
    if (!repairState.validation || !repairState.validation.ready) {
      await validateCurrent();
      if (!repairState.validation || !repairState.validation.ready) return;
    }
    node("repairRunButton").disabled = true;
    try {
      var payload = await fetchJson("/api/repair/synthetic-suffix/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload())
      });
      repairState.job = payload.job;
      renderJob();
      pollJob();
    } catch (error) {
      node("repairJobStatus").textContent = "Run submission failed: " + error.message;
      node("repairRunButton").disabled = false;
    }
  }

  function renderJob() {
    var job = repairState.job;
    if (!job) {
      node("repairJobStatus").textContent = "No active Repair job.";
      return;
    }
    var progress = Math.round(Number(job.progress || 0) * 100);
    node("repairJobStatus").textContent =
      String(job.status || "queued") + " · " + String(job.phase || "queued")
      + " · " + progress + "% · " + String(job.run_id || "");
  }

  async function pollJob() {
    if (repairState.jobTimer) window.clearTimeout(repairState.jobTimer);
    var job = repairState.job;
    if (!job || !job.job_id) return;
    try {
      var payload = await fetchJson(
        "/api/repair/synthetic-suffix/jobs/" + encodeURIComponent(job.job_id),
        { cache: "no-store" }
      );
      repairState.job = payload.job;
      renderJob();
      try {
        var logPayload = await fetchJson(
          "/api/repair/synthetic-suffix/jobs/" + encodeURIComponent(job.job_id) + "/log?tail=100",
          { cache: "no-store" }
        );
        node("repairJobLog").textContent = (logPayload.log || {}).text || "";
      } catch (_) {}
      if (["complete", "failed", "complete_with_errors"].indexOf(payload.job.status) !== -1) {
        node("repairRunButton").disabled = false;
        await Promise.all([loadCatalog(true), loadRuns(true)]);
        if (payload.job.run_id) {
          repairState.selectedRunId = payload.job.run_id;
          await loadRunDetail(payload.job.run_id);
        }
        return;
      }
    } catch (error) {
      node("repairJobStatus").textContent = "Job status error: " + error.message;
    }
    repairState.jobTimer = window.setTimeout(pollJob, 1000);
  }

  async function loadCatalog(force) {
    if (repairState.loading && !force) return;
    repairState.loading = true;
    node("repairPageStatus").textContent = "Loading success rollouts…";
    try {
      var payload = await fetchJson("/api/repair/synthetic-suffix/rollouts", { cache: "no-store" });
      repairState.rollouts = payload.rollouts || [];
      renderFilters();
      renderRolloutSelect();
      var eligibleCount = repairState.rollouts.filter(function (row) {
        return Boolean(row.repair_eligible);
      }).length;
      node("repairPageStatus").textContent =
        repairState.rollouts.length + " success rollout(s) found · "
        + eligibleCount + " Repair-eligible.";
      repairState.loaded = true;
    } catch (error) {
      node("repairPageStatus").textContent = "Repair catalog error: " + error.message;
    } finally {
      repairState.loading = false;
    }
  }

  function renderRuns() {
    var select = node("repairRunSelect");
    if (!repairState.runs.length) {
      select.innerHTML = '<option value="">No Repair runs yet</option>';
      repairState.selectedRunId = "";
      renderRunDetail(null);
      return;
    }
    select.innerHTML = repairState.runs.map(function (run) {
      return '<option value="' + esc(run.run_id) + '">' + esc(run.run_id)
        + " · " + esc(run.status) + " · " + esc(run.source_rollout || "") + "</option>";
    }).join("");
    if (!repairState.runs.some(function (run) { return run.run_id === repairState.selectedRunId; })) {
      repairState.selectedRunId = repairState.runs[0].run_id;
    }
    select.value = repairState.selectedRunId;
  }

  async function loadRuns(force) {
    try {
      var payload = await fetchJson("/api/repair/synthetic-suffix/runs", { cache: "no-store" });
      repairState.runs = payload.runs || [];
      renderRuns();
      if (repairState.selectedRunId) await loadRunDetail(repairState.selectedRunId);
    } catch (error) {
      node("repairResultStatus").textContent = "Could not load Repair runs: " + error.message;
    }
  }

  function videoMarkup(url, role, view) {
    if (!url) return '<div class="repair-video-empty">Unavailable</div>';
    return '<video preload="metadata" playsinline data-repair-video="' + esc(role)
      + '" data-repair-view="' + esc(view) + '" src="' + esc(url) + '"></video>';
  }

  function installVideoSync(detail) {
    var videos = Array.from(document.querySelectorAll("[data-repair-video]"));
    var generatedMaster = videos.find(function (video) {
      return video.dataset.repairVideo === "generated" && video.dataset.repairView === "cam_high";
    }) || videos.find(function (video) { return video.dataset.repairVideo === "generated"; });
    if (!generatedMaster) return;
    var realStart = Number(
      (detail.videos || {}).real_suffix_start_time_seconds
      || (detail.videos || {}).cut_time_seconds
      || 0
    );
    function targetTime(video, generatedTime) {
      return video.dataset.repairVideo === "real" ? realStart + generatedTime : generatedTime;
    }
    function sync(generatedTime) {
      if (repairState.syncing) return;
      repairState.syncing = true;
      videos.forEach(function (video) {
        if (video === generatedMaster) return;
        var target = targetTime(video, generatedTime);
        if (Number.isFinite(target) && Math.abs(video.currentTime - target) > 0.12) {
          try { video.currentTime = target; } catch (_) {}
        }
      });
      var slider = node("repairSeek");
      if (slider && !slider.matches(":active")) slider.value = String(generatedTime);
      node("repairTimeReadout").textContent = generatedTime.toFixed(2) + " s after cut";
      repairState.syncing = false;
    }
    generatedMaster.addEventListener("loadedmetadata", function () {
      node("repairSeek").max = String(Number.isFinite(generatedMaster.duration) ? generatedMaster.duration : 1);
      sync(0);
    }, { once: true });
    generatedMaster.addEventListener("timeupdate", function () { sync(generatedMaster.currentTime); });
    node("repairPlayPause").onclick = function () {
      if (generatedMaster.paused) {
        sync(generatedMaster.currentTime);
        videos.forEach(function (video) {
          var promise = video.play();
          if (promise && promise.catch) promise.catch(function () {});
        });
        this.textContent = "Pause";
      } else {
        videos.forEach(function (video) { video.pause(); });
        this.textContent = "Play";
      }
    };
    node("repairSeek").oninput = function () {
      var value = Number(this.value || 0);
      generatedMaster.currentTime = value;
      sync(value);
    };
    videos.forEach(function (video) {
      video.addEventListener("pause", function () {
        if (!repairState.syncing && video === generatedMaster) {
          videos.forEach(function (other) { if (other !== video) other.pause(); });
          node("repairPlayPause").textContent = "Play";
        }
      });
    });
  }

  function renderMetrics(detail) {
    var metrics = detail.metrics || {};
    var views = metrics.views || {};
    var values = ["cam_high", "cam_wrist"].map(function (view) {
      var item = views[view] || {};
      function fmt(value) {
        return value == null ? "—" : Number(value).toFixed(3);
      }
      return {
        view: view,
        text: "PSNR " + fmt(item.psnr_mean)
          + " · SSIM " + fmt(item.ssim_mean)
          + " · LPIPS " + fmt(item.lpips_mean)
      };
    });
    node("repairMetrics").innerHTML = values.map(function (item) {
      return kv(item.view, item.text);
    }).join("") + kv(
      "LPIPS backend",
      metrics.lpips && metrics.lpips.available ? "available" : "unavailable"
    );
    var evaluation = metrics.human_evaluation || {};
    document.querySelectorAll('input[name="repairUsability"]').forEach(function (input) {
      input.checked = input.value === evaluation.usable_for_policy_training;
    });
    document.querySelectorAll("[data-repair-reason]").forEach(function (input) {
      input.checked = (evaluation.failure_reasons || []).indexOf(input.value) !== -1;
    });
  }

  function renderRunDetail(detail) {
    repairState.detail = detail;
    var host = node("repairCompareGrid");
    if (!detail) {
      node("repairResultStatus").textContent = "Select a completed Repair run.";
      host.innerHTML = '<div class="repair-video-empty">No result selected</div>';
      node("repairMetrics").innerHTML = "";
      return;
    }
    var status = detail.status || {};
    node("repairResultStatus").textContent =
      String(status.status || "unknown") + " · " + String(status.phase || "")
      + (status.error ? " · " + status.error : "");
    var real = (detail.videos || {}).real || {};
    var generated = (detail.videos || {}).generated || {};
    host.innerHTML =
      '<div></div><div class="repair-col-head">Real suffix</div><div class="repair-col-head">Generated suffix</div>'
      + '<div class="repair-view-label">cam_high</div>'
      + '<div class="repair-video-shell">' + videoMarkup(real.cam_high, "real", "cam_high") + "</div>"
      + '<div class="repair-video-shell">' + videoMarkup(generated.cam_high, "generated", "cam_high") + "</div>"
      + '<div class="repair-view-label">cam_wrist</div>'
      + '<div class="repair-video-shell">' + videoMarkup(real.cam_wrist, "real", "cam_wrist") + "</div>"
      + '<div class="repair-video-shell">' + videoMarkup(generated.cam_wrist, "generated", "cam_wrist") + "</div>";
    var cut = Number((detail.provenance || {}).cut_progress);
    if (!Number.isFinite(cut)) {
      var frame = Number((detail.provenance || {}).cut_rgb_frame || 0);
      var total = Number(((detail.videos || {}).total_frames) || (detail.provenance || {}).source_total_frames || 0);
      cut = total > 1 ? frame / (total - 1) : 0.5;
    }
    cut = Math.max(0, Math.min(1, Number.isFinite(cut) ? cut : 0.5));
    ["repairRealPrefix", "repairWmPrefix"].forEach(function (id) { node(id).style.width = (cut * 100) + "%"; });
    ["repairRealSuffix", "repairWmSuffix"].forEach(function (id) { node(id).style.width = ((1 - cut) * 100) + "%"; });
    renderMetrics(detail);
    installVideoSync(detail);
  }

  async function loadRunDetail(runId) {
    if (!runId) return renderRunDetail(null);
    node("repairResultStatus").textContent = "Loading Repair result…";
    try {
      var payload = await fetchJson(
        "/api/repair/synthetic-suffix/run/" + encodeURIComponent(runId),
        { cache: "no-store" }
      );
      renderRunDetail(payload.run);
    } catch (error) {
      node("repairResultStatus").textContent = "Result error: " + error.message;
    }
  }

  async function saveHumanEvaluation() {
    if (!repairState.selectedRunId) return;
    var selected = document.querySelector('input[name="repairUsability"]:checked');
    if (!selected) {
      node("repairHumanStatus").textContent = "Choose Yes, No, or Uncertain.";
      return;
    }
    var reasons = Array.from(document.querySelectorAll("[data-repair-reason]:checked"))
      .map(function (input) { return input.value; });
    try {
      var payload = await fetchJson(
        "/api/repair/synthetic-suffix/run/" + encodeURIComponent(repairState.selectedRunId) + "/review",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            usable_for_policy_training: selected.value,
            failure_reasons: reasons
          })
        }
      );
      node("repairHumanStatus").textContent = "Human evaluation saved.";
      if (repairState.detail) {
        repairState.detail.metrics = repairState.detail.metrics || {};
        repairState.detail.metrics.human_evaluation = payload.human_evaluation;
      }
    } catch (error) {
      node("repairHumanStatus").textContent = "Save failed: " + error.message;
    }
  }

  function installEvents() {
    ["repairManifestFilter", "repairSuiteFilter", "repairTaskFilter"].forEach(function (id) {
      node(id).addEventListener("change", function () {
        renderFilters();
        renderRolloutSelect();
      });
    });
    node("repairRolloutSelect").addEventListener("change", function () {
      repairState.selectedRolloutId = this.value;
      renderRolloutSummary();
    });
    ["repairCutType", "repairCutProgress", "repairCutFrame"].forEach(function (id) {
      node(id).addEventListener("input", updateCutControls);
      node(id).addEventListener("change", updateCutControls);
    });
    node("repairCheckpointType").addEventListener("change", function () {
      syncCheckpointPlaceholder(true);
      renderAdapterPreview();
      repairState.validation = null;
      renderValidation(null);
    });
    [
      "repairCheckpoint",
      "repairBaseCheckpoints",
      "repairDuplicateViews",
      "repairAlignmentPsnr",
      "repairSamplingSteps",
      "repairGuidance",
      "repairSeed",
      "repairHistory"
    ].forEach(function (id) {
      node(id).addEventListener("input", function () {
        repairState.validation = null;
        renderValidation(null);
      });
      node(id).addEventListener("change", function () {
        repairState.validation = null;
        renderValidation(null);
      });
    });
    node("repairValidateButton").addEventListener("click", validateCurrent);
    node("repairRunButton").addEventListener("click", startRun);
    node("repairRefreshRuns").addEventListener("click", function () {
      Promise.all([loadCatalog(true), loadRuns(true)]);
    });
    node("repairRunSelect").addEventListener("change", function () {
      repairState.selectedRunId = this.value;
      loadRunDetail(this.value);
    });
    node("repairSaveHuman").addEventListener("click", saveHumanEvaluation);
    syncCheckpointPlaceholder(false);
    renderValidation(null);
  }

  window.addEventListener("lf3r:viewchange", function (event) {
    if (!event.detail || event.detail.view !== "repair") return;
    if (!repairState.loaded) {
      Promise.all([loadCatalog(false), loadRuns(false)]);
    } else {
      loadRuns(false);
    }
  });

  installEvents();
  window.LF3RRepairSyntheticSuffix = {
    refresh: function () { return Promise.all([loadCatalog(true), loadRuns(true)]); },
    state: repairState
  };
  if (window.workspaceState && window.workspaceState.view === "repair") {
    Promise.all([loadCatalog(false), loadRuns(false)]);
  }
})();
