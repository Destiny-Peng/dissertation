"use strict";

/* Rollout-generation execution for Runs. */

function setRolloutGenerationStatus(message, kind) {
  var status = byId("rolloutGenerationStatus");
  var badgeNode = byId("rolloutGenerationBadge");
  if (status) {
    status.textContent = message;
    status.className = "evaluation-status" + (kind ? " " + kind : "");
  }
  if (badgeNode) {
    var job = state.rolloutGenerationJob;
    var label = job && job.status ? job.status : "idle";
    badgeNode.textContent = label;
    badgeNode.className = "analysis-badge" + (
      label === "complete" ? " ok" :
      label === "failed" || label === "memory_blocked" ? " missing" :
      label === "running" || label === "queued" ? " warning" : ""
    );
  }
}

function updateRolloutGenerationSelection() {
  var suiteNode = byId("rolloutGenerationSuite");
  var suite = suiteNode ? suiteNode.value : "libero_10";
  var isSpatial = suite === "libero_spatial";
  var start = Number(byId("rolloutGenerationTaskStart").value);
  var end = Number(byId("rolloutGenerationTaskEnd").value);
  var trials = Number(byId("rolloutGenerationTrials").value);
  var renderResolution = Number(byId("rolloutGenerationRenderResolution").value);
  var recordResolution = Number(byId("rolloutGenerationRecordResolution").value);
  var videoViewMode = byId("rolloutGenerationVideoViewMode").value;
  var saveLatent = byId("rolloutGenerationLogSafeFeatures").checked;
  var note = byId("rolloutGenerationSelection");
  var button = byId("rolloutGenerationRun");
  var title = byId("rolloutGenerationTitle");
  var description = byId("rolloutGenerationDescription");
  if (title) {
    title.textContent = isSpatial
      ? "Generate natural LIBERO-Spatial rollouts"
      : "Generate natural LIBERO-10 rollouts";
  }
  if (description) {
    var viewText = videoViewMode === "libero_three_view"
      ? " The canonical policy video is cam_high. A no-model replay generates only the additional cam_wrist video; camera identity is stored in camera_video_paths, and Robo-Dopamine later maps a shared wrist view to both wrist input slots."
      : " Only the canonical single-view replay video is generated.";
    description.textContent = (isSpatial
      ? "Uses the existing OpenVLA LIBERO-Spatial checkpoint. Render is " + renderResolution + "x" + renderResolution + ", record is " + recordResolution + "x" + recordResolution + ", and policy preprocessing remains 224x224."
      : "Uses the existing OpenVLA LIBERO-10 natural generator and output root. Render is " + renderResolution + "x" + renderResolution + ", record is " + recordResolution + "x" + recordResolution + ", and policy preprocessing remains 224x224.")
      + viewText + " GPU selection is user-managed; the WebUI does not block launch based on utilization or free memory.";
  }
  var valid = (suite === "libero_10" || suite === "libero_spatial")
    && (videoViewMode === "single_view" || videoViewMode === "libero_three_view")
    && Number.isInteger(start) && Number.isInteger(end) && Number.isInteger(trials)
    && Number.isInteger(renderResolution) && Number.isInteger(recordResolution)
    && renderResolution >= 64 && renderResolution <= 2048 && renderResolution % 2 === 0
    && recordResolution >= 64 && recordResolution <= 2048 && recordResolution % 2 === 0
    && start >= 0 && end >= start && end <= 9 && trials >= 1 && trials <= 50;
  if (!valid) {
    note.textContent = "Task range must be 0-9, trials 1-50, and both resolutions must be even values from 64 to 2048.";
    button.disabled = true;
    return;
  }
  var expected = (end - start + 1) * trials;
  note.textContent = (isSpatial ? "LIBERO-Spatial" : "LIBERO-10")
    + " output: " + expected + " rollout(s), render " + renderResolution + "x" + renderResolution
    + ", record " + recordResolution + "x" + recordResolution
    + ", camera videos " + (videoViewMode === "libero_three_view" ? "high + wrist" : "high only")
    + "; run note is generated automatically. "
    + (saveLatent ? "Latent saving enabled." : "Latent saving disabled.");
  button.disabled = state.rolloutGenerationSubmitting;
}

async function loadRolloutGenerationLog(jobId) {
  if (window.LF3RRunsJobs && typeof window.LF3RRunsJobs.refreshSelectedLog === "function") {
    return window.LF3RRunsJobs.refreshSelectedLog("rollout_generation", jobId);
  }
  try {
    var response = await fetch("/api/rollout-jobs/" + encodeURIComponent(jobId) + "/log?tail=200", { cache: "no-store" });
    var payload = await response.json();
    if (response.ok && state.rolloutGenerationJob && state.rolloutGenerationJob.job_id === jobId) {
      byId("rolloutGenerationLog").textContent = payload.log ? payload.log.text : "";
    }
  } catch (_error) {
    // The status endpoint remains useful while the log is being created.
  }
}

function rolloutGenerationJobMessage(job) {
  var progress = (job.completed_rollouts || 0) + "/" + (job.expected_rollouts || 0);
  var suite = job.task_suite === "libero_spatial" ? "LIBERO-Spatial" : "LIBERO-10";
  var resolution = "render " + (job.render_resolution || "?") + "x" + (job.render_resolution || "?")
    + ", record " + (job.record_resolution || "?") + "x" + (job.record_resolution || "?")
    + ", camera videos " + (job.video_view_mode === "libero_three_view" ? "Robo-Dopamine 3-view" : "canonical only");
  if (job.status === "queued") return suite + " generation queued (" + resolution + ") - " + progress + " rollout(s) complete...";
  if (job.status === "running") return "Generating " + suite + " rollouts (" + resolution + ") - " + progress + " complete...";
  if (job.status === "complete") {
    return suite + " generation complete (" + resolution + "): " + progress + " rollout(s); manifest "
      + (job.manifest_rebuilt ? "rebuilt." : "was not rebuilt.");
  }
  if (job.status === "memory_blocked") {
    return "This older rollout-generation job was stopped by a legacy resource check; inspect its log.";
  }
  return suite + " generation failed after " + progress + " rollout(s); inspect the log below.";
}

async function startRolloutGeneration(event) {
  if (event) event.preventDefault();
  if (state.rolloutGenerationSubmitting) return;
  var suiteNode = byId("rolloutGenerationSuite");
  var taskSuite = suiteNode ? suiteNode.value : "libero_10";
  var suiteLabel = taskSuite === "libero_spatial" ? "LIBERO-Spatial" : "LIBERO-10";
  if (taskSuite !== "libero_10" && taskSuite !== "libero_spatial") {
    setRolloutGenerationStatus("Choose a supported task suite.", "error");
    if (suiteNode) suiteNode.focus();
    return;
  }
  var gpu = byId("rolloutGenerationGpu").value.trim();
  var taskStart = Number(byId("rolloutGenerationTaskStart").value);
  var taskEnd = Number(byId("rolloutGenerationTaskEnd").value);
  var trials = Number(byId("rolloutGenerationTrials").value);
  var seed = Number(byId("rolloutGenerationSeed").value);
  var renderResolution = Number(byId("rolloutGenerationRenderResolution").value);
  var recordResolution = Number(byId("rolloutGenerationRecordResolution").value);
  var videoViewMode = byId("rolloutGenerationVideoViewMode").value;
  var saveLatent = byId("rolloutGenerationLogSafeFeatures").checked;
  var label = byId("rolloutGenerationLabel").value.trim();
  if (!/^\d+$/.test(gpu)) {
    setRolloutGenerationStatus("GPU must be one numeric CUDA device index.", "error");
    byId("rolloutGenerationGpu").focus();
    return;
  }
  if (!Number.isInteger(taskStart) || !Number.isInteger(taskEnd)
      || taskStart < 0 || taskEnd < taskStart || taskEnd > 9) {
    setRolloutGenerationStatus("Task range must satisfy 0 <= start <= end <= 9.", "error");
    return;
  }
  if (!Number.isInteger(trials) || trials < 1 || trials > 50) {
    setRolloutGenerationStatus("Trials must be an integer from 1 to 50.", "error");
    byId("rolloutGenerationTrials").focus();
    return;
  }
  if (!Number.isInteger(seed) || seed < 0) {
    setRolloutGenerationStatus("Seed must be a non-negative integer.", "error");
    byId("rolloutGenerationSeed").focus();
    return;
  }
  if (!Number.isInteger(renderResolution) || renderResolution < 64 || renderResolution > 2048 || renderResolution % 2 !== 0) {
    setRolloutGenerationStatus("Render resolution must be an even integer from 64 to 2048.", "error");
    byId("rolloutGenerationRenderResolution").focus();
    return;
  }
  if (!Number.isInteger(recordResolution) || recordResolution < 64 || recordResolution > 2048 || recordResolution % 2 !== 0) {
    setRolloutGenerationStatus("Record resolution must be an even integer from 64 to 2048.", "error");
    byId("rolloutGenerationRecordResolution").focus();
    return;
  }
  if (label && !/^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$/.test(label)) {
    setRolloutGenerationStatus("Run-note label may use only letters, numbers, dot, underscore, or hyphen.", "error");
    byId("rolloutGenerationLabel").focus();
    return;
  }
  if (videoViewMode !== "single_view" && videoViewMode !== "libero_three_view") {
    setRolloutGenerationStatus("Choose a supported review-video mode.", "error");
    byId("rolloutGenerationVideoViewMode").focus();
    return;
  }
  var expected = (taskEnd - taskStart + 1) * trials;
  var multiviewNote = videoViewMode === "libero_three_view"
    ? " A no-model LIBERO replay will then record the additional wrist view; the canonical rollout remains cam_high."
    : "";
  if (!window.confirm("Generate " + expected + " OpenVLA " + suiteLabel + " rollout(s)? This launches GPU inference." + multiviewNote)) return;
  var button = byId("rolloutGenerationRun");
  state.rolloutGenerationSubmitting = true;
  updateRolloutGenerationSelection();
  byId("rolloutGenerationLog").textContent = "";
  setRolloutGenerationStatus("Starting natural " + suiteLabel + " generation...", "");
  try {
    var response = await fetch("/api/rollouts/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task_suite: taskSuite,
        gpu: gpu,
        task_start: taskStart,
        task_end: taskEnd,
        trials: trials,
        seed: seed,
        run_label: label,
        log_safe_features: saveLatent,
        render_resolution: renderResolution,
        record_resolution: recordResolution,
        video_view_mode: videoViewMode
      })
    });
    var payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not start rollout generation");
    state.rolloutGenerationJob = payload.job;
    rememberPersistentJob(payload.job);
    setRolloutGenerationStatus(rolloutGenerationJobMessage(payload.job), "");
    await loadRolloutGenerationLog(payload.job.job_id);
    pollRolloutGenerationJob(payload.job.job_id);
  } catch (error) {
    setRolloutGenerationStatus("Rollout generation error: " + error.message, "error");
  } finally {
    state.rolloutGenerationSubmitting = false;
    updateRolloutGenerationSelection();
  }
}
async function pollRolloutGenerationJob(jobId) {
  return pollPersistentJob(jobId);
}
