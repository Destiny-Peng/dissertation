"use strict";

window.LF3RProjectTools = (function createProjectToolsController() {
  function install() {
    var view = document.getElementById("runsView");
    if (!view || view.dataset.projectToolsInstalled === "true") return;
    view.dataset.projectToolsInstalled = "true";

    function text(id) {
        var node = document.getElementById(id);
        return node ? String(node.value || "").trim() : "";
      }
    
      function numberValue(id, fallback) {
        var value = Number(text(id));
        return Number.isFinite(value) ? value : fallback;
      }

    var toolPollTimer = null;
      var activeToolJobId = null;
      var manifestRebuildBefore = {};
      var manifestRefreshHandled = {};
    
      var manifestExtraRootsNode = document.getElementById("rebuildManifestExtraRoots");
      if (manifestExtraRootsNode) {
        try {
          manifestExtraRootsNode.value = localStorage.getItem("lf3r.runs.extraManifestScanRoots") || "";
        } catch (_) {}
        manifestExtraRootsNode.addEventListener("input", function () {
          try {
            localStorage.setItem("lf3r.runs.extraManifestScanRoots", manifestExtraRootsNode.value);
          } catch (_) {}
        });
      }
    
      function activityNodes() {
        return [
          {
            status: document.getElementById("projectToolStatus"),
            jobs: document.getElementById("projectToolJobs"),
            log: document.getElementById("projectToolLog")
          },
          {
            status: document.getElementById("projectToolStatusTools"),
            jobs: document.getElementById("projectToolJobsTools"),
            log: document.getElementById("projectToolLogTools")
          }
        ];
      }
    
      function renderToolJobs(jobs) {
        activityNodes().forEach(function (nodes) {
          if (!nodes.jobs) return;
          nodes.jobs.innerHTML = jobs.length ? jobs.slice(0, 5).map(function (job) {
            return '<button type="button" class="runs-tool-job' + (job.job_id === activeToolJobId ? ' active' : '') + '" data-tool-job="' + job.job_id + '"><strong>' + (job.tool_label || job.action || 'Project tool') + '</strong><span>' + (job.status || 'unknown') + '</span></button>';
          }).join("") : '<div class="runs-tool-job-empty">No project-tool jobs yet.</div>';
        });
        Array.prototype.slice.call(document.querySelectorAll("[data-tool-job]")).forEach(function (button) {
          button.addEventListener("click", function () {
            activeToolJobId = button.dataset.toolJob;
            refreshToolJobs();
            pollToolJob();
          });
        });
      }
    
      async function refreshToolJobs() {
        try {
          var response = await fetch("/api/tool-jobs", { cache: "no-store" });
          var payload = await response.json();
          if (!response.ok) throw new Error(payload.error || "Could not read project-tool jobs");
          projectToolsProtocol = payload.project_tools_protocol || null;
          var jobs = payload.jobs || [];
          if (!activeToolJobId && jobs.length) activeToolJobId = jobs[0].job_id;
          var rebuildButton = document.getElementById("rebuildManifestRun");
          if (rebuildButton) {
            rebuildButton.disabled = jobs.some(function (job) {
              return job.action === "rebuild_manifest"
                && (job.status === "queued" || job.status === "running");
            });
          }
          renderToolJobs(jobs);
        } catch (_) {}
      }
    
      async function pollToolJob() {
        if (toolPollTimer) clearTimeout(toolPollTimer);
        if (!activeToolJobId) return;
        try {
          var responses = await Promise.all([
            fetch("/api/tool-jobs/" + encodeURIComponent(activeToolJobId), { cache: "no-store" }),
            fetch("/api/tool-jobs/" + encodeURIComponent(activeToolJobId) + "/log?tail=300", { cache: "no-store" })
          ]);
          var jobPayload = await responses[0].json();
          var logPayload = await responses[1].json();
          if (!responses[0].ok) throw new Error(jobPayload.error || "Could not read job");
          var job = jobPayload.job || {};
          var logText = logPayload.log && logPayload.log.text ? logPayload.log.text : "";
          activityNodes().forEach(function (nodes) {
            if (nodes.status) nodes.status.textContent = (job.tool_label || job.action || "Project tool") + " · " + (job.status || "unknown") + (job.error ? " · " + job.error : "");
            if (nodes.log) nodes.log.textContent = logText;
          });
          await refreshToolJobs();
          if (job.action === "transcode_manifest_videos") {
            var batchSummary = document.getElementById("batchManifestTranscodeSummary");
            if (batchSummary) {
              var match = logText.match(
                /BATCH_H264_SUMMARY selected=(\d+) converted=(\d+) already_h264=(\d+) conflicts=(\d+) missing=(\d+) failed=(\d+)/
              );
              if (match) {
                batchSummary.textContent = "Selected " + match[1]
                  + " · converted " + match[2]
                  + " · already H.264 " + match[3]
                  + " · backup conflicts " + match[4]
                  + " · missing " + match[5]
                  + " · failed " + match[6]
                  + " · " + (job.status || "unknown");
              } else if (job.status === "queued" || job.status === "running") {
                batchSummary.textContent = "Batch H.264 transcode " + job.status + "… see Tool activity for per-video progress.";
              } else if (job.status === "failed") {
                batchSummary.textContent = "Batch H.264 transcode failed: " + (job.error || "see Tool activity log");
              }
            }
          }
          if (
            job.action === "rebuild_manifest"
            && job.status === "failed"
            && !manifestRefreshHandled[job.job_id]
          ) {
            manifestRefreshHandled[job.job_id] = true;
            var failedSummary = document.getElementById("rebuildManifestSummary");
            if (failedSummary) {
              failedSummary.textContent = "Manifest rebuild failed: " + (job.error || "see project-tool log");
            }
          }
          if (
            job.action === "rebuild_manifest"
            && job.status === "complete"
            && !manifestRefreshHandled[job.job_id]
          ) {
            manifestRefreshHandled[job.job_id] = true;
            var beforeCount = Object.prototype.hasOwnProperty.call(manifestRebuildBefore, job.job_id)
              ? manifestRebuildBefore[job.job_id]
              : null;
            var summaryNode = document.getElementById("rebuildManifestSummary");
            if (summaryNode) summaryNode.textContent = "Manifest rebuilt; refreshing rollout catalog…";
            try {
              var preferredId = (typeof state !== "undefined" && state) ? state.selectedId : null;
              if (typeof loadRollouts === "function") {
                await loadRollouts(preferredId);
              }
              await loadRolloutOptions();
              await loadBatchManifestOptions(true);
              var afterCount = (typeof state !== "undefined" && state && Array.isArray(state.rollouts))
                ? state.rollouts.length
                : null;
              if (summaryNode) {
                if (afterCount == null) {
                  summaryNode.textContent = "Manifest rebuilt successfully. Rollout catalog refresh completed.";
                } else if (beforeCount == null) {
                  summaryNode.textContent = "Manifest rebuilt successfully · " + afterCount + " rollout(s) loaded.";
                } else {
                  var delta = afterCount - beforeCount;
                  var deltaText = delta > 0 ? " · +" + delta + " new" : (delta < 0 ? " · " + delta + " net" : " · no net count change");
                  summaryNode.textContent = "Manifest rebuilt successfully · " + afterCount + " rollout(s)" + deltaText + ".";
                }
              }
            } catch (refreshError) {
              if (summaryNode) {
                summaryNode.textContent = "Manifest rebuilt, but automatic catalog refresh failed: " + String(refreshError.message || refreshError);
              }
            }
          }
          if (job.status === "queued" || job.status === "running") {
            toolPollTimer = setTimeout(pollToolJob, 1000);
          }
        } catch (error) {
          activityNodes().forEach(function (nodes) {
            if (nodes.status) nodes.status.textContent = String(error.message || error);
          });
        }
      }
    
      function waitMs(milliseconds) {
        return new Promise(function (resolve) {
          window.setTimeout(resolve, milliseconds);
        });
      }
    
      function makeToolClientRequestId(action) {
        return "web:" + action + ":" + Date.now() + ":" + Math.random().toString(36).slice(2, 10);
      }
    
      async function recoverToolSubmission(action, clientRequestId) {
        for (var attempt = 0; attempt < 5; attempt += 1) {
          try {
            var response = await fetch("/api/tool-jobs", { cache: "no-store" });
            var payload = await response.json();
            if (response.ok) {
              projectToolsProtocol = payload.project_tools_protocol || null;
              var jobs = payload.jobs || [];
              var match = jobs.find(function (job) {
                return job
                  && job.action === action
                  && job.client_request_id === clientRequestId;
              });
              if (match) return match;
            }
          } catch (_) {}
          if (attempt < 4) await waitMs(250 * (attempt + 1));
        }
        return null;
      }
    
      async function submitTool(action, options) {
        activityNodes().forEach(function (nodes) {
          if (nodes.status) nodes.status.textContent = "Submitting " + action + "…";
        });
    
        var clientRequestId = makeToolClientRequestId(action);
        var controller = typeof AbortController === "function" ? new AbortController() : null;
        var timeoutId = window.setTimeout(function () {
          if (controller) controller.abort();
        }, 8000);
        var job = null;
    
        try {
          var response = await fetch("/api/tools/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              action: action,
              options: options || {},
              client_request_id: clientRequestId
            }),
            signal: controller ? controller.signal : undefined
          });
          var payload = await response.json();
          if (!response.ok) {
            var message = payload.error || "Project-tool request failed";
            activityNodes().forEach(function (nodes) {
              if (nodes.status) nodes.status.textContent = message;
            });
            throw new Error(message);
          }
          job = payload.job;
        } catch (error) {
          job = await recoverToolSubmission(action, clientRequestId);
          if (!job) {
            var staleBackend = projectToolsProtocol !== "immediate-registry-v1";
            var message = staleBackend
              ? "Project-tool backend is stale or incompatible. Restart the LF3R WebUI server after git pull."
              : (error && error.name === "AbortError"
                  ? "Project-tool submission timed out before a job could be confirmed."
                  : "Project-tool submission failed: " + String(error.message || error));
            activityNodes().forEach(function (nodes) {
              if (nodes.status) nodes.status.textContent = message;
            });
            throw new Error(message);
          }
          activityNodes().forEach(function (nodes) {
            if (nodes.status) nodes.status.textContent = "Recovered submitted " + action + " job.";
          });
        } finally {
          window.clearTimeout(timeoutId);
        }
    
        activeToolJobId = job.job_id;
        await refreshToolJobs();
        pollToolJob();
        return job;
      }
    
      var prepareStamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..*/, "").replace("T", "_");
      document.getElementById("safePrepareOutput").value = "outputs/safe_training/datasets/web_" + prepareStamp;
      document.getElementById("safeTrainDataset").value = document.getElementById("safePrepareOutput").value;
      document.getElementById("safeValidateDataset").value = document.getElementById("safePrepareOutput").value;
    
      document.getElementById("safePrepareRun").addEventListener("click", function () {
        var output = text("safePrepareOutput");
        document.getElementById("safeTrainDataset").value = output;
        document.getElementById("safeValidateDataset").value = output;
        submitTool("safe_prepare", {
          dataset_role: text("safePrepareRole"),
          partition: text("safePreparePartition"),
          run_name: text("safePrepareRunName"),
          output: output
        }).catch(function () {});
      });
    
      document.getElementById("safeTrainRun").addEventListener("click", function () {
        submitTool("safe_train", {
          dataset_dir: text("safeTrainDataset"),
          model: text("safeTrainModel"),
          gpu: text("safeTrainGpu"),
          epochs: numberValue("safeTrainEpochs", 1000),
          batch_size: numberValue("safeTrainBatch", 512),
          hidden_dim: numberValue("safeTrainHidden", 256),
          seed: text("safeTrainSeed"),
          logs_root: text("safeTrainLogs"),
          normalize: document.getElementById("safeTrainNormalize").checked
        }).catch(function () {});
      });
    
      document.getElementById("safeValidateRun").addEventListener("click", function () {
        submitTool("safe_validate", {
          dataset_dir: text("safeValidateDataset"),
          checkpoint: text("safeValidateCheckpoint"),
          model: text("safeValidateModel"),
          gpu: text("safeValidateGpu"),
          output: text("safeValidateOutput"),
          hidden_dim: numberValue("safeTrainHidden", 256)
        }).catch(function () {});
      });
    
      function exportOptions(dryRun) {
        var outcomes = Array.prototype.slice.call(document.querySelectorAll("#exportOutcomes input:checked")).map(function (node) { return node.value; });
        return {
          outcomes: outcomes,
          dataset_role: text("exportDatasetRole"),
          review_status: text("exportReviewStatus"),
          output_dir: text("exportOutputDir"),
          dry_run: !!dryRun
        };
      }
      document.getElementById("exportDryRun").addEventListener("click", function () { submitTool("export_cases", exportOptions(true)).catch(function () {}); });
      document.getElementById("exportRun").addEventListener("click", function () { submitTool("export_cases", exportOptions(false)).catch(function () {}); });
    
      document.getElementById("roboSweepRun").addEventListener("click", function () {
        var intervals = text("roboSweepIntervals").split(",").map(function (item) { return Number(item.trim()); }).filter(function (item) { return Number.isInteger(item) && item > 0; });
        submitTool("robo_interval_sweep", {
          rollout_ids: [text("roboSweepRollout")],
          intervals: intervals,
          gpu: text("roboSweepGpu"),
          memory_utilization: numberValue("roboSweepMemory", 0.6),
          output_dir: text("roboSweepOutput")
        }).catch(function () {});
      });
    
      var batchManifestCatalog = [];
      var projectToolsProtocol = null;
    
      function selectedBatchManifestPaths() {
        return Array.prototype.slice.call(
          document.querySelectorAll("#batchManifestTranscodeManifests input[data-manifest-path]:checked")
        ).map(function (input) { return input.dataset.manifestPath; });
      }
    
      function updateBatchManifestSelectionCount() {
        var selected = selectedBatchManifestPaths();
        var selectedRows = batchManifestCatalog.reduce(function (total, item) {
          return selected.indexOf(item.path) >= 0 ? total + Number(item.rollouts || 0) : total;
        }, 0);
        var node = document.getElementById("batchManifestTranscodeSelectionCount");
        if (node) {
          node.textContent = selected.length + " / " + batchManifestCatalog.length
            + " manifest(s) selected · " + selectedRows + " manifest row(s)";
        }
        var button = document.getElementById("batchManifestTranscodeRun");
        if (button) button.disabled = selected.length === 0;
      }
    
      function renderBatchManifestOptions(manifests, preserveSelection) {
        var host = document.getElementById("batchManifestTranscodeManifests");
        if (!host) return;
        var previous = preserveSelection ? selectedBatchManifestPaths() : [];
        batchManifestCatalog = (manifests || []).filter(function (item) {
          return item && item.exists && item.path;
        });
        host.innerHTML = "";
        if (!batchManifestCatalog.length) {
          var empty = document.createElement("div");
          empty.className = "runs-manifest-choice-empty";
          empty.textContent = "No loaded manifest files are available.";
          host.appendChild(empty);
          updateBatchManifestSelectionCount();
          return;
        }
    
        batchManifestCatalog.forEach(function (item) {
          var label = document.createElement("label");
          label.className = "runs-manifest-choice";
          var input = document.createElement("input");
          input.type = "checkbox";
          input.dataset.manifestPath = String(item.path);
          input.checked = preserveSelection
            ? previous.indexOf(String(item.path)) >= 0
            : true;
          input.addEventListener("change", updateBatchManifestSelectionCount);
    
          var copy = document.createElement("span");
          var title = document.createElement("strong");
          title.textContent = String(item.label || item.path);
          var meta = document.createElement("small");
          meta.textContent = String(item.path)
            + " · " + Number(item.rollouts || 0) + " rollout(s)"
            + (item.primary ? " · primary" : "");
          copy.appendChild(title);
          copy.appendChild(meta);
          label.appendChild(input);
          label.appendChild(copy);
          host.appendChild(label);
        });
        updateBatchManifestSelectionCount();
      }
    
      async function loadBatchManifestOptions(preserveSelection) {
        var host = document.getElementById("batchManifestTranscodeManifests");
        try {
          var response = await fetch("/api/manifests", { cache: "no-store" });
          var payload = await response.json();
          if (!response.ok) throw new Error(payload.error || "Could not read loaded manifests");
          renderBatchManifestOptions(payload.manifests || [], Boolean(preserveSelection));
        } catch (error) {
          batchManifestCatalog = [];
          if (host) {
            host.innerHTML = "";
            var empty = document.createElement("div");
            empty.className = "runs-manifest-choice-empty error";
            empty.textContent = "Manifest options unavailable: " + String(error.message || error);
            host.appendChild(empty);
          }
          updateBatchManifestSelectionCount();
        }
      }
    
      document.getElementById("batchManifestTranscodeSelectAll").addEventListener("click", function () {
        Array.prototype.slice.call(
          document.querySelectorAll("#batchManifestTranscodeManifests input[data-manifest-path]")
        ).forEach(function (input) { input.checked = true; });
        updateBatchManifestSelectionCount();
      });
    
      document.getElementById("batchManifestTranscodeSelectNone").addEventListener("click", function () {
        Array.prototype.slice.call(
          document.querySelectorAll("#batchManifestTranscodeManifests input[data-manifest-path]")
        ).forEach(function (input) { input.checked = false; });
        updateBatchManifestSelectionCount();
      });
    
      document.getElementById("batchManifestTranscodeRun").addEventListener("click", async function () {
        var button = document.getElementById("batchManifestTranscodeRun");
        var summaryNode = document.getElementById("batchManifestTranscodeSummary");
        var manifests = selectedBatchManifestPaths();
        if (!manifests.length) {
          if (summaryNode) summaryNode.textContent = "Select at least one manifest before starting batch H.264 transcode.";
          return;
        }
        var selectedRows = batchManifestCatalog.reduce(function (total, item) {
          return manifests.indexOf(item.path) >= 0 ? total + Number(item.rollouts || 0) : total;
        }, 0);
        var message = "Transcode canonical video_path files from " + manifests.length + " selected manifest(s)?\n\n"
          + "Selected manifest rows: " + selectedRows
          + "\nDuplicate video_path values are deduplicated."
          + "\nAlready-H.264 videos will be skipped. Existing .orig.mp4 backups will not be overwritten.";
        if (!window.confirm(message)) return;
    
        if (button) button.disabled = true;
        try {
          if (summaryNode) {
            summaryNode.textContent = "Submitting batch H.264 transcode for "
              + manifests.length + " selected manifest(s)…";
          }
          await submitTool("transcode_manifest_videos", { manifest_paths: manifests });
        } catch (error) {
          if (summaryNode) summaryNode.textContent = "Batch H.264 transcode could not start: " + String(error.message || error);
        } finally {
          updateBatchManifestSelectionCount();
        }
      });
    
      document.getElementById("validateBaselinesRun").addEventListener("click", function () { submitTool("validate_baselines", { check_environments: true }).catch(function () {}); });
      document.getElementById("validateVariantsRun").addEventListener("click", function () { submitTool("validate_variants", {}).catch(function () {}); });
      document.getElementById("rebuildManifestRun").addEventListener("click", function () {
        var rebuildButton = document.getElementById("rebuildManifestRun");
        if (rebuildButton) rebuildButton.disabled = true;
        var rawRoots = text("rebuildManifestExtraRoots");
        var extraRoots = rawRoots
          .split(/[\n,]+/)
          .map(function (value) { return value.trim(); })
          .filter(function (value, index, values) { return value && values.indexOf(value) === index; });
        var summaryNode = document.getElementById("rebuildManifestSummary");
        if (summaryNode) {
          summaryNode.textContent = "Submitting manifest rebuild"
            + (extraRoots.length ? " with " + extraRoots.length + " additional scan root(s)…" : " using the default scan roots…");
        }
        var beforeCount = (typeof state !== "undefined" && state && Array.isArray(state.rollouts))
          ? state.rollouts.length
          : null;
        submitTool("rebuild_manifest", { extra_scan_roots: extraRoots })
          .then(function (job) {
            manifestRebuildBefore[job.job_id] = beforeCount;
            if (summaryNode) {
              summaryNode.textContent = "Manifest rebuild running · " + job.job_id;
            }
          })
          .catch(function (error) {
            if (rebuildButton) rebuildButton.disabled = false;
            if (summaryNode) summaryNode.textContent = "Manifest rebuild failed to start: " + String(error.message || error);
          });
      });
    
      async function loadRolloutOptions() {
        var select = document.getElementById("roboSweepRollout");
        try {
          var response = await fetch("/api/rollouts", { cache: "no-store" });
          var payload = await response.json();
          if (!response.ok) throw new Error(payload.error || "Could not load rollouts");
          var rows = payload.rollouts || [];
          select.innerHTML = rows.map(function (row) {
            var label = (row.task_suite || "") + " · task " + row.task_id + " · ep " + row.episode_index + " · " + row.id;
            return '<option value="' + row.id + '">' + label + '</option>';
          }).join("");
        } catch (error) {
          select.innerHTML = '<option value="">' + String(error.message || error) + '</option>';
        }
      }

    refreshToolJobs().then(function () { if (activeToolJobId) pollToolJob(); });
    loadRolloutOptions();
    loadBatchManifestOptions(false);
  }

  return { install: install };
})();
