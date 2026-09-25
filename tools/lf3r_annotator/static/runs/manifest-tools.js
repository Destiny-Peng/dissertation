"use strict";

window.LF3RManifestTools = (function createManifestTools() {
  function install(client, options) {
    options = options || {};
    var refreshRolloutOptions = typeof options.refreshRolloutOptions === "function"
      ? options.refreshRolloutOptions
      : function () { return Promise.resolve(); };

    function text(id) {
      var node = document.getElementById(id);
      return node ? String(node.value || "").trim() : "";
    }

    var manifestRebuildBefore = {};
    var manifestRefreshHandled = {};
    var batchManifestCatalog = [];

    var manifestExtraRootsNode = document.getElementById("rebuildManifestExtraRoots");
    if (manifestExtraRootsNode) {
      try {
        manifestExtraRootsNode.value =
          localStorage.getItem("lf3r.runs.extraManifestScanRoots") || "";
      } catch (_) {}
      manifestExtraRootsNode.addEventListener("input", function () {
        try {
          localStorage.setItem(
            "lf3r.runs.extraManifestScanRoots",
            manifestExtraRootsNode.value
          );
        } catch (_) {}
      });
    }

    function selectedBatchManifestPaths() {
      return Array.prototype.slice.call(
        document.querySelectorAll(
          "#batchManifestTranscodeManifests input[data-manifest-path]:checked"
        )
      ).map(function (input) { return input.dataset.manifestPath; });
    }

    function updateBatchManifestSelectionCount() {
      var selected = selectedBatchManifestPaths();
      var selectedRows = batchManifestCatalog.reduce(function (total, item) {
        return selected.indexOf(item.path) >= 0
          ? total + Number(item.rollouts || 0)
          : total;
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
        input.disabled = item.valid === false || item.exists === false;
        input.checked = !input.disabled && (preserveSelection
          ? previous.indexOf(String(item.path)) >= 0
          : true);
        input.addEventListener("change", updateBatchManifestSelectionCount);

        var copy = document.createElement("span");
        var title = document.createElement("strong");
        title.textContent = String(item.label || item.path);
        var meta = document.createElement("small");
        meta.textContent = String(item.path)
          + " · " + Number(item.rollouts || 0) + " rollout(s)"
          + (item.primary ? " · primary" : "")
          + (item.valid === false ? " · invalid: " + String(item.error || "unavailable") : "");
        copy.appendChild(title);
        copy.appendChild(meta);
        label.appendChild(input);
        label.appendChild(copy);
        host.appendChild(label);
      });
      updateBatchManifestSelectionCount();
    }

    async function loadBatchManifestOptions(preserveSelection, forceNetwork) {
      var host = document.getElementById("batchManifestTranscodeManifests");
      var cached = (typeof state !== "undefined" && state && Array.isArray(state.manifests))
        ? state.manifests
        : [];
      if (!forceNetwork && cached.length) {
        renderBatchManifestOptions(cached, Boolean(preserveSelection));
        return;
      }
      try {
        var response = await fetch("/api/manifests", { cache: "no-store" });
        var payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Could not read loaded manifests");
        }
        var manifests = payload.manifests || [];
        if (typeof state !== "undefined" && state) state.manifests = manifests;
        renderBatchManifestOptions(
          manifests,
          Boolean(preserveSelection)
        );
      } catch (error) {
        batchManifestCatalog = [];
        if (host) {
          host.innerHTML = "";
          var empty = document.createElement("div");
          empty.className = "runs-manifest-choice-empty error";
          empty.textContent = "Manifest options unavailable: "
            + String(error.message || error);
          host.appendChild(empty);
        }
        updateBatchManifestSelectionCount();
      }
    }

    document.getElementById("batchManifestTranscodeSelectAll")
      .addEventListener("click", function () {
        Array.prototype.slice.call(
          document.querySelectorAll(
            "#batchManifestTranscodeManifests input[data-manifest-path]"
          )
        ).forEach(function (input) { input.checked = true; });
        updateBatchManifestSelectionCount();
      });

    document.getElementById("batchManifestTranscodeSelectNone")
      .addEventListener("click", function () {
        Array.prototype.slice.call(
          document.querySelectorAll(
            "#batchManifestTranscodeManifests input[data-manifest-path]"
          )
        ).forEach(function (input) { input.checked = false; });
        updateBatchManifestSelectionCount();
      });

    document.getElementById("batchManifestTranscodeRun")
      .addEventListener("click", async function () {
        var button = document.getElementById("batchManifestTranscodeRun");
        var summaryNode = document.getElementById("batchManifestTranscodeSummary");
        var manifests = selectedBatchManifestPaths();
        if (!manifests.length) {
          if (summaryNode) {
            summaryNode.textContent =
              "Select at least one manifest before starting batch H.264 transcode.";
          }
          return;
        }

        var selectedRows = batchManifestCatalog.reduce(function (total, item) {
          return manifests.indexOf(item.path) >= 0
            ? total + Number(item.rollouts || 0)
            : total;
        }, 0);
        var message = "Transcode all camera_video_paths files from "
          + manifests.length + " selected manifest(s)?\n\n"
          + "Selected manifest rows: " + selectedRows
          + "\nDuplicate camera video paths are deduplicated."
          + "\nAlready-H.264 videos will be skipped. Existing .orig.mp4 backups will not be overwritten.";
        if (!window.confirm(message)) return;

        if (button) button.disabled = true;
        try {
          if (summaryNode) {
            summaryNode.textContent =
              "Submitting batch H.264 transcode for "
              + manifests.length + " selected manifest(s)…";
          }
          await client.submit(
            "transcode_manifest_videos",
            { manifest_paths: manifests }
          );
        } catch (error) {
          if (summaryNode) {
            summaryNode.textContent =
              "Batch H.264 transcode could not start: "
              + String(error.message || error);
          }
        } finally {
          updateBatchManifestSelectionCount();
        }
      });

    document.getElementById("rebuildManifestRun")
      .addEventListener("click", function () {
        var rebuildButton = document.getElementById("rebuildManifestRun");
        if (rebuildButton) rebuildButton.disabled = true;
        var rawRoots = text("rebuildManifestExtraRoots");
        var extraRoots = rawRoots
          .split(/[\n,]+/)
          .map(function (value) { return value.trim(); })
          .filter(function (value, index, values) {
            return value && values.indexOf(value) === index;
          });
        var summaryNode = document.getElementById("rebuildManifestSummary");
        if (summaryNode) {
          summaryNode.textContent = "Submitting manifest rebuild"
            + (extraRoots.length
              ? " with " + extraRoots.length + " additional scan root(s)…"
              : " using the default scan roots…");
        }
        var beforeCount = (
          typeof state !== "undefined"
          && state
          && Array.isArray(state.rollouts)
        ) ? state.rollouts.length : null;

        client.submit("rebuild_manifest", { extra_scan_roots: extraRoots })
          .then(function (job) {
            manifestRebuildBefore[job.job_id] = beforeCount;
            if (summaryNode) {
              summaryNode.textContent =
                "Manifest rebuild running · " + job.job_id;
            }
          })
          .catch(function (error) {
            if (rebuildButton) rebuildButton.disabled = false;
            if (summaryNode) {
              summaryNode.textContent =
                "Manifest rebuild failed to start: "
                + String(error.message || error);
            }
          });
      });

    function onJobsUpdated(jobs) {
      var rebuildButton = document.getElementById("rebuildManifestRun");
      if (!rebuildButton) return;
      rebuildButton.disabled = (jobs || []).some(function (job) {
        return job.action === "rebuild_manifest"
          && (job.status === "queued" || job.status === "running");
      });
    }

    async function onJobUpdate(job, logText) {
      if (job.action === "transcode_manifest_videos") {
        var batchSummary = document.getElementById("batchManifestTranscodeSummary");
        if (batchSummary) {
          var match = String(logText || "").match(
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
            batchSummary.textContent = "Batch H.264 transcode "
              + job.status
              + "… see Tool activity for per-video progress.";
          } else if (job.status === "failed") {
            batchSummary.textContent = "Batch H.264 transcode failed: "
              + (job.error || "see Tool activity log");
          }
        }
      }

      if (job.action === "rebuild_manifest"
          && job.status === "failed"
          && !manifestRefreshHandled[job.job_id]) {
        manifestRefreshHandled[job.job_id] = true;
        var failedSummary = document.getElementById("rebuildManifestSummary");
        if (failedSummary) {
          failedSummary.textContent = "Manifest rebuild failed: "
            + (job.error || "see project-tool log");
        }
      }

      if (job.action === "rebuild_manifest"
          && job.status === "complete"
          && !manifestRefreshHandled[job.job_id]) {
        manifestRefreshHandled[job.job_id] = true;
        var beforeCount = Object.prototype.hasOwnProperty.call(
          manifestRebuildBefore,
          job.job_id
        ) ? manifestRebuildBefore[job.job_id] : null;
        var summaryNode = document.getElementById("rebuildManifestSummary");
        if (summaryNode) {
          summaryNode.textContent =
            "Manifest rebuilt; refreshing rollout catalog…";
        }
        try {
          var preferredId = (
            typeof state !== "undefined" && state
          ) ? state.selectedId : null;
          if (typeof loadRollouts === "function") {
            await loadRollouts(preferredId);
          }
          await refreshRolloutOptions();
          await loadBatchManifestOptions(true, true);
          var afterCount = (
            typeof state !== "undefined"
            && state
            && Array.isArray(state.rollouts)
          ) ? state.rollouts.length : null;

          if (summaryNode) {
            if (afterCount == null) {
              summaryNode.textContent =
                "Manifest rebuilt successfully. Rollout catalog refresh completed.";
            } else if (beforeCount == null) {
              summaryNode.textContent =
                "Manifest rebuilt successfully · "
                + afterCount + " rollout(s) loaded.";
            } else {
              var delta = afterCount - beforeCount;
              var deltaText = delta > 0
                ? " · +" + delta + " new"
                : (delta < 0
                  ? " · " + delta + " net"
                  : " · no net count change");
              summaryNode.textContent =
                "Manifest rebuilt successfully · "
                + afterCount + " rollout(s)" + deltaText + ".";
            }
          }
        } catch (refreshError) {
          if (summaryNode) {
            summaryNode.textContent =
              "Manifest rebuilt, but automatic catalog refresh failed: "
              + String(refreshError.message || refreshError);
          }
        }
      }
    }

    loadBatchManifestOptions(false);
    return {
      onJobUpdate: onJobUpdate,
      onJobsUpdated: onJobsUpdated,
      loadBatchManifestOptions: loadBatchManifestOptions
    };
  }

  return { install: install };
})();
