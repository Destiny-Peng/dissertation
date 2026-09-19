"use strict";

(function installMultiManifestUi() {
  if (!window.state || typeof window.byId !== "function") return;

  var CARD_LIMIT = 250;
  var transcodePollTimer = null;
  var transcodeJobId = null;
  var transcodeRolloutId = null;

  state.manifests = state.manifests || [];
  state.manifestFilter = state.manifestFilter || "all";

  function provenanceClass(record) {
    if (typeof window.isControlled === "function" && window.isControlled(record)) return "controlled";
    if (record && (record.source_kind === "real_robot" || record.analysis_partition === "real_robot_analysis")) {
      return "external";
    }
    if (record && record.manifest_primary === false) return "external";
    return "natural";
  }

  function provenanceLabel(record) {
    if (typeof window.isControlled === "function" && window.isControlled(record)) return "controlled";
    if (record && (record.source_kind === "real_robot" || record.analysis_partition === "real_robot_analysis")) {
      return "real robot";
    }
    if (record && record.manifest_primary === false) return "secondary";
    return "natural";
  }

  function ensureManifestControls() {
    var origin = byId("originFilter");
    if (origin && !origin.querySelector('option[value="real_robot"]')) {
      var option = document.createElement("option");
      option.value = "real_robot";
      option.textContent = "Real-robot only";
      origin.appendChild(option);
    }

    var grid = origin && origin.closest(".filter-grid");
    if (grid && !byId("manifestFilter")) {
      var label = document.createElement("label");
      label.innerHTML = '<span>Manifest</span><select id="manifestFilter"><option value="all">All manifests</option></select>';
      var outcome = byId("outcomeFilter");
      var outcomeLabel = outcome && outcome.closest("label");
      grid.insertBefore(label, outcomeLabel || null);
    }

    var key = document.querySelector(".partition-key");
    if (key && !key.querySelector(".key-dot.external")) {
      var item = document.createElement("span");
      item.innerHTML = '<i class="key-dot external"></i> Other manifest';
      key.appendChild(item);
    }
  }

  function populateManifestFilter(manifests) {
    ensureManifestControls();
    var select = byId("manifestFilter");
    if (!select) return;
    var current = state.manifestFilter || select.value || "all";
    var options = ['<option value="all">All manifests</option>'];
    (manifests || []).forEach(function (manifest) {
      var path = String(manifest.path || "");
      if (!path) return;
      var label = manifest.label && manifest.label !== path
        ? String(manifest.label) + " · " + path
        : path;
      if (manifest.rollouts != null) label += " (" + manifest.rollouts + ")";
      options.push('<option value="' + escapeHtml(path) + '">' + escapeHtml(label) + "</option>");
    });
    select.innerHTML = options.join("");
    var valid = current === "all" || (manifests || []).some(function (manifest) {
      return String(manifest.path) === current;
    });
    select.value = valid ? current : "all";
    state.manifestFilter = select.value;
  }

  function currentManifestFilter() {
    var select = byId("manifestFilter");
    return select ? (select.value || "all") : "all";
  }

  function allFilters(record) {
    var query = (byId("searchInput") && byId("searchInput").value || "").trim().toLowerCase();
    var origin = byId("originFilter") ? byId("originFilter").value : "all";
    var manifest = currentManifestFilter();
    var outcome = byId("outcomeFilter") ? byId("outcomeFilter").value : "all";
    var review = byId("reviewFilter") ? byId("reviewFilter").value : "all";
    var haystack = [
      record.id,
      record.task_description,
      record.task_suite,
      record.task_id,
      record.manifest_source,
      record.manifest_label
    ].join(" ").toLowerCase();
    return (!query || haystack.indexOf(query) !== -1)
      && (origin === "all" || record.source_kind === origin)
      && (manifest === "all" || record.manifest_source === manifest)
      && (outcome === "all" || effectiveOutcome(record) === outcome)
      && (review === "all" || record.annotation_status === review);
  }

  function renderManifestRolloutList() {
    var container = byId("rolloutList");
    if (!container) return;
    var records = state.filtered || [];
    var visible = byId("visibleCount");
    if (visible) visible.textContent = String(records.length);
    if (!records.length) {
      container.innerHTML = '<div class="empty-state"><p>No rollouts match these filters.</p></div>';
      return;
    }

    var shown = records.slice(0, CARD_LIMIT);
    var html = shown.map(function (record) {
      var originClass = provenanceClass(record);
      var selectedClass = record.id === state.selectedId ? " active" : "";
      var title = record.task_description || (record.task_suite + " task " + record.task_id);
      var sourceLabel = record.manifest_label || record.manifest_source || "manifest";
      return '<button class="rollout-card ' + originClass + selectedClass
        + '" data-rollout-id="' + escapeHtml(record.id) + '" type="button">'
        + '<div class="badge-row">'
        + badge(provenanceLabel(record), originClass)
        + badge(effectiveOutcome(record), effectiveOutcome(record))
        + badge(record.annotation_status, record.annotation_status)
        + (record.multiview_video_path ? badge("3-view", "natural") : "")
        + "</div>"
        + '<div class="card-title">' + escapeHtml(title) + "</div>"
        + '<div class="card-footer"><span>' + escapeHtml(record.task_suite)
        + " · task " + escapeHtml(record.task_id)
        + " · " + escapeHtml(sourceLabel)
        + '</span><span>' + escapeHtml(record.total_frames) + "f</span></div>"
        + "</button>";
    }).join("");

    if (records.length > shown.length) {
      html += '<div class="manifest-render-cap">Showing ' + shown.length + " of " + records.length
        + " matching rollouts. Narrow the manifest/search filters to show another subset.</div>";
    }
    container.innerHTML = html;
    container.querySelectorAll("[data-rollout-id]").forEach(function (button) {
      button.addEventListener("click", function () {
        maybeSelectRollout(button.dataset.rolloutId);
      });
    });
  }

  function applyManifestFilters() {
    state.manifestFilter = currentManifestFilter();
    state.filtered = (state.rollouts || []).filter(allFilters);
    renderManifestRolloutList();
  }

  window.renderRolloutList = renderManifestRolloutList;
  try { renderRolloutList = renderManifestRolloutList; } catch (_) {}

  window.applyFilters = applyManifestFilters;
  try { applyFilters = applyManifestFilters; } catch (_) {}

  var originalSelectRollout = window.selectRollout;
  if (typeof originalSelectRollout === "function") {
    var wrappedSelectRollout = function (id) {
      originalSelectRollout(id);
      var record = (state.rollouts || []).find(function (item) { return item.id === id; });
      if (!record) return;
      var host = byId("recordBadges");
      if (!host) return;
      var originClass = provenanceClass(record);
      host.innerHTML = badge(
        provenanceLabel(record) === "natural" ? "natural policy" : provenanceLabel(record),
        originClass
      )
        + badge(record.analysis_partition, originClass)
        + badge(effectiveOutcome(record), effectiveOutcome(record))
        + (record.multiview_video_path ? badge("3-view", "natural") : "");
    };
    window.selectRollout = wrappedSelectRollout;
    try { selectRollout = wrappedSelectRollout; } catch (_) {}
  }

  function installFilterHooks() {
    ensureManifestControls();
    ["searchInput", "originFilter", "outcomeFilter", "reviewFilter", "manifestFilter"].forEach(function (id) {
      var node = byId(id);
      if (!node || node.dataset.manifestFilterHook === "true") return;
      node.dataset.manifestFilterHook = "true";
      node.addEventListener(id === "searchInput" ? "input" : "change", function () {
        window.setTimeout(applyManifestFilters, 0);
      });
    });
  }

  function updateDatasetStatus() {
    var status = byId("datasetStatus");
    if (!status) return;
    var count = state.manifests.length || 1;
    status.textContent = "Dataset online · " + (state.rollouts || []).length
      + " rollouts · " + count + " manifest" + (count === 1 ? "" : "s");
  }

  async function loadManifestMetadata() {
    try {
      var response = await fetch("/api/manifests", { cache: "no-store" });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not load manifest metadata");
      state.manifests = payload.manifests || [];
      populateManifestFilter(state.manifests);
      installFilterHooks();
      updateDatasetStatus();
      applyManifestFilters();
      if (state.selectedId && typeof window.selectRollout === "function") {
        var selected = state.selectedId;
        var record = (state.rollouts || []).find(function (item) { return item.id === selected; });
        if (record) {
          var host = byId("recordBadges");
          var originClass = provenanceClass(record);
          if (host) {
            host.innerHTML = badge(
              provenanceLabel(record) === "natural" ? "natural policy" : provenanceLabel(record),
              originClass
            )
              + badge(record.analysis_partition, originClass)
              + badge(effectiveOutcome(record), effectiveOutcome(record));
          }
        }
      }
    } catch (error) {
      console.error("LF3R manifest metadata unavailable", error);
    }
  }

  function selectedRecord() {
    if (typeof window.selectedRollout === "function") return window.selectedRollout();
    return (state.rollouts || []).find(function (record) { return record.id === state.selectedId; }) || null;
  }

  function transcodeBackupName(videoPath) {
    return String(videoPath || "").replace(/\.mp4$/i, ".orig.mp4");
  }

  function setTranscodeUi(message, busy) {
    var button = byId("manualVideoTranscode");
    var status = byId("manualVideoTranscodeStatus");
    if (button) {
      button.disabled = Boolean(busy);
      button.textContent = busy ? "Transcoding…" : "Transcode H.264";
    }
    if (status) {
      status.hidden = !message;
      status.textContent = message || "";
    }
  }

  function reloadSelectedRawVideo(rolloutId) {
    if (!rolloutId || state.selectedId !== rolloutId) return;
    var video = byId("rolloutVideo");
    if (!video) return;
    video.src = "/api/videos/" + encodeURIComponent(rolloutId)
      + "?v=raw-source-" + Date.now();
    video.load();
    var speed = byId("speedSelect");
    if (speed) video.playbackRate = Number(speed.value);
  }

  function clearTranscodePoll() {
    if (transcodePollTimer) window.clearTimeout(transcodePollTimer);
    transcodePollTimer = null;
  }

  async function pollTranscodeJob() {
    if (!transcodeJobId) return;
    try {
      var response = await fetch("/api/tool-jobs/" + encodeURIComponent(transcodeJobId), { cache: "no-store" });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not read transcode job");
      var job = payload.job || {};
      if (job.status === "complete") {
        var finishedRollout = transcodeRolloutId;
        clearTranscodePoll();
        transcodeJobId = null;
        transcodeRolloutId = null;
        setTranscodeUi("H.264 complete · original kept as .orig.mp4", false);
        reloadSelectedRawVideo(finishedRollout);
        return;
      }
      if (["failed", "cancelled", "memory_blocked"].indexOf(job.status) !== -1) {
        var failedRollout = transcodeRolloutId;
        clearTranscodePoll();
        transcodeJobId = null;
        transcodeRolloutId = null;
        setTranscodeUi("Transcode failed · original file was restored; see Tool activity log", false);
        reloadSelectedRawVideo(failedRollout);
        return;
      }
    } catch (error) {
      setTranscodeUi("Waiting for transcode job status…", true);
    }
    transcodePollTimer = window.setTimeout(pollTranscodeJob, 1000);
  }

  async function startManualTranscode() {
    if (transcodeJobId) return;
    var record = selectedRecord();
    if (!record || !record.video_path) return;
    var playbackPath = record.multiview_video_path || record.video_path;
    var backup = transcodeBackupName(playbackPath);
    var confirmed = window.confirm(
      "Convert this selected video to browser-compatible H.264?\n\n"
      + "The current file will be renamed to:\n" + backup + "\n\n"
      + "The H.264 output will use the original .mp4 path."
    );
    if (!confirmed) return;

    var rolloutId = record.id;
    var video = byId("rolloutVideo");
    if (video) {
      video.pause();
      video.removeAttribute("src");
      video.load();
    }
    setTranscodeUi("Submitting persistent ffmpeg job…", true);

    try {
      var response = await fetch("/api/tools/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "transcode_video",
          options: { video_path: playbackPath }
        })
      });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not start transcode");
      transcodeJobId = payload.job && payload.job.job_id;
      transcodeRolloutId = rolloutId;
      if (!transcodeJobId) throw new Error("Transcode job id is missing");
      setTranscodeUi("Transcoding in background · original will remain as .orig.mp4", true);
      pollTranscodeJob();
    } catch (error) {
      transcodeJobId = null;
      transcodeRolloutId = null;
      setTranscodeUi("Transcode could not start: " + error.message, false);
      reloadSelectedRawVideo(rolloutId);
    }
  }

  function installManualTranscodeControl() {
    var transport = document.querySelector(".transport");
    if (!transport || byId("manualVideoTranscode")) return;
    var button = document.createElement("button");
    button.id = "manualVideoTranscode";
    button.type = "button";
    button.className = "manual-video-transcode";
    button.textContent = "Transcode H.264";
    button.title = "Rename the selected .mp4 to .orig.mp4 and encode an H.264 replacement at the original path";
    button.addEventListener("click", startManualTranscode);
    transport.appendChild(button);

    var status = document.createElement("span");
    status.id = "manualVideoTranscodeStatus";
    status.className = "manual-video-transcode-status";
    status.hidden = true;
    transport.appendChild(status);
  }

  ensureManifestControls();
  installFilterHooks();
  installManualTranscodeControl();
  loadManifestMetadata();
})();
