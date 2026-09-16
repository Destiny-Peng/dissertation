"use strict";

(function installMultiManifestUi() {
  if (!window.state || typeof window.byId !== "function") return;

  var CARD_LIMIT = 250;
  var retryTimer = null;
  var retryRolloutId = null;
  var retryCount = 0;
  var MAX_VIDEO_RETRIES = 80;

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
    byId("visibleCount").textContent = String(records.length);
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
        + badge(effectiveOutcome(record), effectiveOutcome(record));
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

  function ensureVideoStatus() {
    var shell = document.querySelector(".video-shell");
    if (!shell) return null;
    var node = byId("videoCompatStatus");
    if (node) return node;
    node = document.createElement("div");
    node.id = "videoCompatStatus";
    node.className = "video-compat-status";
    node.hidden = true;
    shell.appendChild(node);
    return node;
  }

  function clearVideoRetry() {
    if (retryTimer) window.clearTimeout(retryTimer);
    retryTimer = null;
    retryRolloutId = null;
    retryCount = 0;
  }

  async function pollVideoCompatibility(rolloutId) {
    if (!rolloutId || retryRolloutId !== rolloutId) return;
    var statusNode = ensureVideoStatus();
    try {
      var response = await fetch(
        "/api/video-compatibility/" + encodeURIComponent(rolloutId),
        { cache: "no-store" }
      );
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Video compatibility status unavailable");
      var status = String(payload.status || "idle");
      if (status === "ready") {
        clearVideoRetry();
        if (statusNode) statusNode.hidden = true;
        if (state.selectedId !== rolloutId) return;
        var video = byId("rolloutVideo");
        video.src = "/api/videos/" + encodeURIComponent(rolloutId)
          + "?v=async-h264-" + Date.now();
        video.load();
        return;
      }
      if (status === "passthrough") {
        clearVideoRetry();
        if (statusNode) {
          statusNode.hidden = false;
          statusNode.textContent = "The source is already H.264, but this browser still reported a decode error.";
        }
        return;
      }
      if (status === "failed") {
        clearVideoRetry();
        if (statusNode) {
          statusNode.hidden = false;
          statusNode.textContent = "Browser-compatible video preparation failed. The original file was left unchanged.";
        }
        return;
      }
      if (statusNode) {
        statusNode.hidden = false;
        statusNode.textContent = status === "transcoding"
          ? "Preparing a browser-compatible H.264 copy in the background…"
          : "Checking video compatibility in the background…";
      }
    } catch (error) {
      if (statusNode) {
        statusNode.hidden = false;
        statusNode.textContent = "Waiting for browser-compatible video…";
      }
    }
    retryCount += 1;
    if (retryCount >= MAX_VIDEO_RETRIES) {
      clearVideoRetry();
      return;
    }
    retryTimer = window.setTimeout(function () {
      pollVideoCompatibility(rolloutId);
    }, 1500);
  }

  function installVideoCompatibilityRetry() {
    var video = byId("rolloutVideo");
    if (!video || video.dataset.asyncCompatInstalled === "true") return;
    video.dataset.asyncCompatInstalled = "true";
    video.addEventListener("loadedmetadata", function () {
      clearVideoRetry();
      var status = ensureVideoStatus();
      if (status) status.hidden = true;
    });
    video.addEventListener("error", function () {
      var rolloutId = state.selectedId;
      if (!rolloutId) return;
      clearVideoRetry();
      retryRolloutId = rolloutId;
      var status = ensureVideoStatus();
      if (status) {
        status.hidden = false;
        status.textContent = "Preparing a browser-compatible H.264 copy in the background…";
      }
      pollVideoCompatibility(rolloutId);
    });
  }

  ensureManifestControls();
  installFilterHooks();
  installVideoCompatibilityRetry();
  loadManifestMetadata();
})();
