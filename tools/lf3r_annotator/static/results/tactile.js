"use strict";

(function installResultsTactile() {
  var dock = document.getElementById("resultsVideoDock");
  var video = document.getElementById("rolloutVideo");
  if (!dock || !video || !window.state) return;

  var FINGERS = ["thumb", "index", "middle", "ring", "pinky"];
  var LABELS = {
    thumb: "Thumb",
    index: "Index",
    middle: "Middle",
    ring: "Ring",
    pinky: "Pinky"
  };
  var panel = document.createElement("section");
  panel.id = "resultsTactilePanel";
  panel.className = "results-tactile-panel";
  panel.hidden = true;
  panel.innerHTML =
    '<div class="results-tactile-heading">'
    + '<div><p class="eyebrow">TACTILE</p><h3>Right-hand fingertips</h3></div>'
    + '<label class="results-tactile-kind"><span>Image</span><select id="resultsTactileKind">'
    + '<option value="deform" selected>Deform</option><option value="raw">Raw</option>'
    + '</select></label></div>'
    + '<div id="resultsTactileStatus" class="results-tactile-status">Waiting for synchronized frame…</div>'
    + '<div id="resultsTactileGrid" class="results-tactile-grid"></div>';
  dock.appendChild(panel);

  var kindSelect = document.getElementById("resultsTactileKind");
  var statusNode = document.getElementById("resultsTactileStatus");
  var grid = document.getElementById("resultsTactileGrid");
  var lastKey = "";
  var requestSerial = 0;
  var lastPayload = null;

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function selectedRecord() {
    if (typeof window.selectedRollout === "function") return window.selectedRollout();
    return (state.rollouts || []).find(function (row) { return row.id === state.selectedId; }) || null;
  }

  function supportsTactile(record) {
    return Boolean(
      record
      && record.synchronized_frames_path
      && record.tactile_events_path
      && record.tactile_stream_paths
    );
  }

  function imageUrl(rolloutId, finger, eventId, kind) {
    return "/api/tactile/" + encodeURIComponent(rolloutId) + "/image"
      + "?finger=" + encodeURIComponent(finger)
      + "&event_id=" + encodeURIComponent(eventId)
      + "&kind=" + encodeURIComponent(kind);
  }

  function formatF6(value) {
    if (!Array.isArray(value) || !value.length) return '<span class="results-tactile-empty">No f6</span>';
    return value.map(function (item, index) {
      var number = Number(item);
      var text = Number.isFinite(number) ? number.toFixed(3) : "—";
      return '<span><b>f' + (index + 1) + '</b>' + escapeHtml(text) + '</span>';
    }).join("");
  }

  function renderFinger(rolloutId, finger, data, kind) {
    if (!data) {
      return '<article class="results-tactile-card is-missing"><header><strong>'
        + escapeHtml(LABELS[finger]) + '</strong><span>missing</span></header>'
        + '<div class="results-tactile-image results-tactile-placeholder">No sample</div></article>';
    }
    var eventId = data.event_id;
    var kinds = Array.isArray(data.image_kinds) ? data.image_kinds : [];
    var imageKind = kinds.indexOf(kind) >= 0 ? kind : (kinds[0] || "");
    var stale = data.stale === true;
    var invalid = data.valid === false;
    var status = invalid ? "invalid" : (stale ? "stale" : "valid");
    var age = Number(data.age_ms);
    var meta = "event " + (eventId == null ? "—" : eventId)
      + (Number.isFinite(age) ? " · " + age.toFixed(1) + " ms" : "");
    var image = imageKind && eventId != null
      ? '<img class="results-tactile-image" loading="eager" alt="' + escapeHtml(LABELS[finger])
        + ' tactile ' + escapeHtml(imageKind) + '" src="'
        + escapeHtml(imageUrl(rolloutId, finger, eventId, imageKind)) + '">'
      : '<div class="results-tactile-image results-tactile-placeholder">No image</div>';
    return '<article class="results-tactile-card is-' + status + '">'
      + '<header><strong>' + escapeHtml(LABELS[finger]) + '</strong>'
      + '<span class="results-tactile-chip">' + status + '</span></header>'
      + image
      + '<div class="results-tactile-meta">' + escapeHtml(meta) + '</div>'
      + '<div class="results-tactile-f6">' + formatF6(data.f6) + '</div>'
      + '</article>';
  }

  function render(payload) {
    lastPayload = payload;
    var record = selectedRecord();
    if (!record || !payload) return;
    var kind = String(kindSelect.value || "deform");
    var fingers = payload.fingers || {};
    var completeText = payload.complete === false ? "incomplete" : "complete";
    statusNode.textContent =
      String(payload.camera || "camera") + " frame " + payload.requested_video_frame
      + " → sync video frame " + payload.matched_video_frame
      + " · row " + payload.sync_row + " · " + completeText;
    grid.innerHTML = FINGERS.map(function (finger) {
      return renderFinger(record.id, finger, fingers[finger], kind);
    }).join("");
  }

  async function refresh(force) {
    var record = selectedRecord();
    var resultsView = document.body.dataset.view === "results";
    if (!resultsView || !supportsTactile(record)) {
      panel.hidden = true;
      lastKey = "";
      lastPayload = null;
      return;
    }
    panel.hidden = false;

    var camera = String(video.dataset.videoView || state.reviewVideoView || "cam_high");
    var frame = Math.max(0, Math.round(Number(state.currentFrame) || 0));
    var key = record.id + "|" + camera + "|" + frame;
    if (!force && key === lastKey) return;
    lastKey = key;
    var serial = ++requestSerial;
    statusNode.textContent = "Loading tactile frame…";

    try {
      var response = await fetch(
        "/api/tactile/" + encodeURIComponent(record.id) + "/frame"
        + "?camera=" + encodeURIComponent(camera)
        + "&frame=" + encodeURIComponent(frame),
        { cache: "no-store" }
      );
      var body = await response.json();
      if (serial !== requestSerial) return;
      if (!response.ok) throw new Error(body.error || "Tactile frame request failed");
      render(body.tactile);
    } catch (error) {
      if (serial !== requestSerial) return;
      statusNode.textContent = "Tactile unavailable: " + String(error.message || error);
      grid.innerHTML = "";
    }
  }

  kindSelect.addEventListener("change", function () {
    if (lastPayload) render(lastPayload);
  });
  video.addEventListener("loadedmetadata", function () { refresh(true); });
  video.addEventListener("seeked", function () { refresh(true); });
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) refresh(true);
  });

  window.setInterval(function () { refresh(false); }, 120);
  refresh(true);
})();
