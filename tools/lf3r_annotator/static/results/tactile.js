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
    + '<div class="results-tactile-controls">'
    + '<label class="results-tactile-select"><span>Image</span><select id="resultsTactileKind">'
    + '<option value="deform" selected>Deform</option><option value="raw">Raw</option>'
    + '</select></label>'
    + '<label class="results-tactile-select"><span>f6 curve</span><select id="resultsTactileCurveFinger">'
    + FINGERS.map(function (finger) {
      return '<option value="' + finger + '"' + (finger === "index" ? " selected" : "") + '>'
        + LABELS[finger] + '</option>';
    }).join("")
    + '</select></label></div></div>'
    + '<div id="resultsTactileStatus" class="results-tactile-status">Waiting for synchronized frame…</div>'
    + '<div class="results-tactile-curve-panel">'
    + '<div class="results-tactile-curve-heading"><strong>f6 history</strong>'
    + '<div class="results-tactile-curve-legend">'
    + [0, 1, 2, 3, 4, 5].map(function (index) {
      return '<span><i class="trace-' + index + '"></i>f' + (index + 1) + '</span>';
    }).join("")
    + '</div></div>'
    + '<div id="resultsTactileCurve" class="results-tactile-curve">'
    + '<div class="results-tactile-placeholder">Loading f6 history…</div></div></div>'
    + '<div id="resultsTactileGrid" class="results-tactile-grid"></div>';
  dock.appendChild(panel);

  var kindSelect = document.getElementById("resultsTactileKind");
  var curveFingerSelect = document.getElementById("resultsTactileCurveFinger");
  var statusNode = document.getElementById("resultsTactileStatus");
  var curveNode = document.getElementById("resultsTactileCurve");
  var grid = document.getElementById("resultsTactileGrid");
  var lastKey = "";
  var requestSerial = 0;
  var lastPayload = null;
  var lastSeriesKey = "";
  var loadingSeriesKey = "";
  var seriesSerial = 0;
  var lastSeries = null;

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
    if (!Array.isArray(value) || !value.length) {
      return '<span class="results-tactile-empty">No f6</span>';
    }
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
    var cardClasses = (invalid ? " is-invalid" : "") + (stale ? " is-stale" : "");
    var chips = '<span class="results-tactile-chip ' + (invalid ? "is-invalid" : "is-valid") + '">'
      + (invalid ? "invalid" : "valid") + '</span>'
      + (stale ? '<span class="results-tactile-chip is-stale">stale</span>' : "");
    var age = Number(data.age_ms);
    var meta = "event " + (eventId == null ? "—" : eventId)
      + (Number.isFinite(age) ? " · " + age.toFixed(1) + " ms" : "");
    var image = imageKind && eventId != null
      ? '<img class="results-tactile-image" loading="eager" alt="' + escapeHtml(LABELS[finger])
        + ' tactile ' + escapeHtml(imageKind) + '" src="'
        + escapeHtml(imageUrl(rolloutId, finger, eventId, imageKind)) + '">'
      : '<div class="results-tactile-image results-tactile-placeholder">No image</div>';
    return '<article class="results-tactile-card' + cardClasses + '">'
      + '<header><strong>' + escapeHtml(LABELS[finger]) + '</strong>'
      + '<span class="results-tactile-chips">' + chips + '</span></header>'
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

  function formatAxisNumber(value) {
    if (!Number.isFinite(value)) return "—";
    var magnitude = Math.abs(value);
    if (magnitude >= 1000 || (magnitude > 0 && magnitude < 0.001)) {
      return value.toExponential(2);
    }
    return value.toFixed(magnitude >= 10 ? 2 : 3);
  }

  function renderCurve() {
    if (!lastSeries || !lastSeries.fingers) {
      curveNode.innerHTML = '<div class="results-tactile-placeholder">No f6 history.</div>';
      return;
    }
    var finger = String(curveFingerSelect.value || "index");
    var points = Array.isArray(lastSeries.fingers[finger])
      ? lastSeries.fingers[finger] : [];
    if (!points.length) {
      curveNode.innerHTML = '<div class="results-tactile-placeholder">No f6 history for '
        + escapeHtml(LABELS[finger] || finger) + '.</div>';
      return;
    }

    var frameMin = Number(lastSeries.frame_min);
    var frameMax = Number(lastSeries.frame_max);
    if (!Number.isFinite(frameMin)) frameMin = Number(points[0].frame) || 0;
    if (!Number.isFinite(frameMax)) frameMax = Number(points[points.length - 1].frame) || frameMin + 1;
    if (frameMax <= frameMin) frameMax = frameMin + 1;

    var values = [];
    points.forEach(function (point) {
      (Array.isArray(point.f6) ? point.f6.slice(0, 6) : []).forEach(function (value) {
        var number = Number(value);
        if (Number.isFinite(number)) values.push(number);
      });
    });
    if (!values.length) {
      curveNode.innerHTML = '<div class="results-tactile-placeholder">No numeric f6 history.</div>';
      return;
    }

    var yMin = Math.min.apply(Math, values);
    var yMax = Math.max.apply(Math, values);
    if (yMax <= yMin) {
      yMin -= 0.5;
      yMax += 0.5;
    }
    var width = 640;
    var height = 150;
    var xFor = function (frame) {
      return Math.max(0, Math.min(width, (Number(frame) - frameMin) / (frameMax - frameMin) * width));
    };
    var yFor = function (value) {
      return Math.max(0, Math.min(height, height - (Number(value) - yMin) / (yMax - yMin) * height));
    };

    var traces = [0, 1, 2, 3, 4, 5].map(function (component) {
      var coords = points.map(function (point) {
        var value = Array.isArray(point.f6) ? Number(point.f6[component]) : NaN;
        if (!Number.isFinite(value)) return null;
        return xFor(point.frame).toFixed(2) + "," + yFor(value).toFixed(2);
      }).filter(Boolean);
      if (!coords.length) return "";
      return '<polyline class="results-tactile-trace trace-' + component
        + '" points="' + coords.join(" ") + '"></polyline>';
    }).join("");

    var playheadX = xFor(Number(state.currentFrame) || 0);
    curveNode.innerHTML =
      '<div class="results-tactile-axis"><span>' + escapeHtml(formatAxisNumber(yMax))
      + '</span><span>' + escapeHtml(formatAxisNumber(yMin)) + '</span></div>'
      + '<svg class="results-tactile-curve-svg" viewBox="0 0 ' + width + ' ' + height
      + '" preserveAspectRatio="none" role="img" aria-label="'
      + escapeHtml(LABELS[finger] || finger) + ' f6 history">'
      + '<line class="results-tactile-gridline" x1="0" y1="37.5" x2="640" y2="37.5"></line>'
      + '<line class="results-tactile-gridline" x1="0" y1="75" x2="640" y2="75"></line>'
      + '<line class="results-tactile-gridline" x1="0" y1="112.5" x2="640" y2="112.5"></line>'
      + traces
      + '<line class="results-tactile-playhead" data-tactile-playhead x1="' + playheadX.toFixed(2)
      + '" y1="0" x2="' + playheadX.toFixed(2) + '" y2="' + height + '"></line>'
      + '</svg>';
    curveNode.dataset.frameMin = String(frameMin);
    curveNode.dataset.frameMax = String(frameMax);
  }

  function updateCurvePlayhead() {
    var playhead = curveNode.querySelector("[data-tactile-playhead]");
    if (!playhead) return;
    var frameMin = Number(curveNode.dataset.frameMin);
    var frameMax = Number(curveNode.dataset.frameMax);
    if (!Number.isFinite(frameMin) || !Number.isFinite(frameMax) || frameMax <= frameMin) return;
    var frame = Number(state.currentFrame) || 0;
    var x = Math.max(0, Math.min(640, (frame - frameMin) / (frameMax - frameMin) * 640));
    playhead.setAttribute("x1", x.toFixed(2));
    playhead.setAttribute("x2", x.toFixed(2));
  }

  async function ensureSeries(record, camera) {
    var key = record.id + "|" + camera;
    if (lastSeriesKey === key && lastSeries) {
      updateCurvePlayhead();
      return;
    }
    if (loadingSeriesKey === key) return;
    loadingSeriesKey = key;
    var serial = ++seriesSerial;
    curveNode.innerHTML = '<div class="results-tactile-placeholder">Loading f6 history…</div>';
    try {
      var response = await fetch(
        "/api/tactile/" + encodeURIComponent(record.id) + "/series"
        + "?camera=" + encodeURIComponent(camera),
        { cache: "no-store" }
      );
      var body = await response.json();
      if (serial !== seriesSerial) return;
      if (!response.ok) throw new Error(body.error || "Tactile series request failed");
      lastSeriesKey = key;
      lastSeries = body.tactile;
      renderCurve();
    } catch (error) {
      if (serial !== seriesSerial) return;
      lastSeriesKey = key;
      lastSeries = null;
      curveNode.innerHTML = '<div class="results-tactile-placeholder">f6 history unavailable: '
        + escapeHtml(error.message || error) + '</div>';
    } finally {
      if (serial === seriesSerial) loadingSeriesKey = "";
    }
  }

  async function refresh(force) {
    var record = selectedRecord();
    var resultsView = document.body.dataset.view === "results";
    if (!resultsView || !supportsTactile(record)) {
      panel.hidden = true;
      lastKey = "";
      lastPayload = null;
      lastSeriesKey = "";
      loadingSeriesKey = "";
      lastSeries = null;
      return;
    }
    panel.hidden = false;

    var camera = String(video.dataset.videoView || state.reviewVideoView || "cam_high");
    var frame = Math.max(0, Math.round(Number(state.currentFrame) || 0));
    ensureSeries(record, camera);
    updateCurvePlayhead();

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
      updateCurvePlayhead();
    } catch (error) {
      if (serial !== requestSerial) return;
      statusNode.textContent = "Tactile unavailable: " + String(error.message || error);
      grid.innerHTML = "";
    }
  }

  kindSelect.addEventListener("change", function () {
    if (lastPayload) render(lastPayload);
  });
  curveFingerSelect.addEventListener("change", renderCurve);
  video.addEventListener("loadedmetadata", function () { refresh(true); });
  video.addEventListener("seeked", function () { refresh(true); });
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) refresh(true);
  });

  window.setInterval(function () { refresh(false); }, 120);
  refresh(true);
})();
