"use strict";

(function installReviewVideoViews() {
  var controls = document.getElementById("videoViewControls");
  var video = document.getElementById("rolloutVideo");
  if (!controls || !video || !window.state) return;

  var STORAGE_KEY = "lf3r.review.videoView";
  var LABELS = {
    cam_high: "High",
    cam_wrist: "Wrist",
    cam_left_wrist: "Left wrist",
    cam_right_wrist: "Right wrist"
  };
  var PREFERRED_ORDER = [
    "cam_high",
    "cam_wrist",
    "cam_left_wrist",
    "cam_right_wrist"
  ];

  function escapeAttribute(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function escapeText(value) {
    return escapeAttribute(value).replace(/'/g, "&#039;");
  }

  function cameraLabel(key) {
    if (LABELS[key]) return LABELS[key];
    return String(key || "")
      .replace(/^cam_/, "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, function (match) { return match.toUpperCase(); });
  }

  function availableViews(record) {
    var paths = record && record.camera_video_paths;
    if (!paths || typeof paths !== "object") return [];

    var keys = Object.keys(paths).filter(function (key) {
      return typeof paths[key] === "string" && paths[key].trim().length > 0;
    });
    keys.sort(function (left, right) {
      var leftIndex = PREFERRED_ORDER.indexOf(left);
      var rightIndex = PREFERRED_ORDER.indexOf(right);
      if (leftIndex < 0) leftIndex = PREFERRED_ORDER.length;
      if (rightIndex < 0) rightIndex = PREFERRED_ORDER.length;
      return leftIndex - rightIndex || left.localeCompare(right);
    });
    return keys.map(function (key) {
      return { key: key, label: cameraLabel(key), camera: key };
    });
  }

  function storedView() {
    if (state.reviewVideoView) return String(state.reviewVideoView);
    try {
      return sessionStorage.getItem(STORAGE_KEY) || "";
    } catch (_error) {
      return "";
    }
  }

  function chooseView(record, requested) {
    var views = availableViews(record);
    if (!views.length) return "";
    var key = String(requested || storedView() || "");
    var available = views.some(function (view) { return view.key === key; });
    return available ? key : views[0].key;
  }

  function sourceUrl(record, viewKey) {
    return "/api/videos/" + encodeURIComponent(record.id)
      + "?camera=" + encodeURIComponent(viewKey)
      + "&v=review-view-" + Date.now();
  }

  function renderControls(record, activeKey) {
    var views = availableViews(record);
    controls.hidden = views.length <= 1;
    controls.innerHTML = views.length <= 1 ? "" :
      '<span class="video-view-label">View</span><div class="video-view-buttons" role="group" aria-label="Video camera view">'
      + views.map(function (view) {
        var active = view.key === activeKey;
        return '<button type="button" class="video-view-button' + (active ? " is-active" : "")
          + '" data-video-view="' + escapeAttribute(view.key) + '" aria-pressed="'
          + String(active) + '">' + escapeText(view.label) + "</button>";
      }).join("")
      + '</div><span class="video-view-note">' + views.length + " views</span>";
  }

  function restoreFrame(record, viewKey, frame, resumePlayback) {
    if (!record || state.selectedId !== record.id) return;
    if (video.dataset.videoView !== viewKey || video.dataset.rolloutId !== record.id) return;
    var fps = Number(record.fps);
    if (Number.isFinite(fps) && fps > 0) {
      var targetTime = Math.max(0, Number(frame) || 0) / fps;
      try { video.currentTime = targetTime; } catch (_error) {}
    }
    var speed = document.getElementById("speedSelect");
    if (speed) video.playbackRate = Number(speed.value);
    if (resumePlayback) {
      var playPromise = video.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch(function () {});
      }
    }
  }

  function setView(record, requestedKey, options) {
    if (!record || !record.id) return;
    var opts = options || {};
    var viewKey = chooseView(record, requestedKey);
    if (!viewKey) {
      controls.hidden = true;
      controls.innerHTML = "";
      video.pause();
      video.removeAttribute("src");
      video.load();
      return;
    }
    var frame = Number.isFinite(Number(state.currentFrame)) ? Number(state.currentFrame) : 0;
    var resumePlayback = opts.resumePlayback === true && !video.paused && !video.ended;

    state.reviewVideoView = viewKey;
    try { sessionStorage.setItem(STORAGE_KEY, viewKey); } catch (_error) {}
    renderControls(record, viewKey);

    if (!opts.force
        && video.dataset.rolloutId === record.id
        && video.dataset.videoView === viewKey
        && video.getAttribute("src")) {
      return;
    }

    video.pause();
    video.dataset.rolloutId = record.id;
    video.dataset.videoView = viewKey;
    video.addEventListener("loadedmetadata", function () {
      restoreFrame(record, viewKey, frame, resumePlayback);
    }, { once: true });
    video.src = sourceUrl(record, viewKey);
    video.load();
  }

  function applyRecord(record) {
    if (!record) {
      controls.hidden = true;
      controls.innerHTML = "";
      return;
    }
    setView(record, chooseView(record), { resumePlayback: false });
  }

  function currentView() {
    return String(video.dataset.videoView || state.reviewVideoView || "");
  }

  controls.addEventListener("click", function (event) {
    var button = event.target && event.target.closest
      ? event.target.closest("[data-video-view]")
      : null;
    if (!button || !controls.contains(button)) return;
    var record = typeof window.selectedRollout === "function"
      ? window.selectedRollout()
      : (state.rollouts || []).find(function (item) { return item.id === state.selectedId; });
    if (!record) return;
    setView(record, button.dataset.videoView, { resumePlayback: true });
  });

  window.LF3RReviewVideoViews = {
    availableViews: availableViews,
    applyRecord: applyRecord,
    setView: setView,
    currentView: currentView
  };
})();
