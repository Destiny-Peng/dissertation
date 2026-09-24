"use strict";

(function installRawVideoSource() {
  if (!window.state || typeof window.selectRollout !== "function") return;

  var originalSelectRollout = window.selectRollout;
  var wrappedSelectRollout = function (id) {
    originalSelectRollout(id);
    var video = document.getElementById("rolloutVideo");
    if (!video || state.selectedId !== id) return;
    var record = (state.rollouts || []).find(function (item) { return item.id === id; }) || null;
    if (record && window.LF3RReviewVideoViews
        && typeof window.LF3RReviewVideoViews.applyRecord === "function") {
      window.LF3RReviewVideoViews.applyRecord(record);
      return;
    }
    var src = "/api/videos/" + encodeURIComponent(id) + "?v=raw-source-" + Date.now();
    video.src = src;
    video.load();
    var speed = document.getElementById("speedSelect");
    if (speed) video.playbackRate = Number(speed.value);
  };

  window.selectRollout = wrappedSelectRollout;
  try { selectRollout = wrappedSelectRollout; } catch (_) {}
})();
