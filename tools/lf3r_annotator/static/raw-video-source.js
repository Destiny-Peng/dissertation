"use strict";

(function installRawVideoSource() {
  if (!window.state || typeof window.selectRollout !== "function") return;

  var originalSelectRollout = window.selectRollout;
  var wrappedSelectRollout = function (id) {
    originalSelectRollout(id);
    var video = document.getElementById("rolloutVideo");
    if (!video || state.selectedId !== id) return;
    var src = "/api/videos/" + encodeURIComponent(id) + "?v=raw-source-v1";
    if (video.getAttribute("src") !== src) {
      video.src = src;
      video.load();
      video.playbackRate = Number(document.getElementById("speedSelect").value);
    }
  };

  window.selectRollout = wrappedSelectRollout;
  try { selectRollout = wrappedSelectRollout; } catch (_) {}
})();
