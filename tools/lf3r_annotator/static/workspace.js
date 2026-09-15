"use strict";

(function loadWorkspaceWithResultsEnhancements() {
  function loadScript(src, onload) {
    var script = document.createElement("script");
    script.src = src;
    script.async = false;
    script.onload = onload || null;
    script.onerror = function () {
      console.error("LF3R WebUI could not load " + src);
    };
    document.head.appendChild(script);
  }

  loadScript("/static/workspace-legacy.js?v=results-split-20260915", function () {
    loadScript("/static/results-layout.js?v=results-split-20260915");
  });
})();
