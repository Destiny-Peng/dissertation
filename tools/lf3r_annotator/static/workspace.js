"use strict";

(function loadWorkspaceWithEnhancements() {
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

  function loadStyle(id, href) {
    if (document.getElementById(id)) return;
    var link = document.createElement("link");
    link.id = id;
    link.rel = "stylesheet";
    link.href = href;
    document.head.appendChild(link);
  }

  loadScript("/static/workspace-legacy.js?v=results-split-20260915", function () {
    loadScript("/static/results-layout.js?v=review-polish-20260916b", function () {
      loadStyle("lf3rRunsStyles", "/static/styles-runs.css?v=runs-console-v4-20260916");
      loadScript("/static/runs-layout.js?v=runs-console-v6-20260916");
    });
  });
})();
