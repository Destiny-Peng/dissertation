"use strict";

(function loadWorkspace() {
  var styles = [
    ["lf3rManifestStyles", "/static/styles-manifest.css"],
    ["lf3rRunsStyles", "/static/styles-runs.css"],
    ["lf3rRunConfigStyles", "/static/styles-run-config.css"],
    ["lf3rRunsLogStyles", "/static/styles-runs-log.css"],
    ["lf3rBaselineJobStyles", "/static/styles-baseline-jobs.css"],
    ["lf3rPolishStyles", "/static/styles-polish.css"]
  ];

  var scripts = [
    "/static/frontend-loop-guard.js",
    "/static/raw-video-source.js",
    "/static/manifest-support.js",
    "/static/analysis/live.js",
    "/static/analysis/snapshot.js",
    "/static/analysis/localization.js",
    "/static/analysis/change-point.js",
    "/static/analysis/event-triggered.js",
    "/static/analysis/runs.js",
    "/static/analysis/dashboard.js",
    "/static/analysis/signals.js",
    "/static/analysis/details.js",
    "/static/workspace/settings.js",
    "/static/workspace/router.js",
    "/static/workspace/events.js",
    "/static/workspace-core.js",
    "/static/analysis-robo-hop.js",
    "/static/results/layout.js",
    "/static/results/run-config.js",
    "/static/runs/layout.js",
    "/static/runs/scope.js",
    "/static/runs/jobs.js",
    "/static/baselines/procvlm.js"
  ];

  function loadStyle(id, href) {
    if (document.getElementById(id)) return;
    var link = document.createElement("link");
    link.id = id;
    link.rel = "stylesheet";
    link.href = href;
    document.head.appendChild(link);
  }

  function loadScript(src, onload, attempt) {
    var retry = Number(attempt || 0);
    var script = document.createElement("script");
    script.src = src;
    script.async = false;
    script.onload = onload;
    script.onerror = function () {
      script.remove();
      if (retry < 1) {
        console.warn("LF3R WebUI retrying failed script " + src);
        var separator = src.indexOf("?") === -1 ? "?" : "&";
        window.setTimeout(function () {
          loadScript(src + separator + "lf3r_retry=" + Date.now(), onload, retry + 1);
        }, 80);
        return;
      }
      console.error(
        "LF3R WebUI could not load " + src +
        " after retry; continuing with remaining modules."
      );
      onload();
    };
    document.head.appendChild(script);
  }

  function loadNext(index) {
    if (index >= scripts.length) return;
    loadScript(scripts[index], function () {
      loadNext(index + 1);
    });
  }

  styles.forEach(function (item) {
    loadStyle(item[0], item[1]);
  });

  // The safety guard is first in the ordered list and is installed before any
  // module that creates MutationObserver or ResizeObserver instances.
  loadNext(0);
})();
