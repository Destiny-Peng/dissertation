"use strict";

(function loadWorkspace() {
  var styles = [
    ["lf3rManifestStyles", "/static/styles-manifest.css"],
    ["lf3rRunsStyles", "/static/styles-runs.css"],
    ["lf3rRunConfigStyles", "/static/styles-run-config.css"],
    ["lf3rRunsLogStyles", "/static/styles-runs-log.css"],
    ["lf3rBaselineJobStyles", "/static/styles-baseline-jobs.css"],
    ["lf3rPolishStyles", "/static/styles-polish.css"],
    ["lf3rToolsStyles", "/static/styles-tools.css"]
  ];

  var guardScript = "/static/frontend-loop-guard.js";

  // These modules either declare functions/controllers or install independent
  // DOM enhancements. They do not depend on workspace-core.js and can be
  // downloaded/executed in parallel after the loop guard is active.
  var baseScripts = [
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
    "/static/results/layout.js",
    "/static/results/run-config.js",
    "/static/runs/tools-layout.js",
    "/static/runs/gpu.js",
    "/static/runs/project-tool-client.js",
    "/static/runs/project-tool-actions.js",
    "/static/runs/manifest-tools.js",
    "/static/runs/project-tools.js",
    "/static/runs/scope.js",
    "/static/runs/jobs.js",
    "/static/baselines/procvlm.js"
  ];

  // workspace-core.js performs bootstrap calls immediately, so it waits until
  // all declarations it may call are available.
  var workspaceCoreScript = "/static/workspace-core.js";

  // These modules have top-level initialization that expects either
  // workspaceState or the complete Runs controller graph.
  var postCoreScripts = [
    "/static/analysis-robo-hop.js",
    "/static/runs/layout.js"
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
    script.async = true;
    script.onload = function () {
      if (onload) onload();
    };
    script.onerror = function () {
      script.remove();
      if (retry < 1) {
        console.warn("LF3R WebUI retrying failed script " + src);
        var separator = src.indexOf("?") === -1 ? "?" : "&";
        window.setTimeout(function () {
          loadScript(
            src + separator + "lf3r_retry=" + Date.now(),
            onload,
            retry + 1
          );
        }, 80);
        return;
      }
      console.error(
        "LF3R WebUI could not load " + src
        + " after retry; continuing with remaining modules."
      );
      if (onload) onload();
    };
    document.head.appendChild(script);
  }

  function loadGroup(group, onload) {
    if (!group.length) {
      if (onload) onload();
      return;
    }
    var remaining = group.length;
    group.forEach(function (src) {
      loadScript(src, function () {
        remaining -= 1;
        if (remaining === 0 && onload) onload();
      });
    });
  }

  styles.forEach(function (item) {
    loadStyle(item[0], item[1]);
  });

  // The safety guard is always installed before any enhancement module.
  // Everything inside a phase may load in parallel; phase boundaries encode
  // the actual dependency graph instead of serializing every module.
  loadScript(guardScript, function () {
    loadGroup(baseScripts, function () {
      loadScript(workspaceCoreScript, function () {
        loadGroup(postCoreScripts);
      });
    });
  });
})();
