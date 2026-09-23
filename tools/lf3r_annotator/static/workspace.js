"use strict";

(function loadWorkspaceWithEnhancements() {
  function loadScript(src, onload, attempt) {
    var retry = Number(attempt || 0);
    var script = document.createElement("script");
    script.src = src;
    script.async = false;
    script.onload = function () {
      if (onload) onload();
    };
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
      console.error("LF3R WebUI could not load " + src + " after retry; continuing with remaining enhancements.");
      if (onload) onload();
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

  // Install the observer safety fuse before any enhancement script creates a
  // MutationObserver or ResizeObserver. It disconnects a runaway callback loop
  // instead of letting one extension freeze the whole browser tab.
  loadScript("/static/frontend-loop-guard.js", function () {
    /* app.js has already defined the review globals. Use a fresh raw-source URL
       whenever a rollout is selected so bytes cached before a manual conversion
       are never reused afterwards. */
    loadScript("/static/raw-video-source.js", function () {
      /* Install the queue cap before the large legacy workspace script. */
      loadStyle("lf3rManifestStyles", "/static/styles-manifest.css");
      loadScript("/static/manifest-support.js", function () {
        loadScript("/static/workspace-legacy.js", function () {
          loadScript("/static/analysis-robo-hop.js");
          loadScript("/static/results-layout.js", function () {
            loadScript("/static/results-axis-scale.js");
            loadScript("/static/results-current-values.js");
            loadStyle("lf3rRunsStyles", "/static/styles-runs.css");
            loadStyle("lf3rRunConfigStyles", "/static/styles-run-config.css");
            loadStyle("lf3rRunsLogStyles", "/static/styles-runs-log.css");
            loadStyle("lf3rBaselineJobStyles", "/static/styles-baseline-jobs.css");
            loadScript("/static/results-run-config.js", function () {
              loadScript("/static/results-run-click-bridge.js", function () {
                loadScript("/static/runs-layout.js", function () {
                  loadScript("/static/runs-semantics.js", function () {
                    loadScript("/static/dataset-scope-ui.js", function () {
                      loadScript("/static/progressive-baseline-results.js", function () {
                        loadScript("/static/baseline-job-filter.js", function () {
                          loadScript("/static/runs-log-ui.js", function () {
                            loadScript("/static/runs-job-control.js", function () {
                              loadScript("/static/runs-submit-fix.js", function () {
                                loadScript("/static/procvlm-mode-ui.js");
                              });
                            });
                          });
                        });
                      });
                    });
                  });
                });
              });
            });
          });
        });
      });
    });
  });
})();
