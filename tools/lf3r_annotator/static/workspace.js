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

  // Install the observer safety fuse before any enhancement script creates a
  // MutationObserver or ResizeObserver. It disconnects a runaway callback loop
  // instead of letting one extension freeze the whole browser tab.
  loadScript("/static/frontend-loop-guard.js?v=frontend-loop-guard-v1-20260916", function () {
    /* app.js has already defined the review globals. Use a fresh raw-source URL
       whenever a rollout is selected so bytes cached before a manual conversion
       are never reused afterwards. */
    loadScript("/static/raw-video-source.js?v=raw-video-v2-20260916", function () {
      /* Install the queue cap before the large legacy workspace script. */
      loadStyle("lf3rManifestStyles", "/static/styles-manifest.css?v=multi-manifest-v3-20260916");
      loadScript("/static/manifest-support-v2.js?v=manual-transcode-v1-20260916", function () {
        loadScript("/static/workspace-legacy.js?v=analysis-job-ownership-v2-20260921", function () {
          loadScript("/static/analysis-robo-hop.js?v=isolated-run-catalog-v6-20260921", function () {
            loadScript("/static/analysis-robo-bilstm.js?v=latest-per-rollout-v5-20260921", function () {
              loadScript("/static/analysis-robo-label-loss.js?v=weighted-multievent-v3-20260921");
            });
          });
          loadScript("/static/results-layout.js?v=review-polish-20260916b", function () {
            loadScript("/static/results-axis-scale.js?v=procvlm-progress-0-100-v2-20260916");
            loadScript("/static/results-current-values.js?v=current-numeric-values-v2-20260916");
            loadStyle("lf3rRunsStyles", "/static/styles-runs.css?v=runs-console-v4-20260916");
            loadStyle("lf3rRunConfigStyles", "/static/styles-run-config.css?v=single-baseline-config-v2-20260916");
            loadStyle("lf3rRunsLogStyles", "/static/styles-runs-log.css?v=runs-log-v2-20260916");
            loadStyle("lf3rBaselineJobStyles", "/static/styles-baseline-jobs.css?v=baseline-job-filter-v1-20260916");
            loadScript("/static/results-run-config-v3.js?v=robo-task-goal-v1-20260918", function () {
              loadScript("/static/results-run-click-bridge.js?v=results-config-click-v1-20260916", function () {
                loadScript("/static/runs-layout.js?v=runs-console-v9-20260916", function () {
                  loadScript("/static/runs-semantics.js?v=runs-semantics-v1-20260916", function () {
                    loadScript("/static/dataset-scope-ui.js?v=dynamic-suite-v2-20260916", function () {
                      loadScript("/static/progressive-baseline-results.js?v=progressive-baseline-results-v1-20260917", function () {
                        loadScript("/static/baseline-job-filter.js?v=baseline-job-filter-v1-20260916", function () {
                          loadScript("/static/runs-log-ui.js?v=runs-log-v3-20260917", function () {
                            loadScript("/static/runs-job-control.js?v=baseline-cancel-v2-20260916", function () {
                              loadScript("/static/runs-submit-fix.js?v=batch-submit-v1-20260916", function () {
                                loadScript("/static/procvlm-mode-ui.js?v=procvlm-lora-mode-v4-20260916");
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
