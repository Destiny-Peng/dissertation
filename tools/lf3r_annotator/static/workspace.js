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

  /* app.js has already defined the review globals. Use a fresh raw-source URL
     whenever a rollout is selected so bytes cached before a manual conversion
     are never reused afterwards. */
  loadScript("/static/raw-video-source.js?v=raw-video-v2-20260916", function () {
    /* Install the queue cap before the large legacy workspace script. */
    loadStyle("lf3rManifestStyles", "/static/styles-manifest.css?v=multi-manifest-v3-20260916");
    loadScript("/static/manifest-support-v2.js?v=manual-transcode-v1-20260916", function () {
      loadScript("/static/workspace-legacy.js?v=results-split-20260915", function () {
        loadScript("/static/results-layout.js?v=review-polish-20260916b", function () {
          loadScript("/static/results-axis-scale.js?v=procvlm-progress-0-100-v2-20260916");
          loadScript("/static/results-current-values.js?v=current-numeric-values-v2-20260916");
          loadStyle("lf3rRunsStyles", "/static/styles-runs.css?v=runs-console-v4-20260916");
          loadStyle("lf3rRunConfigStyles", "/static/styles-run-config.css?v=single-baseline-config-v2-20260916");
          loadStyle("lf3rRunsLogStyles", "/static/styles-runs-log.css?v=runs-log-v2-20260916");
          loadStyle("lf3rBaselineJobStyles", "/static/styles-baseline-jobs.css?v=baseline-job-filter-v1-20260916");
          loadScript("/static/results-run-config-v2.js?v=single-baseline-config-v2-20260916", function () {
            loadScript("/static/runs-layout.js?v=runs-console-v9-20260916", function () {
              loadScript("/static/runs-semantics.js?v=runs-semantics-v1-20260916", function () {
                loadScript("/static/dataset-scope-ui.js?v=dynamic-suite-v1-20260916", function () {
                  loadScript("/static/baseline-job-filter.js?v=baseline-job-filter-v1-20260916", function () {
                    loadScript("/static/runs-log-ui.js?v=runs-log-v1-20260916", function () {
                      loadScript("/static/runs-job-control.js?v=baseline-cancel-v2-20260916", function () {
                        loadScript("/static/runs-submit-fix.js?v=batch-submit-v1-20260916", function () {
                          loadScript("/static/procvlm-mode-ui.js?v=procvlm-lora-mode-v1-20260916");
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
