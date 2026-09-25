"use strict";

window.LF3RRunsToolsLayout = (function createRunsToolsLayout() {
  function createPanels(workspace) {
    function toolActivityMarkup() {
      return [
        '<aside class="runs-tool-activity">',
          '<div class="runs-activity-heading"><p class="eyebrow">ACTIVITY</p><h3>Project tool job</h3></div>',
          '<div id="projectToolStatus" class="runs-activity-status">No project-tool job selected.</div>',
          '<div id="projectToolJobs" class="runs-tool-job-list"></div>',
          '<pre id="projectToolLog" class="job-log runs-tool-log" aria-label="Project tool log"></pre>',
        '</aside>'
      ].join("");
    }
  
    var safePanel = document.createElement("section");
    safePanel.className = "evaluation-panel runs-custom-panel";
    safePanel.id = "runsSafePanel";
    safePanel.dataset.runsPanel = "safe";
    safePanel.setAttribute("role", "tabpanel");
    safePanel.setAttribute("aria-labelledby", "runsSafeTab");
    safePanel.innerHTML = [
      '<div class="evaluation-heading runs-custom-heading">',
        '<div><p class="eyebrow">SAFE WORKFLOW</p><h2>Prepare, train, and validate SAFE detectors</h2></div>',
      '</div>',
      '<div class="runs-custom-layout">',
        '<div class="runs-tool-stack">',
          '<section class="runs-tool-card">',
            '<div class="runs-tool-card-heading"><span class="runs-step">1</span><div><h3>Prepare dataset</h3><p>Stage official SAFE-format CSV / pickle files from the LF3R manifest.</p></div></div>',
            '<div class="runs-tool-grid">',
              '<label><span>Dataset role</span><select id="safePrepareRole"><option value="libero_10">LIBERO-10</option><option value="libero_spatial">LIBERO-Spatial</option><option value="controlled_analysis">Controlled analysis</option><option value="all">All</option></select></label>',
              '<label><span>Partition</span><select id="safePreparePartition"><option value="natural_observation">Natural observation</option><option value="controlled_analysis">Controlled analysis</option><option value="all">All</option></select></label>',
              '<label class="runs-tool-wide"><span>Source run name (optional)</span><input id="safePrepareRunName" type="text" placeholder="Use when task/episode basenames are duplicated"></label>',
              '<label class="runs-tool-wide"><span>Output dataset directory</span><input id="safePrepareOutput" type="text"></label>',
            '</div>',
            '<div class="runs-tool-action"><span>Source rollout files are not modified.</span><button id="safePrepareRun" class="save-button" type="button">Prepare dataset</button></div>',
          '</section>',
          '<section class="runs-tool-card">',
            '<div class="runs-tool-card-heading"><span class="runs-step">2</span><div><h3>Train detector</h3><p>Run the official SAFE Hydra trainer through the LF3R wrapper.</p></div></div>',
            '<div class="runs-tool-grid">',
              '<label class="runs-tool-wide"><span>Dataset directory</span><input id="safeTrainDataset" type="text"></label>',
              '<label><span>Model</span><select id="safeTrainModel"><option value="mlp">SAFE-MLP</option><option value="lstm">SAFE-LSTM</option></select></label>',
              '<label><span>GPU</span><input id="safeTrainGpu" type="text" value="0"></label>',
              '<label><span>Epochs</span><input id="safeTrainEpochs" type="number" min="1" value="1000"></label>',
              '<label><span>Batch size</span><input id="safeTrainBatch" type="number" min="1" value="512"></label>',
              '<label><span>Hidden dim</span><input id="safeTrainHidden" type="number" min="1" value="256"></label>',
              '<label><span>Seed</span><input id="safeTrainSeed" type="text" value="0"></label>',
              '<label class="runs-tool-wide"><span>Logs root</span><input id="safeTrainLogs" type="text" value="outputs/safe_training/logs/web"></label>',
              '<label class="runs-check"><input id="safeTrainNormalize" type="checkbox"> Normalize hidden states</label>',
            '</div>',
            '<div class="runs-tool-action"><span>GPU-only; there is no CPU fallback.</span><button id="safeTrainRun" class="save-button" type="button">Train SAFE</button></div>',
          '</section>',
          '<section class="runs-tool-card">',
            '<div class="runs-tool-card-heading"><span class="runs-step">3</span><div><h3>Validate checkpoint</h3><p>Reload one trained checkpoint and write finite validation scores.</p></div></div>',
            '<div class="runs-tool-grid">',
              '<label class="runs-tool-wide"><span>Dataset directory</span><input id="safeValidateDataset" type="text"></label>',
              '<label class="runs-tool-wide"><span>Checkpoint</span><input id="safeValidateCheckpoint" type="text" placeholder="outputs/safe_training/logs/.../model_final.ckpt"></label>',
              '<label><span>Model</span><select id="safeValidateModel"><option value="mlp">SAFE-MLP</option><option value="lstm">SAFE-LSTM</option></select></label>',
              '<label><span>GPU</span><input id="safeValidateGpu" type="text" value="0"></label>',
              '<label class="runs-tool-wide"><span>Output JSON</span><input id="safeValidateOutput" type="text" value="outputs/safe_training/validation/web_scores.json"></label>',
            '</div>',
            '<div class="runs-tool-action"><span>This validates loading and finite scores; it does not retrain.</span><button id="safeValidateRun" class="ghost-button" type="button">Validate checkpoint</button></div>',
          '</section>',
        '</div>',
        toolActivityMarkup(),
      '</div>'
    ].join("");
  
    var toolsPanel = document.createElement("section");
    toolsPanel.className = "evaluation-panel runs-custom-panel";
    toolsPanel.id = "runsToolsPanel";
    toolsPanel.dataset.runsPanel = "tools";
    toolsPanel.setAttribute("role", "tabpanel");
    toolsPanel.setAttribute("aria-labelledby", "runsToolsTab");
    toolsPanel.innerHTML = [
      '<div class="evaluation-heading runs-custom-heading">',
        '<div><p class="eyebrow">PROJECT UTILITIES</p><h2>Data export, diagnostics, and maintenance</h2></div>',
      '</div>',
      '<div class="runs-custom-layout">',
        '<div class="runs-tool-stack">',
          '<section class="runs-tool-card">',
            '<div class="runs-tool-card-heading"><div><h3>Export rollout package</h3><p>Create a self-contained share package with videos, sidecars, annotations, and manifest.</p></div></div>',
            '<div class="runs-outcome-row" id="exportOutcomes">',
              '<label><input type="checkbox" value="failure" checked> Failure</label>',
              '<label><input type="checkbox" value="recovered_success"> Recovered success</label>',
              '<label><input type="checkbox" value="success"> Success</label>',
              '<label><input type="checkbox" value="uncertain"> Uncertain</label>',
            '</div>',
            '<div class="runs-tool-grid">',
              '<label><span>Dataset role</span><select id="exportDatasetRole"><option value="libero_10">LIBERO-10</option><option value="libero_spatial">LIBERO-Spatial</option><option value="controlled_analysis">Controlled analysis</option><option value="all">All</option></select></label>',
              '<label><span>Review status</span><select id="exportReviewStatus"><option value="complete">Complete</option><option value="in_progress">In progress</option><option value="unreviewed">Unreviewed</option><option value="all">All</option></select></label>',
              '<label class="runs-tool-wide"><span>Output directory (optional)</span><input id="exportOutputDir" type="text" placeholder="Blank = timestamped outputs/shares package"></label>',
            '</div>',
            '<div class="runs-tool-action"><button id="exportDryRun" class="ghost-button" type="button">Preview selection</button><button id="exportRun" class="save-button" type="button">Create package</button></div>',
          '</section>',
          '<section class="runs-tool-card">',
            '<div class="runs-tool-card-heading"><div><h3>Robo-Dopamine interval sweep</h3><p>Run one checkpoint once and compare native sampling intervals without mixing outputs.</p></div></div>',
            '<div class="runs-tool-grid">',
              '<label class="runs-tool-wide"><span>Rollout</span><select id="roboSweepRollout"><option value="">Loading rollouts…</option></select></label>',
              '<label><span>Intervals</span><input id="roboSweepIntervals" type="text" value="2,5,10"></label>',
              '<label><span>GPU</span><input id="roboSweepGpu" type="text" value="0"></label>',
              '<label><span>Free-memory fraction</span><input id="roboSweepMemory" type="number" min="0.05" max="1" step="0.05" value="0.60"></label>',
              '<label class="runs-tool-wide"><span>Output parent</span><input id="roboSweepOutput" type="text" value="outputs/baselines/robo_interval_sweeps"></label>',
            '</div>',
            '<div class="runs-tool-action"><span>Diagnostic sweep only; it does not claim detector performance.</span><button id="roboSweepRun" class="save-button" type="button">Run interval sweep</button></div>',
          '</section>',
          '<section class="runs-tool-card">',
            '<div class="runs-tool-card-heading"><div><h3>Batch H.264 transcode</h3><p>Choose which loaded manifests to process. Every <code>camera_video_paths</code> entry in the selected manifests is processed; duplicate file paths are transcoded once.</p></div></div>',
            '<div class="runs-manifest-choice-toolbar">',
              '<div><strong>Manifest selection</strong><span id="batchManifestTranscodeSelectionCount">Loading manifests…</span></div>',
              '<div><button id="batchManifestTranscodeSelectAll" class="ghost-button" type="button">Select all</button><button id="batchManifestTranscodeSelectNone" class="ghost-button" type="button">Clear</button></div>',
            '</div>',
            '<div id="batchManifestTranscodeManifests" class="runs-manifest-choice-list"><div class="runs-manifest-choice-empty">Loading loaded manifests…</div></div>',
            '<div id="batchManifestTranscodeSummary" class="runs-manifest-summary">Already-H.264 videos are skipped. Existing <code>.orig.mp4</code> backups are never overwritten.</div>',
            '<div class="runs-tool-action"><span>Runs sequentially as one persistent background job; duplicate <code>video_path</code> values across selected manifests are converted only once.</span><button id="batchManifestTranscodeRun" class="save-button" type="button" disabled>Transcode selected manifests</button></div>',
          '</section>',
          '<section class="runs-tool-card runs-maintenance-card">',
            '<div class="runs-tool-card-heading"><div><h3>Validation & maintenance</h3><p>Bounded checks around existing project artifacts. No Analysis jobs are launched here.</p></div></div>',
            '<div class="runs-maintenance-actions runs-maintenance-validation">',
              '<button id="validateBaselinesRun" class="ghost-button" type="button">Validate baseline pipelines</button>',
              '<button id="validateVariantsRun" class="ghost-button" type="button">Validate instruction variants</button>',
            '</div>',
            '<div class="runs-manifest-rescan">',
              '<div class="runs-manifest-rescan-heading"><strong>Import / rescan external rollouts</strong><span>Rebuild the main rollout manifest, then refresh Review and Runs automatically.</span></div>',
              '<div class="runs-tool-grid">',
                '<label class="runs-tool-wide"><span>Default scan roots · always included</span><div class="runs-path-list"><code>outputs/openvla_libero</code><code>outputs/openvla_libero_spatial_native</code></div></label>',
                '<label class="runs-tool-wide"><span>Additional scan roots · optional, one project-local directory per line</span><textarea id="rebuildManifestExtraRoots" rows="3" placeholder="outputs/imported_rollouts&#10;outputs/another_rollout_root"></textarea></label>',
              '</div>',
              '<div id="rebuildManifestSummary" class="runs-manifest-summary">No rescan has been run in this page session.</div>',
              '<div class="runs-tool-action"><span>Expected layout: &lt;root&gt;/&lt;run containing natural&gt;/&lt;suite&gt;/taskN--epM--succ0|1.mp4. Default roots are never dropped when extras are supplied.</span><button id="rebuildManifestRun" class="save-button" type="button">Rebuild manifest + refresh</button></div>',
            '</div>',
          '</section>',
        '</div>',
        toolActivityMarkup().replace('id="projectToolStatus"', 'id="projectToolStatusTools"').replace('id="projectToolJobs"', 'id="projectToolJobsTools"').replace('id="projectToolLog"', 'id="projectToolLogTools"'),
      '</div>'
    ].join("");
  
    workspace.appendChild(safePanel);
    workspace.appendChild(toolsPanel);
    return { safePanel: safePanel, toolsPanel: toolsPanel };
  }

  return { createPanels: createPanels };
})();
