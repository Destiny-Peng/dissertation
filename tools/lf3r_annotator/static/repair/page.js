"use strict";

(function mountRepairPage() {
  var mount = document.getElementById("repairMount");
  if (!mount) return;

  mount.innerHTML = `
    <div class="page-container repair-page-container">
      <header class="repair-shell-header runs-shell-header">
        <div class="runs-shell-copy">
          <p class="eyebrow">LF3R REPAIR</p>
          <h2>Synthetic Suffix</h2>
          <p>Prepare one success rollout, define the cut, generate a world-model suffix, then compare it against the real suffix.</p>
        </div>
        <div class="repair-header-actions">
          <button id="repairRefreshRuns" class="ghost-button" type="button">Refresh</button>
        </div>
      </header>

      <div class="repair-mode-switch runs-mode-switch" role="tablist" aria-label="Repair workflow">
        <button id="repairDataTab" type="button" role="tab" data-repair-mode="data">1 · Select Data</button>
        <button id="repairCutTab" type="button" role="tab" data-repair-mode="cut">2 · Define Cut</button>
        <button id="repairGenerateTab" type="button" role="tab" data-repair-mode="generate">3 · Generate Suffix</button>
        <button id="repairCompareTab" type="button" role="tab" data-repair-mode="compare">4 · Compare / Export</button>
      </div>

      <div id="repairGpuMount" class="repair-gpu-mount"></div>
      <div id="repairPageStatus" class="analysis-status repair-page-status" role="status">Open Repair to load success rollouts.</div>

      <div class="repair-console">
        <div class="repair-tab-stack">
          <section id="repairDataPanel" class="analysis-card repair-card repair-tab-panel" data-repair-panel="data" role="tabpanel" aria-labelledby="repairDataTab">
            <div class="repair-step-head">
              <div class="repair-step-title">
                <span class="repair-step-number">1</span>
                <div>
                  <h3>Select Data</h3>
                  <p class="analysis-card-note">Choose one successful rollout from the loaded manifests. The manifest remains the source of truth for physical camera views and prepared controls.</p>
                </div>
              </div>
            </div>
            <div class="repair-filter-grid">
              <label><span>Manifest</span><select id="repairManifestFilter"><option value="all">All manifests</option></select></label>
              <label><span>Suite</span><select id="repairSuiteFilter"><option value="all">All suites</option></select></label>
              <label><span>Task</span><select id="repairTaskFilter"><option value="all">All tasks</option></select></label>
              <label><span>Success rollout</span><select id="repairRolloutSelect" class="repair-rollout-select"></select></label>
            </div>
            <div id="repairRolloutSummary" class="repair-rollout-summary"></div>
          </section>

          <section id="repairCutPanel" class="analysis-card repair-card repair-tab-panel" data-repair-panel="cut" role="tabpanel" aria-labelledby="repairCutTab" hidden>
            <div class="repair-step-head">
              <div class="repair-step-title">
                <span class="repair-step-number">2</span>
                <div>
                  <h3>Define Cut</h3>
                  <p class="analysis-card-note">The cut is an experiment variable, not a failure annotation. LIBERO continuation branches from state[c+1] and actions[c+1:].</p>
                </div>
              </div>
            </div>
            <div class="repair-config-grid">
              <label><span>Cut mode</span>
                <select id="repairCutType">
                  <option value="progress" selected>Fixed progress</option>
                  <option value="frame">Fixed frame (debug)</option>
                </select>
              </label>
              <label id="repairCutProgressWrap"><span>Progress (%)</span>
                <input id="repairCutProgress" type="number" min="1" max="99" step="1" value="50">
              </label>
              <label id="repairCutFrameWrap" class="hidden"><span>RGB frame c</span>
                <input id="repairCutFrame" type="number" min="0" step="1" value="0">
              </label>
              <label><span>Alignment smoke-test minimum PSNR</span>
                <input id="repairAlignmentPsnr" type="number" min="0" step="0.5" value="20">
              </label>
            </div>
            <div id="repairCutReadout" class="analysis-status">Select a rollout.</div>
            <div class="repair-cut-visual">
              <div class="repair-cut-track" aria-label="Real prefix and world-model suffix split">
                <div id="repairCutPrefix" class="repair-cut-prefix" style="width:50%"></div>
                <div id="repairCutSuffix" class="repair-cut-suffix" style="width:50%"></div>
              </div>
              <div class="repair-cut-labels"><span>REAL PREFIX</span><span>WM SUFFIX</span></div>
            </div>
          </section>

          <section id="repairGeneratePanel" class="analysis-card repair-card repair-tab-panel" data-repair-panel="generate" role="tabpanel" aria-labelledby="repairGenerateTab" hidden>
            <div class="repair-step-head">
              <div class="repair-step-title">
                <span class="repair-step-number">3</span>
                <div>
                  <h3>Generate Suffix</h3>
                  <p class="analysis-card-note">A2World and Ctrl-World share the same LIBERO cut/alignment benchmark. Model-specific views, controls, timing, and resize adapters remain explicit in provenance.</p>
                </div>
              </div>
            </div>
            <div class="repair-config-grid">
              <label><span>World model</span>
                <select id="repairWorldModel">
                  <option value="a2world" selected>A2World</option>
                  <option value="ctrl_world">Ctrl-World</option>
                  <option value="wan2_2">Wan2.2-I2V-A14B</option>
                </select>
              </label>
              <label><span>GPU</span>
                <input id="repairGpu" type="text" inputmode="numeric" value="" placeholder="e.g. 0" maxlength="8">
              </label>
              <div class="repair-kv repair-config-span-2"><small>Shared protocol</small><strong>LIBERO alignment → model adapter → synchronized suffix comparison</strong></div>
            </div>

            <div id="repairA2WorldPanel" data-repair-model-panel="a2world">
              <div class="repair-config-grid repair-config-section">
                <label><span>Checkpoint type</span>
                  <select id="repairCheckpointType">
                    <option value="libero_adapted" selected>LIBERO adapted (in-domain upper bound)</option>
                    <option value="generic_pretrained">Generic pretrained</option>
                    <option value="custom">Custom path</option>
                  </select>
                </label>
                <label><span>Checkpoint</span><input id="repairCheckpoint" type="text" value="checkpoints/a2world-libero.pt"></label>
                <label><span>Base checkpoints</span><input id="repairBaseCheckpoints" type="text" value="checkpoints"></label>
                <label><span>Sampling steps</span><input id="repairSamplingSteps" type="number" min="1" step="1" value="35"></label>
                <label><span>Guidance</span><input id="repairGuidance" type="number" min="0" step="0.1" value="0"></label>
                <label><span>Seed</span><input id="repairSeed" type="number" step="1" value="0"></label>
              </div>
              <div class="repair-config-grid repair-config-section">
                <div class="repair-kv"><small>View adapter</small><strong id="repairCameraMapping" style="white-space:pre-line">agentview ← cam_high\neye_in_hand ← cam_wrist</strong></div>
                <label class="repair-kv"><small>Missing view handling</small><span><input id="repairDuplicateViews" type="checkbox"> Explicitly allow cam_high duplication inside A2World adapter</span></label>
                <div class="repair-kv"><small>Action adapter</small><strong id="repairActionAdapter">LIBERO 7D → A2World LIBERO servo</strong></div>
                <label class="repair-kv"><small>History conditioning</small><span><input id="repairHistory" type="checkbox" checked> Enabled (A2World default)</span></label>
              </div>
            </div>

            <div id="repairCtrlWorldPanel" data-repair-model-panel="ctrl_world" class="hidden">
              <div class="repair-config-grid repair-config-section">
                <label><span>Checkpoint</span><input id="repairCtrlCheckpoint" type="text" placeholder="auto-detect project-local Ctrl-World checkpoint"></label>
                <label><span>Ctrl-World source</span><input id="repairCtrlSourceRoot" type="text" placeholder="repos/Ctrl-World (auto-detect)"></label>
                <label><span>Ctrl Python</span><input id="repairCtrlPython" type="text" placeholder="auto-detect configured project-local environment"></label>
                <label><span>SVD base model</span><input id="repairCtrlSvd" type="text" placeholder="auto-detect project-local stable-video-diffusion-img2vid"></label>
                <label><span>CLIP model</span><input id="repairCtrlClip" type="text" placeholder="auto-detect project-local clip-vit-base-patch32"></label>
                <label><span>DROID normalization stats</span><input id="repairCtrlDataStat" type="text" placeholder="repos/Ctrl-World/dataset_meta_info/droid/stat.json"></label>
              </div>
              <div class="repair-config-grid repair-config-section">
                <label><span>Target FPS</span><input id="repairCtrlTargetFps" type="number" min="0.1" step="0.1" value="5"></label>
                <label><span>Inference steps</span><input id="repairCtrlInferenceSteps" type="number" min="1" step="1" value="50"></label>
                <label><span>Guidance scale</span><input id="repairCtrlGuidance" type="number" min="0" step="0.1" value="1"></label>
                <label><span>Seed</span><input id="repairCtrlSeed" type="number" step="1" value="0"></label>
              </div>
              <div class="repair-config-grid repair-config-section">
                <div class="repair-kv"><small>View adapter</small><strong style="white-space:pre-line">exterior_1 ← cam_high\nexterior_2 ← cam_high (adapter-local duplicate)\nwrist ← cam_wrist</strong></div>
                <div class="repair-kv"><small>Control adapter</small><strong>Prepared LIBERO GT replay → DROID-style 7D absolute pose/state</strong></div>
                <div class="repair-kv"><small>Native image geometry</small><strong>source RGB → 192×320 Ctrl-World input (recorded resize)</strong></div>
                <label class="repair-kv"><small>Text conditioning</small><span><input id="repairCtrlTextConditioning" type="checkbox" checked> Use task instruction</span></label>
              </div>
            </div>

            <div data-repair-model-panel="wan2_2" class="hidden">
              <div class="repair-config-grid" style="margin-top:10px">
                <label><span>Wan Python</span><input id="repairWanPython" type="text" placeholder="Absolute path to Wan environment Python"></label>
                <label><span>Wan checkpoint directory</span><input id="repairWanCheckpoint" type="text" placeholder="Absolute path to Wan2.2-I2V-A14B checkpoint directory"></label>
                <label><span>Wan source directory (optional)</span><input id="repairWanSourceRoot" type="text" value="" placeholder="repos/Wan2.2 (default); or set WAN_SOURCE_ROOT"></label>
              </div>
              <p class="analysis-card-note">Input: one cam_high RGB frame at the cut point and the original task instruction. No actions, states, prefix video, or LIBERO runtime required. Wan generates a video at its native duration and FPS.</p>
            </div>
            <div id="repairValidation" class="repair-validation"></div>
            <div class="repair-actions">
              <button id="repairValidateButton" class="ghost-button" type="button">Validate inputs</button>
              <button id="repairRunButton" class="save-button" type="button" disabled>Run A2World</button>
            </div>
          </section>

          <section id="repairComparePanel" class="analysis-card repair-card repair-tab-panel repair-compare-panel" data-repair-panel="compare" role="tabpanel" aria-labelledby="repairCompareTab" hidden>
            <div class="repair-step-head">
              <div class="repair-step-title">
                <span class="repair-step-number">4</span>
                <div>
                  <h3>Compare / Export</h3>
                  <p class="analysis-card-note">Inspect the synchronized real and generated suffixes. Dataset export remains a later phase.</p>
                </div>
              </div>
            </div>

            <div class="repair-run-picker">
              <label><span>Repair run</span><select id="repairRunSelect"><option value="">No Repair runs yet</option></select></label>
            </div>
            <div id="repairResultStatus" class="analysis-status">Select a completed Repair run.</div>

            <div class="repair-timeline" aria-label="Real and world-model suffix timeline">
              <div class="repair-timeline-row"><span>REAL</span><div class="repair-timeline-track">
                <div id="repairRealPrefix" class="repair-timeline-real-prefix" style="width:50%"></div>
                <div id="repairRealSuffix" class="repair-timeline-real-suffix" style="width:50%"></div>
              </div></div>
              <div class="repair-timeline-row"><span>WM</span><div class="repair-timeline-track">
                <div id="repairWmPrefix" class="repair-timeline-wm-prefix" style="width:50%"></div>
                <div id="repairWmSuffix" class="repair-timeline-wm-suffix" style="width:50%"></div>
              </div></div>
            </div>

            <div id="repairCompareGrid" class="repair-compare-grid">
              <div class="repair-video-empty">No result selected</div>
            </div>
            <div class="repair-transport">
              <button id="repairPlayPause" class="ghost-button" type="button">Play</button>
              <input id="repairSeek" type="range" min="0" max="1" step="0.01" value="0" aria-label="Synchronized suffix time">
              <span id="repairTimeReadout">0.00 s after cut</span>
            </div>

            <h4>Visual metrics</h4>
            <div id="repairMetrics" class="repair-metrics"></div>
            <p class="analysis-card-note">PSNR / SSIM / LPIPS are diagnostics only; policy-training validity is not inferred from them.</p>

            <div class="repair-review">
              <h4>Task-aware human evaluation</h4>
              <p class="analysis-card-note">Usable for policy training?</p>
              <div class="repair-review-options">
                <label><input type="radio" name="repairUsability" value="yes"> Yes</label>
                <label><input type="radio" name="repairUsability" value="no"> No</label>
                <label><input type="radio" name="repairUsability" value="uncertain"> Uncertain</label>
              </div>
              <p class="analysis-card-note">Failure reason</p>
              <div class="repair-reasons">
                <label><input type="checkbox" data-repair-reason value="robot_motion"> robot motion</label>
                <label><input type="checkbox" data-repair-reason value="gripper"> gripper</label>
                <label><input type="checkbox" data-repair-reason value="object_motion"> object motion</label>
                <label><input type="checkbox" data-repair-reason value="contact"> contact</label>
                <label><input type="checkbox" data-repair-reason value="geometry"> geometry</label>
                <label><input type="checkbox" data-repair-reason value="visual_corruption"> visual corruption</label>
                <label><input type="checkbox" data-repair-reason value="temporal_drift"> temporal drift</label>
              </div>
              <div class="repair-actions">
                <span id="repairHumanStatus" class="analysis-card-note"></span>
                <button id="repairSaveHuman" class="ghost-button" type="button">Save evaluation</button>
              </div>
            </div>
          </section>
        </div>

        <aside class="runs-activity repair-activity" aria-label="Repair job activity">
          <div class="runs-activity-heading repair-activity-heading">
            <div><p class="eyebrow">ACTIVITY</p><h3>Repair jobs</h3></div>
            <div class="runs-job-filter repair-job-filter">
              <label for="repairJobFilter"><span>Show</span>
                <select id="repairJobFilter">
                  <option value="active">Active</option>
                  <option value="all">All</option>
                  <option value="running">Running</option>
                  <option value="queued">Queued</option>
                  <option value="complete">Complete</option>
                  <option value="problem">Failed / cancelled</option>
                </select>
              </label>
              <span id="repairJobFilterCount" class="runs-job-filter-count">0 / 0</span>
            </div>
          </div>
          <div id="repairJobStatus" class="runs-activity-status" role="status">No active Repair job.</div>
          <div id="repairJobList" class="persistent-job-list runs-filtered-job-list"></div>
          <pre id="repairJobLog" class="job-log repair-log" aria-label="Repair job log" hidden></pre>
        </aside>
      </div>
    </div>
  `;

  var validModes = ["data", "cut", "generate", "compare"];
  var buttons = Array.prototype.slice.call(mount.querySelectorAll("[data-repair-mode]"));
  var panels = Array.prototype.slice.call(mount.querySelectorAll("[data-repair-panel]"));
  var currentMode = "data";
  try {
    var stored = localStorage.getItem("lf3r.repair.mode");
    if (validModes.indexOf(stored) >= 0) currentMode = stored;
  } catch (_) {}

  buttons.forEach(function (button) {
    var panel = mount.querySelector('[data-repair-panel="' + button.dataset.repairMode + '"]');
    if (panel) button.setAttribute("aria-controls", panel.id);
  });

  function setMode(mode) {
    currentMode = validModes.indexOf(mode) >= 0 ? mode : "data";
    buttons.forEach(function (button) {
      var active = button.dataset.repairMode === currentMode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    panels.forEach(function (panel) {
      panel.hidden = panel.dataset.repairPanel !== currentMode;
    });
    try { localStorage.setItem("lf3r.repair.mode", currentMode); } catch (_) {}
  }

  buttons.forEach(function (button, index) {
    button.addEventListener("click", function () { setMode(button.dataset.repairMode); });
    button.addEventListener("keydown", function (event) {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      var next = event.key === "ArrowRight"
        ? (index + 1) % buttons.length
        : (index - 1 + buttons.length) % buttons.length;
      buttons[next].focus();
      setMode(buttons[next].dataset.repairMode);
    });
  });

  var gpuAttempts = 0;
  function installGpuStrip() {
    var host = document.getElementById("repairGpuMount");
    if (!host || host.dataset.gpuInstalled === "true") return true;
    if (!window.LF3RRunsGpu || typeof window.LF3RRunsGpu.createStrip !== "function") {
      gpuAttempts += 1;
      if (gpuAttempts < 20) window.setTimeout(installGpuStrip, 100);
      return false;
    }
    host.dataset.gpuInstalled = "true";
    var strip = window.LF3RRunsGpu.createStrip({
      prefix: "repairGpuStatus",
      eyebrow: "GPU STATUS",
      title: "Current device state"
    });
    host.replaceChildren(strip);
    if (typeof window.LF3RRunsGpu.install === "function") {
      window.LF3RRunsGpu.install("repairGpuStatus");
    }
    return true;
  }

  setMode(currentMode);
  installGpuStrip();

  window.LF3RRepairPage = {
    setMode: setMode,
    installGpu: installGpuStrip
  };
})();