# LF3R Failure Review

A zero-dependency, local-only browser tool for reviewing OpenVLA/LIBERO rollouts. It serves video byte ranges for seeking and stores authoritative annotations as project-local JSON.

## Current dataset snapshot

The current versioned manifest contains 136 natural observations:

| Dataset role | Suite | Rollouts | Evaluator outcomes |
| --- | --- | ---: | --- |
| `primary_natural` | `libero_10` | 125 | 73 success, 52 failure |
| `reference_natural` | `libero_spatial` | 11 | 11 success |
| `controlled_analysis` | — | 0 | no controlled rows currently installed |

The primary LIBERO-10 set includes tasks 0–9. Task counts are 16 each for tasks 0–3, 11 for task 4, and 10 each for tasks 5–9. The reference LIBERO-Spatial rows remain available for comparison but are not part of the primary natural-failure summary. The manifest can later include controlled injections, but those rows must stay in the explicit `controlled_analysis` partition.

`task` and `trial` are separate dimensions. A task identifies one LIBERO instruction/environment; a trial is an independent episode for that task. The filename suffix `succ0`/`succ1` is the evaluator outcome, not a human annotation. The Review page lets an annotator replace that provisional outcome with `clean_success`, `recovered_success`, `terminal_failure`, or `uncertain`.

## Pipeline overview

    generate_libero10_natural.sh
        -> run_openvla_libero10_natural.py
        -> official OpenVLA run_libero_eval.py
        -> MP4 + per-step CSV under outputs/openvla_libero/

    generate_libero_spatial_native.sh
        -> the same suite-aware runner/evaluator
        -> native 256x256 replay MP4 + per-step CSV under outputs/openvla_libero_spatial_native/
        -> build_manifest.py
        -> manifest.jsonl + summary.json
        -> server.py + static browser UI
        -> atomic annotation JSON + append-only event log
        -> optional explicit baseline batch jobs + temporal analysis jobs + Review rollout-generation jobs
        -> verify_pipeline.py and unit tests

Each module has one responsibility. Generation does not create human labels, manifest construction does not alter rollouts, and the annotation server launches model work only for an explicit, bounded single-rollout validation, one-method batch baseline request, or user-started natural rollout-generation request. Any baseline batch may divide that request across explicit rollout workers; SAFE runs per-rollout workers, while the model-backed methods create one independent persistent engine per shard.

## Start the server

From the LF3R server:

    cd /mnt/hdd/qiuxia/pyr/LF3R
    bash tools/lf3r_annotator/run_server.sh 8765

Stop it cleanly with Ctrl-C in that terminal, or from another terminal:

    bash tools/lf3r_annotator/stop_server.sh

The launcher records one project-local PID and rejects duplicate starts. It does not automatically restart the service.

`run_server.sh` loads `project_env.sh`, binds the Python server, records its PID at `cache/tmp/lf3r_annotator.pid`, and mirrors output to a timestamped log. `stop_server.sh` checks that the PID belongs to this project before sending `SIGTERM`; it never force-kills an unrelated process.

The service binds to `127.0.0.1` by default. Do not bind it to a public interface.

Web-started baseline, temporal-analysis, and rollout-generation commands are persistent tmux jobs. The server records each job under logs/annotator_jobs/<job-id>/job.json, so stopping or restarting the web server does not terminate a running command. On restart it reattaches monitoring to recorded LF3R sessions only; it never attaches to or changes unrelated tmux sessions. There is no web cancel button; ProcVLM and Robo-Dopamine command-line resume support remains available for interrupted runs.

On your own computer, create an SSH tunnel:

    ssh -N -L 8765:127.0.0.1:8765 USER@SERVER

Then open `http://127.0.0.1:8765` locally. If local port 8765 is occupied, use another local port, for example `-L 9876:127.0.0.1:8765`, and open `http://127.0.0.1:9876`.

## Pages and navigation

The annotator is one native HTML/CSS/JavaScript application with three hash-routed pages:

- `#/review` — the existing rollout queue, video/frame annotation form, and per-rollout baseline cards. A rollout selected from Analysis opens as `#/review/<rollout-id>` and keeps the selected record.
- `#/analysis` — live annotation/manifest statistics plus the latest complete baseline temporal-analysis snapshot. The page does not start model inference.
- `#/settings` — project-wide appearance settings. Changing a control previews it immediately; leaving Review or Settings with unsaved changes asks for confirmation.

The current rollout selection, Review queue filters, Analysis filters, and annotation state remain in browser memory while switching tabs. The default Analysis scope is `primary_natural`; `reference_natural`, `natural_observation` (all natural roles), and `controlled_analysis` must be selected explicitly.

## Shared Settings

Settings are stored in `config/lf3r_annotator.json`, not in browser local storage, so every browser connected to this loopback service reads the same project configuration. The initial/default document is:

    {
      "background_color": "#0b0d10",
      "surface_color": "#11151a",
      "surface_raised_color": "#171c22",
      "control_color": "#0f1318",
      "text_color": "#f4f6f7",
      "muted_color": "#98a3ad",
      "accent_color": "#67d9b5",
      "font_scale": 1.0,
      "review_font_scale": 1.0,
      "analysis_font_scale": 1.0,
      "control_font_scale": 1.0,
      "density": "comfortable"
    }

The color fields are six-digit hexadecimal colors. font_scale accepts 0.75 through 1.60 in 0.05 steps (75%-160%). review_font_scale, analysis_font_scale, and control_font_scale each accept 0.85 through 1.30 (85%-130%) in the same step size. The active page combines the global scale with its category scale; controls and help text additionally use the control scale. density is compact, comfortable, or spacious. Midnight, Slate, and Warm are convenience presets; they only change the configurable appearance fields.

Font scale changes the document root font size and all rem-based UI text. It does not use CSS `zoom`, so the video, frame slider, and SVG geometry retain their layout behavior. Density adjusts shared page/panel spacing.

The API is:

- `GET /api/settings` — returns `settings`, `defaults`, and the file update time. A missing or malformed project file falls back to defaults.
- `PUT /api/settings` — accepts exactly the fields above, validates colors/range/density, and writes through a temporary file followed by an atomic replacement. Unknown or missing fields return `400`.

## Analysis page

The live section uses the `/api/rollouts` manifest records already loaded by the Review page and recalculates when an annotation is saved; saved annotation outcomes override provisional manifest outcomes. Its default `primary_natural` scope reports:

- rollout count, completed annotation count, resolved count, resolved success rate, failure event count, and observable-onset coverage;
- clean-success, recovered-success, terminal-failure, and uncertain outcome distribution;
- event-level failure-type counts;
- causal-to-observable latency, normalized observable onset position (`observable_frame / (total_frames - 1)`), and observable-to-recovery latency, shown as median/IQR summaries;
- per-task stacked outcome counts with clickable task filtering.

Live filters are partition (`all`, `primary_natural`, `reference_natural`, `natural_observation`, or `controlled_analysis`), task suite, task, and outcome. Natural and controlled counts are never combined unless `All partitions` is selected. Clicking an outcome/task mark applies the same live filter.

The baseline section calls GET /api/analysis. The server chooses the newest directory under outputs/baseline_signal_analysis/ containing metadata.json, event_metrics.jsonl, and all required summary CSVs. On the current checkout this is highres_primary_20260830; it contains the 125-rollout primary-natural high-resolution analysis, its source time, selection count, manifest hash match, and stale status are shown. A stale snapshot remains viewable for description, but it is not silently presented as current.

The snapshot API returns method coverage, method/signal/outcome summaries, onset threshold/direction statistics, compact temporal event metrics, and the failure-localization summary/threshold/failure-type tables joined with rollout and task metadata. Large frame arrays are intentionally removed before JSON is sent to the browser. The page visualizes coverage, recovered-versus-terminal normalized response magnitude (median/IQR), post-event persistence, recovered-event recovery-to-baseline fraction, clean-success Q95 threshold exceedance/direction consistency, and the top anomalous event table. Method, snapshot-outcome, and snapshot-task filters are independent of the live filters; Review buttons navigate back to the corresponding rollout.

The snapshot is descriptive signal analysis, not detector-performance evaluation. Refreshing it only rereads existing output files; it never reruns SAFE, ProcVLM, RynnValue, or Robo-Dopamine inference. If no complete snapshot exists, the live section remains usable and the baseline section reports Unavailable. The high-resolution snapshot also exposes localization_event_metrics.jsonl, localization_summary.csv, localization_thresholds.csv, and localization_by_failure_type.csv. Q90, Q95, and Q99 thresholds are calibrated from clean-success pseudo-events for every native signal. Localization uses native crossing samples only, reports early alarms/misses and +/-4, 8, 16, and 30 frame tolerances, and leaves missing method outputs outside metric denominators. These are descriptive diagnostics on the annotated trajectories, not results from an independent detector test set.


### Analysis information hierarchy

Analysis is split into Overview, Method comparison, Failure types, Event explorer, Signal shape, and Archive & downloads. The hash route #/analysis aliases #/analysis/overview. The dashboard defaults to primary_natural, Q95, local level, 16 frames, and all events. Categorical comparisons use readable HTML bars and heatmaps; only the single onset-aligned Signal shape view uses SVG. Legacy charts are collapsed instead of being dumped into the first view.

The default GET /api/analysis response is dashboard/compact mode. It contains live statistics, snapshot provenance/freshness, coverage, aggregate rows, detail counts, available tabs, and declared artifact links; large event/frame arrays are kept out of the initial response. Use GET /api/analysis?view=full only for legacy compatibility.

The Event explorer loads detail rows on demand through GET /api/analysis/details?kind=...&page=...&page_size=...&sort=.... The server accepts only declared kinds, filters, and sort fields, and limits page_size to 1--100. GET /api/analysis/artifacts/<name> serves only declared files from the current legal snapshot, so it cannot be used for arbitrary path reads. The page keeps 25 rows per page and retains Review links for rollout-level inspection.

### Local change-point view

The Analysis page uses the latest lf3r_baseline_change_points snapshot as its primary analysis and retains the legacy temporal snapshot for comparison only. It reads changepoint_summary.csv, changepoint_reference_summary.csv, changepoint_by_failure_type.csv, changepoint_scales.csv, method_coverage.csv, the compact changepoint_event_metrics.jsonl, and comparison_with_full_136_20260827.csv. Filters keep method, signal, local feature (level, variance, or slope), frame scale, threshold, outcome, and task separate. The page shows recall versus trajectory-level clean-success false alarms, peak distance/tolerance hits, failure-type summaries with precision/F1/AUROC/AP, legacy/current comparisons, and event rows with Review links.

The current project snapshot is outputs/baseline_signal_analysis/changepoint_primary_20260830/: 125 primary-natural rollouts, 58 observable events, four local scales, and method coverage SAFE 125/125, ProcVLM 125/125, RynnValue 124/125, Robo-Dopamine 125/125. The RynnValue missing ID is explicit and is not filled by an older run. The local features use native samples only; the scale is a video-frame radius, so different baseline sampling densities remain visible. These are descriptive diagnostics from the annotated collection, not independently validated detector-performance estimates.

### Event-triggered onset view

The Analysis page also discovers the newest lf3r_event_triggered_signal_analysis snapshot and renders its complementary onset-aligned view. The event-triggered card provides method, signal, event-group, and local-scale filters; raw and onset-normalized median/IQR curves; local change-score curves; strongest case-versus-matched-clean separation with exact before/at/after-onset labels; and a compact event-level peak table with Review links. The UI shows source time, 125-rollout scope, 58 observable events, signal-level coverage, missing raw rollout IDs, clean-control matching, and native sampling intervals.

The checked-in snapshot is outputs/baseline_signal_analysis/event_triggered_primary_20260831/. SAFE, ProcVLM, and Robo-Dopamine have 125/125 signal coverage; RynnValue has 124/125, with libero_10-task02-ep005-natural-e8fc18cf25 unavailable and not filled from an older low-sampling run. The snapshot reports 1-frame SAFE/ProcVLM, 2-frame Robo-Dopamine, and 4-frame RynnValue median native intervals. Curves are aligned by observable onset and retain native sample offsets; connecting SVG segments are visual guides, not interpolated measurements. The event-triggered result is descriptive and complementary to the primary local change-point analysis.

## Review workflow

Choose a rollout, play or step through it, and label:

- outcome as clean success, recovered success, failure, or uncertain;
- one failure event for every failed attempt or visible deviation;
- an independent failure type, causal/injected onset, and observable onset for each event;
- either a recovery frame or terminal-failure frame for each event;
- confidence and notes.

Use **Add failure event** as many times as needed; there is no fixed event-count limit. Click or focus an event card to make it active. Arrow keys step frames, Shift+arrows step ten frames, keys 1/2/3 set causal/observable/terminal on the active event, and S saves. Recovery frames use the event card's **Use current** button. The header readout shows both video frame and corresponding environment timestep.

`recovered_success` requires at least one event and does not allow a terminal-failure frame, because terminal means recovery is no longer plausible. Setting a recovery frame clears terminal for that event and vice versa.

Annotations are atomically written to:

    annotations/failure_annotations/v1/records/<rollout-id>.json

Append-only save events are kept in `annotations/failure_annotations/v1/events/`.

### Annotation service and UI modules

- server.py serves static UI files, health and rollout APIs, video byte ranges, annotation read/write endpoints, shared Settings APIs, the read-only Analysis API, baseline evaluation/job APIs, batch baseline jobs, temporal-analysis jobs, and rollout-generation jobs. Manifest video paths, output paths, and job logs are confined to PROJECT_ROOT.
- static/index.html, styles.css, app.js, and workspace.js implement the Review, Analysis, and Settings pages, hash navigation, shared theme application, rollout filters, playback, frame stepping, frame-to-environment-timestep display, keyboard shortcuts, label editing, baseline signal charts, native SVG analysis charts, model-output readout, CLI help tooltips, rollout-generation polling, and bounded validation polling. static/parameter_help.json is the single metadata source for visible CLI names, defaults, effects, and upstream forwarding.
- `annotation.schema.json` documents schema v2's variable-length `failure_events[]` while remaining compatible with existing v1 records. The UI maps every old single-onset record to event #1 without rewriting it; it upgrades only when that annotation is saved again. Server validation enforces frame bounds, onset ordering, recovery/terminal exclusivity, and recovered-success semantics.
- Annotation records are written through a temporary file plus atomic rename. Every save also appends a small audit event; rollout media and manifest records remain unchanged.


### Baseline evaluation in the annotator

Selecting a rollout loads any completed baseline output that is available for that exact rollout. The Review page also exposes a one-method batch panel and an OpenVLA/LIBERO-10 natural rollout-generation panel.

- GET /api/baselines/<rollout-id> returns normalized samples for SAFE, ProcVLM, RynnValue, Robo-Dopamine, and DenseReward, together with the source run and parser validation status.
- Missing outputs are explicit rather than silently treated as zero or failure. A warning is shown when a baseline emits a raw frame index outside the video bounds; the displayed sample is clipped only for alignment and the raw index remains visible.
- Each baseline signal chart uses the full video frame domain (0 ... total_frames-1) and overlays the same color-coded causal/observable/terminal/recovery markers as the playback timeline. The curve and onset overlay share a responsive plot track with the video timeline, so they resize together; markers refresh while editing.
- The baseline chart legend is interactive: click or keyboard-focus a signal label to show/hide that curve. Visibility is remembered per rollout and method during the current page session. Multi-perspective Robo-Dopamine charts show fused `progress` and `hop` by default; incremental/forward/backward component progress and hop curves remain available from the labels.
- Each baseline card has a **Result run** selector. `Automatic` keeps the existing newest-readable-run behavior; selecting a concrete run makes only the current rollout read that run. **Apply to all** stores the same run choice for the current instruction condition while navigating Review. Other rollouts use it only when that run contains their raw output; otherwise the card explicitly shows unavailable and never silently mixes in another run. Selecting `Automatic` and applying it clears the method-wide override.
- The selector is populated from `GET /api/baselines/runs?scope=all`; run paths remain project-local and are validated server-side under `outputs/baselines`.
- The per-method **Run baseline** button calls POST /api/baselines/run/<rollout-id>. It is deliberately bounded to one rollout; independent method jobs may run at the same time and each is tracked separately until its parsed result is refreshed.
- GPU selects CUDA_VISIBLE_DEVICES; Memory is the target fraction of currently free GPU memory. The runner converts it to the vLLM total-memory parameter immediately before each model worker. The gate is based on whether available GPU memory can satisfy the run. GPU utilization percentage is informational and is not required to be 100%.

#### Instruction-condition view

The Review header exposes an **Instruction condition** selector for the same rollout video and annotation:

- **Full instruction** uses the original main-manifest rollout ID and existing baseline outputs.
- **A** and **B** use the prepared diagnostic rows in tools/lf3r_annotator/instruction_variants/libero_10_v1/manifest.jsonl. They are canonical single-subtask labels on the same video, not claims about execution order.
- A/B baseline cards remain explicitly unavailable until a baseline is run against the variant manifest. The UI never reuses a full-instruction raw directory for an A/B view.
- The completed Robo-Dopamine aggregate run outputs/baselines/web_runs/robo_dopamine-batch-326e663e25bc/robo_dopamine_20260908_105724_234057/ is recorded as full_instruction in run.json (166/166 complete, 0 failed). No raw output was moved or copied.

The selector changes only the displayed instruction and condition-aware baseline lookup. Annotation files, video paths, and the rollout queue remain keyed by the original source rollout ID. Tasks without two validated atomic goals expose only Full instruction.

The server remains loopback-only. Existing completed outputs are read-only from the annotator; an explicit Run baseline action creates a timestamped run under outputs/baselines/web_runs/ and a project-local log under logs/baselines/web_runs/.


### Batch baseline runs from Review

The Review page has a **Batch baseline** panel. Select one method per request and a rollout scope. All five methods use the same worker table with one GPU ID and a right-open `[start,end)` range per worker. The selected method determines whether a worker runs SAFE per-rollout extraction or owns an independent persistent ProcVLM, RynnValue, Robo-Dopamine, or DenseReward engine.

| Web scope | Runner selection |
| --- | --- |
| `all` | `--partition all` |
| `natural_observation` | `--partition natural_observation` |
| `primary_natural` | `--partition natural_observation --dataset-role primary_natural` |
| `reference_natural` | `--partition natural_observation --dataset-role reference_natural` |
| `controlled_analysis` | `--partition controlled_analysis` |

`Total start index` and `Total end index` define the scope-relative total range for every method, with `end` excluded. The old `Limit` field remains a CLI/API compatibility option but is not used by the worker table. Adding or removing a worker recomputes all worker ranges evenly; after the number of workers is fixed, any row can be edited manually. The panel reports each worker count, overlap, gap, repeated GPU, and the number of unique rollouts that will actually execute.

All methods use rollout-level parallelism. A two-worker range `[10,20)` is represented as `[10,15)` and `[15,20)` by default. SAFE workers run per-rollout commands; ProcVLM, Robo-Dopamine, and DenseReward load one independent persistent engine per worker; RynnValue keeps its selected temporal `batch_size` unchanged. Every worker receives one `CUDA_VISIBLE_DEVICES` value. The GPU list is user-controlled: reuse of the same GPU is allowed, no utilization/memory/conflict admission check is performed for this scheduler, and a comma GPU list is only legacy tensor-parallel input when no worker rows are used. The resulting `outputs/baselines/web_runs/<job-id>/` contains one aggregate timestamped run with shared `raw/<rollout-id>` output and worker progress. Overlap is accepted but later duplicate assignments are recorded and skipped; a gap leaves the aggregate run partial and unsuitable as a complete Analysis input.

DenseReward is available in the same selector for both the single-rollout button and Batch baseline panel. Its advanced web fields are `--densereward-frame-interval` (default `1`) and `--densereward-max-new-tokens` (default `32`). The worker calls the official 3-frame model path once per sampled current frame, stores `reward` in `densereward_raw.jsonl`, and reuses the model instance across the worker's assigned rollouts. DenseReward is currently a Review baseline; the existing temporal Analysis runner still compares the four methods for which its analyzer has signal adapters.

The API equivalent is `POST /api/baselines/run-batch` with `baseline`, `scope`, `gpu`, `memory_utilization`, `start_index`, `end_index`, `workers`, `parallel_workers`, and an `options` object. A RynnValue example is:

    {
      "baseline": "rynnvalue",
      "scope": "primary_natural",
      "start_index": 10,
      "end_index": 20,
      "workers": [
        {"gpu": "0", "start_index": 10, "end_index": 15},
        {"gpu": "1", "start_index": 15, "end_index": 20}
      ],
      "options": {"rynn_batch_size": 1, "rynn_num_frames": 16, "rynn_evaluation_interval": 4}
    }

`GET /api/baselines/runs?scope=<scope>` discovers project-local completed runs and reports method, run root, status, selection/completion/failure counts, timestamps, manifest hash, rollout-ID coverage, and whether the run covers the requested scope. Paths outside the project or outside `outputs/baselines` are rejected. `GET /api/baseline-jobs/<job-id>` reports aggregate and per-worker progress; `/log` returns the recent log tail. All methods keep their existing advanced fields, and all visible fields use the canonical CLI metadata in `static/parameter_help.json`. Worker rows are forwarded as repeated `--worker-spec GPU:START:END` arguments for every method.

### CLI parameter help

All baseline and rollout-generation fields with a help marker show a body-mounted tooltip on hover or keyboard focus. The tooltip reports the exact command-line flag, default, scope, effect, and, where applicable, the upstream flag received by the existing worker. The canonical metadata is static/parameter_help.json; tools/baselines/README.md remains the detailed command reference and links back to this file.

### Running temporal analysis from Analysis

### Robo-Dopamine multi-perspective outputs

The Review baseline UI defaults to `fused` for Robo-Dopamine, for both the single-rollout button and the Batch baseline panel. `fused` runs `incremental`, `forward`, and `backward` with one persistent GRM/vLLM engine and averages their native progress outputs. Explicit CLI or batch-API selection of `forward`, `incremental`, or `backward` remains available for compatibility. The same native sampled frame grid is required for fusion. The aggregate raw directory keeps the three official `pred_vllm.json` files separately and adds `multi_perspective/fused_progress.json`, `progress_curves.csv`, `progress_curves.png`, and `metadata.json`. The annotator reads fused progress as the primary curve and exposes component progress/hop signals when present. The arithmetic mean follows the official Robo-Dopamine README recommendation; it is descriptive and does not rerun inference from Analysis.

The Analysis page's **Run temporal analysis** panel performs analysis using existing baseline output directories; it never starts a baseline model. Choose a scope and one compatible completed run for SAFE, ProcVLM, and Robo-Dopamine; RynnValue may use one or more disjoint source runs selected together. Compatibility is checked by rollout ID, not only by the run's numeric count. A larger run is valid when its jobs.jsonl contains every selected ID; missing raw files are retained as explicit method coverage gaps by the analyzer. Multiple RynnValue roots are merged by rollout ID, and duplicate IDs are rejected.

The default windows are `pre_window_frames=60`, `post_window_frames=60`, and `background_stride_frames=30`. The output label is limited to letters, numbers, dot, underscore, and hyphen. The API payload is:

    {
      "scope": "primary_natural",
      "runs": {
        "safe": "outputs/baselines/...",
        "procvlm": "outputs/baselines/...",
        "rynnvalue": ["outputs/baselines/run_a", "outputs/baselines/run_b"],
        "robo_dopamine": "outputs/baselines/..."
      },
      "pre_window_frames": 60,
      "post_window_frames": 60,
      "background_stride_frames": 30,
      "output_label": "web_analysis"
    }

POST /api/analysis/run creates a hidden project-local selection at outputs/baseline_signal_analysis/.web_jobs/<job-id>/selection.json, runs analyze_baseline_temporal_signals.py on the four selected inputs (including repeated --rynnvalue-run values when RynnValue is multi-source), and atomically moves a complete result into outputs/baseline_signal_analysis/web_<timestamp>_<label>_<job-suffix>/. It only reads baseline outputs and annotations. `GET /api/analysis-jobs/<job-id>` and `/log` expose status and recent output; after success the browser rereads `GET /api/analysis` and shows the new snapshot freshness/source metadata.

If the current scope is empty or any method has no compatible run, the button stays disabled with the missing method/scope reason. The default `primary_natural` scope covers the fair 125-rollout comparison. `natural_observation` includes both natural dataset roles and is an explicit broader selection; controlled data remain excluded until `controlled_analysis` is selected explicitly.

### Timeline and font-scale behavior

The video slider, timeline pins, and each baseline SVG use the same `.timeline-track` inset and the same normalized frame domain (`0` through `total_frames - 1`). Curves and markers are clamped to the track bounds, and baseline charts no longer use a negative horizontal margin. The SVG keeps its own viewBox typography while the surrounding layout can shrink or scroll horizontally.

The Settings page changes the global root rem scale from 75% through 160% in 5% steps and independently adjusts Review, Analysis, and control/help category scales from 85% through 130%. It uses responsive rem-based layout rules, not CSS zoom, and applies Compact, Comfortable, or Spacious density.

## Analysis environment

The host server may use the system Python, but temporal analysis always uses the project-local interpreter:

    conda_envs/LF3R-ananlyse/bin/python

The directory name intentionally preserves the requested ananlyse spelling. The host has no Conda, so the environment is a uv-managed Python 3.10.21 venv with pinned numpy==2.2.6, pandas==2.3.3, and matplotlib==3.10.9. Recreate or repair it with:

    bash tools/lf3r_annotator/setup_analysis_env.sh

project_env.sh exports LF3R_ENV_ANALYSE and LF3R_ANALYSIS_PYTHON. The server preserves the venv entrypoint symlink; do not replace it with the resolved uv-cache interpreter. MPLBACKEND=Agg is set for every Analysis job, so the temporal analyzer is CPU-only and does not trigger baseline inference. GET /api/health reports dependency readiness. If the environment is missing or an import fails, the Analysis run button is disabled and POST /api/analysis/run returns 503 instead of falling back to sys.executable. See environment_reports/ANNOTATOR_ANALYSIS_ENV_20260828.md for the validated package freeze.

## Persistent jobs and concurrency

Every web-started baseline, Analysis, or rollout-generation command runs in its own tmux session named lf3r-annotator-<job-id>. A wrapper records stdout/stderr, exit code, completion time, full argv, interpreter, working directory, log path, and tmux state. Job records stay under logs/annotator_jobs/, and GET /api/jobs returns the persistent list with optional job_type and status filters.

Baseline method jobs are independent. Submit SAFE, ProcVLM, RynnValue, and Robo-Dopamine separately to run them in parallel; within each batch, the worker table can also split rollout ranges across independent workers. GPU IDs and memory competition are the user's responsibility. The memory-only checks in the existing runners remain unchanged, and GPU utilization is informational.

Temporal Analysis may run while baseline jobs are still active, but it validates only completed baseline output directories that cover the selected rollout IDs. Rollout generation remains exclusive because it writes media and rebuilds the manifest; an active generation job returns 409 for a competing baseline or Analysis request, and vice versa. The UI restores active jobs from /api/jobs after a page refresh and exposes each job's tmux session and recent-log action. If tmux is unavailable, the server returns a clear 503 and never falls back to an ordinary detached process.

## Data partitions

The manifest deliberately prevents injected trajectories from being counted as natural failures:

- `natural_policy / natural_observation`: no action or environment intervention;
- `controlled_injected / controlled_analysis`: known interventions for controlled analysis only.

LIBERO-10 natural rollouts carry `dataset_role=primary_natural`. Existing LIBERO-Spatial natural runs are `reference_natural`; injected runs are always `controlled_analysis`.

Rebuild the manifest after generating rollouts:

    python3 tools/lf3r_annotator/build_manifest.py

The versioned manifest and summary are stored under `datasets/lf3r_failure_rollouts/v1/`.

### Manifest construction

build_manifest.py recursively scans both outputs/openvla_libero/ and outputs/openvla_libero_spatial_native/ by default (or explicit repeated --scan-root values) for filenames matching taskN--epN--succ0|1.mp4. 
For every recognized rollout it:

1. derives task, episode, and evaluator outcome from the filename;
2. identifies provenance from the run directory name;
3. uses `ffprobe` to record frame count, FPS, and duration;
4. reads the companion CSV for first and last environment timesteps;
5. attaches task metadata, checkpoint identity, and a stable rollout ID;
6. atomically rewrites `manifest.jsonl` and `summary.json`.

Unknown run-name provenance is skipped instead of guessed. Only registered controlled run names receive injection metadata. A natural LIBERO-10 record becomes `primary_natural`; natural rollouts from other suites remain `reference_natural`.

### LIBERO-10 instruction-variant diagnostic manifest

The separate `tools/lf3r_annotator/instruction_variants/libero_10_v1/manifest.jsonl` contains the original `full_instruction` rows and validated `subtask_a`/`subtask_b` rows for the compatible two-goal LIBERO-10 tasks. It preserves each source rollout's video and provenance, marks counterfactual rows with `instruction_variant` and `instruction_type`, and assigns them the isolated output namespace `outputs/baselines/instruction_variants/libero_10/`. A/B labels are canonical task/object labels, not observed execution order. Task 5 is intentionally full-instruction-only because its official BDDL has one unique goal atom. Rebuild or validate it with:

    python3 tools/prepare_libero10_instruction_variants.py
    python3 tools/prepare_libero10_instruction_variants.py --check-only

This preparation step is read-only with respect to the source manifest, media, annotations, and existing baseline outputs; it does not run inference.

## Generate natural LIBERO-10 data

The generation wrapper enforces the current memory-only GPU gate and refuses to overwrite a run. GPU utilization is informational and does not block a run; GPU memory must remain below 50% used with at least 30 GiB free. It accepts a single GPU id, inclusive task range, trials, optional seed, and optional generated run note:

    bash tools/lf3r_annotator/generate_libero10_natural.sh 0 0 3 1 7 lf3r-data-natural-libero10-manual

The six positional arguments are:

1. GPU index;
2. inclusive first task index;
3. inclusive last task index;
4. trials per selected task;
5. seed (default 7);
6. run note (optional; the web service always supplies the required natural prefix).

Thus the example selects tasks 0, 1, 2, and 3 and produces at most four episodes. `... 0 4 4 50` would select only task 4 and request all 50 indexed initial states; it is an explanation of argument semantics, not a recommendation to run more data.

### Generation module flow

`generate_libero10_natural.sh` is the resource and environment wrapper. Headless MuJoCo rendering is explicitly configured with MUJOCO_GL=egl and PYOPENGL_PLATFORM=egl; the wrappers do not use the failing OSMesa backend. It verifies the GPU-memory gate, creates a unique natural-provenance run name and timestamped log, selects the project-local OpenVLA environment, disables external W&B reporting, and invokes the Python runner. If inference succeeds, it rebuilds the manifest. Existing output directories are never overwritten.

`run_openvla_libero10_natural.py` validates task indices (`0–9`), trials (`1–50`, matching available initial-state indices), checkpoint presence, and the natural run-name prefix. It installs a process-local no-op W&B module so `--use_wandb=False` does not require a working W&B installation; this does not modify OpenVLA source or package versions. It then delegates evaluation to the official `experiments/robot/libero/run_libero_eval.py` entry point with:

- `task_suite_name=libero_10`;
- the dedicated `openvla-7b-finetuned-libero-10` checkpoint;
- `use_wandb=False` and `save_logs=True`;
- no action injection or environment intervention;
- eager attention enabled; direct CLI hidden-state output is opt-in, while the web Generate rollout form enables SAFE latent saving by default and exposes a checkbox to disable it.

The web request field is `log_safe_features` (default `true`). When enabled, the selected wrapper passes `--log-safe-features`, which forwards `output_hidden_states=True` to the official evaluator and writes the official hidden-state artifact plus LF3R numeric sidecars. When disabled, generation keeps only the normal video/CSV/log outputs.

The evaluator writes one MP4 and one per-step CSV per completed episode under `outputs/openvla_libero/<run-name>/libero_10/`. The `succ0`/`succ1` suffix is the evaluator outcome, not a human annotation.

### Optional SAFE OpenVLA feature logging

Ordinary rollout generation leaves hidden-state logging disabled. To produce the
representation consumed by the SAFE OpenVLA loader, pass the opt-in flag to the
LF3R wrapper:

    $LF3R_ENV_OPENVLA/bin/python \
        tools/lf3r_annotator/run_openvla_libero10_natural.py \
        --task-suite libero_10 --task-start 0 --task-end 0 --trials 1 \
        --run-note lf3r-data-natural-libero10-safe-features \
        --seed 7 --log-safe-features

The flag forwards output_hidden_states=True to the official evaluator already
present in the local safe-openvla checkout. The web Generate rollout form passes
this flag by default; direct CLI and shell-wrapper users must opt in explicitly.
After evaluation, the LF3R-owned wrapper reads the official .pkl files and writes the numeric sidecars below;
there is no evaluator-side save_safe_features option. It does not change the
policy sampling, image preprocessing, action post-processing, termination, or
success checks.
The official SAFE artifact remains a same-basename .pkl containing
(T, 7, 4096) hidden states: for each generated action token, the last
transformer layer at the last sequence position. SAFE's existing
failure_prob/data/openvla.py can load that file directly and applies its
configured token_idx_rel selection (the default 1.0 yields (T, 4096)).

When enabled, each episode also receives:

- <episode>.safe_features.npz: compressed numeric arrays for hidden_states,
  policy-step index, environment timestep, replay-video frame index, executed
  7-DoF action, task/episode numeric identifiers, and the final success label;
- <episode>.safe_features.json: lightweight provenance and alignment metadata,
  including checkpoint, feature-selection rule, tensor shape/dtypes, image
  preprocessing, and source git identity.

The sidecar action is the post-normalization/post-gripper-inversion vector
actually passed to env.step. There is one row per policy decision. Dense
features are never stored in JSON. The optional flag is intended for a single-rollout validation first; no existing
rollout or annotation is modified. The shell wrapper accepts the same mode as
its optional seventh argument, `--log-safe-features`.

It uses the official OpenVLA LIBERO-10 checkpoint and evaluator with no action or environment intervention. Outputs are named lf3r-data-natural-libero10-<timestamp>-<label>, logged under logs/, and added to the manifest only as primary_natural. Exit code 75 with WAITING_FOR_GPU_MEMORY means the memory gate did not pass; do not bypass it.

## Generate native LIBERO-Spatial data

The project already contains the LIBERO-Spatial BDDL files, initialization files, ten-task suite registration, and the local openvla-7b-finetuned-libero-spatial checkpoint. The old outputs/openvla_libero/lf3r-feasibility-natural/libero_spatial/ sample is retained as reference data and is not modified.

Use the separate wrapper when you want a new native-resolution run:

    bash tools/lf3r_annotator/generate_libero_spatial_native.sh 0 0 3 1 7 lf3r-data-natural-libero-spatial-256-manual

Its positional arguments are the same as the LIBERO-10 wrapper: GPU, inclusive task start, inclusive task end, trials per task, seed, and a run note. The run note must start with lf3r-data-natural-libero-spatial-256-. Outputs are written only below outputs/openvla_libero_spatial_native/<run-note>/libero_spatial/; an existing run directory is never overwritten.

This path uses the existing Spatial OpenVLA checkpoint and official evaluator. The simulator camera and replay video are native 256x256, while the policy preprocessing remains the official 224x224 get_image_resize_size path. The evaluator keeps the policy image separate from the recorded replay image, so changing the replay resolution does not change model input semantics. The resulting manifest rows are classified as reference_natural because LIBERO-Spatial is not the primary LIBERO-10 dataset.

The Review rollout-generation form exposes the same choice as task_suite=libero_spatial. It reports the suite, output root, and 256/224 resolution metadata in the persistent job record. This change only prepares the command/UI path; no Spatial rollout generation is started automatically.

### Rollout generation from Review

The Review page's Generate rollouts panel is a browser form over the existing suite-aware wrapper -> runner -> official evaluator -> build_manifest.py flow. Select either OpenVLA + LIBERO-10 natural or OpenVLA + LIBERO-Spatial natural native-256 data. Both suites accept one numeric GPU id, task start/end 0-9 inclusive, 1-50 trials per task, a non-negative seed, and an optional short label. The service creates a suite-specific provenance prefix, rejects path-like labels and existing output directories, keeps each run under its dedicated output root, and refreshes the manifest and Review queue after success.


POST /api/rollouts/generate starts the selected suite; GET /api/rollout-jobs/<job-id> and /log expose progress and recent output. A web submission is passed to the same TmuxJobSupervisor.submit() used by baseline jobs: it creates lf3r-annotator-<job-id>, runs a project-local wrapper, persists the job record, and reattaches monitoring after a server restart. The direct shell wrapper remains a normal CLI entry point; only web-started jobs are tmux-managed. Generation is the manifest/media writer and is mutually exclusive with any active baseline or temporal-analysis job. SAFE, ProcVLM, RynnValue, and Robo-Dopamine jobs may otherwise be submitted independently, including with overlapping user-specified GPU IDs; the server does not schedule or reject GPU contention. Jobs expose task suite, resolution metadata, requested/completed counts, run root, command, log, tmux session, return code, manifest rebuild state, and the memory-only gate. Failed or memory-blocked jobs leave existing rollout and annotation files untouched. The browser has no cancel action; command-line recovery remains supported.


Controlled injected rollouts use a separate run-name registry in `build_manifest.py`, always receive `analysis_partition=controlled_analysis`, and must not be used to estimate natural failure rates.

## Validation

Run the standard-library test suite:

    python3 -m unittest discover -s tools/lf3r_annotator/tests -p 'test_*.py' -v
    python3 -m py_compile tools/lf3r_annotator/server.py tools/lf3r_annotator/task_supervisor.py tools/lf3r_annotator/verify_pipeline.py tools/analyze_baseline_temporal_signals.py

The tests cover manifest loading, byte-range video delivery, atomic annotation persistence, validation failures, project-root path confinement, parsing the four baseline output formats through the evaluation API, shared Settings round trips/validation, complete-snapshot selection, event-metric compaction, baseline run discovery/batch validation/job progress, temporal-analysis selection/atomic output, rollout-generation validation/progress/logging/memory blocking, persistent tmux job records and recovery behavior, parameter metadata, and the three-page frontend contract.

The requirement-level verifier can be run after data generation:

    python3 tools/lf3r_annotator/verify_pipeline.py

`verify_pipeline.py` checks primary dataset size and outcomes, natural/controlled partition integrity, MP4 frame counts, LIBERO-10 checkpoint normalization metadata, required annotation-tool files, and project-local storage. Its historical small-smoke gate expects 3–12 primary LIBERO-10 rollouts; the current expanded 125-rollout primary manifest therefore reports that size check as expected. It is read-only with respect to rollouts and annotations. The unit tests separately exercise the frontend contract, byte-range playback, atomic save/reload, Settings/Analysis APIs, invalid onset ordering, and path confinement.
