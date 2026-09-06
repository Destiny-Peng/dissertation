# LF3R Baseline Execution

These wrappers run existing baseline implementations over the versioned LF3R manifest without changing upstream repositories or environments. Every invocation creates a unique timestamped run directory, a timestamped combined log, an exact command record, per-rollout status records, and raw baseline outputs.

## Supported methods

| Baseline          | Existing environment              | Raw output                                                      |
| ----------------- | --------------------------------- | --------------------------------------------------------------- |
| `safe`          | `conda_envs/LF3R-safe`          | Official SAFE handcrafted OpenVLA feature CSV                   |
| `procvlm`       | `repos/ProcVLM/.venv`           | JSONL including each unmodified`model_output`                 |
| `rynnvalue`     | `repos/RynnValue/.venv`         | Value-head samples, frame indices, and generated analysis JSON  |
| `robo_dopamine` | `conda_envs/LF3R-robo-dopamine` | Official`pred_vllm.json`, including raw `<score>` responses |

SAFE publishes no trained detector checkpoint in the local checkout, so `safe` intentionally runs only its official handcrafted OpenVLA signals. FAIL-Detect is not exposed as an LF3R runner: its official policy/UQ checkpoints and required policy embeddings are absent. The earlier synthetic schema smoke is not presented as a reproduced baseline.

## Command interface

All examples use:
The annotator web forms use the same runner interface. Their canonical tooltip metadata is stored in tools/lf3r_annotator/static/parameter_help.json, including the exact CLI spelling, default, effect, and upstream forwarding flag; update that file when the web mapping changes so the UI and this command reference do not drift.


```bash
bash tools/baselines/run_baseline.sh [OPTIONS]
```

`run_baseline.sh` only loads the project environment and forwards options to `run_lf3r_baseline.py`. The runner validates the manifest and local paths, then launches one persistent worker for a legacy single-worker request or multiple independent rollout workers when `--parallel-workers`/`--worker-spec` is supplied. SAFE workers execute per-rollout extraction, while ProcVLM and Robo-Dopamine workers each own an independent persistent model engine. No extra persistence flag is required. Use `python3 tools/baselines/run_lf3r_baseline.py --help` for the argparse-generated option list.

### Required, path, and selection parameters

| Option                  | Default                 | Applies to                        | Meaning and effect                                                                                                                                                                        |
| ----------------------- | ----------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--baseline NAME`     | required                | all                               | Selects`safe`, `procvlm`, `rynnvalue`, or `robo_dopamine`. This chooses the existing repository, Python environment, checkpoint default, worker command, and raw-output contract. |
| `--manifest PATH`     | required for new runs  | all                               | Versioned LF3R JSONL manifest. Relative media and CSV paths in each record are resolved under `--data-root`; it is restored from `run.json` when `--resume-run` is used.                                                                             |
| `--data-root PATH`    | project root            | all                               | Root used for relative`video_path` and `csv_path` values. Source media are read in place and are not copied or modified.                                                              |
| `--output-dir PATH`   | required for new runs  | all                               | Parent directory for a new timestamped run directory. The runner refuses to overwrite an existing timestamped directory; it is inferred from the saved run when resuming.                                                                  |
| `--logs-dir PATH`     | `logs/baselines`      | all                               | Directory for the timestamped combined stdout and stderr log. |
| `--resume-run PATH`   | unset                 | ProcVLM, Robo-Dopamine                     | Reopens an existing persistent run directory, reads its saved baseline-specific job plan, and runs only rollouts whose status is not terminal. The manifest hash and local model paths are rechecked. |
| `--model-path PATH`   | configured checkpoint   | ProcVLM, RynnValue, Robo-Dopamine | Overrides the local checkpoint configured for the selected baseline. SAFE has no configured model checkpoint; do not pass this option for SAFE.                                           |
| `--partition NAME`    | `natural_observation` | all                               | Selects`natural_observation`, `controlled_analysis`, or `all`. The default excludes injected and controlled-analysis rollouts.                                                      |
| `--dataset-role ROLE` | unset                   | all                               | Exact match against the manifest field`dataset_role`, for example `primary_natural`.                                                                                                  |
| `--rollout-id ID`     | unset                   | all                               | Selects a rollout by ID. Repeat the option for several IDs; manifest order is preserved. IDs are checked against the full manifest before other filters.                                  |
| `--start-index N`     | `0`                   | all                               | Drops the first`N` records after partition, dataset-role, and ID filtering. Must be non-negative.                                                                                       |
| `--end-index N`      | scope end              | all                               | Exclusive end of the scope-relative range. The selected interval is `[start-index,end-index)`. It is mutually exclusive with`--limit`.                                                                                              |
| `--limit N`           | unset                   | all                               | Keeps at most`N` records after the preceding filters. Must be positive. Use with `--start-index` to schedule reproducible chunks.                                                     |

Selection is applied in this order: `partition → dataset role → rollout IDs → start index → limit`. An empty result is an error rather than a successful no-op.

For model baselines, omitting `--model-path` selects the configured local checkpoint: `checkpoints/ProcVLM-2B`, `checkpoints/RynnValue-4B`, or `checkpoints/Robo-Dopamine-GRM-2.0-4B-Preview`. SAFE uses the manifest `csv_path` and its official handcrafted feature code instead of a trained checkpoint. Each selected model record must provide `id`, `video_path`, and `task_description`; SAFE records must provide `id` and `csv_path`.

### Execution, GPU, and failure-handling parameters

| Option                     | Default | Applies to               | Meaning and effect                                                                                                                                                                                                                                                |
| -------------------------- | ------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--gpu ID[,ID...]`       | `0`   | all                      | Sets the GPU list used by legacy single-worker mode or automatic rollout-worker assignment. Explicit worker rows use one numeric ID each; IDs may repeat and are not conflict-checked. For a legacy single ProcVLM/Robo-Dopamine worker, a comma list remains available for tensor parallelism. |
| `--dry-run`              | off     | all                      | Validates selection and paths, then writes`run.json`, `commands.jsonl`, and planned `jobs.jsonl` without loading a model or creating `raw/`. Free-memory conversion is marked deferred because a real run measures memory immediately before its persistent vLLM worker. |
| `--validate-environment` | off     | all                      | Imports the selected baseline core packages in its configured existing environment before any rollout worker starts. This is a preflight and does not perform model inference.                                                                                    |
| `--continue-on-error`    | off     | all                               | Worker-mode jobs continue other rollout shards after an ordinary failure. Persistent ProcVLM and Robo-Dopamine workers also continue ordinary per-rollout failures; an engine-fatal event exits for resume. |
| `--render-video`         | off     | RynnValue, Robo-Dopamine | Enables the optional upstream visualization video without changing raw-output capture. The top-level option is accepted for SAFE and ProcVLM but is not forwarded by their wrappers.                                                                              |

### vLLM memory parameter

| Option                                  | Default  | Applies to             | Meaning and effect                                                                                                                                                                                                                                                           |
| --------------------------------------- | -------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--vllm-free-memory-fraction VALUE`   | `0.80` | ProcVLM, Robo-Dopamine | Target fraction of GPU memory that is free at worker startup. It must be in`(0, 1]`. The runner reads current `nvidia-smi` `memory.free` and `memory.total` immediately before the persistent vLLM worker, then converts the target to the total-memory fraction expected by vLLM. |
| `--vllm-gpu-memory-utilization VALUE` | alias    | ProcVLM, Robo-Dopamine | Backward-compatible spelling for the same free-memory target. Despite the legacy name, the runner does not interpret this value as a fraction of total GPU memory.                                                                                                           |

The conversion is:

```text
vllm_total_fraction = floor_6(
    requested_free_fraction × min(selected_gpu.free_mib / selected_gpu.total_mib)
)
```

For example, with 71,842 MiB free out of 97,887 MiB total on GPU0, `0.80` becomes `0.587142` for vLLM. This requests about 80% of the memory free at measurement time and leaves the remaining free-memory fraction as headroom for other workloads. With multiple selected GPUs, the smallest free/total ratio is used. GPU utilization percentage is recorded for monitoring only and is not a gate. A failure to query memory or an invalid GPU selection stops before model loading.

The resolved value is recorded in `commands.jsonl` under `vllm_memory_budget`, together with the requested free-memory fraction and per-GPU snapshot. The raw downstream command therefore contains a converted `--gpu_memory_utilization` or `--gpu-memory-utilization` value; that downstream value is a total-memory fraction, not the user-facing free-memory target. ProcVLM and Robo-Dopamine resolve this once per persistent worker; each engine is then reused for that worker's assigned rollout set. GPU utilization percentage is informational and is not a gate.

### ProcVLM parameters

| Option                             | Default  | Forwarded upstream as      | Meaning and effect                                                                                                                                                                                                |
| ---------------------------------- | -------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--procvlm-window-size N`        | `4`    | `--window_size N`        | Number of sampled images supplied in each temporal inference window. Larger windows provide more temporal context but increase prompt size, work, and memory use.                                                 |
| `--procvlm-max-sampled-frames N` | unset    | `--max_sampled_frames N` | Optional cap on frames sent to ProcVLM. If omitted, the upstream default of`512` remains active; a longer video is uniformly sampled to that cap. This is a frame-count cap, not a window size or fixed stride. |
| `--procvlm-max-new-tokens N`     | `4096` | `--max_new_tokens N`     | Maximum generated tokens for each frame/window response. Higher values increase worst-case generation time and KV-cache demand.                                                                                   |
| `--dtype TYPE`                   | `bf16` | `--torch_dtype TYPE`     | ProcVLM model and inference dtype string. The dense-sampling configuration uses`bf16`.                                                                                                                          |
| `--tensor-parallel-size N`       | `1`    | `--tp N`                 | vLLM/model tensor-parallel degree. For example, pair`--gpu 0,1` with `--tensor-parallel-size 2`; this is independent of temporal frame sampling.                                                              |

With the defaults, a 414-frame video produces 414 ProcVLM records, while a 520-frame video is limited by the upstream default to 512 records. The `window_size=4` context is retained in each ProcVLM raw sample as `window_frame_indices`.

### ProcVLM persistent execution and resume

ProcVLM uses one worker process and one cached vLLM engine and processor for the selected rollout set. Rollouts are processed sequentially, and each result is atomically upserted into `jobs.jsonl`; `procvlm_progress.jsonl` records engine initialization, rollout timing, and fatal events. A normal parsing or inference error marks only that rollout as `failed` and the worker continues. CUDA OOM, a dead vLLM engine, or another classified engine-fatal error marks the current rollout as `interrupted`, saves state, exits with code `70`, and leaves later jobs pending. Resume it with:

```bash
bash tools/baselines/run_baseline.sh --baseline procvlm --resume-run outputs/baselines/procvlm_<timestamp>
```

Resume uses the saved plan and skips `complete` and ordinary `failed` jobs; it reruns `interrupted` and unstarted jobs. It does not create a second run directory. `run.json` records `resume_count`, engine initialization time, per-rollout `inference_seconds`, memory budgets, and completed, failed, and pending counts.

### RynnValue parameters

| Option                        | Default                                   | Forwarded upstream as         | Meaning and effect                                                                                                                                            |
| ----------------------------- | ----------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--rynn-num-frames N` | `16` | `--num_frames N` | Number of frames uniformly resampled inside each causal prefix. Zero/all-frame-per-prefix mode is disabled because variable-length prefixes can exhaust GPU memory. |
| `--rynn-evaluation-interval K` | `8` | per-rollout `--num_steps N` | Target source-frame spacing. Before each rollout command, the LF3R runner computes `N = ceil((T-1)/K) + 1`, matching the count of `0, K, 2K, ...` plus the final frame, then passes `N` to the official CLI. The official sampler distributes these endpoints uniformly over the full frame domain, so the realized interval is approximate. For example, 414 frames and `K=8` produce 53 samples. |
| `--rynn-num-steps N` | deprecated | `--num_steps N` (computed) | Legacy compatibility option. The runner does not use this value; per-rollout endpoint count is derived from `--rynn-evaluation-interval`. |
| `--rynn-batch-size N` | `1` | `--batch_size N` | Number of prefix sub-samples per forward pass. Any positive value is accepted; larger values may improve throughput but increase peak GPU memory, so choose it according to the available memory. |
| `--rynn-max-image-side N`   | `448`                                   | `--max_image_side N`        | Resizes each frame so its longer side is at most`N` pixels before inference. Lower values reduce image memory and compute at the cost of detail.            |
| `--rynn-max-new-tokens N`   | `128`                                   | `--max_new_tokens N`        | Maximum generated tokens for the Analysis text block. It does not change the number of value-head samples.                                                    |
| `--robot-description TEXT`  | Franka Panda description                  | `--robot_description TEXT`  | Robot embodiment text inserted into RynnValue metadata. Keep it consistent with the data and model assumptions unless prompt sensitivity is being tested.     |
| `--camera-description TEXT` | fixed third-person agent-view description | `--camera_description TEXT` | Camera viewpoint text inserted into metadata. It describes the source video and does not select a physical camera file.                                       |

The wrapper saves sampled indices, value-head outputs, analysis text, parsed analysis, and model/video metadata in `raw_model_outputs.json`. RynnValue raw output additionally records `sampling_mode=approximate_fixed_interval_prefix_uniform`, the requested `evaluation_interval`, and the computed `num_steps`; existing `values` and `sampled_indices` arrays remain unchanged for temporal analysis.

### Rollout-level parallelism for all methods

`--parallel-workers` and repeated `--worker-spec GPU:START:END` apply to every baseline. The selected range is scope-relative and right-open: `[START,END)`. Filtering still happens in the fixed order `partition → dataset role → rollout IDs → total range → worker shard`. Without explicit rows, the runner divides the total range continuously and assigns any remainder to earlier workers. With explicit rows, each row receives one numeric GPU ID; repeated IDs are allowed, overlapping ranges are deduplicated by first-worker order, and gaps are recorded as partial coverage.

SAFE launches one isolated per-rollout command inside each shard. ProcVLM and Robo-Dopamine launch one independent persistent worker process and model engine per shard, then aggregate their per-worker plans and results under one run root. This means parallelism is at the rollout level, not an additional temporal batch dimension. Keep `--tensor-parallel-size`/`--robo-batch-size`/`--rynn-batch-size` under user control; the runner does not add a GPU-utilization or GPU-ID conflict scheduler. Multiple workers may reuse a GPU, but each model worker can load its own model copy and compete for memory and compute.

Example for two ProcVLM workers:

```bash
bash tools/baselines/run_baseline.sh \
  --baseline procvlm \
  --manifest datasets/lf3r_failure_rollouts/v1/manifest.jsonl \
  --data-root /mnt/hdd/qiuxia/pyr/LF3R \
  --output-dir outputs/baselines \
  --logs-dir logs/baselines \
  --dataset-role primary_natural \
  --start-index 0 --end-index 20 \
  --parallel-workers 2 --gpu 0,1 \
  --continue-on-error
```

The automatic assignment is `[0,10)` on GPU 0 and `[10,20)` on GPU 1. For exact placement, use `--worker-spec 0:0:10 --worker-spec 1:10:20`. The same syntax is valid for SAFE and Robo-Dopamine; the worker table in the annotator sends these flags for all four methods. A legacy invocation with no worker options keeps the existing single-worker behavior, including ProcVLM/Robo-Dopamine multi-GPU tensor parallel configuration.

RynnValue keeps its existing temporal semantics inside each rollout worker: `--rynn-batch-size` is the prefix batch size, and `--rynn-evaluation-interval` is converted to the official sampler's approximate endpoint count. Each RynnValue rollout subprocess receives one `CUDA_VISIBLE_DEVICES` value.

Overlapping explicit ranges are accepted. The first worker in command order owns a repeated rollout; later workers record `duplicate_assignment` and never overwrite the shared `raw/<rollout-id>` directory. Gaps are also accepted and recorded as partial coverage, so the aggregate run finishes as `complete_with_errors` and cannot satisfy a full-scope Analysis selection. One invocation creates one aggregate `run.json`, `jobs.jsonl`, and `commands.jsonl`; each record includes worker index, GPU, range, command, timestamps, and return code.

### Robo-Dopamine parameters

| Option                      | Default                     | Forwarded upstream as  | Meaning and effect                                                                                                                                                                                                                                                                                                                               |
| --------------------------- | --------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--robo-frame-interval N` | `4`                       | `--frame-interval N` | Samples every`N` source frames. The upstream sampler always includes frame 0 and ensures the final source frame is included, so smaller values produce more prediction samples and require more time and memory.                                                                                                                               |
| `--robo-batch-size N`     | `1`                       | `--batch-size N`     | Number of sampled inputs per inference batch. Increase only when GPU memory allows; lower values reduce peak memory.                                                                                                                                                                                                                             |
| `--robo-eval-mode MODE`   | `fused`                   | `--eval-mode MODE`   | `fused` runs `incremental`, `forward`, and `backward` with one persistent model and averages their native progress outputs. Explicit `forward`, `incremental`, or `backward` retains the single-perspective compatibility mode. |
| `--goal-image PATH`       | `examples/blank_goal.png` | `--goal-image PATH`  | Goal/reference image for the official pipeline. If omitted, the blank goal image is used. The wrapper passes the same LF3R video for all three camera streams because LF3R supplies one view.                                                                                                                                                    |

Robo-Dopamine writes the official `pred_vllm.json` under its native timestamped output subdirectory and records that location in `worker_result.json`.

### Robo-Dopamine persistent execution and resume

Robo-Dopamine uses one worker process and one cached `GRMInference` object (including its vLLM engine and processor) for the selected rollout set. Rollouts are processed sequentially with the configured frame interval, batch size, goal image, and evaluation mode. Each rollout gets its own `raw/<rollout-id>/` directory, so the official timestamped output and `pred_vllm.json` remain isolated while the model stays resident. Ordinary parsing or inference errors mark only that rollout as `failed` and continue. CUDA OOM, a dead vLLM engine, or another classified engine-fatal error marks the current rollout as `interrupted`, saves state, exits with code `70`, and leaves later jobs pending.

Resume the same run with:

```bash
bash tools/baselines/run_baseline.sh \
  --baseline robo_dopamine \
  --resume-run outputs/baselines/robo_dopamine_<timestamp>
```

Resume uses `robo_dopamine_jobs.jsonl` as the authoritative plan, skips `complete` and ordinary `failed` jobs, and reruns `interrupted` or unstarted jobs without creating a second run directory. `run.json` records engine initialization time, per-rollout `inference_seconds`, memory budgets, and terminal counts.

### Robo-Dopamine multi-perspective inference

The default command now runs the official three-perspective fused path with one persistent model:

```bash
bash tools/baselines/run_baseline.sh \
  --baseline robo_dopamine \
  --gpu 1 \
  --robo-eval-mode fused \
  --robo-frame-interval 10
```

To run the official three perspectives with one persistent `GRMInference`/vLLM engine, pass all three modes explicitly:

```bash
bash tools/baselines/run_baseline.sh \
  --baseline robo_dopamine \
  --manifest datasets/lf3r_failure_rollouts/v1/manifest.jsonl \
  --data-root . \
  --output-dir outputs/baselines/robo_dopamine_multi \
  --logs-dir logs/baselines/robo_dopamine_multi \
  --dataset-role primary_natural \
  --gpu 1 \
  --vllm-free-memory-fraction 0.8 \
  --robo-frame-interval 10 \
  --robo-batch-size 1 \
  --robo-eval-modes incremental forward backward \
  --goal-image repos/Robo-Dopamine/examples/blank_goal.png
```

`--robo-frame-interval` is still forwarded as the official `--frame-interval`; the official sampler uses `range(0, total_frames, interval)` and appends `total_frames-1` when needed. For each sampled AFTER frame, the official prompt has eight images: reference start, reference goal, three BEFORE views, and three AFTER views. Incremental uses the preceding sampled frame as BEFORE; forward uses the first frame; backward uses the goal image for all three BEFORE views. LF3R has one camera stream, so it passes the same video for high/left-wrist/right-wrist, matching the existing wrapper.

The official README recommends averaging the three inference reward results, but the official repository contains no executable fusion function. LF3R therefore applies that documented rule to the official per-mode `progress` values at exactly matching native frame indices:

```text
fused_progress(frame) = (incremental_progress + forward_progress + backward_progress) / 3
```

No interpolation or resampling is performed. The wrapper writes each native `pred_vllm.json` unchanged, then writes `multi_perspective/fused_progress.json`, `progress_curves.csv`, `progress_curves.png`, and `metadata.json`. The fused `hop` is the finite difference of fused progress from the same initial zero convention; the component official `hop` values remain in `component_hop` and in each raw mode file. `worker_result.json` records checkpoint, mode list, frame interval, goal image, fusion rule, official Robo-Dopamine commit, per-mode paths, and timings.

`blank_goal.png` is the official supported no-goal placeholder. The official code only requires a non-null goal path and passes it through backward mode as the BEFORE image, so the same blank image is valid for incremental, forward, and backward in this LF3R setup.

A small interval report can compare already-generated multi-mode runs without loading the checkpoint again:

```bash
MPLBACKEND=Agg conda_envs/LF3R-ananlyse/bin/python \
  tools/baselines/robo_dopamine_interval_sanity.py \
  --run-root outputs/baselines/robo_dopamine_multi/robo_dopamine_<interval-2> \
  --run-root outputs/baselines/robo_dopamine_multi/robo_dopamine_<interval-5> \
  --run-root outputs/baselines/robo_dopamine_multi/robo_dopamine_<interval-10> \
  --output-dir outputs/baselines/robo_dopamine_interval_sanity
```

The report emits JSON/CSV plus a plot of progress-difference standard deviation. Use it to check whether interval `2` increases incremental hop noise; it is a descriptive sampling diagnostic, not a detector-performance claim. To run a small one-model interval sweep directly, use `tools/baselines/run_robo_dopamine_interval_sanity.py` with repeated `--frame-interval 2 --frame-interval 5 --frame-interval 10` and one or more `--rollout-id` values; it keeps the checkpoint resident and writes separate `interval_<N>/` outputs before the offline comparison. A run now defaults to `fused`, which expands to `--robo-eval-modes incremental forward backward` and produces the fused curve while preserving each native `pred_vllm.json`. To retain the old behavior explicitly pass `--robo-eval-mode forward`.

### Defaults at a glance

| Baseline      | Temporal sampling default                                                  | Batch and memory behavior                                                         |
| ------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| SAFE          | Every row in the input CSV                                                 | No vLLM; official handcrafted feature extraction only                             |
| ProcVLM       | `window_size=4`; no explicit cap, so upstream `max_sampled_frames=512` | vLLM target is 80% of current free GPU memory;`tp=1`                            |
| RynnValue     | `16` frames per prefix; endpoint count derived from `evaluation_interval` | `batch_size=1` by default; any positive value is forwarded to the official worker |
| Robo-Dopamine | `frame_interval=4`; first and last source frames are retained            | `batch_size=1`; vLLM target is 80% of current free GPU memory; `forward` mode |

## Temporal analysis and failure localization

tools/analyze_baseline_temporal_signals.py reads existing raw outputs and annotations; it does not run a baseline model. The analyzer preserves each method's native sample points and aligns them to video-frame coordinates. It supports repeated --rynnvalue-run arguments and merges RynnValue outputs by rollout ID. Duplicate IDs across selected RynnValue roots are rejected; missing selected IDs remain explicit in method_coverage.csv and are excluded from metric denominators.

The high-resolution primary-natural comparison currently uses 125 labeled rollouts:

~~~bash
conda_envs/LF3R-ananlyse/bin/python tools/analyze_baseline_temporal_signals.py \
  --selection outputs/baseline_signal_analysis/highres_primary_selection.json \
  --manifest datasets/lf3r_failure_rollouts/v1/manifest.jsonl \
  --annotations-dir annotations/failure_annotations/v1/records \
  --output-dir outputs/baseline_signal_analysis/highres_primary_<timestamp> \
  --safe-run outputs/baselines/full_136/safe_20260826_212113_540452 \
  --procvlm-run outputs/baselines/full_136_01/procvlm_20260827_190222_580731 \
  --rynnvalue-run outputs/baselines/web_runs/rynnvalue-batch-fe9e05c5ba54/rynnvalue_20260828_222438_749771 \
  --rynnvalue-run outputs/baselines/web_runs/rynnvalue-batch-cc8deb3f2e09/rynnvalue_20260829_221918_087826 \
  --robo-dopamine-run outputs/baselines/full_136_01/robo_dopamine_20260827_200551_687444
~~~

The current generated snapshot is outputs/baseline_signal_analysis/highres_primary_20260830/. SAFE, ProcVLM, and Robo-Dopamine cover 125/125 selected rollouts; RynnValue covers 124/125. The missing RynnValue ID is retained in the coverage metadata and is not filled from the older low-sampling run. The 11 reference_natural rows are excluded.

In addition to response magnitude, persistence, recovery-to-baseline, clean-success Q95 response, and native sampling alignment, the analyzer writes:

- localization_thresholds.csv: clean-success Q90, Q95, and Q99 score thresholds per method/signal;
- localization_event_metrics.jsonl: compact event rows with first native alarm, post-onset alarm, early alarm/miss, signed lead/lag, absolute error, and +/-4/8/16/30 frame flags;
- localization_summary.csv: event recall, clean-success pseudo-event false-alarm rate, precision/F1, AUROC, average precision, and error summaries;
- localization_by_failure_type.csv: the same threshold diagnostics grouped by failure type;
- comparison_with_full_136_20260827.csv: high-resolution versus the prior full snapshot for response, persistence, recovery, and Q95 localization metrics.

The default interpretation is Q95, but all thresholds are reported. A detected event is the first threshold crossing anywhere in the bounded native-sample window; an earlier crossing is separately marked as an early alarm and the first post-onset crossing is retained. Misses have no localization-error denominator. These metrics use clean-success pseudo-events as negatives and are descriptive diagnostics, not independently validated detector-performance conclusions. Run with MPLBACKEND=Agg and conda_envs/LF3R-ananlyse/bin/python for the CPU-only path used by the annotator.


## Local change-point analysis

tools/analyze_baseline_change_points.py provides the primary local-change analysis over the same raw baseline outputs. The legacy global temporal tables are retained only for historical comparison: each native signal is evaluated independently with local level, variance, and slope changes at several half-window scales. The reference distribution combines clean-success trajectories with non-onset regions of event-bearing trajectories, excluding the current scale around annotated observable onsets. Q90/Q95/Q99 detector thresholds are calibrated from clean-success trajectory pseudo-event maxima; non-onset maxima remain a separate within-rollout reference.

The generated primary snapshot is outputs/baseline_signal_analysis/changepoint_primary_20260830/. It uses the 125 primary_natural labeled rollouts, 58 observable-onset events, and scales 8/16/32/64 video frames. SAFE, ProcVLM, and Robo-Dopamine cover 125/125 rollouts; RynnValue covers 124/125, with libero_10-task02-ep005-natural-e8fc18cf25 retained as an explicit unavailable output. RynnValue is logically merged from the two selected aggregate runs by rollout ID, and no older low-sampling output is used as a fallback.

To reproduce it without inference:

~~~bash
MPLBACKEND=Agg conda_envs/LF3R-ananlyse/bin/python tools/analyze_baseline_change_points.py \
  --selection outputs/baseline_signal_analysis/highres_primary_selection.json \
  --manifest datasets/lf3r_failure_rollouts/v1/manifest.jsonl \
  --annotations-dir annotations/failure_annotations/v1/records \
  --output-dir outputs/baseline_signal_analysis/changepoint_primary_<timestamp> \
  --safe-run outputs/baselines/full_136/safe_20260826_212113_540452 \
  --procvlm-run outputs/baselines/full_136_01/procvlm_20260827_190222_580731 \
  --rynnvalue-run outputs/baselines/web_runs/rynnvalue-batch-fe9e05c5ba54/rynnvalue_20260828_222438_749771 \
  --rynnvalue-run outputs/baselines/web_runs/rynnvalue-batch-cc8deb3f2e09/rynnvalue_20260829_221918_087826 \
  --robo-dopamine-run outputs/baselines/full_136_01/robo_dopamine_20260827_200551_687444 \
  --scales 8,16,32,64
~~~

The snapshot writes changepoint_event_metrics.jsonl plus the localization_event_metrics.jsonl compatibility name, summary/reference/failure-type/scale CSVs, localization_summary.csv, localization_by_failure_type.csv, localization_thresholds.csv, native-sampling metadata, three Q95 scale plots, and comparison_with_full_136_20260827.csv. The comparison table uses the high-resolution global temporal snapshot for response/persistence/recovery versus the legacy full snapshot. New local level/variance/slope metrics are marked as non-comparable when no legacy equivalent exists. The report and Analysis page show Q90/Q95/Q99 thresholds, onset-near unusual changes, peak distance, first-threshold errors, tolerance hits, trajectory-level clean-success false alarms, precision/F1, AUROC/AP, and direction only after an unusual change is identified. Native samples are never interpolated or resampled.

## Event-triggered onset analysis

tools/analyze_baseline_event_triggered.py is a complementary, event-aligned view. It aligns every annotated observable_onset_frame to relative frame 0, reports raw and per-instance normalized median/IQR curves, native local level-change scores at 8/16/32/64-frame half-windows, case-versus-matched-clean separation, strongest before/at/after-onset phase, and event-level peak lag. Terminal failures, recovered successes, and uncertain events remain separate; clean-success controls are matched by task suite/task ID when possible and their provenance is recorded.

The checked-in snapshot is outputs/baseline_signal_analysis/event_triggered_primary_20260831/. It contains 125 selected primary-natural rollouts, 58 observable events (48 terminal failures, 9 recovered successes, 1 uncertain), and 59 clean-success controls. SAFE, ProcVLM, and Robo-Dopamine have signal output for 125/125 rollouts; RynnValue has 124/125 and retains libero_10-task02-ep005-natural-e8fc18cf25 as unavailable. Median native intervals are SAFE/ProcVLM 1 frame, Robo-Dopamine 2 frames, and RynnValue 4 frames. This density difference is part of the interpretation and is not hidden by interpolation.

Reproduce the snapshot without GPU inference:

~~~bash
MPLBACKEND=Agg conda_envs/LF3R-ananlyse/bin/python tools/analyze_baseline_event_triggered.py --selection outputs/baseline_signal_analysis/highres_primary_selection.json --manifest datasets/lf3r_failure_rollouts/v1/manifest.jsonl --annotations-dir annotations/failure_annotations/v1/records --safe-run outputs/baselines/full_136/safe_20260826_212113_540452 --procvlm-run outputs/baselines/full_136_01/procvlm_20260827_190222_580731 --rynnvalue-run outputs/baselines/web_runs/rynnvalue-batch-fe9e05c5ba54/rynnvalue_20260828_222438_749771 --rynnvalue-run outputs/baselines/web_runs/rynnvalue-batch-cc8deb3f2e09/rynnvalue_20260829_221918_087826 --robo-dopamine-run outputs/baselines/full_136_01/robo_dopamine_20260827_200551_687444 --output-dir outputs/baseline_signal_analysis/event_triggered_primary_<timestamp>
~~~

The output tables are event_triggered_curves.csv, event_triggered_change_scores.csv, event_triggered_separation.csv, event_triggered_summary.csv, event_triggered_peak_events.csv, event_triggered_controls.csv, and method_coverage.csv; PNG plots are under plots/, and REPORT.md records the matching and native-sampling policy. The analysis only reads raw baseline outputs and annotations and never modifies them. All values are descriptive diagnostics, not independently validated detector or causal-performance estimates.

## Validate without inference

From `/mnt/hdd/qiuxia/pyr/LF3R`:

```bash
python3 tools/baselines/validate_pipeline.py --check-environments --execute-safe-smoke
python3 tools/baselines/test_worker_contracts.py
python3 tools/baselines/test_all_parallel.py -v
```

The first command plans one job for every supported method, imports core packages from each existing environment, and executes only the lightweight SAFE extraction. The second verifies RynnValue and Robo-Dopamine raw-output capture against fake official workers. The third exercises SAFE, ProcVLM, and Robo-Dopamine rollout-worker aggregation with fake subprocesses. None of these checks loads a vision-language model.

To inspect one planned model command:

```bash
bash tools/baselines/run_baseline.sh \
  --baseline procvlm \
  --manifest datasets/lf3r_failure_rollouts/v1/manifest.jsonl \
  --data-root /mnt/hdd/qiuxia/pyr/LF3R \
  --output-dir outputs/baselines \
  --logs-dir logs/baselines \
  --limit 1 \
  --dry-run --validate-environment
```

## Execute a bounded run

Remove `--dry-run` only when the selected GPU has sufficient free memory for the target fraction; GPU utilization percentage is informational. For example:

```bash
bash tools/baselines/run_baseline.sh \
  --baseline rynnvalue \
  --manifest datasets/lf3r_failure_rollouts/v1/manifest.jsonl \
  --data-root /mnt/hdd/qiuxia/pyr/LF3R \
  --output-dir outputs/baselines \
  --logs-dir logs/baselines \
  --rollout-id libero_10-task00-ep000-natural-07acb763b6 \
  --gpu 0
```

The same interface selects `safe`, `procvlm`, `robo_dopamine`, or `rynnvalue`. Natural rollouts are the default. Use `--dataset-role primary_natural`, repeated `--rollout-id`, or `--start-index` plus `--limit` to create reproducible chunks. Pass `--model-path` to override a local checkpoint. Optional visualization is disabled by default; `--render-video` enables it without changing raw-output capture.

The current dense-sampling smoke defaults are deliberate: ProcVLM uses `--procvlm-window-size 4` and omits `--procvlm-max-sampled-frames`, leaving its upstream cap of 512 so short rollouts are sampled densely; `--procvlm-max-sampled-frames` is available only as an explicit optional cap. Robo-Dopamine uses `--robo-frame-interval 4`. Both vLLM-backed workers default to `--vllm-free-memory-fraction 0.80` (the older `--vllm-gpu-memory-utilization` spelling remains an alias). For actual inference, the runner reads current `nvidia-smi` free/total memory immediately before the persistent worker and converts the target to the vLLM total-memory parameter; the measured budget is recorded in `commands.jsonl`. Each selected ProcVLM or Robo-Dopamine run creates one vLLM process and reuses its engine across all selected rollouts. Dry-run plans defer this conversion.

## Run artifacts

Each invocation writes:

```text
outputs/baselines/<baseline>_<timestamp>/
├── run.json          # inputs, manifest SHA-256, repo commit, environment, status
├── commands.jsonl    # exact argv and working directory for every rollout
├── jobs.jsonl        # return codes and discovered raw-output files
├── procvlm_jobs.jsonl      # persistent ProcVLM job plan and command metadata
├── procvlm_progress.jsonl  # append-only engine, rollout, and fatal progress
├── procvlm_state.json      # atomic resumable worker state snapshot
├── robo_dopamine_jobs.jsonl      # persistent Robo-Dopamine job plan and command metadata
├── robo_dopamine_progress.jsonl  # append-only engine, rollout, and fatal progress
├── robo_dopamine_state.json      # atomic resumable worker state snapshot
└── raw/<rollout-id>/ # baseline-native raw outputs

logs/baselines/<baseline>_<timestamp>.log
```

Outputs are never written beside rollout media, and timestamped run directories are created with overwrite protection. The three baseline-specific `procvlm_*` or `robo_dopamine_*` files are created for legacy single persistent runs. Worker-mode ProcVLM/Robo-Dopamine runs place one plan/progress/state group under `workers/worker-*/`; SAFE and RynnValue use the common aggregate files. All worker-mode methods isolate ordinary per-rollout failures and continue other worker ranges.

Full-dataset execution is intentionally not performed by validation. A legacy ProcVLM/Robo-Dopamine invocation uses one persistent worker per selected run, while an explicit worker-mode invocation starts one worker per shard for all four methods; SAFE workers are per-rollout commands and model baselines each load one engine per shard. Use chunks and GPU assignments appropriate to the available memory.

## Useful bash example

```Shell
bash tools/baselines/run_baseline.sh \
  --baseline safe \
  --manifest datasets/lf3r_failure_rollouts/v1/manifest.jsonl \
  --data-root "$PROJECT_ROOT" \
  --output-dir outputs/baselines/full_136 \
  --logs-dir logs/baselines \
  --dataset-role primary_natural \
  --continue-on-error
```
