# Robo-Dopamine multi-perspective sanity report — 2026-09-05

## Scope

This report covers the LF3R wrapper upgrade only. It did not rerun the full
baseline dataset and it did not modify any existing forward-only raw output.
The test used one labeled clean rollout and one labeled failure rollout:

- clean: `libero_10-task00-ep000-natural-07acb763b6` (414 frames)
- failure: `libero_10-task00-ep001-natural-477e2f3af1` (520 frames)

Output root:

`outputs/baselines/robo_dopamine_multi_sanity_20260905/robo_dopamine_20260905_201426_242626/`

The run completed `2/2`, with `frame_interval=10`, `batch_size=1`, GPU `1`,
blank goal image, and requested free-memory fraction `0.6` on the retry. The
first attempt at `0.8` was correctly rejected by vLLM because available memory
fell from 45.9 GiB at the runner query to 30.55 GiB at engine startup, below
the requested 35.65 GiB. GPU utilization was not used as a blocking condition.

## Official state-pair construction

The local official source is Robo-Dopamine commit
`2c714abca66329f8e2113587226bc9556726f6c2`.
`examples/inference.py` creates one shared native grid with
`range(0, total_frames, frame_interval)` and appends the final frame when it is
not already present. The first grid point (frame 0) is the reference/start
point; the emitted prediction rows are the transitions to each subsequent
sampled AFTER index. Thus, for 414 frames at interval 2, the native grid has
208 points (`0, 2, ..., 412, 413`) and each mode emits 207 prediction rows
(`2, 4, ..., 412, 413`). For every sampled AFTER index it passes eight images:

1. reference start (front camera, frame 0);
2. reference goal;
3. BEFORE high/front;
4. BEFORE left wrist;
5. BEFORE right wrist;
6. AFTER high/front;
7. AFTER left wrist;
8. AFTER right wrist.

The BEFORE triplet is mode-specific:

- `incremental`: the preceding native sampled index;
- `forward`: the first sampled frame;
- `backward`: the goal image repeated for all three BEFORE views.

LF3R has one video stream, so the wrapper passes the same video path for all
three camera arguments, exactly as the existing LF3R Robo-Dopamine path did.
The checkpoint is loaded once into one `GRMInference` object; the worker then
calls its existing `run_pipeline` method once per mode and never reloads the
model between modes or rollouts.

## Goal image and fusion

The official README explicitly recommends `examples/blank_goal.png` when no
target image is available. The official code requires only a non-null goal
path and uses that path in backward mode as the BEFORE triplet, so the blank
goal is valid for all three modes in this test.

The official repository does not contain an executable fusion function. Its
README recommendation is to average the three inference reward results. LF3R
implements the requested fused progress signal as the documented equal-weight
mean at the exact matching native frame indices:

```text
fused_progress[t] =
    (incremental_progress[t] + forward_progress[t] + backward_progress[t]) / 3
```

No interpolation, padding, or resampling is performed. The wrapper rejects a
mode-grid mismatch. Each official `pred_vllm.json` remains separate; the
wrapper-derived files are `multi_perspective/fused_progress.json`,
`progress_curves.csv`, `progress_curves.png`, and `metadata.json`. The fused
JSON also retains component progress and component official hop values. Its
`hop` is the finite difference of fused progress from the same zero-origin
convention, so this distinction is explicit rather than silently replacing
native mode outputs.

## Validation results

Both rollouts had equal native lengths and frame indices for all three modes
and the fused output:

| Rollout | Incremental / forward / backward rows | Fused rows | Frame grid | finite | max mean error |
| --- | ---: | ---: | --- | --- | ---: |
| clean | 42 / 42 / 42 | 42 | identical | yes | `5.6e-17` |
| failure | 52 / 52 / 52 | 52 | identical | yes | `1.1e-16` |

The four-curve plots are saved under each rollout’s `raw/<rollout-id>/multi_perspective/progress_curves.png`.
The compact numeric tables are in the corresponding `progress_curves.csv`.

Runtime after the single engine initialization:

| Rollout | incremental | forward | backward | three-mode total | total / forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 106.0 s | 73.3 s | 71.6 s | 250.9 s | 3.42x |
| failure | 124.4 s | 90.4 s | 89.1 s | 303.9 s | 3.36x |

The one-time engine initialization was 236.2 s and is not repeated for the
three modes. The three-mode wall time is therefore about 3.4x the forward-only
inference time for this batch-size-1 test, plus the same one-time initialization
cost.

For a descriptive stability check, the standard deviation of consecutive
progress differences was:

| Rollout | forward | backward | incremental | fused | fused reduction vs forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.0726 | 0.0903 | 0.0118 | 0.0409 | 43.7% |
| failure | 0.0626 | 0.0336 | 0.0091 | 0.0261 | 58.3% |

On these two examples, fused is numerically smoother than forward-only by this
measure. This is not enough to claim detector performance or general stability;
it is only the requested sanity comparison.

### Frame-interval sweep

After the two-rollout interval-10 validation, a separate one-model sweep was
run on the clean rollout only. It used the same checkpoint, GPU 1,
`batch_size=1`, blank goal, and one persistent engine for interval 2, 5, and
10. The sweep output and comparison plot are under:

`outputs/baselines/robo_dopamine_interval_sanity_20260905/interval_sweep_20260905_204010_369904/`

| interval | native grid / emitted rows | incremental diff std | incremental mean abs diff | sign-change rate | three-mode seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 208 / 207 | 0.00343 | 0.000710 | 0.49% | 1176.1 |
| 5 | 84 / 83 | 0.00810 | 0.00480 | 2.47% | 447.97 |
| 10 | 43 / 42 | 0.01238 | 0.01071 | 2.50% | 213.04 |

The interval comparison artifact is
`comparison/interval_comparison.json`, with the plot in
`comparison/interval_comparison.png`. On this clean rollout, interval 2 does
not show excessive incremental noise by the raw sampled-step measures: its
absolute difference and sign-change rate are lower than intervals 5 and 10.
That comparison is not scale-neutral because interval 2 has shorter temporal
steps. Dividing `diff_std` by the interval gives approximately `0.00172`,
`0.00162`, and `0.00124` per source frame for 2, 5, and 10 respectively, so
interval 2 is only modestly higher on that rough normalization, not an
explosive noise regime. This is one clean rollout and a descriptive sampling
check, not a detector-performance conclusion.

The interval report tool is
`tools/baselines/robo_dopamine_interval_sanity.py`; the dedicated inference
runner is `tools/baselines/run_robo_dopamine_interval_sanity.py`. No full
dataset rerun was performed.

## Compatibility and tests

- Existing `--robo-eval-mode forward` remains unchanged.
- New CLI option: `--robo-eval-modes incremental forward backward`.
- Raw outputs are never overwritten; multi-mode output is placed under the
  selected run’s per-rollout directory.
- `worker_result.json` and aggregate `jobs.jsonl` record mode paths, checkpoint,
  source commit, fusion metadata, and per-mode timing.
- No new head or checkpoint was introduced.
- Fake-model/native-grid fusion tests pass, existing persistent-worker and
  baseline worker contracts pass, and modified Python files compile.
