> Historical note: this report documents the temporary all-frame compatibility mode. It was superseded by the memory-safe approximate fixed-interval configuration in `environment_reports/RYNNVALUE_APPROX_FIXED_INTERVAL_20260828.md`; current workers reject `--rynn-num-frames 0`.

# RynnValue All-Frames Fix Report — 2026-08-28

## Result

`--rynn-num-frames 0` now follows the RynnValue-documented all-frames behavior in the LF3R worker. Each evaluated prefix uses every decoded frame from frame 0 through that prefix endpoint. The upstream checkout under `repos/RynnValue` was not modified.

## Root cause

The previous worker forwarded `--num_frames 0` correctly, but the current upstream `rynn_infer/inference.py` constructs prefix inputs with `np.linspace(0, end_idx, args.num_frames, dtype=int)`. With zero, this produced an empty index array and the processor raised `ValueError: process_episode requires at least one prediction image.` The upstream README and `sample_frame_indices` helper document zero as all frames, but that helper was not used by the current prefix construction path.

## Implementation

- `tools/baselines/rynnvalue_worker.py` installs a module-local NumPy compatibility proxy only when the requested frame count is zero. The zero-count `linspace` call becomes the inclusive range from the prefix start to its endpoint; nonzero NumPy behavior is unchanged.
- All-frame prefixes have different sequence lengths. The worker therefore forwards an effective upstream `--batch_size 1` in this mode to avoid invalid tensor concatenation, while recording both `requested_batch_size` and `effective_batch_size` in `raw_model_outputs.json`.
- The worker validates that `--num-frames` and `--num-steps` are non-negative and documents their zero semantics.

## Verification

- `python3 tools/baselines/test_worker_contracts.py`: `BASELINE_WORKER_CONTRACTS_OK`. The fake official module verifies prefix indices `[0]`, `[0, 1, 2]`, and `[0, 1, 2, 3, 4]`, and verifies effective batch size `1`.
- `python3 tools/baselines/validate_pipeline.py`: `BASELINE_PIPELINE_VALIDATION_OK`. The RynnValue dry-run accepts and forwards `--rynn-num-frames 0` as `--num-frames 0`.
- `python3 -m py_compile tools/baselines/rynnvalue_worker.py tools/baselines/test_worker_contracts.py`: passed.
- Real smoke: one 78-frame natural rollout, 16 prefix steps, GPU1, `--rynn-num-frames 0`. It completed all `16/16` value steps, generated the final analysis, saved `output_with_trend.mp4` and `raw_model_outputs.json`, and finished with runner status `complete` and `1/1` job complete.
- Real smoke output: `outputs/baselines/rynnvalue_allframes_smoke_20260828_retry/rynnvalue_20260828_142933_320206/`.

The existing 125-rollout attempt that used zero frames failed before this fix with the empty-image error; the full 136-rollout evaluation was not rerun as part of this bounded fix validation.
