# RynnValue Approximate Fixed-Interval Sampling — 2026-08-28

## Result

The LF3R runner now supports a target RynnValue source-frame interval without changing the upstream `repos/RynnValue/rynn_infer/inference.py` implementation.

For each rollout with `T` decoded frames and target interval `K`, the command planner computes:

```
num_steps = min(T, ceil((T - 1) / K) + 1)
```

This is the number of points in `0, K, 2K, ...` plus the final frame. The official CLI then uniformly distributes those `num_steps` endpoints over `0 ... T-1`, so the realized spacing is approximate by design. For the 414-frame rollout that previously used 16 prefix steps, `K=8` now produces 53 uniformly spaced prefix evaluations (about 7.94 frames apart).

## Memory-safe defaults

- `--rynn-num-frames 16`: uniformly resample at most 16 images inside each prefix.
- `--rynn-batch-size 1` by default: run one prefix sub-sample per model forward. Any positive batch size is accepted and forwarded; the user is responsible for choosing a value that fits available GPU memory.
- `--rynn-evaluation-interval 8`: target source-frame spacing.
- `--rynn-num-frames 0` is rejected; the former all-frame-per-prefix compatibility path is disabled.
- The legacy `--rynn-num-steps` option remains accepted for command compatibility, but the runner computes and forwards the per-rollout value from the interval.
- `batch_size` is not a correctness requirement; it controls how many equal-length prefix samples the official `run_batch` combines in one forward.

The raw output contract remains compatible: `values` and `sampled_indices` keep their existing meanings. New metadata records `sampling_mode`, `evaluation_interval`, and the computed `num_steps`.

## Verification

- `python3 tools/baselines/test_worker_contracts.py`: `BASELINE_WORKER_CONTRACTS_OK`.
- `python3 tools/baselines/validate_pipeline.py`: `BASELINE_PIPELINE_VALIDATION_OK`.
- `python3 -m unittest discover -s tools/lf3r_annotator/tests -p 'test_*.py' -v`: 28 tests passed.
- `python3 -m py_compile tools/baselines/rynnvalue_worker.py tools/baselines/run_lf3r_baseline.py tools/baselines/validate_pipeline.py`: passed.
- No real GPU inference was run for this change.
