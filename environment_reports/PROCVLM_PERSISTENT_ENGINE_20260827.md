# ProcVLM persistent-engine validation — 2026-08-27

## Result

The LF3R ProcVLM runner now starts one persistent worker process for a selected run. The worker initializes the upstream vLLM engine and processor once, processes rollouts sequentially, atomically upserts one status row per rollout, and records resumable progress. A bounded representative run completed 8/8 rollouts without starting the full 136-rollout evaluation.

Run artifact:

`outputs/baselines/persistent_procvlm_representative_20260827/procvlm_20260827_180045_610051/`

## Representative execution

- Baseline: ProcVLM-2B
- Selected rollouts: 8 natural-observation records from `outputs/baselines/representative_rollouts_20260826.json`
- GPU: physical GPU 0
- Memory target: `0.80` of currently free memory
- Measurement: 73,145 MiB free / 97,887 MiB total
- Resolved vLLM total-memory fraction: `0.597791`
- Bounded settings: `window_size=4`, `max_sampled_frames=16`, `max_new_tokens=128`
- Engine initialization: `71.3648` seconds
- Worker return code: `0`
- Final status: `complete`
- Completed / failed / pending: `8 / 0 / 0`

Per-rollout inference times, in seconds:

```text
libero_10-task00-ep000-natural-5ef44c034f   8.646
libero_10-task00-ep001-natural-477e2f3af1   6.226
libero_10-task02-ep000-natural-d0c9300789   5.883
libero_10-task03-ep007-natural-2dd9870cd4   6.180
libero_10-task05-ep003-natural-fefb27649d   5.984
libero_10-task06-ep002-natural-888f529fd3   6.147
libero_10-task08-ep002-natural-2bc533b685   6.157
libero_spatial-task00-ep000-natural-be4a4090b6 5.754
```

## Contract audit

- `run.json`: `procvlm_persistent_worker=true`, engine initialization time, per-rollout inference times, requested free-memory fraction, resolved total fraction, GPU snapshot, and final counts are present.
- `procvlm_progress.jsonl`: exactly one `engine_initialized` event and eight `rollout_finished` events; no fatal event.
- `procvlm_state.json`: `status=complete` after inference; the later no-op resume records `status=no_pending_jobs` with `total_jobs=8`, `completed_jobs=8`, `failed_jobs=0`, and `pending_jobs=0`.
- `jobs.jsonl`: eight unique rollout rows, all `status=complete`, each with one raw output file and `inference_seconds`.
- Raw output: each rollout has 16 JSONL records containing `frame_index` and a four-element `window_frame_indices`; all expected ProcVLM fields (`progress`, `parsed_progress`, `reasoning`, and unmodified `model_output`) are present.
- Downstream compatibility: the existing `temporal-analysis` ProcVLM loader consumed all eight outputs and returned 128 aligned frame/value samples.

The existing `--resume-run` CLI was also invoked against this completed run. It detected no pending jobs, did not initialize a second engine, kept the final status `complete`, and recorded `resume_count=1`.

## Recovery validation

`tools/baselines/test_procvlm_persistent.py` uses a model-free fake engine and covers:

1. One initialization reused across three sequential jobs.
2. An ordinary parser/inference error isolated to one job while later jobs continue.
3. A simulated `EngineCore encountered an issue` fatal event saved as `interrupted` with exit code `70`, followed by resume of only the interrupted/unstarted jobs.
4. A simulated CUDA OOM during initialization saved pending state and exited with code `70`.

Other validations passed:

- `tools/baselines/validate_pipeline.py`
- `tools/baselines/test_worker_contracts.py`
- 12 `tools/lf3r_annotator/tests` unit tests
- Python compilation for the runner, worker, and validation scripts

The vLLM shutdown warning emitted after normal worker completion is from the upstream async event-loop shutdown path; it occurred after all eight outputs were written and the worker returned success. GPU utilization was observed only for monitoring; memory availability was the execution gate.
