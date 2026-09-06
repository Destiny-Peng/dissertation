# Robo-Dopamine Persistent Engine Report — 2026-08-27

## Result

The Robo-Dopamine baseline now uses one persistent worker process for every selected run. The worker constructs one upstream `GRMInference` object, which keeps the vLLM engine and processor resident, and calls `run_pipeline` sequentially for each rollout. The old single-rollout `robo_dopamine_worker.py` entry point was removed as requested; the only runner-owned Robo-Dopamine entry point is `tools/baselines/robo_dopamine_persistent_worker.py`.

The bounded representative run completed successfully:

- Run root: `outputs/baselines/persistent_robo_dopamine_representative_20260827/robo_dopamine_20260827_193829_137208`
- Selection: 8 natural representative rollouts (2 clean success, 2 recovered success, 4 failure)
- Status: `complete`
- Jobs: 8 complete, 0 failed, 0 interrupted, 0 pending
- Engine initialization events: 1
- Engine initialization time: 53.3471 seconds
- vLLM process: 1 persistent worker process for all 8 rollouts
- GPU memory at startup: GPU0 had 73,147 MiB free of 97,887 MiB total
- Requested memory target: `0.80` of currently free memory
- Resolved vLLM total-memory fraction: `0.597807`
- GPU utilization: observed at 100%, informational only and not used as a gate
- Raw output contract: all 8 rollouts have an official `pred_vllm.json` and `worker_result.json`

The detailed atomic job, progress, and state records are in the run root: `jobs.jsonl`, `robo_dopamine_jobs.jsonl`, `robo_dopamine_progress.jsonl`, and `robo_dopamine_state.json`.

## Implementation

- `tools/baselines/robo_dopamine_persistent_worker.py` imports the official `examples.inference` module once, patches its module-local `LLM` symbol once to apply the runner-resolved memory budget, constructs `GRMInference` once, and reuses it across pending jobs.
- Each job passes its own `raw/<rollout-id>/` directory as the official `out_root`; this preserves native timestamped output isolation while avoiding engine recreation.
- Ordinary per-rollout errors are recorded as `failed` and do not stop later jobs.
- CUDA OOM, dead-engine markers, and other classified engine-fatal errors are recorded as `interrupted`; the worker saves state and exits with code `70` for resume.
- `robo_dopamine_runner.py` creates the persistent job plan, resolves the free-memory budget once per run, records one worker command, and supports `--resume-run`.
- `run_lf3r_baseline.py` routes Robo-Dopamine to the persistent runner by default. No additional persistence parameter is required.

## Verification

- `python3 tools/baselines/test_robo_dopamine_persistent.py`: `ROBODOPAMINE_PERSISTENT_WORKER_TESTS_OK`; verifies one model initialization across three jobs, recoverable-error continuation, fatal state, and resume selection.
- `python3 tools/baselines/test_worker_contracts.py`: `BASELINE_WORKER_CONTRACTS_OK`; verifies RynnValue and the persistent Robo-Dopamine official raw-output contract with fake workers.
- `python3 tools/baselines/validate_pipeline.py`: `BASELINE_PIPELINE_VALIDATION_OK`; dry-run command is `robo_dopamine_persistent_worker.py`, with deferred free-memory conversion and execution scope `persistent_robo_dopamine_worker`.
- Existing Robo-Dopamine temporal loader read all 8 representative outputs. Native signal counts were 8/8 for both `progress` and `hop`, with no frame-bound records missing.
- A post-completion `--resume-run` attempt did not start a worker because the shared manifest had changed after the run was created. The runner correctly rejected the changed manifest before model loading; the worker-level resume behavior remains covered by the model-free tests.

The full 136-rollout evaluation was not rerun.
