# Annotator RynnValue Cross-Rollout Parallelism

Date: 2026-08-29

## Status

The LF3R baseline runner and annotator Review batch form now support RynnValue rollout-level parallelism. The implementation preserves the existing temporal `batch_size`; it creates independent model workers for separate rollout ranges. No real GPU inference was run for this change.

## Selection semantics

The selected manifest is filtered by partition and dataset role first. The total positional range is scope-relative and right-open: `[start_index, end_index)`. Worker rows are then applied inside that range. When `--worker-spec` is omitted, `--parallel-workers` divides the range continuously with any remainder assigned to earlier workers.

Explicit overlap is allowed. The first worker in argument/table order owns a rollout; later assignments produce a `duplicate_assignment` job record and never overwrite `raw/<rollout-id>`. Explicit gaps are allowed but are recorded in `run.json`; the aggregate status is `complete_with_errors`, and Analysis full-coverage validation will reject the run for a selection that includes the gap.

## GPU and execution behavior

Each RynnValue worker launches its rollout subprocesses sequentially and sets `CUDA_VISIBLE_DEVICES` to its one configured numeric GPU. `--gpu 0,1` supplies IDs for automatic worker assignment; it does not request RynnValue tensor parallelism. Explicit worker rows may reuse a GPU, and the worker count may exceed the number of listed GPUs. The scheduler intentionally performs no GPU utilization, free-memory, or GPU-ID conflict admission check. Users are responsible for model-copy memory and compute contention. Temporal `--rynn-batch-size` remains independently user-controlled.

One aggregate run contains `run.json`, `jobs.jsonl`, `commands.jsonl`, shared `raw/<rollout-id>` directories, worker assignments, worker progress, overlap/gap metadata, and per-rollout return codes. Nonzero rollout workers do not stop other worker ranges; the aggregate records `complete_with_errors` when failures or uncovered ranges remain.

## Web behavior

Review adds Total start/end fields and a RynnValue worker table. Adding or removing a row rebalances all ranges. Manual edits are retained until the next add/remove/rebalance action. The live summary shows requested range size, per-worker counts, overlaps, gaps, repeated GPU use, and unique execution count. Non-RynnValue batch forms are unchanged.

The batch API forwards worker rows as repeated `--worker-spec GPU:START:END` arguments to `run_lf3r_baseline.py`. Baseline jobs remain independently submit-able; the UI does not lock a new baseline submission because another method is running. Existing failed `web_runs` are not resumed or modified.

## Validation

The following checks are intended to remain CPU/fake-runner only:

```bash
python3 tools/baselines/test_rynn_parallel.py -v
python3 -m unittest discover -s tools/lf3r_annotator/tests -p 'test_*.py' -v
python3 -m py_compile tools/baselines/run_lf3r_baseline.py tools/lf3r_annotator/server.py
```

The test coverage includes right-open selection, automatic splitting, overlap/gap detection, duplicate ownership, same-GPU worker reuse, per-worker CUDA environment assignment, failure isolation, web command forwarding, and aggregate run progress. No rollout/media/annotation files are modified by the feature tests.

The final validation on 2026-08-29 completed successfully: 29 annotator service/contract/supervisor tests, 4 RynnValue fake parallel-runner tests, Python compilation for the changed modules, browser JavaScript syntax parsing, and imports plus `--help` for the dedicated Analysis environment. No real GPU inference was run.
