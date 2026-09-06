# Annotator All-Baseline Rollout Workers

Date: 2026-08-29

## Status

The command-line baseline runner and Review batch form now expose the same rollout-level worker allocator for SAFE, ProcVLM, RynnValue, and Robo-Dopamine. The existing RynnValue temporal batch_size and the legacy no-worker paths remain unchanged. No real GPU inference was run for this extension.

## Execution model

Selection remains scope-relative and right-open: [start_index, end_index). Manifest partition, dataset role, and rollout-ID filters are applied before the total range and worker shards. Repeated --worker-spec GPU:START:END rows use one numeric GPU per worker; automatic --parallel-workers N splitting is continuous and gives any remainder to earlier workers.

- SAFE starts one isolated per-rollout command in each worker shard.
- ProcVLM and Robo-Dopamine start one independent persistent worker process and model engine per shard, then merge their plans and results into one aggregate run.
- RynnValue keeps its existing per-rollout temporal inference and only parallelizes separate rollouts across workers.
- Worker processes receive their configured CUDA_VISIBLE_DEVICES value. GPU IDs may repeat, worker count may exceed the number of listed IDs, and no utilization, memory, or GPU-conflict scheduler is added. Users are responsible for model-copy memory and compute contention.

One invocation writes one aggregate run.json, jobs.jsonl, and commands.jsonl, with shared raw/<rollout-id> directories. Each worker assignment and per-rollout record includes worker index, GPU, right-open range, command, timestamps, and return code. Overlapping explicit ranges are accepted; first worker order owns the rollout and later rows receive duplicate_assignment. Gaps are recorded and leave the aggregate run partial, so it cannot satisfy a complete Analysis selection.

## Web behavior

The Review Batch baseline panel shows the worker table for all four methods. It reports each worker range and count, total and unique counts, overlap, gap, and repeated-GPU warnings. Adding or removing a worker rebalances the total range; manual edits can then be made before submission. The API forwards rows as repeated worker-spec arguments and multiple baseline methods may be submitted independently.

A legacy batch request without worker rows still uses the existing single persistent ProcVLM/Robo-Dopamine path. Supplying worker rows or more than one worker opts every method into the aggregate worker runner. Existing failed outputs/baselines/web_runs are not resumed, rewritten, or deleted.

## Validation

The checks are CPU/fake-runner only:

~~~bash
python3 -m unittest discover -s tools/lf3r_annotator/tests -p test_*.py -v
python3 tools/baselines/test_rynn_parallel.py -v
python3 tools/baselines/test_all_parallel.py -v
python3 -m py_compile tools/baselines/run_lf3r_baseline.py tools/lf3r_annotator/server.py
~~~

The full validation on 2026-08-29 passed after this extension: the annotator service/contract/supervisor suite, four RynnValue fake parallel-runner tests, and three all-method fake parallel-runner tests. JavaScript syntax and existing baseline worker contracts were also checked separately. No real model inference, rollout generation, manifest rewrite, or user-output deletion was performed.
