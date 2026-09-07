# DenseReward Web Baseline Integration — 2026-09-06

## Scope

DenseReward is now registered as the fifth LF3R baseline method for the command-line wrapper and the annotator Review page. Existing SAFE, ProcVLM, RynnValue, and Robo-Dopamine behavior is unchanged. The existing temporal-analysis runner remains a four-method analyzer; DenseReward is exposed for Review, run discovery, raw-output inspection, and future analysis adapters, but is not silently inserted into old temporal snapshots.

## Runtime

- Python: `conda_envs/LF3R-densereward/bin/python`
- Checkpoint: `checkpoints/densereward-3frame-thinking`
- Worker: `tools/baselines/densereward_worker.py`
- Runner method: `--baseline densereward`
- Web method value: `densereward`
- Model loading: one processor/model instance per persistent rollout worker
- Decode: PyAV sequential decode, RGB PIL images kept in a three-frame window
- Official call: exactly three chronological images plus the task instruction, greedy decoding, BF16

## Sampling and output

`--densereward-frame-interval K` evaluates current frames `2, 2+K, 2+2K, ...`; the default is `K=1`. Each row in `raw/<rollout-id>/densereward_raw.jsonl` preserves the current `frame_index`, the three `sampled_frame_indices`, raw model text, parsed reason, scalar `reward`, input/generated token counts, and clipping status. `worker_result.json` records the checkpoint, system-prompt hash, sampling rule, frame interval, output count, and timing.

The annotator maps the scalar to a `reward` signal and leaves it separate from the existing `progress` and `hop` signals. Signal history and current-frame readout use the existing Review chart contract.

## Web integration

- Single-rollout Review card: DenseReward appears with Run baseline/Re-run rollout.
- Batch panel: DenseReward can be selected with the common scope and worker table.
- Advanced fields: `--densereward-frame-interval` and `--densereward-max-new-tokens`.
- CLI tooltip metadata: `tools/lf3r_annotator/static/parameter_help.json`.
- Run discovery and normalized parser: `GET /api/baselines/runs` and `GET /api/baselines/<rollout-id>`.
- DenseReward uses no vLLM memory-fraction conversion. The common free-memory form field is accepted for API compatibility, while the worker uses Transformers `device_map=auto` and the explicitly assigned `CUDA_VISIBLE_DEVICES`.

## Validation

- `python3 -m py_compile tools/baselines/densereward_worker.py tools/baselines/run_lf3r_baseline.py tools/lf3r_annotator/server.py`
- Runner `--help` exposes `densereward`, `--densereward-frame-interval`, and `--densereward-max-new-tokens`.
- One natural rollout dry-run with `--validate-environment --densereward-frame-interval 2` completed with a project-local worker command and the dedicated environment import check.
- Server/frontend contract tests cover the DenseReward raw parser, run discovery, batch command forwarding, method selector, and help metadata.
- No full dataset DenseReward inference was started during this integration change.

## Analysis boundary

DenseReward is intentionally not required by `POST /api/analysis/run` and is not added to existing four-method temporal analysis snapshots. This prevents an output with a different reward semantic and native three-frame sampling contract from being mislabeled as the existing progress/value/hop comparison. A future DenseReward analysis adapter should define its event metrics explicitly before adding it to that comparison.
