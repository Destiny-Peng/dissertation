# LF3R SAFE training layer

This directory contains LF3R-owned setup and wrappers for the official `vla-safe/SAFE` repository. The upstream checkout remains at `repos/SAFE`; its training source is not patched.

## Environment

```bash
source project_env.sh
bash tools/setup_safe_training_env.sh
uv pip check --python "$LF3R_SAFE_PYTHON"
```

The validated project-local environment is `conda_envs/LF3R-safe` with Python 3.10.21, PyTorch 2.11.0+cu128, CUDA runtime 12.8, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, Hydra 1.3.5, OmegaConf 2.3.1, scikit-learn 1.7.2, matplotlib 3.10.9, OpenCV 5.0.0.93, W&B 0.28.2, and the remaining pinned packages in `tools/requirements-safe-training.txt`. `uv pip check` is the equivalent dependency check because this uv environment does not seed a `pip` module.

## Official training entrypoints

The upstream entrypoint is `python -m failure_prob.train`. The official model names are:

- SAFE-MLP: `model=indep` (`IndepModelConfig`)
- SAFE-LSTM: `model=lstm` (`LstmModelConfig`)

Direct commands:

```bash
cd repos/SAFE
"$LF3R_SAFE_PYTHON" -m failure_prob.train \
  dataset=openvla model=indep \
  dataset.data_path_prefix= \
  dataset.data_path=/absolute/path/to/safe_dataset/ \
  dataset.load_to_cuda=true dataset.token_idx_rel=1.0

"$LF3R_SAFE_PYTHON" -m failure_prob.train \
  dataset=openvla model=lstm \
  dataset.data_path_prefix= \
  dataset.data_path=/absolute/path/to/safe_dataset/ \
  dataset.load_to_cuda=true dataset.token_idx_rel=1.0
```

Use the LF3R wrapper when the dataset and logs are inside this project. It always sets `dataset.load_to_cuda=true` and requires an explicit GPU; there is no CPU fallback:

```bash
"$LF3R_SAFE_PYTHON" tools/safe_training/run_safe_training.py \
  --dataset-dir outputs/safe_training/datasets/primary_natural \
  --model mlp --gpu 1 --epochs 1000 --batch-size 512 \
  --logs-root outputs/safe_training/logs/primary_mlp

"$LF3R_SAFE_PYTHON" tools/safe_training/run_safe_training.py \
  --dataset-dir outputs/safe_training/datasets/primary_natural \
  --model lstm --gpu 1 --epochs 1000 --batch-size 512 \
  --logs-root outputs/safe_training/logs/primary_lstm
```

The official config fragments reference `base_*` entries that are registered by `failure_prob.conf` through Hydra `ConfigStore`; they are not missing upstream files and no synthetic YAML was added.

## Dataset schema

`failure_prob.data.openvla.load_rollouts` scans a flat directory using `dataset.data_path + "*.csv"`. Each rollout uses the official basename:

```text
task<TASK_ID>--ep<EPISODE_ID>--succ<0|1>.csv
task<TASK_ID>--ep<EPISODE_ID>--succ<0|1>.pkl
task<TASK_ID>--ep<EPISODE_ID>--succ<0|1>.mp4   # needed only for video export
```

The CSV must contain these action columns:

```text
action/dx, action/dy, action/dz, action/droll,
action/dpitch, action/dyaw, action/dgripper
```

`action/timestep` is also retained by LF3R sidecar alignment. The same-stem pickle is a dictionary containing `hidden_states`, `task_suite_name`, `task_id`, `task_description`, `episode_idx`, and binary `episode_success`. The official feature tensor is `(T, 7, 4096)`; the seven rows are generated action-token states.

With the official default `token_idx_rel=1.0`, SAFE calls:

```text
round((7 - 1) * 1.0) = 6
```

and selects the final token row, producing `(T, 4096)`. The LF3R `prepare_dataset.py` wrapper verifies this source shape, links official CSV/pickle/video files, or converts an existing `*.safe_features.npz` sidecar into an official-compatible pickle without changing the source sidecar or rollout.

Example:

```bash
"$LF3R_SAFE_PYTHON" tools/safe_training/prepare_dataset.py \
  --manifest datasets/lf3r_failure_rollouts/v1/manifest.jsonl \
  --dataset-role primary_natural \
  --run-name lf3r-data-natural-libero10-20260824_210557 \
  --output outputs/safe_training/datasets/primary_natural
```

Repeated task/episode basenames are rejected unless `--run-name` or explicit `--rollout-id` selection makes the source unique. This prevents silently mixing generated runs.

The official split first shuffles task IDs, assigns `round(unseen_task_ratio * task_count)` tasks to `val_unseen`, then splits each seen task by `seen_train_ratio` into `train` and `val_seen`. The official evaluation uses `val_seen` for conformal calibration and `val_unseen` for held-out evaluation. Labels remain `episode_success=1` for success and `0` for failure; model evaluation internally uses `1 - episode_success` for failure-positive metrics.

## GPU-only mock smoke

`run_gpu_smoke.sh` creates a synthetic official-format dataset with 32 `(80,7,4096)` rollouts, runs one complete official SAFE-MLP epoch and one SAFE-LSTM epoch, saves each official `state_dict`, reloads each checkpoint, and writes finite validation scores. It is GPU-only and gates only on free memory (16 GiB by default); utilization and existing compute processes are informational, and it never falls back to CPU:

```bash
bash tools/safe_training/run_gpu_smoke.sh 1
```

`validate_safe_checkpoint.py` is also available for explicit checkpoint reload and score generation. The first 5-step mock attempt reached training but exposed an upstream functional-conformal small-calibration edge case; the current 80-step/32-rollout fixture supplies enough calibration samples for the official path.

## Compatibility status

- Official `.pkl` artifacts are directly consumable by SAFE.
- LF3R `.safe_features.npz` contains the same un-reduced `(T,7,4096)` tensor plus alignment arrays; `.safe_features.json` records provenance and the selection rule. When present, the wrapper validates the LF3R schema and declared shape, preserves the JSON beside the staged files, and converts only when an official `.pkl` is absent. The official loader ignores the JSON and consumes the generated same-stem `.pkl`.
- A two-rollout sidecar fixture was staged and loaded successfully: source `(80,7,4096)` -> official loader `(80,4096)` with `token_idx_rel=1.0`, finite values, and JSON metadata validation.
- No detector loss, model head, split rule, or upstream training code is changed.
- The 44-test annotator suite, SAFE CLI help, wrapper compilation, and `uv pip check` pass. Real LIBERO-10 SAFE training has not been started.
- The MLP/LSTM GPU smoke remains pending. The latest memory-only attempt observed 3,836 MiB free, below the default 16 GiB threshold, and exited before model loading. Utilization and existing compute processes are not gates; the script never falls back to CPU.
