# SAFE training environment and compatibility report

Date: 2026-09-07  
Project: `/mnt/hdd/qiuxia/pyr/LF3R`  
Official checkout: `repos/SAFE`  
Official revision inspected: `b6036ab`

## Environment

The project-local environment is `conda_envs/LF3R-safe` and is exposed as `LF3R_SAFE_PYTHON`. The reproducible setup entrypoint is `tools/setup_safe_training_env.sh`; pinned non-Torch dependencies are in `tools/requirements-safe-training.txt`.

Validated versions:

```text
Python 3.10.21
PyTorch 2.11.0+cu128
CUDA runtime 12.8
numpy 2.2.6
pandas 2.3.3
scipy 1.15.3
hydra-core 1.3.5
omegaconf 2.3.1
scikit-learn 1.7.2
matplotlib 3.10.9
opencv-python 5.0.0.93
wandb 0.28.2
```

`uv pip check --python "$LF3R_SAFE_PYTHON"` passed for 83 packages. The environment is uv-managed and has no `pip` module; no system package was installed.

## Official semantics confirmed

- Entry point: `python -m failure_prob.train`.
- SAFE-MLP: `model=indep`; SAFE-LSTM: `model=lstm`.
- Config fragments register `base_config`, `base_openvla`, `base_indep`, `base_lstm`, and `base_train` through Hydra `ConfigStore` at import time.
- OpenVLA loader scans a flat CSV prefix, loads same-stem pickle metadata/features, and casts hidden states before `process_tensor_idx_rel`.
- Official source hidden states are `(T,7,4096)`. `token_idx_rel=1.0` selects index 6 and yields `(T,4096)`.
- Split semantics are task-level `val_unseen`, per-seen-task `train`/`val_seen`; `val_seen` is the conformal calibration split and `val_unseen` is the held-out evaluation split.
- Official `BaseModel.train_epoch`, `IndepModel.forward_compute_loss`, and `LstmModel.forward_compute_loss` remain the training semantics used by the LF3R wrapper.

## LF3R compatibility layer

`tools/safe_training/prepare_dataset.py` selects manifest rows and materializes a flat official loader directory. Existing `.csv/.pkl/.mp4` files are symlinked when possible. If only `*.safe_features.npz` exists, the wrapper validates `(T,7,4096)` and writes a same-stem official pickle containing the un-reduced tensor and manifest metadata. It records all sources in `selection.json`; source rollout/media/sidecar files are not modified. Duplicate official basenames are rejected.

## Bounded mock validation

Generated fixtures:

- `outputs/safe_training/gpu_smoke_20260907/mock_dataset`: 16 rollouts, `(5,7,4096)`; used for loader/split/token preflight.
- `outputs/safe_training/gpu_smoke_20260907/mock_dataset_80`: 32 rollouts, `(80,7,4096)`; provides enough successful calibration sequences for the official functional-conformal evaluator.

The official loader preflight passed: 16 rollouts loaded, split counts were train 6 / val_seen 6 / val_unseen 4, both labels were present in every split, and `token_idx_rel=1.0` returned `(3,4096)` for a `(3,7,4096)` input.

An initial GPU1 SAFE-MLP run loaded the model and completed one training epoch (`Avg Loss: -0.1025`) but the official functional conformal evaluator failed on the deliberately tiny 5-step/3-success calibration fixture with an empty modulation array. This is an upstream small-sample edge case, not a CUDA or memory failure. The 80-step fixture and GPU-only rerun entrypoint are ready.

The final MLP/LSTM checkpoint, reload, validation-score, and finite-value results remain pending until GPU1 has enough free memory for the bounded smoke. No CPU substitute is used, and no real LIBERO training has started.

## Additional non-GPU verification

- `failure_prob.train --help`, all SAFE wrapper compilation checks, and `uv pip check --python "$LF3R_SAFE_PYTHON"` passed.
- The existing annotator test suite passed: 44 tests, 3 optional tests skipped.
- A project-local sidecar fixture passed end to end: two `.safe_features.npz` files with matching `.safe_features.json` metadata were schema/shape-validated, converted to official same-stem `.pkl` records, and loaded by the official SAFE loader. The source tensor was `(80, 7, 4096)` and the loader output at `token_idx_rel=1.0` was `(80, 4096)`.
- No rollout, media, annotation, or upstream SAFE source was modified by these checks.

## Deviations and blockers

- No upstream SAFE source was modified.
- `failure_prob.train` unconditionally calls `model.to("cuda")`; the LF3R wrapper preserves this and refuses CPU fallback.
- The latest memory-only smoke attempt saw 3,836 MiB free, below the default 16 GiB threshold, and exited 75 before model loading. GPU utilization and existing process presence are informational rather than gates; no external process was killed or reconfigured.
