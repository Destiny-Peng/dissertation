# T-Rex environment reuse and verification

Timestamp: 2026-10-03T21:06:07.620768+08:00
Repository: `repos/T-Rex`, commit `f88e10c61da123c68bf0927cf4860bc97a0381f3`
Environment: `repos/ProcVLM/.venv` (shared with ProcVLM)

Only `h5py==3.16.0` was added. The before/after freezes confirm all existing versions were preserved. Core runtime: Python 3.10.21, torch 2.10.0+cu128, torchvision 0.25.0, transformers 4.57.3, tokenizers 0.22.1, accelerate 1.7.0, deepspeed 0.18.9, datasets 4.4.2, numpy 2.2.6. The reused runtime intentionally differs from upstream exact pins; do not run the upstream full dependency install in this shared environment.

## Commands and results

Run from the LF3R root:

```bash
source ./project_env.sh
/home/linxia/.local/bin/uv pip freeze --python repos/ProcVLM/.venv/bin/python
/home/linxia/.local/bin/uv pip install --python repos/ProcVLM/.venv/bin/python h5py==3.16.0
/home/linxia/.local/bin/uv pip check --python repos/ProcVLM/.venv/bin/python
bash -n tools/run_trex.sh
HF_HUB_OFFLINE=1 bash tools/run_trex.sh infer --help
HF_HUB_OFFLINE=1 bash tools/run_trex.sh train --help
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 bash tools/run_trex.sh python environment_reports/trex_20261003_210333/gpu_smoke.py
```

- Installation PASS: one added package, no upgrades/downgrades.
- Launch script syntax PASS; official inference and training CLI help PASS.
- HDF5 tactile-array write/read equality, quickstart schema/hub/decode, F6WindowDataset and existing decord imports PASS.
- Training dependency imports (torch, transformers, accelerate, datasets, deepspeed, wandb, timm, h5py) PASS.
- One GPU smoke PASS on RTX PRO 6000 Blackwell: random small official Qwen3VLVLAModel, three-expert MoT forward with KV cache, action velocity head, embedded per-finger tactile tokenizer and VQ-VAE reconstruction. Peak tensor allocation 17.05 MiB. See `gpu_smoke.py` and `gpu_smoke.log`.
- Dependency check: pre-existing decord wheel platform-tag warning remains unchanged; decord imports successfully. No newly introduced reported incompatibilities.

## Usage

```bash
bash tools/run_trex.sh infer --help
bash tools/run_trex.sh train --help
bash tools/run_trex.sh python -c 'import h5py; from qwen_vla import Qwen3VLVLAModel'
```

The launcher derives paths from project_env.sh, defaults to the reused ProcVLM Python, and accepts LF3R_TREX_PYTHON as an override. It sets the T-Rex and dataset_quickstart source paths locally and uses offline W&B logging by default. Pass the official CLI arguments directly. Original upstream launch scripts contain author-machine paths and are left unchanged; use tools/run_trex.sh locally.

## Validation limits

No pretrained checkpoints or dataset episodes were downloaded. This validates environment startup, data I/O and model kernels with synthetic inputs, not pretrained policy quality, full-checkpoint memory usage, robot control, or distributed/DeepSpeed training. LeRobot and 3D replay extras were not installed because they are optional and were not used. Original ProcVLM dependencies were preserved and decord import checked, but a full ProcVLM inference regression was not run.
