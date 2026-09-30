# Ctrl-World Conda Environment

Updated: 2026-09-29 16:55 +0800  
Repository: `repos/Ctrl-World` at commit `99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d`

## Locations

- Conda environment: `conda_envs/Ctrl-World`
- Python: `conda_envs/Ctrl-World/bin/python` (Python 3.11.16)
- Project-local Conda manager: `tools/miniforge3/bin/conda` (Conda 26.3.2)
- UV package cache: `cache/uv`
- Conda package cache: `cache/conda/pkgs`
- System Conda was also found at `/mnt/hdd/qiuxia/miniconda3/bin/conda` (Conda 26.5.3). The project-local Miniforge is used so project setup does not depend on that external path.

`project_env.sh` now exports `LF3R_CONDA_EXE`, `LF3R_ENV_CTRL_WORLD`, and `LF3R_CTRL_WORLD_PYTHON`, and sources the project-local Conda shell hook when available.

## Installed versions

- Python 3.11.16
- PyTorch 2.7.1+cu128, built for CUDA 12.8
- NumPy 2.4.6
- The remaining versions are listed in `CTRL_WORLD_PIP_FREEZE_20260929.txt`.

The Conda prefix was created with Python and pip. The UV cache existed, but it did not contain the exact dependency set for Torch 2.7.1+cu128: for example, the required `nvidia-cudnn-cu12==9.7.1.26` was absent while newer CUDNN builds were cached. UV does not use another environment's installed `site-packages` as a wheel cache. To avoid downloading the already installed package set again, 35,806 package files were hardlinked into this prefix. The source and destination use the same filesystem and Python 3.11.16 ABI. Core imports passed from the new Conda Python. The old venv was removed after validation; the new prefix retains its own hardlinks.

## Validation

- `conda run --prefix "$LF3R_ENV_CTRL_WORLD" python -c 'import torch, numpy, diffusers, transformers, accelerate, decord, mediapy, wandb, swanlab, scipy, pandas, einops'` passed.
- `accelerate --help` and `torchrun --help` both started successfully from the Conda prefix.
- A CPU tensor operation passed; no GPU kernel or inference was run.
- `torch.__version__` is `2.7.1+cu128`; `torch.version.cuda` is `12.8`.
- `pip check` reports `decord 0.6.0 is not supported on this platform`. Its wheel metadata advertises `cp36-cp36m-manylinux2010_x86_64`; `import decord` succeeded under Python 3.11. This is an existing wheel-tag issue, not a missing dependency.
- No GPU inference was run because all GPUs were occupied by other jobs. No dataset was downloaded.

## Use

From the LF3R project root:

```bash
source ./project_env.sh
conda activate "$LF3R_ENV_CTRL_WORLD"
```

For non-interactive commands, use `conda run --prefix "$LF3R_ENV_CTRL_WORLD" <command>`.

The Conda package specification is recorded in `CTRL_WORLD_CONDA_EXPLICIT_20260929.txt`; the Python package freeze is recorded in `CTRL_WORLD_PIP_FREEZE_20260929.txt`.
