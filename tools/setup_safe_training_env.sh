#!/usr/bin/env bash
set -euo pipefail

source /mnt/hdd/qiuxia/pyr/LF3R/project_env.sh
test "$PROJECT_ROOT" = "/mnt/hdd/qiuxia/pyr/LF3R"

SAFE_ENV="${LF3R_ENV_SAFE}"
SAFE_PYTHON="${LF3R_SAFE_PYTHON:-${SAFE_ENV}/bin/python}"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

if [[ ! -x "$SAFE_PYTHON" ]]; then
    echo "Creating project-local SAFE environment at $SAFE_ENV"
    uv venv --python 3.10.21 "$SAFE_ENV"
fi

uv pip install --python "$SAFE_PYTHON"     --index-url "$PYTORCH_INDEX"     torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128
uv pip install --python "$SAFE_PYTHON"     -r "$PROJECT_ROOT/tools/requirements-safe-training.txt"
uv pip install --python "$SAFE_PYTHON"     -e "$PROJECT_ROOT/repos/SAFE"
uv pip check --python "$SAFE_PYTHON"

"$SAFE_PYTHON" - <<'PY'
import importlib.metadata as metadata
import sys
import torch

print("SAFE Python:", sys.version.split()[0])
print("SAFE torch:", metadata.version("torch"), "CUDA:", torch.version.cuda)
print("SAFE CUDA available:", torch.cuda.is_available())
PY
