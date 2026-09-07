#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_GUESS="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT_GUESS}/project_env.sh"
test "${PROJECT_ROOT}" = "${PROJECT_ROOT_GUESS}"

if [[ -e "${LF3R_ENV_DENSEREWARD}" ]]; then
    echo "Refusing to overwrite existing environment: ${LF3R_ENV_DENSEREWARD}" >&2
    exit 1
fi

uv venv --python 3.12.3 "${LF3R_ENV_DENSEREWARD}"
uv pip install \
    --python "${LF3R_DENSEREWARD_PYTHON}" \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install \
    --python "${LF3R_DENSEREWARD_PYTHON}" \
    -r "${PROJECT_ROOT}/tools/requirements-densereward.txt"

"${LF3R_DENSEREWARD_PYTHON}" -c '
import accelerate
import numpy
import PIL
import qwen_vl_utils
import torch
import transformers

print("DenseReward environment OK")
print("python", __import__("sys").version)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("numpy", numpy.__version__)
print("pillow", PIL.__version__)
print("cuda_available", torch.cuda.is_available())
'
