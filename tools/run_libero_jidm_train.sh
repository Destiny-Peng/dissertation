#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/../project_env.sh"
TRAIN_DATA="${1:-$(cat "${CACHE}/libero_jidm_training_current_run.txt")}"
shift || true
"${LF3R_VERA_JIDM_PYTHON}" - "${TRAIN_DATA}" <<'PY'
import sys,json
from pathlib import Path
p=Path(sys.argv[1]);v=json.loads((p/'verification.json').read_text());assert v['passed']
PY
export CUDA_VISIBLE_DEVICES="${LIBERO_JIDM_GPU:-1}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VERA_CKPT_ROOT="${CHECKPOINTS}/vera-jidm"
export WANDB_MODE=disabled
export WANDB_DIR="${OUTPUTS}/vera-libero-training/wandb"
export WANDB_CONFIG_DIR="${CACHE}/wandb/config"
export WANDB_CACHE_DIR="${CACHE}/wandb"
export WANDB_DATA_DIR="${CACHE}/wandb/data"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export MPLCONFIGDIR="${CACHE}/matplotlib"
mkdir -p "${WANDB_DIR}"
TRAIN_LOG="$(lf3r_log_path libero_jidm_train)"
exec > >(tee -a "${TRAIN_LOG}") 2>&1
cd "${REPOS}/VERA"
exec "${LF3R_VERA_JIDM_PYTHON}" -m vera.main --config-path "${TRAIN_DATA}/configs" --config-name libero_idm "$@"
