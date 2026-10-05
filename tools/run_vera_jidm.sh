#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/../project_env.sh"
export VERA_DATA_PREFIX="${PROJECT_ROOT}"
export VERA_CKPT_ROOT="${CHECKPOINTS}/vera-jidm"
export WANDB_MODE="disabled"
export WANDB_DIR="${OUTPUTS}/vera-jidm/wandb"
export WANDB_CACHE_DIR="${CACHE}/wandb"
export WANDB_CONFIG_DIR="${CACHE}/wandb/config"
export WANDB_DATA_DIR="${CACHE}/wandb/data"
export MPLCONFIGDIR="${CACHE}/matplotlib"
export NUMBA_CACHE_DIR="${CACHE}/numba"
export HF_HUB_DISABLE_IMPLICIT_TOKEN="1"
# Upstream defaults to all visible GPUs. Keep a single selected device by default.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${VERA_CKPT_ROOT}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" \
    "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"
cd "${REPOS}/VERA"
exec "${LF3R_VERA_JIDM_PYTHON}" -m vera.main \
    --config-name=config_pusht_vggt_fusion_jacobian \
    wandb.mode=disabled wandb.entity=lf3r \
    'hydra.run.dir=${oc.env:PROJECT_ROOT}/outputs/vera-jidm/${now:%Y%m%d_%H%M%S}' \
    'hydra.sweep.dir=${oc.env:PROJECT_ROOT}/outputs/vera-jidm/multirun/${now:%Y%m%d_%H%M%S}' \
    "$@"
