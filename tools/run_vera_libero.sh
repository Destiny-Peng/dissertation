#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/../project_env.sh"
cd "${PROJECT_ROOT}"
STAGE="${1:-all}"
VERA_EVAL_OUTPUT="${2:-${OUTPUTS}/vera_libero_jidm/$(lf3r_file_timestamp)}"
VERA_EVAL_GPU="${VERA_LIBERO_GPU:-0}"
export CUDA_VISIBLE_DEVICES="${VERA_EVAL_GPU}"
export MUJOCO_EGL_DEVICE_ID="${VERA_EVAL_GPU}"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export LIBERO_CONFIG_PATH="${CACHE}/libero-vera-jidm"
export MPLCONFIGDIR="${CACHE}/matplotlib"
export NUMBA_CACHE_DIR="${CACHE}/numba"
export VERA_CKPT_ROOT="${CHECKPOINTS}/vera-jidm"
export WANDB_MODE="disabled"
export WANDB_DIR="${OUTPUTS}/vera-jidm/wandb"
export WANDB_CONFIG_DIR="${CACHE}/wandb/config"
export WANDB_CACHE_DIR="${CACHE}/wandb"
export WANDB_DATA_DIR="${CACHE}/wandb/data"
export HF_HUB_OFFLINE="1"
export HF_HUB_DISABLE_IMPLICIT_TOKEN="1"
export OMP_NUM_THREADS="4"
export OPENBLAS_NUM_THREADS="4"
mkdir -p "${LIBERO_CONFIG_PATH}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${VERA_EVAL_OUTPUT}"
"${LF3R_VERA_JIDM_PYTHON}" - "${VERA_EVAL_OUTPUT}" <<'PY'
import os, sys
from pathlib import Path
import yaml
root=Path(os.environ['PROJECT_ROOT'])
assert Path(sys.argv[1]).resolve().is_relative_to(root)
libero=root/'repos/LIBERO/libero/libero'
config={'benchmark_root':str(libero),'bddl_files':str(libero/'bddl_files'),
        'init_states':str(libero/'init_files'),'assets':str(libero/'assets'),
        'datasets':str(root/'datasets/libero_official')}
(Path(os.environ['LIBERO_CONFIG_PATH'])/'config.yaml').write_text(yaml.safe_dump(config))
PY
RUN_LOG="$(lf3r_log_path "vera_libero_${STAGE}")"
exec > >(tee -a "${RUN_LOG}") 2>&1
printf 'Timestamp: %s\nStage: %s\nOutput: %s\nGPU: %s\n' \
    "$(lf3r_timestamp)" "${STAGE}" "${VERA_EVAL_OUTPUT}" "${VERA_EVAL_GPU}"
nvidia-smi --query-gpu=index,memory.free --format=csv
run_stage() {
    local SELECTED_STAGE="$1"
    local SELECTED_PYTHON="${LF3R_VERA_JIDM_PYTHON}"
    case "${SELECTED_STAGE}" in
        prepare)
            "${SELECTED_PYTHON}" "${SCRIPT_DIR}/vera_libero/prepare.py" --output "${VERA_EVAL_OUTPUT}" --demos-per-task 5
            "${SELECTED_PYTHON}" "${SCRIPT_DIR}/vera_libero/diagnose_interface.py" --output "${VERA_EVAL_OUTPUT}"
            ;;
        alignment)
            "${LF3R_ENV_OPENVLA}/bin/python" "${SCRIPT_DIR}/vera_libero/playback.py" --output "${VERA_EVAL_OUTPUT}" --alignment-only
            ;;
        predict)
            "${SELECTED_PYTHON}" "${SCRIPT_DIR}/vera_libero/predict.py" --output "${VERA_EVAL_OUTPUT}" --device cuda:0 --batch 8
            ;;
        playback)
            "${LF3R_ENV_OPENVLA}/bin/python" "${SCRIPT_DIR}/vera_libero/playback.py" --output "${VERA_EVAL_OUTPUT}"
            ;;
        report)
            "${SELECTED_PYTHON}" "${SCRIPT_DIR}/vera_libero/report.py" --output "${VERA_EVAL_OUTPUT}"
            ;;
        verify)
            "${SELECTED_PYTHON}" "${SCRIPT_DIR}/vera_libero/verify_results.py" --output "${VERA_EVAL_OUTPUT}"
            ;;
        *)
            printf 'Unknown stage: %s\n' "${SELECTED_STAGE}" >&2
            return 2
            ;;
    esac
}
if [[ "${STAGE}" == "all" ]]; then
    for VERA_STAGE in prepare alignment predict playback report verify; do
        run_stage "${VERA_STAGE}"
    done
else
    run_stage "${STAGE}"
fi
