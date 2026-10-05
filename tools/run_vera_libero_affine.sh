#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/../project_env.sh"
cd "${PROJECT_ROOT}"
AFFINE_STAGE="${1:-all}"
AFFINE_SOURCE="${2:?Source evaluation directory required}"
AFFINE_OUTPUT="${3:-${OUTPUTS}/vera_libero_affine/$(lf3r_file_timestamp)}"
export CUDA_VISIBLE_DEVICES="${VERA_LIBERO_GPU:-1}"
export MUJOCO_EGL_DEVICE_ID="${VERA_LIBERO_GPU:-1}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="${CACHE}/libero-vera-jidm"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
mkdir -p "${AFFINE_OUTPUT}"
AFFINE_LOG="$(lf3r_log_path "vera_libero_affine_${AFFINE_STAGE}")"
exec > >(tee -a "${AFFINE_LOG}") 2>&1
printf 'Timestamp: %s\nStage: %s\nSource: %s\nOutput: %s\n' "$(lf3r_timestamp)" "${AFFINE_STAGE}" "${AFFINE_SOURCE}" "${AFFINE_OUTPUT}"
if [[ "${AFFINE_STAGE}" == all || "${AFFINE_STAGE}" == diagnose ]]; then
    "${LF3R_VERA_JIDM_PYTHON}" "${SCRIPT_DIR}/vera_libero/affine_diagnostic.py" --source "${AFFINE_SOURCE}" --output "${AFFINE_OUTPUT}"
fi
if [[ "${AFFINE_STAGE}" == all || "${AFFINE_STAGE}" == playback ]]; then
    "${LF3R_ENV_OPENVLA}/bin/python" "${SCRIPT_DIR}/vera_libero/affine_playback.py" --source "${AFFINE_SOURCE}" --output "${AFFINE_OUTPUT}"
fi
