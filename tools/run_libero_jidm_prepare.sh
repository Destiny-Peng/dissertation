#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/../project_env.sh"
cd "${PROJECT_ROOT}"
PREP_STAGE="${1:-pack}"
PREP_DATA="${2:-$(cat "${CACHE}/libero_jidm_training_current_run.txt")}"
export CUDA_VISIBLE_DEVICES="${LIBERO_JIDM_GPU:-1}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export MPLCONFIGDIR="${CACHE}/matplotlib"
PREP_LOG="$(lf3r_log_path "libero_jidm_${PREP_STAGE}")"
exec > >(tee -a "${PREP_LOG}") 2>&1
printf 'Timestamp: %s\nStage: %s\nData: %s\nGPU: %s\n' "$(lf3r_timestamp)" "${PREP_STAGE}" "${PREP_DATA}" "${CUDA_VISIBLE_DEVICES}"
case "${PREP_STAGE}" in
    worker)
        "${LF3R_VERA_JIDM_PYTHON}" "${SCRIPT_DIR}/vera_libero/training_worker.py" --data "${PREP_DATA}"
        ;;
    first)
        "${LF3R_VERA_JIDM_PYTHON}" "${SCRIPT_DIR}/vera_libero/pack_training.py" --data "${PREP_DATA}" --limit 1 --batch 1
        ;;
    pack)
        "${LF3R_VERA_JIDM_PYTHON}" "${SCRIPT_DIR}/vera_libero/pack_training.py" --data "${PREP_DATA}" --batch 1
        ;;
    verify)
        "${LF3R_VERA_JIDM_PYTHON}" "${SCRIPT_DIR}/vera_libero/verify_training.py" --data "${PREP_DATA}"
        ;;
    *)
        printf 'Unknown preparation stage: %s\n' "${PREP_STAGE}" >&2
        exit 2
        ;;
esac
