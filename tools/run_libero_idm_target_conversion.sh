#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/../project_env.sh"
cd "${PROJECT_ROOT}"
TARGET_OUTPUT="${1:-${DATASETS}/libero_idm_targets/libero_10_$(lf3r_file_timestamp)}"
TARGET_LOG="$(lf3r_log_path "libero_idm_target_conversion")"
printf '%s\n' "${TARGET_OUTPUT}" > "${CACHE}/libero_idm_targets_current_run.txt"
exec > >(tee -a "${TARGET_LOG}") 2>&1
printf 'Timestamp: %s\nPython: %s\nOutput: %s\nCommand: convert_idm_targets.py --output %s\n' "$(lf3r_timestamp)" "${LF3R_VERA_JIDM_PYTHON}" "${TARGET_OUTPUT}" "${TARGET_OUTPUT}"
"${LF3R_VERA_JIDM_PYTHON}" "${SCRIPT_DIR}/vera_libero/convert_idm_targets.py" --output "${TARGET_OUTPUT}"
