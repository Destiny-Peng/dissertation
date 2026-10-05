#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/../project_env.sh"
cd "${PROJECT_ROOT}"
SETUP_LOG="$(lf3r_log_path vera_jidm_setup)"
exec > >(tee -a "${SETUP_LOG}") 2>&1
VERA_COMMIT="e9bf1c8033a31be9c42edd6d7348e92c0cfdf9eb"
VERA_PATCH="${SCRIPT_DIR}/patches/vera-jidm-blackwell-project-cache.patch"
printf 'Timestamp: %s\nEnvironment: %s\n' "$(lf3r_timestamp)" "${LF3R_ENV_VERA_JIDM}"
df -h "${PROJECT_ROOT}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv
if [[ ! -d "${REPOS}/VERA/.git" ]]; then
    git clone https://github.com/sizhe-li/VERA.git "${REPOS}/VERA"
    git -C "${REPOS}/VERA" checkout "${VERA_COMMIT}"
fi
if [[ "$(git -C "${REPOS}/VERA" rev-parse HEAD)" != "${VERA_COMMIT}" ]]; then
    printf 'Expected VERA commit %s; refusing to change an existing checkout.\n' "${VERA_COMMIT}" >&2
    exit 1
fi
if ! git -C "${REPOS}/VERA" apply --reverse --check "${VERA_PATCH}"; then
    git -C "${REPOS}/VERA" apply --check "${VERA_PATCH}"
    git -C "${REPOS}/VERA" apply "${VERA_PATCH}"
fi
if [[ ! -x "${LF3R_VERA_JIDM_PYTHON}" ]]; then
    "${LF3R_CONDA_EXE}" create -y -p "${LF3R_ENV_VERA_JIDM}" python=3.11 pip
fi
# CUDA wheels are installed in one step before the remaining shared/IDM deps.
uv pip install --python "${LF3R_VERA_JIDM_PYTHON}" \
    torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "${LF3R_VERA_JIDM_PYTHON}" \
    -e "${REPOS}/VERA[idm]" -r "${SCRIPT_DIR}/requirements-vera-jidm.txt"
"${LF3R_VERA_JIDM_PYTHON}" -m pip check
HF_HUB_OFFLINE=1 WANDB_MODE=disabled MPLCONFIGDIR="${CACHE}/matplotlib" \
    "${LF3R_VERA_JIDM_PYTHON}" "${SCRIPT_DIR}/verify_vera_jidm.py"
"${SCRIPT_DIR}/run_vera_jidm.sh" --help
