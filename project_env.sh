#!/usr/bin/env bash

# ------------------------------------------------------------
# LF3R project root
# ------------------------------------------------------------

# Default to the repository directory that contains this file. This makes one
# checkout portable across servers with different absolute paths. Set
# LF3R_PROJECT_ROOT only when the project data intentionally lives elsewhere.
_LF3R_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PROJECT_ROOT="${LF3R_PROJECT_ROOT:-${_LF3R_ENV_DIR}}"
PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd -P)"
export PROJECT_ROOT
unset _LF3R_ENV_DIR

# ------------------------------------------------------------
# LF3R directories
# ------------------------------------------------------------

export REPOS="${PROJECT_ROOT}/repos"
export DATASETS="${PROJECT_ROOT}/datasets"
export CHECKPOINTS="${PROJECT_ROOT}/checkpoints"
export CONDA_ENVS="${PROJECT_ROOT}/conda_envs"

export CACHE="${PROJECT_ROOT}/cache"
export OUTPUTS="${PROJECT_ROOT}/outputs"
export LOGS="${PROJECT_ROOT}/logs"
export ENV_REPORTS="${PROJECT_ROOT}/environment_reports"
export LF3R_CONDA_EXE="${PROJECT_ROOT}/tools/miniforge3/bin/conda"

# ------------------------------------------------------------
# Package caches
# ------------------------------------------------------------

export CONDA_PKGS_DIRS="${CACHE}/conda/pkgs"
export PIP_CACHE_DIR="${CACHE}/pip"
export UV_CACHE_DIR="${CACHE}/uv"

export HF_HOME="${CACHE}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

export TORCH_HOME="${CACHE}/torch"
export XDG_CACHE_HOME="${CACHE}/xdg"
export TMPDIR="${CACHE}/tmp"

# ------------------------------------------------------------
# LF3R conda environment prefixes
# ------------------------------------------------------------

export LF3R_ENV_SAFE="${CONDA_ENVS}/LF3R-safe"
export LF3R_SAFE_PYTHON="${LF3R_ENV_SAFE}/bin/python"
export LF3R_ENV_OPENVLA="${CONDA_ENVS}/LF3R-openvla"
export LF3R_ENV_ROBODOPAMINE="${CONDA_ENVS}/LF3R-robo-dopamine"
export LF3R_ROBODOPAMINE_PYTHON="${LF3R_ENV_ROBODOPAMINE}/bin/python"
# Project-local uv venv used by temporal analysis. The path preserves the requested ananlyse spelling.
export LF3R_ENV_ANALYSE="${CONDA_ENVS}/LF3R-ananlyse"
export LF3R_ANALYSIS_PYTHON="${LF3R_ENV_ANALYSE}/bin/python"
export LF3R_ENV_DENSEREWARD="${CONDA_ENVS}/LF3R-densereward"
export LF3R_DENSEREWARD_PYTHON="${LF3R_ENV_DENSEREWARD}/bin/python"
export LF3R_DENSEREWARD_CHECKPOINT="${CHECKPOINTS}/densereward-3frame-thinking"
export LF3R_ENV_CTRL_WORLD="${CONDA_ENVS}/Ctrl-World"
export LF3R_CTRL_WORLD_PYTHON="${LF3R_ENV_CTRL_WORLD}/bin/python"

# J-IDM only; Python 3.11 and Blackwell-compatible PyTorch.
export LF3R_ENV_VERA_JIDM="${CONDA_ENVS}/LF3R-vera-jidm"
export LF3R_VERA_JIDM_PYTHON="${LF3R_ENV_VERA_JIDM}/bin/python"

export LF3R_ENV_MANISKILL3="${CONDA_ENVS}/LF3R-maniskill3"
export LF3R_MANISKILL3_PYTHON="${LF3R_ENV_MANISKILL3}/bin/python"
export MS_ASSET_DIR="${DATASETS}/maniskill3"
export SAPIEN_CACHE_DIR="${CACHE}/sapien"
export CUDA_CACHE_PATH="${CACHE}/cuda"
export __GL_SHADER_DISK_CACHE_PATH="${CACHE}/nvidia"

# Make the project-local Conda command available in sourced Bash sessions.
if [[ -f "${PROJECT_ROOT}/tools/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "${PROJECT_ROOT}/tools/miniforge3/etc/profile.d/conda.sh"
fi

# ------------------------------------------------------------
# Timestamp helpers
# ------------------------------------------------------------

lf3r_timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

lf3r_file_timestamp() {
    date '+%Y%m%d_%H%M%S'
}

lf3r_log_path() {
    local component="$1"
    echo "${LOGS}/${component}_$(lf3r_file_timestamp).log"
}

export -f lf3r_timestamp
export -f lf3r_file_timestamp
export -f lf3r_log_path

# ------------------------------------------------------------
# Create LF3R project-local directories
# ------------------------------------------------------------

mkdir -p \
    "${REPOS}" \
    "${DATASETS}" \
    "${CHECKPOINTS}" \
    "${CONDA_ENVS}" \
    "${CACHE}/conda/pkgs" \
    "${CACHE}/pip" \
    "${CACHE}/uv" \
    "${CACHE}/huggingface" \
    "${CACHE}/huggingface/hub" \
    "${CACHE}/huggingface/transformers" \
    "${CACHE}/torch" \
    "${CACHE}/xdg" \
    "${CACHE}/tmp" \
    "${OUTPUTS}" \
    "${LOGS}" \
    "${ENV_REPORTS}"
