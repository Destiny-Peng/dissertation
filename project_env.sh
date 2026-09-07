#!/usr/bin/env bash

# ------------------------------------------------------------
# LF3R project root
# ------------------------------------------------------------

export PROJECT_ROOT="/mnt/hdd/qiuxia/pyr/LF3R"

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
export LF3R_ENV_OPENVLA="${CONDA_ENVS}/LF3R-openvla"
export LF3R_ENV_ROBODOPAMINE="${CONDA_ENVS}/LF3R-robo-dopamine"
# Project-local uv venv used by temporal analysis. The path preserves the requested ananlyse spelling.
export LF3R_ENV_ANALYSE="${CONDA_ENVS}/LF3R-ananlyse"
export LF3R_ANALYSIS_PYTHON="${LF3R_ENV_ANALYSE}/bin/python"
export LF3R_ENV_DENSEREWARD="${CONDA_ENVS}/LF3R-densereward"
export LF3R_DENSEREWARD_PYTHON="${LF3R_ENV_DENSEREWARD}/bin/python"
export LF3R_DENSEREWARD_CHECKPOINT="${CHECKPOINTS}/densereward-3frame-thinking"

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
