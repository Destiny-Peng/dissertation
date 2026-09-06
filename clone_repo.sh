#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/hdd/qiuxia/pyr/LF3R"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
HF_HOME_DIR="${PROJECT_ROOT}/cache/huggingface"

mkdir -p "${CHECKPOINT_DIR}" "${HF_HOME_DIR}"

export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME_DIR}/hub"

cd "${PROJECT_ROOT}"

echo "===== Disk space before download ====="
df -h "${PROJECT_ROOT}"
echo

# Total expected size is ~53 GB.
# Require at least ~120 GB free to leave comfortable headroom.
FREE_GB=$(df --output=avail -BG "${PROJECT_ROOT}" | tail -1 | tr -dc '0-9')

if [ "${FREE_GB}" -lt 120 ]; then
    echo "[ERROR] Only ${FREE_GB} GB free."
    echo "Recommend at least 120 GB free before downloading all checkpoints."
    exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "[ERROR] 'hf' command not found."
    echo "Install Hugging Face CLI first, e.g.:"
    echo "  uv tool install huggingface_hub"
    exit 1
fi

download_model() {
    local repo="$1"
    local local_name="$2"
    local target="${CHECKPOINT_DIR}/${local_name}"

    echo
    echo "===================================================="
    echo "Downloading: ${repo}"
    echo "Target:      ${target}"
    echo "===================================================="

    if [ -d "${target}" ] && [ "$(find "${target}" -mindepth 1 -print -quit 2>/dev/null)" ]; then
        echo "[SKIP] ${target} already exists and is non-empty."
        return
    fi

    mkdir -p "${target}"

    hf download \
        "${repo}" \
        --local-dir "${target}"

    echo "[OK] ${local_name}"
}

download_model \
    "openvla/openvla-7b-finetuned-libero-10" \
    "openvla-7b-finetuned-libero-10"

download_model \
    "openvla/openvla-7b-finetuned-libero-spatial" \
    "openvla-7b-finetuned-libero-spatial"

download_model \
    "ce-amtic/ProcVLM-2B" \
    "ProcVLM-2B"

download_model \
    "Alibaba-DAMO-Academy/RynnValue-4B" \
    "RynnValue-4B"

download_model \
    "tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview" \
    "Robo-Dopamine-GRM-2.0-4B-Preview"

echo
echo "===== Downloaded checkpoint sizes ====="
du -sh "${CHECKPOINT_DIR}"/* 2>/dev/null | sort -h

echo
echo "===== Disk space after download ====="
df -h "${PROJECT_ROOT}"

echo
echo "All checkpoint downloads completed."