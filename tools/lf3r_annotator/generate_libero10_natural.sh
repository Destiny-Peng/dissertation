#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT_GUESS="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT_GUESS/project_env.sh"
test "$PROJECT_ROOT" = "$PROJECT_ROOT_GUESS"

GPU_ID="${1:?usage: generate_libero10_natural.sh GPU_ID [TASK_START] [TASK_END] [TRIALS] [SEED] [RUN_NOTE] [--log-safe-features]}"
TASK_START="${2:-0}"
TASK_END="${3:-3}"
TRIALS="${4:-1}"
SEED="${5:-7}"
RUN_NOTE="${6:-lf3r-data-natural-libero10-$(lf3r_file_timestamp)}"
SAFE_FEATURE_MODE="${7:-}"
if [[ -n "$SAFE_FEATURE_MODE" && "$SAFE_FEATURE_MODE" != "--log-safe-features" ]]; then
    echo "optional seventh argument must be --log-safe-features" >&2
    exit 2
fi
if [[ "$SAFE_FEATURE_MODE" == "--log-safe-features" ]]; then
    SAFE_FEATURE_ARGS=(--log-safe-features)
else
    SAFE_FEATURE_ARGS=()
fi

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be numeric" >&2
    exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "$RUN_NOTE" =~ ^lf3r-data-natural-libero10-[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NOTE must preserve natural LIBERO-10 provenance" >&2
    exit 2
fi

IFS=',' read -r GPU_UTIL MEMORY_USED MEMORY_TOTAL MEMORY_FREE < <(
    nvidia-smi \
        --id="$GPU_ID" \
        --query-gpu=utilization.gpu,memory.used,memory.total,memory.free \
        --format=csv,noheader,nounits | tr -d ' '
)

if (( MEMORY_USED * 2 >= MEMORY_TOTAL || MEMORY_FREE < 30720 )); then
    echo "WAITING_FOR_GPU_MEMORY gpu=$GPU_ID utilization_ignored=$GPU_UTIL used_mib=$MEMORY_USED total_mib=$MEMORY_TOTAL free_mib=$MEMORY_FREE"
    exit 75
fi

LOG_FILE="$(lf3r_log_path openvla_libero10_natural_rollouts)"

echo "RUN_NOTE=$RUN_NOTE" | tee "$LOG_FILE"
echo "GPU=$GPU_ID TASK_START=$TASK_START TASK_END=$TASK_END TRIALS=$TRIALS SEED=$SEED RUN_NOTE=$RUN_NOTE" | tee -a "$LOG_FILE"
echo "GPU_GATE=memory_only utilization_ignored=$GPU_UTIL used_mib=$MEMORY_USED total_mib=$MEMORY_TOTAL free_mib=$MEMORY_FREE" | tee -a "$LOG_FILE"
echo "SAFE_FEATURES=${SAFE_FEATURE_MODE:-disabled}" | tee -a "$LOG_FILE"

set -o pipefail
env \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    LIBERO_CONFIG_PATH="$CACHE/libero" \
    PYTHONPATH="$REPOS/safe-openvla" \
    WANDB_DISABLED=true \
    TOKENIZERS_PARALLELISM=false \
    "$LF3R_ENV_OPENVLA/bin/python" \
    "$PROJECT_ROOT/tools/lf3r_annotator/run_openvla_libero10_natural.py" \
    --task-start "$TASK_START" \
    --task-end "$TASK_END" \
    --trials "$TRIALS" \
    --run-note "$RUN_NOTE" \
    --seed "$SEED" \
    "${SAFE_FEATURE_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE"

python3 "$PROJECT_ROOT/tools/lf3r_annotator/build_manifest.py" | tee -a "$LOG_FILE"
echo "NATURAL_LIBERO10_GENERATION_OK run_note=$RUN_NOTE" | tee -a "$LOG_FILE"
