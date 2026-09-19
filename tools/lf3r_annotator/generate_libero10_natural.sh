#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT_GUESS="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT_GUESS/project_env.sh"
test "$PROJECT_ROOT" = "$PROJECT_ROOT_GUESS"

GPU_ID="${1:?usage: generate_libero10_natural.sh GPU_ID [TASK_START] [TASK_END] [TRIALS] [SEED] [RUN_NOTE] [options]}"
TASK_START="${2:-0}"
TASK_END="${3:-3}"
TRIALS="${4:-1}"
SEED="${5:-7}"
RUN_NOTE="${6:-lf3r-data-natural-libero10-$(lf3r_file_timestamp)}"
shift $(( $# >= 6 ? 6 : $# ))
SAFE_FEATURE_MODE="disabled"
SAFE_FEATURE_ARGS=()
RESOLUTION_ARGS=()
RENDER_RESOLUTION=""
RECORD_RESOLUTION=""
VIDEO_VIEW_MODE="single_view"
VIDEO_VIEW_ARGS=()
while (($# > 0)); do
    case "$1" in
        --log-safe-features)
            SAFE_FEATURE_MODE="enabled"
            SAFE_FEATURE_ARGS+=(--log-safe-features)
            shift
            ;;
        --render-resolution)
            [[ $# -ge 2 ]] || { echo "--render-resolution requires a value" >&2; exit 2; }
            RENDER_RESOLUTION="$2"
            RESOLUTION_ARGS+=(--render-resolution "$2")
            shift 2
            ;;
        --video-view-mode)
            [[ $# -ge 2 ]] || { echo "--video-view-mode requires a value" >&2; exit 2; }
            VIDEO_VIEW_MODE="$2"
            if [[ "$VIDEO_VIEW_MODE" != "single_view" && "$VIDEO_VIEW_MODE" != "libero_three_view" ]]; then
                echo "--video-view-mode must be single_view or libero_three_view" >&2
                exit 2
            fi
            VIDEO_VIEW_ARGS+=(--video-view-mode "$VIDEO_VIEW_MODE")
            shift 2
            ;;
        --record-resolution)
            [[ $# -ge 2 ]] || { echo "--record-resolution requires a value" >&2; exit 2; }
            RECORD_RESOLUTION="$2"
            RESOLUTION_ARGS+=(--record-resolution "$2")
            shift 2
            ;;
        *)
            echo "unknown option: $1" >&2
            exit 2
            ;;
    esac
done

validate_resolution() {
    local name="$1"
    local value="$2"
    if [[ -n "$value" ]] && { [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 64 || value > 2048 || value % 2 != 0 )); }; then
        echo "$name must be an even integer between 64 and 2048" >&2
        exit 2
    fi
}
validate_resolution render-resolution "$RENDER_RESOLUTION"
validate_resolution record-resolution "$RECORD_RESOLUTION"

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

LOG_FILE="$(lf3r_log_path openvla_libero10_natural_rollouts)"
ROBOSUITE_LOG_DIR="${LOGS}/robosuite"
ROBOSUITE_LOG_FILE="${ROBOSUITE_LOG_DIR}/${RUN_NOTE}.log"
mkdir -p "$ROBOSUITE_LOG_DIR"

echo "RUN_NOTE=$RUN_NOTE" | tee "$LOG_FILE"
echo "GPU=$GPU_ID TASK_START=$TASK_START TASK_END=$TASK_END TRIALS=$TRIALS SEED=$SEED RUN_NOTE=$RUN_NOTE" | tee -a "$LOG_FILE"
echo "GPU_SELECTION=user_managed gpu=$GPU_ID" | tee -a "$LOG_FILE"
echo "SAFE_FEATURES=$SAFE_FEATURE_MODE" | tee -a "$LOG_FILE"
echo "RESOLUTION render=${RENDER_RESOLUTION:-suite-default} record=${RECORD_RESOLUTION:-suite-default} policy=224"
echo "VIDEO_VIEW_MODE=$VIDEO_VIEW_MODE" | tee -a "$LOG_FILE"
echo "ROBOSUITE_LOG_PATH=$ROBOSUITE_LOG_FILE" | tee -a "$LOG_FILE"

set -o pipefail
env \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    LIBERO_CONFIG_PATH="$CACHE/libero" \
    PYTHONPATH="$REPOS/safe-openvla" \
    WANDB_DISABLED=true \
    ROBOSUITE_LOG_PATH="$ROBOSUITE_LOG_FILE" \
    TOKENIZERS_PARALLELISM=false \
    "$LF3R_ENV_OPENVLA/bin/python" \
    "$PROJECT_ROOT/tools/lf3r_annotator/run_openvla_libero10_natural.py" \
    --task-start "$TASK_START" \
    --task-end "$TASK_END" \
    --trials "$TRIALS" \
    --run-note "$RUN_NOTE" \
    --seed "$SEED" \
    "${SAFE_FEATURE_ARGS[@]}" \
    "${VIDEO_VIEW_ARGS[@]}" \
    "${RESOLUTION_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE"

python3 "$PROJECT_ROOT/tools/lf3r_annotator/build_manifest.py" | tee -a "$LOG_FILE"
echo "NATURAL_LIBERO10_GENERATION_OK run_note=$RUN_NOTE" | tee -a "$LOG_FILE"
