#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT_GUESS="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT_GUESS/project_env.sh"
test "$PROJECT_ROOT" = "$PROJECT_ROOT_GUESS"

GPU_ID="${1:?usage: generate_libero_spatial_native.sh GPU_ID [TASK_START] [TASK_END] [TRIALS] [SEED] [RUN_NOTE] [--log-safe-features]}"
TASK_START="${2:-0}"
TASK_END="${3:-3}"
TRIALS="${4:-1}"
SEED="${5:-7}"
RUN_NOTE="${6:-lf3r-data-natural-libero-spatial-256-$(lf3r_file_timestamp)}"
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
  echo "GPU_ID must be one numeric CUDA device index" >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "$RUN_NOTE" =~ ^lf3r-data-natural-libero-spatial-256-[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_NOTE must preserve the LIBERO-Spatial native-256 provenance prefix" >&2
  exit 2
fi

IFS=',' read -r GPU_UTIL MEMORY_USED MEMORY_TOTAL MEMORY_FREE < <(
  nvidia-smi --id="$GPU_ID" --query-gpu=utilization.gpu,memory.used,memory.total,memory.free --format=csv,noheader,nounits | tr -d ' '
)
if ((MEMORY_FREE < 20480 )); then
  echo "WAITING_FOR_GPU_MEMORY gpu=$GPU_ID used_mib=$MEMORY_USED total_mib=$MEMORY_TOTAL free_mib=$MEMORY_FREE" >&2
  exit 75
fi

LOG_FILE="$(lf3r_log_path openvla_libero_spatial_native_rollouts)"
echo "RUN_NOTE=$RUN_NOTE" | tee -a "$LOG_FILE"
echo "TASK_SUITE=libero_spatial" | tee -a "$LOG_FILE"
echo "TASK_RANGE start=$TASK_START end=$TASK_END trials=$TRIALS" | tee -a "$LOG_FILE"
echo "RESOLUTION simulator=256x256 policy_input=224x224 replay_video=256x256" | tee -a "$LOG_FILE"
echo "GPU_GATE gpu=$GPU_ID utilization=${GPU_UTIL}% used_mib=$MEMORY_USED total_mib=$MEMORY_TOTAL free_mib=$MEMORY_FREE" | tee -a "$LOG_FILE"
echo "SAFE_FEATURES=${SAFE_FEATURE_MODE:-disabled}" | tee -a "$LOG_FILE"

set -o pipefail
env CUDA_VISIBLE_DEVICES="$GPU_ID" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH="$CACHE/libero" PYTHONPATH="$REPOS/safe-openvla:$REPOS/LIBERO" WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false \
  "$LF3R_ENV_OPENVLA/bin/python" "$PROJECT_ROOT/tools/lf3r_annotator/run_openvla_libero10_natural.py" \
  --task-suite libero_spatial --task-start "$TASK_START" --task-end "$TASK_END" --trials "$TRIALS" --run-note "$RUN_NOTE" --seed "$SEED" \
  "${SAFE_FEATURE_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
python3 "$PROJECT_ROOT/tools/lf3r_annotator/build_manifest.py" | tee -a "$LOG_FILE"
echo "NATURAL_LIBERO_SPATIAL_NATIVE_GENERATION_OK run_note=$RUN_NOTE" | tee -a "$LOG_FILE"
