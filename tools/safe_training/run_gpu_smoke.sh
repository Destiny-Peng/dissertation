#!/usr/bin/env bash
set -euo pipefail

source /mnt/hdd/qiuxia/pyr/LF3R/project_env.sh
test "$PROJECT_ROOT" = "/mnt/hdd/qiuxia/pyr/LF3R"

GPU_ID="${1:-1}"
SMOKE_ROOT="${PROJECT_ROOT}/outputs/safe_training/gpu_smoke_$(date +%Y%m%d_%H%M%S)"
DATASET="${SMOKE_ROOT}/mock_dataset"

# This is intentionally GPU-only. GPU utilization and existing compute
# processes are informational; only free memory gates this bounded mock run.
# Override the conservative threshold when the user has measured a smaller
# safe margin for this workstation.
MIN_FREE_MIB="${SAFE_SMOKE_MIN_FREE_MIB:-16000}"
FREE_MIB="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ ! "$FREE_MIB" =~ ^[0-9]+$ ]] || (( FREE_MIB < MIN_FREE_MIB )); then
    echo "GPU $GPU_ID has insufficient free memory (${FREE_MIB:-unknown} MiB; required ${MIN_FREE_MIB} MiB); refusing SAFE smoke." >&2
    exit 75
fi
echo "GPU $GPU_ID free memory: ${FREE_MIB} MiB; utilization/process contention is not a gate."

mkdir -p "$SMOKE_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$LF3R_SAFE_PYTHON" "$PROJECT_ROOT/tools/safe_training/create_mock_dataset.py" \
    --output "$DATASET" --task-count 4 --episodes-per-task 8 --steps 80

CUDA_VISIBLE_DEVICES="$GPU_ID" WANDB_MODE=offline MPLBACKEND=Agg \
  "$LF3R_SAFE_PYTHON" "$PROJECT_ROOT/tools/safe_training/run_safe_training.py" \
  --dataset-dir "$DATASET" --model mlp --gpu "$GPU_ID" --epochs 1 --batch-size 16 --hidden-dim 16 \
  --logs-root "$SMOKE_ROOT/logs_mlp" --extra train.roc_every=1 --extra train.eval_save_logs=true \
  > "$SMOKE_ROOT/mlp.log" 2>&1
MLP_CKPT="$(find "$SMOKE_ROOT/logs_mlp" -name model_final.ckpt -type f -print -quit)"
[[ -n "$MLP_CKPT" ]]
CUDA_VISIBLE_DEVICES="$GPU_ID" "$LF3R_SAFE_PYTHON" "$PROJECT_ROOT/tools/safe_training/validate_safe_checkpoint.py" \
  --dataset-dir "$DATASET" --checkpoint "$MLP_CKPT" --model mlp --gpu "$GPU_ID" \
  --output "$SMOKE_ROOT/mlp_scores.json"

CUDA_VISIBLE_DEVICES="$GPU_ID" WANDB_MODE=offline MPLBACKEND=Agg \
  "$LF3R_SAFE_PYTHON" "$PROJECT_ROOT/tools/safe_training/run_safe_training.py" \
  --dataset-dir "$DATASET" --model lstm --gpu "$GPU_ID" --epochs 1 --batch-size 16 --hidden-dim 16 \
  --logs-root "$SMOKE_ROOT/logs_lstm" --extra train.roc_every=1 --extra train.eval_save_logs=true \
  > "$SMOKE_ROOT/lstm.log" 2>&1
LSTM_CKPT="$(find "$SMOKE_ROOT/logs_lstm" -name model_final.ckpt -type f -print -quit)"
[[ -n "$LSTM_CKPT" ]]
CUDA_VISIBLE_DEVICES="$GPU_ID" "$LF3R_SAFE_PYTHON" "$PROJECT_ROOT/tools/safe_training/validate_safe_checkpoint.py" \
  --dataset-dir "$DATASET" --checkpoint "$LSTM_CKPT" --model lstm --gpu "$GPU_ID" \
  --output "$SMOKE_ROOT/lstm_scores.json"

cat > "$SMOKE_ROOT/smoke_status.json" <<EOF
{
  "status": "complete",
  "gpu": "$GPU_ID",
  "dataset": "$DATASET",
  "mlp_checkpoint": "$MLP_CKPT",
  "lstm_checkpoint": "$LSTM_CKPT",
  "token_idx_rel": 1.0,
  "feature_shape_before_token_selection": [80, 7, 4096],
  "feature_shape_after_token_selection": [80, 4096]
}
EOF
cat "$SMOKE_ROOT/smoke_status.json"
