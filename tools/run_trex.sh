#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/../project_env.sh"
TREX_REPO="${PROJECT_ROOT}/repos/T-Rex"
TREX_PYTHON="${LF3R_TREX_PYTHON:-${PROJECT_ROOT}/repos/ProcVLM/.venv/bin/python}"
export PYTHONPATH="${TREX_REPO}:${TREX_REPO}/dataset_quickstart/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
case "${1:-}" in
    train|infer)
        TREX_COMMAND="$1"
        shift
        if [[ "$TREX_COMMAND" == "infer" ]]; then
            TREX_COMMAND=test
        fi
        exec "$TREX_PYTHON" "${TREX_REPO}/scripts/${TREX_COMMAND}.py" "$@"
        ;;
    python)
        shift
        exec "$TREX_PYTHON" "$@"
        ;;
    *)
        echo "Usage: bash tools/run_trex.sh {train|infer|python} [arguments...]" >&2
        exit 2
        ;;
esac
