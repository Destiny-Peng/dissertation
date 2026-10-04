#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/../project_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/tools${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
case "${1:-}" in
    prepare|train|verify|verify_online|report)
        SHARPA_COMMAND="$1"
        shift
        exec bash "${PROJECT_ROOT}/tools/run_trex.sh" python -m "sharpa_tactile.${SHARPA_COMMAND}" "$@"
        ;;
    *)
        echo "Usage: bash tools/run_sharpa_tactile_ablation.sh {prepare|train|verify|verify_online|report} [arguments...]" >&2
        exit 2
        ;;
esac
