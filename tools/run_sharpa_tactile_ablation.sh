#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/../project_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/tools${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
case "${1:-}" in
    normalize_align_online|report_align_online|train_align_online|prepare_align_online|align_windows_data|align_windows|verify_align_windows|report_align_windows|publish_align_windows|prepare|train|verify|verify_online|report|prepare_three|train_three|verify_three|report_three|weight_seed_ablation|prepare_intervals|train_intervals|verify_intervals|report_intervals)
        SHARPA_COMMAND="$1"
        shift
        exec bash "${PROJECT_ROOT}/tools/run_trex.sh" python -m "sharpa_tactile.${SHARPA_COMMAND}" "$@"
        ;;
    *)
        echo "Usage: bash tools/run_sharpa_tactile_ablation.sh {normalize_align_online|report_align_online|train_align_online|prepare_align_online|align_windows_data|align_windows|verify_align_windows|report_align_windows|publish_align_windows|prepare_intervals|train_intervals|verify_intervals|report_intervals|weight_seed_ablation|prepare_three|train_three|verify_three|report_three|verify_online|prepare|train|verify|report} [arguments...]" >&2
        exit 2
        ;;
esac
