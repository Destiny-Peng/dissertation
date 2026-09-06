#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Reuse LF3R's project-local cache and environment locations.
source "${PROJECT_DIR}/project_env.sh"
exec /usr/bin/python3 "${SCRIPT_DIR}/run_lf3r_baseline.py" "$@"
