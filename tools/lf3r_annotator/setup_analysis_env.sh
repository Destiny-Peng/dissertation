#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../project_env.sh"

REQUIREMENTS="$PROJECT_ROOT/tools/lf3r_annotator/requirements-analysis.txt"
ANALYZER="$PROJECT_ROOT/tools/analyze_baseline_temporal_signals.py"

command -v uv >/dev/null 2>&1 || {
    echo "uv is required to create the project-local analysis environment" >&2
    exit 1
}
test -f "$REQUIREMENTS"
test -f "$ANALYZER"

if [[ -e "$LF3R_ENV_ANALYSE" && ! -x "$LF3R_ANALYSIS_PYTHON" ]]; then
    echo "Refusing to replace an existing non-venv path: $LF3R_ENV_ANALYSE" >&2
    exit 1
fi

if [[ ! -x "$LF3R_ANALYSIS_PYTHON" ]]; then
    uv venv --python 3.10.21 "$LF3R_ENV_ANALYSE"
fi

uv pip install --python "$LF3R_ANALYSIS_PYTHON" --requirement "$REQUIREMENTS"
MPLBACKEND=Agg "$LF3R_ANALYSIS_PYTHON" -c \
    'import matplotlib, numpy, pandas; print("analysis dependencies OK")'
MPLBACKEND=Agg "$LF3R_ANALYSIS_PYTHON" "$ANALYZER" --help >/dev/null
echo "Analysis environment ready: $LF3R_ENV_ANALYSE"
