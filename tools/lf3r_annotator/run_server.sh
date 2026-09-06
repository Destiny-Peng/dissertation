#!/usr/bin/env bash
set -euo pipefail

source /mnt/hdd/qiuxia/pyr/LF3R/project_env.sh
test "$PROJECT_ROOT" = "/mnt/hdd/qiuxia/pyr/LF3R"

ANNOTATOR_HOST="${LF3R_ANNOTATOR_HOST:-127.0.0.1}"
ANNOTATOR_PORT="${1:-8765}"
LOG_FILE="$(lf3r_log_path annotator_server)"
PID_FILE="$CACHE/tmp/lf3r_annotator.pid"

if [[ -f "$PID_FILE" ]]; then
    EXISTING_PID="$(<"$PID_FILE")"
    if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "LF3R annotator already has a live PID: $EXISTING_PID" >&2
        echo "Stop it with: bash tools/lf3r_annotator/stop_server.sh" >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi

python3 -u "$PROJECT_ROOT/tools/lf3r_annotator/server.py" \
    --host "$ANNOTATOR_HOST" \
    --port "$ANNOTATOR_PORT" \
    > >(tee "$LOG_FILE") 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$PID_FILE"

cleanup() {
    if [[ -f "$PID_FILE" ]] && [[ "$(<"$PID_FILE")" == "$SERVER_PID" ]]; then
        rm -f "$PID_FILE"
    fi
}

request_stop() {
    kill -TERM "$SERVER_PID" 2>/dev/null || true
}

trap request_stop INT TERM
trap cleanup EXIT

echo "PID: $SERVER_PID"
echo "Stop: bash tools/lf3r_annotator/stop_server.sh"
set +e
wait "$SERVER_PID"
STATUS=$?
set -e
if (( STATUS == 130 || STATUS == 143 )); then
    STATUS=0
fi
exit "$STATUS"
