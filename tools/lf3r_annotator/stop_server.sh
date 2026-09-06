#!/usr/bin/env bash
set -euo pipefail

source /mnt/hdd/qiuxia/pyr/LF3R/project_env.sh
test "$PROJECT_ROOT" = "/mnt/hdd/qiuxia/pyr/LF3R"

PID_FILE="$CACHE/tmp/lf3r_annotator.pid"
if [[ ! -f "$PID_FILE" ]]; then
    echo "LF3R annotator is not running (no PID file)."
    exit 0
fi

SERVER_PID="$(<"$PID_FILE")"
if [[ ! "$SERVER_PID" =~ ^[0-9]+$ ]]; then
    echo "Invalid PID file: $PID_FILE" >&2
    exit 1
fi

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Removed stale PID file; LF3R annotator was not running."
    exit 0
fi

CMDLINE="$(tr '\0' ' ' < "/proc/$SERVER_PID/cmdline")"
if [[ "$CMDLINE" != *"$PROJECT_ROOT/tools/lf3r_annotator/server.py"* ]]; then
    echo "Refusing to stop PID $SERVER_PID because it is not the LF3R annotator." >&2
    exit 1
fi

kill -TERM "$SERVER_PID"
for _ in {1..50}; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "LF3R annotator stopped cleanly (PID $SERVER_PID)."
        exit 0
    fi
    sleep 0.1
done

echo "LF3R annotator did not stop within 5 seconds; no forced kill was sent." >&2
exit 1
