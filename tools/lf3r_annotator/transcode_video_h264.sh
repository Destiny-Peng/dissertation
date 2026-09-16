#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 <video.mp4>" >&2
    exit 2
fi

VIDEO="$1"
if [[ "$VIDEO" != *.mp4 ]]; then
    echo "Expected an .mp4 file: $VIDEO" >&2
    exit 2
fi
if [[ ! -f "$VIDEO" ]]; then
    echo "Video not found: $VIDEO" >&2
    exit 2
fi

BACKUP="${VIDEO%.mp4}.orig.mp4"
if [[ -e "$BACKUP" ]]; then
    echo "Backup already exists; refusing to overwrite: $BACKUP" >&2
    exit 3
fi

mv -- "$VIDEO" "$BACKUP"

restore_on_failure() {
    status=$?
    if (( status != 0 )); then
        rm -f -- "$VIDEO"
        if [[ -f "$BACKUP" ]]; then
            mv -- "$BACKUP" "$VIDEO"
        fi
        echo "Transcode failed; original video restored: $VIDEO" >&2
    fi
}
trap restore_on_failure EXIT

ffmpeg -nostdin -hide_banner -y \
    -i "$BACKUP" \
    -map 0:v:0 \
    -c:v libx264 \
    -preset veryfast \
    -crf 20 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    -an \
    "$VIDEO"

CODEC="$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$VIDEO" | head -n 1)"
if [[ "$CODEC" != "h264" ]]; then
    echo "Unexpected output codec: ${CODEC:-unknown}" >&2
    exit 4
fi

trap - EXIT
echo "Transcode complete"
echo "Output: $VIDEO"
echo "Original backup: $BACKUP"
echo "Codec: $CODEC"
