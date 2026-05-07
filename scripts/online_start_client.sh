#!/bin/bash
# ==============================================================================
# BaGLM Online Client — Connect to the server using a video file or webcam.
#
# Usage:
#   # From a video file:
#   bash scripts/online_start_client.sh --video_path /path/to/video.mp4
#
#   # From webcam:
#   bash scripts/online_start_client.sh --webcam
#
#   # Custom server address:
#   bash scripts/online_start_client.sh --host 192.168.1.100 --port 9999 \
#       --video_path /path/to/video.mp4
# ==============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

# ---------- Environment ----------
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
# shellcheck source=/dev/null
source "${CONDA_DIR}/bin/activate" baglm

export PYTHONPATH="src:t2v_metrics:${PYTHONPATH:-}"

# ---------- Defaults ----------
HOST="localhost"
PORT=9999
SEGMENT_DURATION=2
SAMPLING_FPS=2
MAX_SEGMENTS=0

# ---------- Parse args ----------
VIDEO_PATH=""
WEBCAM=false
WEBCAM_ID=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)           HOST="$2";           shift 2 ;;
        --port)           PORT="$2";           shift 2 ;;
        --video_path)     VIDEO_PATH="$2";     shift 2 ;;
        --webcam)         WEBCAM=true;         shift   ;;
        --webcam_id)      WEBCAM_ID="$2";      shift 2 ;;
        --segment_duration) SEGMENT_DURATION="$2"; shift 2 ;;
        --sampling_fps)   SAMPLING_FPS="$2";   shift 2 ;;
        --max_segments)   MAX_SEGMENTS="$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

CLIENT_ARGS=(
    --host "$HOST"
    --port "$PORT"
    --segment_duration "$SEGMENT_DURATION"
    --sampling_fps "$SAMPLING_FPS"
    --max_segments "$MAX_SEGMENTS"
)

if $WEBCAM; then
    CLIENT_ARGS+=(--webcam --webcam_id "$WEBCAM_ID")
    echo "Source: webcam (device ${WEBCAM_ID})"
elif [[ -n "$VIDEO_PATH" ]]; then
    CLIENT_ARGS+=(--video_path "$VIDEO_PATH")
    echo "Source: video (${VIDEO_PATH})"
else
    echo "ERROR: Specify --video_path <path> or --webcam"
    exit 1
fi

echo "Server: ${HOST}:${PORT}"
echo "Segment: ${SEGMENT_DURATION}s @ ${SAMPLING_FPS}fps"
echo "============================================"

python src/online_client.py "${CLIENT_ARGS[@]}"
