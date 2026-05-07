#!/bin/bash
# ==============================================================================
# BaGLM Online Server — Start the real-time step grounding service.
#
# The server receives camera frames from a robot via TCP socket, runs LMM
# inference (VSG + Progress) and incremental Bayesian filtering, then returns
# the current step belief.
#
# Prerequisites:
#   1. Dependency matrix .pt file (run custom_qa.py first, or use --no-prereq)
#   2. LMM model downloaded (or available via HF cache)
#
# Usage:
#   bash scripts/online_start_server.sh
#
#   # With a custom video for testing:
#   bash scripts/online_start_server.sh --test_video /path/to/video.mp4
# ==============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

# ---------- Environment ----------
export HF_HOME="${HF_HOME:-/mnt/public1/dais/hf_cache}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="${HOME}/opt/ffmpeg-n7.1-latest-linux64-gpl-shared-7.1/lib:${LD_LIBRARY_PATH:-}"

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
# shellcheck source=/dev/null
source "${CONDA_DIR}/bin/activate" baglm

export PYTHONPATH="src:t2v_metrics:${PYTHONPATH:-}"

# ---------- Config ----------
HOST="0.0.0.0"
PORT=9999
MODEL="internvl2.5-8b"
DEVICE="cuda"

TASK_CONFIG="configs/restock_cola.json"
PREREQ_DIR="/mnt/public1/dais/baglm_data/restock_cola/results/prereq"

SEGMENT_DURATION=2
SAMPLING_FPS=2
VISUAL_BATCH_SIZE=1
TEXT_BATCH_SIZE=1

# ---------- Parse optional overrides ----------
TEST_VIDEO=""
NO_PREREQ=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test_video)   TEST_VIDEO="$2";    shift 2 ;;
        --no-prereq)    NO_PREREQ=true;      shift   ;;
        --task_config)  TASK_CONFIG="$2";    shift 2 ;;
        --port)         PORT="$2";           shift 2 ;;
        --model)        MODEL="$2";          shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------- Server args ----------
SERVER_ARGS=(
    --host "$HOST"
    --port "$PORT"
    --task_config "$TASK_CONFIG"
    --model "$MODEL"
    --device "$DEVICE"
    --segment_duration "$SEGMENT_DURATION"
    --sampling_fps "$SAMPLING_FPS"
    --visual_batch_size "$VISUAL_BATCH_SIZE"
    --text_batch_size "$TEXT_BATCH_SIZE"
)

if $NO_PREREQ; then
    echo "[INFO] Running without dependency matrix (identity matrix will be used)."
else
    SERVER_ARGS+=(--prereq_dir "$PREREQ_DIR")
fi

echo "============================================"
echo " BaGLM Online Server"
echo "============================================"
echo " Task:   ${TASK_CONFIG}"
echo " Model:  ${MODEL}"
echo " Listen: ${HOST}:${PORT}"
echo "============================================"

# ---------- Start server (background) ----------
python src/online_server.py "${SERVER_ARGS[@]}" &
SERVER_PID=$!

# Give server time to load model
sleep 5

# ---------- Optionally start test client ----------
if [[ -n "$TEST_VIDEO" ]]; then
    echo ""
    echo "[INFO] Starting test client with video: ${TEST_VIDEO}"
    python src/online_client.py \
        --host localhost \
        --port "$PORT" \
        --video_path "$TEST_VIDEO" \
        --segment_duration "$SEGMENT_DURATION" \
        --sampling_fps "$SAMPLING_FPS"
fi

# Wait for server
wait $SERVER_PID
