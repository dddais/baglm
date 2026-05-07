#!/bin/bash
# ==============================================================================
# BaGLM Online Server — RoboBrain2.5 variant
#
# Uses baglm-robobrain conda env (transformers>=4.57, needed by Qwen3-VL).
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

CONDA_DIR="${CONDA_DIR:-/mnt/public1/dais/miniconda3}"
# shellcheck source=/dev/null
source "${CONDA_DIR}/bin/activate" baglm-robobrain

export PYTHONPATH="src:t2v_metrics:${PYTHONPATH:-}"

# ---------- Config ----------
HOST="0.0.0.0"
PORT=9999
MODEL="robobrain2.5-4b"
DEVICE="cuda"

TASK_CONFIG="configs/restock_cola.json"

SEGMENT_DURATION=2
SAMPLING_FPS=2
VISUAL_BATCH_SIZE=1
TEXT_BATCH_SIZE=1

# ---------- Parse optional overrides ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_config)  TASK_CONFIG="$2";    shift 2 ;;
        --port)         PORT="$2";           shift 2 ;;
        --model)        MODEL="$2";          shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo " BaGLM Online Server (RoboBrain)"
echo "============================================"
echo " Task:   ${TASK_CONFIG}"
echo " Model:  ${MODEL}"
echo " Listen: ${HOST}:${PORT}"
echo "============================================"

python src/online_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --task_config "$TASK_CONFIG" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --segment_duration "$SEGMENT_DURATION" \
    --sampling_fps "$SAMPLING_FPS" \
    --visual_batch_size "$VISUAL_BATCH_SIZE" \
    --text_batch_size "$TEXT_BATCH_SIZE"
