#!/bin/bash
# ============================================================
# Step 4: Generate Progress scores using LMM
# ============================================================

# ---------- Environment ----------
export HF_HOME="/mnt/public1/dais/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=32

CONDA_DIR="/mnt/public1/dais/miniconda3"
source "$CONDA_DIR/bin/activate" baglm

cd /home/dais/workspace/baglm
export PYTHONPATH="$PWD/src:$PWD/t2v_metrics:$PYTHONPATH"

# ---------- Paths ----------
DATA_DIR="/mnt/public1/dais/baglm_data/restock_cola"
VIDEO_DIR="$DATA_DIR/videos"
VIDEO_ANNOTS_FILE="$DATA_DIR/video_annots.json"

MODEL="robobrain2.5-4b"
PROMPT_TYPE="prog"
QUESTION_FILE="prompts/$PROMPT_TYPE/question_robot.txt"
RESULT_DIR="$DATA_DIR/results/$PROMPT_TYPE/"
mkdir -p "$RESULT_DIR"

# ---------- Run ----------
python src/custom_eval.py \
    --video_dir "$VIDEO_DIR" \
    --video_annots_file "$VIDEO_ANNOTS_FILE" \
    --result_dir "$RESULT_DIR" \
    --model "$MODEL" \
    --decode_device cpu \
    --visual_batch_size 16 \
    --text_batch_size 1 \
    --segment_duration 4 \
    --sampling_fps 5 \
    --question_file "$QUESTION_FILE" \
    --num_workers 0 \
    --prompt_type "$PROMPT_TYPE"
