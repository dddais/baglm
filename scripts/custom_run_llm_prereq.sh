#!/bin/bash
# ============================================================
# Step 5: Generate prerequisite dependency matrices using LLM
# ============================================================
#
# Choose ONE of the following models:
#   A) llama-3.3-70b-instruct  (local, needs 4x H100 GPUs)
#   B) gpt-4.1-mini            (API, needs --oai_key_path)
#
# Default below uses gpt-4.1-mini via API.

# ---------- Environment ----------
export HF_HOME="/mnt/public1/dais/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENAI_BASE_URL="https://api.vectorengine.ai/v1"
export OMP_NUM_THREADS=32

CONDA_DIR="/mnt/public1/dais/miniconda3"
source "$CONDA_DIR/bin/activate" baglm

cd /home/dais/workspace/baglm
export PYTHONPATH="$PWD/src:$PWD/t2v_metrics:$PYTHONPATH"

# ---------- Paths ----------
DATA_DIR="/mnt/public1/dais/baglm_data/restock_cola"
VIDEO_ANNOTS_FILE="$DATA_DIR/video_annots.json"

MODEL="gpt-4.1-mini"
SYSTEM_FILE="prompts/prereq/system.txt"
QUESTION_FILE="prompts/prereq/question.txt"
RESULT_DIR="$DATA_DIR/results/prereq/"
mkdir -p "$RESULT_DIR"

# For GPT models, put your OpenAI API key in this file (one line):
OAI_KEY_PATH="/home/dais/workspace/baglm/OPENAI_KEY.txt"

# ---------- Run ----------
python src/custom_qa.py \
    --video_annots_file "$VIDEO_ANNOTS_FILE" \
    --result_dir "$RESULT_DIR" \
    --model "$MODEL" \
    --openai_base_url "$OPENAI_BASE_URL" \
    --system_file "$SYSTEM_FILE" \
    --question_file "$QUESTION_FILE" \
    --oai_key_path "$OAI_KEY_PATH" \
    --batch_size 64 #64
