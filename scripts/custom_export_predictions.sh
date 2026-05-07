#!/bin/bash
# ============================================================
# Export readable BaGLM predictions (JSON / CSV / HTML)
# ============================================================

CONDA_DIR="/mnt/public1/dais/miniconda3"
source "$CONDA_DIR/bin/activate" baglm

cd /home/dais/workspace/baglm
export PYTHONPATH="$PWD/src:$PWD/t2v_metrics:$PYTHONPATH"

DATA_DIR="/mnt/public1/dais/baglm_data/restock_cola"
VIDEO_ANNOTS_FILE="$DATA_DIR/video_annots.json"

MODEL="robobrain2.5-4b"
LMM_VSG_DIR="$DATA_DIR/results/vsg/"
LMM_PROG_DIR="$DATA_DIR/results/prog/"
LLM_PREREQ_DIR="$DATA_DIR/results/prereq/gpt-4.1-mini/"
OUTPUT_DIR="$DATA_DIR/results/visualization/"

python src/custom_export_predictions.py \
    --video_annots_file "$VIDEO_ANNOTS_FILE" \
    --model "$MODEL" \
    --lmm_vsg_dir "$LMM_VSG_DIR" \
    --lmm_prog_dir "$LMM_PROG_DIR" \
    --llm_prereq_dir "$LLM_PREREQ_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --segment_duration 1
