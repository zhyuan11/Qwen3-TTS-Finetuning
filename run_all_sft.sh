#!/usr/bin/env bash
set -e

export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CD_DIR="/root/workspace/Qwen3-TTS/finetuning"
CONDA_ENV="qwen3-tts"
MODEL_PATH="/root/workspace/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base"
LOG_DIR="${CD_DIR}/logs"
BATCH_SIZE=2
LR=2e-6
NUM_EPOCHS=10

cd "$CD_DIR"

SPEAKERS=("MT" "安吉拉" "汉克狗" "狗狗本")

for SPEAKER in "${SPEAKERS[@]}"; do
    TRAIN_JSONL="train_${SPEAKER}_with_codes.jsonl"
    OUTPUT_DIR="${CD_DIR}/output_${SPEAKER}"
    LOG_FILE="${LOG_DIR}/sft_${SPEAKER}.log"

    if [ -d "${OUTPUT_DIR}/checkpoint-epoch-$((NUM_EPOCHS - 1))" ]; then
        echo "[$(date)] Skipping ${SPEAKER} -- already has all ${NUM_EPOCHS} checkpoints" | tee -a "$LOG_FILE"
        continue
    fi

    echo "[$(date)] Starting SFT for speaker: ${SPEAKER}" | tee -a "$LOG_FILE"
    echo "[$(date)]   JSONL:  ${TRAIN_JSONL}" | tee -a "$LOG_FILE"
    echo "[$(date)]   Output: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"

    conda run -n "$CONDA_ENV" python sft_12hz.py \
        --init_model_path "$MODEL_PATH" \
        --output_model_path "$OUTPUT_DIR" \
        --train_jsonl "$TRAIN_JSONL" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --num_epochs "$NUM_EPOCHS" \
        --speaker_name "$SPEAKER" \
        2>&1 | tee -a "$LOG_FILE"

    echo "[$(date)] Finished SFT for speaker: ${SPEAKER}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "[$(date)] All speakers done!" | tee -a "${LOG_DIR}/sft_all.log"
