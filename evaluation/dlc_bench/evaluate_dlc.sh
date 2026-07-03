#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model_path> <cache_name>"
    echo "Example: $0 google/gemma-4b-it gemma4_bbox"
    exit 1
fi

MODEL_PATH="$1"
CACHE_NAME="$2"

python evaluation/dlc_bench/inference.py \
    --model_path "${MODEL_PATH}" \
    --cache_name "${CACHE_NAME}" \
    --data_type bf16 \
    --seed 42 \
    --vq_sam2_path Qwen/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth \
    --sam2_path Qwen/sam2.1_hiera_large.pt \