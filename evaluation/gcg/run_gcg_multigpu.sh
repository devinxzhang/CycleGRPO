#!/bin/bash
# Multi-GPU launcher for qwen3vl_gcg_eval.py (Grounded Caption Generation).
# Data-parallel: shard the image folder across N GPUs, one process per GPU.
# Per-image JSON outputs => shards never collide and re-running resumes
# (the dataset filters out already-evaluated images at startup).
#
# Usage:
#   bash run_gcg_multigpu.sh [NUM_GPUS] [MODEL_PATH] [SAVE_DIR]
# Example:
#   bash run_gcg_multigpu.sh 8 zhouyik/Qwen3-VL-4B-SAMTok-co ./results/gcg/

set -u
NUM_GPUS=${1:-8}
MODEL_PATH=${2:-zhouyik/Qwen3-VL-4B-SAMTok-co}
SAVE_DIR=${3:-./results/gcg/}
VQ_SAM2_PATH=${VQ_SAM2_PATH:-Qwen/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth}
SAM2_PATH=${SAM2_PATH:-Qwen/sam2.1_hiera_large.pt}
# image folder hardcoded in the script (IMAGE_FOLDER); keep in sync for the monitor:
IMAGE_FOLDER=${IMAGE_FOLDER:-<PATH_TO_DATA>/Sa2VA-Training/glamm_data/images/grandf/val_test}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL="$SCRIPT_DIR/qwen3vl_gcg_eval.py"

# script hardcodes ./results/... relative to the MaskTokenizer repo root
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
echo "repo root: $REPO_ROOT"
mkdir -p "$SAVE_DIR" logs

echo "Launching $NUM_GPUS shards | model=$MODEL_PATH | save=$SAVE_DIR"
pids=()
for ((t=0; t<NUM_GPUS; t++)); do
    CUDA_VISIBLE_DEVICES=$t python "$EVAL" \
        --model_path "$MODEL_PATH" \
        --vq_sam2_path "$VQ_SAM2_PATH" \
        --sam2_path "$SAM2_PATH" \
        --save_dir "$SAVE_DIR" \
        --launcher none \
        --task_id "$t" --num_tasks "$NUM_GPUS" --gpu_id 0 \
        > "logs/gcg_shard${t}.log" 2>&1 &
    pids+=($!)
    echo "  shard $t -> GPU $t (pid ${pids[-1]}, log logs/gcg_shard${t}.log)"
done

# ---- global progress monitor: count done json in SAVE_DIR vs #images in folder ----
TOTAL=$(find "$IMAGE_FOLDER" -maxdepth 1 -type f | wc -l)
START=$(date +%s)
echo "monitoring progress: $TOTAL images total (status every 20s)"
while true; do
    running=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && running=$((running+1)); done
    done=$(find "$SAVE_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    el=$(( $(date +%s) - START ))
    if [ "$done" -gt 0 ] && [ "$el" -gt 0 ]; then
        awk -v d="$done" -v T="$TOTAL" -v el="$el" -v r="$running" 'BEGIN{
            rate=d/el; eta=(rate>0)?(T-d)/rate:0;
            printf "[%4dm%02ds] %d/%d (%.0f%%) | %.2f/s | %d shards alive | ETA ~%dm%02ds\n",
                   el/60, el%60, d, T, 100*d/T, rate, r, eta/60, eta%60 }'
    else
        printf "[%4dm%02ds] %d/%d | %d shards alive (loading models...)\n" $((el/60)) $((el%60)) "$done" "$TOTAL" "$running"
    fi
    [ "$running" -eq 0 ] && break
    sleep 20
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
done=$(find "$SAVE_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
echo "All shards finished (rc=$rc). $done/$TOTAL outputs in $SAVE_DIR"
exit $rc
