#!/bin/bash
# Multi-GPU launcher for qwen3vl_gres_eval.py (GRES / grefcoco).
# Data-parallel: shard the dataset across N GPUs, one process per GPU.
# Per-case JSON outputs => shards never collide and re-running resumes.
# The last shard to finish auto-computes the metric (gIoU/cIoU/N_acc/T_acc).
#
# Usage:
#   bash run_gres_multigpu.sh [NUM_GPUS] [MODEL_PATH] [SAVE_DIR]
# Example:
#   bash run_gres_multigpu.sh 8 zhouyik/Qwen3-VL-4B-SAMTok-co ./results/grefcoco/

set -u
NUM_GPUS=${1:-8}
MODEL_PATH=${2:-zhouyik/Qwen3-VL-4B-SAMTok-co}
SAVE_DIR=${3:-./results/grefcoco/}
VQ_SAM2_PATH=${VQ_SAM2_PATH:-./checkpoints/Qwen3-VL-4B-SAMTok-co/mask_tokenizer_256x2.pth}
SAM2_PATH=${SAM2_PATH:-./checkpoints/Qwen3-VL-4B-SAMTok-co/sam2.1_hiera_large.pt}
# Directory holding COCO train2014/; empty keeps the image paths stored in DATASET.
COCO_ROOT=${COCO_ROOT:-}
DATASET=${DATASET:-./data/PaDT-MLLM/RefCOCO/grefcoco_val.json}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL="$SCRIPT_DIR/qwen3vl_gres_eval.py"

# Relative paths (./data, ./checkpoints, ./results) resolve from the repo root.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
echo "repo root: $REPO_ROOT"
mkdir -p "$SAVE_DIR" logs

COMMON_ARGS=(--model_path "$MODEL_PATH" --vq_sam2_path "$VQ_SAM2_PATH" --sam2_path "$SAM2_PATH"
             --dataset "$DATASET" --save_dir "$SAVE_DIR")
[[ -n "$COCO_ROOT" ]] && COMMON_ARGS+=(--coco_root "$COCO_ROOT")
python "$EVAL" "${COMMON_ARGS[@]}" --check_only || exit 1

echo "Launching $NUM_GPUS shards | model=$MODEL_PATH | save=$SAVE_DIR"
pids=()
for ((t=0; t<NUM_GPUS; t++)); do
    CUDA_VISIBLE_DEVICES=$t python "$EVAL" "${COMMON_ARGS[@]}" \
        --task_id "$t" --num_tasks "$NUM_GPUS" --gpu_id 0 \
        > "logs/gres_shard${t}.log" 2>&1 &
    pids+=($!)
    echo "  shard $t -> GPU $t (pid ${pids[-1]}, log logs/gres_shard${t}.log)"
done

# ---- global progress monitor: count per-case json in SAVE_DIR vs dataset size ----
TOTAL=$(python3 - "$DATASET" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as source:
    print(len(json.load(source)))
PY
)
START=$(date +%s)
echo "monitoring progress: $TOTAL samples total (status every 20s)"
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

# reap shards; exit non-zero if any failed
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
done=$(find "$SAVE_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
echo "All shards finished (rc=$rc). $done/$TOTAL outputs in $SAVE_DIR"
if ! python3 "$SCRIPT_DIR/result_store.py" \
    --dataset "$DATASET" --model_path "$MODEL_PATH" \
    --vq_sam2_path "$VQ_SAM2_PATH" --sam2_path "$SAM2_PATH" \
    --save_dir "$SAVE_DIR" --require_complete; then
    echo "GRES output verification failed: some results are missing or stale" >&2
    rc=1
fi
echo "(metric was printed by the last shard's log; or re-run with --metric_only)"
exit $rc
