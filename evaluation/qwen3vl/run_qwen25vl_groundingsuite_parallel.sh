#!/usr/bin/env bash
# Multi-GPU data-parallel launcher for qwen25vl_groundingsuite_infer_bbox.py
#
# Each GPU runs an independent process on a disjoint chunk of the dataset
# (split via --task_id / --num_tasks). All processes write to the same
# --save_dir, indexed by sample idx, so there is no write conflict and
# any process can be safely re-run to resume.
#
# Usage:
#   bash run_qwen25vl_groundingsuite_parallel.sh
#   NUM_GPUS=4 GPUS=0,1,2,3 bash run_qwen25vl_groundingsuite_parallel.sh
#   MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct bash run_qwen25vl_groundingsuite_parallel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/qwen25vl_groundingsuite_infer_bbox.py"

# ---- configurable knobs (override via env) ---------------------------------
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
SAVE_DIR="${SAVE_DIR:-./results/groundingsuite_bbox_qwen25vl/}"
DATASET="${DATASET:-./data/GroundingSuiteEval/GroundingSuite-Eval.jsonl}"
LOG_DIR="${LOG_DIR:-./logs/qwen25vl_groundingsuite}"
# ----------------------------------------------------------------------------

IFS=',' read -ra GPU_ARR <<< "${GPUS}"
NUM_GPUS="${NUM_GPUS:-${#GPU_ARR[@]}}"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

echo "Launching ${NUM_GPUS} workers on GPUs: ${GPUS}"
echo "  model_path = ${MODEL_PATH}"
echo "  dataset    = ${DATASET}"
echo "  save_dir   = ${SAVE_DIR}"
echo "  log_dir    = ${LOG_DIR}"

pids=()
cleanup() {
    echo "Caught signal, killing child workers..."
    for pid in "${pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait
    exit 130
}
trap cleanup INT TERM

for i in "${!GPU_ARR[@]}"; do
    GPU_ID="${GPU_ARR[$i]}"
    LOG_FILE="${LOG_DIR}/task_${i}_of_${NUM_GPUS}_gpu${GPU_ID}.log"
    echo "  -> task_id=${i}/${NUM_GPUS} on cuda:${GPU_ID} | log=${LOG_FILE}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${PY_SCRIPT}" \
        --model_path "${MODEL_PATH}" \
        --save_dir "${SAVE_DIR}" \
        --dataset "${DATASET}" \
        --task_id "${i}" \
        --num_tasks "${NUM_GPUS}" \
        > "${LOG_FILE}" 2>&1 &
    pids+=("$!")
done

# Wait for all workers; if any fails, surface a non-zero exit code.
fail=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        fail=1
    fi
done

if [[ "${fail}" -ne 0 ]]; then
    echo "One or more workers failed. Check logs in ${LOG_DIR}." >&2
    exit 1
fi

echo "All ${NUM_GPUS} workers finished. Results in ${SAVE_DIR}"
