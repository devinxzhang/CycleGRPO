#!/bin/bash
# CycleGRPO — image-level RL training (Qwen3-VL-4B + mask-token / SAM2).
# Caption<->grounding cycle-consistency reward. See README for data / cold-start
# checkpoint setup. Paths are configured through environment variables.

set -euo pipefail
set -x
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-./verl_wandb_logs}"

export MODELSCOPE_CACHE='./modelscope_cache/shared'
export HF_DATASETS_CACHE='./hf_dataset_cache'

# Download zhouyik/Qwen3-VL-4B-SAMTok-co locally first (see README/TRAIN.md).
# That snapshot contains the co-SFT model and both SAMTok weight files.
MODEL_PATH="${MODEL_PATH:-./checkpoints/Qwen3-VL-4B-SAMTok-co}"
DATA_DIR="${DATA_DIR:-./data/cyclegrpo}"
MASK_TOKENIZER_PATH="${MASK_TOKENIZER_PATH:-${MODEL_PATH}/mask_tokenizer_256x2.pth}"
SAM2_PATH="${SAM2_PATH:-${MODEL_PATH}/sam2.1_hiera_large.pt}"

for path in \
    "${DATA_DIR}/denseworld_train.parquet" \
    "${DATA_DIR}/gres_no_target_train.parquet" \
    "${MASK_TOKENIZER_PATH}" \
    "${SAM2_PATH}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        echo "See TRAIN.md for checkpoint and dataset setup." >&2
        exit 1
    fi
done

# Validate parquet schema and the first rows (image paths, mask tokens).
python3 projects/rl/datasets/check_rl_parquet.py \
    "${DATA_DIR}/denseworld_train.parquet" "${DATA_DIR}/gres_no_target_train.parquet"

# DRY_RUN=1 resolves the full config (including appended overrides), prints it, and exits before Ray.
ENTRYPOINT=verl.trainer.main
[[ "${DRY_RUN:-0}" == 1 ]] && ENTRYPOINT=verl.trainer.resolve_config
python3 -m "${ENTRYPOINT}" \
    config=projects/rl/config.yaml \
    data.train_files="['${DATA_DIR}/denseworld_train.parquet', '${DATA_DIR}/gres_no_target_train.parquet']" \
    data.val_files=[] \
    data.format_prompt=./projects/rl/format_prompt/non_thinking.jinja \
    data.region_format=mask_token \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.mask_tokenizer_path=${MASK_TOKENIZER_PATH} \
    worker.reward.sam2_pretrained_weight=${SAM2_PATH} \
    worker.actor.optimize_captioner=true \
    worker.actor.optimize_segmenter=true \
    worker.rollout.n=8 \
    trainer.experiment_name=cyclegrpo_qwen3vl_4b \
    trainer.total_epochs=1 \
    trainer.val_freq=-1 \
    trainer.save_freq=5 \
    trainer.val_before_train=false \
    trainer.save_limit=20 \
    trainer.logger=["file","wandb"] \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.rollout_batch_size=128 \
    worker.actor.global_batch_size=128 \
    "$@"

# ---- Optional tuning for large multi-image samples / OOM ----
# Extra arguments to this script are appended as config overrides, e.g.
#   bash projects/rl/qwen3vl_4b_mt.sh worker.actor.micro_batch_size_per_device_for_update=1
# Long multi-image prompts make the actor backward memory-heavy. If you hit OOM,
# pass some of these (see README "Memory tuning"):
#    data.max_prompt_length=24576 \
#    worker.rollout.max_num_batched_tokens=32768 \
#    data.mini_rollout_batch_size=16 \
#    worker.actor.micro_batch_size_per_device_for_experience=1 \
#    worker.actor.micro_batch_size_per_device_for_update=1 \
#    trainer.max_try_make_batch=64 \
