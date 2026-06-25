#!/bin/bash
# CycleGRPO — image-level RL training (Qwen3-VL-4B + mask-token / SAM2).
# Caption<->grounding cycle-consistency reward. See README for data / cold-start
# checkpoint setup. Replace the <PATH_TO_*> placeholders before running.

set -x
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE=offline
export WANDB_DIR='./verl_wandb_logs'

export MODELSCOPE_CACHE='./modelscope_cache/shared'
export HF_DATASETS_CACHE='./hf_dataset_cache'

# Cold-start (co-SFT) checkpoint that RL starts from. See README for how to obtain it.
MODEL_PATH="<PATH_TO_COLD_START_CKPT>"

python3 -m verl.trainer.main \
    config=projects/rl/config.yaml \
    data.train_files="['<PATH_TO_DATA>/denseworld_train.parquet', '<PATH_TO_DATA>/gres_no_target_train.parquet']" \
    data.val_files="['<PATH_TO_DATA>/val.parquet']" \
    data.format_prompt=./projects/rl/format_prompt/non_thinking.jinja \
    data.region_format=mask_token \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
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
    worker.actor.global_batch_size=128

# ---- Optional tuning for large multi-image samples / OOM (append as needed) ----
# Long multi-image prompts make the actor backward memory-heavy. If you hit OOM,
# add some of these (see README "Memory tuning"):
#    data.max_prompt_length=24576 \
#    worker.rollout.max_num_batched_tokens=32768 \
#    data.mini_rollout_batch_size=16 \
#    worker.actor.micro_batch_size_per_device_for_experience=1 \
#    worker.actor.micro_batch_size_per_device_for_update=1 \
#    trainer.max_try_make_batch=64 \
