#!/bin/bash

set -x
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE=offline
export WANDB_DIR='./verl_wandb_logs'

export MODELSCOPE_CACHE='./modelscope_cache/shared'
export HF_DATASETS_CACHE='./hf_dataset_cache'

MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=projects/rl/config.yaml \
    data.train_files="['rl_dataset/denseworld_10k_img_45977_samples_train_bbox.parquet']" \
    data.val_files="['rl_dataset/denseworld_10k_img_45977_samples_train_bbox.parquet']" \
    data.format_prompt=./projects/rl/format_prompt/non_thinking.jinja \
    data.region_format=bbox \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.optimize_captioner=true \
    worker.actor.optimize_segmenter=true \
    worker.rollout.n=6 \
    trainer.experiment_name=qwen3vl_4b_bbox_10k_dw0311 \
    trainer.total_epochs=1 \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_limit=5 \
    trainer.logger=["file","wandb"] \
    trainer.n_gpus_per_node=2 \
    data.rollout_batch_size=4 \
    worker.actor.global_batch_size=4 \