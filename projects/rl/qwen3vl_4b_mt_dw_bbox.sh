#!/bin/bash

set -x
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE=offline
export WANDB_DIR='./verl_wandb_logs'

export MODELSCOPE_CACHE='./modelscope_cache/shared'
export HF_DATASETS_CACHE='./hf_dataset_cache'

# MODEL_PATH=../pretrained/Qwen3.5-4B  # replace it with your local file path
# MODEL_PATH=../pretrained/Qwen3-VL-8B-Instruct  # replace it with your local file path
MODEL_PATH=../pretrained/Qwen2.5-VL-3B-Instruct
      

python3 -m verl.trainer.main \
    config=projects/rl/config.yaml \
    data.train_files="['./rl_dataset/denseworld_5k_img_4856_samples_train_bbox.parquet','./rl_dataset/denseworld_5k_img_4920_samples_train_bbox.parquet','./rl_dataset/denseworld_5k_img_4938_samples_train_bbox.parquet','./rl_dataset/denseworld_5k_img_4941_samples_train_bbox.parquet']" \
    data.val_files="['./rl_dataset/denseworld_5k_img_4856_samples_train_bbox.parquet']" \
    data.format_prompt=./projects/rl/format_prompt/non_thinking.jinja \
    data.region_format=bbox \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.optimize_captioner=true \
    worker.actor.optimize_segmenter=true \
    worker.rollout.n=8 \
    trainer.experiment_name=qwen2.5_3b_bbox_dw0311_v4 \
    trainer.total_epochs=1 \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_limit=10 \
    trainer.logger=["file","wandb"] \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    data.rollout_batch_size=128 \
    worker.actor.global_batch_size=128 \