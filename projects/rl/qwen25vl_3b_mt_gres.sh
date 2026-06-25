#!/bin/bash

set -x

export WANDB_API_KEY="5af7c29cbce6c69b564f36557a148e3e40979477"
export WANDB_MODE=offline
export WANDB_DIR='./verl_wandb_logs'

MODEL_PATH=Qwen/qwen25vl_3b_mt_cold_start  # replace it with your local file path

python3 -m verl.trainer.main \
    config=projects/rl/config.yaml \
    data.train_files=./zhouyik/gres8k_train.parquet \
    data.val_files=./zhouyik/gres8k_train.parquet \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen25vl_3b_gres \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_limit=3