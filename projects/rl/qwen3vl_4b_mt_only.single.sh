#!/bin/bash

set -x
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE=offline
export WANDB_DIR='./verl_wandb_logs'

export MODELSCOPE_CACHE='./modelscope_cache/shared'
export HF_DATASETS_CACHE='./hf_dataset_cache'

# MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct  # replace it with your local file path
MODEL_PATH=Qwen/Qwen3-VL-4B-SAMTok  # replace it with your local file path
# MODEL_PATH=checkpoints/mask_tokenizer_grpo/qwen3vl_4b_dw_multiple_all/global_step_190/actor/huggingface  # replace it with your local file path

    # data.train_files="['./rl_dataset/denseworld_5k_img_21219_single_target_samples_train.parquet', './rl_dataset/denseworld_5k_img_22872_single_target_samples_train.parquet', './rl_dataset/denseworld_5k_img_23105_single_target_samples_train.parquet', './rl_dataset/denseworld_5k_img_24794_single_target_samples_train.parquet', './rl_dataset/gres_no_target_14665_samples_train.parquet', './rl_dataset/gres_no_target_14665_samples_train.parquet']" \
    # data.val_files="['./rl_dataset/denseworld_5k_img_4856_samples_train.parquet']" \

python3 -m verl.trainer.main \
    config=projects/rl/config.yaml \
    data.train_files="['./rl_dataset/denseworld_5k_img_21219_single_target_samples_train.parquet', './rl_dataset/denseworld_5k_img_22872_single_target_samples_train.parquet', './rl_dataset/denseworld_5k_img_23105_single_target_samples_train.parquet', './rl_dataset/denseworld_5k_img_24794_single_target_samples_train.parquet']" \
    data.val_files="['./rl_dataset/denseworld_5k_img_4856_samples_train.parquet']" \
    data.format_prompt=./projects/rl/format_prompt/non_thinking.jinja \
    data.region_format=mask_token \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.optimize_captioner=true \
    worker.actor.optimize_segmenter=true \
    worker.rollout.n=8 \
    trainer.experiment_name=cyclegrpo_4b_single.only \
    trainer.total_epochs=1 \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_limit=20 \
    trainer.logger=["file","wandb"] \
    trainer.save_limit=15 \
    trainer.n_gpus_per_node=8 \
    data.rollout_batch_size=128 \
    worker.actor.global_batch_size=128 \