#!/bin/bash

set -x
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE=offline
export WANDB_DIR='./verl_wandb_logs'

export MODELSCOPE_CACHE='./modelscope_cache/shared'
export HF_DATASETS_CACHE='./hf_dataset_cache'

# MODEL_PATH=../highlight_project/highlight_tools/checkpoints/Qwen3-VL-4B-Instruct  # replace it with your local file path
# MODEL_PATH="<PATH_TO_DATA>"/4b_cold_start_sft_46k_multi_32k_timelens_1epoch_grpo_format_tiou_cacc_caption_length_multi_10k/global_step_154
# MODEL_PATH=../pretrained/Qwen3.5-4B  # replace it with your local file path
# MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct  # replace it with your local file path
MODEL_PATH=Qwen/Qwen3-VL-4B-SAMTok  # replace it with your local file path

# ./rl_dataset/denseworld_5k_img_21219_single_target_samples_train.parquet
# ./rl_dataset/tg_multi_merged_train_rl.parquet
# ./rl_dataset/tg_multi_merged_test_rl.parquet
# ./rl_dataset/tg_single_merged_train_rl.parquet
# ./rl_dataset/tg_single_merged_test_rl.parquet

# data.max_pixels=8388608 \

# python3 -m verl.trainer.main \
#     config=projects/rl/config.yaml \
#     data.train_files="['./rl_dataset/tg_single_merged_train_rl.parquet']" \
#     data.val_files="['./rl_dataset/tg_single_merged_test_rl.parquet']" \
#     data.format_prompt=./projects/rl/format_prompt/non_thinking.jinja \
#     data.region_format=mask_token \
#     data.video_fps=0.25 \
#     data.max_pixels=262144 \
#     data.max_prompt_length=16384 \
#     worker.rollout.max_num_batched_tokens=24576 \
#     data.mini_rollout_batch_size=8 \
#     worker.actor.model.freeze_vision_tower=true \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     worker.actor.optimize_captioner=true \
#     worker.actor.optimize_segmenter=true \
#     worker.rollout.n=4 \
#     worker.actor.micro_batch_size_per_device_for_update=1 \
#     worker.actor.micro_batch_size_per_device_for_experience=1 \
#     worker.reward.reward_function=./projects/rl/reward_function/text2mask.py:compute_score \
#     trainer.experiment_name=cyclegrpo_video_debug \
#     trainer.total_epochs=1 \
#     trainer.val_freq=-1 \
#     trainer.val_before_train=false \
#     trainer.save_limit=20 \
#     trainer.logger=["file","wandb"] \
#     trainer.save_limit=15 \
#     trainer.n_gpus_per_node=2 \
#     trainer.max_try_make_batch=64 \
#     data.rollout_batch_size=8 \
#     worker.actor.global_batch_size=8 \

MODEL_PATH=../pretrained/gemma-4-E2B-it  # replace it with your local file path
# MODEL_PATH=../pretrained/Qwen3.5-4B  # replace it with your local file path
# MODEL_PATH=../pretrained/Qwen3-VL-4B-Instruct  # replace it with your local file path
# MODEL_PATH=../pretrained/Qwen2.5-VL-3B-Instruct

python3 -m verl.trainer.main \
    config=projects/rl/config.yaml \
    data.train_files="['./rl_dataset/denseworld_5k_img_4856_samples_train_bbox.parquet', './rl_dataset/denseworld_5k_img_4920_samples_train_bbox.parquet', './rl_dataset/denseworld_5k_img_4938_samples_train_bbox.parquet', './rl_dataset/denseworld_5k_img_4941_samples_train_bbox.parquet']" \
    data.val_files="['./rl_dataset/denseworld_5k_img_4856_samples_train_bbox.parquet']" \
    data.format_prompt=./projects/rl/format_prompt/non_thinking.jinja \
    data.region_format=bbox \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.optimize_captioner=true \
    worker.actor.optimize_segmenter=true \
    worker.rollout.n=8 \
    trainer.experiment_name=gemma_4_e2b_it_bbox_dw0311 \
    trainer.total_epochs=1 \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_limit=10 \
    trainer.logger=["file","wandb"] \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    data.rollout_batch_size=8 \
    worker.actor.global_batch_size=8 \