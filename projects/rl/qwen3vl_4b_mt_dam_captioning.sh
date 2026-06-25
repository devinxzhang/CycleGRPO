#!/bin/bash

set -x
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE=offline
export WANDB_DIR='./verl_wandb_logs'

export MODELSCOPE_CACHE='./modelscope_cache/shared'
export HF_DATASETS_CACHE='./hf_dataset_cache'

MODEL_PATH=Qwen/Qwen3-VL-4B-SAMTok  # replace it with your local file path
# MODEL_PATH=Qwen/Qwen2.5-VL-7B-SAMTok-co

export LLM_AS_A_JUDGE_BASES="http://<LLM_JUDGE_HOST>:<PORT>/v1"
export LLM_AS_A_JUDGE_KEY="EMPTY"
export LLM_AS_A_JUDGE_MODEL="qwen-judge"

python3 -m verl.trainer.main \
    config=projects/rl/config.yaml \
    data.train_files="['./rl_dataset/dam_1k_samples_train_captioning.parquet','./rl_dataset/dam_1k_samples_train_grounding.parquet']" \
    data.val_files="['./rl_dataset/dam_1k_samples_train_captioning.parquet']" \
    data.format_prompt=./projects/rl/format_prompt/non_thinking.jinja \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.optimize_captioner=true \
    worker.actor.optimize_segmenter=true \
    worker.rollout.n=8 \
    worker.rollout.temperature=1 \
    trainer.experiment_name=qwen3vl_4b_mt_dam_captioning_grounding \
    trainer.total_epochs=10 \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_limit=10 \
    trainer.logger=["file","wandb"] \
    trainer.n_gpus_per_node=8 \
    data.rollout_batch_size=128 \
    worker.actor.global_batch_size=128