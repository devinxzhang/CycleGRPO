#!/usr/bin/env bash
# Run in the foreground on the Ray head; all nodes must share this checkout.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
PROFILE="${1:-}"
case "$PROFILE" in
  temporal_qwen)
    DEFAULT_BATCH=128; DEFAULT_FPS=0.25; DEFAULT_PIXELS=262144
    DEFAULT_PROMPT=8192; DEFAULT_RESPONSE=8192; DEFAULT_FORMAT=mask_token
    DEFAULT_DENSE=0; DEFAULT_MINI=32; DEFAULT_SAVE=5
    ;;
  temporal_omtg)
    DEFAULT_BATCH=64; DEFAULT_FPS=2; DEFAULT_PIXELS=786432
    DEFAULT_PROMPT=16384; DEFAULT_RESPONSE=768; DEFAULT_FORMAT=mask_token
    DEFAULT_DENSE=0; DEFAULT_MINI=16; DEFAULT_SAVE=5
    export CYCLEGRPO_VIDEO_MIN_FRAMES="${CYCLEGRPO_VIDEO_MIN_FRAMES:-4}"
    export CYCLEGRPO_VIDEO_MAX_FRAMES="${CYCLEGRPO_VIDEO_MAX_FRAMES:-768}"
    export CYCLEGRPO_VIDEO_TOTAL_PIXELS="${CYCLEGRPO_VIDEO_TOTAL_PIXELS:-8388608}"
    ;;
  openo3_mixed)
    DEFAULT_BATCH=128; DEFAULT_FPS=2; DEFAULT_PIXELS=262144
    DEFAULT_PROMPT=16384; DEFAULT_RESPONSE=768; DEFAULT_FORMAT=bbox
    DEFAULT_DENSE=2; DEFAULT_MINI=128; DEFAULT_SAVE=10
    export CYCLEGRPO_VIDEO_MIN_FRAMES="${CYCLEGRPO_VIDEO_MIN_FRAMES:-4}"
    export CYCLEGRPO_VIDEO_MAX_FRAMES="${CYCLEGRPO_VIDEO_MAX_FRAMES:-16}"
    ;;
  *) echo 'Usage: bash generalized/train.sh {temporal_qwen|temporal_omtg|openo3_mixed} [Hydra overrides...]' >&2; exit 2 ;;
esac
shift
: "${MODEL_PATH:?Set MODEL_PATH to the correct bidirectional co-SFT initialization}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the prepared RL parquet}"
export MODEL_PATH TRAIN_FILE
export CYCLEGRPO_INNER_ROLLOUT_N="${CYCLEGRPO_INNER_ROLLOUT_N:-4}"
export CYCLEGRPO_TEMPORAL_DENSE_REWARD_WEIGHT="${CYCLEGRPO_TEMPORAL_DENSE_REWARD_WEIGHT:-$DEFAULT_DENSE}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export CYCLEGRPO_DECORD_NUM_THREADS="${CYCLEGRPO_DECORD_NUM_THREADS:-2}"
export CYCLEGRPO_VIDEO_DECODE_TIMEOUT_S="${CYCLEGRPO_VIDEO_DECODE_TIMEOUT_S:-60}"
export CYCLEGRPO_INNER_ROLLOUT_TIMEOUT_S="${CYCLEGRPO_INNER_ROLLOUT_TIMEOUT_S:-600}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USE_MODELSCOPE_HUB=0
export NNODES="${NNODES:-1}" GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-$DEFAULT_BATCH}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$DEFAULT_BATCH}"
export MIN_PIXELS="${MIN_PIXELS:-3136}"
if [[ "$PROFILE" == temporal_omtg ]]; then
  export MIN_PIXELS="${OMTG_MIN_PIXELS:-2048}"
fi

python3 -m generalized.preflight --profile "$PROFILE"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "Preflight passed; DRY_RUN=1, not starting training."
  exit 0
fi

EXTRA_ARGS=()
if [[ "$PROFILE" == openo3_mixed ]]; then
  EXTRA_ARGS+=(data.sampling_weight_key=sampling_weight)
fi
if [[ -n "${LOAD_CHECKPOINT_PATH:-}" ]]; then
  EXTRA_ARGS+=(trainer.load_checkpoint_path="$LOAD_CHECKPOINT_PATH")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(trainer.max_steps="$MAX_STEPS")
fi
python3 -m generalized.verl.trainer.main \
  config=generalized/config.yaml \
  data.train_files="['${TRAIN_FILE}']" \
  data.val_files="['${VAL_FILE:-$TRAIN_FILE}']" \
  data.region_format="${REGION_FORMAT:-$DEFAULT_FORMAT}" \
  data.video_fps="${VIDEO_FPS:-$DEFAULT_FPS}" \
  data.min_pixels="$MIN_PIXELS" \
  data.max_pixels="${MAX_PIXELS:-$DEFAULT_PIXELS}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-$DEFAULT_PROMPT}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-$DEFAULT_RESPONSE}" \
  data.rollout_batch_size="$ROLLOUT_BATCH_SIZE" \
  data.mini_rollout_batch_size="${MINI_ROLLOUT_BATCH_SIZE:-$DEFAULT_MINI}" \
  worker.actor.model.model_path="$MODEL_PATH" \
  worker.actor.global_batch_size="$GLOBAL_BATCH_SIZE" \
  worker.rollout.n="$ROLLOUT_N" \
  worker.rollout.max_model_len="${MAX_MODEL_LEN:-$((DEFAULT_PROMPT + DEFAULT_RESPONSE))}" \
  worker.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-32768}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-$PROFILE}" \
  trainer.save_checkpoint_path="${SAVE_CHECKPOINT_PATH:-$ROOT_DIR/checkpoints/$PROFILE}" \
  trainer.save_freq="${SAVE_FREQ:-$DEFAULT_SAVE}" \
  trainer.nnodes="$NNODES" \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  "${EXTRA_ARGS[@]}" "$@"
