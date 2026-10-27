# Spatial CycleGRPO training

This repository releases the spatial CycleGRPO training path used for region
captioning and referring localization. Commands below assume one node with
eight CUDA GPUs.

## 1. Install

```bash
git clone https://github.com/devinxzhang/CycleGRPO.git
cd CycleGRPO
pip install -r requirements.txt
pip install -e .
```

## 2. Download the co-SFT initialization

The public co-SFT snapshot contains the Qwen3-VL-4B model, SAMTok tokenizer,
and SAM2 weights required by the cycle reward:

```bash
hf download zhouyik/Qwen3-VL-4B-SAMTok-co \
  --local-dir checkpoints/Qwen3-VL-4B-SAMTok-co
```

## 3. Prepare RL data

Use the converters in `projects/rl/datasets/` to build the DenseWorld cycle
data and GRES no-target data:

```bash
python projects/rl/datasets/prepare_dw_rl_dataset.py \
  --input_json /path/to/denseworld_mask_annotations.json \
  --vq_sam2_path checkpoints/Qwen3-VL-4B-SAMTok-co/mask_tokenizer_256x2.pth \
  --sam2_path checkpoints/Qwen3-VL-4B-SAMTok-co/sam2.1_hiera_large.pt \
  --output_path data/cyclegrpo/denseworld_train.parquet \
  --crop_dir data/cyclegrpo/denseworld_crops

python projects/rl/datasets/prepare_gres_no_target_rl_dataset.py \
  --input_json /path/to/gres_no_target_annotations.json \
  --output_path data/cyclegrpo/gres_no_target_train.parquet
```

The launcher expects:

```text
data/cyclegrpo/
  denseworld_train.parquet
  gres_no_target_train.parquet
```

No validation parquet is needed: cycle-aware validation is not implemented
for the spatial runtime, so the launcher trains with validation disabled.

The RL stage consumes region masks but does not require reference captions for
the DenseWorld cycle samples. Dataset images remain in their original public
dataset locations; the parquet files store the paths expected by the loaders.

## 4. Launch

```bash
MODEL_PATH=checkpoints/Qwen3-VL-4B-SAMTok-co \
DATA_DIR=data/cyclegrpo \
WANDB_MODE=offline \
bash projects/rl/qwen3vl_4b_mt.sh
```

Optional overrides:

```bash
MASK_TOKENIZER_PATH=/path/to/mask_tokenizer_256x2.pth \
SAM2_PATH=/path/to/sam2.1_hiera_large.pt \
MODEL_PATH=/path/to/co_sft_checkpoint \
DATA_DIR=/path/to/parquets \
bash projects/rl/qwen3vl_4b_mt.sh
```

Before starting Ray, the launcher checks that the parquet and SAMTok weight
files exist, validates the parquet schema and first rows
(`projects/rl/datasets/check_rl_parquet.py`), and resolves the full config.
Extra arguments are appended as config overrides. To run every check and print
the resolved config without training (no GPU needed):

```bash
DRY_RUN=1 MODEL_PATH=checkpoints/Qwen3-VL-4B-SAMTok-co DATA_DIR=data/cyclegrpo \
  bash projects/rl/qwen3vl_4b_mt.sh data.mini_rollout_batch_size=16
```

Each run saves `resolved_config.yaml` in its checkpoint directory. Resuming
with a `trainer.load_checkpoint_path` that does not exist is an error.
Memory-related overrides are listed at the bottom of
`projects/rl/qwen3vl_4b_mt.sh`.
