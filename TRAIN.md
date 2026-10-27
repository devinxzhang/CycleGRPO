# Spatial CycleGRPO training

This repository releases the spatial CycleGRPO training path used for region
captioning and referring localization. Commands below assume one node with
eight CUDA GPUs.

## 1. Install

```bash
git clone https://github.com/devinxzhang/CycleGRPO.git
cd CycleGRPO
uv sync && source .venv/bin/activate
```

This installs the exact versions recorded by the training runs (see the
README "Setup" section; `requirements.txt` is the pip equivalent).

## 2. Download the co-SFT initialization

The public co-SFT snapshot contains the Qwen3-VL-4B model, SAMTok tokenizer,
and SAM2 weights required by the cycle reward:

```bash
hf download zhouyik/Qwen3-VL-4B-SAMTok-co \
  --local-dir checkpoints/Qwen3-VL-4B-SAMTok-co
```

## 3. Prepare RL data

### Released data (recommended)

[XinNUS/CycleGRPO-RL-Data](https://huggingface.co/datasets/XinNUS/CycleGRPO-RL-Data)
contains the exact parquets and zoom-in crops used to train
[CycleGRPO-4B](https://huggingface.co/XinNUS/CycleGRPO-4B). V3Det and COCO
images are not redistributed; download them from the official sources and
let `prepare_images.py` link the ones that are used:

```bash
hf download XinNUS/CycleGRPO-RL-Data --repo-type dataset --local-dir data/CycleGRPO-RL-Data
cd data/CycleGRPO-RL-Data
tar -xf images/crops.tar
python prepare_images.py --v3det_images /path/to/V3Det/images --coco_train2014 /path/to/coco/train2014
cd -
```

Then train as in section 5. Sections 3.1-3.3 describe how to build
equivalent data from your own annotations.

### Building your own data

By default the launcher reads two parquet files from `DATA_DIR`:

```text
data/cyclegrpo/
  denseworld_train.parquet       # region-caption cycle samples
  gres_no_target_train.parquet   # "No target." grounding samples
```

No validation parquet is needed: cycle-aware validation is not implemented
for the spatial runtime, so the launcher trains with validation disabled.
Image paths are stored as given in the input JSON and resolved relative to the
directory you launch from (the repo root) unless they are absolute.

### 3.1 DenseWorld cycle samples

`prepare_dw_rl_dataset.py --input_json` expects a JSON list in the SAMTok
conversation format. Only the image and the region tokens are used; the
caption text in the answer is ignored, so the RL stage needs no reference
captions. Each region is a SAMTok 256x2 token,
`<|mt_start|><|mt_AAAA|><|mt_BBBB|><|mt_end|>` with `0 <= AAAA < 256 <= BBBB < 512`,
either inline in the second turn or as `mask_2d` fields in a fenced `json` block:

```json
[
  {
    "image": "./data/V3Det/images/x00000001/000000004.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nPlease describe the contents of the image. ..."},
      {"from": "gpt", "value": "... a white ceramic plate <|mt_start|><|mt_0087|><|mt_0290|><|mt_end|> on a checkered cloth <|mt_start|><|mt_0186|><|mt_0400|><|mt_end|> ..."}
    ]
  }
]
```

The paper run used SAMTok's DenseWorld mask-generation annotations (872,485
images from Objects365, V3Det and SA-1B). The converter needs a CUDA GPU: it
decodes every region with VQ-SAM2, drops regions whose mask is empty, and for
regions whose box covers less than 30% of the image writes a zoom-in crop to
`--crop_dir` and appends it as an extra `<image>`. Crops are kept in memory
until the end of a run, so convert large inputs in chunks with `--start` and
`--end` and concatenate the parquets.

```bash
python projects/rl/datasets/prepare_dw_rl_dataset.py \
  --input_json data/denseworld_mask_generation.json \
  --vq_sam2_path checkpoints/Qwen3-VL-4B-SAMTok-co/mask_tokenizer_256x2.pth \
  --sam2_path checkpoints/Qwen3-VL-4B-SAMTok-co/sam2.1_hiera_large.pt \
  --output_path data/cyclegrpo/denseworld_train.parquet \
  --crop_dir data/cyclegrpo/denseworld_crops
```

### 3.2 GRES no-target samples

`prepare_gres_no_target_rl_dataset.py --input_json` expects the same
conversation format, with gRefCOCO no-target expressions on COCO `train2014`
images. The answer must contain `<answer> No target. </answer>` (or the phrase
"No target"); a trailing reasoning-format instruction in the question is
stripped. The converter asserts that every image exists.

```json
[
  {
    "image": "./data/ref_seg/grefs/coco2014/train2014/COCO_train2014_000000098304.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nCould you provide a segmentation mask for the man with remote in this image?"},
      {"from": "gpt", "value": "<think> ... </think><answer> No target. </answer>"}
    ]
  }
]
```

```bash
python projects/rl/datasets/prepare_gres_no_target_rl_dataset.py \
  --input_json data/gres_no_target.json \
  --output_path data/cyclegrpo/gres_no_target_train.parquet
```

### 3.3 Check the parquets

Both converters write the same columns:

| Column | Content |
|---|---|
| `images` | list of image paths; the first is the full image, the rest are zoom-in crops |
| `cap_problem` | captioning prompt with one `<image>` per entry in `images` and the region tokens |
| `seg_answer` | `<answer>` + the target region tokens, or `<answer>No target.</answer>` |
| `source` | `denseworld_multiple` or `gres_no_target` |
| `cap_answer`, `seg_problem`, `masks` | unused, `null` |

Validate the schema and the first rows (image paths, `<image>` counts, token
ranges) without loading the whole file. The launcher runs the same check:

```bash
python projects/rl/datasets/check_rl_parquet.py \
  data/cyclegrpo/denseworld_train.parquet data/cyclegrpo/gres_no_target_train.parquet
```

## 4. Launch

```bash
MODEL_PATH=checkpoints/Qwen3-VL-4B-SAMTok-co \
DATA_DIR=data/cyclegrpo \
WANDB_MODE=offline \
bash projects/rl/qwen3vl_4b_mt.sh
```

Optional overrides (`TRAIN_FILES` is a comma-separated parquet list that replaces
the two `DATA_DIR` files; `IMAGE_DIR` is the base for relative image paths):

```bash
TRAIN_FILES=/data/a.parquet,/data/b.parquet \
IMAGE_DIR=/path/to/images \
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

## 5. Reproduce CycleGRPO-4B

The released checkpoint came from three consecutive runs on the released data
(full lineage and settings in the
[dataset card](https://huggingface.co/datasets/XinNUS/CycleGRPO-RL-Data)). It
starts from `zhouyik/Qwen3-VL-4B-SAMTok`, not from the co-SFT snapshot of
section 2. Each stage after the first starts from the previous stage's last
checkpoint, merged to Hugging Face format:

```bash
D=data/CycleGRPO-RL-Data
SAMTOK=checkpoints/Qwen3-VL-4B-SAMTok
hf download zhouyik/Qwen3-VL-4B-SAMTok --local-dir $SAMTOK
export IMAGE_DIR=$D MASK_TOKENIZER_PATH=$SAMTOK/mask_tokenizer_256x2.pth SAM2_PATH=$SAMTOK/sam2.1_hiera_large.pt
CKPT=checkpoints/mask_tokenizer_grpo
MERGE="python -m generalized.verl.utils.checkpoint.fsdp_hf_merger"

# Stage 1: 5 epochs over 4,941 multi-region samples = 190 steps, 1 node x 8 GPUs.
MODEL_PATH=$SAMTOK TRAIN_FILES=$D/data/denseworld_multi_4941.parquet \
  bash projects/rl/qwen3vl_4b_mt.sh trainer.experiment_name=stage1 \
  worker.rollout.n=6 trainer.total_epochs=5
$MERGE $CKPT/stage1/global_step_190/actor $CKPT/stage1/ckpt_190

# Stage 2: 40 steps, 2 nodes x 8 GPUs (start a 2-node Ray cluster first).
MODEL_PATH=$CKPT/stage1/ckpt_190 \
TRAIN_FILES=$D/data/denseworld_multi_4941.parquet,$D/data/denseworld_multi_4920.parquet,$D/data/gres_no_target_1000.parquet \
  bash projects/rl/qwen3vl_4b_mt.sh trainer.experiment_name=stage2 trainer.nnodes=2 \
  trainer.max_steps=40 data.max_pixels=262144 data.mini_rollout_batch_size=16 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.ref.micro_batch_size_per_device_for_experience=1 trainer.max_try_make_batch=64
$MERGE $CKPT/stage2/global_step_40/actor $CKPT/stage2/ckpt_40

# Stage 3: the release is the checkpoint after step 1 of a 30-step schedule.
MODEL_PATH=$CKPT/stage2/ckpt_40 \
TRAIN_FILES=$D/data/denseworld_single_21219.parquet,$D/data/gres_no_target_14665.parquet \
  bash projects/rl/qwen3vl_4b_mt.sh trainer.experiment_name=stage3 trainer.nnodes=2 \
  trainer.max_steps=30 trainer.save_freq=1 worker.actor.optim.lr=2e-6 data.seed=12 \
  data.mini_rollout_batch_size=16 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.ref.micro_batch_size_per_device_for_experience=1 trainer.max_try_make_batch=64
$MERGE $CKPT/stage3/global_step_1/actor $CKPT/stage3/ckpt_1
```

The merger writes the weights in the dtype stored by FSDP (fp32), about twice
the size of the bf16 release. RL sampling is stochastic, so a rerun matches the
release in setup, not bit for bit.
