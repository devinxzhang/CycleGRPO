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
holds the 19,655 DenseWorld cycle samples, the 1,000 gRefCOCO no-target
samples and the zoom-in crops they reference. V3Det and COCO images are not
redistributed; download them from the official sources and let
`prepare_images.py` link the ones that are used:

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

A row may hold several target regions. The data loader draws one of them at
random on every pass, together with that target's zoom-in crop, so a row with
many regions contributes a different single-target sample each epoch.

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

## 5. Train the paper configuration

One epoch over the released data, starting from `zhouyik/Qwen3-VL-4B-SAMTok`
rather than from the co-SFT snapshot of section 2, with the caption group size
and the grounding rollout count both set to 6:

```bash
D=data/CycleGRPO-RL-Data
SAMTOK=checkpoints/Qwen3-VL-4B-SAMTok
hf download zhouyik/Qwen3-VL-4B-SAMTok --local-dir $SAMTOK

MODEL_PATH=$SAMTOK IMAGE_DIR=$D \
MASK_TOKENIZER_PATH=$SAMTOK/mask_tokenizer_256x2.pth SAM2_PATH=$SAMTOK/sam2.1_hiera_large.pt \
TRAIN_FILES=$D/data/denseworld_multi_4856.parquet,$D/data/denseworld_multi_4920.parquet,$D/data/denseworld_multi_4938.parquet,$D/data/denseworld_multi_4941.parquet,$D/data/gres_no_target_1000.parquet \
  bash projects/rl/qwen3vl_4b_mt.sh worker.rollout.n=6
```

Merge the resulting FSDP checkpoint to Hugging Face format with

```bash
python -m generalized.verl.utils.checkpoint.fsdp_hf_merger \
  checkpoints/mask_tokenizer_grpo/<experiment>/global_step_<n>/actor out_dir
```

which writes the weights in the dtype stored by FSDP (fp32), about twice the
size of a bf16 release. RL sampling is stochastic, so a rerun matches the
published setup, not bit for bit.
