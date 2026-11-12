<div align="center">

# CycleGRPO

**The actor is its own critic: closing the caption ↔ grounding loop for fine-grained multimodal understanding.**

Official implementation, checkpoints and training data for CycleGRPO (ECCV 2026)
and its extension to video, Generalized CycleGRPO.

<a href="https://arxiv.org/abs/2607.11581"><img src="https://img.shields.io/badge/CycleGRPO-arXiv%202607.11581-b31b1b?logo=arxiv&logoColor=white" alt="CycleGRPO arXiv"></a>
<a href="https://arxiv.org/abs/2610.29552"><img src="https://img.shields.io/badge/Generalized-arXiv%202610.29552-b31b1b?logo=arxiv&logoColor=white" alt="Generalized CycleGRPO arXiv"></a>
<a href="https://devinxzhang.github.io/CycleGRPO-Page/"><img src="https://img.shields.io/badge/Project-Pages-blue?logo=googlechrome&logoColor=white" alt="Project pages"></a>
<a href="https://huggingface.co/XinNUS"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models%20%26%20Data-FFD21E" alt="Hugging Face"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"></a>

</div>

---

## What is CycleGRPO?

CycleGRPO improves region captioning and referring localization **jointly**, with
no caption ground truth during RL.

The policy is a Qwen3-VL-4B model that emits **mask tokens**, decoded by a VQ-SAM2
mask tokenizer. To score a caption, the reward runs an **inner grounding rollout**
conditioned on the model's own caption, then measures mask IoU against the region
that was captioned. A caption only scores well if it is specific enough to find the
region again — so captions are optimized to be **distinctive and locatable**, and
the actor supplies its own critic signal.

**Generalized CycleGRPO** carries the same region → text → region cycle into video,
over temporal intervals and timestamped spatial evidence.

## Releases

| | Paper | Venue |
|---|---|---|
| **CycleGRPO** — images, spatial | [Actor as Its Own Critic: Unifying Region Understanding and Localization via CycleGRPO](https://arxiv.org/abs/2607.11581) | ECCV 2026 |
| **Generalized CycleGRPO** — images + video | [Generalized CycleGRPO for Fine-Grained Multimodal Understanding and Grounding](https://arxiv.org/abs/2610.29552) | arXiv preprint |

## Model and data zoo

**Models**

| Model | Task | Link |
|---|---|---|
| `CycleGRPO-4B` | Spatial: region captioning + localization | [🤗](https://huggingface.co/XinNUS/CycleGRPO-4B) |
| `Generalized-CycleGRPO-Temporal-4B` | Temporal (Qwen route): captioning + grounding | [🤗](https://huggingface.co/XinNUS/Generalized-CycleGRPO-Temporal-4B) |
| `Generalized-CycleGRPO-Temporal-OMTG-4B` | Temporal (OMTG route): grounding | [🤗](https://huggingface.co/XinNUS/Generalized-CycleGRPO-Temporal-OMTG-4B) |
| `Generalized-CycleGRPO-SpatioTemporal-4B` | Spatio-temporal (Open-o3 route) | [🤗](https://huggingface.co/XinNUS/Generalized-CycleGRPO-SpatioTemporal-4B) |

These are final inference models. RL training starts from the co-SFT
initialization below, not from these.

**Data and initialization**

| Resource | Contents | Link |
|---|---|---|
| `Qwen3-VL-4B-SAMTok-co` | Co-SFT init that spatial RL starts from | [🤗](https://huggingface.co/zhouyik/Qwen3-VL-4B-SAMTok-co) |
| `CycleGRPO-RL-Data` | 19,655 DenseWorld region + 1,000 GRES no-target samples, with zoom-in crops | [🤗](https://huggingface.co/datasets/XinNUS/CycleGRPO-RL-Data) |
| Generalized annotations v1 | 73,364 RL rows across 3 profiles + 21,443 co-SFT conversations | [Release](https://github.com/devinxzhang/CycleGRPO/releases/tag/generalized-data-v1) |
| `DOVE-Bench` | Evaluation benchmark | [🤗](https://huggingface.co/datasets/XinNUS/DOVE-Bench) |

## Contents

- [Installation](#installation)
- [Spatial CycleGRPO](#spatial-cyclegrpo-images) — [training](#training) · [evaluation](#evaluation) · [results](#results)
- [Generalized CycleGRPO](#generalized-cyclegrpo-video) — [data](#training-annotations) · [training](#training-1)
- [Repository layout](#repository-layout)
- [Citation](#citation)

## Installation

One environment serves both releases, pinned to the stack every released run
used: Python 3.11, torch 2.8.0 (CUDA 12.8), vLLM 0.11.0, transformers 4.57.1,
flash-attn 2.8.3, on Linux x86_64.

```bash
uv sync                    # creates .venv from uv.lock, including the flash-attn wheel
source .venv/bin/activate
```

Without uv, install the same pins into a Python 3.11 environment:

```bash
pip install -r requirements.txt && pip install -e . --no-deps
```

`requirements.txt` is exported from `uv.lock`.

---

## Spatial CycleGRPO (images)

### Training

Download the co-SFT initialization and the released RL data, then launch.
[`TRAIN.md`](TRAIN.md) covers the data layout, the paper configuration, and how
to build your own parquets.

```bash
# 1. Initialization checkpoint
hf download zhouyik/Qwen3-VL-4B-SAMTok-co \
  --local-dir checkpoints/Qwen3-VL-4B-SAMTok-co

# 2. RL data, plus the images it references
D=data/CycleGRPO-RL-Data
hf download XinNUS/CycleGRPO-RL-Data --repo-type dataset --local-dir $D
(cd $D && tar -xf images/crops.tar && python prepare_images.py \
  --v3det_images /path/to/V3Det/images --coco_train2014 /path/to/coco/train2014)

# 3. Train (defaults assume 1 node x 8 GPUs)
MODEL_PATH=checkpoints/Qwen3-VL-4B-SAMTok-co IMAGE_DIR=$D \
TRAIN_FILES=$D/data/denseworld_multi_4856.parquet,$D/data/denseworld_multi_4920.parquet,$D/data/denseworld_multi_4938.parquet,$D/data/denseworld_multi_4941.parquet,$D/data/gres_no_target_1000.parquet \
bash projects/rl/qwen3vl_4b_mt.sh
```

The launcher reads all paths from environment variables and verifies every
required parquet and SAMTok weight file before starting. `WANDB_API_KEY` is
optional; `WANDB_MODE` defaults to `offline`.

To build your own data instead, see the scripts in `projects/rl/datasets/`
(e.g. `prepare_dw_rl_dataset.py`, `prepare_gres_no_target_rl_dataset.py`). A
typical mix combines DenseWorld region samples with GRES no-target samples.

<details>
<summary><b>Memory tuning for multi-image samples / OOM</b></summary>

Multi-image samples produce long prompts, and the actor backward pass can run
out of memory. Pass any of these as extra arguments to the launcher, e.g.
`bash projects/rl/qwen3vl_4b_mt.sh data.mini_rollout_batch_size=16`:

| Knob | Effect |
|---|---|
| `data.max_prompt_length`, `worker.rollout.max_num_batched_tokens` | Raise to fit long prompts (costs memory) |
| `worker.actor.micro_batch_size_per_device_for_{experience,update}=1` | Smallest micro-batch |
| `data.mini_rollout_batch_size=16` | Smaller vLLM generation batch; lowers rollout-phase memory |
| `worker.rollout.n` | Use a multiple of `world_size` (`nnodes x n_gpus`) when mixing cycle and no-target sources, so per-source sub-batches divide evenly across ranks |

</details>

### Evaluation

Each benchmark lives in `evaluation/<benchmark>/`. The multi-GPU launchers
validate every input path before loading a model, shard the dataset across GPUs,
and write one JSON result per sample. SAMTok weights default to
`./checkpoints/Qwen3-VL-4B-SAMTok-co/`; override with `VQ_SAM2_PATH` and
`SAM2_PATH`.

Pass `--check_only` to the GRES, GroundingSuite or GCG inference script to
verify paths without a GPU — it reports every missing input at once and exits.

| Benchmark | Data to download | How to point the scripts at it |
|---|---|---|
| GRES | [gRefCOCO](https://github.com/henghuiding/gRefCOCO) `grefs(unc).json` + `instances.json`, and COCO `train2014` | Build the case file once with `python evaluation/gres/qwen3vl_gres_eval.py --prepare_split val --grefs_dir <dir> --dataset data/grefcoco_val.json`, then set `DATASET=data/grefcoco_val.json`, `COCO_ROOT=<dir containing train2014/>` |
| GroundingSuite | [GSEval](https://huggingface.co/datasets/hustvl/GSEval) | `DATASET=<.../GroundingSuite-Eval.jsonl>`; images are read from `unlabeled2017/` next to it (override with `IMAGE_ROOT`) |
| GCG | GranDf `val_test` images from [Sa2VA-Training](https://huggingface.co/datasets/Dense-World/Sa2VA-Training) | `IMAGE_FOLDER=<...>/glamm_data/images/grandf/val_test` |
| GAR-Bench | [Grasp-Any-Region](https://github.com/Haochen-Wang409/Grasp-Any-Region) `evaluation/GAR-Bench/annotations` | `GAR_BENCH_ROOT=<annotations dir>` or `--gar_bench_root` |

```bash
# Referring segmentation (GRES)
DATASET=data/grefcoco_val.json COCO_ROOT=/data/coco \
bash evaluation/gres/run_gres_multigpu.sh 8 <MODEL_PATH> ./results/gres/

# GroundingSuite
DATASET=data/GroundingSuiteEval/GroundingSuite-Eval.jsonl \
bash evaluation/groundingsuite/run_groundingsuite_multigpu.sh 8 <MODEL_PATH> ./results/groundingsuite/

# Grounded caption generation (GCG)
IMAGE_FOLDER=/data/Sa2VA-Training/glamm_data/images/grandf/val_test \
bash evaluation/gcg/run_gcg_multigpu.sh 8 <MODEL_PATH> ./results/gcg/

# GAR-Bench VQA (inference, then metrics)
python evaluation/gar/qwen3vl_gar_vqa_infer.py <MODEL_PATH> \
  --gar_bench_root /data/GAR-Bench/annotations --output results/gar/vqa.json
python evaluation/gar/gar_vqa_metrics.py results/gar/vqa.json

# DLC-Bench (start the judge server in a separate shell, then infer + eval)
bash evaluation/dlc_bench/serve_judge.sh
bash evaluation/dlc_bench/evaluate_dlc.sh <MODEL_PATH> <CACHE_NAME>
python evaluation/dlc_bench/eval_llama_without_image.py \
  --pred evaluation/dlc_bench/model_outputs/<CACHE_NAME>.json --base-url http://localhost:8007/v1
```

Bounding-box generalization variants live in `evaluation/bbox/` and still use
the `<PATH_TO_*>` placeholders inside the scripts.

### Results

Baseline is **[SAMTok](https://huggingface.co/zhouyik/Qwen3-VL-4B-SAMTok)**
(Qwen3-VL-4B). Two CycleGRPO rows are reported: **paper** is the ECCV 2026
numbers, **release** is the public [`XinNUS/CycleGRPO-4B`](https://huggingface.co/XinNUS/CycleGRPO-4B)
checkpoint, which comes from a rerun and scores similarly or slightly higher
overall.

**Region captioning — DLC-Bench** (100 samples)

| Method | Pos. | Neg. | Avg. |
|---|---:|---:|---:|
| SAMTok | 43.5 | 80.4 | 61.9 |
| CycleGRPO (paper) | 51.2 | 84.2 | 67.7 |
| **CycleGRPO (release)** | **52.4** | 83.2 | **67.8** |

**Text-to-mask — GroundingSuite** (gIoU, %)

| Method | Stuff | Part | Multi | Single | All |
|---|---:|---:|---:|---:|---:|
| SAMTok | 80.9 | 12.4 | 62.0 | 52.9 | 57.5 |
| CycleGRPO (paper) | 90.7 | 20.9 | 76.3 | 61.6 | 67.6 |
| **CycleGRPO (release)** | 90.5 | **21.2** | **78.3** | **62.3** | **68.2** |

**Referring segmentation + target rejection — GRES** (%)

| Method | Avg gIoU | Avg cIoU | Avg N-acc |
|---|---:|---:|---:|
| SAMTok | 71.2 | 70.2 | 58.7 |
| CycleGRPO (paper) | 78.2 | 74.1 | 92.1 |
| **CycleGRPO (release)** | **78.7** | **74.4** | **92.5** |

<details>
<summary>Full GRES splits (val / testA / testB)</summary>

| Method | Val gIoU | Val cIoU | Val N-acc | TestA gIoU | TestA cIoU | TestA N-acc | TestB gIoU | TestB cIoU | TestB N-acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAMTok | 71.3 | 69.2 | 61.4 | 75.3 | 75.4 | 59.0 | 66.9 | 66.0 | 55.6 |
| CycleGRPO (paper) | 81.8 | 74.6 | 94.2 | 79.9 | 77.8 | 93.1 | 73.0 | 70.0 | 89.0 |
| CycleGRPO (release) | 82.2 | 74.8 | 94.7 | 80.3 | 78.2 | 93.0 | 73.5 | 70.2 | 89.9 |

</details>

<details>
<summary>Region VQA — GAR-Bench-VQA (%)</summary>

| Method | Overall | Color | Shape | Texture | Material | Position | Non-Entity | Relation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAMTok | 64.2 | 58.0 | 48.4 | 48.3 | 58.3 | 76.6 | 54.1 | 83.2 |
| CycleGRPO (paper) | 65.1 | 62.3 | 50.0 | 48.3 | 61.1 | 73.4 | 57.4 | 82.2 |
| CycleGRPO (release) | 64.9 | 60.9 | 50.0 | 48.3 | 61.1 | 73.4 | 54.1 | 84.2 |

</details>

<details>
<summary>Interleaved text-mask — GCG (METEOR / CIDEr / AP50 / mIoU / Recall)</summary>

| Method | val M | val C | val AP50 | val mIoU | val Rec | test M | test C | test AP50 | test mIoU | test Rec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAMTok | 16.1 | 48.2 | 34.7 | 69.4 | 46.6 | 16.4 | 51.4 | 34.4 | 68.4 | 48.3 |
| CycleGRPO (paper) | 17.2 | 54.7 | 35.9 | 69.6 | 49.6 | 17.1 | 54.0 | 35.2 | 68.6 | 49.7 |
| CycleGRPO (release) | 17.3 | 54.3 | 36.8 | 70.2 | 50.2 | 17.2 | 53.7 | 35.0 | 69.2 | 49.8 |

</details>

---

## Generalized CycleGRPO (video)

Extends the image region–text–region cycle to video intervals and timestamped
spatial evidence. Full setup, media preparation and training instructions:
**[`generalized/README.md`](generalized/README.md)**.

### Training annotations

```bash
python generalized/download_data.py
python data/generalized-cyclegrpo-rl-data/scripts/release_data.py validate
```

The downloader verifies a pinned SHA256 and extracts under
`data/generalized-cyclegrpo-rl-data/`. The package holds **73,364 RL rows**
(18,742 `temporal_qwen`, 18,735 `temporal_omtg`, 35,887 `openo3_mixed`) plus
12,242 OMTG and 9,201 Open-o3 co-SFT conversations.

Images and videos are downloaded separately from the original
[OMTG](https://huggingface.co/datasets/insomnia7/omtg56k) and
[Open-o3-Video](https://huggingface.co/datasets/marinero4972/Open-o3-Video)
sources; the included tools connect that media and generate training parquets.
See [data preparation](generalized/README.md#data-preparation).

For an offline transfer, download the
[archive](https://github.com/devinxzhang/CycleGRPO/releases/download/generalized-data-v1/generalized-cyclegrpo-training-data-20261112.tar.gz)
and run `python generalized/download_data.py --archive /path/to/archive.tar.gz`.

### Training

```bash
bash generalized/train.sh <profile>   # temporal_qwen | temporal_omtg | openo3_mixed
```

Set the corresponding model and data paths first — see
[`generalized/README.md`](generalized/README.md).

---

## Repository layout

```
verl/                       # RL engine (forked EasyR1/veRL): trainer, FSDP workers, vLLM rollout
projects/
  rl/
    qwen3vl_4b_mt.sh        # >>> main image CycleGRPO training entry <<<
    config.yaml             # default RL config (algorithm, rollout, fsdp, reward)
    reward_function/        # text2mask.py: cycle-consistency + mask-IoU reward
    format_prompt/          # prompt templates (non_thinking.jinja)
    datasets/               # scripts that build the RL parquet datasets
  transformers/             # VQ-SAM2 mask tokenizer + SAM2 model code
  vlm/                      # eval helpers: gRefCOCO/refcoco loaders, IoU metrics, GCG mapping
generalized/
  verl/                     # video RL runtime
  train.sh                  # temporal_qwen / temporal_omtg / openo3_mixed
  download_data.py          # versioned Generalized training annotations
  README.md                 # video setup, media preparation and training
evaluation/
  gres/                     # referring segmentation (GRES)
  groundingsuite/           # GroundingSuite grounding
  gcg/                      # grounded caption generation (GCG)
  gar/                      # GAR-Bench VQA / detailed caption
  dlc_bench/                # dense-captioning eval (DLC-Bench)
  bbox/                     # bbox-format generalization variants
tests/                      # CPU regression tests (make check)
TRAIN.md                    # spatial data layout + paper training configuration
README_EasyR1.md            # upstream EasyR1/veRL framework documentation
```

## Citation

For the spatial CycleGRPO model, cite the ECCV paper:

```bibtex
@inproceedings{cyclegrpo2026,
  title     = {Actor as Its Own Critic: Unifying Region Understanding and
               Localization via CycleGRPO},
  author    = {Zhang, Xin and Wang, Haochen and Zhou, Yikang and Wang, Zhuochen
               and Li, Xiangtai and Tan, Robby T.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

For the generalized image and video framework, cite the arXiv preprint:

```bibtex
@misc{zhang2026generalizedcyclegrpo,
  title         = {Generalized CycleGRPO for Fine-Grained Multimodal Understanding and Grounding},
  author        = {Zhang, Xin and Wang, Haochen and Lin, Beibei and Zhang, Zhaoxiang
                   and Tan, Robby T. and Li, Jason and Tao, Dacheng},
  year          = {2026},
  eprint        = {2610.29552},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.29552}
}
```

## Acknowledgements

Built on [EasyR1](https://github.com/hiyouga/EasyR1) and
[veRL](https://github.com/volcengine/verl), with segmentation via
[SAM2](https://github.com/facebookresearch/sam2). Released under
[Apache 2.0](LICENSE); upstream copyright and license notices are retained.
