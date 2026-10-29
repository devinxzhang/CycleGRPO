# CycleGRPO and Generalized CycleGRPO

<p align="center">
  <a href="https://devinxzhang.github.io/CycleGRPO-Page/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/pdf/2607.11581"><img src="https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Paper"></a>
  <a href="https://arxiv.org/abs/2607.11581"><img src="https://img.shields.io/badge/arXiv-2607.11581-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/XinNUS/CycleGRPO-4B"><img src="https://img.shields.io/badge/HuggingFace-Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Models"></a>
  <a href="https://eccv.ecva.net/"><img src="https://img.shields.io/badge/ECCV-2026-1a73e8?style=for-the-badge" alt="ECCV 2026"></a>
</p>

## Generalized CycleGRPO: images and videos

The follow-up paper, **Generalized CycleGRPO for Fine-Grained Multimodal
Understanding and Grounding**, extends the image region–text–region cycle to
video intervals and timestamped spatial evidence.

- [Generalized CycleGRPO paper](https://arxiv.org/html/2610.29552v1)
- [Generalized project page](https://devinxzhang.github.io/Generalized-CycleGRPO-Page/)
- [Video training, data preparation, and checkpoint instructions](generalized/README.md)
- Training entry: `bash generalized/train.sh <profile>`, where `<profile>` is
  `temporal_qwen`, `temporal_omtg`, or `openo3_mixed`. Set the corresponding
  model and data paths first.

The image release and its commands appear below. The Models badge above links
to the ECCV **spatial** model. The video releases are:

| Model | Hugging Face |
|---|---|
| Temporal: Qwen route, captioning and grounding | [Generalized-CycleGRPO-Temporal-4B](https://huggingface.co/XinNUS/Generalized-CycleGRPO-Temporal-4B) |
| Temporal: OMTG route, grounding | [Generalized-CycleGRPO-Temporal-OMTG-4B](https://huggingface.co/XinNUS/Generalized-CycleGRPO-Temporal-OMTG-4B) |
| Spatio-temporal: Open-o3 route | [Generalized-CycleGRPO-SpatioTemporal-4B](https://huggingface.co/XinNUS/Generalized-CycleGRPO-SpatioTemporal-4B) |

These are final inference models. Training uses the corresponding co-SFT
initialization described in the video documentation, not these final models.

## ECCV 2026 spatial CycleGRPO

**Actor as Its Own Critic: Unifying Region Understanding and Localization via CycleGRPO**

This repository provides the official implementation of **CycleGRPO** for
jointly improving region captioning and referring localization with multimodal
reinforcement learning. CycleGRPO uses a **caption ↔ grounding cycle-consistency**
reward. The policy is a Qwen3-VL-4B model that emits **mask tokens** decoded by
a VQ-SAM2 mask tokenizer; the reward runs
an inner grounding rollout conditioned on the model's own caption and scores it
by mask IoU — so captions are optimized to be **distinctive and
locatable**, without requiring caption ground truth during RL.

> Built on [EasyR1](https://github.com/hiyouga/EasyR1) /
> [veRL](https://github.com/volcengine/verl) (see `README_EasyR1.md` for the
> underlying framework).

## Repo layout

```
verl/                     # RL engine (forked EasyR1/veRL): trainer, FSDP workers, vLLM rollout
projects/
  rl/
    qwen3vl_4b_mt.sh       # >>> main image CycleGRPO training entry <<<
    config.yaml            # default RL config (algorithm, rollout, fsdp, reward)
    reward_function/       # text2mask.py: cycle-consistency + mask-IoU reward
    format_prompt/         # prompt templates (non_thinking.jinja)
    datasets/              # scripts that build the RL parquet datasets
  transformers/            # VQ-SAM2 mask tokenizer + SAM2 model code
  vlm/                     # model + eval helpers (refcoco loaders, IoU metrics)
evaluation/
  gres/                    # referring segmentation (GRES)
  groundingsuite/          # GroundingSuite grounding
  gcg/                     # grounded caption generation (GCG)
  gar/                     # GAR-Bench VQA / detailed caption
  dlc_bench/               # dense-captioning eval (DLC-Bench)
  bbox/                    # bbox-format generalization variants
```

## Setup

One environment serves both the spatial release and the video release
(`generalized/`). It is pinned to the stack every released training run used:
Python 3.11, torch 2.8.0 (CUDA 12.8), vLLM 0.11.0, transformers 4.57.1 and
flash-attn 2.8.3, on Linux x86_64 with NVIDIA drivers for CUDA 12.8.

```bash
uv sync                    # creates .venv from uv.lock, including the flash-attn wheel
source .venv/bin/activate
```

Without uv, install the same pins into a Python 3.11 environment with
`pip install -r requirements.txt && pip install -e . --no-deps`.
`requirements.txt` is exported from `uv.lock`.

## Data & checkpoints

Configure the public training launcher with the following paths:

- **`MODEL_PATH`** — the cold-start (co-SFT) Qwen3-VL-4B + mask-token
  checkpoint that RL starts from. The public initialization is
  [`zhouyik/Qwen3-VL-4B-SAMTok-co`](https://huggingface.co/zhouyik/Qwen3-VL-4B-SAMTok-co).
- **`DATA_DIR`** — directory holding the RL `*.parquet` files. Build them
  with the scripts in `projects/rl/datasets/` (e.g. `prepare_dw_rl_dataset.py`,
  `prepare_gres_no_target_rl_dataset.py`). A typical training mix combines
  DenseWorld region samples with GRES no-target samples.
- **`<PATH_TO_COCO2014>`** — COCO2014 `train2014/` images, used only by the
  gres/groundingsuite **eval** scripts.
- **`<PATH_TO_GAR_BENCH>`** — GAR-Bench annotations directory (holds
  `GAR-Bench-VQA.json` / `GAR-Bench-Caption-Detailed.json` and the `images/`),
  used by the GAR eval scripts. GAR-Bench is a separate public benchmark — get it
  from the official Grasp-Any-Region release.

## Training

Download the public co-SFT initialization, prepare the parquet files
described in [`TRAIN.md`](TRAIN.md), and launch:

```bash
hf download zhouyik/Qwen3-VL-4B-SAMTok-co \
  --local-dir checkpoints/Qwen3-VL-4B-SAMTok-co

MODEL_PATH=checkpoints/Qwen3-VL-4B-SAMTok-co \
DATA_DIR=data/cyclegrpo \
bash projects/rl/qwen3vl_4b_mt.sh
```

Defaults assume 1 node × 8 GPUs. The launcher reads paths from environment
variables and checks all required parquet and SAMTok weight files before
starting. `WANDB_API_KEY` is optional and `WANDB_MODE` defaults to `offline`.

### Memory tuning (multi-image / OOM)

Multi-image samples produce long prompts; the actor backward pass can run out
of memory. Pass any of these as extra arguments to the launcher, e.g.
`bash projects/rl/qwen3vl_4b_mt.sh data.mini_rollout_batch_size=16`:

- `data.max_prompt_length` / `worker.rollout.max_num_batched_tokens` — raise to fit long prompts (costs memory).
- `worker.actor.micro_batch_size_per_device_for_{experience,update}=1` — smallest micro-batch.
- `data.mini_rollout_batch_size=16` — smaller vLLM generation batch (lowers rollout-phase memory).
- `worker.rollout.n` should be a multiple of `world_size` (`nnodes × n_gpus`) when mixing
  cycle (region) and no-target sources, so the per-source sub-batches divide evenly across ranks.

## Inference / evaluation

Each benchmark lives under `evaluation/<benchmark>/`. The multi-GPU launchers
validate every input path before loading a model, shard the dataset across
GPUs, and save one JSON result per sample. SAMTok weights default to
`./checkpoints/Qwen3-VL-4B-SAMTok-co/` (the snapshot from [TRAIN.md](TRAIN.md));
override them with `VQ_SAM2_PATH` and `SAM2_PATH`.

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

# GAR-Bench VQA (single-process inference, then metrics)
python evaluation/gar/qwen3vl_gar_vqa_infer.py <MODEL_PATH> \
  --gar_bench_root /data/GAR-Bench/annotations --output results/gar/vqa.json
python evaluation/gar/gar_vqa_metrics.py results/gar/vqa.json

# DLC-Bench (start the Llama judge server in a separate shell, then infer + eval)
bash evaluation/dlc_bench/serve_judge.sh
bash evaluation/dlc_bench/evaluate_dlc.sh <MODEL_PATH> <CACHE_NAME>
python evaluation/dlc_bench/eval_llama_without_image.py \
  --pred evaluation/dlc_bench/model_outputs/<CACHE_NAME>.json --base-url http://localhost:8007/v1
```

To check paths without a GPU, pass `--check_only` to the GRES, GroundingSuite,
or GCG inference script; it reports every missing input at once and exits.
Bounding-box generalization variants live in `evaluation/bbox/` and still use
the `<PATH_TO_*>` placeholders inside the scripts.

## Results

Base **[SAMTok](https://huggingface.co/zhouyik/Qwen3-VL-4B-SAMTok)** (Qwen3-VL-4B)
versus **CycleGRPO** (this work). Two CycleGRPO rows are reported: **paper**
shows the numbers in the ECCV 2026 paper, and **release** shows results from
the public checkpoint [`XinNUS/CycleGRPO-4B`](https://huggingface.co/XinNUS/CycleGRPO-4B).
The release comes from a rerun and has similar or slightly higher overall scores.

**Region captioning — DLC-Bench** (100 samples):

| Method | Pos. | Neg. | Avg. |
|---|---:|---:|---:|
| SAMTok | 43.5 | 80.4 | 61.9 |
| CycleGRPO (paper) | 51.2 | 84.2 | 67.7 |
| CycleGRPO (release) | 52.4 | 83.2 | 67.8 |

**Text-to-mask — GroundingSuite** (gIoU, %):

| Method | Stuff | Part | Multi | Single | All |
|---|---:|---:|---:|---:|---:|
| SAMTok | 80.9 | 12.4 | 62.0 | 52.9 | 57.5 |
| CycleGRPO (paper) | 90.7 | 20.9 | 76.3 | 61.6 | 67.6 |
| CycleGRPO (release) | 90.5 | 21.2 | 78.3 | 62.3 | 68.2 |

**Region VQA — GAR-Bench-VQA** (%):

| Method | Overall | Color | Shape | Texture | Material | Position | Non-Entity | Relation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAMTok | 64.2 | 58.0 | 48.4 | 48.3 | 58.3 | 76.6 | 54.1 | 83.2 |
| CycleGRPO (paper) | 65.1 | 62.3 | 50.0 | 48.3 | 61.1 | 73.4 | 57.4 | 82.2 |
| CycleGRPO (release) | 64.9 | 60.9 | 50.0 | 48.3 | 61.1 | 73.4 | 54.1 | 84.2 |

**Interleaved text-mask — GCG** (METEOR / CIDEr / AP50 / mIoU / Recall):

| Method | val M | val C | val AP50 | val mIoU | val Rec | test M | test C | test AP50 | test mIoU | test Rec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAMTok | 16.1 | 48.2 | 34.7 | 69.4 | 46.6 | 16.4 | 51.4 | 34.4 | 68.4 | 48.3 |
| CycleGRPO (paper) | 17.2 | 54.7 | 35.9 | 69.6 | 49.6 | 17.1 | 54.0 | 35.2 | 68.6 | 49.7 |
| CycleGRPO (release) | 17.3 | 54.3 | 36.8 | 70.2 | 50.2 | 17.2 | 53.7 | 35.0 | 69.2 | 49.8 |

**Referring segmentation + target rejection — GRES** (gIoU / cIoU / N-acc, %):

| Method | Val gIoU | Val cIoU | Val N-acc | TestA gIoU | TestA cIoU | TestA N-acc | TestB gIoU | TestB cIoU | TestB N-acc | Avg gIoU | Avg cIoU | Avg N-acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAMTok | 71.3 | 69.2 | 61.4 | 75.3 | 75.4 | 59.0 | 66.9 | 66.0 | 55.6 | 71.2 | 70.2 | 58.7 |
| CycleGRPO (paper) | 81.8 | 74.6 | 94.2 | 79.9 | 77.8 | 93.1 | 73.0 | 70.0 | 89.0 | 78.2 | 74.1 | 92.1 |
| CycleGRPO (release) | 82.2 | 74.8 | 94.7 | 80.3 | 78.2 | 93.0 | 73.5 | 70.2 | 89.9 | 78.7 | 74.4 | 92.5 |

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
  author        = {Zhang, Xin and Wang, Haochen and Lin, Beibei and Zhang, Zhaoxiang and Tan, Robby T. and Li, Jason and Tao, Dacheng},
  year          = {2026},
  eprint        = {2610.29552},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.29552}
}
```

## Acknowledgements

Built on [EasyR1](https://github.com/hiyouga/EasyR1) and
[veRL](https://github.com/volcengine/verl); segmentation via
[SAM2](https://github.com/facebookresearch/sam2). See `LICENSE`.
