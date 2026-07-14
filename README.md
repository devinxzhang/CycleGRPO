# Actor as Its Own Critic: Unifying Region Understanding and Localization via CycleGRPO

<p align="center">
  <a href="https://devinxzhang.github.io/CycleGRPO-Page/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/pdf/2607.11581"><img src="https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Paper"></a>
  <a href="https://arxiv.org/abs/2607.11581"><img src="https://img.shields.io/badge/arXiv-2607.11581-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/XinNUS/CycleGRPO-4B"><img src="https://img.shields.io/badge/HuggingFace-Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Models"></a>
  <a href="https://eccv.ecva.net/"><img src="https://img.shields.io/badge/ECCV-2026-1a73e8?style=for-the-badge" alt="ECCV 2026"></a>
</p>

> **🚧 Work in progress.** This repository is still under active development and
> is **not** the final/official release.

RL fine-tuning for **referring segmentation + region captioning** with a
**caption ↔ grounding cycle-consistency** reward. The policy is a Qwen3-VL-4B
that emits **mask tokens** decoded by a VQ-SAM2 mask tokenizer; the reward runs
an inner grounding rollout conditioned on the model's own caption and scores it
by mask/temporal IoU — so captions are optimized to be **distinctive and
locatable**, with no caption ground-truth needed in the RL stage.

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

```bash
pip install -r requirements.txt
pip install -e .          # editable install of the verl package
```

Requires CUDA GPUs, PyTorch, and vLLM (SPMD mode). See `requirements.txt`.

## Data & checkpoints

The scripts use placeholders you must fill in:

- **`<PATH_TO_COLD_START_CKPT>`** — the cold-start (co-SFT) Qwen3-VL-4B + mask-token
  checkpoint that RL starts from. (Train it with SFT first, or download the
  released checkpoint — see project page.)
- **`<PATH_TO_DATA>`** — directory holding the RL `*.parquet` files. Build them
  with the scripts in `projects/rl/datasets/` (e.g. `prepare_dw_rl_dataset.py`,
  `prepare_gres_no_target_rl_dataset.py`). A training mix typically combines
  dense-region (denseworld) + no-target (gres) parquets.
- **`<PATH_TO_COCO2014>`** — COCO2014 `train2014/` images, used only by the
  gres/groundingsuite **eval** scripts.
- **`<PATH_TO_GAR_BENCH>`** — GAR-Bench annotations directory (holds
  `GAR-Bench-VQA.json` / `GAR-Bench-Caption-Detailed.json` and the `images/`),
  used by the GAR eval scripts. GAR-Bench is a separate public benchmark — get it
  from the official Grasp-Any-Region release.

## Training

```bash
bash projects/rl/qwen3vl_4b_mt.sh
```

Edit the script first: set `MODEL_PATH`, `data.train_files`, `data.val_files`.
Defaults assume 1 node × 8 GPUs. `WANDB_API_KEY` is read from the environment
(defaults to empty / offline).

### Memory tuning (multi-image / OOM)

Multi-image samples produce long prompts; the actor backward can OOM. The main
script has a commented block of levers — append them to the `python3 -m
verl.trainer.main` command as needed:

- `data.max_prompt_length` / `worker.rollout.max_num_batched_tokens` — raise to fit long prompts (costs memory).
- `worker.actor.micro_batch_size_per_device_for_{experience,update}=1` — smallest micro-batch.
- `data.mini_rollout_batch_size=16` — smaller vLLM generation batch (lowers rollout-phase memory).
- `worker.rollout.n` should be a multiple of `world_size` (`nnodes × n_gpus`) when mixing
  cycle (region) and no-target sources, so the per-source sub-batches divide evenly across ranks.

## Inference / evaluation

Each benchmark lives under `evaluation/<benchmark>/`. The multi-GPU launchers
shard the dataset across GPUs and auto-merge:

```bash
# Referring segmentation (GRES)
bash evaluation/gres/run_gres_multigpu.sh                     8 <MODEL_PATH> ./results/gres/
# GroundingSuite
bash evaluation/groundingsuite/run_groundingsuite_multigpu.sh 8 <MODEL_PATH> ./results/groundingsuite/
# Grounded caption generation (GCG)
bash evaluation/gcg/run_gcg_multigpu.sh                       8 <MODEL_PATH> ./results/gcg/

# GAR-Bench VQA (single-process inference, then metrics)
python evaluation/gar/qwen3vl_gar_vqa_infer.py <MODEL_PATH> --output results/gar/vqa.json
python evaluation/gar/gar_vqa_metrics.py results/gar/vqa.json

# DLC-Bench (start the Llama judge server in a separate shell, then infer + eval)
bash evaluation/dlc_bench/serve_judge.sh
bash evaluation/dlc_bench/evaluate_dlc.sh <MODEL_PATH> <CACHE_NAME>
python evaluation/dlc_bench/eval_llama_without_image.py \
  --pred evaluation/dlc_bench/model_outputs/<CACHE_NAME>.json --base-url http://localhost:8007/v1
```

Fill the dataset placeholders inside the eval scripts where noted:
`<PATH_TO_COCO2014>` (gres / groundingsuite) and `<PATH_TO_GAR_BENCH>` (gar).
bbox-format generalization variants live in `evaluation/bbox/`.

## Results

Base **[SAMTok](https://huggingface.co/zhouyik/Qwen3-VL-4B-SAMTok)** (Qwen3-VL-4B) vs **CycleGRPO** (this work). Two CycleGRPO rows are
reported: **paper** = the numbers in the ECCV 2026 paper, and **release** = the
public checkpoint [`XinNUS/CycleGRPO-4B`](https://huggingface.co/XinNUS/CycleGRPO-4B),
a re-run that varies slightly from the paper (overall on par / marginally higher).

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

## Acknowledgements

Built on [EasyR1](https://github.com/hiyouga/EasyR1) and
[veRL](https://github.com/volcengine/verl); segmentation via
[SAM2](https://github.com/facebookresearch/sam2). See `LICENSE`.
