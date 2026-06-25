# CycleGRPO (image-level)

RL fine-tuning for **referring segmentation + region captioning** with a
**caption ↔ grounding cycle-consistency** reward. The policy is a Qwen3-VL-4B
that emits **mask tokens** decoded by a VQ-SAM2 mask tokenizer; the reward runs
an inner grounding rollout conditioned on the model's own caption and scores it
by mask/temporal IoU — so captions are optimized to be **distinctive and
locatable**, with no caption ground-truth needed in the RL stage.

> This release covers the **image-level** pipeline only. It is built on
> [EasyR1](https://github.com/hiyouga/EasyR1) / [veRL](https://github.com/volcengine/verl)
> (see `README_EasyR1.md` for the underlying framework).

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
  qwen3vl/                 # inference / eval scripts (gres, gcg, groundingsuite, refcoco)
  DLC-Bench/               # dense-captioning eval scripts
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
  refcoco/gres/groundingsuite **eval** scripts.

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

Image segmentation/grounding evals live in `evaluation/qwen3vl/`. Multi-GPU
launchers shard the dataset across GPUs and auto-merge:

```bash
# Referring segmentation (GRES)
bash evaluation/qwen3vl/run_gres_multigpu.sh        8 <MODEL_PATH> ./results/gres/
# GroundingSuite
bash evaluation/qwen3vl/run_groundingsuite_multigpu.sh 8 <MODEL_PATH> ./results/groundingsuite/
# Grounded caption generation (GCG)
bash evaluation/qwen3vl/run_gcg_multigpu.sh         8 <MODEL_PATH> ./results/gcg/
```

Set the COCO image path (`<PATH_TO_COCO2014>`) inside the eval scripts where noted.

## Acknowledgements

Built on [EasyR1](https://github.com/hiyouga/EasyR1) and
[veRL](https://github.com/volcengine/verl); segmentation via
[SAM2](https://github.com/facebookresearch/sam2). See `LICENSE`.
