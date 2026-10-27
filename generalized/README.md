# Generalized CycleGRPO: video training

This release extends the image implementation with temporal interval cycles
and Open-o3-Video-style timestamped spatial evidence. It contains the actual
nested-rollout trainer, media conditioning, task routing, overlap rewards,
token-level GRPO updates, and checkpoint merging used by the video project.

The ECCV image entry point and root `verl/` remain unchanged. Video code lives
in `generalized/verl/` and runs as `generalized.verl`, so it cannot silently
import the older image trainer. Run commands from the repository root.

## Initialization is task-specific

| Profile | Required RL initialization | Training data |
|---|---|---|
| `temporal_qwen` | Qwen3-VL-4B after bidirectional temporal co-SFT | Video-disjoint temporal RL partition from OMTG |
| `temporal_omtg` | OMTG-4B RL after bidirectional temporal co-SFT | Visually distinguishable temporal examples from OMTG |
| `openo3_mixed` | Open-o3-Video-4B RL after inverse-caption + video-QA co-SFT | Spatial, temporal, spatio-temporal, and auxiliary video-QA mix from Open-o3-Video |

Do **not** substitute a generic Qwen Instruct checkpoint, the original OMTG
checkpoint, or the original Open-o3 checkpoint for the corresponding co-SFT
initialization. The inverse description interface is learned during co-SFT.
Do not start RL from the final CycleGRPO checkpoint unless deliberately
continuing that run.

The existing `XinNUS/CycleGRPO-4B` model link is the **ECCV image model**.
The separate video inference models are:

- [Temporal: Qwen route](https://huggingface.co/XinNUS/Generalized-CycleGRPO-Temporal-4B),
  used for the main DOVE result and the Qwen-initialized TimeLens comparison.
- [Temporal: OMTG route](https://huggingface.co/XinNUS/Generalized-CycleGRPO-Temporal-OMTG-4B),
  used for the OMTG-initialized TimeLens comparison.
- [Spatio-temporal: Open-o3 route](https://huggingface.co/XinNUS/Generalized-CycleGRPO-SpatioTemporal-4B),
  used for the main V-STaR and general-video comparisons.

These are inference releases, not the co-SFT models that start RL. Curated
training artifacts and the corresponding co-SFT initializations must be
provided separately. No private experiment filesystem is a download endpoint.

## Environment

Use a separate CUDA environment for the video release. The original root
requirements file has an older Transformers upper bound and should not be
combined with the video requirements.

```bash
# First install mutually compatible CUDA builds of torch, torchvision,
# vLLM, and flash-attn in your environment.
pip install -r generalized/requirements.txt
```

The version ranges in that file describe the Qwen3-VL runtime family, not a
claim that every CUDA/driver combination has been validated. Install the
compatible FlashAttention build separately. CPU tests do not validate an
entire distributed GPU training environment.

## Training

All commands run in the foreground and stream output to the terminal.
Paths are user-supplied; no API key, private endpoint, or private storage mount
is embedded in the launchers.

```bash
# Choose one profile and its corresponding co-SFT model/data.
MODEL_PATH=/path/to/temporal_qwen_cosft \
TRAIN_FILE=/path/to/temporal_rl.parquet \
bash generalized/train.sh temporal_qwen

MODEL_PATH=/path/to/omtg_bidirectional_cosft \
TRAIN_FILE=/path/to/visually_distinct_temporal_rl.parquet \
bash generalized/train.sh temporal_omtg

MODEL_PATH=/path/to/openo3_inverse_caption_vqa_cosft \
TRAIN_FILE=/path/to/openo3_mixed_temporal_hard.parquet \
bash generalized/train.sh openo3_mixed
```

Defaults are one node with eight GPUs, outer caption samples `N=4`, and inner
localization samples `K=4`. Set `NNODES=2 GPUS_PER_NODE=8` for two shared-filesystem
nodes with sixteen GPUs. Start the Ray head and connect the other node(s)
before launching once on the head. The launcher never allocates remote
machines or kills existing processes. Ray workers receive the runtime and
video-conditioning environment explicitly.

`MODEL_PATH` must be a complete local HF checkpoint. `TRAIN_FILE` must be a
prepared parquet with valid local media paths. Check these inputs without
starting training by adding `DRY_RUN=1`. This checks checkpoint inventory and
parquet schema/source routes, then resolves and prints the full trainer config
including any Hydra overrides you append, so misspelled keys or bad values fail
here. It does not start Ray or certify video decodability or GPU memory
capacity. Set `VAL_FILE` only if you enable validation; it is optional.

```bash
DRY_RUN=1 MODEL_PATH=/path/to/cosft TRAIN_FILE=/path/to/train.parquet \
  bash generalized/train.sh temporal_qwen
```

For OMTG, the video protocol is 2 fps, 4–768 frames, and a total pixel
budget of 8,388,608. An optional deterministic frame cache is supported via
`CYCLEGRPO_FRAME_CACHE_ROOT`; set `CYCLEGRPO_REQUIRE_FRAME_CACHE=true` only after
building a complete compatible cache. The Qwen temporal recipe retains
0.25 fps and a 262,144-pixel frame cap. Open-o3 uses 2 fps, 4–16 frames,
plus timestamped query conditioning. These are training protocols, not a
claim that every evaluation model receives a fixed identical frame count.

Useful environment overrides include `ROLLOUT_N`, `CYCLEGRPO_INNER_ROLLOUT_N`,
`ROLLOUT_BATCH_SIZE`, `GLOBAL_BATCH_SIZE`, `SAVE_FREQ`, `SAVE_CHECKPOINT_PATH`,
and `MAX_STEPS`. Additional configuration overrides can be appended to the
command. Do not increase micro-batch sizes without a memory smoke test.

The launchers use the paper's common `N=K=4` setting. They are configurable
training recipes, not bitwise recreations of every historical training run.
Exact run reproduction also requires the matching co-SFT weights, curated
parquet and split/audit manifests, random state, and GPU software environment.

## What the cycle does

- A region-conditioned policy generates `N` captions.
- Each caption conditions `K` localization rollouts from the same policy.
- Temporal sources use temporal overlap, including after compact multimodal
  transport. They must not fall through to mask-token accuracy.
- Spatio-temporal sources parse normalized timestamp–box supports and score
  their overlap. The main mixed recipe uses `stgr_cyclegrpo`; the atomic and
  single-object trajectory variants are available separately, not aliases
  for the main experiment.
- Captions receive reconstructed-region feedback and GT-free validity terms.
  Localization rollouts are normalized within their own caption's group.
- The mixed Open-o3 recipe additionally retains answer-based auxiliary QA.
  Its optional dense temporal shaping and explicit sampling weights are
  preserved; neither creates new annotation targets.

The independent captioning-only controls use an external caption judge; this
is **not** required for either direction of CycleGRPO. Optional judge clients
read `LLM_AS_A_JUDGE_BASES`, `LLM_AS_A_JUDGE_KEY`, and `LLM_AS_A_JUDGE_MODEL`
from the environment. Never paste credentials into source or launch scripts.

## Data preparation

The public converters under `projects/rl/datasets/` cover:

1. `prepare_open_o3_unified_cyclegrpo.py`: build region-cycle rows from
   Open-o3 `STGR-RL.json`; default data root is `data/Open-o3-Video` and can be
   changed with `--data-root`.
2. `audit_open_o3_unified_cyclegrpo.py`: audit annotation consistency.
3. `prepare_open_o3_unified_cyclegrpo_clean_v2.py`: apply the audit and add
   query conditioning.
4. `prepare_open_o3_cyclegrpo_vqa_mix.py`: restore the 15,000 auxiliary QA rows.
5. `prepare_open_o3_temporal_hard_mix.py`: attach explicit sampling weights.

Run each script with `--help` for its input/output arguments. Use
`python -m projects.rl.datasets.<script_name>` from the repository root so
package imports resolve. These scripts do not download source videos or
reconstruct the private offline audit. The selected main mixture comprises
35,887 rows: 4,143 spatial, 4,820 temporal, 11,924 spatio-temporal, and 15,000 QA.
Custom datasets may differ; do not describe a rebuilt mixture as the exact
paper dataset without checking its audit and split manifests.

Temporal parquets contain at least `videos`, `cap_problem`, `cap_answer`,
`seg_problem`, `seg_answer`, and `source`. The cycle source is `tg_multi_merged`.
The caption prompt specifies the target interval; reconstruction is performed
in the source video. Caption targets are not read by the cycle reward.
Mixed-task rows also carry query timestamps/boxes and `sampling_weight`.

## Checkpoints and tests

Each saved `global_step_*` directory contains distributed model/optimizer and
dataloader state for exact resume. `ckpt_*` is the merged inference model and
is **not** a replacement for optimizer state. Resume with
`LOAD_CHECKPOINT_PATH=/path/to/global_step_5`, not `ckpt_5`. The launcher keeps
one resumable checkpoint and retains merged checkpoints separately.

```bash
pip install -r requirements-test.txt  # plus a CPU build of torch
make check                            # ruff + all CPU tests (spatial, video, evaluation)
```

The tests cover task registries, temporal reward routing, temporal shaping,
spatio-temporal parsing/overlap, and video decoding/cache guards. Real video
smoke tests are optional and skipped when no fixture is provided. A clean
CPU test run does not imply that a multi-node training run was performed.

For evaluation, use the corresponding benchmark's official inference and
scoring protocol. The old `evaluation/` directory remains the image release;
it must not be used to score video checkpoints as if they were SAMTok models.
Temporal captioning supplies the full source video and specifies one target
interval in the prompt. Keep the video-caption model separate from the
caption-only judge. V-STaR additionally requires its own grounded-answer
serialization and official scoring, rather than free-form caption metrics.
