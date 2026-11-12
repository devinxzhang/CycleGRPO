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

The "Data preparation" section below gives the upstream sources for each of
these three and the exact selection rule behind each partition.

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

These are final inference models. The selected training annotations are
available in the [v1 data release](https://github.com/devinxzhang/CycleGRPO/releases/tag/generalized-data-v1);
task-specific co-SFT initialization weights are supplied separately.

## Environment

The video release uses the same environment as the spatial release; the
temporal and spatio-temporal runs were trained with exactly these pins.

```bash
uv sync && source .venv/bin/activate   # from the repository root
```

CPU tests do not validate an entire distributed GPU training environment.

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

### Download the selected annotations

The [v1 data release](https://github.com/devinxzhang/CycleGRPO/releases/tag/generalized-data-v1) contains JSONL annotations, frozen
selection/split manifests and preparation tools. No images, videos, weights
or frame caches are bundled, and no separate Hugging Face dataset download
is required for these annotations.

```bash
python generalized/download_data.py
D=data/generalized-cyclegrpo-rl-data
python "$D/scripts/release_data.py" validate
```

The download is pinned to `generalized-data-v1` and SHA256-verified before extraction.
Rerunning the command verifies an existing installation. To transfer the
archive from another machine, use `python generalized/download_data.py
--archive /path/to/archive.tar.gz`.

| Profile | RL rows | Included co-SFT conversations |
|---|---:|---:|
| `temporal_qwen` | 18,742 | — |
| `temporal_omtg` | 18,735 | 12,242 |
| `openo3_mixed` | 35,887 | 9,201 |

Obtain the original media from [OMTG](https://huggingface.co/datasets/insomnia7/omtg56k)
and [Open-o3-Video](https://huggingface.co/datasets/marinero4972/Open-o3-Video),
following the source-specific download instructions in the package's
`docs/SOURCES.md`. Preserve the source directories listed in
`manifests/media.jsonl`; some video IDs have multiple encoded copies.

Generate the parquet for the profile you want to train:

```bash
# OMTG media root contains cosmo-cap/, moment10m/ and vtimellm/.
python "$D/scripts/release_data.py" materialize --profile temporal_qwen \
  --omtg-root /data/temporal_sources \
  --output "$PWD/data/generalized_prepared/temporal_qwen.parquet"

python "$D/scripts/release_data.py" materialize --profile temporal_omtg \
  --omtg-root /data/temporal_sources \
  --output "$PWD/data/generalized_prepared/temporal_omtg.parquet"

# Open-o3 media root contains videos/ with the upstream directory layout.
python "$D/scripts/release_data.py" materialize --profile openo3_mixed \
  --openo3-root /data/Open-o3-Video-data \
  --output "$PWD/data/generalized_prepared/openo3_mixed.parquet"

MODEL_PATH=/models/temporal_qwen_cosft \
TRAIN_FILE="$PWD/data/generalized_prepared/temporal_qwen.parquet" \
bash generalized/train.sh temporal_qwen
```

The preparation tool checks media availability and recorded byte sizes before
writing, preserves targets and row order, and refuses to overwrite outputs.
`check-media` accepts the same profile/media-root arguments for a check without
writing parquet. For new Open-o3 training, optional
`--normalize-mcq-instruction` makes the output instruction enumerate the
choices actually shown while preserving the original answer labels; the
distributed JSONL retains the source prompts.

For an included co-SFT set, add `--stage cosft` and choose a `.json` output.
For example:

```bash
python "$D/scripts/release_data.py" materialize --profile temporal_omtg --stage cosft \
  --omtg-root /data/temporal_sources \
  --output "$PWD/data/generalized_prepared/temporal_omtg_cosft.json"
```

Qwen temporal co-SFT annotations and model initialization weights are outside
this data release. The package documents its annotation provenance and
component source licenses. The construction notes below describe the released
subsets; downloading them avoids rebuilding the selection pipeline.

### Spatio-temporal: Open-o3 route (`openo3_mixed`)

Upstream sources, none of which this repository redistributes:

- Annotations: [`marinero4972/Open-o3-Video`](https://huggingface.co/datasets/marinero4972/Open-o3-Video).
  The converters read `json_data/STGR-RL.json` from it.
- Videos: that dataset card lists the upstream video releases to fetch
  separately, including VideoEspresso and Video-R1. Place them under
  `<data root>/videos/` so the paths inside the JSON resolve.
- Code and paper: [Open-o3-Video](https://github.com/marinero4972/Open-o3-Video),
  [arXiv:2510.20579](https://arxiv.org/abs/2510.20579).

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
package imports resolve. These scripts support building custom mixtures;
the v1 download provides the selected annotations directly. Source videos are
obtained separately. The selected main mixture comprises
35,887 rows: 4,143 spatial, 4,820 temporal, 11,924 spatio-temporal, and 15,000 QA.
Custom datasets may differ; do not describe a rebuilt mixture as the exact
paper dataset without checking its audit and split manifests.

### Temporal routes (`temporal_qwen`, `temporal_omtg`)

Both selected subsets are included in the v1 download. They come from one
intermediate table: a one-span conversion of the OMTG temporal-grounding
training corpus (its cosmo-cap, Moment-10M, VTimeLLM and TimeLens CoT inputs),
giving 36,892 rows over 9,491 storage-independent video IDs, one query and one
target interval per row, with `source` set to `tg_multi_merged`.

The public OMTG release is the place to start when rebuilding an equivalent
table: [project page](https://insomniaaac.github.io/OMTG/),
[`insomnia7/omtg56k`](https://huggingface.co/datasets/insomnia7/omtg56k) (46k
SFT + 10k RL). Its row counts differ from the corpus used here, so a rebuild is
an equivalent recipe, not the same rows.

The two partitions are then selected as follows.

`temporal_qwen` takes a video-disjoint half. The 9,491 unique video IDs are
split 50/50 under seed 42 into a co-SFT half (4,746 videos) and an RL half
(4,745 videos); the RL rows are the 18,742 rows whose video falls in the RL
half. The property this buys is that the co-SFT initialization and the RL stage
share no videos. It says nothing about the upstream OMTG SFT corpus, which does
overlap these videos.

`temporal_omtg` takes the visually distinguishable subset. Each row was audited
for whether its target interval is visually distinguishable from the rest of its
own video and covers one complete event. A row is kept when the audit completed
and returned `keep` or `keep_singleton`: 18,735 of the 36,892 rows, over 7,240
videos (18,507 `keep`, 228 `keep_singleton`; 148 rejected as non-atomic, 2,716
for visual ambiguity, and 15,293 left at `review` and therefore dropped). The
audit used an external vision-language judge over frame strips. The release
includes the selected rows and frozen decisions, so using the dataset does
not require running that judge again.

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
