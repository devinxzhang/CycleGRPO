# Generalized CycleGRPO training annotations v1

JSONL training annotations and case manifests for the three Generalized
CycleGRPO video profiles:

- Temporal Qwen: **18,742 RL examples**.
- Temporal OMTG: **18,735 RL examples**, plus **12,242 co-SFT conversations**.
- Open-o3 mixed: **35,887 RL examples**, plus **9,201 co-SFT conversations**.

The archive includes selected annotations, video split and selection
manifests, media references, content hashes, and tools to generate training
parquet/JSON files. Download source images and videos separately from
[OMTG](https://huggingface.co/datasets/insomnia7/omtg56k) and
[Open-o3-Video](https://huggingface.co/datasets/marinero4972/Open-o3-Video).

From the repository root:

```bash
python generalized/download_data.py
```

See [data preparation](https://github.com/devinxzhang/CycleGRPO/blob/generalized-data-v1/generalized/README.md#data-preparation)
for the media layout and training commands. Qwen temporal co-SFT annotations
and initialization model weights are outside this annotation release.

Asset: `generalized-cyclegrpo-training-data-20261112.tar.gz` (13,327,403 bytes).

SHA256:

```text
f369777afd186ccecd9837cf9e319509a9145bb72d684d383e234675fe7703dc
```

Annotations are released as they appear in their sources. The preparation tool
takes an optional `--normalize-mcq-instruction` flag, which rewrites
multiple-choice instruction wording only and leaves answer labels unchanged.
Source attribution and component license notes are included in the archive.
