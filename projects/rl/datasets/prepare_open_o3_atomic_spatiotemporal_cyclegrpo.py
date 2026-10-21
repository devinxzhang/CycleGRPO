"""Build atomic timestamp+bbox CycleGRPO data from Open-o3-Video.

Each output row contains exactly one queried timestamp and one queried bbox.
Multi-object Open-o3 annotations are split into independent rows. The original
question/answer remain metadata only: the outer caption prompt is driven
solely by the visualized spatio-temporal query.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from projects.rl.datasets.prepare_open_o3_spatiotemporal_cyclegrpo import (
    DEFAULT_DATA_ROOT,
    TARGET_TASK,
    _split_name,
    build_claims,
    resolve_video_path,
)
from generalized.verl.utils.spatiotemporal import (
    ATOMIC_SPATIOTEMPORAL_SOURCE,
    SpatioTemporalClaim,
    bbox_iou,
    serialize_spatiotemporal_claims,
)


def build_atomic_caption_prompt() -> str:
    """Prompt for the query-conditioned local clip produced by the loader."""

    return (
        "<video>\n"
        "This query video represents exactly one spatio-temporal target. It first "
        "shows local temporal context, then the exact queried frame with the target "
        "marked by a red rectangle, and finally a close-up of that same target. "
        "Describe only the marked target instance. Give a concise but discriminative "
        "description of its visible appearance, action, interaction, and useful "
        "relations to nearby objects so that it can be uniquely located in the "
        "original video. Do not mention the red rectangle, crop, query, coordinates, "
        "timestamps, seconds, frame indices, or object tags. Output description only."
    )


def iter_atomic_claims(
    claims: list[SpatioTemporalClaim],
    *,
    dedupe_iou: float = 0.95,
):
    """Split into atomic targets and remove duplicate same-time regions."""

    atomic_index = 0
    kept: list[SpatioTemporalClaim] = []
    for claim in claims:
        for box in claim.boxes:
            atomic_claim = SpatioTemporalClaim(
                object_name=claim.object_name,
                timestamp=claim.timestamp,
                boxes=(box,),
            )
            duplicate = any(
                abs(previous.timestamp - atomic_claim.timestamp) < 1e-6
                and bbox_iou(previous.boxes[0], atomic_claim.boxes[0])
                >= dedupe_iou
                for previous in kept
            )
            if duplicate:
                continue
            kept.append(atomic_claim)
            yield atomic_index, atomic_claim
            atomic_index += 1


def convert(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = args.data_root.resolve()
    input_json = args.input_json or data_root / "json_data" / "STGR-RL.json"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_json.open(encoding="utf-8") as handle:
        examples = json.load(handle)

    rows = {"train": [], "val": []}
    skipped = Counter()
    source_counts = Counter()
    original_examples = Counter()
    checked_paths: dict[str, bool] = {}

    for example in examples:
        if example.get("task") != TARGET_TASK:
            continue

        source = str(example.get("source") or "")
        try:
            video_path = resolve_video_path(
                data_root,
                source,
                str(example.get("video_path") or ""),
            ).resolve()
        except ValueError:
            skipped["unsupported_source"] += 1
            continue

        video_key = str(video_path)
        if not args.skip_path_validation:
            exists = checked_paths.setdefault(video_key, video_path.is_file())
            if not exists:
                skipped["missing_video"] += 1
                continue

        claims = build_claims(example)
        if not claims:
            skipped["no_valid_claims"] += 1
            continue

        split = _split_name(video_key, args.val_ratio, args.seed)
        original_examples[split] += 1
        question = str(example.get("question") or "")
        answer = str(example.get("answer") or "")
        original_target_count = sum(len(claim.boxes) for claim in claims)
        atomic_claims = list(
            iter_atomic_claims(claims, dedupe_iou=args.dedupe_iou)
        )
        skipped["duplicate_atomic_query"] += (
            original_target_count - len(atomic_claims)
        )
        for atomic_index, claim in atomic_claims:
            if args.max_samples is not None and sum(map(len, rows.values())) >= args.max_samples:
                break
            box = [float(value) for value in claim.boxes[0]]
            row = {
                "videos": [video_key],
                # Used only by the unmarked inner grounding rollout.
                "nframes": [args.nframes],
                # The loader derives a query-local clip from these fields only
                # for the outer caption rollout.
                "query_timestamp": float(claim.timestamp),
                "query_bbox": box,
                "query_context_seconds": float(args.query_context_seconds),
                "query_context_frames": int(args.query_context_frames),
                "cap_problem": build_atomic_caption_prompt(),
                "cap_answer": "",
                # Replaced at runtime by the sampled outer description.
                "seg_problem": "",
                "seg_answer": serialize_spatiotemporal_claims([claim]),
                "masks": None,
                "source": ATOMIC_SPATIOTEMPORAL_SOURCE,
                "extra_info": {
                    "open_o3_id": str(example.get("id") or ""),
                    "open_o3_source": source,
                    "atomic_index": int(atomic_index),
                    "target_object": claim.object_name,
                    "query_timestamp": float(claim.timestamp),
                    "query_bbox_json": json.dumps(box),
                    # Metadata only; neither enters the outer prompt or reward.
                    "question": question,
                    "answer": answer,
                    "video_path": str(example.get("video_path") or ""),
                },
            }
            rows[split].append(row)
            source_counts[source] += 1

        if args.max_samples is not None and sum(map(len, rows.values())) >= args.max_samples:
            break

    if not rows["train"]:
        raise RuntimeError("No atomic training rows were produced.")
    if not rows["val"]:
        fallback_video = rows["train"][-1]["videos"][0]
        moved = [row for row in rows["train"] if row["videos"][0] == fallback_video]
        rows["train"] = [row for row in rows["train"] if row["videos"][0] != fallback_video]
        rows["val"].extend(moved)

    train_videos = {row["videos"][0] for row in rows["train"]}
    val_videos = {row["videos"][0] for row in rows["val"]}
    overlap = train_videos & val_videos
    if overlap:
        raise AssertionError(f"Video leakage across splits: {len(overlap)} videos")

    for split in ("train", "val"):
        for row in rows[split]:
            answer_claim = row["seg_answer"]
            if answer_claim.count("<obj>") != 1 or answer_claim.count("<box>") != 1:
                raise AssertionError("Atomic row contains multiple claims or boxes.")

    train_path = output_dir / args.train_name
    val_path = output_dir / args.val_name
    pd.DataFrame(rows["train"]).to_parquet(train_path, index=False)
    pd.DataFrame(rows["val"]).to_parquet(val_path, index=False)

    print(f"Input: {input_json}")
    print(
        f"Train: {len(rows['train'])} atomic rows from "
        f"{original_examples['train']} examples / {len(train_videos)} videos -> {train_path}"
    )
    print(
        f"Val: {len(rows['val'])} atomic rows from "
        f"{original_examples['val']} examples / {len(val_videos)} videos -> {val_path}"
    )
    print(f"Sources: {dict(source_counts)}")
    print(f"Skipped: {dict(skipped)}")
    print(f"Inner frames per video: {args.nframes}")
    print(
        "Outer query clip: "
        f"{args.query_context_frames} context frames over "
        f"+/-{args.query_context_seconds:g}s, then marked frame and crop"
    )
    return train_path, val_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("rl_dataset"))
    parser.add_argument(
        "--train-name",
        default="open_o3_atomic_stgr_cyclegrpo_train.parquet",
    )
    parser.add_argument(
        "--val-name",
        default="open_o3_atomic_stgr_cyclegrpo_val.parquet",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nframes", type=int, default=32)
    parser.add_argument("--query-context-seconds", type=float, default=1.0)
    parser.add_argument("--query-context-frames", type=int, default=5)
    parser.add_argument("--dedupe-iou", type=float, default=0.95)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-path-validation", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        parser.error("--val-ratio must be in [0, 1).")
    if args.nframes < 2:
        parser.error("--nframes must be at least 2.")
    if args.query_context_seconds <= 0:
        parser.error("--query-context-seconds must be positive.")
    if args.query_context_frames < 3 or args.query_context_frames % 2 == 0:
        parser.error("--query-context-frames must be an odd integer >= 3.")
    if not 0.0 < args.dedupe_iou <= 1.0:
        parser.error("--dedupe-iou must be in (0, 1].")
    return args


if __name__ == "__main__":
    convert(parse_args())
