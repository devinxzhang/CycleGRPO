"""Build the opt-in spatio-temporal CycleGRPO parquet from Open-o3-Video.

Only ``temporal-spatial free-form QA`` examples are used.  The Open-o3
question/answer pair is retained as metadata, but no reference caption is
placed in ``cap_answer`` or used by the reward.  The train/validation split is
deterministic and grouped by video to avoid cross-split video leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from generalized.verl.utils.spatiotemporal import (
    SPATIOTEMPORAL_SOURCE,
    SpatioTemporalClaim,
    serialize_spatiotemporal_claims,
)


DEFAULT_DATA_ROOT = Path(
    "data/Open-o3-Video"
)
TARGET_TASK = "temporal-spatial free-form QA"


def resolve_video_path(data_root: Path, source: str, relative_path: str) -> Path:
    """Resolve Open-o3's source-dependent video roots."""

    videos_root = data_root / "videos"
    if source == "videoespresso_train_video":
        base = videos_root / "videoespresso" / "videos"
    elif source == "STR_plm_rdcap":
        base = videos_root / "stgr" / "plm" / "videos"
    elif source.startswith("STR_"):
        base = videos_root / "stgr" / "temporal_grounding" / "videos"
    else:
        raise ValueError(f"Unsupported spatio-temporal source: {source}")
    return base / relative_path


def _canonical_box(raw_box: Any, coordinate_scale: float = 1000.0):
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return None
    try:
        values = [float(value) for value in raw_box]
    except (TypeError, ValueError):
        return None
    if max(abs(value) for value in values) <= 1.5:
        values = [value * coordinate_scale for value in values]
    values = [min(coordinate_scale, max(0.0, value)) for value in values]
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return tuple(round(value) for value in values)


def build_claims(example: dict[str, Any]) -> list[SpatioTemporalClaim]:
    """Convert key_frames/key_items into sparse timestamped box claims."""

    key_items = example.get("key_items") or {}
    claims: list[SpatioTemporalClaim] = []
    frames = sorted(
        example.get("key_frames") or [],
        key=lambda frame: (float(frame.get("time", 0.0)), int(frame.get("idx", 0))),
    )
    for frame in frames:
        objects = key_items.get(str(frame.get("idx"))) or {}
        if not isinstance(objects, dict):
            continue
        for object_name, raw_boxes in sorted(objects.items()):
            boxes = []
            for raw_box in raw_boxes or []:
                box = _canonical_box(raw_box)
                if box is not None:
                    boxes.append(box)
            if boxes:
                claims.append(
                    SpatioTemporalClaim(
                        object_name=str(object_name).strip() or "target",
                        timestamp=float(frame["time"]),
                        boxes=tuple(boxes),
                    )
                )
    return claims


def build_caption_prompt(question: str, claims: list[SpatioTemporalClaim]) -> str:
    """Expose geometry only; semantic object labels are replaced by ``target``."""

    anonymous_claims = [
        SpatioTemporalClaim("target", claim.timestamp, claim.boxes)
        for claim in claims
    ]
    region = serialize_spatiotemporal_claims(anonymous_claims, wrap_answer=False)
    return (
        "<video>\n"
        f"Question context: {question.strip()}\n"
        "The target sparse spatio-temporal support is:\n"
        f"{region}\n"
        "Describe the visual content and event in the target support precisely enough "
        "that another viewer can localize it again. Output description only. Do not "
        "repeat object tags, bounding-box coordinates, timestamps, seconds, or frame indices."
    )


def _split_name(video_path: str, val_ratio: float, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{video_path}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if bucket < val_ratio else "train"


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
    checked_paths: dict[str, bool] = {}

    for example in examples:
        if example.get("task") != TARGET_TASK:
            continue
        if args.max_samples is not None and sum(map(len, rows.values())) >= args.max_samples:
            break

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

        question = str(example.get("question") or "")
        split = _split_name(video_key, args.val_ratio, args.seed)
        row = {
            "videos": [video_key],
            "nframes": [args.nframes],
            "cap_problem": build_caption_prompt(question, claims),
            # CycleGRPO must not receive a reference caption during RL.
            "cap_answer": "",
            # This placeholder is replaced by each sampled outer caption before
            # the inner grounding rollout.
            "seg_problem": question,
            "seg_answer": serialize_spatiotemporal_claims(claims),
            "masks": None,
            "source": SPATIOTEMPORAL_SOURCE,
            "extra_info": {
                "open_o3_id": str(example.get("id") or ""),
                "open_o3_source": source,
                "question": question,
                "answer": str(example.get("answer") or ""),
                "video_path": str(example.get("video_path") or ""),
                "key_frames_json": json.dumps(example.get("key_frames") or [], ensure_ascii=False),
                "key_items_json": json.dumps(example.get("key_items") or {}, ensure_ascii=False),
            },
        }
        rows[split].append(row)
        source_counts[source] += 1

    if not rows["train"]:
        raise RuntimeError("No training rows were produced.")
    if not rows["val"]:
        # The current dataloader requires a non-empty val file even when
        # validation is disabled.  Move one complete video group deterministically.
        fallback_video = rows["train"][-1]["videos"][0]
        moved = [row for row in rows["train"] if row["videos"][0] == fallback_video]
        rows["train"] = [row for row in rows["train"] if row["videos"][0] != fallback_video]
        rows["val"].extend(moved)

    train_path = output_dir / args.train_name
    val_path = output_dir / args.val_name
    pd.DataFrame(rows["train"]).to_parquet(train_path, index=False)
    pd.DataFrame(rows["val"]).to_parquet(val_path, index=False)

    train_videos = {row["videos"][0] for row in rows["train"]}
    val_videos = {row["videos"][0] for row in rows["val"]}
    overlap = train_videos & val_videos
    if overlap:
        raise AssertionError(f"Video leakage across splits: {len(overlap)} videos")

    print(f"Input: {input_json}")
    print(f"Task: {TARGET_TASK}")
    print(f"Train: {len(rows['train'])} rows / {len(train_videos)} videos -> {train_path}")
    print(f"Val: {len(rows['val'])} rows / {len(val_videos)} videos -> {val_path}")
    print(f"Sources: {dict(source_counts)}")
    print(f"Skipped: {dict(skipped)}")
    print(f"Frames per video: {args.nframes}")
    return train_path, val_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("rl_dataset"))
    parser.add_argument(
        "--train-name",
        default="open_o3_stgr_cyclegrpo_train.parquet",
    )
    parser.add_argument(
        "--val-name",
        default="open_o3_stgr_cyclegrpo_val.parquet",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--nframes",
        type=int,
        default=32,
        help="Uniform native-video frames decoded for each outer/inner rollout.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-path-validation", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        parser.error("--val-ratio must be in [0, 1).")
    if args.nframes < 2:
        parser.error("--nframes must be at least 2.")
    return args


if __name__ == "__main__":
    convert(parse_args())
