"""Build single-object sparse-trajectory CycleGRPO data from Open-o3 SFT.

Open-o3's released RL JSON retains only one key frame per spatio-temporal
example, so it cannot define a trajectory. The SFT reasoning traces retain
multiple grounded evidence anchors. This converter conservatively keeps only
one object name repeated over one short continuous window, with one box at
each of 2--5 distinct timestamps. The original question, answer, and
reasoning remain metadata only and never enter the CycleGRPO prompts/reward.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from generalized.verl.utils.spatiotemporal import (
    TRAJECTORY_SPATIOTEMPORAL_SOURCE,
    SpatioTemporalClaim,
    bbox_iou,
    parse_spatiotemporal_claims,
    serialize_spatiotemporal_claims,
)
from projects.rl.datasets.prepare_open_o3_spatiotemporal_cyclegrpo import (
    DEFAULT_DATA_ROOT,
    TARGET_TASK,
    _split_name,
    resolve_video_path,
)


def canonical_object_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().casefold()


def build_trajectory_caption_prompt() -> str:
    """Prompt for chronological marked-frame/crop pairs."""

    return (
        "<video>\n"
        "This query video represents exactly one physical target during one "
        "continuous event. It shows chronological pairs: the full frame with "
        "the target marked by a red rectangle, followed by a close-up of that "
        "same target. Every pair refers to the same target instance. Write one "
        "concise, discriminative paragraph describing only that target's "
        "visible appearance, motion or action, interactions, and useful "
        "relations to nearby objects, so it can be uniquely relocated in the "
        "original video. Do not mention the query, red rectangle, close-up, "
        "coordinates, timestamps, seconds, frame indices, or object tags. "
        "Output the description only."
    )


def extract_single_object_trajectories(
    example: dict[str, Any],
    *,
    min_anchors: int = 2,
    max_anchors: int = 5,
    max_span_seconds: float = 15.0,
    max_gap_seconds: float = 8.0,
    duplicate_iou: float = 0.95,
) -> Iterable[list[SpatioTemporalClaim]]:
    """Yield conservative same-name trajectories from grounded reasoning."""

    grouped: dict[str, list[SpatioTemporalClaim]] = defaultdict(list)
    for claim in parse_spatiotemporal_claims(
        example.get("reasoning_process")
    ):
        name = canonical_object_name(claim.object_name)
        if name:
            grouped[name].append(claim)

    for object_name in sorted(grouped):
        by_timestamp: dict[float, list[tuple[float, float, float, float]]] = (
            defaultdict(list)
        )
        display_name = grouped[object_name][0].object_name.strip()
        for claim in grouped[object_name]:
            by_timestamp[round(float(claim.timestamp), 6)].extend(claim.boxes)

        anchors = []
        ambiguous = False
        for timestamp in sorted(by_timestamp):
            unique_boxes = []
            for box in by_timestamp[timestamp]:
                if not any(
                    bbox_iou(box, previous) >= duplicate_iou
                    for previous in unique_boxes
                ):
                    unique_boxes.append(box)
            if len(unique_boxes) != 1:
                ambiguous = True
                break
            anchors.append(
                SpatioTemporalClaim(
                    object_name=display_name,
                    timestamp=timestamp,
                    boxes=(unique_boxes[0],),
                )
            )
        if ambiguous or not min_anchors <= len(anchors) <= max_anchors:
            continue

        timestamps = [claim.timestamp for claim in anchors]
        if timestamps[-1] - timestamps[0] > max_span_seconds:
            continue
        if max(
            right - left
            for left, right in zip(timestamps, timestamps[1:])
        ) > max_gap_seconds:
            continue
        yield anchors


def convert(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = args.data_root.resolve()
    input_json = (
        args.input_json
        or data_root / "json_data" / "STGR-SFT.json"
    )
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

        trajectories = list(
            extract_single_object_trajectories(
                example,
                min_anchors=args.min_anchors,
                max_anchors=args.max_anchors,
                max_span_seconds=args.max_span_seconds,
                max_gap_seconds=args.max_gap_seconds,
                duplicate_iou=args.duplicate_iou,
            )
        )
        if not trajectories:
            skipped["no_conservative_trajectory"] += 1
            continue

        split = _split_name(video_key, args.val_ratio, args.seed)
        for trajectory_index, claims in enumerate(trajectories):
            if (
                args.max_samples is not None
                and sum(map(len, rows.values())) >= args.max_samples
            ):
                break
            timestamps = [float(claim.timestamp) for claim in claims]
            boxes = [
                [float(value) for value in claim.boxes[0]]
                for claim in claims
            ]
            rows[split].append(
                {
                    "videos": [video_key],
                    "nframes": [args.nframes],
                    "query_timestamps": timestamps,
                    "query_bboxes": boxes,
                    "cap_problem": build_trajectory_caption_prompt(),
                    "cap_answer": "",
                    "seg_problem": "",
                    "seg_answer": serialize_spatiotemporal_claims(claims),
                    "masks": None,
                    "source": TRAJECTORY_SPATIOTEMPORAL_SOURCE,
                    "extra_info": {
                        "open_o3_id": str(example.get("id") or ""),
                        "open_o3_source": source,
                        "trajectory_index": trajectory_index,
                        "target_object": claims[0].object_name,
                        "query_timestamps_json": json.dumps(timestamps),
                        "query_bboxes_json": json.dumps(boxes),
                        "question": str(example.get("question") or ""),
                        "answer": str(example.get("answer") or ""),
                        "video_path": str(example.get("video_path") or ""),
                    },
                }
            )
            source_counts[source] += 1

        if (
            args.max_samples is not None
            and sum(map(len, rows.values())) >= args.max_samples
        ):
            break

    if not rows["train"]:
        raise RuntimeError("No trajectory training rows were produced.")
    if not rows["val"]:
        fallback_video = rows["train"][-1]["videos"][0]
        moved = [
            row
            for row in rows["train"]
            if row["videos"][0] == fallback_video
        ]
        rows["train"] = [
            row
            for row in rows["train"]
            if row["videos"][0] != fallback_video
        ]
        rows["val"].extend(moved)

    train_videos = {row["videos"][0] for row in rows["train"]}
    val_videos = {row["videos"][0] for row in rows["val"]}
    if train_videos & val_videos:
        raise AssertionError("Video leakage across trajectory splits.")

    for split_rows in rows.values():
        for row in split_rows:
            claims = parse_spatiotemporal_claims(row["seg_answer"])
            names = {
                canonical_object_name(claim.object_name)
                for claim in claims
            }
            if (
                not args.min_anchors <= len(claims) <= args.max_anchors
                or len(names) != 1
                or any(len(claim.boxes) != 1 for claim in claims)
            ):
                raise AssertionError("Non-trajectory row escaped filtering.")

    train_path = output_dir / args.train_name
    val_path = output_dir / args.val_name
    pd.DataFrame(rows["train"]).to_parquet(train_path, index=False)
    pd.DataFrame(rows["val"]).to_parquet(val_path, index=False)

    print(f"Input: {input_json}")
    print(
        f"Train: {len(rows['train'])} trajectories / "
        f"{len(train_videos)} videos -> {train_path}"
    )
    print(
        f"Val: {len(rows['val'])} trajectories / "
        f"{len(val_videos)} videos -> {val_path}"
    )
    print(f"Sources: {dict(source_counts)}")
    print(f"Skipped: {dict(skipped)}")
    print(
        "Filter: "
        f"{args.min_anchors}-{args.max_anchors} anchors, "
        f"span<={args.max_span_seconds:g}s, "
        f"gap<={args.max_gap_seconds:g}s"
    )
    return train_path, val_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("rl_dataset"))
    parser.add_argument(
        "--train-name",
        default="open_o3_trajectory_stgr_cyclegrpo_train.parquet",
    )
    parser.add_argument(
        "--val-name",
        default="open_o3_trajectory_stgr_cyclegrpo_val.parquet",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nframes", type=int, default=32)
    parser.add_argument("--min-anchors", type=int, default=2)
    parser.add_argument("--max-anchors", type=int, default=5)
    parser.add_argument("--max-span-seconds", type=float, default=15.0)
    parser.add_argument("--max-gap-seconds", type=float, default=8.0)
    parser.add_argument("--duplicate-iou", type=float, default=0.95)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-path-validation", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        parser.error("--val-ratio must be in [0, 1).")
    if args.nframes < 2:
        parser.error("--nframes must be at least 2.")
    if not 2 <= args.min_anchors <= args.max_anchors <= 5:
        parser.error("Require 2 <= min_anchors <= max_anchors <= 5.")
    if args.max_span_seconds <= 0 or args.max_gap_seconds <= 0:
        parser.error("Trajectory span/gap limits must be positive.")
    if not 0.0 <= args.duplicate_iou <= 1.0:
        parser.error("--duplicate-iou must be in [0, 1].")
    return args


if __name__ == "__main__":
    convert(parse_args())
