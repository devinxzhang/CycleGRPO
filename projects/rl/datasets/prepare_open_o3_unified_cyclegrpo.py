"""Build the unified CycleGRPO training mixture from Open-o3 RL data.

The released ``STGR-RL.json`` contains three localization supervision levels
used by this CycleGRPO experiment:

* image visual QA with bounding boxes -> spatial CycleGRPO;
* video temporal QA with intervals -> temporal CycleGRPO;
* video QA with timestamped boxes -> spatio-temporal CycleGRPO;

No reference caption is exposed to any cycle row. General video QA (both MCQ
and free-form) is intentionally excluded so that gains cannot be attributed to
an auxiliary understanding objective or an LLM judge. By default all valid
examples are written to train; an optional nonzero
``--val-ratio`` enables a deterministic asset-grouped validation split.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from projects.rl.datasets.prepare_open_o3_spatiotemporal_cyclegrpo import (
    DEFAULT_DATA_ROOT,
    _split_name,
    build_claims,
    resolve_video_path as resolve_stgr_video_path,
)
from generalized.verl.utils.spatiotemporal import (
    SPATIOTEMPORAL_SOURCE,
    serialize_spatiotemporal_claims,
)
from generalized.verl.utils.cyclegrpo_profiles import VIDEO_CAPTION_ANTI_TIME_INSTRUCTION


SPATIAL_TASK = "visual QA"
SPATIOTEMPORAL_TASK = "temporal-spatial free-form QA"
TEMPORAL_TASKS = frozenset({"temporal QA", "temporal QA (MCQ)"})
EXCLUDED_UNDERSTANDING_TASKS = frozenset(
    {"General video QA MCQ", "General video QA Free-form"}
)


def resolve_media_path(data_root: Path, example: dict[str, Any]) -> Path:
    source = str(example.get("source") or "")
    image_path = str(example.get("image_path") or "")
    video_path = str(example.get("video_path") or "")
    videos_root = data_root / "videos"
    if image_path:
        if source != "gqa":
            raise ValueError(f"Unsupported image source: {source}")
        return videos_root / "gqa" / image_path
    if not video_path:
        raise ValueError("Example has neither image_path nor video_path")
    if source.startswith("STR_") or source == "videoespresso_train_video":
        return resolve_stgr_video_path(data_root, source, video_path)
    if source.startswith("TVG_"):
        return videos_root / "tvg_r1" / video_path
    if source == "timerft":
        return videos_root / "timerft" / video_path
    if source.startswith("videor1_"):
        return videos_root / "videor1" / video_path
    raise ValueError(f"Unsupported video source: {source}")


def normalized_bbox(example: dict[str, Any]) -> tuple[str, str]:
    key_items = example.get("key_items") or {}
    if len(key_items) != 1:
        raise ValueError("Spatial rows must contain exactly one target object")
    object_name, raw_boxes = next(iter(key_items.items()))
    if not isinstance(raw_boxes, list) or len(raw_boxes) != 1:
        raise ValueError("Spatial rows must contain exactly one target box")
    width, height = [float(value) for value in example.get("image_size") or []]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image_size")
    x1, y1, x2, y2 = [float(value) for value in raw_boxes[0]]
    box = [
        round(1000.0 * x1 / width),
        round(1000.0 * y1 / height),
        round(1000.0 * x2 / width),
        round(1000.0 * y2 / height),
    ]
    box = [min(1000, max(0, value)) for value in box]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Degenerate normalized bbox")
    return str(object_name).strip() or "target", json.dumps(box)


def temporal_answer(answer: Any) -> str:
    """Extract the released interval and serialize the local TG format."""

    matches = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", str(answer))
    if not matches:
        try:
            value = ast.literal_eval(str(answer).strip())
        except (SyntaxError, ValueError):
            value = None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            matches = [(str(value[0]), str(value[1]))]
    intervals = []
    for raw_start, raw_end in matches:
        start, end = float(raw_start), float(raw_end)
        if start < 0 or end <= start:
            continue
        intervals.append(f"{start:.1f} - {end:.1f} seconds")
    if not intervals:
        raise ValueError("No valid temporal interval in answer")
    return ", ".join(intervals)


def temporal_caption_prompt(interval: str) -> str:
    """Canonical outer caption prompt for temporal CycleGRPO."""

    return (
        "<video>\nThe target event occurs during "
        f"{interval}. Describe only that event precisely enough that another "
        "viewer can localize it again. Do not mention timestamps, seconds, "
        "intervals, or the prompt. Output description only."
        + VIDEO_CAPTION_ANTI_TIME_INSTRUCTION
    )


def spatiotemporal_caption_prompt(
    claims: list,
) -> str:
    """Canonical outer caption prompt for spatio-temporal CycleGRPO."""

    anonymous = [type(claim)("target", claim.timestamp, claim.boxes) for claim in claims]
    support = serialize_spatiotemporal_claims(anonymous, wrap_answer=False)
    return (
        "<video>\nThe target sparse spatio-temporal support is:\n"
        f"{support}\nDescribe only the visible target and event precisely "
        "enough to relocate them. Do not mention object tags, coordinates, "
        "timestamps, seconds, or frame indices. Output description only."
        + VIDEO_CAPTION_ANTI_TIME_INSTRUCTION
    )


def base_metadata(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "open_o3_id": str(example.get("id") or ""),
        "open_o3_source": str(example.get("source") or ""),
        "open_o3_task": str(example.get("task") or ""),
        "question": str(example.get("question") or ""),
        "answer": str(example.get("answer") or ""),
        "image_path": str(example.get("image_path") or ""),
        "video_path": str(example.get("video_path") or ""),
    }


def spatial_row(example: dict[str, Any], media: Path) -> dict[str, Any]:
    object_name, box = normalized_bbox(example)
    return {
        "images": [str(media)],
        "cap_problem": (
            "<image>\nThe target region is marked by the normalized bounding "
            f"box {box}. Describe only the target object precisely enough to "
            "relocate it. Do not mention coordinates, boxes, or the prompt. "
            "Output the description only."
        ),
        "cap_answer": "",
        "seg_problem": "",
        "seg_answer": box,
        "masks": None,
        # Reuse the mature image nested-rollout route in bbox mode.
        "source": "dam_cyclegrpo",
        "extra_info": {**base_metadata(example), "target_object": object_name},
    }


def temporal_row(example: dict[str, Any], media: Path) -> dict[str, Any]:
    interval = temporal_answer(example.get("answer"))
    return {
        "videos": [str(media)],
        "cap_problem": temporal_caption_prompt(interval),
        "cap_answer": "",
        "seg_problem": "",
        "seg_answer": interval,
        "masks": None,
        "source": "tg_multi_merged",
        "extra_info": base_metadata(example),
    }


def spatiotemporal_row(example: dict[str, Any], media: Path) -> dict[str, Any]:
    claims = build_claims(example)
    if not claims:
        raise ValueError("No valid timestamped bbox claims")
    return {
        "videos": [str(media)],
        "cap_problem": spatiotemporal_caption_prompt(claims),
        "cap_answer": "",
        "seg_problem": "",
        "seg_answer": serialize_spatiotemporal_claims(claims),
        "masks": None,
        "source": SPATIOTEMPORAL_SOURCE,
        "extra_info": base_metadata(example),
    }


def convert(args: argparse.Namespace) -> tuple[Path, Path | None]:
    data_root = args.data_root.resolve()
    input_json = args.input_json or data_root / "json_data" / "STGR-RL.json"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = json.loads(input_json.read_text(encoding="utf-8"))

    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    route_counts = Counter()
    source_counts = Counter()
    skipped = Counter()
    checked_paths: dict[str, bool] = {}
    builders = {
        SPATIAL_TASK: ("spatial", lambda ex, path: spatial_row(ex, path)),
        SPATIOTEMPORAL_TASK: (
            "spatial-temporal",
            lambda ex, path: spatiotemporal_row(ex, path),
        ),
    }

    for example in examples:
        task = str(example.get("task") or "")
        if task in TEMPORAL_TASKS:
            route, builder = "temporal", lambda ex, path: temporal_row(ex, path)
        elif task in EXCLUDED_UNDERSTANDING_TASKS:
            skipped["excluded_understanding"] += 1
            continue
        elif task in builders:
            route, builder = builders[task]
        else:
            skipped["unsupported_task"] += 1
            continue
        try:
            # ``data_root`` is already absolute. Avoid resolving every media
            # path: on the shared filesystem that performs tens of thousands
            # of unnecessary metadata lookups even when validation is skipped.
            media = resolve_media_path(data_root, example)
            media_key = str(media)
            if not args.skip_path_validation:
                if not checked_paths.setdefault(media_key, media.is_file()):
                    skipped[f"missing_{route}_media"] += 1
                    continue
            row = builder(example, media)
        except (TypeError, ValueError) as exc:
            skipped[f"invalid_{route}:{type(exc).__name__}"] += 1
            continue
        row["cyclegrpo_route"] = route
        split = (
            _split_name(media_key, args.val_ratio, args.seed)
            if args.val_ratio > 0
            else "train"
        )
        rows[split].append(row)
        route_counts[route] += 1
        source_counts[str(example.get("source") or "")] += 1

    if not rows["train"]:
        raise RuntimeError("Training split must be non-empty")
    train_assets = {
        (row.get("images") or row.get("videos"))[0] for row in rows["train"]
    }
    val_assets = {
        (row.get("images") or row.get("videos"))[0] for row in rows["val"]
    }
    if train_assets & val_assets:
        raise AssertionError("Visual asset leakage across train/validation")

    train_path = output_dir / args.train_name
    pd.DataFrame(rows["train"]).to_parquet(train_path, index=False)
    val_path = None
    if rows["val"]:
        val_path = output_dir / args.val_name
        pd.DataFrame(rows["val"]).to_parquet(val_path, index=False)
    route_files: dict[str, dict[str, str]] = {}
    for route in ("spatial", "temporal", "spatial-temporal"):
        route_files[route] = {}
        safe_route = route.replace("-", "_")
        splits = ("train", "val") if rows["val"] else ("train",)
        for split in splits:
            subset = [row for row in rows[split] if row["cyclegrpo_route"] == route]
            route_path = output_dir / f"open_o3_{safe_route}_{split}.parquet"
            pd.DataFrame(subset).to_parquet(route_path, index=False)
            route_files[route][split] = str(route_path)
    manifest = {
        "input": str(input_json),
        "train_rows": len(rows["train"]),
        "val_rows": len(rows["val"]),
        "route_counts": dict(route_counts),
        "source_counts": dict(source_counts),
        "skipped": dict(skipped),
        "train_assets": len(train_assets),
        "val_assets": len(val_assets),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "video_sampling": "2 FPS from the training config; no fixed nframes in rows",
        "route_files": route_files,
    }
    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Train -> {train_path}")
    if val_path is not None:
        print(f"Val   -> {val_path}")
    print(f"Audit -> {manifest_path}")
    return train_path, val_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("rl_dataset"))
    parser.add_argument("--train-name", default="open_o3_unified_cyclegrpo_train.parquet")
    parser.add_argument("--val-name", default="open_o3_unified_cyclegrpo_val.parquet")
    parser.add_argument("--manifest-name", default="open_o3_unified_cyclegrpo_manifest.json")
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-path-validation", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        parser.error("--val-ratio must be in [0, 1)")
    return args


if __name__ == "__main__":
    convert(parse_args())
