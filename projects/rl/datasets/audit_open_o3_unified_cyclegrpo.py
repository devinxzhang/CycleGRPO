"""Audit the materialized Open-o3 unified CycleGRPO RL parquet.

This audit is intentionally independent from the dataset converter: rows are
validated from the serialized training inputs and from media metadata rather
than being accepted merely because the converter emitted them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from generalized.verl.utils.spatiotemporal import parse_spatiotemporal_claims


INTERVAL_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*seconds",
    flags=re.IGNORECASE,
)
BBOX_RE = re.compile(
    r"^\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*$"
)


def media_path(row: pd.Series) -> str:
    values = row.get("images")
    if values is None:
        values = row.get("videos")
    if values is None or len(values) != 1:
        return ""
    return str(values[0])


def parse_interval(text: str) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b in INTERVAL_RE.findall(str(text))]


def parse_bbox(text: str) -> tuple[float, float, float, float] | None:
    match = BBOX_RE.match(str(text))
    if not match:
        return None
    return tuple(float(value) for value in match.groups())


def sampled_times(info: dict[str, Any], sample_fps: float = 2.0) -> list[float]:
    """Reproduce qwen-vl-utils' 2 FPS, 4--16 frame linspace sampler."""

    duration = float(info["duration"])
    if not math.isfinite(duration) or duration <= 0:
        return []
    native_fps = float(info.get("fps") or 0)
    total_frames = int(info.get("total_frames") or 0)
    if native_fps <= 0 or total_frames <= 0:
        native_fps = 1.0
        total_frames = max(2, round(duration))
    nframes = int(min(max(total_frames / native_fps * sample_fps, 4), 16, total_frames))
    nframes -= nframes % 2
    nframes = max(2, nframes)
    indices = [
        round(index * (total_frames - 1) / (nframes - 1))
        for index in range(nframes)
    ]
    return [index / native_fps for index in indices]


def ffprobe(path: str) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,duration,nb_frames,avg_frame_rate:format=duration",
        "-of",
        "json",
        path,
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip()[-1000:]}
    try:
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [])[0]
        raw_duration = (payload.get("format") or {}).get("duration")
        if raw_duration in {None, "N/A"}:
            raw_duration = stream.get("duration")
        duration = float(raw_duration)
        width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
        rate = str(stream.get("avg_frame_rate") or "0/1")
        numerator, denominator = rate.split("/", 1)
        denominator_value = float(denominator)
        fps = float(numerator) / denominator_value if denominator_value else 0.0
        raw_frames = stream.get("nb_frames")
        total_frames = (
            int(raw_frames)
            if raw_frames not in {None, "N/A"}
            else round(duration * fps)
        )
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"metadata:{type(exc).__name__}:{exc}"}
    return {
        "ok": math.isfinite(duration) and duration > 0 and width > 0 and height > 0,
        "duration": duration,
        "width": width,
        "height": height,
        "codec": stream.get("codec_name"),
        "total_frames": total_frames,
        "fps": fps,
    }


def verify_image(path: str) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        return {"ok": width > 0 and height > 0, "width": width, "height": height}
    except Exception as exc:  # PIL raises several backend-specific exceptions.
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}"}


def verify_video_frames(path: str, target_times: list[float]) -> dict[str, Any]:
    """Decode every frame the audit needs through the training-time backend."""

    try:
        from decord import VideoReader

        reader = VideoReader(path, num_threads=1)
        total_frames = len(reader)
        fps = float(reader.get_avg_fps())
        if total_frames <= 0 or not math.isfinite(fps) or fps <= 0:
            return {"ok": False, "error": "invalid_decord_metadata"}
        # qwen-vl-utils uniformly requests at most 16 frames. Add every
        # annotated boundary/center/key time so corrupt target frames cannot
        # hide behind otherwise valid container metadata.
        uniform_count = min(16, total_frames)
        uniform = {
            round(index * (total_frames - 1) / max(uniform_count - 1, 1))
            for index in range(uniform_count)
        }
        targets = {
            min(total_frames - 1, max(0, round(float(value) * fps)))
            for value in target_times
            if math.isfinite(float(value))
        }
        indices = sorted(uniform | targets)
        for start in range(0, len(indices), 64):
            reader.get_batch(indices[start : start + 64]).asnumpy()
        return {
            "ok": True,
            "fps": fps,
            "total_frames": total_frames,
            "decoded_frames": len(indices),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}"}


def check_media(paths: list[str], worker, workers: int, label: str) -> dict[str, dict]:
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, path): path for path in paths}
        for index, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                results[path] = future.result()
            except Exception as exc:
                results[path] = {"ok": False, "error": f"worker:{type(exc).__name__}:{exc}"}
            if index % 500 == 0 or index == len(paths):
                print(f"{label}: {index}/{len(paths)}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet", type=Path, default=Path("rl_dataset/open_o3_unified_cyclegrpo_train.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("rl_dataset/open_o3_unified_cyclegrpo_audit.json")
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--decode-videos",
        action="store_true",
        help="Decode uniform and annotated frames with decord, not just ffprobe metadata.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    df["_media"] = df.apply(media_path, axis=1)
    issues: dict[str, list[int]] = defaultdict(list)

    # Model-equivalent duplicates: metadata fields ignored by training do not
    # distinguish rows. Keep the first occurrence deterministically.
    # The outer model sees only media + cap_problem. For the ST route object
    # labels in seg_answer are ignored by the category-agnostic geometric
    # reward, so rows that differ only by such a label are duplicates too.
    signatures = list(zip(df["cyclegrpo_route"], df["_media"], df["cap_problem"]))
    seen: dict[tuple, int] = {}
    duplicate_of: dict[int, int] = {}
    for index, signature in enumerate(signatures):
        if signature in seen:
            duplicate_of[index] = seen[signature]
            issues["duplicate_model_equivalent"].append(index)
        else:
            seen[signature] = index

    for index, row in df.iterrows():
        route = row["cyclegrpo_route"]
        if not row["_media"]:
            issues["bad_media_cardinality"].append(index)
        if str(row.get("cap_answer") or "").strip():
            issues["nonempty_caption_ground_truth"].append(index)
        if route == "spatial":
            box = parse_bbox(row["seg_answer"])
            if box is None:
                issues["spatial_unparseable_bbox"].append(index)
            else:
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                    issues["spatial_invalid_bbox"].append(index)
                area = (x2 - x1) * (y2 - y1) / 1_000_000
                if area < 0.001:
                    issues["spatial_tiny_bbox_lt_0.1pct"].append(index)
                if area > 0.95:
                    issues["spatial_huge_bbox_gt_95pct"].append(index)
        elif route == "temporal":
            intervals = parse_interval(row["seg_answer"])
            if not intervals:
                issues["temporal_unparseable_interval"].append(index)
            elif any(not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b) for a, b in intervals):
                issues["temporal_invalid_interval"].append(index)
        elif route == "spatial-temporal":
            claims = parse_spatiotemporal_claims(row["seg_answer"])
            if not claims:
                issues["st_unparseable_claims"].append(index)
            for claim in claims:
                for x1, y1, x2, y2 in claim.boxes:
                    area = (x2 - x1) * (y2 - y1) / 1_000_000
                    if area < 0.001:
                        issues["st_tiny_bbox_lt_0.1pct"].append(index)
                        break
                    if area > 0.95:
                        issues["st_huge_bbox_gt_95pct"].append(index)
                        break
        else:
            issues["unknown_route"].append(index)

    image_paths = sorted(set(df.loc[df["cyclegrpo_route"] == "spatial", "_media"]))
    video_paths = sorted(set(df.loc[df["cyclegrpo_route"] != "spatial", "_media"]))
    image_info = check_media(image_paths, verify_image, args.workers, "images")
    video_info = check_media(video_paths, ffprobe, args.workers, "videos")
    video_decode_info: dict[str, dict] = {}
    if args.decode_videos:
        target_times: dict[str, list[float]] = defaultdict(list)
        for _, row in df.loc[df["cyclegrpo_route"] != "spatial"].iterrows():
            if row["cyclegrpo_route"] == "temporal":
                for start, end in parse_interval(row["seg_answer"]):
                    target_times[row["_media"]].extend([start, (start + end) / 2, end])
            else:
                target_times[row["_media"]].extend(
                    claim.timestamp for claim in parse_spatiotemporal_claims(row["seg_answer"])
                )
        video_decode_info = check_media(
            video_paths,
            lambda path: verify_video_frames(path, target_times[path]),
            args.workers,
            "video-decode",
        )

    for index, row in df.iterrows():
        route, path = row["cyclegrpo_route"], row["_media"]
        info = image_info.get(path) if route == "spatial" else video_info.get(path)
        if not info or not info.get("ok"):
            issues[f"{route}_media_unreadable"].append(index)
            continue
        if (
            route != "spatial"
            and args.decode_videos
            and not video_decode_info.get(path, {}).get("ok")
        ):
            issues[f"{route}_media_decode_failed"].append(index)
        if route == "spatial":
            continue
        duration = float(info["duration"])
        samples = sampled_times(info)
        if route == "temporal":
            intervals = parse_interval(row["seg_answer"])
            if any(a >= duration or b > duration + 0.5 for a, b in intervals):
                issues["temporal_interval_outside_video"].append(index)
            coverage = sum(max(0.0, min(b, duration) - a) for a, b in intervals) / duration
            if coverage > 0.90:
                issues["temporal_interval_covers_gt_90pct_video"].append(index)
            if coverage > 0.95:
                issues["temporal_interval_covers_gt_95pct_video"].append(index)
            if intervals and not any(a <= t <= min(b, duration) for a, b in intervals for t in samples):
                issues["temporal_no_uniform_sample_in_interval"].append(index)
        else:
            claims = parse_spatiotemporal_claims(row["seg_answer"])
            if any(claim.timestamp > duration + 0.5 for claim in claims):
                issues["st_timestamp_outside_video"].append(index)
            if claims and samples:
                worst_distance = max(min(abs(t - claim.timestamp) for t in samples) for claim in claims)
                if worst_distance > 1.0:
                    issues["st_nearest_uniform_sample_gt_1s"].append(index)
                if worst_distance > 3.0:
                    issues["st_nearest_uniform_sample_gt_3s"].append(index)
                if worst_distance > 5.0:
                    issues["st_nearest_uniform_sample_gt_5s"].append(index)

    route_counts = df["cyclegrpo_route"].value_counts().to_dict()
    issue_summary = {}
    for name, indices in sorted(issues.items()):
        unique_indices = sorted(set(indices))
        issue_summary[name] = {
            "rows": len(unique_indices),
            "indices": unique_indices,
            "by_route": dict(Counter(df.iloc[index]["cyclegrpo_route"] for index in unique_indices)),
            "by_source": dict(
                Counter(
                    df.iloc[index]["extra_info"]["open_o3_source"]
                    for index in unique_indices
                )
            ),
            "example_indices": unique_indices[:20],
            "example_ids": [df.iloc[index]["extra_info"]["open_o3_id"] for index in unique_indices[:20]],
        }

    hard_filter_reasons = {
        "duplicate_model_equivalent",
        "bad_media_cardinality",
        "nonempty_caption_ground_truth",
        "spatial_unparseable_bbox",
        "spatial_invalid_bbox",
        "temporal_unparseable_interval",
        "temporal_invalid_interval",
        "st_unparseable_claims",
        "spatial_media_unreadable",
        "temporal_media_unreadable",
        "spatial-temporal_media_unreadable",
        "temporal_media_decode_failed",
        "spatial-temporal_media_decode_failed",
        "temporal_interval_outside_video",
        "st_timestamp_outside_video",
        "unknown_route",
    }
    hard_indices = sorted(
        set().union(*(set(issues.get(reason, [])) for reason in hard_filter_reasons))
    )
    report = {
        "parquet": str(args.parquet.resolve()),
        "rows": len(df),
        "route_counts": route_counts,
        "unique_images": len(image_paths),
        "unique_videos": len(video_paths),
        "hard_filter_rows": len(hard_indices),
        "hard_filter_by_route": dict(Counter(df.iloc[index]["cyclegrpo_route"] for index in hard_indices)),
        "hard_filter_reasons": sorted(hard_filter_reasons),
        "issues": issue_summary,
        "notes": {
            "uniform_sampling": "estimated 2 FPS clamped to 4--16 frames, matching the launcher",
            "decode_scope": (
                "images fully decoded with PIL; videos decoded at uniform and annotated frames with decord"
                if args.decode_videos
                else "images fully decoded with PIL; videos validated by ffprobe stream metadata"
            ),
            "soft_flags": "tiny/huge boxes and uniform-sampling coverage require policy judgement",
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
