#!/usr/bin/env python3
"""Attach transparent hard-case sampling weights to the clean VQA mix.

No rows or labels are invented. The output has exactly the input rows plus two
audit columns. Training uses a WeightedRandomSampler, so repeated exposure is
explicitly sampling policy rather than being presented as additional data.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


INTERVAL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–—~]\s*(\d+(?:\.\d+)?)\s*seconds?", re.I)
MOTION_RE = re.compile(
    r"\b(?:move|moving|walk|walking|run|running|enter|leave|approach|depart|"
    r"turn|turning|cross|passing|fall|jump|throw|catch|open|close|pick|put|"
    r"place|remove|lift|lower|pour|cut|slice|transition|change)\w*\b", re.I
)
ORDER_RE = re.compile(
    r"\b(?:before|after|then|next|first|last|until|while|when|prior|following)\b", re.I
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--duration-cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    return parser.parse_args()


def probe_duration(path: str) -> tuple[str, float | None]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path,
            ],
            text=True, capture_output=True, timeout=20, check=True,
        )
        value = float(result.stdout.strip())
        return path, value if value > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return path, None


def load_duration_cache(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def target_interval(row: dict) -> tuple[float, float] | None:
    timestamps = row.get("query_timestamps") or []
    if timestamps:
        values = [float(value) for value in timestamps]
        return min(values), max(values)
    match = INTERVAL_RE.search(str(row.get("seg_answer") or ""))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


def main() -> None:
    args = parse_args()
    table = pq.read_table(args.input)
    rows = table.to_pylist()
    video_paths = sorted({
        str((row.get("videos") or [""])[0])
        for row in rows
        if row.get("cyclegrpo_route") in {"temporal", "spatial-temporal"}
        and (row.get("videos") or [""])[0]
    })
    durations = load_duration_cache(args.duration_cache)
    missing = [path for path in video_paths if path not in durations]
    if missing:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for path, duration in pool.map(probe_duration, missing):
                durations[path] = duration
        args.duration_cache.parent.mkdir(parents=True, exist_ok=True)
        args.duration_cache.write_text(
            json.dumps(durations, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )

    weights: list[float] = []
    all_tags: list[list[str]] = []
    source_counts = Counter()
    tag_counts = Counter()
    weighted_mass = defaultdict(float)
    for row in rows:
        route = str(row.get("cyclegrpo_route") or "")
        source_counts[route] += 1
        tags: list[str] = []
        # Preserve VQA for capability retention, but make localization the
        # majority of expected draws. Spatial-only remains useful but is not
        # the target of this temporal ablation.
        weight = {
            "understanding": 0.80,
            "spatial": 1.00,
            "temporal": 1.25,
            "spatial-temporal": 1.00,
        }.get(route, 1.0)

        query = str((row.get("extra_info") or {}).get("question") or "")
        if MOTION_RE.search(query):
            tags.append("motion")
        if ORDER_RE.search(query):
            tags.append("before_after_or_order")
        if route == "spatial-temporal":
            timestamps = [float(value) for value in (row.get("query_timestamps") or [])]
            if len(set(timestamps)) <= 2:
                tags.append("sparse_temporal_support")

        path = str((row.get("videos") or [""])[0])
        duration = durations.get(path)
        interval = target_interval(row)
        if duration and interval:
            start, end = interval
            center_ratio = ((start + end) / 2.0) / duration
            target_duration = max(end - start, 0.0)
            target_ratio = target_duration / duration
            if duration >= 120:
                tags.append("long_video")
            if center_ratio >= 2 / 3:
                tags.append("late_target")
            if target_ratio <= 0.10:
                tags.append("short_relative_window")
            if route == "temporal" and target_duration <= 4:
                tags.append("short_absolute_window")
        elif route in {"temporal", "spatial-temporal"}:
            tags.append("duration_unavailable")

        genuine_hard_tags = [tag for tag in tags if tag != "duration_unavailable"]
        if route in {"temporal", "spatial-temporal"}:
            # Several simultaneous difficulties receive more exposure, with a
            # cap so a tiny tail cannot dominate an epoch.
            weight *= min(1.0 + 0.25 * len(genuine_hard_tags), 2.0)
        weight = round(weight, 4)
        weights.append(weight)
        all_tags.append(tags)
        weighted_mass[route] += weight
        tag_counts.update(tags)

    output = table.append_column("sampling_weight", pa.array(weights, type=pa.float32()))
    output = output.append_column("hard_case_tags", pa.array(all_tags, type=pa.list_(pa.string())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output, args.output, compression="zstd")

    total_mass = sum(weighted_mass.values())
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "rows": len(rows),
        "row_count_unchanged": len(rows) == table.num_rows,
        "source_rows": dict(source_counts),
        "hard_tag_counts": dict(tag_counts.most_common()),
        "duration_videos": len(video_paths),
        "duration_probe_failures": sum(durations.get(path) is None for path in video_paths),
        "sampling_weight": {
            "min": min(weights), "mean": sum(weights) / len(weights), "max": max(weights),
        },
        "expected_route_fraction": {
            key: value / total_mass for key, value in sorted(weighted_mass.items())
        },
        "note": "Weighted replacement sampling; rows and supervision are unchanged.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
