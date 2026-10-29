"""Materialize the independently audited Open-o3 CycleGRPO clean-v2 set.

This script never overwrites the original 22,227-row parquet.  Filtering is
driven by the complete audit artifact, while all expected counts and route
unions are asserted so a stale audit cannot silently produce a new mixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from generalized.verl.utils.spatiotemporal import parse_spatiotemporal_claims
from projects.rl.datasets.prepare_open_o3_unified_cyclegrpo import (
    temporal_caption_prompt,
)


EXPECTED_INPUT_ROWS = 22_227
EXPECTED_OUTPUT_COUNTS = {
    "spatial": 4_143,
    "temporal": 4_820,
    "spatial-temporal": 11_924,
}
SEVERE_TEMPORAL_IDS = {"timerft_201", "timerft_227", "timerft_782"}
INTERVAL_RE = re.compile(
    r"(?P<start>-?\d+(?:\.\d+)?)\s*-\s*(?P<end>-?\d+(?:\.\d+)?)\s*seconds"
)


def _indices(audit: dict[str, Any], issue: str) -> set[int]:
    return {int(value) for value in audit["issues"][issue]["indices"]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extra_info(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _final_frame_timestamp(video: str) -> float:
    from decord import VideoReader

    reader = VideoReader(video)
    fps = float(reader.get_avg_fps())
    if len(reader) <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video metadata: {video}")
    return (len(reader) - 1) / fps


def _repair_temporal_endpoint(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    matches = list(INTERVAL_RE.finditer(str(row["seg_answer"])))
    if not matches:
        raise ValueError("Cannot parse temporal interval for endpoint repair")
    duration = _final_frame_timestamp(str(row["videos"][0]))
    repaired = []
    for match in matches:
        start = float(match.group("start"))
        end = min(float(match.group("end")), duration)
        # Floor rather than round so the serialized endpoint never exceeds the
        # last real frame timestamp.
        end = math.floor(end * 10.0) / 10.0
        if end <= start:
            raise ValueError(
                f"Endpoint repair collapsed interval {start}--{end} for {row['videos'][0]}"
            )
        repaired.append(f"{start:.1f} - {end:.1f} seconds")
    answer = ", ".join(repaired)
    return answer, {
        "original_seg_answer": str(row["seg_answer"]),
        "final_frame_timestamp": duration,
        "repaired_seg_answer": answer,
    }


def _attach_joint_st_fields(row: dict[str, Any], distractor_count: int) -> None:
    claims = parse_spatiotemporal_claims(str(row["seg_answer"]))
    if not claims:
        raise ValueError("Clean-v2 ST row has no parseable claim")
    timestamps: list[float] = []
    bboxes: list[list[float]] = []
    for claim in claims:
        for box in claim.boxes:
            timestamps.append(float(claim.timestamp))
            bboxes.append([float(value) for value in box])
    row["query_timestamps"] = timestamps
    row["query_bboxes"] = bboxes
    row["inner_distractor_count"] = int(distractor_count)
    row["nframes"] = [16]


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    audit_path = args.audit.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    frame = pd.read_parquet(input_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if len(frame) != EXPECTED_INPUT_ROWS or audit.get("rows") != EXPECTED_INPUT_ROWS:
        raise AssertionError("Input parquet/audit is not the audited 22,227-row version")
    if Path(audit["parquet"]).resolve() != input_path:
        raise AssertionError("Audit artifact points to a different input parquet")
    audited_hash = audit.get("parquet_sha256")
    if not audited_hash or audited_hash != _sha256(input_path):
        raise AssertionError("Input parquet contents differ from the audited version")

    duplicate = _indices(audit, "duplicate_model_equivalent")
    remove_by_reason: dict[str, set[int]] = {
        "duplicate_model_equivalent": duplicate,
        "spatial_tiny_bbox_lt_0.1pct": _indices(
            audit, "spatial_tiny_bbox_lt_0.1pct"
        ),
        "spatial_huge_bbox_gt_95pct": _indices(
            audit, "spatial_huge_bbox_gt_95pct"
        ),
        "temporal_no_uniform_sample_in_interval": _indices(
            audit, "temporal_no_uniform_sample_in_interval"
        ),
        "temporal_interval_covers_gt_90pct_video": _indices(
            audit, "temporal_interval_covers_gt_90pct_video"
        ),
        "st_timestamp_outside_video": _indices(
            audit, "st_timestamp_outside_video"
        ),
        "st_huge_bbox_gt_95pct": _indices(audit, "st_huge_bbox_gt_95pct"),
    }

    severe_indices = {
        int(index)
        for index, row in frame.iterrows()
        if _extra_info(row.get("extra_info")).get("open_o3_id")
        in SEVERE_TEMPORAL_IDS
    }
    remove_by_reason["temporal_endpoint_exceeds_video_gt_5s"] = severe_indices

    route_remove: dict[str, set[int]] = {
        "spatial": set(),
        "temporal": set(),
        "spatial-temporal": set(),
    }
    for reason, indices in remove_by_reason.items():
        for index in indices:
            route = str(frame.at[index, "cyclegrpo_route"])
            # Route-specific issue sets are enforced here; the duplicate set is
            # intentionally shared and split by the actual row route.
            if reason.startswith("spatial_") and route != "spatial":
                continue
            if reason.startswith("temporal_") and route != "temporal":
                continue
            if reason.startswith("st_") and route != "spatial-temporal":
                continue
            route_remove[route].add(index)

    expected_removed = {
        "spatial": 855,
        "temporal": 364,
        "spatial-temporal": 121,
    }
    actual_removed = {route: len(indices) for route, indices in route_remove.items()}
    if actual_removed != expected_removed:
        raise AssertionError(
            f"Clean-v2 filter union changed: {actual_removed} != {expected_removed}"
        )

    outside = _indices(audit, "temporal_interval_outside_video")
    repair_indices = outside - severe_indices - route_remove["temporal"]
    # Some endpoint rows may already be removed by another temporal rule.  The
    # remaining audited rows are the ones that must be repaired in-place.
    expected_repair_ids = {
        _extra_info(frame.at[index, "extra_info"]).get("open_o3_id")
        for index in repair_indices
    }

    removed_all = set().union(*route_remove.values())
    rows: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    for index, series in frame.iterrows():
        if index in removed_all:
            continue
        row = series.to_dict()
        route = str(row["cyclegrpo_route"])
        if index in repair_indices:
            answer, repair = _repair_temporal_endpoint(row)
            row["seg_answer"] = answer
            row["cap_problem"] = temporal_caption_prompt(answer)
            repair["index"] = int(index)
            repair["open_o3_id"] = _extra_info(row.get("extra_info")).get(
                "open_o3_id"
            )
            repairs.append(repair)
        if route == "spatial-temporal":
            _attach_joint_st_fields(row, args.distractor_count)
        info = _extra_info(row.get("extra_info"))
        info["cyclegrpo_clean_version"] = "v2"
        info["original_row_index"] = int(index)
        row["extra_info"] = info
        rows.append(row)

    output = pd.DataFrame(rows)
    counts = Counter(output["cyclegrpo_route"].tolist())
    if dict(counts) != EXPECTED_OUTPUT_COUNTS or len(output) != 20_887:
        raise AssertionError(f"Unexpected clean-v2 counts: {dict(counts)}")
    if len(repairs) != len(expected_repair_ids):
        raise AssertionError("Temporal endpoint repair count mismatch")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)
    manifest = {
        "version": "open_o3_unified_cyclegrpo_clean_v2",
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "audit": str(audit_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "input_rows": len(frame),
        "output_rows": len(output),
        "output_route_counts": dict(counts),
        "removed_route_counts": actual_removed,
        "removed_reason_raw_counts": {
            reason: len(indices) for reason, indices in remove_by_reason.items()
        },
        "temporal_endpoint_repairs": repairs,
        "temporal_repair_ids": sorted(expected_repair_ids),
        "st_outer_interface": "exact target frame with all boxes marked plus one crop per box",
        "st_inner_interface": (
            "16 uniform unmarked frames plus exact target frame(s) and "
            f"{args.distractor_count} deterministic unmarked distractor frames"
        ),
        "original_parquet_modified": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("rl_dataset/open_o3_unified_cyclegrpo_train.parquet"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("rl_dataset/open_o3_unified_cyclegrpo_audit_full.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl_dataset/open_o3_unified_cyclegrpo_clean_v2_train.parquet"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("rl_dataset/open_o3_unified_cyclegrpo_clean_v2_manifest.json"),
    )
    parser.add_argument("--distractor-count", type=int, default=4)
    args = parser.parse_args()
    if args.distractor_count < 2:
        parser.error("--distractor-count must be at least 2")
    return args


if __name__ == "__main__":
    build(parse_args())
