"""Sparse spatio-temporal region parsing and overlap utilities.

The serialization follows Open-o3-Video's timestamped-box convention::

    <obj>target</obj><box>[x1,y1,x2,y2]</box>at<t>12.5</t>s

Coordinates are represented on a normalized 0--1000 canvas.  For
backward-compatibility with the source annotations, boxes in [0, 1] are
automatically rescaled to that canvas.

CycleGRPO uses the helpers in this module only for the opt-in legacy
``stgr_cyclegrpo``, atomic ``stgr_atomic_cyclegrpo``, and single-object
trajectory ``stgr_trajectory_cyclegrpo`` sources. Existing image-mask and
temporal-interval paths do not call these functions.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

from .cyclegrpo_profiles import (
    ATOMIC_SPATIOTEMPORAL_SOURCE,
    SPATIOTEMPORAL_SOURCE,
    TRAJECTORY_LOCALIZATION_SOURCE,
    TRAJECTORY_SPATIOTEMPORAL_SOURCE,
    TRAJECTORY_UNDERSTANDING_SOURCE,
)

# Used only to remove already-valid claims in strict format checks below.  The
# character classes deliberately cannot cross tag delimiters, so malformed
# model output cannot trigger the nested ``.*?`` backtracking of the old parser.
_CLAIM_PATTERN = re.compile(
    r"<obj>[^<]*</obj>"
    r"(?:\s*<box>\s*\[[^\]<>]*\]\s*</box>)+"
    r"\s*at\s*<t>\s*[-+]?\d+(?:\.\d+)?\s*</t>\s*s",
    flags=re.IGNORECASE,
)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
_THINK_ANSWER_PATTERN = re.compile(
    r"<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)
_TRAILING_GENERATION_TERMINATOR_PATTERN = re.compile(
    r"(?:\s*<\|(?:im_end|endoftext)\|>)+\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SpatioTemporalClaim:
    """One timestamped object claim containing one or more boxes."""

    object_name: str
    timestamp: float
    boxes: tuple[tuple[float, float, float, float], ...]


def _normalize_box(box: Sequence[float], coordinate_scale: float) -> tuple[float, float, float, float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        coords = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in coords):
        return None

    # Open-o3's raw key_items use [0, 1], while Qwen-style boxes commonly use
    # [0, 1000].  Canonicalize both without tying reward to decoded frame size.
    if max(abs(value) for value in coords) <= 1.5:
        coords = [value * coordinate_scale for value in coords]
    coords = [min(coordinate_scale, max(0.0, value)) for value in coords]
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def parse_spatiotemporal_claims(
    text: str | None,
    *,
    only_answer: bool = False,
    coordinate_scale: float = 1000.0,
) -> list[SpatioTemporalClaim]:
    """Parse valid timestamped-box claims from model or ground-truth text."""

    if text is None:
        return []
    content = str(text)
    if only_answer:
        answer_match = _ANSWER_PATTERN.search(content)
        if answer_match:
            content = answer_match.group(1)

    # Do not express the whole claim grammar as one nested, backtracking
    # regular expression. Model generations can contain many unmatched
    # ``<obj>``/``<box>`` fragments; the former pattern then took minutes in
    # ``finditer`` (and blocked an entire distributed rollout rank). A small
    # delimiter scanner is linear in the response length and rejects malformed
    # fragments locally.
    lowered = content.lower()
    claims: list[SpatioTemporalClaim] = []
    search_pos = 0
    max_claims = 256
    max_boxes_per_claim = 256
    while len(claims) < max_claims:
        object_start = lowered.find("<obj>", search_pos)
        if object_start < 0:
            break
        object_body_start = object_start + len("<obj>")
        object_end = lowered.find("</obj>", object_body_start)
        if object_end < 0:
            break

        cursor = object_end + len("</obj>")
        raw_boxes: list[str] = []
        while len(raw_boxes) < max_boxes_per_claim:
            while cursor < len(content) and content[cursor].isspace():
                cursor += 1
            if not lowered.startswith("<box>", cursor):
                break
            box_body_start = cursor + len("<box>")
            box_end = lowered.find("</box>", box_body_start)
            if box_end < 0:
                break
            raw_boxes.append(content[box_body_start:box_end].strip())
            cursor = box_end + len("</box>")

        # Always advance beyond this object on a malformed claim. This is what
        # guarantees progress without regex backtracking.
        search_pos = max(object_body_start, cursor)
        if not raw_boxes:
            continue

        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if not lowered.startswith("at", cursor):
            continue
        cursor += 2
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if not lowered.startswith("<t>", cursor):
            continue
        time_start = cursor + len("<t>")
        time_end = lowered.find("</t>", time_start)
        if time_end < 0:
            continue
        try:
            timestamp = float(content[time_start:time_end].strip())
        except (TypeError, ValueError):
            continue
        if not math.isfinite(timestamp) or timestamp < 0:
            continue

        boxes = []
        for raw_box_text in raw_boxes:
            try:
                raw_box = json.loads(raw_box_text)
            except (json.JSONDecodeError, TypeError):
                continue
            normalized = _normalize_box(raw_box, coordinate_scale)
            if normalized is not None:
                boxes.append(normalized)
        if boxes:
            claims.append(
                SpatioTemporalClaim(
                    object_name=content[object_body_start:object_end].strip(),
                    timestamp=timestamp,
                    boxes=tuple(boxes),
                )
            )
    return claims


def serialize_spatiotemporal_claims(
    claims: Iterable[SpatioTemporalClaim],
    *,
    wrap_answer: bool = True,
) -> str:
    """Serialize claims deterministically for dataset ground truth."""

    parts = []
    for claim in claims:
        boxes = "".join(
            "<box>[{},{},{},{}]</box>".format(
                *[f"{coordinate:g}" for coordinate in box]
            )
            for box in claim.boxes
        )
        parts.append(
            f"<obj>{claim.object_name}</obj>{boxes}"
            f"at<t>{claim.timestamp:g}</t>s"
        )
    body = "\n".join(parts)
    return f"<answer>{body}</answer>" if wrap_answer else body


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    width = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    height = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    return width * height


def _flatten_claims(
    claims: Sequence[SpatioTemporalClaim],
) -> list[tuple[float, tuple[float, float, float, float]]]:
    return [
        (claim.timestamp, box)
        for claim in claims
        for box in claim.boxes
    ]


def _maximum_weight_matching(weights: Sequence[Sequence[float]]) -> float:
    """Exact maximum bipartite matching for typical sparse evidence sets.

    Open-o3 examples normally contain only a handful of boxes.  A bitmask DP is
    exact up to 18 columns; a deterministic greedy fallback avoids pathological
    memory use for malformed generations containing hundreds of claims.
    """

    if not weights or not weights[0]:
        return 0.0
    rows = [list(map(float, row)) for row in weights]
    if len(rows) < len(rows[0]):
        rows = [list(column) for column in zip(*rows)]
    n_rows, n_columns = len(rows), len(rows[0])

    if n_columns > 18:
        candidates = sorted(
            (
                (weight, row_idx, column_idx)
                for row_idx, row in enumerate(rows)
                for column_idx, weight in enumerate(row)
                if weight > 0
            ),
            reverse=True,
        )
        used_rows, used_columns = set(), set()
        total = 0.0
        for weight, row_idx, column_idx in candidates:
            if row_idx not in used_rows and column_idx not in used_columns:
                total += weight
                used_rows.add(row_idx)
                used_columns.add(column_idx)
        return total

    @lru_cache(maxsize=None)
    def solve(row_idx: int, used_columns: int) -> float:
        if row_idx == n_rows:
            return 0.0
        best = solve(row_idx + 1, used_columns)
        for column_idx, weight in enumerate(rows[row_idx]):
            bit = 1 << column_idx
            if weight > 0 and not used_columns & bit:
                best = max(best, weight + solve(row_idx + 1, used_columns | bit))
        return best

    return solve(0, 0)


def spatiotemporal_support_iou(
    predicted: Sequence[SpatioTemporalClaim],
    ground_truth: Sequence[SpatioTemporalClaim],
    *,
    temporal_tolerance: float = 3.0,
) -> float:
    """Space--time support IoU for sparse timestamped boxes.

    Boxes can intersect only when their timestamps are within
    ``temporal_tolerance`` seconds.  A maximum one-to-one matching prevents
    duplicate predictions from reusing the same ground-truth area.  Unmatched
    and extra boxes remain in the union, so both missing and spammed evidence
    are penalized.  Object names are deliberately not matched, following
    Open-o3-Video's category-agnostic geometric reward.
    """

    pred_items = _flatten_claims(predicted)
    gt_items = _flatten_claims(ground_truth)
    if not pred_items and not gt_items:
        return 1.0
    if not pred_items or not gt_items:
        return 0.0

    weights = [
        [
            _intersection_area(pred_box, gt_box)
            if abs(pred_time - gt_time) <= temporal_tolerance
            else 0.0
            for gt_time, gt_box in gt_items
        ]
        for pred_time, pred_box in pred_items
    ]
    intersection = _maximum_weight_matching(weights)
    total_pred = sum(_box_area(box) for _, box in pred_items)
    total_gt = sum(_box_area(box) for _, box in gt_items)
    union = total_pred + total_gt - intersection
    if union <= 0:
        return 0.0
    return min(1.0, max(0.0, intersection / union))


def spatiotemporal_iou_from_text(
    prediction: str | None,
    ground_truth: str | None,
    *,
    temporal_tolerance: float = 3.0,
) -> float:
    """Parse two serializations and compute sparse space--time support IoU."""

    pred_claims = parse_spatiotemporal_claims(prediction, only_answer=True)
    gt_claims = parse_spatiotemporal_claims(ground_truth, only_answer=False)
    return spatiotemporal_support_iou(
        pred_claims,
        gt_claims,
        temporal_tolerance=temporal_tolerance,
    )


def spatiotemporal_temporal_proximity_reward(
    prediction: str | None,
    ground_truth: str | None,
    *,
    temporal_scale: float = 5.0,
) -> float:
    """Dense timestamp-only score for sparse spatio-temporal claims.

    The normal joint support IoU correctly requires both time and space, but it
    becomes exactly zero when either side misses.  This auxiliary term ignores
    boxes and supplies ordered temporal credit using the same annotated
    timestamps.  It is intended as a small add-on, never a replacement for the
    joint objective.
    """

    if temporal_scale <= 0:
        raise ValueError("temporal_scale must be positive.")
    predicted = parse_spatiotemporal_claims(prediction)
    target = parse_spatiotemporal_claims(ground_truth)
    if not predicted or not target:
        return 0.0
    pred_times = [claim.timestamp for claim in predicted]
    scores = [
        math.exp(-min(abs(timestamp - candidate) for candidate in pred_times) / temporal_scale)
        for timestamp in (claim.timestamp for claim in target)
    ]
    return sum(scores) / len(scores)


def _single_box_claim(
    claims: Sequence[SpatioTemporalClaim],
) -> SpatioTemporalClaim | None:
    """Return the only atomic claim, rejecting extra claims or boxes."""

    if len(claims) != 1 or len(claims[0].boxes) != 1:
        return None
    return claims[0]


def bbox_iou(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """IoU for two canonical ``xyxy`` boxes."""

    intersection = _intersection_area(box_a, box_b)
    union = _box_area(box_a) + _box_area(box_b) - intersection
    if union <= 0:
        return 0.0
    return min(1.0, max(0.0, intersection / union))


def atomic_spatiotemporal_components(
    prediction: str | None,
    ground_truth: str | None,
    *,
    temporal_sigma: float = 2.0,
    temporal_tolerance: float = 3.0,
) -> tuple[float, float, float]:
    """Return ``(joint, temporal, spatial)`` for one timestamp and one box.

    The temporal and spatial terms are multiplicatively coupled, so neither
    dimension can compensate for failure in the other. Predictions containing
    zero or multiple claims/boxes receive zero on every component.
    """

    if temporal_sigma <= 0:
        raise ValueError("temporal_sigma must be positive.")
    if temporal_tolerance <= 0:
        raise ValueError("temporal_tolerance must be positive.")

    # Open-o3 SFT places grounded evidence inside <think>, while our earlier
    # interface allowed it directly inside <answer>. Parse the entire response
    # and accept either wrapper, but still require exactly one atomic claim.
    predicted = _single_box_claim(parse_spatiotemporal_claims(prediction))
    target = _single_box_claim(parse_spatiotemporal_claims(ground_truth))
    if predicted is None or target is None:
        return 0.0, 0.0, 0.0

    time_error = abs(predicted.timestamp - target.timestamp)
    if time_error > temporal_tolerance:
        temporal = 0.0
    else:
        temporal = math.exp(
            -(time_error * time_error) / (2.0 * temporal_sigma * temporal_sigma)
        )
    spatial = bbox_iou(predicted.boxes[0], target.boxes[0])
    return temporal * spatial, temporal, spatial


def atomic_spatiotemporal_reward(
    prediction: str | None,
    ground_truth: str | None,
    *,
    temporal_sigma: float = 2.0,
    temporal_tolerance: float = 3.0,
) -> float:
    """Multiplicative timestamp-proximity and bbox-IoU reward."""

    joint, _, _ = atomic_spatiotemporal_components(
        prediction,
        ground_truth,
        temporal_sigma=temporal_sigma,
        temporal_tolerance=temporal_tolerance,
    )
    return joint


def spatiotemporal_format_reward(text: str | None) -> float:
    """Return 1 only for a non-empty, strictly formatted ``<answer>`` block."""

    if text is None:
        return 0.0
    answer_match = _ANSWER_PATTERN.search(str(text))
    if not answer_match:
        return 0.0
    body = answer_match.group(1).strip()
    claims = parse_spatiotemporal_claims(body)
    if not claims:
        return 0.0
    residue = _CLAIM_PATTERN.sub("", body)
    return 1.0 if not residue.strip() else 0.0


def atomic_spatiotemporal_format_reward(text: str | None) -> float:
    """Strict format reward for one claim in either supported SFT wrapper."""

    if text is None:
        return 0.0
    content = str(text).strip()
    answer_match = _ANSWER_PATTERN.fullmatch(content)
    if not answer_match:
        reasoned_match = _THINK_ANSWER_PATTERN.fullmatch(content)
        if not reasoned_match:
            return 0.0
        think_body = reasoned_match.group("think").strip()
        answer_body = reasoned_match.group("answer").strip()
        claims = parse_spatiotemporal_claims(think_body)
        if _single_box_claim(claims) is None or not answer_body:
            return 0.0
        if re.search(r"</?(?:obj|box|t)>", answer_body, flags=re.IGNORECASE):
            return 0.0
        # Open-o3's native SFT format includes natural-language reasoning
        # around the grounded evidence. Keep that compatible while enforcing
        # the actual atomic invariant: exactly one parsed claim/box and a
        # short, tag-free answer label.
        if len(answer_body.split()) > 12:
            return 0.0
        return 1.0

    answer_body = answer_match.group(1).strip()
    claims = parse_spatiotemporal_claims(answer_body)
    if _single_box_claim(claims) is None:
        return 0.0
    residue = _CLAIM_PATTERN.sub("", answer_body)
    return 1.0 if not residue.strip() else 0.0


def _single_object_trajectory(
    claims: Sequence[SpatioTemporalClaim],
    *,
    min_anchors: int = 2,
    max_anchors: int = 5,
    require_chronological: bool = False,
) -> list[SpatioTemporalClaim] | None:
    """Validate one sparse trajectory with one box at each unique timestamp."""

    if not min_anchors <= len(claims) <= max_anchors:
        return None
    if any(len(claim.boxes) != 1 for claim in claims):
        return None
    object_names = {
        re.sub(r"\s+", " ", claim.object_name).strip().casefold()
        for claim in claims
    }
    if len(object_names) != 1 or not next(iter(object_names)):
        return None
    timestamps = [claim.timestamp for claim in claims]
    if len(set(timestamps)) != len(timestamps):
        return None
    if require_chronological and timestamps != sorted(timestamps):
        return None
    return list(claims)


def trajectory_spatiotemporal_components(
    prediction: str | None,
    ground_truth: str | None,
    *,
    temporal_sigma: float = 2.0,
    temporal_tolerance: float = 3.0,
    min_anchors: int = 2,
    max_anchors: int = 5,
) -> tuple[float, float, float]:
    """Soft-F1 trajectory alignment for one object across sparse anchors.

    A maximum one-to-one match prevents duplicate predictions from reusing a
    ground-truth anchor. The denominator penalizes missing and extra anchors.
    Spatial overlap is gated by temporal proximity, while the joint term
    smoothly multiplies Gaussian temporal proximity and bbox IoU.
    """

    if temporal_sigma <= 0:
        raise ValueError("temporal_sigma must be positive.")
    if temporal_tolerance <= 0:
        raise ValueError("temporal_tolerance must be positive.")

    predicted = _single_object_trajectory(
        parse_spatiotemporal_claims(prediction),
        min_anchors=min_anchors,
        max_anchors=max_anchors,
    )
    target = _single_object_trajectory(
        parse_spatiotemporal_claims(ground_truth),
        min_anchors=min_anchors,
        max_anchors=max_anchors,
    )
    if predicted is None or target is None:
        return 0.0, 0.0, 0.0

    temporal_weights = []
    spatial_weights = []
    joint_weights = []
    for pred_claim in predicted:
        temporal_row = []
        spatial_row = []
        joint_row = []
        for target_claim in target:
            time_error = abs(pred_claim.timestamp - target_claim.timestamp)
            if time_error > temporal_tolerance:
                temporal = 0.0
                spatial = 0.0
            else:
                temporal = math.exp(
                    -(time_error * time_error)
                    / (2.0 * temporal_sigma * temporal_sigma)
                )
                spatial = bbox_iou(
                    pred_claim.boxes[0],
                    target_claim.boxes[0],
                )
            temporal_row.append(temporal)
            spatial_row.append(spatial)
            joint_row.append(temporal * spatial)
        temporal_weights.append(temporal_row)
        spatial_weights.append(spatial_row)
        joint_weights.append(joint_row)

    denominator = len(predicted) + len(target)
    scale = 2.0 / denominator
    return (
        scale * _maximum_weight_matching(joint_weights),
        scale * _maximum_weight_matching(temporal_weights),
        scale * _maximum_weight_matching(spatial_weights),
    )


def trajectory_spatiotemporal_reward(
    prediction: str | None,
    ground_truth: str | None,
    *,
    temporal_sigma: float = 2.0,
    temporal_tolerance: float = 3.0,
    min_anchors: int = 2,
    max_anchors: int = 5,
) -> float:
    """Return the joint single-object trajectory alignment."""

    joint, _, _ = trajectory_spatiotemporal_components(
        prediction,
        ground_truth,
        temporal_sigma=temporal_sigma,
        temporal_tolerance=temporal_tolerance,
        min_anchors=min_anchors,
        max_anchors=max_anchors,
    )
    return joint


def trajectory_spatiotemporal_format_reward(text: str | None) -> float:
    """Strict answer-only format for a chronological 2--5-anchor trajectory."""

    if text is None:
        return 0.0
    # Online trajectory reward intentionally decodes with special tokens so
    # existing mask-token tasks retain their structural tokens. Qwen may
    # therefore leave its generation terminator after an otherwise strict
    # answer. Normalize only known trailing terminators in this source-specific
    # checker; arbitrary text before or after the answer remains invalid.
    normalized = _TRAILING_GENERATION_TERMINATOR_PATTERN.sub(
        "",
        str(text).strip(),
    ).strip()
    match = _ANSWER_PATTERN.fullmatch(normalized)
    if not match:
        return 0.0
    answer_body = match.group(1).strip()
    if not answer_body:
        return 0.0
    claims = parse_spatiotemporal_claims(answer_body)
    residue = _CLAIM_PATTERN.sub("", answer_body)
    if residue.strip():
        return 0.0
    trajectory = _single_object_trajectory(
        claims,
        require_chronological=True,
    )
    return 1.0 if trajectory is not None else 0.0


_COORDINATE_LIST_PATTERN = re.compile(
    r"\[\s*[-+]?\d+(?:\.\d+)?"
    r"(?:\s*,\s*[-+]?\d+(?:\.\d+)?){3}\s*\]"
)
_EXPLICIT_TIME_PATTERN = re.compile(
    r"(?:<t>.*?</t>|"
    r"\b(?:at|around|from|between)\s+\d+(?:\.\d+)?\s*(?:s|sec(?:ond)?s?)\b|"
    r"\b\d+(?:\.\d+)?\s*(?:s|sec(?:ond)?s?)\b)",
    flags=re.IGNORECASE | re.DOTALL,
)


def spatiotemporal_caption_no_leak_reward(text: str | None) -> float:
    """Reward a caption only when it contains no coordinates or timestamps."""

    if text is None:
        return 0.0
    content = str(text)
    leaked = (
        re.search(r"</?(?:box|t)>", content, flags=re.IGNORECASE) is not None
        or _COORDINATE_LIST_PATTERN.search(content) is not None
        or _EXPLICIT_TIME_PATTERN.search(content) is not None
    )
    return 0.0 if leaked else 1.0


def strip_spatiotemporal_priors(text: str | None, replacement: str = "") -> str:
    """Remove leaked boxes/timestamps before constructing the inner prompt."""

    if text is None:
        return ""
    content = re.sub(
        r"<obj>(.*?)</obj>\s*(?:<box>.*?</box>\s*)+at\s*<t>.*?</t>\s*s",
        lambda match: f"{match.group(1).strip()} {replacement}".strip(),
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )
    content = re.sub(r"</?(?:obj|box|t)>", "", content, flags=re.IGNORECASE)
    content = _COORDINATE_LIST_PATTERN.sub(replacement, content)
    content = _EXPLICIT_TIME_PATTERN.sub(replacement, content)
    return re.sub(r"\s+", " ", content).strip()
