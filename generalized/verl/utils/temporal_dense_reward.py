"""Dense temporal localization rewards for CycleGRPO.

Plain tIoU gives every disjoint interval exactly zero.  These components retain
tIoU as the primary objective while providing bounded proximity information for
wrong-occurrence rollouts.  No additional annotation is used: all terms are
computed from the same target interval already consumed by tIoU.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


Interval = tuple[float, float]


def _valid(interval: Sequence[float]) -> Interval | None:
    if len(interval) < 2:
        return None
    start, end = float(interval[0]), float(interval[1])
    if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
        return None
    return start, end


def temporal_dense_components(
    predicted: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Return bounded dense scores for the closest predicted/target pair.

    The current Open-o3 temporal CycleGRPO data contains one target interval.
    Supporting a list here keeps malformed or future multi-window rows safe:
    each target uses its best predicted match and components are averaged.
    """

    preds = [item for value in predicted if (item := _valid(value)) is not None]
    gts = [item for value in target if (item := _valid(value)) is not None]
    if not preds or not gts:
        return {"center": 0.0, "boundary": 0.0, "duration": 0.0, "dense": 0.0}

    per_target: list[tuple[float, float, float, float]] = []
    for gs, ge in gts:
        gd = max(ge - gs, 1e-6)
        gc = (gs + ge) / 2.0
        candidates = []
        for ps, pe in preds:
            pd = max(pe - ps, 1e-6)
            pc = (ps + pe) / 2.0
            # Smooth, scale-adaptive terms. A prediction one GT duration away
            # still gets a small ordered signal instead of tying at zero.
            center = math.exp(-abs(pc - gc) / gd)
            boundary = 0.5 * (
                math.exp(-abs(ps - gs) / gd) + math.exp(-abs(pe - ge) / gd)
            )
            duration = math.exp(-abs(math.log(pd / gd)))
            # Do not let an interval earn duration credit merely by matching
            # the GT length at an unrelated occurrence. Gate shape credit by
            # temporal proximity to avoid a trivial fixed-duration strategy.
            gated_duration = duration * max(center, boundary)
            dense = 0.50 * center + 0.35 * boundary + 0.15 * gated_duration
            candidates.append((dense, center, boundary, duration))
        dense, center, boundary, duration = max(candidates)
        per_target.append((center, boundary, duration, dense))

    count = float(len(per_target))
    return {
        "center": sum(x[0] for x in per_target) / count,
        "boundary": sum(x[1] for x in per_target) / count,
        "duration": sum(x[2] for x in per_target) / count,
        "dense": sum(x[3] for x in per_target) / count,
    }
