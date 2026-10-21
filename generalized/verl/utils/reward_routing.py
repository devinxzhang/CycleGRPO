"""Pure helpers for routing multimodal rollout rewards."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# These sources are temporal-video tasks even after compact rollout transport
# moves their media under ``multi_modal_data``.
TEMPORAL_GROUNDING_SOURCES = frozenset({"tg_multi_merged", "tg_grounding"})


def _has_nonempty_media(value: Any) -> bool:
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True


def is_temporal_video_sample(
    non_tensor_batch: Mapping[str, Any], index: int, source: Any
) -> bool:
    """Return whether one rollout row must use temporal tIoU reward.

    ``DataProto`` stores multimodal payloads as an object array under
    ``multi_modal_data``.  Looking only for a top-level ``videos`` column is
    therefore insufficient after sharding/compact rollout transport.
    Explicit temporal sources are also recognized so loss of optional media
    metadata cannot silently reroute them to an image/mask reward.
    """

    if str(source) in TEMPORAL_GROUNDING_SOURCES:
        return True

    top_level_videos = non_tensor_batch.get("videos")
    if top_level_videos is not None:
        try:
            if _has_nonempty_media(top_level_videos[index]):
                return True
        except (IndexError, KeyError, TypeError):
            if _has_nonempty_media(top_level_videos):
                return True

    multimodal_rows = non_tensor_batch.get("multi_modal_data")
    if multimodal_rows is None:
        return False
    try:
        multimodal_row = multimodal_rows[index]
    except (IndexError, KeyError, TypeError):
        multimodal_row = multimodal_rows
    return (
        isinstance(multimodal_row, Mapping)
        and _has_nonempty_media(multimodal_row.get("videos"))
    )
