"""Deterministic offline frame cache for long-video CycleGRPO.

The cache stores the exact frames returned by qwen-vl-utils after temporal
sampling and spatial resizing.  Frames are losslessly stored as uint8 numpy
arrays (qwen-vl-utils returns integer-valued float32 tensors) and restored to
float32 tensors at runtime.  One sidecar JSON file per video records both the
sampling protocol and video metadata so stale/incompatible caches are never
used silently.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


CACHE_FORMAT_VERSION = 1


class VideoFrameCacheError(RuntimeError):
    """A required frame cache is missing, corrupt, or incompatible."""


def cache_root() -> Optional[Path]:
    value = os.environ.get("CYCLEGRPO_FRAME_CACHE_ROOT", "").strip()
    return Path(value) if value else None


def cache_required() -> bool:
    return os.environ.get("CYCLEGRPO_REQUIRE_FRAME_CACHE", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def normalized_video_path(video: str) -> str:
    return os.path.abspath(os.path.expanduser(str(video)))


def video_cache_key(video: str) -> str:
    return hashlib.sha256(normalized_video_path(video).encode("utf-8")).hexdigest()


def cache_paths(root: Path, video: str) -> tuple[Path, Path]:
    key = video_cache_key(video)
    directory = root / key[:2]
    return directory / f"{key}.npy", directory / f"{key}.json"


def current_protocol(
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    video_fps: float,
) -> dict[str, Any]:
    def optional_int_env(name: str) -> Optional[int]:
        value = os.environ.get(name)
        return int(value) if value not in (None, "") else None

    return {
        "min_pixels": int(min_pixels) if min_pixels is not None else None,
        "max_pixels": int(max_pixels) if max_pixels is not None else None,
        "video_fps": float(video_fps),
        "total_pixels": optional_int_env("CYCLEGRPO_VIDEO_TOTAL_PIXELS"),
        "min_frames": optional_int_env("CYCLEGRPO_VIDEO_MIN_FRAMES"),
        "max_frames": optional_int_env("CYCLEGRPO_VIDEO_MAX_FRAMES"),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    tensor = frames.detach().cpu()
    if tensor.ndim != 4:
        raise VideoFrameCacheError(
            f"expected a 4-D video tensor [T,C,H,W], got shape={tuple(tensor.shape)}"
        )
    if tensor.dtype == torch.uint8:
        array = tensor.numpy()
    else:
        if not tensor.is_floating_point():
            raise VideoFrameCacheError(f"unsupported frame dtype: {tensor.dtype}")
        rounded = tensor.round()
        if not torch.equal(tensor, rounded):
            raise VideoFrameCacheError(
                "qwen-vl frame tensor contains non-integer values; refusing lossy uint8 cache"
            )
        if tensor.numel() and (float(tensor.min()) < 0 or float(tensor.max()) > 255):
            raise VideoFrameCacheError("frame values are outside uint8 range [0,255]")
        array = rounded.to(torch.uint8).numpy()
    return np.ascontiguousarray(array)


def write_cache_entry(
    root: Path,
    video: str,
    frames: torch.Tensor,
    video_metadata: dict[str, Any],
    sample_fps: float,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    array_path, metadata_path = cache_paths(root, video)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    array = frames_to_uint8(frames)
    source = Path(normalized_video_path(video))
    stat = source.stat()
    record = {
        "format_version": CACHE_FORMAT_VERSION,
        "video": str(source),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "protocol": _jsonable(protocol),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sample_fps": float(sample_fps),
        "video_metadata": _jsonable(video_metadata),
        "array_nbytes": int(array.nbytes),
    }
    unique = f".tmp-{os.getpid()}"
    temporary_array = Path(str(array_path) + unique)
    temporary_metadata = Path(str(metadata_path) + unique)
    try:
        with temporary_array.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        with temporary_metadata.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_array, array_path)
        os.replace(temporary_metadata, metadata_path)
    finally:
        temporary_array.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    return record


def _float_equal(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def validate_record(
    record: dict[str, Any],
    video: str,
    protocol: dict[str, Any],
    requested_nframes: Optional[int] = None,
    *,
    check_source: bool = True,
) -> None:
    if record.get("format_version") != CACHE_FORMAT_VERSION:
        raise VideoFrameCacheError(
            f"unsupported cache format_version={record.get('format_version')}"
        )
    expected_video = normalized_video_path(video)
    if record.get("video") != expected_video:
        raise VideoFrameCacheError(
            f"cache source mismatch: cached={record.get('video')} requested={expected_video}"
        )
    cached_protocol = record.get("protocol", {})
    for key, expected in protocol.items():
        actual = cached_protocol.get(key)
        matches = _float_equal(actual, expected) if key == "video_fps" else actual == expected
        if not matches:
            raise VideoFrameCacheError(
                f"cache protocol mismatch for {key}: cached={actual} requested={expected}"
            )
    shape = record.get("shape")
    if not isinstance(shape, list) or len(shape) != 4 or min(shape) <= 0:
        raise VideoFrameCacheError(f"invalid cached frame shape: {shape}")
    if requested_nframes is not None and int(requested_nframes) != int(shape[0]):
        raise VideoFrameCacheError(
            f"cached nframes={shape[0]} does not match requested nframes={requested_nframes}"
        )
    if record.get("dtype") != "uint8":
        raise VideoFrameCacheError(f"unsupported cached dtype={record.get('dtype')}")
    if check_source:
        stat = Path(expected_video).stat()
        if int(record.get("source_size", -1)) != int(stat.st_size):
            raise VideoFrameCacheError("source video size changed after cache creation")
        if int(record.get("source_mtime_ns", -1)) != int(stat.st_mtime_ns):
            raise VideoFrameCacheError("source video mtime changed after cache creation")


def load_cache_entry(
    video: str,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    video_fps: float,
    *,
    requested_nframes: Optional[int] = None,
) -> Optional[tuple[torch.Tensor, dict[str, Any], float]]:
    root = cache_root()
    if root is None:
        return None
    array_path, metadata_path = cache_paths(root, video)
    if not array_path.is_file() or not metadata_path.is_file():
        if cache_required():
            raise VideoFrameCacheError(
                f"required frame cache is missing for {video}: {array_path}"
            )
        return None
    protocol = current_protocol(min_pixels, max_pixels, video_fps)
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
        check_source = os.environ.get(
            "CYCLEGRPO_FRAME_CACHE_CHECK_SOURCE", "true"
        ).lower() in {"1", "true", "yes", "on"}
        validate_record(
            record,
            video,
            protocol,
            requested_nframes,
            check_source=check_source,
        )
        array = np.load(array_path, allow_pickle=False)
        if list(array.shape) != record["shape"] or array.dtype != np.uint8:
            raise VideoFrameCacheError(
                f"cached array does not match sidecar: shape={array.shape} dtype={array.dtype}"
            )
        if int(array.nbytes) != int(record.get("array_nbytes", -1)):
            raise VideoFrameCacheError("cached array byte count does not match sidecar")
        # qwen-vl-utils returns float32 frames in the 0--255 range.  Restoring
        # this exact public contract keeps Processor/vLLM behavior unchanged.
        frames = torch.from_numpy(array).to(dtype=torch.float32)
        metadata = dict(record["video_metadata"])
        return frames, metadata, float(record["sample_fps"])
    except Exception as exc:
        if cache_required():
            if isinstance(exc, VideoFrameCacheError):
                raise
            raise VideoFrameCacheError(f"failed to load frame cache for {video}: {exc}") from exc
        return None


def validate_cache_entry_files(
    root: Path,
    video: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Validate an entry without materializing its frame payload."""

    array_path, metadata_path = cache_paths(root, video)
    if not array_path.is_file() or not metadata_path.is_file():
        raise VideoFrameCacheError(f"cache entry is incomplete: {array_path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    validate_record(record, video, protocol)
    array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != record["shape"] or array.dtype != np.uint8:
        raise VideoFrameCacheError(
            f"cached array does not match sidecar: shape={array.shape} dtype={array.dtype}"
        )
    if int(array.nbytes) != int(record.get("array_nbytes", -1)):
        raise VideoFrameCacheError("cached array byte count does not match sidecar")
    return record
