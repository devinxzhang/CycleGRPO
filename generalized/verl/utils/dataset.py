# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import math
import os
import re
import random
from collections import OrderedDict, defaultdict
from io import BytesIO
from typing import Any, Optional, Union, Tuple, List

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from jinja2 import Template
from PIL import Image, ImageDraw
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF
from .cyclegrpo_profiles import (
    NFRAME_PRESERVING_SOURCES,
    OPEN_O3_UNDERSTANDING_SOURCE,
    REPOSITORY_ROOT,
    SPATIOTEMPORAL_CYCLE_SOURCES,
    VIDEO_CAPTION_ANTI_TIME_INSTRUCTION,
)
from .spatiotemporal import (
    ATOMIC_SPATIOTEMPORAL_SOURCE,
    SPATIOTEMPORAL_SOURCE,
    TRAJECTORY_SPATIOTEMPORAL_SOURCE,
    TRAJECTORY_UNDERSTANDING_SOURCE,
    TRAJECTORY_LOCALIZATION_SOURCE,
)


NO_THINK_PREFIX = "<think>\n\n</think>\n\n"


ATOMIC_GROUNDING_SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user provides a video and "
    "asks a question, and the Assistant solves it. The assistant MUST first "
    "think about the reasoning process in the mind and then provide the user "
    "with the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively. All reasoning "
    "must be grounded in visual evidence from the video. When mentioning the "
    "single target, strictly use this format in the reasoning process: "
    "<obj>object_name</obj><box>[x1,y1,x2,y2]</box>"
    "at<t>time_in_seconds</t>s."
)

TRAJECTORY_GROUNDING_SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user provides timestamped "
    "frames from a video and asks it to ground one tracked target. The "
    "assistant must return only one <answer> </answer> block containing the "
    "final sparse trajectory. Do not output a <think> block or any explanation. "
    "Every grounded anchor must use this format: "
    "<obj>object_name</obj><box>[x1,y1,x2,y2]</box>"
    "at<t>time_in_seconds</t>s."
)


def build_atomic_grounding_instruction(description: str) -> str:
    """Open-o3-SFT-compatible prompt for one description-to-region query."""

    return (
        "Locate exactly one target instance matching the description below in "
        "the video:\n"
        f"{description}\n"
        "Inside <think>, output exactly one grounded evidence claim containing "
        "exactly one bounding box in this format:\n"
        "<obj>object_name</obj><box>[x1,y1,x2,y2]</box>"
        "at<t>time_in_seconds</t>s\n"
        "Coordinates must be integers normalized to [0,1000]. Do not return "
        "multiple timestamps, multiple boxes, or alternative candidates. "
        "Inside <answer>, output only a short object name."
    )


def build_trajectory_grounding_instruction(description: str) -> str:
    """Answer-only prompt for one sparse single-object trajectory."""

    return (
        "Locate the one physical target instance matching the description "
        "below throughout one continuous event in the video:\n"
        f"{description}\n"
        "Return only one <answer> block containing 2 to 5 chronological "
        "grounded trajectory anchors for that same target instance. Each "
        "anchor must contain exactly one bounding box in this format:\n"
        "<obj>object_name</obj><box>[x1,y1,x2,y2]</box>"
        "at<t>time_in_seconds</t>s\n"
        "Use the same object name in every claim. Coordinates must be integers "
        "normalized to [0,1000]. Do not return multiple objects, disconnected "
        "occurrences, multiple boxes at one timestamp, or alternative "
        "candidates. Do not output a <think> block, object-name-only summary, "
        "or any explanation outside <answer>."
    )


def add_no_think_prefix(prompt: str, enabled: bool) -> str:
    if not enabled:
        return prompt
    if prompt.endswith(NO_THINK_PREFIX):
        return prompt
    return prompt + NO_THINK_PREFIX


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    if not features:
        return {}

    # Mixed-route parquet rows do not necessarily expose the same optional
    # metadata keys. Tensor model inputs must be present for every sample;
    # non-tensor metadata is aligned to the full batch with None placeholders.
    # Appending only when a key happens to exist produces short arrays (for
    # example 26 ``images`` values in a 32-sample image/video batch), which
    # violates DataProto's batch consistency contract.
    all_keys = set().union(*(feature.keys() for feature in features))
    tensor_keys = {
        key
        for key in all_keys
        if any(isinstance(feature.get(key), torch.Tensor) for feature in features)
    }
    tensors = {}
    for key in tensor_keys:
        values = [feature.get(key) for feature in features]
        if not all(isinstance(value, torch.Tensor) for value in values):
            missing = [i for i, value in enumerate(values) if not isinstance(value, torch.Tensor)]
            raise ValueError(f"Tensor field {key!r} is missing for samples {missing}")
        tensors[key] = torch.stack(values, dim=0)

    non_tensors = {
        key: np.array([feature.get(key) for feature in features], dtype=object)
        for key in all_keys - tensor_keys
    }
    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def get_video_metadata(video_path: str) -> dict:
    """Get native video metadata (real fps and total frame count) via decord.VideoReader."""
    from decord import VideoReader
    vr = VideoReader(video_path, num_threads=int(os.environ.get("CYCLEGRPO_DECORD_NUM_THREADS", "2")))
    return {
        "fps": float(vr.get_avg_fps()),
        "total_num_frames": len(vr),
    }


def process_video(
    video: str,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    video_fps: float,
    return_fps: bool = False,
    return_video_metadata: bool = False,
    nframes: Optional[int] = None,
    use_sample_fps_metadata: bool = False,
) -> Union[list[ImageObject], tuple[list[ImageObject], float], tuple[list[ImageObject], dict]]:
    from .video_frame_cache import load_cache_entry

    cached = load_cache_entry(
        video,
        min_pixels,
        max_pixels,
        video_fps,
        requested_nframes=nframes,
    )
    if cached is not None:
        frames, video_metadata, sample_fps = cached
        if return_video_metadata:
            video_metadata = dict(video_metadata)
            if use_sample_fps_metadata:
                video_metadata["fps"] = float(sample_fps)
            return frames, video_metadata
        if return_fps:
            return frames, sample_fps
        return frames

    total_pixels = os.environ.get("CYCLEGRPO_VIDEO_TOTAL_PIXELS")
    vision_info = {
        "video": video,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }
    if total_pixels is not None:
        # OMTG-style processing: retain the requested temporal density and
        # dynamically lower per-frame spatial resolution so the entire video
        # stays inside one total-pixel budget.
        vision_info["total_pixels"] = int(total_pixels)
    else:
        # Historical CycleGRPO recipes use a deterministic fixed frame size.
        vision_info["resized_height"] = 224
        vision_info["resized_width"] = 224
    # Use explicit nframes if provided (for deterministic frame count across
    # dataset tokenization and feature extraction). Otherwise use fps-based sampling.
    if nframes is not None:
        vision_info["nframes"] = nframes
    else:
        vision_info["fps"] = video_fps
        # Optional experiment-level cap for recipes whose reference sampler
        # uses FPS together with explicit min/max frame bounds. Leave unset to
        # preserve the repository-wide Qwen-VL defaults.
        min_frames = os.environ.get("CYCLEGRPO_VIDEO_MIN_FRAMES")
        max_frames = os.environ.get("CYCLEGRPO_VIDEO_MAX_FRAMES")
        if min_frames is not None:
            vision_info["min_frames"] = int(min_frames)
        if max_frames is not None:
            vision_info["max_frames"] = int(max_frames)
    if return_video_metadata:
        (frames, video_metadata), sample_fps = fetch_video(
            vision_info,
            return_video_sample_fps=True,
            return_video_metadata=True,
        )
        video_metadata = dict(video_metadata)
        if use_sample_fps_metadata:
            video_metadata["fps"] = float(sample_fps)
        return frames, video_metadata
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


def process_atomic_spatiotemporal_query_video(
    video: str,
    query_timestamp: float,
    query_bbox: list[float] | tuple[float, ...],
    *,
    context_seconds: float = 1.0,
    context_frames: int = 5,
    output_size: int = 224,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Render an atomic ``(timestamp, bbox)`` query as a short visual clip.

    The first ``context_frames`` frames are chronological local context around
    the queried time. They are followed by the exact queried frame with a red
    rectangle and a target crop. This media is used only by the outer caption
    rollout; the inner grounding rollout receives the original unmarked video.
    """

    if context_seconds <= 0:
        raise ValueError("context_seconds must be positive.")
    if context_frames < 3 or context_frames % 2 == 0:
        raise ValueError("context_frames must be an odd integer >= 3.")
    if len(query_bbox) != 4:
        raise ValueError("query_bbox must contain exactly four coordinates.")

    from decord import VideoReader

    reader = VideoReader(video, num_threads=int(os.environ.get("CYCLEGRPO_DECORD_NUM_THREADS", "2")))
    total_frames = len(reader)
    if total_frames <= 0:
        raise ValueError(f"Video contains no frames: {video}")
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video fps for {video}: {fps}")

    duration = max(0.0, (total_frames - 1) / fps)
    query_timestamp = min(duration, max(0.0, float(query_timestamp)))
    context_times = np.linspace(
        query_timestamp - context_seconds,
        query_timestamp + context_seconds,
        context_frames,
    )
    context_times = np.clip(context_times, 0.0, duration)
    frame_indices = [
        min(total_frames - 1, max(0, int(round(float(time) * fps))))
        for time in context_times
    ]
    raw_frames = reader.get_batch(frame_indices).asnumpy()
    frames = [
        Image.fromarray(frame).convert("RGB").resize(
            (output_size, output_size),
            Image.Resampling.BICUBIC,
        )
        for frame in raw_frames
    ]

    exact_frame = frames[context_frames // 2]
    coords = [min(1000.0, max(0.0, float(value))) for value in query_bbox]
    x1, y1, x2, y2 = [
        int(round(value * output_size / 1000.0))
        for value in coords
    ]
    x1, x2 = sorted((min(output_size - 1, x1), min(output_size, x2)))
    y1, y2 = sorted((min(output_size - 1, y1), min(output_size, y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate query_bbox after scaling: {query_bbox}")

    marked_frame = exact_frame.copy()
    line_width = max(3, output_size // 56)
    ImageDraw.Draw(marked_frame).rectangle(
        (x1, y1, x2 - 1, y2 - 1),
        outline=(255, 0, 0),
        width=line_width,
    )

    padding = max(2, int(round(0.1 * max(x2 - x1, y2 - y1))))
    crop_box = (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(output_size, x2 + padding),
        min(output_size, y2 + padding),
    )
    target_crop = exact_frame.crop(crop_box).resize(
        (output_size, output_size),
        Image.Resampling.BICUBIC,
    )
    rendered_frames = [*frames, marked_frame, target_crop]
    video_tensor = torch.stack(
        [
            torch.from_numpy(np.array(frame, copy=True))
            .permute(2, 0, 1)
            .to(torch.float32)
            for frame in rendered_frames
        ]
    )

    # The final two frames are query visualizations rather than chronological
    # video frames, so expose a synthetic local-clip timeline to the processor.
    synthetic_fps = 2.0
    metadata = {
        "fps": synthetic_fps,
        "frames_indices": list(range(len(rendered_frames))),
        "total_num_frames": len(rendered_frames),
        "video_backend": "atomic_spatiotemporal_query",
    }
    return video_tensor, metadata


def process_trajectory_spatiotemporal_query_video(
    video: str,
    query_timestamps: list[float] | tuple[float, ...],
    query_bboxes: list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    output_size: int = 224,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Render one trajectory as chronological marked-frame/crop pairs."""

    if not 2 <= len(query_timestamps) <= 5:
        raise ValueError("Trajectory queries require 2 to 5 timestamps.")
    if len(query_timestamps) != len(query_bboxes):
        raise ValueError("Trajectory timestamps and bboxes must align.")
    timestamps = [float(value) for value in query_timestamps]
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(
        timestamps
    ):
        raise ValueError(
            "Trajectory timestamps must be unique and chronological."
        )
    if any(len(box) != 4 for box in query_bboxes):
        raise ValueError("Every trajectory anchor must contain one bbox.")

    from decord import VideoReader

    reader = VideoReader(video, num_threads=int(os.environ.get("CYCLEGRPO_DECORD_NUM_THREADS", "2")))
    total_frames = len(reader)
    if total_frames <= 0:
        raise ValueError(f"Video contains no frames: {video}")
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video fps for {video}: {fps}")
    duration = max(0.0, (total_frames - 1) / fps)
    timestamps = [
        min(duration, max(0.0, timestamp))
        for timestamp in timestamps
    ]
    frame_indices = [
        min(
            total_frames - 1,
            max(0, int(round(timestamp * fps))),
        )
        for timestamp in timestamps
    ]
    decoded = reader.get_batch(frame_indices).asnumpy()

    rendered_frames = []
    line_width = max(3, output_size // 56)
    for raw_frame, raw_box in zip(decoded, query_bboxes):
        frame = Image.fromarray(raw_frame).convert("RGB").resize(
            (output_size, output_size),
            Image.Resampling.BICUBIC,
        )
        coords = [
            min(1000.0, max(0.0, float(value)))
            for value in raw_box
        ]
        x1, y1, x2, y2 = [
            int(round(value * output_size / 1000.0))
            for value in coords
        ]
        x1, x2 = sorted(
            (min(output_size - 1, x1), min(output_size, x2))
        )
        y1, y2 = sorted(
            (min(output_size - 1, y1), min(output_size, y2))
        )
        # Very small valid boxes can collapse after mapping normalized
        # coordinates to the rendered pixel grid. Preserve the anchor instead
        # of dropping the evaluation sample by expanding it to one pixel.
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)

        marked_frame = frame.copy()
        ImageDraw.Draw(marked_frame).rectangle(
            (x1, y1, x2 - 1, y2 - 1),
            outline=(255, 0, 0),
            width=line_width,
        )
        padding = max(
            2,
            int(round(0.1 * max(x2 - x1, y2 - y1))),
        )
        crop = frame.crop(
            (
                max(0, x1 - padding),
                max(0, y1 - padding),
                min(output_size, x2 + padding),
                min(output_size, y2 + padding),
            )
        ).resize(
            (output_size, output_size),
            Image.Resampling.BICUBIC,
        )
        rendered_frames.extend([marked_frame, crop])

    video_tensor = torch.stack(
        [
            torch.from_numpy(np.array(frame, copy=True))
            .permute(2, 0, 1)
            .to(torch.float32)
            for frame in rendered_frames
        ]
    )
    metadata = {
        "fps": 2.0,
        "frames_indices": list(range(len(rendered_frames))),
        "total_num_frames": len(rendered_frames),
        "video_backend": "trajectory_spatiotemporal_query",
    }
    return video_tensor, metadata


def process_joint_spatiotemporal_query_video(
    video: str,
    query_timestamps: list[float] | tuple[float, ...],
    query_bboxes: list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    output_size: int = 224,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Render exact target frames with all same-time boxes marked and crops.

    Unlike the trajectory interface, a released Open-o3 ST row may contain
    several target boxes at one timestamp.  The outer captioner sees one
    marked full frame per timestamp followed by one crop per target box.  The
    unmarked inner grounding interface is constructed separately.
    """

    if len(query_timestamps) == 0 or len(query_timestamps) != len(query_bboxes):
        raise ValueError("Joint ST timestamps and bboxes must be non-empty and align.")
    if any(len(box) != 4 for box in query_bboxes):
        raise ValueError("Every joint ST target must contain one bbox.")

    from decord import VideoReader

    reader = VideoReader(video, num_threads=int(os.environ.get("CYCLEGRPO_DECORD_NUM_THREADS", "2")))
    total_frames = len(reader)
    if total_frames <= 0:
        raise ValueError(f"Video contains no frames: {video}")
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video fps for {video}: {fps}")
    duration = max(0.0, (total_frames - 1) / fps)

    grouped: dict[float, list[list[float]]] = defaultdict(list)
    for timestamp, box in zip(query_timestamps, query_bboxes):
        grouped[min(duration, max(0.0, float(timestamp)))].append(
            [float(value) for value in box]
        )
    ordered_times = sorted(grouped)
    frame_indices = [
        min(total_frames - 1, max(0, int(round(timestamp * fps))))
        for timestamp in ordered_times
    ]
    decoded = reader.get_batch(frame_indices).asnumpy()

    rendered_frames: list[ImageObject] = []
    line_width = max(3, output_size // 56)
    for raw_frame, timestamp in zip(decoded, ordered_times):
        frame = Image.fromarray(raw_frame).convert("RGB").resize(
            (output_size, output_size), Image.Resampling.BICUBIC
        )
        pixel_boxes = []
        for raw_box in grouped[timestamp]:
            coords = [min(1000.0, max(0.0, value)) for value in raw_box]
            x1, y1, x2, y2 = [
                int(round(value * output_size / 1000.0)) for value in coords
            ]
            x1, x2 = sorted((min(output_size - 1, x1), min(output_size, x2)))
            y1, y2 = sorted((min(output_size - 1, y1), min(output_size, y2)))
            x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
            pixel_boxes.append((x1, y1, x2, y2))

        marked = frame.copy()
        draw = ImageDraw.Draw(marked)
        for x1, y1, x2, y2 in pixel_boxes:
            draw.rectangle(
                (x1, y1, x2 - 1, y2 - 1), outline=(255, 0, 0), width=line_width
            )
        rendered_frames.append(marked)
        for x1, y1, x2, y2 in pixel_boxes:
            padding = max(2, int(round(0.1 * max(x2 - x1, y2 - y1))))
            rendered_frames.append(
                frame.crop(
                    (
                        max(0, x1 - padding),
                        max(0, y1 - padding),
                        min(output_size, x2 + padding),
                        min(output_size, y2 + padding),
                    )
                ).resize((output_size, output_size), Image.Resampling.BICUBIC)
            )

    video_tensor = torch.stack(
        [
            torch.from_numpy(np.array(frame, copy=True))
            .permute(2, 0, 1)
            .to(torch.float32)
            for frame in rendered_frames
        ]
    )
    return video_tensor, {
        "fps": 2.0,
        "frames_indices": list(range(len(rendered_frames))),
        "total_num_frames": len(rendered_frames),
        "video_backend": "joint_spatiotemporal_query",
    }


def process_open_o3_frame_sequence(
    video: str,
    *,
    nframes: int = 16,
    key_timestamps: Optional[list[float]] = None,
    distractor_count: int = 0,
    output_size: int = 224,
) -> tuple[list[ImageObject], list[float]]:
    """Decode the explicit timestamped-image interface used by Open-o3.

    Open-o3 did not train its spatio-temporal examples as a native Qwen video.
    It uniformly sampled frames, inserted annotated key frames in chronological
    order, and paired every image token with ``Frame N at {seconds}s`` text.
    Keeping this helper separate makes that compatibility path opt-in.
    """

    if nframes < 2:
        raise ValueError("nframes must be at least 2.")

    from decord import VideoReader

    reader = VideoReader(video, num_threads=int(os.environ.get("CYCLEGRPO_DECORD_NUM_THREADS", "2")))
    total_frames = len(reader)
    if total_frames <= 0:
        raise ValueError(f"Video contains no frames: {video}")
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video fps for {video}: {fps}")

    uniform_indices = np.linspace(0, total_frames - 1, nframes)
    uniform_indices = np.rint(uniform_indices).astype(np.int64).tolist()
    indexed_times = [
        (int(index), float(index) / fps, False)
        for index in uniform_indices
    ]

    duration = max(0.0, (total_frames - 1) / fps)
    if distractor_count < 0:
        raise ValueError("distractor_count must be non-negative.")
    if distractor_count:
        # A lone exact annotation frame leaks the target time because it is the
        # only off-grid frame.  Add deterministic, video-specific off-grid
        # frames so the target remains visible but is not identifiable from the
        # sampling pattern alone.
        occupied = set(uniform_indices)
        occupied.update(
            min(total_frames - 1, max(0, int(round(float(t) * fps))))
            for t in (key_timestamps or [])
        )
        candidates = [index for index in range(total_frames) if index not in occupied]
        if candidates:
            seed = int.from_bytes(
                hashlib.sha256(str(video).encode("utf-8")).digest()[:8], "big"
            )
            rng = np.random.default_rng(seed)
            chosen = rng.choice(
                candidates,
                size=min(int(distractor_count), len(candidates)),
                replace=False,
            )
            key_timestamps = [
                *(key_timestamps or []),
                *(float(index) / fps for index in chosen.tolist()),
            ]
    for timestamp in sorted(set(key_timestamps or [])):
        timestamp = min(duration, max(0.0, float(timestamp)))
        index = min(
            total_frames - 1,
            max(0, int(round(timestamp * fps))),
        )
        # Preserve the annotation's decimal timestamp in the prompt while
        # decoding the nearest real frame.
        indexed_times.append((index, timestamp, True))

    # A key frame replaces a uniform frame only when both resolve to the exact
    # same decoded frame; otherwise it is inserted, matching Open-o3.
    indexed_times.sort(key=lambda item: (item[0], not item[2]))
    deduplicated = []
    for item in indexed_times:
        if deduplicated and deduplicated[-1][0] == item[0]:
            if item[2]:
                deduplicated[-1] = item
            continue
        deduplicated.append(item)

    decoded = reader.get_batch([item[0] for item in deduplicated]).asnumpy()
    frames = [
        Image.fromarray(frame).convert("RGB").resize(
            (output_size, output_size),
            Image.Resampling.BICUBIC,
        )
        for frame in decoded
    ]
    timestamps = [item[1] for item in deduplicated]
    return frames, timestamps


def build_open_o3_frame_messages(
    description: str,
    timestamps: list[float],
) -> list[dict[str, Any]]:
    """Build atomic grounding messages matching Open-o3's SFT modality."""

    content = []
    for frame_index, timestamp in enumerate(timestamps, start=1):
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"Frame {frame_index} at {timestamp:.1f}s: ",
                },
                {"type": "image"},
                {"type": "text", "text": "\n"},
            ]
        )
    content.append(
        {
            "type": "text",
            "text": build_atomic_grounding_instruction(description),
        }
    )
    return [
        {"role": "system", "content": ATOMIC_GROUNDING_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_open_o3_trajectory_frame_messages(
    description: str,
    timestamps: list[float],
) -> list[dict[str, Any]]:
    """Build timestamped-image messages for trajectory grounding."""

    content = []
    for frame_index, timestamp in enumerate(timestamps, start=1):
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"Frame {frame_index} at {timestamp:.1f}s: ",
                },
                {"type": "image"},
                {"type": "text", "text": "\n"},
            ]
        )
    content.append(
        {
            "type": "text",
            "text": build_trajectory_grounding_instruction(description),
        }
    )
    return [
        {
            "role": "system",
            "content": TRAJECTORY_GROUNDING_SYSTEM_PROMPT,
        },
        {"role": "user", "content": content},
    ]


def build_open_o3_joint_frame_messages(
    description: str,
    timestamps: list[float],
) -> list[dict[str, Any]]:
    """Build timestamped-image messages for released multi-box ST rows."""

    content = []
    for frame_index, timestamp in enumerate(timestamps, start=1):
        content.extend(
            [
                {"type": "text", "text": f"Frame {frame_index} at {timestamp:.1f}s: "},
                {"type": "image"},
                {"type": "text", "text": "\n"},
            ]
        )
    content.append(
        {
            "type": "text",
            "text": (
                "Based on the description below, reconstruct all matching "
                "spatio-temporal evidence in these frames:\n"
                f"{description}\n"
                "Return only an <answer> block containing one or more claims "
                "formatted as <obj>object_name</obj><box>[x1,y1,x2,y2]</box>"
                "at<t>time_in_seconds</t>s. Coordinates must be integers "
                "normalized to [0,1000]. Do not output explanations."
            ),
        }
    )
    return [{"role": "user", "content": content}]


_OPEN_O3_FRAME_CACHE: OrderedDict = OrderedDict()
_OPEN_O3_FRAME_CACHE_MAX = 192


def process_open_o3_frame_sequence_cached(
    video: str,
    *,
    nframes: int = 16,
    key_timestamps: Optional[list[float]] = None,
    distractor_count: int = 0,
    output_size: int = 224,
) -> tuple[list[ImageObject], list[float]]:
    """Cached driver-side wrapper for Open-o3 timestamped image sequences."""

    normalized_keys = tuple(
        sorted(round(float(value), 6) for value in (key_timestamps or []))
    )
    key = (
        video,
        int(nframes),
        normalized_keys,
        int(distractor_count),
        int(output_size),
    )
    hit = _OPEN_O3_FRAME_CACHE.get(key)
    if hit is not None:
        _OPEN_O3_FRAME_CACHE.move_to_end(key)
        return hit
    from .video_decode_guard import guarded_video_decode

    result = guarded_video_decode(
        "open_o3_frames",
        video,
        nframes=nframes,
        key_timestamps=list(normalized_keys),
        distractor_count=distractor_count,
        output_size=output_size,
        media_path=video,
        context="driver-inner-open-o3-frames",
    )
    _OPEN_O3_FRAME_CACHE[key] = result
    while len(_OPEN_O3_FRAME_CACHE) > _OPEN_O3_FRAME_CACHE_MAX:
        _OPEN_O3_FRAME_CACHE.popitem(last=False)
    return result


# Bounded LRU decode cache for the DRIVER-side seg-batch build
# (_gen_seg_preprocess): without it the same video is decoded once per rollout
# row — n=6/8 identical decodes per prompt — dominating cycle-step wall clock.
# Decode is deterministic, so sibling/cross-step hits are correct.  Bound the
# cache by bytes rather than entry count: an OMTG 2-FPS video can occupy about
# 100 MB as float32, whereas an older 0.25-FPS entry was only a few MB.
_VIDEO_DECODE_CACHE: OrderedDict = OrderedDict()
_VIDEO_DECODE_CACHE_MAX = 192
_VIDEO_DECODE_CACHE_BYTES = 0


def _video_cache_value_nbytes(value) -> int:
    frames, _ = value
    if isinstance(frames, torch.Tensor):
        return int(frames.numel() * frames.element_size())
    if isinstance(frames, np.ndarray):
        return int(frames.nbytes)
    return 0


def process_video_cached(
    video,
    min_pixels,
    max_pixels,
    video_fps,
    nframes=None,
    use_sample_fps_metadata=False,
):
    global _VIDEO_DECODE_CACHE_BYTES

    key = (
        video,
        min_pixels,
        max_pixels,
        video_fps,
        nframes,
        use_sample_fps_metadata,
    )
    hit = _VIDEO_DECODE_CACHE.get(key)
    if hit is not None:
        _VIDEO_DECODE_CACHE.move_to_end(key)
        frames, metadata = hit
        return frames, dict(metadata)
    from .video_decode_guard import guarded_video_decode

    frames, metadata = guarded_video_decode(
        "video",
        video,
        min_pixels,
        max_pixels,
        video_fps,
        return_video_metadata=True,
        nframes=nframes,
        use_sample_fps_metadata=use_sample_fps_metadata,
        media_path=video,
        context="driver-inner-video",
    )
    _VIDEO_DECODE_CACHE[key] = (frames, dict(metadata))
    _VIDEO_DECODE_CACHE_BYTES += _video_cache_value_nbytes(
        _VIDEO_DECODE_CACHE[key]
    )
    max_bytes = int(
        os.environ.get("CYCLEGRPO_VIDEO_MEMORY_CACHE_MAX_BYTES", str(1024**3))
    )
    while (
        len(_VIDEO_DECODE_CACHE) > _VIDEO_DECODE_CACHE_MAX
        or (_VIDEO_DECODE_CACHE_BYTES > max_bytes and len(_VIDEO_DECODE_CACHE) > 1)
    ):
        _, evicted = _VIDEO_DECODE_CACHE.popitem(last=False)
        _VIDEO_DECODE_CACHE_BYTES -= _video_cache_value_nbytes(evicted)
    return frames, dict(metadata)


def sample_single_target_from_multi_target(
    prompt: str,
    ground_truth: str,
    images: Optional[List[str]] = None,
    max_targets: int = 5,
    region_format: str = "mask_token",  # "mask_token" or "bbox"
) -> Tuple[str, str, Optional[List[str]]]:
    """
    从多目标 region captioning prompt 中随机采样最多 max_targets 个目标。
    不使用 zoom-in 图片，只保留原图。

    Args:
        prompt: 多目标的 prompt 字符串
        ground_truth: 对应的 ground truth 字符串
        images: 图片路径列表，只保留第一张原图
        max_targets: 最多保留的目标数量，默认为5
        region_format: 区域表示格式，"mask_token" 或 "bbox"

    Returns:
        Tuple[str, str, Optional[List[str]]]: (处理后的 prompt, 处理后的 ground truth, 只包含原图的列表)

    如果目标数量 <= max_targets，返回原始内容不变。
    如果目标数量 > max_targets，随机选择 max_targets 个目标保留。
    """
    # 根据 region_format 选择不同的正则表达式
    if region_format == "mask_token":
        region_pattern = r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>"
    elif region_format == "bbox":
        region_pattern = r"\[\d+,\s*\d+,\s*\d+,\s*\d+\]"
    else:
        raise ValueError(f"Unknown region_format: {region_format}, expected 'mask_token' or 'bbox'")

    # 只保留原图
    new_images = None
    if images is not None and len(images) > 0:
        new_images = [images[0]]  # 只保留原图

    # 1. 找到主句部分，提取所有 region tokens
    # 支持两种格式：有或没有 "Please respond with interleaved segmentation masks" 后缀
    main_part_match = re.search(
        r"(Could you please give me a detailed description of the following regions?\?)\s*(.*?)\s*(\.\s*(?:Please respond with interleaved segmentation masks.*?)?)(?=\s*(?:Zoom in|$))",
        prompt,
        re.DOTALL
    )
    # 匹配格式: "Provide a detailed description of this region [x1, y1, x2, y2]."
    # 注意：没有问号，bbox 直接跟在 "this region" 后面，句尾是句号
    main_part_match2 = re.search(
        r"(Provide a detailed description of this region)\s*(.*?)\s*\.$",
        prompt,
        re.DOTALL
    )

    if not main_part_match and not main_part_match2:
        # 不匹配预期格式，返回原始内容（但只保留原图）
        return prompt, ground_truth, new_images

    # 提取主句中的 region tokens (mask token 或 bbox)
    main_tokens_str = main_part_match.group(2) if main_part_match else main_part_match2.group(2)
    main_tokens = re.findall(region_pattern, main_tokens_str)

    if len(main_tokens) <= max_targets:
        # 目标数量不超过 max_targets，使用所有 tokens
        selected_indices = list(range(len(main_tokens)))
        selected_tokens = main_tokens
    else:
        # 随机选择 max_targets 个 token（保持原始顺序）
        selected_indices = sorted(random.sample(range(len(main_tokens)), max_targets))
        selected_tokens = [main_tokens[i] for i in selected_indices]

    # 2. 构建新的 prompt（不包含 zoom-in 部分）
    if region_format == "bbox":
        # bbox 格式使用简洁的 prompt
        new_prompt = "<image>\n"
        prompt_parts = []
        for token in selected_tokens:
            part = f"Given a detailed description of the region at bounding box {token}."
            prompt_parts.append(part)
        new_prompt += " ".join(prompt_parts)
    else:
        # mask_token 格式
        region_word = "region" if len(selected_tokens) == 1 else "regions"
        new_prompt = f"<image>\nCould you please give me a detailed description of the following {region_word}? "
        new_prompt += ", ".join(selected_tokens)
        new_prompt += "."

    # 3. 处理 ground truth
    new_ground_truth = extract_multiple_ground_truths(ground_truth, selected_tokens, selected_indices, len(main_tokens), region_format)

    return new_prompt, new_ground_truth, new_images


def extract_multiple_ground_truths(
    ground_truth: str,
    selected_tokens: List[str],
    selected_indices: List[int],
    total_count: int,
    region_format: str = "mask_token",  # "mask_token" or "bbox"
) -> str:
    """
    从多目标的 ground truth 中提取与选中 tokens 对应的 ground truths。

    Args:
        ground_truth: 原始的多目标 ground truth
        selected_tokens: 选中的 region tokens 列表 (mask token 或 bbox)
        selected_indices: 选中 tokens 在原始列表中的索引
        total_count: 原始目标总数
        region_format: 区域表示格式，"mask_token" 或 "bbox"

    Returns:
        str: 提取后的 ground truth
    """
    # 根据 region_format 选择不同的正则表达式
    if region_format == "mask_token":
        region_pattern = r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>"
    else:  # bbox
        region_pattern = r"\[\d+,\s*\d+,\s*\d+,\s*\d+\]"

    # 检查是否有 <answer>...</answer> 包裹
    answer_match = re.search(r"<answer>(.*?)</answer>", ground_truth, re.DOTALL)
    if answer_match:
        inner_content = answer_match.group(1).strip()
        gt_tokens = re.findall(region_pattern, inner_content)

        # 按索引提取选中的 tokens
        extracted_tokens = []
        for idx in selected_indices:
            if idx < len(gt_tokens):
                extracted_tokens.append(gt_tokens[idx])

        if extracted_tokens:
            return f"<answer>{', '.join(extracted_tokens)}</answer>"

    # 方式2: 如果 ground truth 中包含 object_ref 格式
    # 格式: <|object_ref_start|>...<|object_ref_end|><|mt_start|>...<|mt_end|>
    object_ref_parts = []
    for token in selected_tokens:
        object_ref_pattern = (
            r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>\s*" +
            re.escape(token)
        )
        match = re.search(object_ref_pattern, ground_truth, re.DOTALL)
        if match:
            object_ref_parts.append(f"<|object_ref_start|>{match.group(1)}<|object_ref_end|>{token}")

    if object_ref_parts:
        return " ".join(object_ref_parts)

    # 方式3: 按 <|object_ref_start|> 分割
    if "<|object_ref_start|>" in ground_truth:
        parts = re.split(r"(?=<\|object_ref_start\|>)", ground_truth)
        parts = [p for p in parts if p.strip()]
        extracted_parts = []
        for idx in selected_indices:
            if idx < len(parts):
                extracted_parts.append(parts[idx])
        if extracted_parts:
            return " ".join(extracted_parts)

    # 方式4: 按句号分割
    sentences = re.split(r'(?<=[.。])\s*', ground_truth)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) == total_count:
        extracted_sentences = []
        for idx in selected_indices:
            if idx < len(sentences):
                extracted_sentences.append(sentences[idx])
        if extracted_sentences:
            return " ".join(extracted_sentences)

    # 无法解析，返回原始 ground truth
    return ground_truth

class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        # data_path: str,
        data_path: List[str],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        cap_prompt_key: str = "cap_prompt",
        cap_answer_key: str = "cap_answer",
        seg_prompt_key: str = "seg_prompt",
        seg_answer_key: str = "seg_answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        region_format: str = "mask_token",  # "mask_token" or "bbox"
        enable_no_think_prefix: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.cap_prompt_key = cap_prompt_key
        self.cap_answer_key = cap_answer_key
        self.seg_prompt_key = seg_prompt_key
        self.seg_answer_key = seg_answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.region_format = region_format
        self.enable_no_think_prefix = enable_no_think_prefix

        # 支持多路径加载
        datasets_list = []
        for single_path in data_path:
            if "@" in single_path:
                single_path, data_split = single_path.split("@")
            else:
                data_split = "train"

            if os.path.isdir(single_path):
                # when we use dataset builder, we should always refer to the train split
                file_type = os.path.splitext(os.listdir(single_path)[0])[-1][1:].replace("jsonl", "json")
                ds = load_dataset(file_type, data_dir=single_path, split=data_split)
            elif os.path.isfile(single_path):
                file_type = os.path.splitext(single_path)[-1][1:].replace("jsonl", "json")
                ds = load_dataset(file_type, data_files=single_path, split=data_split)
            else:
                # load remote dataset from huggingface hub
                ds = load_dataset(single_path, split=data_split)

            datasets_list.append(ds)
            print(f"Loaded {len(ds)} samples from {single_path}")

        # 合并所有数据集
        if len(datasets_list) == 1:
            self.dataset = datasets_list[0]
        else:
            self.dataset = concatenate_datasets(datasets_list)
        print(f"Total samples after concatenation: {len(self.dataset)}")

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

    def _build_gen_seg_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        # A unified parquet can contain both media columns in its schema; the
        # inactive one is null for each row. Membership alone therefore
        # misclassifies image rows as video rows.
        is_video_sample = (
            self.video_key in example and example.get(self.video_key) is not None
        )
        media_token = "<video>" if is_video_sample else "<image>"
        # 视频样本使用时间戳定位模板；图像样本沿用 bbox/mask 模板
        if (
            is_video_sample
            and example.get("source")
            == TRAJECTORY_SPATIOTEMPORAL_SOURCE
        ):
            PROMPT_TEMPLATE = (
                "{media_token}\n"
                + build_trajectory_grounding_instruction("{description}")
            )
        elif is_video_sample and example.get("source") == ATOMIC_SPATIOTEMPORAL_SOURCE:
            PROMPT_TEMPLATE = (
                "{media_token}\n"
                + build_atomic_grounding_instruction("{description}")
            )
        elif is_video_sample and example.get("source") == SPATIOTEMPORAL_SOURCE:
            PROMPT_TEMPLATE = """{media_token}
Based on the description below, reconstruct all matching spatio-temporal evidence in the video:
        {description}
Return only an <answer> block containing one or more claims in this exact format:
<obj>object_name</obj><box>[x1,y1,x2,y2]</box>at<t>time_in_seconds</t>s
Coordinates must be integers normalized to [0,1000]. Repeat <box>...</box> before the timestamp when one object has multiple boxes at that time. Repeat the complete claim for different timestamps or objects. Do not output explanations or any other text."""
        elif is_video_sample:
            PROMPT_TEMPLATE = """{media_token}\nBased on the description below, locate the timestamp interval in the video where the event occurs:
        {description}
            Output only the time intervals in the format 'x.x - x.x seconds'. - Each interval should be represented as a start and end time in seconds (e.g., '115.5 - 127.0 seconds'). - If there are multiple segments, separate them with commas (e.g., '10.0 - 20.0 seconds, 35.5 - 50.0 seconds'). - If there is only one segment, output a single interval. - If the event cannot be located, output 'null'. Do not output any explanation."""
        elif self.region_format == "bbox":
            PROMPT_TEMPLATE = """{media_token}\nAll spatial relationships are defined from the viewer's perspective, where 'front' means closer to the viewer and 'back' means farther from the viewer. Please provide the bounding box of the object the following statement describes:
        {description}
        Ensure that all details mentioned about the object are accurate. If a matching object is found, provide its bounding box in the format `[x1, y1, x2, y2]` where coordinates are normalized to [0, 1000]. If no matching object is found, output null."""
        else:
            PROMPT_TEMPLATE = """{media_token}\nAll spatial relationships are defined from the viewer's perspective, where 'front' means closer to the viewer and 'back' means farther from the viewer. Please provide the segmentation mask of the object the following statement describes:
        {description}
        Ensure that all details mentioned about the object are accurate. If a matching object is found, provide its segmentation mask in the format `<|mt_start|><|mt_xxxx|><|mt_xxxx|><|mt_end|>`. If no matching object is found, output null."""

        prompt_str: str = example[self.seg_prompt_key]
        # Defensive cleanup: strip any vision-related markers the upstream caller may
        # have left in the description, otherwise the re.split below treats them as
        # extra media references and breaks the processor.
        prompt_str = re.sub(
            r'<(?:image|video)>|<\|(?:image|video)_pad\|>|<\|vision_(?:start|end)\|>',
            '',
            prompt_str,
        )
        prompt_str = PROMPT_TEMPLATE.format(description=prompt_str, media_token=media_token)

        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example or self.video_key in example:
            # Handle mixed <image> and <video> tags using regex (reference: video_dataloader.py)
            content_list = []
            segments = re.split(r"(<image>|<video>)", prompt_str)
            segments = [s for s in segments if s]
            for segment in segments:
                if segment == "<image>":
                    content_list.append({"type": "image"})
                elif segment == "<video>":
                    content_list.append({"type": "video", "fps": self.video_fps})
                else:
                    if content_list and content_list[-1].get("type") in {"image", "video"}:
                        segment = segment.lstrip("\n")
                    content_list.append({"type": "text", "text": segment})

            messages = [{"role": "user", "content": content_list}]
            if example.get("source") in {
                ATOMIC_SPATIOTEMPORAL_SOURCE,
                TRAJECTORY_SPATIOTEMPORAL_SOURCE,
            }:
                system_prompt = (
                    TRAJECTORY_GROUNDING_SYSTEM_PROMPT
                    if example.get("source")
                    == TRAJECTORY_SPATIOTEMPORAL_SOURCE
                    else ATOMIC_GROUNDING_SYSTEM_PROMPT
                )
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                )
            return messages
        else:
            messages = [{"role": "user", "content": prompt_str}]
            if example.get("source") in {
                ATOMIC_SPATIOTEMPORAL_SOURCE,
                TRAJECTORY_SPATIOTEMPORAL_SOURCE,
            }:
                system_prompt = (
                    TRAJECTORY_GROUNDING_SYSTEM_PROMPT
                    if example.get("source")
                    == TRAJECTORY_SPATIOTEMPORAL_SOURCE
                    else ATOMIC_GROUNDING_SYSTEM_PROMPT
                )
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                )
            return messages

    def _get_image_and_mask(self, sample: dict[str, Any]):
        """
        从样本中获取image path和随机采样的一个mask_2d
        支持两种格式:
        1. mask_2d直接嵌入文本中: <|mt_start|><|mt_xxxx|><|mt_xxxx|><|mt_end|>
        2. mask_2d在```json```代码块中

        Args:
            sample: 包含'image'和'conversations'字段的样本字典

        Returns:
            tuple: (image_path, sampled_mask_2d)
                如果没有找到mask_2d，返回 (image_path, None)
        """
        # 1. 获取images路径
        image_path = sample['image']
        if isinstance(image_path, list):
            image_path = image_path[0]

        # 2. 从gpt回答中提取mask_2d
        gpt_value = sample['conversations'][1]['value']

        # 方式1: 尝试从```json```代码块中提取
        json_match = re.search(r'```json\s*(.*?)\s*```', gpt_value, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                all_mask_2d = [item['mask_2d'] for item in result]
                if all_mask_2d:
                    return image_path, random.choice(all_mask_2d)
            except:
                pass

        # 方式2: 直接从文本中提取mask_2d格式
        pattern = r'<\|mt_start\|><\|mt_\d+\|><\|mt_\d+\|><\|mt_end\|>'
        all_mask_2d = re.findall(pattern, gpt_value)
        if all_mask_2d:
            return image_path, random.choice(all_mask_2d)

        return image_path, None

    def _gen_seg_preprocess(self, example: dict[str, Any]):
        seg_messages = self._build_gen_seg_messages(example)
        example['cap_responses'] = example.pop(self.seg_prompt_key)
        if self.image_key in example:
            seg_prompt = self.processor.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = add_no_think_prefix(seg_prompt, self.enable_no_think_prefix)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            seg_model_inputs = self.processor(text=[seg_prompt], images=processed_images[:1], add_special_tokens=False, return_tensors="pt")
            seg_input_ids = seg_model_inputs.pop("input_ids")[0]
            seg_attention_mask = seg_model_inputs.pop("attention_mask")[0]
            example["seg_multi_modal_data"] = {"images": images[:1]}
        elif self.video_key in example:
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            stored_nframes = example.get('nframes')
            source = example.get("source")
            is_atomic = source == ATOMIC_SPATIOTEMPORAL_SOURCE
            is_trajectory = (
                source == TRAJECTORY_SPATIOTEMPORAL_SOURCE
            )
            is_joint = (
                source == SPATIOTEMPORAL_SOURCE
                and example.get("query_timestamps") is not None
            )
            if is_atomic or is_trajectory or is_joint:
                if len(videos) != 1:
                    raise ValueError(
                        "Open-o3 spatio-temporal samples must contain one video."
                    )
                nf = int(stored_nframes[0]) if stored_nframes is not None else 16
                if is_trajectory or is_joint:
                    key_timestamps = [
                        float(value)
                        for value in example["query_timestamps"]
                    ]
                else:
                    key_timestamps = [
                        float(
                            example.get(
                                "query_timestamp",
                                example.get("extra_info", {}).get(
                                    "query_timestamp"
                                ),
                            )
                        )
                    ]
                frames, timestamps = (
                    process_open_o3_frame_sequence_cached(
                        videos[0],
                        nframes=nf,
                        key_timestamps=key_timestamps,
                        distractor_count=(
                            int(example.get("inner_distractor_count", 0))
                            if is_joint
                            else 0
                        ),
                    )
                )
                message_builder = (
                    build_open_o3_joint_frame_messages
                    if is_joint
                    else (
                        build_open_o3_trajectory_frame_messages
                        if is_trajectory
                        else build_open_o3_frame_messages
                    )
                )
                seg_messages = message_builder(
                    example["cap_responses"], timestamps
                )
                seg_prompt = self.processor.apply_chat_template(
                    seg_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                seg_prompt = add_no_think_prefix(
                    seg_prompt,
                    self.enable_no_think_prefix,
                )
                seg_model_inputs = self.processor(
                    images=frames,
                    text=[seg_prompt],
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                example["seg_multi_modal_data"] = {
                    "videos": videos,
                    "open_o3_frame_sequence": True,
                    "nframes": [nf],
                    "key_timestamps": (
                        [key_timestamps]
                        if is_trajectory or is_joint
                        else key_timestamps
                    ),
                    "distractor_count": (
                        int(example.get("inner_distractor_count", 0))
                        if is_joint
                        else 0
                    ),
                }
            else:
                seg_prompt = self.processor.apply_chat_template(
                    seg_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                seg_prompt = add_no_think_prefix(
                    seg_prompt,
                    self.enable_no_think_prefix,
                )
                processed_videos = [] if len(videos) != 0 else None
                video_metadata_list = []
                for i, video in enumerate(videos):
                    nf = stored_nframes[i] if stored_nframes is not None else None
                    processed_video, video_metadata = process_video_cached(
                        video,
                        self.min_pixels,
                        self.max_pixels,
                        self.video_fps,
                        nframes=nf,
                    )
                    processed_videos.append(processed_video)
                    video_metadata_list.append(video_metadata)

                seg_model_inputs = self.processor(
                    videos=processed_videos,
                    text=[seg_prompt],
                    add_special_tokens=False,
                    return_tensors="pt",
                    videos_kwargs={
                        "video_metadata": video_metadata_list,
                        "do_sample_frames": False,
                    },
                )
                # Fallback: manually compute second_per_grid_ts if processor
                # did not provide it.
                if "second_per_grid_ts" in self.processor.model_input_names:
                    seg_model_inputs["second_per_grid_ts"] = [
                        2.0 / vm["fps"] for vm in video_metadata_list
                    ]
                example["seg_multi_modal_data"] = {
                    "videos": videos,
                    "fps": self.video_fps,
                    "nframes": [pv.shape[0] for pv in processed_videos],
                }

            seg_input_ids = seg_model_inputs.pop("input_ids")[0]
            seg_attention_mask = seg_model_inputs.pop("attention_mask")[0]
        else:
            seg_prompt = self.tokenizer.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = add_no_think_prefix(seg_prompt, self.enable_no_think_prefix)
            seg_model_inputs = self.tokenizer([seg_prompt], add_special_tokens=False, return_tensors="pt")
            seg_input_ids = seg_model_inputs.pop("input_ids")[0]
            seg_attention_mask = seg_model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            seg_vision_position_ids = get_rope_index(
                self.processor,
                input_ids=seg_input_ids,
                image_grid_thw=seg_model_inputs.get("image_grid_thw", None),
                video_grid_thw=seg_model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=seg_model_inputs.get("second_per_grid_ts", None),
                attention_mask=seg_attention_mask,
            )  # (3, seq_length)
            seg_text_position_ids = torch.arange(len(seg_input_ids)).unsqueeze(0)  # (1, seq_length)
            seg_position_ids = torch.cat((seg_text_position_ids, seg_vision_position_ids), dim=0)  # (4, seq_length)
        else:
            seg_position_ids = torch.clip(seg_attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        seg_input_ids, seg_attention_mask, seg_position_ids = VF.postprocess_data(
            input_ids=seg_input_ids,
            attention_mask=seg_attention_mask,
            position_ids=seg_position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        seg_raw_prompt_ids = self.tokenizer.encode(seg_prompt, add_special_tokens=False)
        if len(seg_raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                seg_raw_prompt_ids = seg_raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                seg_raw_prompt_ids = seg_raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(seg_raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["seg_input_ids"] = seg_input_ids
        example["seg_attention_mask"] = seg_attention_mask
        example["seg_position_ids"] = seg_position_ids
        example["seg_raw_prompt_ids"] = seg_raw_prompt_ids
        example["seg_ground_truth"] = example.pop('seg_ground_truth')
        return example

    def _judge_preprocess(self, example: dict[str, Any]) -> dict[str, Any]:
        """Prompt-only inputs for the v3 readback judge (video-mode only).

        example: {'videos': [path], 'nframes': [n] or None, 'judge_prompt': str}
        judge_prompt must contain one <video> tag. Returns left-padded
        input_ids / attention_mask / position_ids of the prompt, ready for a
        caption response to be appended (compute_log_probs scoring path).
        Uses process_video_cached, so per-batch repeated videos decode once.
        """
        prompt_str = example["judge_prompt"]
        content_list = []
        for segment in (s for s in re.split(r"(<video>)", prompt_str) if s):
            if segment == "<video>":
                content_list.append({"type": "video", "fps": self.video_fps})
            else:
                if content_list and content_list[-1].get("type") == "video":
                    segment = segment.lstrip("\n")
                content_list.append({"type": "text", "text": segment})
        messages = [{"role": "user", "content": content_list}]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt = add_no_think_prefix(prompt, self.enable_no_think_prefix)

        videos = example["videos"]
        stored_nframes = example.get("nframes")
        processed_videos, video_metadata_list = [], []
        for i, video in enumerate(videos):
            nf = stored_nframes[i] if stored_nframes is not None else None
            pv, vm = process_video_cached(video, self.min_pixels, self.max_pixels, self.video_fps, nframes=nf)
            processed_videos.append(pv)
            video_metadata_list.append(vm)

        model_inputs = self.processor(
            videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt",
            videos_kwargs={"video_metadata": video_metadata_list, "do_sample_frames": False},
        )
        if "second_per_grid_ts" in self.processor.model_input_names:
            model_inputs["second_per_grid_ts"] = [2.0 / vm["fps"] for vm in video_metadata_list]
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, L)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "multi_modal_data": {"videos": videos, "fps": self.video_fps,
                                 "nframes": [pv.shape[0] for pv in processed_videos]},
        }

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        cap_prompt_str: str = example.get(self.cap_prompt_key) or ""
        seg_prompt_str: str = example.get(self.seg_prompt_key) or ""
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            cap_prompt_str = format_prompt.render(content=cap_prompt_str)
            seg_prompt_str = format_prompt.render(content=seg_prompt_str)

        if self.image_key in example or self.video_key in example:
            # Handle mixed <image> and <video> tags using regex (reference: video_dataloader.py)
            def parse_content(text):
                content_list = []
                segments = re.split(r"(<image>|<video>)", text)
                segments = [s for s in segments if s]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video", "fps": self.video_fps})
                    else:
                        if content_list and content_list[-1].get("type") in {"image", "video"}:
                            segment = segment.lstrip("\n")
                        content_list.append({"type": "text", "text": segment})
                return content_list

            cap_content_list = parse_content(cap_prompt_str)
            seg_content_list = parse_content(seg_prompt_str)

            return [{"role": "user", "content": cap_content_list}], [{"role": "user", "content": seg_content_list}]
        else:
            return [{"role": "user", "content": cap_prompt_str}], [{"role": "user", "content": seg_prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        cap_messages, seg_messages = self._build_messages(example)
        if self.image_key in example:
            cap_prompt = self.processor.apply_chat_template(cap_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = self.processor.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
            cap_prompt = add_no_think_prefix(cap_prompt, self.enable_no_think_prefix)
            seg_prompt = add_no_think_prefix(seg_prompt, self.enable_no_think_prefix)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            cap_model_inputs = self.processor(processed_images, [cap_prompt], add_special_tokens=False, return_tensors="pt")
            seg_model_inputs = self.processor(processed_images[:1], [seg_prompt], add_special_tokens=False, return_tensors="pt")
            return cap_model_inputs["input_ids"].size(-1) <= self.max_prompt_length and seg_model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            cap_prompt = self.processor.apply_chat_template(cap_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = self.processor.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
            cap_prompt = add_no_think_prefix(cap_prompt, self.enable_no_think_prefix)
            seg_prompt = add_no_think_prefix(seg_prompt, self.enable_no_think_prefix)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_metadata_list = []
            stored_nframes = (
                example.get("nframes")
                if example.get("source") in SPATIOTEMPORAL_CYCLE_SOURCES
                else None
            )
            for i, video in enumerate(videos):
                nf = stored_nframes[i] if stored_nframes is not None else None
                processed_video, video_metadata = process_video(
                    video,
                    self.min_pixels,
                    self.max_pixels,
                    self.video_fps,
                    return_video_metadata=True,
                    nframes=nf,
                )
                processed_videos.append(processed_video)
                video_metadata_list.append(video_metadata)

            videos_kwargs = {"video_metadata": video_metadata_list, "do_sample_frames": False}
            cap_model_inputs = self.processor(
                videos=processed_videos, text=[cap_prompt], add_special_tokens=False, return_tensors="pt",
                videos_kwargs=videos_kwargs,
            )
            seg_model_inputs = self.processor(
                videos=processed_videos, text=[seg_prompt], add_special_tokens=False, return_tensors="pt",
                videos_kwargs=videos_kwargs,
            )
            return cap_model_inputs["input_ids"].size(-1) <= self.max_prompt_length and seg_model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            cap_prompt = self.tokenizer.apply_chat_template(cap_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = self.tokenizer.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
            cap_prompt = add_no_think_prefix(cap_prompt, self.enable_no_think_prefix)
            seg_prompt = add_no_think_prefix(seg_prompt, self.enable_no_think_prefix)
            cap_input_ids = self.tokenizer.encode(cap_prompt, add_special_tokens=False)
            seg_input_ids = self.tokenizer.encode(seg_prompt, add_special_tokens=False)
            return len(cap_input_ids) <= self.max_prompt_length and len(seg_input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        # HuggingFace datasets materialize the union parquet schema for every
        # row. Remove inactive nullable media columns before route-specific
        # processing so they cannot leak into a mixed batch as partial arrays.
        if example.get(self.image_key) is None:
            example.pop(self.image_key, None)
        if example.get(self.video_key) is None:
            example.pop(self.video_key, None)
        project_root = os.environ.get("SAMTOK_RL_ROOT", str(REPOSITORY_ROOT))
        if self.image_key in example and example[self.image_key] is not None:
            if isinstance(example[self.image_key], list):
                example[self.image_key] = [
                    os.path.join(project_root, img[2:]) if isinstance(img, str) and img.startswith("./") else img
                    for img in example[self.image_key]
                ]
            elif isinstance(example[self.image_key], str) and example[self.image_key].startswith("./"):
                example[self.image_key] = os.path.join(project_root, example[self.image_key][2:])

        # Only apply anti-time-leak instruction for video captioning prompts.
        # Skip tg_grounding: there cap_problem is a *grounding* prompt ("locate the
        # timestamp interval ..."), so suppressing time info contradicts the task
        # and makes the model emit a description instead of a time interval.
        if (
            self.video_key in example
            and example[self.video_key] is not None
            and example.get("source") not in {
                "tg_grounding",
                TRAJECTORY_LOCALIZATION_SOURCE,
                OPEN_O3_UNDERSTANDING_SOURCE,
            }
        ):
            if VIDEO_CAPTION_ANTI_TIME_INSTRUCTION not in example["cap_problem"]:
                example["cap_problem"] += VIDEO_CAPTION_ANTI_TIME_INSTRUCTION

        cap_messages, seg_messages = self._build_messages(example)

        # messages: ['role', 'content']
        example.pop(self.cap_prompt_key, None)
        example.pop(self.seg_prompt_key, None)
        if self.image_key in example and example[self.image_key] is not None:
            cap_prompt = self.processor.apply_chat_template(cap_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = self.processor.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
            cap_prompt = add_no_think_prefix(cap_prompt, self.enable_no_think_prefix)
            seg_prompt = add_no_think_prefix(seg_prompt, self.enable_no_think_prefix)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            cap_model_inputs = self.processor(processed_images, [cap_prompt], add_special_tokens=False, return_tensors="pt")
            seg_model_inputs = self.processor(processed_images[:1], [seg_prompt], add_special_tokens=False, return_tensors="pt")
            cap_input_ids = cap_model_inputs.pop("input_ids")[0]
            seg_input_ids = seg_model_inputs.pop("input_ids")[0]
            cap_attention_mask = cap_model_inputs.pop("attention_mask")[0]
            seg_attention_mask = seg_model_inputs.pop("attention_mask")[0]
            example["cap_multi_modal_data"] = {"images": images}
            example["seg_multi_modal_data"] = {"images": images[:1]}
        elif self.video_key in example and example[self.video_key] is not None:
            cap_prompt = self.processor.apply_chat_template(cap_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = self.processor.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
            cap_prompt = add_no_think_prefix(cap_prompt, self.enable_no_think_prefix)
            seg_prompt = add_no_think_prefix(seg_prompt, self.enable_no_think_prefix)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            cap_processed_videos = [] if len(videos) != 0 else None
            seg_processed_videos = [] if len(videos) != 0 else None
            cap_video_metadata_list = []
            seg_video_metadata_list = []
            stored_nframes = (
                example.get("nframes")
                if example.get("source") in NFRAME_PRESERVING_SOURCES
                else None
            )
            is_atomic_query = (
                example.get("source") == ATOMIC_SPATIOTEMPORAL_SOURCE
            )
            is_trajectory_query = (
                example.get("source") in {
                    TRAJECTORY_SPATIOTEMPORAL_SOURCE,
                    TRAJECTORY_UNDERSTANDING_SOURCE,
                }
            )
            is_joint_query = (
                example.get("source") == SPATIOTEMPORAL_SOURCE
                and example.get("query_timestamps") is not None
                and example.get("query_bboxes") is not None
            )
            if is_joint_query and example.get("nframes") is not None:
                stored_nframes = example["nframes"]
            if is_atomic_query or is_trajectory_query or is_joint_query:
                if len(videos) != 1:
                    raise ValueError(
                        "Query-conditioned spatio-temporal samples must "
                        "contain one video."
                    )
            if is_atomic_query:
                query_timestamp = float(example["query_timestamp"])
                query_bbox = list(example["query_bbox"])
                query_context_seconds = float(example["query_context_seconds"])
                query_context_frames = int(example["query_context_frames"])
            elif is_trajectory_query:
                query_trajectory_timestamps = [
                    float(value)
                    for value in example["query_timestamps"]
                ]
                query_trajectory_bboxes = [
                    [float(value) for value in box]
                    for box in example["query_bboxes"]
                ]
            elif is_joint_query:
                query_joint_timestamps = [
                    float(value) for value in example["query_timestamps"]
                ]
                query_joint_bboxes = [
                    [float(value) for value in box]
                    for box in example["query_bboxes"]
                ]
                inner_distractor_count = int(
                    example.get("inner_distractor_count", 4)
                )
            for i, video in enumerate(videos):
                nf = stored_nframes[i] if stored_nframes is not None else None
                seg_video, seg_metadata = process_video(
                    video,
                    self.min_pixels,
                    self.max_pixels,
                    self.video_fps,
                    return_video_metadata=True,
                    nframes=nf,
                    use_sample_fps_metadata=(
                        is_atomic_query or is_trajectory_query or is_joint_query
                    ),
                )
                seg_processed_videos.append(seg_video)
                seg_video_metadata_list.append(seg_metadata)
                if is_atomic_query:
                    cap_video, cap_metadata = (
                        process_atomic_spatiotemporal_query_video(
                            video,
                            query_timestamp,
                            query_bbox,
                            context_seconds=query_context_seconds,
                            context_frames=query_context_frames,
                        )
                    )
                elif is_trajectory_query:
                    cap_video, cap_metadata = (
                        process_trajectory_spatiotemporal_query_video(
                            video,
                            query_trajectory_timestamps,
                            query_trajectory_bboxes,
                        )
                    )
                elif is_joint_query:
                    cap_video, cap_metadata = (
                        process_joint_spatiotemporal_query_video(
                            video,
                            query_joint_timestamps,
                            query_joint_bboxes,
                        )
                    )
                else:
                    cap_video, cap_metadata = seg_video, seg_metadata
                cap_processed_videos.append(cap_video)
                cap_video_metadata_list.append(cap_metadata)

            cap_videos_kwargs = {
                "video_metadata": cap_video_metadata_list,
                "do_sample_frames": False,
            }
            seg_videos_kwargs = {
                "video_metadata": seg_video_metadata_list,
                "do_sample_frames": False,
            }
            cap_model_inputs = self.processor(
                videos=cap_processed_videos,
                text=[cap_prompt],
                add_special_tokens=False,
                return_tensors="pt",
                videos_kwargs=cap_videos_kwargs,
            )
            seg_model_inputs = self.processor(
                videos=seg_processed_videos,
                text=[seg_prompt],
                add_special_tokens=False,
                return_tensors="pt",
                videos_kwargs=seg_videos_kwargs,
            )
            # Fallback: manually compute second_per_grid_ts if processor didn't provide it
            if "second_per_grid_ts" in self.processor.model_input_names:
                cap_model_inputs["second_per_grid_ts"] = [
                    2.0 / vm["fps"] for vm in cap_video_metadata_list
                ]
                seg_model_inputs["second_per_grid_ts"] = [
                    2.0 / vm["fps"] for vm in seg_video_metadata_list
                ]

            cap_input_ids = cap_model_inputs.pop("input_ids")[0]
            seg_input_ids = seg_model_inputs.pop("input_ids")[0]
            cap_attention_mask = cap_model_inputs.pop("attention_mask")[0]
            seg_attention_mask = seg_model_inputs.pop("attention_mask")[0]
            example["cap_multi_modal_data"] = {
                "videos": videos,
                "fps": self.video_fps,
                "nframes": [pv.shape[0] for pv in cap_processed_videos],
            }
            if is_atomic_query:
                example["cap_multi_modal_data"].update(
                    {
                        "query_timestamps": [query_timestamp],
                        "query_bboxes": [query_bbox],
                        "query_context_seconds": query_context_seconds,
                        "query_context_frames": query_context_frames,
                    }
                )
            elif is_trajectory_query:
                example["cap_multi_modal_data"].update(
                    {
                        "trajectory_query_timestamps": [
                            query_trajectory_timestamps
                        ],
                        "trajectory_query_bboxes": [
                            query_trajectory_bboxes
                        ],
                    }
                )
            elif is_joint_query:
                example["cap_multi_modal_data"].update(
                    {
                        "joint_query_timestamps": [query_joint_timestamps],
                        "joint_query_bboxes": [query_joint_bboxes],
                    }
                )
            example["seg_multi_modal_data"] = {
                "videos": videos,
                "fps": self.video_fps,
                "nframes": [pv.shape[0] for pv in seg_processed_videos],
            }
            if is_atomic_query:
                # The online inner rollout re-decodes this media through the
                # explicit timestamped-image interface used by Open-o3 SFT.
                # Keep the raw path here so the initial outer rollout remains
                # a query video and does not receive any inner key-frame data.
                example["seg_multi_modal_data"].update(
                    {
                        "open_o3_frame_sequence": True,
                        "nframes": [
                            int(stored_nframes[0])
                            if stored_nframes is not None
                            else int(seg_processed_videos[0].shape[0])
                        ],
                        "key_timestamps": [query_timestamp],
                    }
                )
            elif is_trajectory_query:
                example["seg_multi_modal_data"].update(
                    {
                        "open_o3_frame_sequence": True,
                        "nframes": [
                            int(stored_nframes[0])
                            if stored_nframes is not None
                            else int(seg_processed_videos[0].shape[0])
                        ],
                        "key_timestamps": [
                            query_trajectory_timestamps
                        ],
                    }
                )
            elif is_joint_query:
                example["seg_multi_modal_data"].update(
                    {
                        "open_o3_frame_sequence": True,
                        "nframes": [
                            int(stored_nframes[0])
                            if stored_nframes is not None
                            else int(seg_processed_videos[0].shape[0])
                        ],
                        "key_timestamps": [
                            sorted(set(query_joint_timestamps))
                        ],
                        "distractor_count": inner_distractor_count,
                    }
                )
        else:
            cap_prompt = self.tokenizer.apply_chat_template(cap_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = self.tokenizer.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
            cap_prompt = add_no_think_prefix(cap_prompt, self.enable_no_think_prefix)
            seg_prompt = add_no_think_prefix(seg_prompt, self.enable_no_think_prefix)
            cap_model_inputs = self.tokenizer([cap_prompt], add_special_tokens=False, return_tensors="pt")
            seg_model_inputs = self.tokenizer([seg_prompt], add_special_tokens=False, return_tensors="pt")
            cap_input_ids = cap_model_inputs.pop("input_ids")[0]
            seg_input_ids = seg_model_inputs.pop("input_ids")[0]
            cap_attention_mask = cap_model_inputs.pop("attention_mask")[0]
            seg_attention_mask = seg_model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            cap_vision_position_ids = get_rope_index(
                self.processor,
                input_ids=cap_input_ids,
                image_grid_thw=cap_model_inputs.get("image_grid_thw", None),
                video_grid_thw=cap_model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=cap_model_inputs.get("second_per_grid_ts", None),
                attention_mask=cap_attention_mask,
            )  # (3, seq_length)
            seg_vision_position_ids = get_rope_index(
                self.processor,
                input_ids=seg_input_ids,
                image_grid_thw=seg_model_inputs.get("image_grid_thw", None),
                video_grid_thw=seg_model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=seg_model_inputs.get("second_per_grid_ts", None),
                attention_mask=seg_attention_mask,
            )  # (3, seq_length)
            cap_text_position_ids = torch.arange(len(cap_input_ids)).unsqueeze(0)  # (1, seq_length)
            seg_text_position_ids = torch.arange(len(seg_input_ids)).unsqueeze(0)  # (1, seq_length)
            cap_position_ids = torch.cat((cap_text_position_ids, cap_vision_position_ids), dim=0)  # (4, seq_length)
            seg_position_ids = torch.cat((seg_text_position_ids, seg_vision_position_ids), dim=0)  # (4, seq_length)
        else:
            cap_position_ids = torch.clip(cap_attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
            seg_position_ids = torch.clip(seg_attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        cap_input_ids, cap_attention_mask, cap_position_ids = VF.postprocess_data(
            input_ids=cap_input_ids,
            attention_mask=cap_attention_mask,
            position_ids=cap_position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        seg_input_ids, seg_attention_mask, seg_position_ids = VF.postprocess_data(
            input_ids=seg_input_ids,
            attention_mask=seg_attention_mask,
            position_ids=seg_position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        cap_raw_prompt_ids = self.tokenizer.encode(cap_prompt, add_special_tokens=False)
        if len(cap_raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                cap_raw_prompt_ids = cap_raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                cap_raw_prompt_ids = cap_raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(cap_raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        seg_raw_prompt_ids = self.tokenizer.encode(seg_prompt, add_special_tokens=False)
        if len(seg_raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                seg_raw_prompt_ids = seg_raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                seg_raw_prompt_ids = seg_raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(seg_raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["cap_input_ids"] = cap_input_ids
        example["cap_attention_mask"] = cap_attention_mask
        example["cap_position_ids"] = cap_position_ids
        example["cap_raw_prompt_ids"] = cap_raw_prompt_ids
        example["cap_ground_truth"] = example.pop(self.cap_answer_key)
        example["cap_prompt"] = cap_prompt
        example["seg_raw_prompt_ids"] = seg_raw_prompt_ids
        example["seg_prompt"] = seg_prompt
        example["seg_ground_truth"] = example.pop(self.seg_answer_key)
        return example


def _process_multi_modal_data_for_vllm(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> Optional[dict[str, Any]]:
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        nframes_list = multi_modal_data.get("nframes")
        query_timestamps = multi_modal_data.get("query_timestamps")
        query_bboxes = multi_modal_data.get("query_bboxes")
        trajectory_query_timestamps = multi_modal_data.get(
            "trajectory_query_timestamps"
        )
        trajectory_query_bboxes = multi_modal_data.get(
            "trajectory_query_bboxes"
        )
        joint_query_timestamps = multi_modal_data.get("joint_query_timestamps")
        joint_query_bboxes = multi_modal_data.get("joint_query_bboxes")
        open_o3_frame_sequence = bool(
            multi_modal_data.get("open_o3_frame_sequence", False)
        )
        key_timestamps = multi_modal_data.get("key_timestamps")
        distractor_count = int(multi_modal_data.get("distractor_count", 0))
        use_sample_fps_metadata = bool(
            multi_modal_data.get("use_sample_fps_metadata", False)
        )
        for vi, video in enumerate(multi_modal_data["videos"]):
            if open_o3_frame_sequence:
                nf = nframes_list[vi] if nframes_list is not None else 16
                selected_key_timestamps = (
                    key_timestamps[vi]
                    if key_timestamps is not None
                    else None
                )
                if selected_key_timestamps is not None and not isinstance(
                    selected_key_timestamps,
                    (list, tuple, np.ndarray),
                ):
                    selected_key_timestamps = [selected_key_timestamps]
                frame_images, _ = process_open_o3_frame_sequence(
                    video,
                    nframes=nf,
                    key_timestamps=selected_key_timestamps,
                    distractor_count=distractor_count,
                )
                images.extend(frame_images)
                continue
            elif joint_query_timestamps is not None and joint_query_bboxes is not None:
                frames, metadata = process_joint_spatiotemporal_query_video(
                    video,
                    joint_query_timestamps[vi],
                    joint_query_bboxes[vi],
                )
            elif (
                trajectory_query_timestamps is not None
                and trajectory_query_bboxes is not None
            ):
                frames, metadata = (
                    process_trajectory_spatiotemporal_query_video(
                        video,
                        trajectory_query_timestamps[vi],
                        trajectory_query_bboxes[vi],
                    )
                )
            elif query_timestamps is not None and query_bboxes is not None:
                frames, metadata = process_atomic_spatiotemporal_query_video(
                    video,
                    query_timestamps[vi],
                    query_bboxes[vi],
                    context_seconds=float(
                        multi_modal_data.get("query_context_seconds", 1.0)
                    ),
                    context_frames=int(
                        multi_modal_data.get("query_context_frames", 5)
                    ),
                )
            else:
                nf = nframes_list[vi] if nframes_list is not None else None
                frames, metadata = process_video(
                    video,
                    min_pixels,
                    max_pixels,
                    video_fps,
                    return_video_metadata=True,
                    nframes=nf,
                    use_sample_fps_metadata=use_sample_fps_metadata,
                )
            videos.append((frames, metadata))

    if len(images) != 0:
        return {"image": images}
    if len(videos) != 0:
        return {"video": videos}
    return None


def _generate_sequences_with_vllm(gen_batch, llm, sampling_params, pad_token_id):
    from tensordict import TensorDict

    from ..protocol import DataProto

    input_ids: torch.Tensor = gen_batch.batch["input_ids"]
    attention_mask: torch.Tensor = gen_batch.batch["attention_mask"]
    position_ids: torch.Tensor = gen_batch.batch["position_ids"]
    eos_token_id: int = gen_batch.meta_info["eos_token_id"]
    batch_size = input_ids.size(0)

    non_tensor_batch = dict(gen_batch.non_tensor_batch)
    batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
    batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)

    if batch_multi_modal_data is not None:
        vllm_inputs = []
        for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
            vllm_inputs.append(
                {
                    "prompt_token_ids": list(raw_prompt_ids),
                    "multi_modal_data": _process_multi_modal_data_for_vllm(
                        multi_modal_data,
                        gen_batch.meta_info["min_pixels"],
                        gen_batch.meta_info["max_pixels"],
                        gen_batch.meta_info["video_fps"],
                    ),
                }
            )
    else:
        vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

    completions = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
    response_ids = [output.token_ids for completion in completions for output in completion.outputs]
    response_ids = VF.pad_2d_list_to_length(response_ids, pad_token_id, max_length=sampling_params.max_tokens)
    response_ids = response_ids.to(input_ids.device)

    if sampling_params.n > 1:
        batch_size = batch_size * sampling_params.n
        input_ids = input_ids.repeat_interleave(sampling_params.n, dim=0)
        attention_mask = attention_mask.repeat_interleave(sampling_params.n, dim=0)
        position_ids = position_ids.repeat_interleave(sampling_params.n, dim=0)
        if batch_multi_modal_data is not None:
            batch_multi_modal_data = np.repeat(batch_multi_modal_data, sampling_params.n, axis=0)

    sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
    response_length = response_ids.size(1)
    delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
    delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
    if position_ids.ndim == 3:
        delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

    response_position_ids = position_ids[..., -1:] + delta_position_id
    position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
    response_mask = VF.get_response_mask(response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype)
    attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

    batch = TensorDict(
        {
            "prompts": input_ids,
            "responses": response_ids,
            "input_ids": sequence_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )
    output_non_tensor_batch = {"multi_modal_data": batch_multi_modal_data} if batch_multi_modal_data is not None else {}
    return DataProto(batch=batch, non_tensor_batch=output_non_tensor_batch, meta_info=gen_batch.meta_info)


if __name__ == "__main__":
    """
    测试脚本：加载数据集 → 加载模型 → 推理生成 → 对比 ground truth。

    使用方式（在项目根目录运行）：
        python -m verl.utils.dataset \
            --model_path Qwen/Qwen3-VL-4B-Instruct \
            --data_path ./rl_dataset/tg_multi_merged_train_rl.parquet \
            --num_samples 2 \
            --max_new_tokens 512

    也可以测试图像数据：
        python -m verl.utils.dataset \
            --model_path Qwen/Qwen3-VL-4B-Instruct \
            --data_path ./rl_dataset/denseworld_0k_img_2_samples_train.parquet \
            --num_samples 1
    """
    import argparse
    import multiprocessing as mp
    import os

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    from ..protocol import DataProto

    parser = argparse.ArgumentParser(description="测试 RLHFDataset 数据加载 + vLLM 推理")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径，如 Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data_path", type=str, nargs="+", required=True,
                        help="数据集路径（支持多个），如 ./rl_dataset/xxx.parquet")
    parser.add_argument("--num_samples", type=int, default=2, help="测试的样本数量")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--max_prompt_length", type=int, default=8192, help="最大 prompt 长度")
    parser.add_argument("--image_dir", type=str, default=None, help="图片目录前缀")
    parser.add_argument("--video_fps", type=float, default=0.25, help="视频采样帧率")
    parser.add_argument("--min_pixels", type=int, default=3136, help="最小像素数")
    parser.add_argument("--max_pixels", type=int, default=262144, help="最大像素数")
    parser.add_argument("--region_format", type=str, default="mask_token", choices=["mask_token", "bbox"],
                        help="区域表示格式")
    parser.add_argument("--enable_no_think_prefix", action="store_true", help="是否在生成起始处添加空 think 块")
    parser.add_argument("--format_prompt", type=str, default=None, help="format prompt 模板文件路径")
    parser.add_argument("--sample_index", type=int, default=None, help="指定测试的样本索引（默认从头开始）")
    parser.add_argument("--do_sample", action="store_true", help="是否使用采样生成（默认 greedy）")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p 采样")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="重复惩罚系数")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.6, help="vLLM 显存占用比例")
    args = parser.parse_args()

    print("=" * 60)
    print("1. 加载 Processor 和 Tokenizer")
    print("=" * 60)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    print(f"   Processor: {processor.__class__.__name__}")
    print(f"   Tokenizer: {tokenizer.__class__.__name__}")
    print(f"   Vocab size: {tokenizer.vocab_size}")

    print("\n" + "=" * 60)
    print("2. 构建 RLHFDataset 加载数据")
    print("=" * 60)
    dataset = RLHFDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        processor=processor,
        cap_prompt_key="cap_problem",
        cap_answer_key="cap_answer",
        seg_prompt_key="seg_problem",
        seg_answer_key="seg_answer",
        image_key="images",
        video_key="videos",
        image_dir=args.image_dir,
        video_fps=args.video_fps,
        max_prompt_length=args.max_prompt_length,
        truncation="left",
        format_prompt=args.format_prompt,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        filter_overlong_prompts=False,
        region_format=args.region_format,
        enable_no_think_prefix=args.enable_no_think_prefix,
    )
    print(f"   数据集大小: {len(dataset)}")

    print("\n" + "=" * 60)
    print("3. 加载 vLLM")
    print("=" * 60)
    print(f"   模型路径: {args.model_path}")
    llm = LLM(
        args.model_path,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_new_tokens,
        disable_mm_preprocessor_cache=True,
    )
    sampling_params = SamplingParams(
        n=1,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        detokenize=False,
    )
    print("   推理后端: vLLM")
    print(f"   Sampling params: {sampling_params}")

    print("\n" + "=" * 60)
    print("4. 逐样本推理并对比 ground truth")
    print("=" * 60)

    # 确定要测试的样本索引
    if args.sample_index is not None:
        indices = [args.sample_index]
    else:
        indices = list(range(min(args.num_samples, len(dataset))))

    for i, idx in enumerate(indices):
        print(f"\n{'─' * 60}")
        print(f"样本 [{i+1}/{len(indices)}]  index={idx}")
        print(f"{'─' * 60}")

        # 获取数据集处理后的样本
        sample = dataset[idx]

        # 提取 ground truth
        cap_gt = sample.get("cap_ground_truth", None)
        seg_gt = sample.get("seg_ground_truth", None)

        raw_example = dataset.dataset[idx]
        is_video = "videos" in raw_example and raw_example["videos"] is not None
        is_image = "images" in raw_example and raw_example["images"] is not None

        print(f"\n  [Caption 任务]")
        print(f"  cap_prompt: {sample.get('cap_prompt', None)}")

        cap_task_dict = {
            k.replace("cap_", ""): sample[k].unsqueeze(0)
            for k in ["cap_input_ids", "cap_attention_mask", "cap_position_ids"]
        }
        cap_task_dict["raw_prompt_ids"] = np.array([sample["cap_raw_prompt_ids"]], dtype=object)
        if sample.get("cap_multi_modal_data") is not None:
            cap_task_dict["multi_modal_data"] = np.array([sample["cap_multi_modal_data"]], dtype=object)

        meta_info = {
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "video_fps": args.video_fps,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        cap_gen_batch = DataProto.from_single_dict(cap_task_dict, meta_info=meta_info)
        cap_gen_batch_output = _generate_sequences_with_vllm(
            cap_gen_batch, llm, sampling_params, tokenizer.pad_token_id
        )
        cap_response_ids = cap_gen_batch_output.batch["responses"][0]
        cap_valid_len = int(cap_gen_batch_output.batch["response_mask"][0].sum().item())
        cap_pred_text = tokenizer.decode(cap_response_ids[:cap_valid_len].tolist(), skip_special_tokens=True)

        print(f"  cap_prediction: {cap_pred_text}")
        print(f"  cap_ground_truth: {str(cap_gt)}")

        # print(f"\n  [Segmentation 任务]")
        # print(f"  seg_prompt: {sample.get('seg_prompt', None)}")
        # seg_task_dict = {
        #     k.replace("seg_", ""): sample[k].unsqueeze(0)
        #     for k in ["seg_input_ids", "seg_attention_mask", "seg_position_ids"]
        # }
        # seg_task_dict["raw_prompt_ids"] = np.array([sample["seg_raw_prompt_ids"]], dtype=object)
        # if sample.get("seg_multi_modal_data") is not None:
        #     seg_task_dict["multi_modal_data"] = np.array([sample["seg_multi_modal_data"]], dtype=object)
        # seg_gen_batch = DataProto.from_single_dict(seg_task_dict, meta_info=meta_info)
        # seg_gen_batch_output = _generate_sequences_with_vllm(
        #     seg_gen_batch, llm, sampling_params, tokenizer.pad_token_id
        # )
        # seg_response_ids = seg_gen_batch_output.batch["responses"][0]
        # seg_valid_len = int(seg_gen_batch_output.batch["response_mask"][0].sum().item())
        # seg_pred_text = tokenizer.decode(seg_response_ids[:seg_valid_len].tolist(), skip_special_tokens=True)

        # print(f"  seg_prediction: {seg_pred_text}")
        # print(f"  seg_ground_truth: {str(seg_gt)}")

        # ========== 多模态数据信息 ==========
        print(f"\n  [多模态信息]")
        if is_video:
            print(f"  类型: 视频")
            print(f"  视频路径: {raw_example['videos']}")
        elif is_image:
            print(f"  类型: 图像")
            img_paths = raw_example['images']
            if isinstance(img_paths, list):
                print(f"  图像数量: {len(img_paths)}")
                print(f"  图像路径: {img_paths[0] if isinstance(img_paths[0], str) else '(bytes)'}")
            else:
                print(f"  图像路径: {img_paths}")
        else:
            print(f"  类型: 纯文本")

    print(f"\n{'=' * 60}")
    print(f"测试完成，共处理 {len(indices)} 个样本")
    print(f"{'=' * 60}")
