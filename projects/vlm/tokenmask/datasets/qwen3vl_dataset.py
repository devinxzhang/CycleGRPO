import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence
from pycocotools import mask as mask_utils

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
# from torchcodec.decoders import VideoDecoder
import transformers
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER

from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

from projects.vlm.tokenmask.datasets.data import data_list
from projects.vlm.tokenmask.datasets.rope2d import get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": img} for img in images
    ]
    video_pool = [
        {"type": "video", "video": vid} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages

def preprocess_qwen_visual(
    sources,
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


def _normalize_conversations(obj):
    # 如果是字符串，尽量先解析为 Python 对象
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            # 保底兜底：包成 assistant 一条
            return [{"from": "human", "value": obj}]

    # 如果是单条 dict，包成 list
    if isinstance(obj, dict):
        return [obj]

    # 如果已经是 list，确认每一项都是 dict；不是的话兜底包装
    if isinstance(obj, list):
        fixed = []
        for x in obj:
            if isinstance(x, dict):
                fixed.append(x)
            elif isinstance(x, str):
                # 尝试解析字符串消息
                try:
                    xj = json.loads(x)
                    fixed.append(xj if isinstance(xj, dict) else {"from": "human", "value": x})
                except Exception:
                    fixed.append({"from": "human", "value": x})
            else:
                # 其他类型兜底
                fixed.append({"from": "human", "value": str(x)})
        return fixed

    # 其他类型兜底
    return [{"from": "human", "value": str(obj)}]


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args["min_pixels"]
        ip.max_pixels = data_args["max_pixels"]
        rank0_print(f"✅ Updated image_processor min_pixels to {ip.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {ip.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args["min_pixels"]
        ip.size["longest_edge"] = data_args["max_pixels"]
        min_pixels = data_args["min_pixels"]
        max_pixels = data_args["max_pixels"]
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args["video_min_pixels"]
            vp.max_pixels = data_args["video_max_pixels"]
            video_min_pixels = data_args["video_min_pixels"]
            video_max_pixels = data_args["video_max_pixels"]
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args["video_min_frames"]
            vp.max_frames = data_args["video_max_frames"]
            rank0_print(
                f"✅ Updated video_processor min_frames to {vp.min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {vp.max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args["video_fps"]
            rank0_print(f"✅ Updated video_processor fps to {vp.fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args["video_min_pixels"]
            vp.size["longest_edge"] = data_args["video_max_pixels"]
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor

class Qwen3VLDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer, 
        data_args,
        sam_preprocessor,
    ):
        super(Qwen3VLDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = data_args.get(
            "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = data_args.get(
            "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.get("model_type", "qwen3vl")
        self.get_rope_index = get_rope_index_3

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                with open(data["annotation_path"], "r") as f:
                    annotations = json.load(f)
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
                annotations = annotations * int(sampling_rate)
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        # self.tokenizer = BUILDER.build(tokenizer)
        self.list_data_dict = list_data_dict
        image_processor = data_args['image_processor']
        image_processor = BUILDER.build(image_processor)
        image_processor = update_processor_pixels(image_processor, data_args)
        self.processor = image_processor
        self.tokenizer = image_processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(image_processor.image_processor, "merge_size", 2)

        self.data_args = data_args

        self.sam_preprocessor = BUILDER.build(sam_preprocessor)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            new_conversations = _normalize_conversations(sample["conversations"])
            length_list.append(
                sum(len(conv["value"].split()) for conv in new_conversations)
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            new_conversations = _normalize_conversations(sample["conversations"])
            cur_len = sum(
                len(conv["value"].split()) for conv in new_conversations
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def modality_length(self):
        return self.modality_lengths

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))
    
    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        _max_refetch = 1000
        index = i
        for _ in range(_max_refetch + 1):
            try:
                sample = self._get_item(index)
                if sample is None:
                    print(
                        f"[Try other #{_+1}] Failed to fetch sample {i}."
                    )
                    index = self._rand_another()
                else:
                    return sample
            except Exception as e:
                print(
                    f"[Try other #{_+1}] Failed to fetch sample {i}. Exception:",
                    e,
                )
                index = self._rand_another()
        

    def get_data(self, input_sources) -> Dict[str, torch.Tensor]:
        assert len(input_sources) == 1

        data_dict = preprocess_qwen_visual(
            input_sources,
            self.processor,
        )

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        # text = self.processor.tokenizer.decode(
        #     data_dict["input_ids"][0], skip_special_tokens=False
        # )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        # label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_item(self, i) -> Dict[str, torch.Tensor]:

        sources = self.list_data_dict[i]

        if isinstance(sources, dict):
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self.get_data(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(i, int):
                    source = [source]
                assert (
                    len(source) == 1
                ), "Don't know why it is wrapped to a list"  # FIXME
                data_list.append(self.get_data(source))
                if data_list[-1] is None:
                    return None

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None
            }
            
            if any("pixel_values" in d for d in data_list):
                new_data_dict.update({
                    "pixel_values": torch.cat([d["pixel_values"] for d in data_list if "pixel_values" in d], dim=0),
                    "image_grid_thw": torch.cat([d["image_grid_thw"] for d in data_list if "image_grid_thw" in d], dim=0)
                })
            
            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update({
                    "pixel_values_videos": torch.cat([d["pixel_values_videos"] for d in data_list if "pixel_values_videos" in d], dim=0),
                    "video_grid_thw": torch.cat([d["video_grid_thw"] for d in data_list if "video_grid_thw" in d], dim=0)
                })
            return new_data_dict

