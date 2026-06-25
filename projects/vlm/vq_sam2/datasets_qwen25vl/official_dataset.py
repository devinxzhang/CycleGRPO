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

from projects.vlm.vq_sam2.datasets_qwen25vl.data import data_list
from projects.vlm.vq_sam2.datasets_qwen25vl.rope2d import get_rope_index_25, get_rope_index_2

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


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
    grid_thw_image_not_merged: List = [],
    grid_thw_video_not_merged: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        if isinstance(source, str):
            source = json.loads(source)

        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        if grid_thw_image is not None and len(grid_thw_image) != 0:
            assert source[0]['from'] == 'human'
            content = source[0]['value']
            if '<image>' not in content:
                source[0]['value'] = "<image>\n" + content
                # print(source)
                # exit(0)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            # replace \"bbox_2d\" with actual absolute bbox coordinates
            if 'bbox_2d' in conv:
                normalized_bbox_2d = conv['bbox_2d']
                image_idx = conv["image_id"] if "image_id" in conv else 0
                grid_thw = grid_thw_image_not_merged[image_idx]
                
                actual_h, actual_w = grid_thw[1] * 14, grid_thw[2] * 14
                if isinstance(normalized_bbox_2d, List):
                    abs_bbox_2d = [
                        int(actual_w * normalized_bbox_2d[0] + 0.5),
                        int(actual_h * normalized_bbox_2d[1] + 0.5),
                        int(actual_w * normalized_bbox_2d[2] + 0.5),
                        int(actual_h * normalized_bbox_2d[3] + 0.5)
                    ]
                    content = content.format(bbox_2d=abs_bbox_2d)
                elif isinstance(normalized_bbox_2d, Dict):
                    for k, v in normalized_bbox_2d.items():
                        abs_bbox_2d = [
                            int(actual_w * v[0] + 0.5),
                            int(actual_h * v[1] + 0.5),
                            int(actual_w * v[2] + 0.5),
                            int(actual_h * v[3] + 0.5)
                        ]
                        abs_bbox_2d_str = f"{abs_bbox_2d}"
                        content = content.replace(k, abs_bbox_2d_str)
                else:
                    raise NotImplementedError
            
            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|video_pad|>"
                            * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


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


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer, 
        data_args,
        sam_preprocessor,
    ):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = data_args.get(
            "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = data_args.get(
            "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.get("model_type", "qwen2.5vl")
        if self.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

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
        self.tokenizer = BUILDER.build(tokenizer)
        self.list_data_dict = list_data_dict
        image_processor = data_args['image_processor']
        image_processor = BUILDER.build(image_processor).image_processor
        image_processor.max_pixels = data_args.get("max_pixels")
        image_processor.min_pixels = data_args.get("min_pixels")
        image_processor.size["longest_edge"] = data_args.get("max_pixels")
        image_processor.size["shortest_edge"] = data_args.get("min_pixels")
        data_args.update({'image_processor': image_processor})
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

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args["image_processor"])
        try:
            image = Image.open(image_file).convert("RGB")
        except:
            return None
        
        ori_width, ori_height = image.size
        if ori_width < ori_height and ori_width < 28:
            resized_image = image.resize((28, int(ori_height/ori_width * 28)))
        elif ori_height < ori_width and ori_height < 28:
            resized_image = image.resize((int(ori_width/ori_height * 28), 28))
        elif ori_height==ori_width and ori_width < 28:
            return None
        else:
            resized_image = image

        visual_processed = processor.preprocess(resized_image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 3
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(video_file)
                return decord_video
                if decord_video:
                    break
            except Exception as e:
                print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_file)
            return torchcodec_video
        except Exception as e:
            print(f"torchcodec attempt failed: {e}")

    def video_decord(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
            return None
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts
    
    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))
    
    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            if isinstance(object_mask, dict):
                if isinstance(object_mask["counts"], list):
                    # convert to compressed RLE
                    object_mask = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
                m = mask_utils.decode(object_mask)
                m = m.astype(np.uint8).squeeze()
            elif object_mask:
                rles = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
                rle = mask_utils.merge(rles)
                m = mask_utils.decode(rle).astype(np.uint8).squeeze()
            else:
                m = np.zeros((ori_height, ori_width), dtype=np.uint8)
            binary_masks.append(m)
        if len(binary_masks) == 0:
            binary_masks.append(np.zeros((ori_height, ori_width), dtype=np.uint8))
        masks = np.stack(binary_masks, axis=0)
        return masks

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
        # define some variables
        image = None
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        new_sources = []
        for source in input_sources:
            copied_source = {}
            for k, v in source.items():
                if k == "image" and len(v) > 0 and v is not None:
                    copied_source[k] = copy.deepcopy(v)
                elif k == "image":
                    continue
                else:
                    copied_source[k] = copy.deepcopy(v)
            new_sources.append(copied_source)
        sources = new_sources
            
        if "image" in sources[0] and len(sources[0]["image"]) != 0:
            image_folder = sources[0]["data_path"]
            image_file = sources[0]["image"]

            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    results = [self.process_image_unified(file) for file in image_file]
                    if None in results:
                        return None
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    ret__ = self.process_image_unified(image_file)
                    if ret__ is None:
                        return None
                    image, grid_thw = ret__
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                ret__ = self.process_image_unified(image_file)
                if ret__ is None:
                    return None
                image, grid_thw = ret__
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args["image_processor"].merge_size**2
                for merged_thw in grid_thw_merged
            ]
        if "video" in sources[0]:
            video_file = sources[0]["video"]
            video_folder = sources[0]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, video_grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, video_grid_thw, second_per_grid_ts = self.process_video(
                        video_file
                    )
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, video_grid_thw, second_per_grid_ts = self.process_video(
                    video_file
                )
                video = [video]
            video_grid_thw_merged = copy.deepcopy(video_grid_thw)
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw_merged = [video_grid_thw_merged]
                video_grid_thw = [video_grid_thw]
            video_grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in video_grid_thw_merged
            ]
        # chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        chat_sources = []
        for e in sources:
            convs = _normalize_conversations(e["conversations"])
            chat_sources.append(convs)
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
            grid_thw_image_not_merged=grid_thw,
            grid_thw_video_not_merged=video_grid_thw,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args["image_processor"].merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        if "image" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            # sources = copy.deepcopy([e["conversations"] for e in sources])
            chat_sources = []
            for e in sources:
                convs = _normalize_conversations(e["conversations"])
                chat_sources.append(convs)
            data_dict = preprocess_qwen_2_visual(
                chat_sources, self.tokenizer, grid_thw_image=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        data_dict["position_ids"] = position_ids

        # handle too long sequence
        if data_dict["input_ids"][0].size(0) > self.tokenizer.model_max_length:
            truncate_input_ids = data_dict["input_ids"][:, :self.tokenizer.model_max_length]
            truncate_labels = data_dict["labels"][:, :self.tokenizer.model_max_length]
            truncate_position_ids = data_dict["position_ids"][:, :, :self.tokenizer.model_max_length]
            data_dict.update({
                "input_ids": truncate_input_ids,
                "labels": truncate_labels,
                "position_ids": truncate_position_ids
            })
            print("self.tokenizer.model_max_length: ", self.tokenizer.model_max_length)
            for key, value in data_dict.items():
                print(key, value.shape)
            
            if any(truncate_input_ids[:, -1]==IMAGE_TOKEN_INDEX) or any(truncate_input_ids[:, -1]==VIDEO_TOKEN_INDEX):
                print("Visual tokens were truncated!!!!! Skip this case")
                return None
        
        if torch.all(data_dict['labels']==IGNORE_INDEX):
            return None

        data_dict["attention_mask"] = torch.LongTensor([1] * data_dict["input_ids"][0].size(0))

        if "image" in sources[0] and len(sources[0]["image"]) != 0 and image is not None:
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw], dim=0
            )
        # video exist in the data
        elif "video" in sources[0]:
            data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
            data_dict["video_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
            )

        # handdle mask generation data
        if "segmentation" not in sources[0] or sources[0]["segmentation"] is None:
            data_dict["sam2_pixel_values"] = None
            data_dict["masks"] = None
        else:
            if not isinstance(image_file, List):
                image_file = [image_file]
            images = [Image.open(_image_file_).convert('RGB') for _image_file_ in image_file]
            sam2_images = [np.array(image) for image in images]
            sam2_images = [self.sam_preprocessor.apply_image(image) for image in sam2_images]
            sam2_pixel_values = [torch.from_numpy(image).permute(2, 0, 1).contiguous() for image in sam2_images]
            segmentation_image_indices = sources[0]['segmentation_image_indices']
            data_dict["sam2_pixel_values"] = [sam2_pixel_values[img_idx] for img_idx in segmentation_image_indices]        
            
            segmentations = sources[0]["segmentation"]
            masks = []
            for img_idx, segm in zip(segmentation_image_indices, segmentations):
                ori_width, ori_height = images[img_idx].size
                mask = self.decode_mask([segm], ori_height, ori_width)
                mask = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in mask])
                masks.append(mask)
            data_dict['masks'] = masks
            
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

