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

import math
import os
import re
import random
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union, Tuple, List

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

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


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)

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
            
            # # 禁用 Image 自动解码，保持为字符串路径
            # from datasets import Sequence, Value, Image as HFImage
            # new_features = ds.features.copy()
            # schema_changed = False
            # for col_name, col_type in ds.features.items():
            #     # 检查是否是 Sequence(Image) 类型，转换为 Sequence(string)
            #     if hasattr(col_type, 'feature') and isinstance(col_type.feature, HFImage):
            #         new_features[col_name] = Sequence(Value('string'))
            #         schema_changed = True
            #     # 检查是否是单个 Image 类型，转换为 string
            #     elif isinstance(col_type, HFImage):
            #         new_features[col_name] = Value('string')
            #         schema_changed = True
            # if schema_changed:
            #     ds = ds.cast(new_features)
            #     print(f"Casted Image columns to string for {single_path}")
            
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

        # if filter_overlong_prompts:
        #     self.dataset = self.dataset.filter(
        #         self._filter_overlong_prompts,
        #         desc="Filtering overlong prompts",
        #         num_proc=filter_overlong_prompts_workers,
        #     )

    def _build_gen_seg_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        # 根据 region_format 选择不同的 prompt 模板
        if self.region_format == "bbox":
            PROMPT_TEMPLATE = """<image>\nAll spatial relationships are defined from the viewer's perspective, where 'front' means closer to the viewer and 'back' means farther from the viewer. Please provide the bounding box of the object the following statement describes:
        {description}
        Ensure that all details mentioned about the object are accurate. If a matching object is found, provide its bounding box in the format `[x1, y1, x2, y2]` where coordinates are normalized to [0, 1000]. If no matching object is found, output null."""
        else:
            PROMPT_TEMPLATE = """<image>\nAll spatial relationships are defined from the viewer's perspective, where 'front' means closer to the viewer and 'back' means farther from the viewer. Please provide the segmentation mask of the object the following statement describes:
        {description}
        Ensure that all details mentioned about the object are accurate. If a matching object is found, provide its segmentation mask in the format `<|mt_start|><|mt_xxxx|><|mt_xxxx|><|mt_end|>`. If no matching object is found, output null."""

        prompt_str: str = example[self.seg_prompt_key]
        prompt_str = PROMPT_TEMPLATE.format(description=prompt_str)

        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

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
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            seg_model_inputs = self.processor(processed_images[:1], [seg_prompt], add_special_tokens=False, return_tensors="pt")
            seg_input_ids = seg_model_inputs.pop("input_ids")[0]
            seg_attention_mask = seg_model_inputs.pop("attention_mask")[0]
            example["seg_multi_modal_data"] = {"images": images[:1]}
        elif self.video_key in example:
            seg_prompt = self.processor.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            # TODO: check the video length
            seg_model_inputs = self.processor(
                videos=processed_videos[:1], text=[seg_prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                seg_model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            seg_input_ids = seg_model_inputs.pop("input_ids")[0]
            seg_attention_mask = seg_model_inputs.pop("attention_mask")[0]
            # TODO: check the video length
            example["seg_multi_modal_data"] = {"videos": videos[:1]}
        else:
            seg_prompt = self.tokenizer.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
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

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        cap_prompt_str: str = example[self.cap_prompt_key]
        seg_prompt_str: str = example[self.seg_prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            cap_prompt_str = format_prompt.render(content=cap_prompt_str)
            seg_prompt_str = format_prompt.render(content=seg_prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            cap_content_list = []
            for i, content in enumerate(cap_prompt_str.split("<image>")):
                if i != 0:
                    cap_content_list.append({"type": "image"})

                if content:
                    cap_content_list.append({"type": "text", "text": content})

            seg_content_list = []
            for i, content in enumerate(seg_prompt_str.split("<image>")):
                if i != 0:
                    seg_content_list.append({"type": "image"})

                if content:
                    seg_content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": cap_content_list}], [{"role": "user", "content": seg_content_list}]
        elif self.video_key in example:
            cap_content_list = []
            for i, content in enumerate(cap_prompt_str.split("<video>")):
                if i != 0:
                    cap_content_list.append({"type": "video"})

                if content:
                    cap_content_list.append({"type": "text", "text": content})

            seg_content_list = []
            for i, content in enumerate(seg_prompt_str.split("<video>")):
                if i != 0:
                    seg_content_list.append({"type": "video"})

                if content:
                    seg_content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": cap_content_list}], [{"role": "user", "content": seg_content_list}]
        else:
            return [{"role": "user", "content": cap_prompt_str}], [{"role": "user", "content": seg_prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        cap_messages, seg_messages = self._build_messages(example)
        if self.image_key in example:
            cap_prompt = self.processor.apply_chat_template(cap_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = self.processor.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
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
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            cap_model_inputs = self.processor(
                videos=processed_videos, text=[cap_prompt], add_special_tokens=False, return_tensors="pt"
            )
            ## TODO: check the video length
            seg_model_inputs = self.processor(
                videos=processed_videos[:1], text=[seg_prompt], add_special_tokens=False, return_tensors="pt"
            )
            return cap_model_inputs["input_ids"].size(-1) <= self.max_prompt_length and seg_model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            cap_input_ids = self.tokenizer.apply_chat_template(cap_messages, add_generation_prompt=True)
            seg_input_ids = self.tokenizer.apply_chat_template(seg_messages, add_generation_prompt=True)
            return len(cap_input_ids) <= self.max_prompt_length and len(seg_input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        # example: ['images', 'seg_answer', 'cap_answer', 'seg_problem', 'masks', 'source']
        # 如果超过5个目标，随机保留5个；否则不变
        # print('before: ', example)
        # example[self.cap_prompt_key], example[self.seg_answer_key], example[self.image_key] = sample_single_target_from_multi_target(
        #     prompt=example.get(self.cap_prompt_key, ""),
        #     ground_truth=example.get(self.seg_answer_key, None),
        #     images=example.get(self.image_key, None),
        #     max_targets=1,
        #     region_format=self.region_format,
        # )
        # print('after: ', example)
        cap_messages, seg_messages = self._build_messages(example)
        # messages: ['role', 'content']
        example.pop(self.cap_prompt_key, None)
        example.pop(self.seg_prompt_key, None)
        if self.image_key in example:
            cap_prompt = self.processor.apply_chat_template(cap_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = self.processor.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
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
        elif self.video_key in example:
            cap_prompt = self.processor.apply_chat_template(cap_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = self.processor.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            cap_model_inputs = self.processor(
                videos=processed_videos, text=[cap_prompt], add_special_tokens=False, return_tensors="pt"
            )
            # TODO: check the video length
            seg_model_inputs = self.processor(
                videos=processed_videos[:1], text=[seg_prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                cap_model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]
                seg_model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            cap_input_ids = cap_model_inputs.pop("input_ids")[0]
            seg_input_ids = seg_model_inputs.pop("input_ids")[0]
            cap_attention_mask = cap_model_inputs.pop("attention_mask")[0]
            seg_attention_mask = seg_model_inputs.pop("attention_mask")[0]
            example["cap_multi_modal_data"] = {"videos": videos}
            # TODO: check the video length
            example["seg_multi_modal_data"] = {"videos": videos[:1]}
        else:
            cap_prompt = self.tokenizer.apply_chat_template(cap_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = self.tokenizer.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
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
        # example["seg_input_ids"] = seg_input_ids
        example["cap_attention_mask"] = cap_attention_mask
        # example["seg_attention_mask"] = seg_attention_mask
        example["cap_position_ids"] = cap_position_ids
        # example["seg_position_ids"] = seg_position_ids
        example["cap_raw_prompt_ids"] = cap_raw_prompt_ids
        # example["seg_raw_prompt_ids"] = seg_raw_prompt_ids
        example["cap_ground_truth"] = example.pop(self.cap_answer_key)
        example["seg_ground_truth"] = example.pop(self.seg_answer_key)
        return example
