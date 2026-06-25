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


NO_THINK_PREFIX = "<think>\n\n</think>\n\n"


def add_no_think_prefix(prompt: str, enabled: bool) -> str:
    if not enabled:
        return prompt
    if prompt.endswith(NO_THINK_PREFIX):
        return prompt
    return prompt + NO_THINK_PREFIX


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


def get_video_metadata(video_path: str) -> dict:
    """Get native video metadata (real fps and total frame count) via decord.VideoReader."""
    from decord import VideoReader
    vr = VideoReader(video_path)
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
) -> Union[list[ImageObject], tuple[list[ImageObject], float], tuple[list[ImageObject], dict]]:
    # Force all videos to fixed 224x224 for deterministic spatial size.
    vision_info = {
        "video": video,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "resized_height": 224,
        "resized_width": 224,
    }
    # Use explicit nframes if provided (for deterministic frame count across
    # dataset tokenization and feature extraction). Otherwise use fps-based sampling.
    if nframes is not None:
        vision_info["nframes"] = nframes
    else:
        vision_info["fps"] = video_fps
    if return_video_metadata:
        (frames, video_metadata), sample_fps = fetch_video(
            vision_info,
            return_video_sample_fps=True,
            return_video_metadata=True,
        )
        video_metadata = dict(video_metadata)
        return frames, video_metadata
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

        if filter_overlong_prompts:
            # Drops samples whose EXPANDED image-token prompt exceeds max_prompt_length.
            # Required for multi-image samples: otherwise the prompt is silently truncated
            # (losing image_pad tokens) while pixel_values keeps all images, causing
            # "Image features and image tokens do not match" in the model forward.
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_gen_seg_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        is_video_sample = self.video_key in example
        media_token = "<video>" if is_video_sample else "<image>"
        # 视频样本使用时间戳定位模板；图像样本沿用 bbox/mask 模板
        if is_video_sample:
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
            seg_prompt = self.processor.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = add_no_think_prefix(seg_prompt, self.enable_no_think_prefix)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_metadata_list = []
            stored_nframes = example.get('nframes')
            for i, video in enumerate(videos):
                nf = stored_nframes[i] if stored_nframes is not None else None
                processed_video, video_metadata = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_video_metadata=True, nframes=nf
                )
                processed_videos.append(processed_video)
                video_metadata_list.append(video_metadata)

            seg_model_inputs = self.processor(
                videos=processed_videos, text=[seg_prompt], add_special_tokens=False, return_tensors="pt",
                videos_kwargs={"video_metadata": video_metadata_list, "do_sample_frames": False},
            )
            # Fallback: manually compute second_per_grid_ts if processor didn't provide it
            if "second_per_grid_ts" in self.processor.model_input_names:
                seg_model_inputs["second_per_grid_ts"] = [2.0 / vm["fps"] for vm in video_metadata_list]

            seg_input_ids = seg_model_inputs.pop("input_ids")[0]
            seg_attention_mask = seg_model_inputs.pop("attention_mask")[0]
            example["seg_multi_modal_data"] = {"videos": videos, "fps": self.video_fps, "nframes": [pv.shape[0] for pv in processed_videos]}
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
            for video in videos:
                processed_video, video_metadata = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_video_metadata=True
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
        masktokenizer_root = "."
        if self.image_key in example and example[self.image_key] is not None:
            if isinstance(example[self.image_key], list):
                example[self.image_key] = [
                    os.path.join(masktokenizer_root, img[2:]) if isinstance(img, str) and img.startswith("./") else img
                    for img in example[self.image_key]
                ]
            elif isinstance(example[self.image_key], str) and example[self.image_key].startswith("./"):
                example[self.image_key] = os.path.join(masktokenizer_root, example[self.image_key][2:])

        # Only apply anti-time-leak instruction for video captioning prompts.
        # Skip tg_grounding: there cap_problem is a *grounding* prompt ("locate the
        # timestamp interval ..."), so suppressing time info contradicts the task
        # and makes the model emit a description instead of a time interval.
        if (
            self.video_key in example
            and example[self.video_key] is not None
            and example.get("source") != "tg_grounding"
        ):
            example["cap_problem"] += (
                "\nImportant: In your answer, do not mention any explicit time information "
                "(timestamps, seconds, minutes, frame indices, time ranges, or phrases like "
                "'during 0-3 seconds'). Replace them with neutral phrases like "
                "'in this segment'. Output description only."
            )
        
        # example: ['images', 'seg_answer', 'cap_answer', 'seg_problem', 'masks', 'source']
        # 如果超过5个目标，随机保留5个；否则不变
        # print('before: ', example)
        example[self.cap_prompt_key], example[self.seg_answer_key], example[self.image_key] = sample_single_target_from_multi_target(
            prompt=example.get(self.cap_prompt_key, ""),
            ground_truth=example.get(self.seg_answer_key, None),
            images=example.get(self.image_key, None),
            max_targets=1,
            region_format=self.region_format,
        )
        # print('after: ', example)
        cap_messages, seg_messages = self._build_messages(example)

        # messages: ['role', 'content']
        example.pop(self.cap_prompt_key, None)
        example.pop(self.seg_prompt_key, None)
        if self.image_key in example:
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
        elif self.video_key in example:
            cap_prompt = self.processor.apply_chat_template(cap_messages, add_generation_prompt=True, tokenize=False)
            seg_prompt = self.processor.apply_chat_template(seg_messages, add_generation_prompt=True, tokenize=False)
            cap_prompt = add_no_think_prefix(cap_prompt, self.enable_no_think_prefix)
            seg_prompt = add_no_think_prefix(seg_prompt, self.enable_no_think_prefix)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_metadata_list = []
            for video in videos:
                processed_video, video_metadata = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_video_metadata=True
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
            # Fallback: manually compute second_per_grid_ts if processor didn't provide it
            if "second_per_grid_ts" in self.processor.model_input_names:
                cap_model_inputs["second_per_grid_ts"] = [2.0 / vm["fps"] for vm in video_metadata_list]
                seg_model_inputs["second_per_grid_ts"] = [2.0 / vm["fps"] for vm in video_metadata_list]

            cap_input_ids = cap_model_inputs.pop("input_ids")[0]
            seg_input_ids = seg_model_inputs.pop("input_ids")[0]
            cap_attention_mask = cap_model_inputs.pop("attention_mask")[0]
            seg_attention_mask = seg_model_inputs.pop("attention_mask")[0]
            example["cap_multi_modal_data"] = {"videos": videos, "fps": self.video_fps, "nframes": [pv.shape[0] for pv in processed_videos]}
            example["seg_multi_modal_data"] = {"videos": videos, "fps": self.video_fps, "nframes": [pv.shape[0] for pv in processed_videos]}
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
        for vi, video in enumerate(multi_modal_data["videos"]):
            nf = nframes_list[vi] if nframes_list is not None else None
            frames, metadata = process_video(video, min_pixels, max_pixels, video_fps, return_video_metadata=True, nframes=nf)
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
