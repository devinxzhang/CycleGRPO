import copy
import random
import glob
import json
import logging
import os
from typing import Literal, Dict, Optional, Sequence, List, Tuple

import torch
import transformers

from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset

from projects.vlm.qwen2_5_vl_vq_sam2.datasets.rope2d import get_rope_index_25

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
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
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

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

class LLaVADataset(Dataset):

    IMG_CONTEXT_TOKEN = '<|image_pad|>'
    IMG_START_TOKEN = '<|vision_start|>'
    IMG_END_TOKEN = '<|vision_end|>'

    def __init__(
        self,
        tokenizer,
        data_path,
        image_folder=None,
        max_length=8192,
        preprocessor=None,
        repeats=1,
    ):
        super().__init__()
        self.tokenizer = BUILDER.build(tokenizer)

        self.data_path = data_path
        self.image_folder = image_folder
        self.max_length = max_length
        self.repeats = repeats

        self.get_rope_index = get_rope_index_25

        self.preprocessor = BUILDER.build(preprocessor)

        self.data = self._load_annotations(data_path)

        self._max_refetch = 1000

    def _load_annotations(self, data_path):
        data = json.load(open(data_path))
        return data
    
    @property
    def modality_length(self):
        length_list = [100] * int(len(self.data) * self.repeats)
        return length_list
    
    def __len__(self):
        return int(len(self.data) * self.repeats)
    
    @property
    def real_len(self):
        return len(self.data)
    
    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))
    
    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.preprocessor.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw
    
    def prepare_data(self, index):
        data_dict = self.data[index]

        grid_thw_merged = None
        grid_thw = None

        # print("=============================>", self.tokenizer.convert_tokens_to_ids("<|mt_start|>"))

        if data_dict.get('image', None) is not None:
            image_path = data_dict['image']
            if self.image_folder is not None:
                image_path = os.path.join(self.image_folder, image_path)
        
            if 'dam_data/sam' in image_path:
                image_path = image_path.replace('dam_data/sam', 'dam_data/SAM/images')

            image_processor = copy.deepcopy(self.preprocessor.image_processor)
            image, grid_thw = self.process_image_unified(image_path)
            image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]

        chat_sources = copy.deepcopy([data_dict["conversations"]])
        out_data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=None,
        )
        position_ids, _ = self.get_rope_index(
            self.preprocessor.image_processor.merge_size,
            out_data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=None,
            second_per_grid_ts=None,
        )
        if data_dict.get('image', None) is None:
            grid_thw_merged = None
            sources = copy.deepcopy([data_dict["conversations"]])
            out_data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw_image=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, out_data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )
        
        out_data_dict["position_ids"] = position_ids
        out_data_dict["attention_mask"] = [out_data_dict["input_ids"][0].size(0)]
        
        if data_dict.get('image', None) is not None:
            out_data_dict["pixel_values"] = torch.cat(image, dim=0)
            out_data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw], dim=0
            )
        
        return out_data_dict
        

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            if self.repeats >= 1:
                real_index = index % self.real_len
            else:
                real_index = random.randint(0, self.real_len-1)
            try:
                data = self.prepare_data(real_index)
            except:
                data = None
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data