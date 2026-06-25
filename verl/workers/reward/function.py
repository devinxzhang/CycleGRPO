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

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict
import re
import copy
from PIL import Image
from io import BytesIO
import numpy as np
from pycocotools import mask as mask_utils
import hydra

import torch
from torchvision.transforms.functional import to_pil_image
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig

class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]

def decode_mask(object_masks, ori_height, ori_width):
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
    return binary_masks

class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer


    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            # # pred masks
            # pred_masks = data.non_tensor_batch["pred_masks"][i]
            # if pred_masks is not None:
            #     pred_masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in pred_masks])

            # # gt masks
            # segmentations = data.non_tensor_batch["masks"][i]
            # if segmentations is None:
            #     gt_masks = None
            # else:
            #     ori_height, ori_width = segmentations[0]['size']
            #     gt_masks = decode_mask(segmentations, ori_height, ori_width)
            #     gt_masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in gt_masks])
            pred_masks = None
            gt_masks = None
            
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                    "source": data.non_tensor_batch["source"][i],
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, step: int, data: DataProto, task: str) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        # ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        mask_token_accuracy = data.non_tensor_batch.get("mask_token_accuracy") if task == 'segmentation' else None
        seg_ground_truth = data.non_tensor_batch.get("seg_ground_truth")
        cap_ground_truth = data.non_tensor_batch.get("cap_ground_truth")
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "source": data.non_tensor_batch["source"][i],
                    "task": task,
                    "iou_scores": data.non_tensor_batch["iou_scores"][i] if data.non_tensor_batch["source"][i] in ['groundingme', 'denseworld_single', 'denseworld_multiple', 'tg_multi_merged', 'dam_cyclegrpo', None] else None,
                    "mask_token_accuracy": data.non_tensor_batch["mask_token_accuracy"][i] if task == 'segmentation' else None,
                    # "correct_mask": data.non_tensor_batch["correct_mask"][i], 
                    # "image": images[i]['images'][0] if task == 'segmentation' else None,
                    # "gt_masks": masks[i] if task == 'segmentation' else None,
                    "cap_ground_truth": cap_ground_truth[i] if cap_ground_truth is not None else None,
                    "seg_ground_truth": seg_ground_truth[i] if task == 'segmentation' else None,
                    "extra_info": data.non_tensor_batch["extra_info"][i] if "extra_info" in data.non_tensor_batch else None,
                    # "cap_images": cap_images[i] if task == 'segmentation' else None,
                    # "cap_responses": cap_responses[i] if task == 'segmentation' else None,
                }
            )

        # if step % 5 == 1:
        #     torch.save(reward_inputs, f"grpo_analysis/cap_reward_inputs_{step}.pt")

        scores = self.reward_fn(reward_inputs, task=task)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["cap_overall"] if task == 'caption' else score["seg_overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)
            if task == 'caption' and data.non_tensor_batch["source"][i] in ['groundingme', 'denseworld_single', 'denseworld_multiple', 'tg_multi_merged', 'dam_cyclegrpo', '', None]:
                reward_metrics['cap_correct_mask'] = data.non_tensor_batch["correct_mask"][i]

        return reward_tensor, reward_metrics


