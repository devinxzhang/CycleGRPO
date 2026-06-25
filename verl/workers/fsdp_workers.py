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
"""
The main entry point to run the PPO algorithm
"""

from typing import Literal, Optional, Union, cast, Tuple, List

import re
import hydra
import copy
from PIL import Image
from io import BytesIO
import numpy as np
import psutil
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image
from accelerate import init_empty_weights
from codetiming import Timer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    GenerationConfig,
    PreTrainedModel,
)
try:
    from transformers.modeling_utils import no_init_weights  # old HF
except ImportError:
    try:
        from transformers.utils import no_init_weights  # mid HF
    except ImportError:
        from contextlib import contextmanager
        import torch.nn as nn
        from transformers.modeling_utils import PreTrainedModel
        @contextmanager
        def no_init_weights():
            old_linear = nn.Linear.reset_parameters
            old_init = PreTrainedModel.init_weights
            nn.Linear.reset_parameters = lambda self: None
            PreTrainedModel.init_weights = lambda self: None
            try:
                yield
            finally:
                nn.Linear.reset_parameters = old_linear
                PreTrainedModel.init_weights = old_init

from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.dataset import process_image, process_video
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import AnyPrecisionAdamW, get_constant_schedule_with_warmup
from .config import ActorConfig, CriticConfig, FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from .rollout import vLLMRollout
from .sharding_manager import FSDPVLLMShardingManager
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from projects.transformers.vq_sam2 import SAM2Config, VQ_SAM2Config, VQ_SAM2

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))
    
def extract_mt_token_ids_v1(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]

def extract_mt_token_ids_v2(text):
    pattern = re.compile(r'<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>')
    matches = pattern.findall(text)
    ret_list = []
    for num1, num2 in matches:
        ret_list.append(int(num1))
        ret_list.append(int(num2))
    return ret_list

def fix_mt_format_comprehensive(text):
    """
    全面修正 <|mt_...> 格式的函数。
    它会处理以下几种情况：
    1. 标记太少 (1个): <|mt_start|><|mt_0198|><|mt_end|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    2. 标记太少 (1个, 无end): <|mt_start|><|mt_0198|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    3. 标记太多 (3个或以上): <|mt_start|><|mt_0186|><|mt_0410|><|mt_0186|><|mt_end|> -> <|mt_start|><|mt_0186|><|mt_0410|><|mt_end|>
    4. 正确格式: <|mt_start|><|mt_0044|><|mt_0442|><|mt_end|> -> 不变
    """
    # 规则 1: 处理标记太多的情况 (3个或以上)
    # 捕获前两个，匹配掉多余的，然后用前两个重构
    pattern_too_many = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>'
    replacement_too_many = r'\1\2\3<|mt_end|>'
    text = re.sub(pattern_too_many, replacement_too_many, text)
    # 规则 2: 处理标记太少的情况 (只有1个，且有<|mt_end|>)
    pattern_too_few_with_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)'
    replacement_too_few = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_with_end, replacement_too_few, text)
    # 规则 3: 处理标记太少的情况 (只有1个，且没有<|mt_end|>)
    # 使用负向前瞻确保后面不是另一个mt_token
    pattern_too_few_no_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)'
    replacement_too_few_no_end = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_no_end, replacement_too_few_no_end, text)
    return text

def extract_bbox_from_response(response_str: str) -> Optional[Tuple[int, int, int, int]]:
    """
    从 response 字符串中提取 bounding box 坐标。
    
    Args:
        response_str: 模型生成的响应字符串，例如 '[7, 296, 998, 885]<|im_end|>'
    
    Returns:
        Tuple[int, int, int, int]: (x1, y1, x2, y2) 坐标元组，如果未找到则返回 None
    
    Examples:
        >>> extract_bbox_from_response('[7, 296, 998, 885]<|im_end|>')
        (7, 296, 998, 885)
        >>> extract_bbox_from_response('The bbox is [100, 200, 300, 400].')
        (100, 200, 300, 400)
    """
    if response_str is None:
        return None
    
    # 匹配 [x1, y1, x2, y2] 格式的 bbox
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(bbox_pattern, response_str)
    
    if match:
        x1, y1, x2, y2 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        return (x1, y1, x2, y2)
    
    return None

def extract_all_bboxes_from_response(response_str: str) -> List[Tuple[int, int, int, int]]:
    """
    从 response 字符串中提取所有 bounding box 坐标。
    
    Args:
        response_str: 模型生成的响应字符串
    
    Returns:
        List[Tuple[int, int, int, int]]: 所有 (x1, y1, x2, y2) 坐标元组的列表
    
    Examples:
        >>> extract_all_bboxes_from_response('[7, 296, 998, 885] and [100, 200, 300, 400]')
        [(7, 296, 998, 885), (100, 200, 300, 400)]
    """
    if response_str is None:
        return []
    
    # 匹配所有 [x1, y1, x2, y2] 格式的 bbox
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(bbox_pattern, response_str)
    
    return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in matches]

def compute_bbox_iou(
    pred_bbox: Optional[Tuple[int, int, int, int]],
    gt_bbox: Optional[Tuple[int, int, int, int]]
) -> float:
    """
    计算两个 bounding box 之间的 IoU (Intersection over Union)。
    
    Args:
        pred_bbox: 预测的 bbox (x1, y1, x2, y2)，如果为 None 则返回 0.0
        gt_bbox: ground truth 的 bbox (x1, y1, x2, y2)，如果为 None 则返回 0.0
    
    Returns:
        float: IoU 值，范围 [0.0, 1.0]
    
    Examples:
        >>> compute_bbox_iou((0, 0, 100, 100), (50, 50, 150, 150))
        0.14285714285714285
        >>> compute_bbox_iou((0, 0, 100, 100), (0, 0, 100, 100))
        1.0
        >>> compute_bbox_iou(None, (0, 0, 100, 100))
        0.0
    """
    if pred_bbox is None or gt_bbox is None:
        return 0.0
    
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
    
    # 计算交集区域
    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)
    
    # 如果没有交集
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # 计算各自的面积
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    
    # 计算并集面积
    union_area = pred_area + gt_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    iou = inter_area / union_area

    return iou


def extract_time_intervals_from_response(sentence: Optional[str], only_result: bool = False) -> List[List[float]]:
    """从文本中提取时间区间，兼容 `<time>a-b</time>`、`[a, b]` 等格式。"""
    if sentence is None:
        return []

    if only_result:
        think_end_match = re.search(r'</think>', sentence, re.I)
        if think_end_match:
            sentence = sentence[think_end_match.end():]

        answer_match = re.search(r'<answer>(.*?)</answer>', sentence, re.I | re.DOTALL)
        if answer_match:
            sentence = answer_match.group(1)

    intervals = []

    time_blocks = re.findall(r'<time>(.*?)</time>', sentence, flags=re.I | re.DOTALL)
    if time_blocks:
        for blk in time_blocks:
            blk = blk.strip()
            if not blk:
                continue
            match = re.fullmatch(
                r'\s*(\d+(?:\.\d+)?)\s*[-–—~,]\s*(\d+(?:\.\d+)?)\s*(?:seconds?|s)?\s*',
                blk,
                flags=re.I,
            )
            if match:
                start, end = float(match.group(1)), float(match.group(2))
                if start < end:
                    intervals.append([start, end])
        if intervals:
            return intervals

    bracket_matches = re.findall(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', sentence)
    if bracket_matches:
        for start, end in bracket_matches:
            start_val, end_val = float(start), float(end)
            if start_val < end_val:
                intervals.append([start_val, end_val])
        if intervals:
            return intervals

    from_to_matches = re.findall(
        r'[Ff]rom\s+(\d+(?:\.\d+)?)\s*s?\s+to\s+(\d+(?:\.\d+)?)\s*s?',
        sentence,
    )
    if from_to_matches:
        for start, end in from_to_matches:
            start_val, end_val = float(start), float(end)
            if start_val < end_val:
                intervals.append([start_val, end_val])
        if intervals:
            return intervals

    general_matches = re.findall(r'(\d+(?:\.\d+)?)\s*s?\s*[-–—~]\s*(\d+(?:\.\d+)?)\s*s?', sentence)
    if general_matches:
        for start, end in general_matches:
            start_val, end_val = float(start), float(end)
            if start_val < end_val:
                intervals.append([start_val, end_val])
        if intervals:
            return intervals

    return []


def calculate_temporal_iou(gt_windows: List[List[float]], pred_windows: List[List[float]]) -> float:
    """计算多个时间区间的 tIoU。"""

    def merge_intervals(intervals: List[List[float]]) -> List[List[float]]:
        if not intervals:
            return []
        valid_intervals = [[start, end] for start, end in intervals if start < end]
        if not valid_intervals:
            return []

        sorted_intervals = sorted(valid_intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0][:]]
        for current in sorted_intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                merged[-1] = [last[0], max(last[1], current[1])]
            else:
                merged.append(current[:])
        return merged

    merged_gt = merge_intervals(gt_windows)
    merged_pred = merge_intervals(pred_windows)

    if not merged_gt and not merged_pred:
        return 1.0
    if not merged_gt or not merged_pred:
        return 0.0

    total_gt = sum(end - start for start, end in merged_gt)
    total_pred = sum(end - start for start, end in merged_pred)

    intersection = 0.0
    i = j = 0
    while i < len(merged_gt) and j < len(merged_pred):
        gt_start, gt_end = merged_gt[i]
        pred_start, pred_end = merged_pred[j]
        intersect_start = max(gt_start, pred_start)
        intersect_end = min(gt_end, pred_end)
        intersection += max(0.0, intersect_end - intersect_start)

        if gt_end < pred_end:
            i += 1
        else:
            j += 1

    union = total_gt + total_pred - intersection
    return intersection / union if union > 0 else 0.0


def compute_temporal_iou_reward(pred_windows: List[List[float]], gt_windows: List[List[float]]) -> Tuple[float, List[List[float]], List[List[float]]]:
    """计算视频 grounding 的 tIoU reward（输入应为已解析的时间区间）。"""
    if not pred_windows or not gt_windows:
        return 0.0, pred_windows, gt_windows
    return calculate_temporal_iou(gt_windows, pred_windows), pred_windows, gt_windows


def video_grounding_format_reward(response_str: Optional[str]) -> int:
    """检查视频 grounding 输出中是否包含 'X - X seconds' 结构（X 为整数或小数）。"""
    if response_str is None:
        return 0

    answer_content = response_str
    think_end_match = re.search(r'</think>', response_str, re.I)
    if think_end_match:
        answer_content = response_str[think_end_match.end():]

    answer_match = re.search(r'<answer>(.*?)</answer>', answer_content, re.I | re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1)

    interval_pattern = r"\d+(?:\.\d+)?\s*[-–—~]\s*\d+(?:\.\d+)?\s*seconds?"
    return int(
        re.search(interval_pattern, answer_content, flags=re.I) is not None
    )
    
def extract_think_and_answer_robust(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts the content between <think> and <answer> tags from a string,
    regardless of their order or position, as long as the tags exist.
    Args:
        response (str): The input string, potentially containing <think> and <answer> tags.
    Returns:
        Tuple[Optional[str], Optional[str]]: 
            A tuple containing (think_content, answer_content).
            Each element will be a string if found, or None if the corresponding tag is not found.
    """
    think_content = None
    answer_content = None
    # Pattern for <think> tag content
    # re.DOTALL allows '.' to match any character, including newlines.
    # Non-greedy match (.*?) ensures it stops at the first </think>.
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    # Pattern for <answer> tag content
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    # Search for <think> content
    think_match = think_pattern.search(response)
    if think_match:
        think_content = think_match.group(1) # group(1) gets the content of the first capture group
    # Search for <answer> content
    answer_match = answer_pattern.search(response)
    if answer_match:
        answer_content = answer_match.group(1) # group(1) gets the content of the first capture group
    
    if answer_content is None or think_content is None:
        if '<answer>' in response:
            head, tail = response.split('<answer>', 1)
            if think_content is None:
                think_content = head
            if answer_content is None:
                answer_content = tail
        elif '</think>' in response:
            head, tail = response.split('</think>', 1)
            if think_content is None:
                think_content = head
            if answer_content is None:
                answer_content = tail

    return think_content, answer_content


class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self.role = role
        self._cache = {}

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # improve numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self._has_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._has_critic = self.role == "critic"
        self._has_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._has_ref = self.role in ["ref", "actor_rollout_ref"]
        if self._has_actor and self._has_critic:
            raise ValueError("Actor and critic cannot be both initialized.")

        if self.config.actor.disable_kl:
            self._has_ref = False

        self._use_param_offload = False
        self._use_optimizer_offload = False
        self._use_ref_param_offload = False
        if self._has_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
            self._init_dist_mesh(self.config.actor, "actor")

        if self._has_critic:
            self._use_param_offload = self.config.critic.offload.offload_params
            self._use_optimizer_offload = self.config.critic.offload.offload_optimizer
            self._init_dist_mesh(self.config.critic, "critic")

        if self._has_ref:  # NOTE: it seems that manual offload is slower than FSDP offload
            self._use_ref_param_offload = self.config.ref.offload.offload_params

        # mask tokenizer
        self.vq_sam2 = None
        self.sam2_image_processor = None

    def _init_dist_mesh(self, config: Union[ActorConfig, CriticConfig], role: Literal["actor", "critic"]):
        world_size = dist.get_world_size()
        # create main device mesh
        fsdp_size = config.fsdp.fsdp_size
        if fsdp_size <= 0 or fsdp_size >= world_size:
            self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        else:  # hsdp
            self.device_mesh = init_device_mesh(
                "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp")
            )

        # create ulysses device mesh
        if config.ulysses_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // config.ulysses_size, config.ulysses_size),
                mesh_dim_names=("dp", "sp"),
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # validate and normalize config
        if self.config.rollout.n > 1:
            config.global_batch_size *= self.config.rollout.n
            self.print_rank0(f"{role} will use global batch size {config.global_batch_size}.")

        config.global_batch_size_per_device = config.global_batch_size // (world_size // config.ulysses_size)
        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size * ulysses size must be larger than num gpus.")

        if config.global_batch_size_per_device % config.micro_batch_size_per_device_for_update != 0:
            raise ValueError(f"{role} global batch size per device must be divisible by the micro batch size.")

        if (
            config.fsdp.enable_cpu_offload
            and config.global_batch_size_per_device != config.micro_batch_size_per_device_for_update
        ):
            raise ValueError(f"{role} cannot use FSDP's CPU offload when gradient accumulation is enabled.")

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool,
        role: Literal["actor", "critic", "ref"],
    ) -> None:
        if role != "ref":  # ref model's tokenizer is same as actor
            self.tokenizer = get_tokenizer(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.processor = get_processor(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.model_config = AutoConfig.from_pretrained(
                model_config.model_path,
                trust_remote_code=model_config.trust_remote_code,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **model_config.override_config,
            )

            try:
                self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
            except Exception:
                self.generation_config = GenerationConfig.from_model_config(self.model_config)

            self.print_rank0(f"Model config: {self.model_config}")

        if padding_free:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied!")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if role != "ref" else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if role == "critic":
            AutoClass = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForImageTextToText._model_mapping.keys():
            AutoClass = AutoModelForImageTextToText
        else:
            AutoClass = AutoModelForCausalLM

        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            model = AutoClass.from_pretrained(
                model_config.model_path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                model = AutoClass.from_config(
                    self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=model_config.trust_remote_code,
                )

        model = cast(PreTrainedModel, model)  # lint
        model.tie_weights()  # avoid hanging
        model = model.to(torch_dtype)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if role == "ref":
            model.requires_grad_(False)

        if model_config.freeze_vision_tower:
            if hasattr(model, "model") and hasattr(model.model, "visual"):  # transformers >= 4.52.0
                model.model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            elif hasattr(model, "visual"):  # transformers < 4.52.0
                model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            else:
                self.print_rank0("No vision tower found.")

        dist.barrier()
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")

        if self.device_mesh.ndim == 2:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            else:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        else:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            else:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After FSDP module init")

        if role in ["actor", "critic"]:
            self.fsdp_module = fsdp_module
            if optim_config.strategy == "adamw":
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                self.optimizer = AnyPrecisionAdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")

            if optim_config.lr_warmup_steps is not None:
                num_warmup_steps = optim_config.lr_warmup_steps
            else:
                num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)

            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
            )
            print_gpu_memory_usage("After optimizer init")
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")
        else:
            self.ref_fsdp_module = fsdp_module
            if self._use_ref_param_offload:
                offload_fsdp_model(self.ref_fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

    def _build_rollout(self) -> None:
        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        if self.world_size % tp_size != 0:
            raise ValueError(f"rollout world size {self.world_size} is not divisible by tp size {tp_size}.")

        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        self.rollout = vLLMRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
            use_param_offload=self._use_param_offload,
        )
        print_gpu_memory_usage("After vllm init")

    def _build_ref_rollout_sharding_manager(self) -> None:
        """Build a sharding manager for reference model that reuses the same vLLM engine."""
        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        if self.world_size % tp_size != 0:
            raise ValueError(f"rollout world size {self.world_size} is not divisible by tp size {tp_size}.")

        ref_rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        # Reuse the same vLLM inference engine, but use ref_fsdp_module for weight sync
        self.ref_rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.ref_fsdp_module,  # Use ref model weights
            inference_engine=self.rollout.inference_engine,  # Reuse the same vLLM engine
            device_mesh=ref_rollout_device_mesh,
            use_param_offload=self._use_ref_param_offload,
        )
        print_gpu_memory_usage("After ref rollout sharding manager init")

    def _build_mask_tokenizer(self):
        """
        Initializes vq_sam2 and other components for reward computation on CPU.
        The model will be moved to the correct GPU later.
        """
        if not self._has_rollout:
            return
        self.print_rank0("Building mask tokenizer (vq_sam2)...")
        reward_config = self.config.reward
        with hydra.initialize(version_base=None, config_path=reward_config.sam2_config_dir_path):
            sam2_config = SAM2Config(
                cfg_path="sam2.1_hiera_l.yaml",
                ckpt_path=reward_config.sam2_pretrained_weight,
            )
            vq_sam2_config = VQ_SAM2Config(
                sam2_config=sam2_config,
                codebook_size=reward_config.codebook_size,
                codebook_depth=reward_config.codebook_depth,
                shared_codebook=False,
                latent_dim=256,
            )
        self.vq_sam2 = VQ_SAM2(vq_sam2_config)
        self.print_rank0(f"Loading mask tokenizer weights from {reward_config.mask_tokenizer_path}")
        state = torch.load(reward_config.mask_tokenizer_path, map_location="cpu")
        self.vq_sam2.load_state_dict(state)
        self.sam2_image_processor = DirectResize(1024)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # self._build_mask_tokenizer()

        if self._has_critic:
            self._build_model_optimizer(
                model_config=self.config.critic.model,
                fsdp_config=self.config.critic.fsdp,
                optim_config=self.config.critic.optim,
                padding_free=self.config.critic.padding_free,
                role="critic",
            )

        if self._has_actor:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.actor.fsdp,
                optim_config=self.config.actor.optim,
                padding_free=self.config.actor.padding_free,
                role="actor",
            )

        if self._has_ref:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.ref.fsdp,
                optim_config=None,
                padding_free=self.config.ref.padding_free,
                role="ref",
            )

        if self._has_actor:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )

        if self._has_critic:
            from .critic.dp_critic import DataParallelPPOCritic  # lazy import

            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._has_rollout:  # must after actor
            self._build_rollout()

        if self._has_ref:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.ref_fsdp_module,
            )
            # Build ref rollout sharding manager for generation with pretrained model
            # Reuses the same vLLM engine but syncs ref model weights
            if self._has_rollout:
                self._build_ref_rollout_sharding_manager()

        if self._has_actor or self._has_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor or self.tokenizer,
            )
        
        if self.vq_sam2 is not None:
            device = torch.cuda.current_device()
            self.print_rank0(f"Moving mask tokenizer to device: {device}")
            self.vq_sam2.to(device)
            self.vq_sam2.eval()
            print_gpu_memory_usage("After moving vq_sam2 to device")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, save_model_only: bool = False):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.save_checkpoint(path, save_model_only)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:  # avoid OOM in resuming
            offload_fsdp_optimizer(self.optimizer)

    def _process_multi_modal_inputs(self, data: DataProto):
        if "multi_modal_data" not in data.non_tensor_batch:
            return

        # Check if cache needs to be cleared (different uid array)
        # First check length, then check content to avoid broadcast error
        if "uid" in self._cache:
            cached_uid = self._cache["uid"]
            data_uid = data.non_tensor_batch["uid"]
            if len(cached_uid) != len(data_uid) or not np.all(data_uid == cached_uid):
                self._cache.clear()

        if "multi_modal_inputs_per_uid" not in self._cache:
            min_pixels = data.meta_info["min_pixels"]
            max_pixels = data.meta_info["max_pixels"]
            video_fps = data.meta_info["video_fps"]
            multi_modal_inputs_cache = {}  # uid -> pristine mm dict
            for index, multi_modal_data in zip(
                data.non_tensor_batch["uid"], data.non_tensor_batch["multi_modal_data"]
            ):  # process multi modal data per sample
                if index not in multi_modal_inputs_cache:
                    images, videos = [], []
                    video_metadata_list = []
                    if "images" in multi_modal_data:
                        for image in multi_modal_data["images"]:
                            images.append(process_image(image, min_pixels, max_pixels))

                    if "videos" in multi_modal_data:
                        sample_video_fps = multi_modal_data.get("fps", video_fps)
                        nframes_list = multi_modal_data.get("nframes")
                        for vi, video in enumerate(multi_modal_data["videos"]):
                            nf = nframes_list[vi] if nframes_list is not None else None
                            processed_video, video_metadata = process_video(
                                video,
                                min_pixels,
                                max_pixels,
                                sample_video_fps,
                                return_video_metadata=True,
                                nframes=nf,
                            )
                            videos.append(processed_video)
                            video_metadata_list.append(video_metadata)

                    if len(images) != 0:
                        # it's necessary to add `dict` to properly convert batch features to dict
                        # otherwise the batch features will be converted to dict keys
                        # see https://github.com/hiyouga/EasyR1/pull/339
                        multi_modal_inputs = dict(self.processor.image_processor(images=images, return_tensors="pt"))
                    elif len(videos) != 0:
                        videos_kwargs = {"video_metadata": video_metadata_list, "do_sample_frames": False}
                        multi_modal_inputs = dict(
                            self.processor(
                                videos=videos,
                                text=[""],
                                add_special_tokens=False,
                                return_tensors="pt",
                                videos_kwargs=videos_kwargs,
                            )
                        )
                        multi_modal_inputs.pop("input_ids", None)
                        multi_modal_inputs.pop("attention_mask", None)
                    else:
                        multi_modal_inputs = {}

                    multi_modal_inputs_cache[index] = multi_modal_inputs

            self._cache["uid"] = data.non_tensor_batch["uid"]
            self._cache["multi_modal_inputs_per_uid"] = multi_modal_inputs_cache

        # ALWAYS rebuild fresh shallow copies on every call. This prevents
        # cross-call contamination: when reconcile mutates the per-sample mm
        # dict (mm.pop / slicing), those mutations stay within this call's
        # data, not in self._cache. Otherwise actor's reconcile leaves an
        # inconsistent state for ref's subsequent call (input_ids fresh from
        # Ray serialization, but cached mm already pop'd).
        batch_multi_modal_inputs = [
            dict(self._cache["multi_modal_inputs_per_uid"][index])
            for index in data.non_tensor_batch["uid"]
        ]
        data.non_tensor_batch["multi_modal_inputs"] = np.array(batch_multi_modal_inputs, dtype=object)

        # Reconcile video features with actual video tokens in input_ids.
        # If prompts were truncated (exceeding max_prompt_length), some video_pad
        # tokens may have been removed. Trim pixel_values_videos and video_grid_thw
        # to match the actual token count per sample.
        self._reconcile_video_features_with_tokens(data)

    def _reconcile_video_features_with_tokens(self, data: DataProto):
        """Trim per-sample video features if input_ids has fewer video tokens than expected."""
        if "multi_modal_inputs" not in data.non_tensor_batch:
            return
        video_token_id = getattr(self.tokenizer, 'video_token_id', None)
        if video_token_id is None:
            video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        if video_token_id is None:
            return

        merge_size = getattr(self.processor.video_processor, 'merge_size', 2) if hasattr(self.processor, 'video_processor') else 2

        input_ids = data.batch["input_ids"]  # (batch_size, seq_len)
        for i in range(len(data.non_tensor_batch["multi_modal_inputs"])):
            mm = data.non_tensor_batch["multi_modal_inputs"][i]
            if "video_grid_thw" not in mm:
                continue

            n_tokens = (input_ids[i] == video_token_id).sum().item()
            grid_thw = mm["video_grid_thw"]  # (num_videos, 3)
            # Compute expected features per video
            features_per_video = [(t * h * w // (merge_size ** 2)).item() for t, h, w in grid_thw]
            n_features = sum(features_per_video)

            if n_tokens == n_features:
                continue  # no truncation

            if n_tokens > n_features:
                # More tokens than features — shouldn't happen, skip
                continue

            # Trim: keep only full videos whose cumulative features <= n_tokens
            cum = 0
            keep_videos = 0
            for fv in features_per_video:
                if cum + fv <= n_tokens:
                    cum += fv
                    keep_videos += 1
                else:
                    break

            if keep_videos == 0:
                # No full video fits — clear video features entirely AND zero out
                # all video_pad tokens in input_ids[i]. Skipping the zeroing here
                # leaves orphan tokens that break the (n_tokens == n_features)
                # invariant in qwen3_vl._get_input_embeds.
                mm.pop("video_grid_thw", None)
                mm.pop("pixel_values_videos", None)
                vid_positions = (input_ids[i] == video_token_id).nonzero(as_tuple=True)[0]
                if vid_positions.numel() > 0:
                    input_ids[i, vid_positions] = self.tokenizer.pad_token_id
                continue

            new_grid_thw = grid_thw[:keep_videos]
            new_n_features = sum(features_per_video[:keep_videos])

            if "pixel_values_videos" in mm:
                mm["pixel_values_videos"] = mm["pixel_values_videos"][:new_n_features]
            mm["video_grid_thw"] = new_grid_thw

            # Also need to zero out the orphan video tokens in input_ids
            # (tokens belonging to partially-truncated videos that no longer have features)
            if new_n_features < n_tokens:
                # Find positions of video tokens and zero out excess ones
                vid_positions = (input_ids[i] == video_token_id).nonzero(as_tuple=True)[0]
                excess_positions = vid_positions[new_n_features:]
                input_ids[i, excess_positions] = self.tokenizer.pad_token_id

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.lr_scheduler.step()

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )
            # Metrics do not need post processing since their batch size is 1

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def accumulate_actor_gradients(self, data: DataProto):
        """Compute gradients for actor without optimizer step (for gradient accumulation).
        
        Args:
            data: DataProto containing the batch data.
                  grad_weight should be passed via data.meta_info['grad_weight'] (default 1.0)
        """
        assert self._has_actor
        
        # Get grad_weight from meta_info (default to 1.0)
        grad_weight = data.meta_info.get('grad_weight', 1.0)

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="accumulate_gradients", logger=None) as timer:
                # Only compute gradients, don't step optimizer
                metrics = self.actor.update_policy(data=data, do_optimizer_step=False, grad_weight=grad_weight)

            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)

            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )

        # Don't offload yet - we need to accumulate more gradients
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def step_actor_optimizer(self):
        """Perform optimizer step after gradient accumulation."""
        assert self._has_actor

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            metrics = self.actor.optimizer_step()

            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.lr_scheduler.step()

            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_rollout_engine(self):
        self.rollout_sharding_manager.load_vllm_and_sync_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rollout_engine(self):
        self.rollout_sharding_manager.offload_vllm()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_multi_modal_cache(self):
        """Clear the multi-modal inputs cache. 
        
        Call this when switching between different tasks (e.g., caption to segmentation)
        to avoid uid mismatch issues.
        """
        self._cache.clear()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_ref_rollout_engine(self):
        """Prepare the reference model's vLLM engine for generation."""
        assert self._has_ref and hasattr(self, 'ref_rollout_sharding_manager')
        self.ref_rollout_sharding_manager.load_vllm_and_sync_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_ref_rollout_engine(self):
        """Release the reference model's vLLM engine."""
        assert self._has_ref and hasattr(self, 'ref_rollout_sharding_manager')
        self.ref_rollout_sharding_manager.offload_vllm()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def swap_to_ref_weights(self):
        """Swap vLLM weights to ref model weights.
        
        Call this when the vLLM engine is already loaded (via prepare_rollout_engine)
        and you want to temporarily use ref model weights instead of actor weights.
        """
        assert self._has_ref and hasattr(self, 'ref_rollout_sharding_manager')
        self.ref_rollout_sharding_manager.sync_weights_only()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def swap_to_actor_weights(self):
        """Swap vLLM weights back to actor model weights.
        
        Call this after using ref weights to restore actor weights.
        """
        self.rollout_sharding_manager.sync_weights_only()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences_with_ref(self, prompts: DataProto):
        """Generate sequences using the reference (pretrained) model.
        
        This method uses the same vLLM engine as the actor, but temporarily
        swaps in the ref_fsdp_module (pretrained weights) for generation.
        After generation, it swaps back to actor weights.
        """
        assert self._has_ref and hasattr(self, 'ref_rollout_sharding_manager')

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        # Swap to ref model weights before generation
        self.ref_rollout_sharding_manager.sync_weights_only()
        
        prompts = self.ref_rollout_sharding_manager.preprocess_data(prompts)
        # Use the same rollout engine with ref model weights
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.ref_rollout_sharding_manager.postprocess_data(output)
        
        # Swap back to actor weights after generation
        self.rollout_sharding_manager.sync_weights_only()
        
        output = output.to("cpu")

        # Compare mask tokens directly with seg_ground_truth (without decoding to masks)
        # Using graded matching: 1.0 for exact match, 0.8 for both tokens match, 0.4 for first token match
        if output.meta_info.get('task') == 'segmentation' and 'seg_ground_truth' in prompts.non_tensor_batch:
            response_ids = output.batch["responses"]
            response_length = torch.sum(output.batch["response_mask"], dim=-1)
            seg_ground_truths = prompts.repeat(repeat_times=prompts.meta_info["n"], interleave=True).non_tensor_batch["seg_ground_truth"]
            seg_problems = prompts.repeat(repeat_times=prompts.meta_info["n"], interleave=True).non_tensor_batch["seg_problems"]
            
            def compute_mask_token_accuracy_by_grading(one_answer: str, one_ground_truth: str) -> float:
                """Compute graded match score between two mask tokens."""
                answer_tokens = re.findall(r"<\|[^|]*\|>", one_answer)
                gt_tokens = re.findall(r"<\|[^|]*\|>", one_ground_truth)
                if one_answer == one_ground_truth:
                    return 1.0
                elif ''.join(answer_tokens[:-1]) == ''.join(gt_tokens[:-1]):  # both of the two tokens are matched
                    return 0.8
                elif ''.join(answer_tokens[:-2]) == ''.join(gt_tokens[:-2]):  # only the first token matches
                    return 0.4
                else:
                    return 0.0
            
            def mask_token_accuracy_reward(answer_content: str, ground_truth: str) -> float:
                """Compute mask token accuracy reward using graded matching."""
                if answer_content is None:
                    return 0.0
                regex_pattern = r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>"
                target_mask_tokens = re.findall(regex_pattern, ground_truth) if ground_truth else []
                target_mask_tokens = [target_mask_tokens[0]] if len(target_mask_tokens) > 1 else target_mask_tokens
                pred_mask_tokens = re.findall(regex_pattern, answer_content)
                if len(target_mask_tokens) == 0 and len(pred_mask_tokens) == 0:
                    return 1.0
                if len(target_mask_tokens) == 0 or len(pred_mask_tokens) == 0:
                    return 0.0
                unique_target_mask_tokens = list(set(target_mask_tokens))
                unique_pred_mask_tokens = list(set(pred_mask_tokens))
                max_N = max(len(pred_mask_tokens), len(target_mask_tokens))
                recall_N = 0.0
                for mask_token in unique_pred_mask_tokens:
                    max_match_score = 0.0
                    for gt_mask_token in unique_target_mask_tokens:
                        match_score = compute_mask_token_accuracy_by_grading(mask_token, gt_mask_token)
                        if max_match_score < match_score:
                            max_match_score = match_score
                    recall_N += max_match_score
                return recall_N / max_N
            
            mask_token_accuracy_list = []
            format_correct_list = []
            debug_responses = []  # Collect responses for debugging
            
            for i in range(len(output)):
                cur_response_length = int(response_length[i].item())
                valid_response_ids = response_ids[i][:cur_response_length]
                response_str = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=self.config.reward.skip_special_tokens
                )
                
                # Collect response for debugging
                debug_responses.append(f"Sample {i}:\n{response_str}\n")
                
                # Extract answer content from response
                think_content, answer_content = extract_think_and_answer_robust(response_str)
                # Check format correctness (should have the correct mask token format)
                regex_pattern = r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>"
                pred_mask_tokens = re.findall(regex_pattern, answer_content) if answer_content else []
                format_correct = int(len(pred_mask_tokens) > 0)
                
                # Get ground truth
                gt_text = seg_ground_truths[i]
                
                # Compute graded mask token accuracy
                accuracy = mask_token_accuracy_reward(answer_content, gt_text)
                
                mask_token_accuracy_list.append(accuracy)
                format_correct_list.append(format_correct)
            
            output.non_tensor_batch["mask_token_accuracy"] = np.array(mask_token_accuracy_list, dtype=object)
            output.non_tensor_batch["format_correct"] = np.array(format_correct_list, dtype=object)
            
            # Save all responses to file once after the loop (minimal I/O overhead)
            with open("./debug_response_new.txt", "w") as f:
                f.write("\n".join(debug_responses))  # Only save first 5 samples to keep file small     

        # if output.meta_info['task'] == 'segmentation':
        #     # decoding text-format mask tokens to binary mask array
        #     response_ids = output.batch["responses"]
        #     response_length = torch.sum(output.batch["response_mask"], dim=-1)
        #     batch_multi_modal_data = output.non_tensor_batch["multi_modal_data"]
        #     batch_pred_masks = []
        #     successes = []
        #     for i in range(len(output)):
        #         cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
        #         valid_response_ids = response_ids[i][:cur_response_length]
        #         response_str = self.tokenizer.decode(
        #             valid_response_ids, skip_special_tokens=self.config.reward.skip_special_tokens
        #         )
        #         think_content, answer_content = extract_think_and_answer_robust(response_str)
        #         # decode mask tokens to masks
        #         multi_modal_data = batch_multi_modal_data[i]
        #         image = multi_modal_data["images"][0]
        #         if isinstance(image, str):
        #             image = Image.open(image)
        #         elif isinstance(image, dict):
        #             image = Image.open(BytesIO(image["bytes"]))
        #         elif isinstance(image, bytes):
        #             image = Image.open(BytesIO(image))
                
        #         if image.mode != "RGB":
        #             image = image.convert("RGB")
        #         ori_width, ori_height = image.size

        #         quant_ids = extract_mt_token_ids_v1(answer_content) if answer_content is not None else []
        #         successes.append(int(len(quant_ids) == 2))
        #         if len(quant_ids) % self.config.reward.codebook_depth != 0:
        #             answer_content = fix_mt_format_comprehensive(answer_content)
        #             quant_ids = extract_mt_token_ids_v2(answer_content)
        #         if len(quant_ids) == 0:
        #             pred_masks = {'pred_masks': np.zeros((ori_height, ori_width), dtype=np.uint8)}
        #         else:
        #             batch_size = len(quant_ids) // self.config.reward.codebook_depth
        #             remap_quant_ids = []
        #             for bs_id in range(batch_size):
        #                 chunk_quant_ids = quant_ids[bs_id*self.config.reward.codebook_depth:(bs_id+1)*self.config.reward.codebook_depth]
        #                 remap_chunk_quant_ids = [quant_id - book_id*self.config.reward.codebook_size for book_id, quant_id in enumerate(chunk_quant_ids)]
        #                 code1 = remap_chunk_quant_ids[0]
        #                 code2 = remap_chunk_quant_ids[1]
        #                 if not (code1 >= 0 and code1 < self.config.reward.codebook_size):
        #                     continue
        #                 if not (code2 >= 0 and code2 < self.config.reward.codebook_size):
        #                     code2 = -1
        #                 remap_chunk_quant_ids_error_handle = [code1, code2]
        #                 remap_quant_ids.append(remap_chunk_quant_ids_error_handle)
                    
        #             if len(remap_quant_ids) == 0:
        #                 pred_masks = {'pred_masks': np.zeros((ori_height, ori_width), dtype=np.uint8)}
        #             else:
        #                 batch_size = len(remap_quant_ids)
        #                 # assert batch_size == 1 , f"currently only support batch size 1, but got {batch_size}"
        #                 if batch_size > 1:
        #                     batch_size = 1

        #                 sam2_image = np.array(image)
        #                 sam2_image = self.sam2_image_processor.apply_image(sam2_image)
        #                 sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        #                 sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(self.vq_sam2.dtype).to(self.vq_sam2.device)

        #                 quant_ids = torch.LongTensor(remap_quant_ids).to(self.vq_sam2.device)
        #                 pred_masks = []
        #                 for batch_i in range(batch_size):
        #                     with torch.no_grad():
        #                         pred_masks_i = self.vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids[batch_i:batch_i+1])
        #                         pred_masks_i = pred_masks_i.detach()
        #                         pred_masks_i = torch.nn.functional.interpolate(pred_masks_i, size=(ori_height, ori_width), mode='bilinear')
        #                         pred_masks_i = pred_masks_i > 0.5
        #                         pred_masks_i = pred_masks_i[:, 0, :, :].cpu().numpy().astype(np.uint8)
        #                         pred_masks.append(pred_masks_i)
        #                 pred_masks = {'pred_masks': np.concatenate(pred_masks, axis=0)[0]}
        #         batch_pred_masks.append(pred_masks)
        #     output.non_tensor_batch["pred_masks"] = np.array(batch_pred_masks, dtype=object)
        #     output.non_tensor_batch["correct_mask"] = np.array(successes, dtype=object)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._has_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        # prompts.batch: ['attention_mask', 'input_ids', 'position_ids']
        output = self.rollout.generate_sequences(prompts=prompts) #sampling_params
        # ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
        output = self.rollout_sharding_manager.postprocess_data(output)
        # ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
        output = output.to("cpu")

        # 图像样本比较 mask token；视频样本比较时间区间 tIoU。
        if output.meta_info.get('task') == 'segmentation' and 'seg_ground_truth' in prompts.non_tensor_batch:
            response_ids = output.batch["responses"]
            response_length = torch.sum(output.batch["response_mask"], dim=-1)
            seg_ground_truths = prompts.repeat(repeat_times=prompts.meta_info["n"], interleave=True).non_tensor_batch["seg_ground_truth"]
            seg_problems = prompts.repeat(repeat_times=prompts.meta_info["n"], interleave=True).non_tensor_batch["seg_problems"]
            repeated_non_tensor = prompts.repeat(repeat_times=prompts.meta_info["n"], interleave=True).non_tensor_batch
            is_video_segmentation = "videos" in repeated_non_tensor

            def compute_mask_token_accuracy_by_grading(one_answer: str, one_ground_truth: str) -> float:
                """Compute graded match score between two mask tokens."""
                answer_tokens = re.findall(r"<\|[^|]*\|>", one_answer)
                gt_tokens = re.findall(r"<\|[^|]*\|>", one_ground_truth)
                if one_answer == one_ground_truth:
                    return 1.0
                elif ''.join(answer_tokens[:-1]) == ''.join(gt_tokens[:-1]):  # both of the two tokens are matched
                    return 0.8
                elif ''.join(answer_tokens[:-2]) == ''.join(gt_tokens[:-2]):  # only the first token matches
                    return 0.4
                else:
                    return 0.0
            
            def mask_token_accuracy_reward(answer_content: str, ground_truth: str) -> float:
                """Compute mask token accuracy reward using graded matching."""
                if answer_content is None:
                    return 0.0
                regex_pattern = r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>"
                target_mask_tokens = re.findall(regex_pattern, ground_truth) if ground_truth else []
                # target_mask_tokens = [target_mask_tokens[0]] if len(target_mask_tokens) > 1 else target_mask_tokens
                pred_mask_tokens = re.findall(regex_pattern, answer_content)
                if len(target_mask_tokens) == 0 and len(pred_mask_tokens) == 0:
                    return 1.0
                if len(target_mask_tokens) == 0 or len(pred_mask_tokens) == 0:
                    return 0.0
                unique_target_mask_tokens = list(set(target_mask_tokens))
                unique_pred_mask_tokens = list(set(pred_mask_tokens))
                max_N = max(len(pred_mask_tokens), len(target_mask_tokens))
                recall_N = 0.0
                for mask_token in unique_pred_mask_tokens:
                    max_match_score = 0.0
                    for gt_mask_token in unique_target_mask_tokens:
                        match_score = compute_mask_token_accuracy_by_grading(mask_token, gt_mask_token)
                        if max_match_score < match_score:
                            max_match_score = match_score
                    recall_N += max_match_score
                return recall_N / max_N
            
            mask_token_accuracy_list = []
            format_correct_list = []
            debug_responses = []  # Collect responses for debugging
            
            for i in range(len(output)):
                cur_response_length = int(response_length[i].item())
                valid_response_ids = response_ids[i][:cur_response_length]
                response_str = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=self.config.reward.skip_special_tokens
                )
                # Collect response for debugging
                debug_responses.append(f"Sample {i}:\n{response_str}\n")
                
                gt_text = seg_ground_truths[i]

                if is_video_segmentation:
                    parsed_gt_windows = extract_time_intervals_from_response(gt_text)
                    parsed_pred_windows = extract_time_intervals_from_response(response_str, only_result=True)

                    accuracy, _, _ = compute_temporal_iou_reward(parsed_pred_windows, parsed_gt_windows)
                    format_correct = video_grounding_format_reward(response_str)
                else:
                    # Extract answer content from response
                    # think_content, answer_content = extract_think_and_answer_robust(response_str)
                    # Check format correctness (should have the correct mask token format)
                    regex_pattern = r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>"
                    pred_mask_tokens = re.findall(regex_pattern, response_str) if response_str else []
                    format_correct = int(len(pred_mask_tokens) > 0)
                    accuracy = mask_token_accuracy_reward(response_str, gt_text)

                    # pred_bbox = extract_bbox_from_response(response_str)
                    # if pred_bbox is not None:
                    #     format_correct = 1
                    # else:
                    #     format_correct = 0
                    # gt_bbox = extract_bbox_from_response(seg_ground_truths[i])
                    # accuracy = compute_bbox_iou(pred_bbox, gt_bbox)
                
                mask_token_accuracy_list.append(accuracy)
                format_correct_list.append(format_correct)

            output.non_tensor_batch["mask_token_accuracy"] = np.array(mask_token_accuracy_list, dtype=object)
            output.non_tensor_batch["format_correct"] = np.array(format_correct_list, dtype=object)
            output.non_tensor_batch["seg_responses"] = debug_responses
            output.non_tensor_batch["cap_responses"] = seg_problems
            output.non_tensor_batch["seg_ground_truth"] = seg_ground_truths
            
            # Save all responses to file once after the loop (minimal I/O overhead)
            with open("./debug_response_cap_debug0223.txt", "w") as f:
                f.write("\n".join(debug_responses[:32]))
                f.write("\n".join(seg_problems[:32]))
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output}, meta_info={"temperature": self.config.rollout.temperature}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_probs(self, data: DataProto):
        assert self._has_ref

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_ref_param_offload:
            load_fsdp_model(self.ref_fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_probs": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_fsdp_module._handle.reshard(True)

        if self._use_ref_param_offload:
            offload_fsdp_model(self.ref_fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_critic"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )
            # Metrics do not need post processing since their batch size is 1

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output
