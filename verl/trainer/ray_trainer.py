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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import math
import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from torch.utils.data._utils.collate import default_collate

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from pycocotools import mask as mask_utils

class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def strip_temporal_priors(response_text: str, replacement: str = "in this segment") -> str:
    """Remove explicit temporal expressions from a free-form response.

    This is a post-processing guard to prevent leaking timestamp priors into
    temporal grounding prompts.
    """
    if not isinstance(response_text, str):
        return response_text

    text = response_text
    # Normalize dash variants to simplify matching.
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    interval_patterns = [
        # e.g. "100.3 - 104.2 seconds", "100.3 to 104.2 sec"
        r"\b\d+(?:\.\d+)?\s*(?:-|to|~)\s*\d+(?:\.\d+)?\s*(?:s|sec(?:ond)?s?|seconds?|mins?|minutes?|分钟|秒)\b",
        # e.g. "00:01:40 - 00:01:44", "1:40 to 1:44"
        r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:-|to|~)\s*\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b",
        # e.g. "during 100.3 to 104.2 seconds", "from 3 to 7 s"
        r"\b(?:during|from|between|at|around|approximately|about|在|约|大约)\s+\d+(?:\.\d+)?\s*(?:-|to|~|至|到)\s*\d+(?:\.\d+)?\s*(?:s|sec(?:ond)?s?|seconds?|mins?|minutes?|分钟|秒)\b",
    ]

    point_patterns = [
        # e.g. "at 103.2 seconds", "around 1.5 min"
        r"\b(?:at|around|approximately|about|timestamp|time|在|约|大约)\s+\d+(?:\.\d+)?\s*(?:s|sec(?:ond)?s?|seconds?|mins?|minutes?|分钟|秒)\b",
        # e.g. "at 01:42", "timestamp 00:01:42"
        r"\b(?:at|around|timestamp|time)\s+\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b",
    ]

    for pattern in interval_patterns:
        text = re.sub(pattern, " __TIME__ ", text, flags=re.IGNORECASE)
    for pattern in point_patterns:
        text = re.sub(pattern, " __TIME__ ", text, flags=re.IGNORECASE)

    # Replace context phrases around removed timestamps with neutral wording.
    text = re.sub(
        r"\b(?:during|from|between|at|around|approximately|about|within)\s+__TIME__\b",
        f" {replacement} ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:the\s+)?(?:time\s*frame|time\s*window|timestamp(?:\s*interval)?)\s*(?:of)?\s*__TIME__\b",
        f" {replacement} ",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("__TIME__", replacement)

    # Cleanup spacing and punctuation after substitution.
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,", ", ", text)

    return text.strip()


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> tuple[Optional[DataProto], Optional[DataProto]]:
        cycle_batch = None
        non_cycle_batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0

        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }

            DW_SOURCES = ['denseworld_single', 'denseworld_multiple', 'tg_multi_merged', 'dam_cyclegrpo', None]
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # prepare gen_batch for generation of captioning / segmentation training
            task_dict = {
                k.replace('cap_', ''): new_batch.batch.pop(k)
                for k in ['cap_input_ids', 'cap_attention_mask', 'cap_position_ids']
            }
            task_dict.update(
                {
                    k.replace('cap_', ''): new_batch.non_tensor_batch.pop(k)
                    for k in ['cap_raw_prompt_ids', 'cap_multi_modal_data']
                }
            )
            gen_batch = DataProto.from_single_dict(task_dict, meta_info=meta_info)
            gen_batch.meta_info.update({'task': 'caption'})
            # gen_batch.meta_info["mm_processor_kwargs"] = {"fps": 0.5, "do_sample_frames": True,}

            # generate on the whole batch directly
            # gen_batch.batch.keys(): ['input_ids', 'attention_mask', 'position_ids']
            # gen_batch.non_tensor_batch.keys(): ['raw_prompt_ids', 'multi_modal_data']
            # gen_batch.meta_info: {'min_pixels': 3136, 'max_pixels': 1605632, 'video_fps': 0.5, 'task': 'caption'}
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            # split after rollout
            rollout_sources = new_batch.non_tensor_batch["source"]
            cycle_indices = [i for i, src in enumerate(rollout_sources) if src in DW_SOURCES]
            non_cycle_indices = [i for i, src in enumerate(rollout_sources) if src not in DW_SOURCES]

            if cycle_indices:
                cycle_part = new_batch[cycle_indices]
                cycle_batch = DataProto.concat([cycle_batch, cycle_part]) if cycle_batch is not None else cycle_part
            if non_cycle_indices:
                non_cycle_part = new_batch[non_cycle_indices]
                non_cycle_batch = (
                    DataProto.concat([non_cycle_batch, non_cycle_part]) if non_cycle_batch is not None else non_cycle_part
                )

            # 检查累积的数据是否足够
            cycle_batch_size = len(cycle_batch) // self.config.worker.rollout.n if cycle_batch is not None else 0
            non_cycle_batch_size = len(non_cycle_batch) // self.config.worker.rollout.n if non_cycle_batch is not None else 0
            total_batch_size = cycle_batch_size + non_cycle_batch_size
            rollout_batch_size = self.config.data.rollout_batch_size
            if total_batch_size < rollout_batch_size:
                print(f"{total_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{total_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                # cycle_batch / non_cycle_batch are balanced & dispatched SEPARATELY, so each
                # must have a sample count divisible by world_size. Their sizes are data-dependent
                # (how many DW vs no-target samples landed this rollout), so trim each down to a
                # whole number of groups such that groups*n % world_size == 0. Avoids the
                # "len(seqlen_list) % k_partitions != 0" assertion in _balance_batch for any n /
                # node count. Drops at most (group_align-1) groups per sub-batch (typically <1%).
                n = self.config.worker.rollout.n
                world_size = self.actor_rollout_ref_wg.world_size
                group_align = world_size // math.gcd(n, world_size)  # group count must be a multiple of this

                def _aligned_groups(num_groups: int) -> int:
                    return (num_groups // group_align) * group_align

                cycle_groups = _aligned_groups(cycle_batch_size) if cycle_batch is not None else 0
                non_cycle_groups = _aligned_groups(non_cycle_batch_size) if non_cycle_batch is not None else 0
                if cycle_batch is not None and cycle_groups != cycle_batch_size:
                    print(f"[make_batch] trim cycle groups {cycle_batch_size}->{cycle_groups} (divisible by world_size={world_size})")
                if non_cycle_batch is not None and non_cycle_groups != non_cycle_batch_size:
                    print(f"[make_batch] trim non_cycle groups {non_cycle_batch_size}->{non_cycle_groups} (divisible by world_size={world_size})")

                cycle_batch_ret = cycle_batch[: cycle_groups * n] if cycle_groups > 0 else None
                non_cycle_batch_ret = non_cycle_batch[: non_cycle_groups * n] if non_cycle_groups > 0 else None
                return cycle_batch_ret, non_cycle_batch_ret

    def _make_seg_batch_data_for_caption(self, batch: DataProto) -> DataProto:

        all_seg_problems = []
        gen_seg_batch_list = []
        for i in range(len(batch.non_tensor_batch['multi_modal_data'])):
            seg_problem = self.tokenizer.decode(batch.batch['responses'][i], skip_special_tokens=True)
            seg_mm_data = batch.non_tensor_batch['seg_multi_modal_data'][i]
            # Remove empty thinking tags if present
            seg_problem = re.sub(r'<think>\s*</think>\s*', '', seg_problem)
            # Strip vision-related markers the captioner may have echoed back; otherwise
            # they get parsed as extra <image>/<video> references and the processor
            # IndexErrors on video_metadata.
            seg_problem = re.sub(
                r'<(?:image|video)>|<\|(?:image|video)_pad\|>|<\|vision_(?:start|end)\|>',
                '',
                seg_problem,
            )
            if 'videos' in seg_mm_data:
                seg_problem = strip_temporal_priors(seg_problem)
            all_seg_problems.append(seg_problem)
            cap_mm_data = batch.non_tensor_batch['multi_modal_data'][i]
            example = {'seg_problem': seg_problem,
                        'seg_ground_truth': batch.non_tensor_batch['seg_ground_truth'][i],
                        'source': batch.non_tensor_batch['source'][i],
                        'masks': batch.non_tensor_batch['masks'][i],
                        'cap_ground_truth': batch.non_tensor_batch['cap_ground_truth'][i]}
            if 'images' in seg_mm_data:
                example['images'] = seg_mm_data['images']
                example['cap_images'] = cap_mm_data['images']
            elif 'videos' in seg_mm_data:
                example['videos'] = seg_mm_data['videos']
                example['nframes'] = seg_mm_data.get('nframes')
                example['cap_videos'] = cap_mm_data['videos']

            gen_seg_batch_list.append(self.train_dataloader.dataset._gen_seg_preprocess(example))

        gen_seg_batch_dict = {}
        for k in gen_seg_batch_list[0].keys():
            values = [d[k] for d in gen_seg_batch_list]
            if isinstance(values[0], torch.Tensor):
                gen_seg_batch_dict[k] = torch.stack(values, dim=0)
            else:
                gen_seg_batch_dict[k] = np.array(values, dtype=object)

        gen_seg_batch: DataProto = DataProto.from_single_dict(gen_seg_batch_dict, meta_info=batch.meta_info)
        gen_seg_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(gen_seg_batch.batch))], dtype=object
        )

        # prepare gen_batch for generation of segmentation training
        task_dict = {k.replace(f'seg_', ''): gen_seg_batch.batch.pop(k) for k in [f'seg_input_ids', f'seg_attention_mask', f'seg_position_ids']}
        task_dict.update({k.replace(f'seg_', ''): gen_seg_batch.non_tensor_batch.pop(k) for k in [f'seg_raw_prompt_ids', f'seg_multi_modal_data']})
        gen_batch = DataProto.from_single_dict(task_dict, meta_info=batch.meta_info)
        gen_batch.meta_info.update({'task': 'segmentation'})
        gen_batch.non_tensor_batch.update({'seg_ground_truth': gen_seg_batch.non_tensor_batch['seg_ground_truth']})
        gen_batch.non_tensor_batch.update({'seg_problems': np.array(all_seg_problems, dtype=object)})
        # Store media under a unified 'media' key for downstream compatibility (images or videos)
        first_mm = gen_batch.non_tensor_batch['multi_modal_data'][0]
        media_key = 'images' if 'images' in first_mm else 'videos'
        gen_batch.non_tensor_batch.update({media_key: gen_batch.non_tensor_batch['multi_modal_data']})
        
        # generate a batch using ref (pretrained) model
        # generate_sequences_with_ref automatically swaps weights to ref model and back to actor
        ori_rollout_n = self.config.worker.rollout.n
        self.config.worker.rollout.n = 6
        
        # gen_batch_output = self.actor_rollout_ref_wg.generate_sequences_with_ref(gen_batch)
        gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
        # gen_batch_output.batch.keys(): ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
        # gen_batch_output.non_tensor_batch.keys(): ['multi_modal_data', 'mask_token_accuracy', 'format_correct']

        # Aggregate mask_token_accuracy and format_correct from gen_batch_output to batch
        # gen_batch_output has n_prompts * n_samples results, aggregate to n_prompts
        n_samples = self.config.worker.rollout.n  # 32
        n_prompts = len(gen_batch)
        
        # Add uid to gen_batch_output: repeat each uid n_samples times to match the expanded batch size
        gen_batch_output.non_tensor_batch['uid'] = np.repeat(gen_seg_batch.non_tensor_batch["uid"], n_samples)
        gen_batch_output.non_tensor_batch['source'] = np.repeat(gen_seg_batch.non_tensor_batch["source"], n_samples)
        
        if 'mask_token_accuracy' in gen_batch_output.non_tensor_batch:
            flat_accuracy = np.array(gen_batch_output.non_tensor_batch['mask_token_accuracy']).astype(float)
            # Compute mean accuracy for each prompt
            mean_accuracy = [
                np.mean(flat_accuracy[b * n_samples : (b + 1) * n_samples])
                for b in range(n_prompts)
            ]
            batch.non_tensor_batch['iou_scores'] = np.array(mean_accuracy, dtype=object)
            # Also add iou_scores to gen_batch_output: repeat each mean_accuracy n_samples times
            gen_batch_output.non_tensor_batch['iou_scores'] = np.repeat(mean_accuracy, n_samples).astype(object)
        
        if 'format_correct' in gen_batch_output.non_tensor_batch:
            flat_format_correct = np.array(gen_batch_output.non_tensor_batch['format_correct']).astype(int)
            # Count how many samples have correct format for each prompt
            correct_counts = [
                np.sum(flat_format_correct[b * n_samples : (b + 1) * n_samples])
                for b in range(n_prompts)
            ]
            batch.non_tensor_batch['correct_mask'] = np.array(correct_counts, dtype=object)

        self.config.worker.rollout.n = ori_rollout_n
        # batch.batch: ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
        # batch.non_tensor_batch: ['masks', 'source', 'seg_multi_modal_data', 'cap_ground_truth', 'seg_ground_truth', 'uid', 'multi_modal_data', 'iou_scores', 'correct_mask']
        # gen_batch_output.batch.keys(): ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
        # gen_batch_output.non_tensor_batch.keys(): ['multi_modal_data', 'mask_token_accuracy', 'format_correct', 'iou_scores']
        return batch, gen_batch_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    cycle_batch, non_cycle_batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                # 对分离后的每个子集分别进行 balance
                if non_cycle_batch is not None:
                    self._balance_batch(non_cycle_batch, metrics=metrics)
                    # compute global valid tokens
                    non_cycle_batch.meta_info["global_token_num"] = torch.sum(non_cycle_batch.batch["attention_mask"], dim=-1).tolist()

                if cycle_batch is not None:
                    self._balance_batch(cycle_batch, metrics=metrics)
                    # compute global valid tokens
                    cycle_batch.meta_info["global_token_num"] = torch.sum(cycle_batch.batch["attention_mask"], dim=-1).tolist()

                if non_cycle_batch is not None:

                    if "token_level_scores" not in non_cycle_batch.batch:
                        with timer("reward", timing_raw):
                            reward_ref = self.reward_fn.compute_reward.remote(self.global_step, non_cycle_batch, task='caption')

                    # recompute old_log_probs
                    with timer("old", timing_raw):
                        # batch.batch.keys(): ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
                        # batch.non_tensor_batch.keys(): ['masks', 'source', 'cap_ground_truth', 'seg_ground_truth', 'uid', 'multi_modal_data', 'iou_scores', 'correct_mask']
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(non_cycle_batch) 
                        non_cycle_batch = non_cycle_batch.union(old_log_probs)

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(non_cycle_batch)
                            non_cycle_batch = non_cycle_batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(non_cycle_batch)
                            non_cycle_batch = non_cycle_batch.union(values)

                    with timer("adv", timing_raw):
                        if "token_level_scores" not in non_cycle_batch.batch:
                            # get token level scores asynchronously
                            reward_tensor, reward_metrics = ray.get(reward_ref)
                            non_cycle_batch.batch["token_level_scores"] = reward_tensor  # [8, 8192]
                            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                            metrics.update(reward_metrics)

                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            non_cycle_batch, kl_metrics = apply_kl_penalty(non_cycle_batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            non_cycle_batch.batch["token_level_rewards"] = non_cycle_batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        non_cycle_batch = compute_advantage(
                            non_cycle_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                    # update critic
                    if self.use_critic:
                        with timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(non_cycle_batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)
                    
                    # Clear multi-modal cache before switching to segmentation task
                    self.actor_rollout_ref_wg.clear_multi_modal_cache()

                cycle_cap_batch = None
                cycle_seg_batch = None
                seg_batch = None
                if cycle_batch is not None and (
                    self.config.worker.actor.optimize_captioner or self.config.worker.actor.optimize_segmenter
                ):
                    # build both cycle caption and cycle segmentation batches once
                    with timer("gen", timing_raw):
                        cycle_batch.meta_info["n"] = 6
                        self.actor_rollout_ref_wg.prepare_rollout_engine()
                        cycle_cap_batch, cycle_seg_batch = self._make_seg_batch_data_for_caption(cycle_batch)
                        cycle_cap_batch.meta_info.pop("n", None)
                        self.actor_rollout_ref_wg.release_rollout_engine()

                if cycle_cap_batch is not None and self.config.worker.actor.optimize_captioner:

                    if "token_level_scores" not in cycle_cap_batch.batch:
                        with timer("reward", timing_raw):
                            reward_ref = self.reward_fn.compute_reward.remote(self.global_step, cycle_cap_batch, task='caption')

                    # recompute old_log_probs
                    with timer("old", timing_raw):
                        # cycle_cap_batch.batch.keys(): ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
                        # cycle_cap_batch.non_tensor_batch.keys(): ['masks', 'source', 'cap_ground_truth', 'seg_ground_truth', 'uid', 'multi_modal_data', 'iou_scores', 'correct_mask']
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(cycle_cap_batch)
                        cycle_cap_batch = cycle_cap_batch.union(old_log_probs)

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(cycle_cap_batch)
                            cycle_cap_batch = cycle_cap_batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(cycle_cap_batch)
                            cycle_cap_batch = cycle_cap_batch.union(values)

                    with timer("adv", timing_raw):
                        if "token_level_scores" not in cycle_cap_batch.batch:
                            # get token level scores asynchronously
                            reward_tensor, reward_metrics = ray.get(reward_ref)
                            cycle_cap_batch.batch["token_level_scores"] = reward_tensor  # [8, 8192]
                            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                            metrics.update(reward_metrics)

                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            cycle_cap_batch, kl_metrics = apply_kl_penalty(cycle_cap_batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            cycle_cap_batch.batch["token_level_rewards"] = cycle_cap_batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        cycle_cap_batch = compute_advantage(
                            cycle_cap_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                    # update critic
                    if self.use_critic:
                        with timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(cycle_cap_batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)
                    
                    # Clear multi-modal cache before switching to segmentation task
                    self.actor_rollout_ref_wg.clear_multi_modal_cache()

                if cycle_seg_batch is not None and self.config.worker.actor.optimize_segmenter:
                    ## Start Segmentation Session
                    # seg_batch.batch.keys(): ['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids']
                    # seg_batch.non_tensor_batch.keys(): ['multi_modal_data', 'mask_token_accuracy', 'format_correct', 'iou_scores', 'uid']
                    self._balance_batch(cycle_seg_batch, metrics=metrics)
                    cycle_seg_batch.meta_info["global_token_num"] = torch.sum(cycle_seg_batch.batch["attention_mask"], dim=-1).tolist()

                    # compute reward
                    if cycle_seg_batch is not None and "token_level_scores" not in cycle_seg_batch.batch:
                        with timer("reward", timing_raw):
                            seg_reward_ref = self.reward_fn.compute_reward.remote(self.global_step, cycle_seg_batch, task='segmentation')
                    
                    # recompute old_log_probs
                    if cycle_seg_batch is not None:
                        with timer("old", timing_raw):
                            old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(cycle_seg_batch) 
                            cycle_seg_batch = cycle_seg_batch.union(old_log_probs)

                    # compute ref_log_probs
                    if cycle_seg_batch is not None and self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(cycle_seg_batch)
                            cycle_seg_batch = cycle_seg_batch.union(ref_log_probs)

                    # compute values
                    if cycle_seg_batch is not None and self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(cycle_seg_batch)
                            cycle_seg_batch = cycle_seg_batch.union(values)

                    if cycle_seg_batch is not None:
                        with timer("adv", timing_raw):
                            if "token_level_scores" not in cycle_seg_batch.batch:
                                # get token level scores asynchronously
                                reward_tensor, reward_metrics = ray.get(seg_reward_ref)
                                cycle_seg_batch.batch["token_level_scores"] = reward_tensor
                                reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                                metrics.update(reward_metrics)

                            # apply kl penalty if available
                            if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                                # apply kl penalty to reward
                                cycle_seg_batch, kl_metrics = apply_kl_penalty(cycle_seg_batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                cycle_seg_batch.batch["token_level_rewards"] = cycle_seg_batch.batch["token_level_scores"]

                            # compute advantages, executed on the driver process
                            cycle_seg_batch = compute_advantage(
                                cycle_seg_batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                            )

                    # update critic
                    if cycle_seg_batch is not None and self.use_critic:
                        with timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(cycle_seg_batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)
                    
                    # Clear multi-modal cache before switching to segmentation task
                    self.actor_rollout_ref_wg.clear_multi_modal_cache()

                    seg_batch = cycle_seg_batch

                # Concatenate non_cycle_batch and cycle_cap_batch into cap_batch
                if non_cycle_batch is not None and cycle_cap_batch is not None:
                    # Remove iou_scores and correct_mask from batch before concat
                    if 'iou_scores' in cycle_cap_batch.non_tensor_batch:
                        del cycle_cap_batch.non_tensor_batch['iou_scores']
                    if 'correct_mask' in cycle_cap_batch.non_tensor_batch:
                        del cycle_cap_batch.non_tensor_batch['correct_mask']
                    cap_batch = DataProto.concat([non_cycle_batch, cycle_cap_batch])
                elif non_cycle_batch is not None:
                    cap_batch = non_cycle_batch
                elif cycle_cap_batch is not None:
                    cap_batch = cycle_cap_batch
                else:
                    cap_batch = None

                if self.config.worker.actor.optimize_segmenter and self.config.worker.actor.optimize_captioner:
                    # Case 1: Both tasks - Use gradient accumulation for cap_batch and seg_batch
                    cap_batch_size = len(cap_batch) if cap_batch is not None else 0
                    seg_batch_size = len(seg_batch) if seg_batch is not None else 0
                    total_size = cap_batch_size + seg_batch_size
                    
                    # Caption 和 Segmentation 各占 0.5
                    cap_grad_weight = 0.5
                    seg_grad_weight = 0.5

                    if self.config.trainer.critic_warmup <= self.global_step:
                        self.actor_rollout_ref_wg.clear_multi_modal_cache()
                        
                        with timer("update_actor", timing_raw):
                            actor_metrics = {}
                            
                            # Step 1: Accumulate gradients from cap_batch (combined non_single + single)
                            if cap_batch is not None and cap_batch_size > 0:
                                cap_batch.meta_info['grad_weight'] = cap_grad_weight
                                cap_batch.meta_info['global_batch_size_per_device'] = len(cap_batch) // self.actor_rollout_ref_wg.world_size
                                cap_output = self.actor_rollout_ref_wg.accumulate_actor_gradients(cap_batch)
                                actor_metrics.update({f"cap_{k}": v for k, v in reduce_metrics(cap_output.non_tensor_batch).items()})
                            
                            # Step 2: Accumulate gradients from segmentation batch
                            if seg_batch is not None and seg_batch_size > 0:
                                seg_batch.meta_info['grad_weight'] = seg_grad_weight
                                seg_batch.meta_info['global_batch_size_per_device'] = len(seg_batch) // self.actor_rollout_ref_wg.world_size
                                seg_output = self.actor_rollout_ref_wg.accumulate_actor_gradients(seg_batch)
                                actor_metrics.update({f"seg_{k}": v for k, v in reduce_metrics(seg_output.non_tensor_batch).items()})
                            
                            # Step 3: Perform optimizer step with accumulated gradients
                            opt_output = self.actor_rollout_ref_wg.step_actor_optimizer()

                        # opt_output is a list from ONE_TO_ALL dispatch, take first element's metrics
                        if opt_output and len(opt_output) > 0 and hasattr(opt_output[0], 'non_tensor_batch'):
                            actor_metrics.update(reduce_metrics(opt_output[0].non_tensor_batch))
                        metrics.update(actor_metrics)

                elif self.config.worker.actor.optimize_captioner and not self.config.worker.actor.optimize_segmenter:
                    # Case 2: Only captioner - Use cap_batch to update
                    if self.config.trainer.critic_warmup <= self.global_step:
                        self.actor_rollout_ref_wg.clear_multi_modal_cache()
                        
                        with timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_ref_wg.update_actor(cap_batch)

                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                elif self.config.worker.actor.optimize_segmenter and not self.config.worker.actor.optimize_captioner:
                    # Case 3: Only segmenter - Use seg_batch to update
                    if self.config.trainer.critic_warmup <= self.global_step:
                        self.actor_rollout_ref_wg.clear_multi_modal_cache()
                        
                        with timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_ref_wg.update_actor(seg_batch)

                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=cap_batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=cap_batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=cap_batch, timing_raw=timing_raw, num_gpus=num_gpus))
            if seg_batch is not None:           
                metrics.update(compute_data_metrics(batch=seg_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=seg_batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=seg_batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
